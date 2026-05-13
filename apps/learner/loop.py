"""Self-learning loop driver.

For each agent in [agent_1, agent_2, agent_3]:
  1. run baseline (N paired sims with active prompt)
  2. find weakest persona buckets
  3. ask Opus to propose K variant prompts targeting the weakness
  4. compliance pre-flight (subset of probes, fast)
  5. run K paired evaluations with each candidate variant
  6. stat gate → adopt or reject; log either way

For Day 3 we run on agent_1 only. agent_2/agent_3 wiring is mechanical and
follows the same pattern; full-pipeline integration comes on Day 4.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from packages.agents.base import BaseAgent
from packages.agents.agent_1 import AssessmentAgent
from packages.compliance import run_probe_suite
from packages.evaluator.metrics import (
    AgentScores,
    ConvScores,
    cheap_compliance,
    primary_metric,
)
from packages.llm import AnthropicClient, count_tokens
from packages.simulator.borrower import BorrowerProfile, BorrowerSimulator, load_borrowers
from packages.simulator.runner import run_chat_conversation
from packages.stats import GateDecision, evaluate_gate
from packages.storage.repos import (
    get_active_prompt,
    set_active_prompt,
    upsert_prompt_version,
)
from apps.learner.prompt_engineer import PromptProposal, propose_variants

log = logging.getLogger("learner")
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_evaluations"
RAW_DIR.mkdir(parents=True, exist_ok=True)

_AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "agent_1": AssessmentAgent,
}


@dataclass
class VariantResult:
    variant_idx: int
    rationale: str
    prompt_tokens: int
    compliance_preflight_passed: bool
    compliance_preflight_failures: list[str] = field(default_factory=list)
    scores: Optional[AgentScores] = None
    gate_decision: Optional[str] = None
    gate_reasons: list[str] = field(default_factory=list)
    primary_diff: float = 0.0
    cohens_d: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0


@dataclass
class IterationResult:
    iteration_id: int
    agent_id: str
    baseline_prompt_version: int
    baseline_prompt_tokens: int
    baseline_primary_mean: float
    baseline_compliance_rate: float
    weak_dims: list[str]
    variants: list[VariantResult] = field(default_factory=list)
    adopted_variant_idx: Optional[int] = None
    adopted_new_version: Optional[int] = None
    cost_usd_iteration: float = 0.0


def _pick_borrowers(n: int) -> list[BorrowerProfile]:
    """Pick n borrowers — balanced across personas if possible."""
    by_persona: dict[str, list[BorrowerProfile]] = {}
    for b in load_borrowers():
        by_persona.setdefault(b.persona, []).append(b)
    per = max(1, n // len(by_persona))
    picked: list[BorrowerProfile] = []
    for persona, lst in by_persona.items():
        picked.extend(lst[:per])
    # top up if short
    rest = [b for plist in by_persona.values() for b in plist[per:]]
    while len(picked) < n and rest:
        picked.append(rest.pop(0))
    return picked[:n]


def _evaluate_prompt(
    agent_id: str,
    prompt_text: str,
    prompt_version: int,
    borrowers: list[BorrowerProfile],
    iteration_id: int,
    client: Optional[AnthropicClient] = None,
    label: str = "baseline",
) -> AgentScores:
    """Run agent against each borrower, score, return AgentScores."""
    agent_cls = _AGENT_CLASSES[agent_id]
    client = client or AnthropicClient()
    scores = AgentScores()
    for i, profile in enumerate(borrowers):
        agent: BaseAgent = agent_cls(
            client=client,
            override_prompt_text=prompt_text,
            override_prompt_version=prompt_version,
        )
        sim = BorrowerSimulator(profile, client=client)
        conv_id, result = run_chat_conversation(
            agent,
            sim,
            max_turns=7,
            handoff="",
            workflow_id=f"learner-{label}-{iteration_id}-{i}",
            iteration_id=iteration_id,
            persona=profile.persona,
        )
        primary, outcome_metrics = primary_metric(agent_id, result.transcript, result.tool_calls)
        compliance_dict = cheap_compliance(result.transcript, result.tool_calls)
        compliance_rate = sum(compliance_dict.values()) / len(compliance_dict)
        # Day-3 system metric = "reached terminal outcome cleanly" (1 if outcome != no_response, else 0)
        system_value = 0.0 if result.outcome == "no_response" else 1.0
        cs = ConvScores(
            conversation_id=str(conv_id),
            persona=profile.persona,
            agent_id=agent_id,
            primary=primary,
            outcome_metrics=outcome_metrics,
            compliance=compliance_dict,
            compliance_pass_rate=compliance_rate,
        )
        scores.add(cs, system_value)
        log.info("  %s/%-15s persona=%-12s primary=%.3f compliance=%.2f outcome=%s",
                 label, str(profile.id)[:8], profile.persona, primary, compliance_rate, result.outcome)
    return scores


def _weak_dims(baseline: AgentScores, agent_id: str) -> list[str]:
    """Identify which metrics are weak so the prompt engineer can target them."""
    if not baseline.convs:
        return []
    weaknesses: list[str] = []
    # ID coverage
    id_ver_rate = sum(1 for c in baseline.convs if c.outcome_metrics.get("identity_verified")) / len(baseline.convs)
    if id_ver_rate < 0.7:
        weaknesses.append(f"identity_verification_rate={id_ver_rate:.2f} (target ≥ 0.85)")
    # Fields captured (Agent 1)
    if agent_id == "agent_1":
        fields_mean = sum(c.outcome_metrics.get("fields_captured", 0) for c in baseline.convs) / len(baseline.convs)
        if fields_mean < 2.0:
            weaknesses.append(f"financial_fields_captured_mean={fields_mean:.2f} (target ≥ 2.5)")
    # Compliance regex
    comp_mean = sum(c.compliance_pass_rate for c in baseline.convs) / len(baseline.convs)
    if comp_mean < 0.95:
        weaknesses.append(f"regex_compliance_pass_rate={comp_mean:.2f} (target ≥ 0.95)")
    # Primary
    p_mean = sum(c.primary for c in baseline.convs) / len(baseline.convs)
    if p_mean < 0.75:
        weaknesses.append(f"primary_metric_mean={p_mean:.2f} (room to improve)")
    return weaknesses


def _failures_for_prompt_engineer(baseline: AgentScores, n: int = 4) -> list[dict]:
    ranked = sorted(baseline.convs, key=lambda c: c.primary)[:n]
    # need transcripts — we re-load from DB to keep memory simple
    from packages.storage.repos import load_turns
    out: list[dict] = []
    for cs in ranked:
        turns = load_turns(uuid.UUID(cs.conversation_id))
        out.append({
            "persona": cs.persona,
            "primary": cs.primary,
            "outcome_metrics": cs.outcome_metrics,
            "transcript": [{"role": t["role"], "content": t["content"]} for t in turns],
        })
    return out


def _compliance_preflight(
    new_prompt: str, agent_id: str, client: AnthropicClient
) -> tuple[bool, list[str]]:
    """Run the subset of compliance probes targeting this agent under the new prompt.

    We temporarily install the candidate prompt as active, run probes, restore.
    Caller passes the candidate prompt text and we wrap it.
    """
    # We swap the active prompt to the candidate, run probes that target this agent,
    # then revert. The DB constraint requires a real PromptVersion row, so write
    # the candidate first with status='candidate' and switch the pointer.
    from sqlalchemy import select

    from packages.storage.db import session_scope
    from packages.storage.models import ActivePrompt, PromptVersion

    with session_scope() as s:
        cand = PromptVersion(
            agent_id=agent_id,
            version=10_000 + int(datetime.utcnow().timestamp()) % 10_000,
            prompt_text=new_prompt,
            prompt_tokens=count_tokens(new_prompt),
            status="candidate_preflight",
        )
        s.add(cand); s.flush()
        cand_id = cand.id
        prev_active = s.get(ActivePrompt, agent_id).version_id
        s.get(ActivePrompt, agent_id).version_id = cand_id
    try:
        suite = run_probe_suite(only_agent_id=agent_id, judge_client=client)
        failed = [o.probe_id for o in suite.failed()]
        return len(failed) == 0, failed
    finally:
        with session_scope() as s:
            s.get(ActivePrompt, agent_id).version_id = prev_active


def run_iteration(
    *,
    agent_id: str,
    iteration_id: int,
    n_borrowers: int = 15,
    n_variants: int = 2,
    client: Optional[AnthropicClient] = None,
) -> IterationResult:
    client = client or AnthropicClient()
    log.info("\n%s  ITERATION %d  agent=%s  N=%d  %s", "=" * 12, iteration_id, agent_id, n_borrowers, "=" * 12)

    active = get_active_prompt(agent_id)
    borrowers = _pick_borrowers(n_borrowers)
    log.info("baseline: prompt v%d (%d tokens), %d borrowers",
             active.version, active.prompt_tokens, len(borrowers))

    # ---- baseline ----
    baseline = _evaluate_prompt(
        agent_id, active.prompt_text, active.version,
        borrowers, iteration_id, client, label="baseline",
    )
    p_mean = sum(baseline.primary) / len(baseline.primary)
    c_mean = sum(baseline.compliance) / len(baseline.compliance)
    log.info("baseline: primary_mean=%.3f compliance_mean=%.3f", p_mean, c_mean)
    weak = _weak_dims(baseline, agent_id)
    log.info("baseline weak dims: %s", weak or ["(none above thresholds)"])

    iteration = IterationResult(
        iteration_id=iteration_id,
        agent_id=agent_id,
        baseline_prompt_version=active.version,
        baseline_prompt_tokens=active.prompt_tokens,
        baseline_primary_mean=p_mean,
        baseline_compliance_rate=c_mean,
        weak_dims=weak,
    )

    # If baseline is strong AND nothing weak → still try variants? For Day 3 we always try.
    failures = _failures_for_prompt_engineer(baseline)
    proposals: list[PromptProposal] = propose_variants(
        agent_id=agent_id,
        current_prompt=active.prompt_text,
        weak_dims=weak,
        failures=failures,
        n_variants=n_variants,
        client=client,
        iteration_id=iteration_id,
    )
    log.info("prompt engineer produced %d valid proposals", len(proposals))

    for idx, prop in enumerate(proposals):
        log.info("-- variant %d (%d tokens) — %s", idx, prop.tokens, prop.rationale[:80])
        v = VariantResult(
            variant_idx=idx,
            rationale=prop.rationale,
            prompt_tokens=prop.tokens,
            compliance_preflight_passed=False,
        )
        # compliance pre-flight
        try:
            passed, failed_probes = _compliance_preflight(prop.prompt_text, agent_id, client)
        except Exception as e:
            log.warning("    preflight error: %s", e)
            passed, failed_probes = False, [f"preflight_error: {e}"]
        v.compliance_preflight_passed = passed
        v.compliance_preflight_failures = failed_probes
        if not passed:
            log.info("    REJECT pre-flight: %s", failed_probes)
            iteration.variants.append(v)
            continue

        # paired eval
        vscores = _evaluate_prompt(
            agent_id, prop.prompt_text, -1,
            borrowers, iteration_id, client, label=f"variant{idx}",
        )
        v.scores = vscores
        gate = evaluate_gate(
            baseline_primary=baseline.primary,
            variant_primary=vscores.primary,
            baseline_compliance=baseline.compliance,
            variant_compliance=vscores.compliance,
            baseline_system=baseline.system,
            variant_system=vscores.system,
        )
        v.gate_decision = gate.decision.value
        v.gate_reasons = gate.reasons
        v.primary_diff = gate.primary_diff
        v.cohens_d = gate.cohens_d
        v.ci_lower = gate.bootstrap.ci_lower
        v.ci_upper = gate.bootstrap.ci_upper
        log.info(
            "    gate=%s  primary_diff=%+.3f  d=%.2f  CI=[%+.3f, %+.3f]",
            v.gate_decision, v.primary_diff, v.cohens_d, v.ci_lower, v.ci_upper,
        )
        if gate.decision == GateDecision.ADOPT:
            new_version = active.version + 1
            new_id = upsert_prompt_version(
                agent_id=agent_id,
                version=new_version,
                prompt_text=prop.prompt_text,
                prompt_tokens=prop.tokens,
                status="active",
                parent_version=active.id,
                adoption_data={
                    "iteration_id": iteration_id,
                    "primary_diff": gate.primary_diff,
                    "cohens_d": gate.cohens_d,
                    "ci_lower": gate.bootstrap.ci_lower,
                    "ci_upper": gate.bootstrap.ci_upper,
                    "compliance_baseline": gate.compliance_baseline,
                    "compliance_variant": gate.compliance_variant,
                    "system_p": gate.system_p,
                    "rationale": prop.rationale,
                },
            )
            set_active_prompt(agent_id, new_id)
            iteration.adopted_variant_idx = idx
            iteration.adopted_new_version = new_version
            log.info("    >>> ADOPTED as %s v%d <<<", agent_id, new_version)
        else:
            upsert_prompt_version(
                agent_id=agent_id,
                version=10_000 + iteration_id * 100 + idx,  # synthetic version for rejects
                prompt_text=prop.prompt_text,
                prompt_tokens=prop.tokens,
                status="rejected",
                parent_version=active.id,
                adoption_data={
                    "iteration_id": iteration_id,
                    "gate_decision": v.gate_decision,
                    "gate_reasons": v.gate_reasons,
                    "primary_diff": gate.primary_diff,
                    "cohens_d": gate.cohens_d,
                    "ci_lower": gate.bootstrap.ci_lower,
                    "ci_upper": gate.bootstrap.ci_upper,
                },
            )
            log.info("    REJECTED: %s", v.gate_reasons)
        iteration.variants.append(v)
        if iteration.adopted_variant_idx is not None:
            break  # stop at first winner

    _write_iteration_csv(iteration, baseline)
    return iteration


def _write_iteration_csv(iteration: IterationResult, baseline: AgentScores) -> None:
    import csv

    path = RAW_DIR / f"iter_{iteration.iteration_id:02d}_{iteration.agent_id}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "iteration_id", "agent_id", "label", "borrower_idx", "persona",
            "primary", "compliance_pass_rate", "outcome_metrics", "conv_id",
        ])
        for i, cs in enumerate(baseline.convs):
            w.writerow([
                iteration.iteration_id, iteration.agent_id, "baseline", i, cs.persona,
                f"{cs.primary:.4f}", f"{cs.compliance_pass_rate:.4f}",
                json.dumps(cs.outcome_metrics), cs.conversation_id,
            ])
        for v in iteration.variants:
            if v.scores is None:
                continue
            for i, cs in enumerate(v.scores.convs):
                w.writerow([
                    iteration.iteration_id, iteration.agent_id, f"variant{v.variant_idx}", i, cs.persona,
                    f"{cs.primary:.4f}", f"{cs.compliance_pass_rate:.4f}",
                    json.dumps(cs.outcome_metrics), cs.conversation_id,
                ])
    log.info("wrote %s", path)

    summary = RAW_DIR / f"iter_{iteration.iteration_id:02d}_{iteration.agent_id}_summary.json"
    summary.write_text(json.dumps(
        {
            "iteration_id": iteration.iteration_id,
            "agent_id": iteration.agent_id,
            "baseline_prompt_version": iteration.baseline_prompt_version,
            "baseline_primary_mean": iteration.baseline_primary_mean,
            "baseline_compliance_rate": iteration.baseline_compliance_rate,
            "weak_dims": iteration.weak_dims,
            "adopted_variant_idx": iteration.adopted_variant_idx,
            "adopted_new_version": iteration.adopted_new_version,
            "variants": [
                {
                    "idx": v.variant_idx,
                    "rationale": v.rationale,
                    "tokens": v.prompt_tokens,
                    "preflight_passed": v.compliance_preflight_passed,
                    "preflight_failures": v.compliance_preflight_failures,
                    "gate_decision": v.gate_decision,
                    "gate_reasons": v.gate_reasons,
                    "primary_diff": v.primary_diff,
                    "cohens_d": v.cohens_d,
                    "ci_lower": v.ci_lower,
                    "ci_upper": v.ci_upper,
                }
                for v in iteration.variants
            ],
        },
        indent=2,
    ))
    log.info("wrote %s", summary)
