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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from apps.learner.prompt_engineer import PromptProposal, propose_variants
from packages.agents.agent_1 import AssessmentAgent
from packages.agents.agent_2 import ResolutionAgent
from packages.agents.agent_3 import FinalNoticeAgent
from packages.agents.base import BaseAgent
from packages.compliance import run_probe_suite
from packages.evaluator.metrics import (
    AgentScores,
    ConvScores,
    cheap_compliance,
    primary_metric,
)
from packages.evaluator.system_judge import judge_full_pipeline
from packages.llm import AnthropicClient, count_tokens
from packages.simulator.borrower import BorrowerProfile, BorrowerSimulator, load_borrowers
from packages.simulator.runner import run_chat_conversation
from packages.stats import GateDecision, evaluate_gate
from packages.storage.repos import (
    get_active_prompt,
    set_active_prompt,
    upsert_prompt_version,
)

log = logging.getLogger("learner")
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_evaluations"
RAW_DIR.mkdir(parents=True, exist_ok=True)

_AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "agent_1": AssessmentAgent,
    "agent_2": ResolutionAgent,
    "agent_3": FinalNoticeAgent,
}

# Representative handoff payloads for evaluating agent_2 / agent_3 in isolation.
# Each is hand-tuned to be a plausible output of the prior stage(s) for the named
# persona. Keeps the learning loop cheap by avoiding a full pipeline run per eval.
_STUB_HANDOFFS_TO_AGENT_2: dict[str, str] = {
    "cooperative": (
        '{"identity":{"verified":true,"method":"last4_ssn+dob","confidence":"high"},'
        '"debt":{"amount_acknowledged":4250.0,"borrower_disputes":false},'
        '"financial_situation":{"employment":"part_time","monthly_income_band":"1k-2k",'
        '"stated_hardship":[],"ability_to_pay_lump":"no","ability_to_pay_plan":"yes_under_200_mo"},'
        '"offers_made":[],"objections_raised":[],"emotional_state":"engaged",'
        '"compliance_flags":{"opt_out_requested":false,"hardship_program_offered":false,'
        '"sensitive_disclosure":null},"open_threads":["awaiting_resolution_options"],'
        '"borrower_quotes":["I want to get this resolved"]}'
    ),
    "distressed": (
        '{"identity":{"verified":true,"method":"last4_ssn+dob","confidence":"high"},'
        '"debt":{"amount_acknowledged":8500.0,"borrower_disputes":false},'
        '"financial_situation":{"employment":"unemployed","monthly_income_band":"under_1k",'
        '"stated_hardship":["medical","job_loss"],"ability_to_pay_lump":"no",'
        '"ability_to_pay_plan":"unknown"},'
        '"offers_made":[],"objections_raised":[],"emotional_state":"distressed",'
        '"compliance_flags":{"opt_out_requested":false,"hardship_program_offered":true,'
        '"sensitive_disclosure":"medical"},"open_threads":["hardship_referral_recommended"],'
        '"borrower_quotes":["I lost my job in february","I can barely hold it together"]}'
    ),
    "combative": (
        '{"identity":{"verified":true,"method":"last4_ssn+dob","confidence":"medium"},'
        '"debt":{"amount_acknowledged":null,"borrower_disputes":true,'
        '"dispute_basis":"never_agreed_to_this_debt"},'
        '"financial_situation":{"employment":null,"monthly_income_band":null,'
        '"stated_hardship":[],"ability_to_pay_lump":"unknown","ability_to_pay_plan":"unknown"},'
        '"offers_made":[],"objections_raised":["debt_disputed","wants_documentation"],'
        '"emotional_state":"hostile",'
        '"compliance_flags":{"opt_out_requested":false,"hardship_program_offered":false,'
        '"sensitive_disclosure":null},"open_threads":["dispute_resolution_pending"],'
        '"borrower_quotes":["I never agreed to this","Prove I owe this"]}'
    ),
}

_STUB_HANDOFFS_TO_AGENT_3: dict[str, str] = {
    "cooperative": (
        '{"identity":{"verified":true,"method":"last4_ssn+dob","confidence":"high"},'
        '"debt":{"amount_acknowledged":4250.0,"borrower_disputes":false},'
        '"financial_situation":{"employment":"part_time","monthly_income_band":"1k-2k",'
        '"stated_hardship":[],"ability_to_pay_lump":"no","ability_to_pay_plan":"yes_under_200_mo"},'
        '"offers_made":[{"type":"lump_30","borrower_response":"declined"},'
        '{"type":"plan_12","borrower_response":"considering"}],'
        '"objections_raised":["payment_too_high"],"emotional_state":"reluctant_but_engaged",'
        '"compliance_flags":{"opt_out_requested":false,"hardship_program_offered":false,'
        '"sensitive_disclosure":null},"open_threads":["awaiting_decision_48h"],'
        '"borrower_quotes":["I need to think about it","I cant do 300 a month"]}'
    ),
    "distressed": (
        '{"identity":{"verified":true,"method":"last4_ssn+dob","confidence":"high"},'
        '"debt":{"amount_acknowledged":8500.0,"borrower_disputes":false},'
        '"financial_situation":{"employment":"unemployed","monthly_income_band":"under_1k",'
        '"stated_hardship":["medical","job_loss"],"ability_to_pay_lump":"no",'
        '"ability_to_pay_plan":"unknown"},'
        '"offers_made":[{"type":"hardship_referral","borrower_response":"accepted"}],'
        '"objections_raised":[],"emotional_state":"distressed",'
        '"compliance_flags":{"opt_out_requested":false,"hardship_program_offered":true,'
        '"sensitive_disclosure":"medical"},"open_threads":["hardship_program_pending"],'
        '"borrower_quotes":["I cant work right now"]}'
    ),
    "combative": (
        '{"identity":{"verified":true,"method":"last4_ssn+dob","confidence":"medium"},'
        '"debt":{"amount_acknowledged":null,"borrower_disputes":true,'
        '"dispute_basis":"never_agreed_to_this_debt"},'
        '"financial_situation":{"employment":null,"monthly_income_band":null,'
        '"stated_hardship":[],"ability_to_pay_lump":"unknown","ability_to_pay_plan":"unknown"},'
        '"offers_made":[{"type":"plan_12","borrower_response":"declined"}],'
        '"objections_raised":["debt_disputed","wants_documentation"],'
        '"emotional_state":"hostile",'
        '"compliance_flags":{"opt_out_requested":false,"hardship_program_offered":false,'
        '"sensitive_disclosure":null},"open_threads":["dispute_resolution_pending"],'
        '"borrower_quotes":["Im calling my lawyer"]}'
    ),
}


def _handoff_for(agent_id: str, persona: str) -> str:
    if agent_id == "agent_1":
        return ""
    table = _STUB_HANDOFFS_TO_AGENT_2 if agent_id == "agent_2" else _STUB_HANDOFFS_TO_AGENT_3
    return table.get(persona, table["cooperative"])


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
    seamlessness_baseline: Optional[float] = None
    seamlessness_variant: Optional[float] = None
    system_check_ran: bool = False


@dataclass
class IterationResult:
    iteration_id: int
    agent_id: str
    eval_mode: str
    baseline_prompt_version: int
    baseline_prompt_tokens: int
    baseline_primary_mean: float
    baseline_compliance_rate: float
    weak_dims: list[str]
    variants: list[VariantResult] = field(default_factory=list)
    adopted_variant_idx: Optional[int] = None
    adopted_new_version: Optional[int] = None
    cost_usd_iteration: float = 0.0


@dataclass
class PipelineStageRun:
    agent_id: str
    conversation_id: uuid.UUID
    outcome: str
    transcript: list[dict]
    tool_calls: list[dict]


def _pick_borrowers(n: int) -> list[BorrowerProfile]:
    """Pick n borrowers — balanced across personas if possible."""
    by_persona: dict[str, list[BorrowerProfile]] = {}
    for b in load_borrowers():
        by_persona.setdefault(b.persona, []).append(b)
    per = max(1, n // len(by_persona))
    picked: list[BorrowerProfile] = []
    for _persona, lst in by_persona.items():
        picked.extend(lst[:per])
    # top up if short
    rest = [b for plist in by_persona.values() for b in plist[per:]]
    while len(picked) < n and rest:
        picked.append(rest.pop(0))
    return picked[:n]


# Personas we consider "real test fixtures" — used to filter DB reuse and to
# exclude human-chat / probe / vapi conversations from learning samples.
_SIM_PERSONAS = ("cooperative", "combative", "evasive", "confused", "distressed")


def _aggregate_tool_calls_from_turns(conv_id: uuid.UUID) -> tuple[list[dict], list[dict]]:
    """Re-load a conversation's transcript + flattened tool_calls from the DB.
    Used by the baseline-reuse path (no LLM cost)."""
    from sqlalchemy import select

    from packages.storage.db import session_scope
    from packages.storage.models import Turn

    with session_scope() as s:
        rows = s.execute(
            select(Turn).where(Turn.conversation_id == conv_id).order_by(Turn.seq)
        ).scalars().all()
        transcript = [
            {
                "agent_id": t.agent_id,
                "role": t.role,
                "content": t.content,
                "seq": t.seq,
            }
            for t in rows
        ]
        tool_calls: list[dict] = []
        for t in rows:
            if t.tool_calls and t.tool_calls.get("calls"):
                tool_calls.extend(t.tool_calls["calls"])
        return transcript, tool_calls


def _load_baseline_from_db(
    agent_id: str,
    active_version: int,
    n_target: int,
    current_iteration_id: int,
) -> Optional[tuple[AgentScores, list[BorrowerProfile]]]:
    """Try to assemble a baseline AgentScores from previously-stored conversations.

    Returns (AgentScores, borrowers) if we find >= n_target conversations matching:
      - iteration_id IS NOT NULL (was a learning-loop run, not chat/probe/vapi)
      - agent_versions[agent_id] == active_version
      - persona ∈ _SIM_PERSONAS
      - borrower_id resolvable to a profile in seeds.json (so the variant can be
        evaluated on the SAME borrowers and the paired bootstrap stays valid)

    No LLM calls — scores are re-computed from stored turns + tool_calls. The
    returned borrower list is the EXACT set the variant must be paired against.
    """
    from sqlalchemy import select

    from packages.storage.db import session_scope
    from packages.storage.models import Conversation

    with session_scope() as s:
        candidates = s.execute(
            select(Conversation)
            .where(
                Conversation.iteration_id.isnot(None),
                Conversation.persona.in_(_SIM_PERSONAS),
            )
            .order_by(Conversation.id.desc())
            .limit(n_target * 8)
        ).scalars().all()
        matching = [
            c for c in candidates
            if c.agent_versions and c.agent_versions.get(agent_id) == active_version
        ]
        # Index profiles by id so we can pair the variant on the SAME borrowers.
        profiles_by_id = {b.id: b for b in load_borrowers()}
        matching = [c for c in matching if str(c.borrower_id) in profiles_by_id]
        # Dedupe by borrower_id — at most one stored convo per borrower (most recent).
        seen: set[str] = set()
        unique: list[Conversation] = []
        for c in matching:
            bid = str(c.borrower_id)
            if bid in seen:
                continue
            seen.add(bid)
            unique.append(c)
        # Balance across personas first, then pad up to n_target from remainder.
        per_persona = max(1, n_target // len(_SIM_PERSONAS))
        by_persona: dict[str, list[Conversation]] = {}
        for c in unique:
            by_persona.setdefault(c.persona, []).append(c)
        picked: list[Conversation] = []
        leftover: list[Conversation] = []
        for plist in by_persona.values():
            picked.extend(plist[:per_persona])
            leftover.extend(plist[per_persona:])
        while len(picked) < n_target and leftover:
            picked.append(leftover.pop(0))
        if len(picked) < n_target:
            return None
        picked = picked[:n_target]
        for c in picked:
            s.expunge(c)

    paired_profiles: list[BorrowerProfile] = [profiles_by_id[str(c.borrower_id)] for c in picked]
    scores = AgentScores()
    for c in picked:
        transcript, tool_calls = _aggregate_tool_calls_from_turns(c.id)
        primary, outcome_metrics = primary_metric(agent_id, transcript, tool_calls)
        comp_dict = cheap_compliance(transcript, tool_calls)
        comp_rate = sum(comp_dict.values()) / len(comp_dict)
        system_value = 0.0 if c.outcome == "no_response" else 1.0
        scores.add(
            ConvScores(
                conversation_id=str(c.id),
                persona=c.persona,
                agent_id=agent_id,
                primary=primary,
                outcome_metrics=outcome_metrics,
                compliance=comp_dict,
                compliance_pass_rate=comp_rate,
            ),
            system_value,
        )
    return scores, paired_profiles


def _run_full_pipeline_inline(
    profile: BorrowerProfile,
    *,
    variant_agent_id: Optional[str],
    variant_prompt: Optional[str],
    variant_prompt_version: Optional[int] = None,
    iteration_id: int,
    client: AnthropicClient,
    label: str,
) -> list[PipelineStageRun]:
    """Run A1 → summarize → A2 → summarize → A3 inline (no Temporal).

    When ``variant_agent_id`` is given, that agent uses ``variant_prompt``; the
    other two agents use their currently-active prompts. When variant_agent_id
    is None, all three agents use active prompts (pure baseline).

    Returns stage records for each stage actually run.
    Short-circuits early if Agent 1 yielded no usable data or Agent 2 closed a deal.
    """
    from packages.summarizer import summarize_for_handoff

    def _prompt_for(agent_id: str) -> tuple[str, int]:
        if variant_agent_id == agent_id and variant_prompt is not None:
            return variant_prompt, variant_prompt_version or -1
        pv = get_active_prompt(agent_id)
        return pv.prompt_text, pv.version

    stages: list[PipelineStageRun] = []

    # --- stage 1 ---
    p1_text, p1_ver = _prompt_for("agent_1")
    a1 = AssessmentAgent(client=client, override_prompt_text=p1_text, override_prompt_version=p1_ver)
    sim1 = BorrowerSimulator(profile, client=client)
    conv1, res1 = run_chat_conversation(
        a1, sim1, max_turns=7, handoff="",
        workflow_id=f"sys-{label}-{iteration_id}-{profile.id[:8]}-1",
        iteration_id=iteration_id, persona=profile.persona,
    )
    stages.append(PipelineStageRun(
        agent_id="agent_1",
        conversation_id=conv1,
        outcome=res1.outcome,
        transcript=res1.transcript,
        tool_calls=res1.tool_calls,
    ))
    if res1.outcome in ("opt_out", "no_response", "identity_unverified"):
        return stages

    # --- handoff 1 → 2 ---
    turns1 = _aggregate_tool_calls_from_turns(conv1)[0]
    h2 = summarize_for_handoff(
        turns1,
        to_agent="to_agent_2",
        conversation_id=str(conv1),
        iteration_id=iteration_id,
    )

    # --- stage 2 ---
    p2_text, p2_ver = _prompt_for("agent_2")
    a2 = ResolutionAgent(client=client, override_prompt_text=p2_text, override_prompt_version=p2_ver)
    sim2 = BorrowerSimulator(profile, client=client)
    conv2, res2 = run_chat_conversation(
        a2, sim2, max_turns=7, handoff=h2.payload.to_compact_json(),
        workflow_id=f"sys-{label}-{iteration_id}-{profile.id[:8]}-2",
        iteration_id=iteration_id, persona=profile.persona,
    )
    stages.append(PipelineStageRun(
        agent_id="agent_2",
        conversation_id=conv2,
        outcome=res2.outcome,
        transcript=res2.transcript,
        tool_calls=res2.tool_calls,
    ))
    if res2.outcome in ("deal_agreed", "opt_out"):
        return stages

    # --- handoff 2 → 3 ---
    turns_combined = _aggregate_tool_calls_from_turns(conv1)[0] + _aggregate_tool_calls_from_turns(conv2)[0]
    h3 = summarize_for_handoff(
        turns_combined,
        to_agent="to_agent_3",
        conversation_id=str(conv2),
        iteration_id=iteration_id,
    )

    # --- stage 3 ---
    p3_text, p3_ver = _prompt_for("agent_3")
    a3 = FinalNoticeAgent(client=client, override_prompt_text=p3_text, override_prompt_version=p3_ver)
    sim3 = BorrowerSimulator(profile, client=client)
    conv3, res3 = run_chat_conversation(
        a3, sim3, max_turns=7, handoff=h3.payload.to_compact_json(),
        workflow_id=f"sys-{label}-{iteration_id}-{profile.id[:8]}-3",
        iteration_id=iteration_id, persona=profile.persona,
    )
    stages.append(PipelineStageRun(
        agent_id="agent_3",
        conversation_id=conv3,
        outcome=res3.outcome,
        transcript=res3.transcript,
        tool_calls=res3.tool_calls,
    ))
    return stages


def _system_level_check(
    agent_id: str,
    variant_prompt: str,
    iteration_id: int,
    client: AnthropicClient,
    n: int = 3,
) -> tuple[list[float], list[float]]:
    """Run N=3 full A1→A2→A3 pipelines for baseline (all active) and variant
    (variant in its slot, active elsewhere). Score handoff seamlessness via
    system_judge. Returns paired score lists in [1..5].

    Cost: ~$0.50-1 per call. Only invoked AFTER a variant passes the per-agent gate.
    Never called for agent_3 (no downstream agents to disturb).
    """
    borrowers = _pick_borrowers(n)
    baseline_scores: list[float] = []
    variant_scores: list[float] = []

    for profile in borrowers:
        # baseline run — no variant
        try:
            bt = _run_full_pipeline_inline(
                profile, variant_agent_id=None, variant_prompt=None,
                iteration_id=iteration_id, client=client, label="sys-baseline",
            )
            br = judge_full_pipeline(
                [(s.agent_id, s.transcript) for s in bt],
                client=client,
                iteration_id=iteration_id,
            )
            if br and "handoff_seamlessness" in br:
                baseline_scores.append(float(br["handoff_seamlessness"]))
        except Exception as e:
            log.warning("    system check: baseline pipeline failed: %s", e)

        # variant run
        try:
            vt = _run_full_pipeline_inline(
                profile, variant_agent_id=agent_id, variant_prompt=variant_prompt,
                iteration_id=iteration_id, client=client, label="sys-variant",
            )
            vr = judge_full_pipeline(
                [(s.agent_id, s.transcript) for s in vt],
                client=client,
                iteration_id=iteration_id,
            )
            if vr and "handoff_seamlessness" in vr:
                variant_scores.append(float(vr["handoff_seamlessness"]))
        except Exception as e:
            log.warning("    system check: variant pipeline failed: %s", e)

    return baseline_scores, variant_scores


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
            handoff=_handoff_for(agent_id, profile.persona),
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


def _evaluate_prompt_full_pipeline(
    agent_id: str,
    prompt_text: str,
    prompt_version: int,
    borrowers: list[BorrowerProfile],
    iteration_id: int,
    client: Optional[AnthropicClient] = None,
    label: str = "baseline",
) -> AgentScores:
    """Run full A1→A2→A3 pipelines and score the requested agent's stage.

    Unlike ``_evaluate_prompt`` this does not use stub handoffs for Agent 2/3.
    The candidate agent is evaluated in its real upstream/downstream context:
      - agent_1 candidate: A1 candidate → real summary → active A2 → active A3
      - agent_2 candidate: active A1 → real summary → A2 candidate → active A3
      - agent_3 candidate: active A1 → real summary → active A2 → real summary → A3 candidate
    """
    client = client or AnthropicClient()
    scores = AgentScores()

    for i, profile in enumerate(borrowers):
        stages = _run_full_pipeline_inline(
            profile,
            variant_agent_id=agent_id,
            variant_prompt=prompt_text,
            variant_prompt_version=prompt_version,
            iteration_id=iteration_id,
            client=client,
            label=f"{label}-{i}",
        )
        target = next((s for s in stages if s.agent_id == agent_id), None)
        if target is None:
            # Upstream stage short-circuited. Keep the paired sample, but score the
            # target as a miss because the full borrower journey never reached it.
            transcript: list[dict] = []
            tool_calls: list[dict] = []
            conv_id = str(stages[-1].conversation_id if stages else uuid.uuid4())
            outcome = "upstream_short_circuit"
        else:
            transcript = target.transcript
            tool_calls = target.tool_calls
            conv_id = str(target.conversation_id)
            outcome = target.outcome

        primary, outcome_metrics = primary_metric(agent_id, transcript, tool_calls)
        compliance_dict = cheap_compliance(transcript, tool_calls)
        compliance_rate = sum(compliance_dict.values()) / len(compliance_dict)
        # System metric is about the whole journey, not only the target stage.
        last_outcome = stages[-1].outcome if stages else "no_response"
        system_value = 0.0 if last_outcome in ("no_response", "upstream_short_circuit") else 1.0
        scores.add(
            ConvScores(
                conversation_id=conv_id,
                persona=profile.persona,
                agent_id=agent_id,
                primary=primary,
                outcome_metrics=outcome_metrics,
                compliance=compliance_dict,
                compliance_pass_rate=compliance_rate,
            ),
            system_value,
        )
        log.info(
            "  %s/%-15s persona=%-12s primary=%.3f compliance=%.2f "
            "target_outcome=%s pipeline_last=%s stages=%s",
            label,
            profile.id[:8],
            profile.persona,
            primary,
            compliance_rate,
            outcome,
            last_outcome,
            "→".join(s.agent_id for s in stages),
        )

    return scores


def _weak_dims(baseline: AgentScores, agent_id: str) -> list[str]:
    """Identify which metrics are weak so the prompt engineer can target them."""
    if not baseline.convs:
        return []
    weaknesses: list[str] = []
    if agent_id == "agent_1":
        id_ver_rate = sum(1 for c in baseline.convs if c.outcome_metrics.get("identity_verified")) / len(baseline.convs)
        if id_ver_rate < 0.7:
            weaknesses.append(f"identity_verification_rate={id_ver_rate:.2f} (target ≥ 0.85)")
        fields_mean = sum(c.outcome_metrics.get("fields_captured", 0) for c in baseline.convs) / len(baseline.convs)
        if fields_mean < 2.0:
            weaknesses.append(f"financial_fields_captured_mean={fields_mean:.2f} (target ≥ 2.5)")
    elif agent_id == "agent_2":
        offer_rate = sum(1 for c in baseline.convs if c.outcome_metrics.get("presented_offer")) / len(baseline.convs)
        commit_rate = sum(1 for c in baseline.convs if c.outcome_metrics.get("obtained_commitment")) / len(baseline.convs)
        if offer_rate < 0.8:
            weaknesses.append(f"present_offer_rate={offer_rate:.2f} (target ≥ 0.85)")
        if commit_rate < 0.3:
            weaknesses.append(f"commitment_rate={commit_rate:.2f} (room to improve)")
    elif agent_id == "agent_3":
        issued_rate = sum(1 for c in baseline.convs if c.outcome_metrics.get("final_offer_issued")) / len(baseline.convs)
        if issued_rate < 0.8:
            weaknesses.append(f"final_offer_issued_rate={issued_rate:.2f} (target ≥ 0.9)")
    comp_mean = sum(c.compliance_pass_rate for c in baseline.convs) / len(baseline.convs)
    if comp_mean < 0.95:
        weaknesses.append(f"regex_compliance_pass_rate={comp_mean:.2f} (target ≥ 0.95)")
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
        s.add(cand)
        s.flush()
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
    eval_mode: str = "full",
    client: Optional[AnthropicClient] = None,
) -> IterationResult:
    client = client or AnthropicClient()
    if eval_mode not in ("full", "isolated"):
        raise ValueError(f"unknown eval_mode={eval_mode!r}; expected 'full' or 'isolated'")
    log.info("\n%s  ITERATION %d  agent=%s  N=%d  eval_mode=%s  %s",
             "=" * 12, iteration_id, agent_id, n_borrowers, eval_mode, "=" * 12)

    active = get_active_prompt(agent_id)
    log.info("baseline: prompt v%d (%d tokens), targeting %d borrowers",
             active.version, active.prompt_tokens, n_borrowers)

    # ---- baseline ----
    # Full mode deliberately does NOT reuse stored isolated conversations because
    # the whole point is to test each prompt inside the real A1→A2→A3 path.
    if eval_mode == "full":
        borrowers = _pick_borrowers(n_borrowers)
        log.info("baseline: running %d fresh FULL PIPELINES", n_borrowers)
        baseline = _evaluate_prompt_full_pipeline(
            agent_id, active.prompt_text, active.version,
            borrowers, iteration_id, client, label="baseline-full",
        )
    else:
        # Cheap fallback: try DB reuse first, fall back to fresh isolated run.
        # When we reuse, the variant MUST run on the exact same borrowers.
        reuse = _load_baseline_from_db(
            agent_id=agent_id,
            active_version=active.version,
            n_target=n_borrowers,
            current_iteration_id=iteration_id,
        )
        if reuse is not None:
            baseline, borrowers = reuse
            log.info("baseline: REUSED %d stored conversations for agent_%s v%d "
                     "(isolated mode; no fresh LLM calls)",
                     len(baseline.convs), agent_id[-1], active.version)
        else:
            borrowers = _pick_borrowers(n_borrowers)
            log.info("baseline: no reusable stored data — running %d isolated conversations",
                     n_borrowers)
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
        eval_mode=eval_mode,
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
        if eval_mode == "full":
            vscores = _evaluate_prompt_full_pipeline(
                agent_id, prop.prompt_text, -1,
                borrowers, iteration_id, client, label=f"variant{idx}-full",
            )
        else:
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
            "    per-agent gate=%s  primary_diff=%+.3f  d=%.2f  CI=[%+.3f, %+.3f]",
            v.gate_decision, v.primary_diff, v.cohens_d, v.ci_lower, v.ci_upper,
        )

        # ---- system-level check (only needed for cheap isolated mode) ----
        # Full mode has already run every paired sample through A1→A2→A3.
        if eval_mode == "isolated" and gate.decision == GateDecision.ADOPT \
                and agent_id in ("agent_1", "agent_2"):
            log.info("    running system-level check: %d full pipelines × 2 (baseline + variant) …", 3)
            sys_baseline, sys_variant = _system_level_check(
                agent_id=agent_id,
                variant_prompt=prop.prompt_text,
                iteration_id=iteration_id,
                client=client,
                n=3,
            )
            v.system_check_ran = True
            v.seamlessness_baseline = (
                sum(sys_baseline) / len(sys_baseline) if sys_baseline else None
            )
            v.seamlessness_variant = (
                sum(sys_variant) / len(sys_variant) if sys_variant else None
            )
            log.info(
                "    system check: seamlessness baseline=%.2f variant=%.2f (n=%d/%d)",
                v.seamlessness_baseline or 0.0, v.seamlessness_variant or 0.0,
                len(sys_baseline), len(sys_variant),
            )
            # Re-run gate with seamlessness data included
            gate = evaluate_gate(
                baseline_primary=baseline.primary,
                variant_primary=vscores.primary,
                baseline_compliance=baseline.compliance,
                variant_compliance=vscores.compliance,
                baseline_system=baseline.system,
                variant_system=vscores.system,
                baseline_seamlessness=sys_baseline,
                variant_seamlessness=sys_variant,
            )
            v.gate_decision = gate.decision.value
            v.gate_reasons = gate.reasons

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
                    "eval_mode": eval_mode,
                    "primary_diff": gate.primary_diff,
                    "cohens_d": gate.cohens_d,
                    "ci_lower": gate.bootstrap.ci_lower,
                    "ci_upper": gate.bootstrap.ci_upper,
                    "compliance_baseline": gate.compliance_baseline,
                    "compliance_variant": gate.compliance_variant,
                    "system_p": gate.system_p,
                    "seamlessness_baseline": v.seamlessness_baseline,
                    "seamlessness_variant": v.seamlessness_variant,
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
                    "eval_mode": eval_mode,
                    "gate_decision": v.gate_decision,
                    "gate_reasons": v.gate_reasons,
                    "primary_diff": gate.primary_diff,
                    "cohens_d": gate.cohens_d,
                    "ci_lower": gate.bootstrap.ci_lower,
                    "ci_upper": gate.bootstrap.ci_upper,
                    "seamlessness_baseline": v.seamlessness_baseline,
                    "seamlessness_variant": v.seamlessness_variant,
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
            "eval_mode": iteration.eval_mode,
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
                    "system_check_ran": v.system_check_ran,
                    "seamlessness_baseline": v.seamlessness_baseline,
                    "seamlessness_variant": v.seamlessness_variant,
                }
                for v in iteration.variants
            ],
        },
        indent=2,
    ))
    log.info("wrote %s", summary)
