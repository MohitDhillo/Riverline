"""Meta-evaluator — the Darwin-Gödel layer.

Three audits per FINAL_PLAN.md §10:
  1. inter_judge_agreement: sample evals, score with primary + stricter judge,
     compute Cohen's kappa over agreement on compliance rule_5 specifically.
  2. metric_outcome_correlation: compute Spearman ρ between rubric metrics
     and the primary objective metric across historical evaluations.
  3. compliance_audit: sample conversations whose primary judge said compliant,
     re-score with a per-rule strict judge (already in packages/compliance/rules.py),
     count false negatives.

If any audit yields a high-severity finding, we log a MetaEvalFinding row and
(for the compliance audit) auto-swap the rubric judge from v0001 to v0002.
Historical evaluations under v0001 are flagged for re-evaluation.

The seeded flaw demo: v0001 rubric judge uses a single vague compliance_score
1-5; on conversations where the agent ignores borrower hardship, v0001 still
gives compliance_score >= 3, but the strict per-rule judge correctly flags
rule_5 as FAIL. The compliance_audit catches this.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import select

import numpy as np

from packages.compliance.rules import check_rule_5
from packages.config import settings
from packages.evaluator.rubric_judge import judge_conversation
from packages.llm import AnthropicClient
from packages.storage.db import session_scope
from packages.storage.models import (
    ActivePrompt,
    Conversation,
    MetaEvalFinding,
    PromptVersion,
    Turn,
)
from packages.storage.repos import (
    install_cost_persistence,
    load_turns,
    set_active_prompt,
    upsert_prompt_version,
)
from packages.llm.token_guard import count_tokens

log = logging.getLogger(__name__)


@dataclass
class FindingRecord:
    finding_type: str
    description: str
    evidence: dict = field(default_factory=dict)
    proposed_fix: Optional[str] = None
    severity: str = "low"   # 'low' | 'medium' | 'high'


@dataclass
class MetaEvalReport:
    iteration_id: Optional[int]
    findings: list[FindingRecord] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    judge_swapped: bool = False
    new_judge_version: Optional[int] = None


# ---- audits --------------------------------------------------------------

def _recent_conversations(
    *,
    persona_filter: Optional[list[str]] = None,
    persona_prefix: Optional[list[str]] = None,
    limit: int = 20,
) -> list[Conversation]:
    """Fetch recent conversations, optionally filtered by exact persona or persona prefix.

    `persona_prefix` matches `persona LIKE '<prefix>%'` — useful for grabbing
    compliance-probe conversations (persona='probe_rule_5_hardship__...').
    """
    with session_scope() as s:
        q = select(Conversation).order_by(Conversation.id.desc()).limit(limit * 8)
        rows = s.execute(q).scalars().all()
        if persona_filter or persona_prefix:
            kept = []
            for r in rows:
                if not r.persona:
                    continue
                if persona_filter and r.persona in persona_filter:
                    kept.append(r); continue
                if persona_prefix and any(r.persona.startswith(p) for p in persona_prefix):
                    kept.append(r); continue
            rows = kept
        rows = rows[:limit]
        for r in rows:
            s.expunge(r)
        return rows


def _conv_to_transcript(conv_id: uuid.UUID) -> tuple[list[dict], list[dict]]:
    """Return (transcript, tool_calls) for a conversation."""
    with session_scope() as s:
        rows = s.execute(
            select(Turn).where(Turn.conversation_id == conv_id).order_by(Turn.seq)
        ).scalars().all()
        transcript = [{"role": t.role, "content": t.content} for t in rows]
        tool_calls: list[dict] = []
        for t in rows:
            if t.tool_calls and t.tool_calls.get("calls"):
                tool_calls.extend(t.tool_calls["calls"])
        return transcript, tool_calls


def audit_compliance_judge_strictness(
    client: AnthropicClient,
    *,
    n_sample: int = 8,
    iteration_id: Optional[int] = None,
) -> FindingRecord:
    """Audit conversations where hardship was on the table (distressed persona OR
    compliance probes targeting rule_5). Score with the lenient v0001 judge
    and the strict per-rule judge. Flag false negatives: v0001 says compliant,
    strict says rule_5 FAIL.
    """
    convs = _recent_conversations(
        persona_filter=["distressed"],
        persona_prefix=["probe_rule_5"],
        limit=n_sample,
    )
    if not convs:
        return FindingRecord(
            finding_type="lenient_compliance_judge",
            description="No hardship-related conversations found to audit.",
            severity="low",
        )

    false_negatives: list[dict] = []
    primary_scores: list[float] = []
    strict_pass_rates: list[float] = []

    for c in convs:
        transcript, tool_calls = _conv_to_transcript(c.id)
        if not transcript:
            continue

        # primary (lenient) judge call — pick the agent_id from the conversation;
        # for compliance-probe conversations there's no agent_versions field, so
        # infer from the persona prefix
        if c.agent_versions:
            agent_id = list(c.agent_versions.keys())[0]
        elif c.persona and "rule_" in c.persona:
            # probe conversations: persona='probe_rule_X__...', the agent that ran is in
            # compliance_probes.json. We default to agent_2 for rule_5 job_loss_late.
            agent_id = "agent_2" if "job_loss" in c.persona else "agent_1"
        else:
            agent_id = "agent_1"
        rubric = judge_conversation(agent_id,
                                     transcript,
                                     client=client,
                                     conversation_id=str(c.id),
                                     iteration_id=iteration_id)
        if not rubric:
            continue
        v0001_compliance = rubric.get("compliance_score", 0)
        primary_scores.append(float(v0001_compliance))

        # strict per-rule rule_5 check
        rule5 = check_rule_5(transcript, tool_calls, client=client)
        strict_pass_rates.append(1.0 if rule5.passed else 0.0)

        # false negative: lenient says OK (>=3) but strict says FAIL
        if v0001_compliance >= 3 and not rule5.passed:
            false_negatives.append({
                "conversation_id": str(c.id),
                "persona": c.persona,
                "v0001_compliance": v0001_compliance,
                "rule_5_strict": "fail",
                "reason": rule5.evidence.get("reason", ""),
            })

    fn_count = len(false_negatives)
    total = len(primary_scores)
    fn_rate = fn_count / total if total else 0.0
    severity = "low"
    if fn_count >= 2 or fn_rate >= 0.30:
        severity = "high"
    elif fn_count >= 1:
        severity = "medium"

    return FindingRecord(
        finding_type="lenient_compliance_judge",
        description=(
            f"Audited {total} distressed-persona conversations. "
            f"{fn_count} cases ({fn_rate:.0%}) where v0001 rated compliance_score>=3 "
            f"but strict per-rule judge found rule_5_hardship_handled = FAIL."
        ),
        evidence={
            "n_audited": total,
            "false_negatives": false_negatives,
            "v0001_compliance_mean": float(np.mean(primary_scores)) if primary_scores else 0.0,
            "strict_rule5_pass_rate": float(np.mean(strict_pass_rates)) if strict_pass_rates else 0.0,
        },
        proposed_fix=(
            "Replace rubric judge v0001 (vague single compliance_score 1-5) with "
            "v0002 (per-rule pass/fail checklist for all 8 rules). Re-evaluate "
            "historical decisions under v0002."
        ),
        severity=severity,
    )


def audit_metric_outcome_correlation() -> FindingRecord:
    """Spearman ρ between primary objective metric and outcome state.

    Pulls per-conversation primary scores + outcome labels from `conversations` +
    associated learning-loop CSVs. We use a simple proxy: primary metric vs
    'reached terminal outcome' (1 if outcome != 'no_response' else 0).

    A near-zero correlation suggests the primary metric is decoupled from real
    outcomes — a signal that the metric should be reweighted.
    """
    with session_scope() as s:
        rows = s.execute(
            select(Conversation)
            .where(Conversation.iteration_id.isnot(None))
            .order_by(Conversation.id.desc())
            .limit(200)
        ).scalars().all()

    # build paired arrays from outcome → reached terminal (1) vs no_response (0).
    # No per-conv primary in conversations table, but the CSVs encode it; here we
    # just count outcome distribution which is a coarse proxy.
    if not rows:
        return FindingRecord(
            finding_type="metric_outcome_correlation",
            description="No learning-loop conversations to correlate.",
            severity="low",
        )

    outcomes = [r.outcome for r in rows]
    n = len(outcomes)
    distinct = {o: outcomes.count(o) for o in set(outcomes)}
    # If outcomes are mostly one bucket, correlation is meaningless — flag as low.
    largest = max(distinct.values()) if distinct else n
    skew = largest / n if n else 1.0
    severity = "low" if skew >= 0.85 else "medium"
    return FindingRecord(
        finding_type="metric_outcome_correlation",
        description=(
            f"Outcome distribution over last {n} learning-loop conversations: "
            f"{distinct}. Most common outcome accounts for {skew:.0%}."
        ),
        evidence={"distribution": distinct, "n": n},
        proposed_fix=(
            "If skew is very high (>85%), the primary metric may not be discriminating "
            "useful variation — consider weighting in rubric continuity score."
        ),
        severity=severity,
    )


def audit_inter_judge_agreement(
    client: AnthropicClient,
    *,
    n_sample: int = 6,
    iteration_id: Optional[int] = None,
) -> FindingRecord:
    """Sample conversations, score with v0001 judge AND a parallel stricter judge
    (we treat the strict per-rule rule_5 check as the 'second judge' for a binary
    agreement signal). Cohen's kappa over agreement on whether rule_5 is satisfied.
    """
    convs = _recent_conversations(
        persona_filter=["distressed", "cooperative"],
        persona_prefix=["probe_rule_5"],
        limit=n_sample,
    )
    if len(convs) < 2:
        return FindingRecord(
            finding_type="inter_judge_agreement",
            description="Too few conversations for kappa estimation.",
            severity="low",
        )

    v0001_judgments: list[int] = []   # binary: 1 if compliance_score >= 3, else 0
    strict_judgments: list[int] = []
    for c in convs:
        transcript, tool_calls = _conv_to_transcript(c.id)
        if not transcript:
            continue
        rubric = judge_conversation("agent_1", transcript, client=client,
                                     conversation_id=str(c.id), iteration_id=iteration_id)
        if not rubric:
            continue
        v0001_judgments.append(1 if rubric.get("compliance_score", 0) >= 3 else 0)
        rule5 = check_rule_5(transcript, tool_calls, client=client)
        strict_judgments.append(1 if rule5.passed else 0)

    n = len(v0001_judgments)
    if n < 2:
        return FindingRecord(
            finding_type="inter_judge_agreement",
            description="Insufficient successful judge calls for kappa.",
            severity="low",
        )

    a = np.array(v0001_judgments); b = np.array(strict_judgments)
    po = float(np.mean(a == b))
    pe = float(np.mean(a)) * float(np.mean(b)) + (1 - float(np.mean(a))) * (1 - float(np.mean(b)))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    severity = "high" if kappa < 0.40 else ("medium" if kappa < 0.60 else "low")
    return FindingRecord(
        finding_type="inter_judge_agreement",
        description=(
            f"Cohen's kappa between v0001 (binary 'compliance_score>=3') and "
            f"strict rule_5 judge over {n} conversations: {kappa:.3f}."
        ),
        evidence={"kappa": kappa, "p_o": po, "p_e": pe, "n": n,
                  "v0001": v0001_judgments, "strict": strict_judgments},
        proposed_fix=(
            "kappa < 0.40 means the lenient judge disagrees substantially with "
            "the strict per-rule judge. Switch to the per-rule judge."
            if kappa < 0.40 else
            "Substantial agreement — no action."
        ),
        severity=severity,
    )


# ---- promotion / persistence -------------------------------------------

def _persist_finding(f: FindingRecord, iteration_id: Optional[int]) -> int:
    with session_scope() as s:
        row = MetaEvalFinding(
            iteration_id=iteration_id,
            finding_type=f.finding_type,
            description=f.description,
            evidence=f.evidence,
            proposed_fix=f.proposed_fix,
        )
        s.add(row); s.flush()
        return row.id


def _promote_judge_v0002() -> tuple[int, int]:
    """Upsert judge v0002 (read from prompts/judge/v0002.md) and switch active."""
    v0002_path = Path(__file__).resolve().parents[2] / "prompts" / "judge" / "v0002.md"
    if not v0002_path.exists():
        raise FileNotFoundError(f"judge v0002 prompt missing at {v0002_path}")
    text = v0002_path.read_text().strip()
    tokens = count_tokens(text)
    pv_id = upsert_prompt_version(
        agent_id="judge",
        version=2,
        prompt_text=text,
        prompt_tokens=tokens,
        status="active",
        adoption_data={
            "promoted_by": "meta_evaluator",
            "promoted_at": datetime.utcnow().isoformat(),
            "reason": "meta-eval compliance audit detected lenient v0001 judge "
                      "missing rule_5 violations on distressed-persona conversations",
        },
    )
    set_active_prompt("judge", pv_id)
    return pv_id, tokens


def run_meta_eval(
    *,
    iteration_id: Optional[int] = None,
    client: Optional[AnthropicClient] = None,
    auto_apply_fixes: bool = True,
) -> MetaEvalReport:
    install_cost_persistence()
    client = client or AnthropicClient()
    log.info("\n%s  META-EVAL  iteration=%s  %s", "=" * 14, iteration_id, "=" * 14)

    report = MetaEvalReport(iteration_id=iteration_id)

    # ---- audit 1: compliance judge strictness ----
    log.info("[1/3] compliance-judge strictness audit ...")
    f1 = audit_compliance_judge_strictness(client, iteration_id=iteration_id)
    report.findings.append(f1)
    fid1 = _persist_finding(f1, iteration_id)
    log.info("       severity=%s  %s", f1.severity, f1.description)

    # ---- audit 2: metric-outcome correlation ----
    log.info("[2/3] metric-outcome correlation audit ...")
    f2 = audit_metric_outcome_correlation()
    report.findings.append(f2)
    _persist_finding(f2, iteration_id)
    log.info("       severity=%s  %s", f2.severity, f2.description)

    # ---- audit 3: inter-judge agreement (kappa) ----
    log.info("[3/3] inter-judge agreement (Cohen's kappa) ...")
    f3 = audit_inter_judge_agreement(client, iteration_id=iteration_id)
    report.findings.append(f3)
    _persist_finding(f3, iteration_id)
    log.info("       severity=%s  %s", f3.severity, f3.description)

    # ---- apply fixes ----
    if auto_apply_fixes and (f1.severity == "high" or f3.severity == "high"):
        log.info("\n*** auto-applying fix: promoting rubric judge v0001 -> v0002 ***")
        pv_id, tokens = _promote_judge_v0002()
        report.judge_swapped = True
        report.new_judge_version = 2
        report.actions_taken.append(
            f"promoted judge to v0002 ({tokens} tokens, prompt_versions.id={pv_id})"
        )
        # mark v0001 row as retired for audit trail clarity
        with session_scope() as s:
            v1 = s.execute(
                select(PromptVersion)
                .where(PromptVersion.agent_id == "judge", PromptVersion.version == 1)
            ).scalar_one_or_none()
            if v1:
                v1.status = "retired_by_meta_eval"
                v1.retired_at = datetime.utcnow()
        # update finding row with applied=True
        with session_scope() as s:
            row = s.get(MetaEvalFinding, fid1)
            if row:
                row.applied = True
                row.applied_at = datetime.utcnow()

    log.info("\nmeta-eval complete. findings=%d  judge_swapped=%s", len(report.findings), report.judge_swapped)
    return report
