"""End-to-end smoke test.

Day 1: Agent 1 only.
Day 2 (now): full A1 → summarize → A2 → summarize → A3 pipeline + handoff tokens
verified <=500 + compliance probe suite against v0 prompts.

Success criteria:
  1. Workflow completes without error.
  2. All three agent stages produced turns (unless an opt-out short-circuited).
  3. Each handoff payload <=500 tokens.
  4. Every agent turn has token_counts.total <= 2000.
  5. cost_ledger populated; total < $1 for one full pipeline.
  6. Compliance probe suite passes 100% on v0 prompts (or we know exactly which rules fail and why).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select
from temporalio.client import Client
from temporalio.worker import Worker

from apps.workflow.activities import run_chat_agent, summarize_handoff
from apps.workflow.collections import CollectionsInput, CollectionsWorkflow
from packages.compliance import run_probe_suite
from packages.config import settings
from packages.llm.budget_tracker import budget
from packages.simulator.borrower import load_borrowers
from packages.storage import init_schema, session_scope
from packages.storage.models import CostLedgerEntry, Handoff, Turn
from packages.storage.repos import install_cost_persistence

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smoke")
log.setLevel(logging.INFO)


async def run_pipeline_once() -> tuple[bool, dict]:
    install_cost_persistence()
    init_schema()

    s = settings()
    if not s.anthropic_api_key or not s.anthropic_api_key.startswith("sk-ant-"):
        return False, {"error": "ANTHROPIC_API_KEY missing or invalid in .env"}

    coop_borrowers = load_borrowers("cooperative")
    if not coop_borrowers:
        return False, {"error": "no cooperative borrowers — run scripts/seed_db.py"}
    borrower = coop_borrowers[0]
    print(f"borrower: {borrower.name} (id={borrower.id[:8]}…) persona={borrower.persona} "
          f"debt=${borrower.debt_amount:,.2f}")

    print(f"connecting to temporal at {s.temporal_host} ...")
    client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)

    workflow_id = f"smoke-{uuid.uuid4().hex[:8]}"
    activity_executor = ThreadPoolExecutor(max_workers=4)

    async with Worker(
        client,
        task_queue=s.temporal_task_queue,
        workflows=[CollectionsWorkflow],
        activities=[run_chat_agent, summarize_handoff],
        activity_executor=activity_executor,
    ):
        print(f"worker up. starting workflow {workflow_id} ...")
        handle = await client.start_workflow(
            CollectionsWorkflow.run,
            CollectionsInput(borrower_id=borrower.id, iteration_id=None),
            id=workflow_id,
            task_queue=s.temporal_task_queue,
        )
        result = await handle.result()

    print()
    print("=" * 72)
    print(f"WORKFLOW OUTCOME: {result.outcome}")
    print("=" * 72)
    print(f"agent_1 conv={result.assessment_conversation_id}")
    print(f"agent_2 conv={result.resolution_conversation_id}  handoff={result.handoff_1_to_2_tokens}t")
    print(f"agent_3 conv={result.final_conversation_id}  handoff={result.handoff_2_to_3_tokens}t")
    print()
    for stage in ("agent_1", "agent_2", "agent_3"):
        excerpt = result.excerpts.get(stage)
        if excerpt:
            print(f"--- {stage} excerpt ---")
            print(excerpt)
            print()

    # ---- assertions ----
    ok = True
    failures: list[str] = []

    convs = [c for c in (
        result.assessment_conversation_id,
        result.resolution_conversation_id,
        result.final_conversation_id,
    ) if c]
    if not convs:
        failures.append("no conversations recorded")
        ok = False

    with session_scope() as sess:
        # token budget check across every recorded turn
        all_turns = sess.execute(
            select(Turn).where(Turn.conversation_id.in_([uuid.UUID(c) for c in convs]))
        ).scalars().all()
        over = [t for t in all_turns if t.token_counts and t.token_counts.get("total", 0) > 2000]
        if over:
            failures.append(f"{len(over)} turns exceeded 2000-token budget")
            ok = False
        max_tok = max((t.token_counts.get("total", 0) for t in all_turns if t.token_counts), default=0)
        print(f"turn count={len(all_turns)}  max token_counts.total = {max_tok} (cap 2000)")

        # handoff budget check
        handoffs = sess.execute(
            select(Handoff).where(Handoff.conversation_id.in_([uuid.UUID(c) for c in convs]))
        ).scalars().all()
        for h in handoffs:
            tag = f"{h.from_agent}→{h.to_agent}"
            ok_h = h.payload_tokens <= 500
            print(f"  handoff {tag:30s}  {h.payload_tokens}t  {'✓' if ok_h else 'FAIL'}")
            if not ok_h:
                failures.append(f"handoff {tag} = {h.payload_tokens}t > 500")
                ok = False

        # cost ledger
        rows = sess.execute(
            select(CostLedgerEntry.purpose,
                   func.count().label("calls"),
                   func.sum(CostLedgerEntry.cost_usd).label("usd"))
            .where(CostLedgerEntry.conversation_id.in_([uuid.UUID(c) for c in convs]))
            .group_by(CostLedgerEntry.purpose)
        ).all()
        print("cost_ledger (this pipeline):")
        pipeline_cost = 0.0
        for r in rows:
            print(f"  {r.purpose:25s}  calls={r.calls:3d}  ${float(r.usd):.6f}")
            pipeline_cost += float(r.usd)
        print(f"  pipeline total = ${pipeline_cost:.6f}")
        if pipeline_cost > 1.0:
            failures.append(f"pipeline cost ${pipeline_cost:.4f} > $1 (unexpected)")
            ok = False

    print(f"\nin-memory budget tracker: ${budget().spent():.6f}")

    return ok, {
        "outcome": result.outcome,
        "convs": convs,
        "max_token": max_tok,
        "handoff_tokens": [h.payload_tokens for h in handoffs],
        "failures": failures,
    }


def run_compliance_probes() -> tuple[bool, dict]:
    print("\n" + "=" * 72)
    print("COMPLIANCE PROBE SUITE (v0 prompts)")
    print("=" * 72)
    suite = run_probe_suite()
    print(suite.summary())
    failed = suite.failed()
    if failed:
        print(f"\nFAILED PROBES ({len(failed)}):")
        for f in failed:
            print(f"  - {f.probe_id} ({f.rule_id})  evidence={f.evidence}")
    return suite.all_pass(), {
        "total": len(suite.outcomes),
        "passed": sum(1 for o in suite.outcomes if o.passed),
        "failed_ids": [f.probe_id for f in failed],
    }


async def main() -> int:
    print("\n========= Day 2 smoke test: full A1→A2→A3 pipeline =========\n")
    pipeline_ok, pipeline_info = await run_pipeline_once()

    print("\n========= Compliance probe suite (16 probes) =========")
    compliance_ok, compliance_info = run_compliance_probes()

    print("\n========= TOTAL SPEND =========")
    print(f"  ${budget().spent():.6f} of $20 budget")

    print("\n========= RESULT =========")
    print(f"  pipeline   : {'PASS ✓' if pipeline_ok else 'FAIL ✗'}")
    print(f"  compliance : {compliance_info['passed']}/{compliance_info['total']} probes passed")
    if pipeline_ok and compliance_info["passed"] == compliance_info["total"]:
        print("\nDay 2 smoke test: PASS ✓")
        return 0
    print("\nDay 2 smoke test: see failures above")
    return 0 if pipeline_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
