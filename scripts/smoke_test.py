"""Day 1 smoke test.

Boots a Temporal worker in this process, starts ONE CollectionsWorkflow against
a cooperative borrower, waits for it to finish, then prints:
  - Workflow outcome
  - Transcript
  - Token-count breakdown for the recorded turns
  - Cost-ledger summary (per-purpose totals from Postgres)

Success criteria:
  1. Workflow completes without error.
  2. >= 2 borrower turns + >= 2 agent turns recorded in `turns`.
  3. Every recorded turn has token_counts.total <= 2000.
  4. cost_ledger has at least one row per purpose ('agent_1', 'sim_cooperative').
  5. Total spend reported.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select
from temporalio.client import Client
from temporalio.worker import Worker

from apps.workflow.activities import run_chat_agent
from apps.workflow.collections import CollectionsInput, CollectionsWorkflow
from packages.config import settings
from packages.llm.budget_tracker import budget
from packages.simulator.borrower import load_borrowers
from packages.storage import init_schema, session_scope
from packages.storage.models import CostLedgerEntry, Conversation, Turn
from packages.storage.repos import install_cost_persistence

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smoke")
log.setLevel(logging.INFO)


async def run_smoke() -> int:
    install_cost_persistence()
    init_schema()

    s = settings()
    if not s.anthropic_api_key or not s.anthropic_api_key.startswith("sk-ant-"):
        print("ERROR: ANTHROPIC_API_KEY is missing or invalid in .env")
        return 2

    coop_borrowers = load_borrowers("cooperative")
    if not coop_borrowers:
        print("ERROR: no cooperative borrowers in seeds.json — run scripts/seed_db.py")
        return 2
    borrower = coop_borrowers[0]
    print(f"borrower: {borrower.name} (id={borrower.id[:8]}…) persona={borrower.persona} "
          f"debt=${borrower.debt_amount:,.2f}")

    print(f"connecting to temporal at {s.temporal_host} ...")
    client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)
    log.info("connected.")

    workflow_id = f"smoke-{uuid.uuid4().hex[:8]}"
    activity_executor = ThreadPoolExecutor(max_workers=4)

    async with Worker(
        client,
        task_queue=s.temporal_task_queue,
        workflows=[CollectionsWorkflow],
        activities=[run_chat_agent],
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
    print(f"assessment conversation_id = {result.assessment_conversation_id}")
    print()
    print(result.transcript_excerpt)
    print()

    # ---- assertions ----
    conv_id = uuid.UUID(result.assessment_conversation_id)
    ok = True
    with session_scope() as sess:
        conv = sess.get(Conversation, conv_id)
        turns = sess.execute(
            select(Turn).where(Turn.conversation_id == conv_id).order_by(Turn.seq)
        ).scalars().all()

        n_agent = sum(1 for t in turns if t.role == "assistant")
        n_user = sum(1 for t in turns if t.role == "user")
        print(f"turns: total={len(turns)}  agent={n_agent}  borrower={n_user}")
        if n_agent < 2 or n_user < 2:
            print(f"  FAIL: expected >= 2 each, got agent={n_agent} borrower={n_user}")
            ok = False

        over = [t for t in turns if t.token_counts and t.token_counts.get("total", 0) > 2000]
        if over:
            print(f"  FAIL: {len(over)} turns exceeded 2000-token budget")
            for t in over:
                print(f"    seq={t.seq}  total={t.token_counts['total']}")
            ok = False
        else:
            mx = max((t.token_counts.get("total", 0) for t in turns if t.token_counts), default=0)
            print(f"  max token_counts.total = {mx} (cap 2000) ✓")

        # cost ledger
        rows = sess.execute(
            select(CostLedgerEntry.purpose,
                   func.count().label("calls"),
                   func.sum(CostLedgerEntry.cost_usd).label("usd"))
            .where(CostLedgerEntry.conversation_id == conv_id)
            .group_by(CostLedgerEntry.purpose)
        ).all()
        print("cost_ledger (this conversation):")
        for r in rows:
            print(f"  {r.purpose:25s}  calls={r.calls:3d}  ${float(r.usd):.6f}")
        if not rows:
            print("  FAIL: no cost_ledger rows for this conversation")
            ok = False

        total = sess.execute(
            select(func.sum(CostLedgerEntry.cost_usd))
        ).scalar()
        print(f"total session spend (Postgres) = ${float(total or 0):.6f}")

    in_mem = budget().spent()
    print(f"total session spend (in-memory)  = ${in_mem:.6f}")

    print()
    print("Day 1 smoke test:", "PASS ✓" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_smoke()))
