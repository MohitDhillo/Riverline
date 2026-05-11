"""Re-run only the four probes that errored in the previous smoke run.

Cheaper than re-running the whole suite. Useful after fixes that don't change
the broader system.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.compliance import run_probe
from packages.llm import AnthropicClient
from packages.llm.budget_tracker import budget
from packages.storage import init_schema
from packages.storage.repos import install_cost_persistence

FAILING_IDS = [
    "rule_2_no_false_threats__escalation",
    "rule_4_offers__demand_higher_discount",
    "rule_5_hardship__job_loss_late",
    "rule_8_privacy__full_account_request",
]


def main() -> int:
    install_cost_persistence()
    init_schema()
    judge = AnthropicClient()

    start_spend = budget().spent()
    print(f"start spend: ${start_spend:.6f}\n")

    passes = 0
    for pid in FAILING_IDS:
        out = run_probe(pid, judge_client=judge)
        status = "PASS ✓" if out.passed else "FAIL ✗"
        print(f"{status}  {pid:50s} ({out.rule_id})")
        if not out.passed:
            print(f"        evidence: {out.evidence}")
        if out.passed:
            passes += 1

    print(f"\n{passes}/{len(FAILING_IDS)} probes pass after fix")
    print(f"this run spent: ${budget().spent() - start_spend:.6f}")
    return 0 if passes == len(FAILING_IDS) else 1


if __name__ == "__main__":
    sys.exit(main())
