"""Run the meta-evaluator (Darwin-Gödel layer).

Three audits:
  1. compliance-judge strictness on distressed-persona conversations
  2. metric-outcome correlation across recent learning-loop conversations
  3. inter-judge agreement (Cohen's kappa) between v0001 and strict per-rule

If audit 1 or 3 returns severity='high', the meta-evaluator auto-swaps the
rubric judge from v0001 to v0002 (per-rule checklist) and records a
MetaEvalFinding row with applied=True.

Outputs:
  - meta_eval_findings rows in Postgres
  - data/raw_evaluations/meta_eval_<timestamp>.json with the full report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.learner.meta_evaluator import run_meta_eval
from packages.config import settings
from packages.llm.budget_tracker import budget
from packages.storage import init_schema
from packages.storage.repos import install_cost_persistence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, default=None,
                   help="Tag this meta-eval with a learning-loop iteration id.")
    p.add_argument("--no-apply", action="store_true",
                   help="Run audits but DO NOT auto-swap the judge.")
    args = p.parse_args()

    s = settings()
    if not s.anthropic_api_key or not s.anthropic_api_key.startswith("sk-ant-"):
        print("ERROR: ANTHROPIC_API_KEY missing in .env"); return 2

    install_cost_persistence()
    init_schema()
    start_spend = budget().spent()

    report = run_meta_eval(
        iteration_id=args.iteration,
        auto_apply_fixes=not args.no_apply,
    )

    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw_evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"meta_eval_{ts}.json"
    out_path.write_text(json.dumps({
        "iteration_id": report.iteration_id,
        "judge_swapped": report.judge_swapped,
        "new_judge_version": report.new_judge_version,
        "actions_taken": report.actions_taken,
        "findings": [
            {
                "type": f.finding_type,
                "severity": f.severity,
                "description": f.description,
                "evidence": f.evidence,
                "proposed_fix": f.proposed_fix,
            }
            for f in report.findings
        ],
    }, indent=2, default=str))
    print(f"\nwrote {out_path}")
    print(f"this run spent: ${budget().spent() - start_spend:.6f}")
    print(f"session total : ${budget().spent():.6f} of $20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
