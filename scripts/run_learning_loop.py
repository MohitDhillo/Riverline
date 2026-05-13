"""Run the self-learning loop.

By default: 2 iterations on agent_1 with N=15 paired borrowers, 2 variants per
iteration. Adopted prompts land in `prompt_versions` (status='active'),
rejects land with status='rejected', evidence in `adoption_data`. CSV + JSON
summary per iteration in data/raw_evaluations/.

    python scripts/run_learning_loop.py --agent agent_1 --iters 2 --n 15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.learner.loop import run_iteration
from packages.config import settings
from packages.llm.budget_tracker import BudgetExhausted, budget
from packages.storage import init_schema
from packages.storage.repos import install_cost_persistence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="agent_1",
                   choices=["agent_1"],   # agent_2/3 in Day 4
                   help="Agent to evolve in this run.")
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--n", type=int, default=15, help="paired borrowers per iteration")
    p.add_argument("--variants", type=int, default=2, help="variants proposed per iteration")
    args = p.parse_args()

    s = settings()
    if not s.anthropic_api_key or not s.anthropic_api_key.startswith("sk-ant-"):
        print("ERROR: ANTHROPIC_API_KEY missing in .env"); return 2

    install_cost_persistence()
    init_schema()
    start_spend = budget().spent()
    print(f"start session spend: ${start_spend:.6f}\n")

    try:
        for it in range(1, args.iters + 1):
            result = run_iteration(
                agent_id=args.agent,
                iteration_id=it,
                n_borrowers=args.n,
                n_variants=args.variants,
            )
            spent_now = budget().spent()
            print(f"\niteration {it} done. session spend: ${spent_now:.6f}  "
                  f"(this iter: ${spent_now - start_spend:.6f})")
            if result.adopted_variant_idx is None:
                print(f"  no variant adopted at iteration {it}.")
            else:
                print(f"  ADOPTED: variant {result.adopted_variant_idx} → "
                      f"{args.agent} v{result.adopted_new_version}")
    except BudgetExhausted as e:
        print(f"\n!! BUDGET EXHAUSTED — stopping cleanly: {e}")
        return 1

    print(f"\nfinal session spend: ${budget().spent():.6f} of $20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
