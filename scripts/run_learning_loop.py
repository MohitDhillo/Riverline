"""Run the self-learning loop.

By default: 2 iterations on agent_1 with N=15 paired borrowers, 2 variants per
iteration. Adopted prompts land in `prompt_versions` (status='active'),
rejects land with status='rejected', evidence in `adoption_data`. CSV + JSON
summary per iteration in data/raw_evaluations/.

    python scripts/run_learning_loop.py --agent agent_1 --iters 2 --n 15

Default evaluation mode is `full`: every baseline and variant sample runs through
the real A1→summarizer→A2→summarizer→A3 path. Use `--eval-mode isolated` only
when you intentionally want the old cheaper stub-handoff evaluation.
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
                   choices=["agent_1", "agent_2", "agent_3"],
                   help="Agent to evolve. Ignored when --agents is set.")
    p.add_argument("--agents", default=None,
                   help="Comma-separated agents to rotate through per cycle, e.g. "
                        "'agent_1,agent_2,agent_3'. When set, --iters is the number "
                        "of CYCLES; every cycle runs one iteration per agent in order. "
                        "Lets the loop try multiple agents instead of repeatedly hammering one.")
    p.add_argument("--iters", type=int, default=2,
                   help="Iterations per agent (single-agent mode) or full cycles (--agents mode).")
    p.add_argument("--n", type=int, default=15, help="paired borrowers per iteration")
    p.add_argument("--variants", type=int, default=2, help="variants proposed per iteration")
    p.add_argument("--eval-mode", choices=["full", "isolated"], default="full",
                   help="full = run A1→A2→A3 for each sample; isolated = old cheap stub handoffs")
    args = p.parse_args()

    s = settings()
    if not s.anthropic_api_key or not s.anthropic_api_key.startswith("sk-ant-"):
        print("ERROR: ANTHROPIC_API_KEY missing in .env")
        return 2

    install_cost_persistence()
    init_schema()
    start_spend = budget().spent()
    print(f"start session spend: ${start_spend:.6f}\n")

    # Build the (agent, iteration_id) schedule. Single-agent mode keeps the old
    # behavior. --agents mode rotates: cycle 1 = (a1, a2, a3), cycle 2 = (a1, a2, a3), ...
    if args.agents:
        rotation = [a.strip() for a in args.agents.split(",") if a.strip()]
        for a in rotation:
            if a not in ("agent_1", "agent_2", "agent_3"):
                print(f"ERROR: unknown agent '{a}' in --agents"); return 2
        schedule: list[tuple[str, int]] = []
        iter_counter = 0
        for cycle in range(args.iters):
            for a in rotation:
                iter_counter += 1
                schedule.append((a, iter_counter))
        print(f"rotation: {' → '.join(rotation)} × {args.iters} cycle(s) = {len(schedule)} iterations\n")
    else:
        schedule = [(args.agent, i) for i in range(1, args.iters + 1)]

    adoptions: list[tuple[str, int, int]] = []
    try:
        for agent_id, it in schedule:
            print(f"\n══ cycle iteration {it}: agent={agent_id} ══")
            result = run_iteration(
                agent_id=agent_id,
                iteration_id=it,
                n_borrowers=args.n,
                n_variants=args.variants,
                eval_mode=args.eval_mode,
            )
            spent_now = budget().spent()
            print(f"\niteration {it} ({agent_id}) done. session spend: ${spent_now:.6f}  "
                  f"(this iter: ${spent_now - start_spend:.6f})")
            if result.adopted_variant_idx is None:
                print(f"  no variant adopted at iteration {it} for {agent_id}.")
            else:
                adoptions.append((agent_id, result.adopted_new_version, it))
                print(f"  ADOPTED on {agent_id}: variant {result.adopted_variant_idx} → "
                      f"{agent_id} v{result.adopted_new_version}")
    except BudgetExhausted as e:
        print(f"\n!! BUDGET EXHAUSTED — stopping cleanly: {e}")
        return 1

    print(f"\nfinal session spend: ${budget().spent():.6f} of ${settings().budget_total_usd:.2f}")
    if adoptions:
        print(f"\nadoptions this run ({len(adoptions)}):")
        for a, v, it in adoptions:
            print(f"  iter {it}: {a} → v{v}")
    else:
        print("\nno adoptions this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
