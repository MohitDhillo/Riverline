"""Lift every currently-active agent prompt from the DB to disk.

After the learning loop adopts a new version (or the meta-evaluator promotes
the judge), the new prompt only lives in Postgres. This script writes it to
``prompts/<agent_id>/v0NNN.md`` so it shows up in the repo and survives a
fresh `make seed`.

Usage:
    make lift-prompts            # all 4 (agent_1, agent_2, agent_3, judge)
    uv run python scripts/lift_active_prompts.py --agent agent_2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.storage.repos import get_active_prompt

REPO = Path(__file__).resolve().parents[1]


def lift_one(agent_id: str) -> Path:
    pv = get_active_prompt(agent_id)
    sub_dir = REPO / "prompts" / agent_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    out_path = sub_dir / f"v{pv.version:04d}.md"
    out_path.write_text(pv.prompt_text)
    print(f"  {agent_id} v{pv.version} → {out_path}  ({pv.prompt_tokens} tokens)")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default=None,
                   help="Limit to one agent. Default: all 4 (agent_1, agent_2, agent_3, judge).")
    args = p.parse_args()

    targets = [args.agent] if args.agent else ["agent_1", "agent_2", "agent_3", "judge"]
    for a in targets:
        try:
            lift_one(a)
        except Exception as e:
            print(f"  {a}: SKIP ({e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
