"""Interactive chat CLI.

Run the full A1 → summarize → A2 → summarize → A3 pipeline against either:
  - yourself (you type the borrower replies — default)
  - one of the seeded LLM personas (--mode sim)

Bypasses Temporal so the loop is snappy and easy to demo. Production / batch path
(Temporal workflow + simulated borrowers) lives in ``scripts/smoke_test.py``.

Default: human mode, random persona. Override with --persona / --borrower-id /
--mode if you want a specific setup.

Examples
--------
    make chat                              # human, random borrower
    make chat PERSONA=distressed           # human, distressed profile
    python scripts/chat.py --mode sim      # autoplay (LLM borrower)
    python scripts/chat.py --max-stage 2   # stop after Resolution
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.agents.agent_1 import AssessmentAgent
from packages.agents.agent_2 import ResolutionAgent
from packages.agents.agent_3 import FinalNoticeAgent
from packages.agents.base import BaseAgent
from packages.config import settings
from packages.llm.budget_tracker import budget
from packages.simulator.borrower import BorrowerProfile, BorrowerSimulator, load_borrowers
from packages.simulator.human_borrower import HumanBorrower
from packages.simulator.runner import run_chat_conversation
from packages.storage import init_schema
from packages.storage.repos import (
    create_conversation,
    install_cost_persistence,
    load_turns,
    record_handoff,
)
from packages.summarizer import summarize_for_handoff

# ANSI colors
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"


def _banner(title: str, width: int = 64, color: str = _GREEN) -> None:
    line = "═" * width
    pad = max(0, (width - len(title) - 2) // 2)
    inner = "═" * pad + " " + title + " " + "═" * pad
    inner = inner.ljust(width, "═")
    print(f"\n{color}{_BOLD}{line}{_RESET}")
    print(f"{color}{_BOLD}{inner}{_RESET}")
    print(f"{color}{_BOLD}{line}{_RESET}")


def _system(msg: str) -> None:
    print(f"{_DIM}[system] {msg}{_RESET}")


def _toolcall_line(tc: dict) -> str:
    args = tc.get("input") or {}
    return f"{_MAGENTA}🔧 {tc['name']}({json.dumps(args, separators=(',', ':'))}){_RESET}"


def pick_borrower(persona: str | None, borrower_id: str | None) -> BorrowerProfile:
    bs = load_borrowers()
    if borrower_id:
        for b in bs:
            if b.id.startswith(borrower_id):
                return b
        raise SystemExit(f"no borrower starting with id {borrower_id}")
    if persona:
        candidates = [b for b in bs if b.persona == persona]
        if not candidates:
            raise SystemExit(f"no borrowers with persona {persona}")
        return random.choice(candidates)
    return random.choice(bs)


def make_borrower(mode: str, profile: BorrowerProfile):
    if mode == "human":
        return HumanBorrower(profile)
    if mode == "sim":
        return BorrowerSimulator(profile)
    raise SystemExit(f"unknown borrower mode: {mode}")


def run_stage(
    stage: int,
    agent: BaseAgent,
    borrower,
    handoff_json: str,
    max_turns: int,
    conv_id: uuid.UUID,
) -> tuple[uuid.UUID, str, list[dict]]:
    label = {
        1: "STAGE 1 — ASSESSMENT (Agent 1, chat)",
        2: "STAGE 2 — RESOLUTION (Agent 2, text-mode voice)",
        3: "STAGE 3 — FINAL NOTICE (Agent 3, chat)",
    }[stage]
    _banner(label, color=_GREEN)
    _system(f"prompt v{agent.prompt_version_num} ({agent.system_prompt_tokens} tok)  "
            f"handoff in: {len(handoff_json)} char")
    new_conv_id, result = run_chat_conversation(
        agent,
        borrower,
        max_turns=max_turns,
        handoff=handoff_json,
        workflow_id=f"chat-stage{stage}",
        conversation_id=conv_id,
        persona=borrower.profile.persona,
    )
    # Surface tool calls visibly so the demo can show what the agent decided.
    if result.tool_calls:
        print(f"\n{_DIM}tool calls this stage:{_RESET}")
        for tc in result.tool_calls:
            print(f"  {_toolcall_line(tc)}")
    print()
    _system(f"stage {stage} outcome: {_BOLD}{result.outcome}{_RESET}{_DIM}  "
            f"({result.turns} turns, {len(result.tool_calls)} tool calls){_RESET}")
    return new_conv_id, result.outcome, result.tool_calls


def summarize_stage(conv_ids: list[uuid.UUID], to_agent: str) -> str:
    _banner(f"HANDOFF → {to_agent}", color=_CYAN)
    _system(f"summarizing {len(conv_ids)} conversation(s) into a ≤500-token JSON payload …")
    combined: list[dict] = []
    for cid in conv_ids:
        combined.extend(load_turns(cid))
    res = summarize_for_handoff(
        combined,
        to_agent=to_agent,
        conversation_id=str(conv_ids[-1]),
    )
    payload = res.payload.model_dump()
    record_handoff(
        conversation_id=conv_ids[-1],
        from_agent=combined[-1].get("agent_id", "unknown") if combined else "unknown",
        to_agent=to_agent,
        payload=payload,
        payload_tokens=res.payload_tokens,
        trimmed_fields=res.trimmed_fields or None,
    )
    print(f"{_CYAN}{_BOLD}handoff JSON ({res.payload_tokens}/500 tokens; trimmed: {res.trimmed_fields or 'none'}){_RESET}")
    print(f"{_DIM}{json.dumps(payload, indent=2)}{_RESET}")
    return res.payload.to_compact_json()


def main() -> int:
    p = argparse.ArgumentParser(description="Interactive chat with the 3-agent collections pipeline")
    p.add_argument("--mode", choices=["human", "sim"],
                   default=os.getenv("BORROWER_MODE", "human"),
                   help="Who plays the borrower. Default: env BORROWER_MODE or 'human'.")
    p.add_argument("--persona", default=None,
                   choices=[None, "cooperative", "combative", "evasive", "confused", "distressed"],
                   help="Filter to a specific persona. Default: random pick across all.")
    p.add_argument("--borrower-id", default=None,
                   help="Specific borrower UUID prefix (e.g. 2d877835).")
    p.add_argument("--max-stage", type=int, default=3, choices=[1, 2, 3],
                   help="Stop after this stage (default 3 = run full pipeline).")
    p.add_argument("--max-turns", type=int, default=12)
    args = p.parse_args()

    s = settings()
    if not s.anthropic_api_key or not s.anthropic_api_key.startswith("sk-ant-"):
        print("ERROR: ANTHROPIC_API_KEY missing in .env")
        return 2

    install_cost_persistence()
    init_schema()

    profile = pick_borrower(args.persona, args.borrower_id)
    borrower = make_borrower(args.mode, profile)
    start_spend = budget().spent()

    _banner("Riverline Collections — interactive CLI", color=_YELLOW)
    print(f"  mode    : {args.mode}")
    print(f"  borrower: {profile.name}  (persona={profile.persona}, debt=${profile.debt_amount:,.2f})")
    print(f"  prompts : agent_1 v? • agent_2 v? • agent_3 v? (looked up at stage start)")
    if args.mode == "human":
        print(f"  {_DIM}Press Ctrl+C to end early. Type 'stop contacting me' to test opt-out.{_RESET}")

    conv_ids: list[uuid.UUID] = []
    handoff_json = ""

    # ---- stage 1 ----
    a1 = AssessmentAgent()
    cid1 = create_conversation(
        borrower_id=uuid.UUID(profile.id),
        persona=profile.persona,
        agent_versions={"agent_1": a1.prompt_version_num},
    )
    conv_ids.append(cid1)
    _, outcome1, _ = run_stage(1, a1, borrower, "", args.max_turns, cid1)

    if outcome1 == "opt_out" or args.max_stage < 2:
        _system(f"pipeline ended after Agent 1 ({outcome1}).")
        _system(f"this session spent: ${budget().spent() - start_spend:.6f}")
        return 0

    # ---- handoff 1→2 ----
    handoff_json = summarize_stage(conv_ids, "to_agent_2")

    # ---- stage 2 ----
    a2 = ResolutionAgent()
    cid2 = create_conversation(
        borrower_id=uuid.UUID(profile.id),
        persona=profile.persona,
        agent_versions={"agent_2": a2.prompt_version_num},
    )
    conv_ids.append(cid2)
    _, outcome2, _ = run_stage(2, a2, borrower, handoff_json, args.max_turns, cid2)

    if outcome2 == "deal_agreed":
        _banner("** DEAL AGREED **", color=_GREEN)
        _system(f"pipeline complete. this session spent: ${budget().spent() - start_spend:.6f}")
        return 0
    if outcome2 == "opt_out" or args.max_stage < 3:
        _system(f"pipeline ended after Agent 2 ({outcome2}).")
        _system(f"this session spent: ${budget().spent() - start_spend:.6f}")
        return 0

    # ---- handoff 2→3 ----
    handoff_json = summarize_stage(conv_ids, "to_agent_3")

    # ---- stage 3 ----
    a3 = FinalNoticeAgent()
    cid3 = create_conversation(
        borrower_id=uuid.UUID(profile.id),
        persona=profile.persona,
        agent_versions={"agent_3": a3.prompt_version_num},
    )
    conv_ids.append(cid3)
    _, outcome3, _ = run_stage(3, a3, borrower, handoff_json, args.max_turns, cid3)

    if outcome3 == "resolved":
        _banner("** RESOLVED AT FINAL NOTICE **", color=_GREEN)
    else:
        _banner(f"** Final outcome: {outcome3} — account flagged **", color=_YELLOW)
    _system(f"this session spent: ${budget().spent() - start_spend:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
