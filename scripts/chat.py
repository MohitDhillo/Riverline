"""Interactive chat CLI.

Run a full A1 → summarize → A2 → summarize → A3 pipeline against either:
  - yourself (you type the borrower replies), or
  - one of the seeded LLM personas (cooperative / combative / evasive / confused / distressed).

This script bypasses Temporal so the loop is snappy and easy to demo. The
production / batch path (Temporal workflow + simulated borrowers) lives in
``scripts/smoke_test.py`` and the upcoming learning-loop driver.

Examples:
    # play yourself against the default cooperative borrower profile
    python scripts/chat.py

    # play yourself but as the distressed persona (so the agent's hardship-rule
    # behavior gets tested)
    python scripts/chat.py --persona distressed

    # autoplay — LLM cooperative borrower vs the agents (same as smoke test)
    python scripts/chat.py --mode sim --persona cooperative

    # stop after Agent 2
    python scripts/chat.py --max-stage 2
"""

from __future__ import annotations

import argparse
import os
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

_BOLD = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def pick_borrower(persona: str, borrower_id: str | None) -> BorrowerProfile:
    if borrower_id:
        for b in load_borrowers():
            if b.id.startswith(borrower_id):
                return b
        raise SystemExit(f"no borrower starting with id {borrower_id}")
    candidates = load_borrowers(persona)
    if not candidates:
        raise SystemExit(f"no borrowers with persona {persona}")
    return candidates[0]


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
) -> tuple[uuid.UUID, str]:
    label = {1: "ASSESSMENT (Agent 1, chat)",
             2: "RESOLUTION (Agent 2, text-mode voice)",
             3: "FINAL NOTICE (Agent 3, chat)"}[stage]
    print(f"\n{_BOLD}{_GREEN}========== {label} =========={_RESET}")
    new_conv_id, result = run_chat_conversation(
        agent,
        borrower,
        max_turns=max_turns,
        handoff=handoff_json,
        workflow_id=f"chat-{stage}",
        conversation_id=conv_id,
        persona=borrower.profile.persona,
    )
    print(f"\n{_BOLD}{_YELLOW}-- stage {stage} outcome: {result.outcome}  ({result.turns} turns, {result.summary_note}) --{_RESET}")
    return new_conv_id, result.outcome


def summarize_stage(conv_ids: list[uuid.UUID], to_agent: str) -> str:
    print(f"\n{_BOLD}{_YELLOW}-- summarizing {len(conv_ids)} conversation(s) for {to_agent} --{_RESET}")
    combined: list[dict] = []
    for cid in conv_ids:
        combined.extend(load_turns(cid))
    res = summarize_for_handoff(
        combined,
        to_agent=to_agent,
        conversation_id=str(conv_ids[-1]),
    )
    record_handoff(
        conversation_id=conv_ids[-1],
        from_agent=combined[-1].get("agent_id", "unknown") if combined else "unknown",
        to_agent=to_agent,
        payload=res.payload.model_dump(),
        payload_tokens=res.payload_tokens,
        trimmed_fields=res.trimmed_fields or None,
    )
    print(f"   handoff_tokens = {res.payload_tokens}/500   trimmed = {res.trimmed_fields or 'none'}")
    return res.payload.to_compact_json()


def main() -> int:
    p = argparse.ArgumentParser(description="Interactive chat with the 3-agent collections pipeline")
    p.add_argument("--mode", choices=["human", "sim"],
                   default=os.getenv("BORROWER_MODE", "human"),
                   help="Who plays the borrower. Default: env BORROWER_MODE or 'human'.")
    p.add_argument("--persona", default="cooperative",
                   choices=["cooperative", "combative", "evasive", "confused", "distressed"],
                   help="Persona to use (sim mode) or to play in-character (human mode).")
    p.add_argument("--borrower-id", default=None,
                   help="Specific borrower UUID prefix (e.g. 2d877835).")
    p.add_argument("--max-stage", type=int, default=3, choices=[1, 2, 3],
                   help="Stop after this stage (default 3 = run full pipeline).")
    p.add_argument("--max-turns", type=int, default=12)
    args = p.parse_args()

    s = settings()
    if not s.anthropic_api_key or not s.anthropic_api_key.startswith("sk-ant-"):
        print("ERROR: ANTHROPIC_API_KEY missing in .env"); return 2

    install_cost_persistence()
    init_schema()

    profile = pick_borrower(args.persona, args.borrower_id)
    borrower = make_borrower(args.mode, profile)
    start_spend = budget().spent()

    print(f"\n{_BOLD}Riverline chat CLI{_RESET}")
    print(f"  mode={args.mode}  persona={profile.persona}  borrower={profile.name} (debt ${profile.debt_amount:,.2f})")
    if args.mode == "human":
        print(f"  Press Ctrl+C to end early. Type 'stop contacting me' to test the opt-out path.")

    conv_ids: list[uuid.UUID] = []
    handoff_json = ""

    # ---- stage 1 ----
    cid1 = create_conversation(
        borrower_id=uuid.UUID(profile.id),
        persona=profile.persona,
        agent_versions={"agent_1": AssessmentAgent().prompt_version_num},
    )
    conv_ids.append(cid1)
    _, outcome1 = run_stage(1, AssessmentAgent(), borrower, "", args.max_turns, cid1)

    if outcome1 == "opt_out" or args.max_stage < 2:
        print(f"\n{_BOLD}pipeline ended after Agent 1.{_RESET}")
        print(f"this session spent: ${budget().spent() - start_spend:.6f}")
        return 0

    # ---- handoff 1→2 ----
    handoff_json = summarize_stage(conv_ids, "to_agent_2")

    # ---- stage 2 ----
    cid2 = create_conversation(
        borrower_id=uuid.UUID(profile.id),
        persona=profile.persona,
        agent_versions={"agent_2": ResolutionAgent().prompt_version_num},
    )
    conv_ids.append(cid2)
    _, outcome2 = run_stage(2, ResolutionAgent(), borrower, handoff_json, args.max_turns, cid2)

    if outcome2 == "deal_agreed":
        print(f"\n{_BOLD}{_GREEN}** DEAL AGREED at Resolution. Pipeline complete. **{_RESET}")
        print(f"this session spent: ${budget().spent() - start_spend:.6f}")
        return 0
    if outcome2 == "opt_out" or args.max_stage < 3:
        print(f"\n{_BOLD}pipeline ended after Agent 2.{_RESET}")
        print(f"this session spent: ${budget().spent() - start_spend:.6f}")
        return 0

    # ---- handoff 2→3 ----
    handoff_json = summarize_stage(conv_ids, "to_agent_3")

    # ---- stage 3 ----
    cid3 = create_conversation(
        borrower_id=uuid.UUID(profile.id),
        persona=profile.persona,
        agent_versions={"agent_3": FinalNoticeAgent().prompt_version_num},
    )
    conv_ids.append(cid3)
    _, outcome3 = run_stage(3, FinalNoticeAgent(), borrower, handoff_json, args.max_turns, cid3)

    if outcome3 == "resolved":
        print(f"\n{_BOLD}{_GREEN}** RESOLVED at Final Notice. **{_RESET}")
    else:
        print(f"\n{_BOLD}{_YELLOW}** Final outcome: {outcome3}. Account flagged. **{_RESET}")
    print(f"this session spent: ${budget().spent() - start_spend:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
