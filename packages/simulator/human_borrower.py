"""Human-driven borrower for interactive testing.

Drop-in replacement for ``BorrowerSimulator`` — same ``.reply(history, ...) -> str``
interface, but reads the borrower turn from stdin instead of calling an LLM.

Used by ``scripts/chat.py``. The automated learning loop always uses the LLM
simulator (you cannot have a human in the loop for 30 paired sims per iteration).
"""

from __future__ import annotations

import sys
import uuid
from typing import Optional

from packages.simulator.borrower import BorrowerProfile

_BLUE = "\033[94m"
_GREY = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


class HumanBorrower:
    """Asks for borrower replies on stdin. The profile (if given) is shown
    once at the start so the user can play in-character."""

    def __init__(
        self,
        profile: Optional[BorrowerProfile] = None,
        show_profile: bool = True,
    ) -> None:
        if profile is None:
            profile = BorrowerProfile(
                id=str(uuid.uuid4()),
                persona="human",
                name="You",
                age=35,
                debt_amount=0.0,
                last4_ssn="0000",
                dob="1990-01-01",
                employment="unknown",
                monthly_income="unknown",
                hardship=None,
                phone="+10000000000",
            )
        self.profile = profile
        self._first_reply = True
        self._show_profile = show_profile

    def _print_profile_card(self) -> None:
        p = self.profile
        print(f"\n{_BOLD}— You are playing the borrower —{_RESET}")
        print(f"{_GREY}  persona       {_RESET}{p.persona}")
        print(f"{_GREY}  name          {_RESET}{p.name}")
        print(f"{_GREY}  debt          {_RESET}${p.debt_amount:,.2f}")
        print(f"{_GREY}  last4 SSN     {_RESET}{p.last4_ssn}")
        print(f"{_GREY}  DOB           {_RESET}{p.dob}")
        print(f"{_GREY}  employment    {_RESET}{p.employment}")
        print(f"{_GREY}  monthly inc   {_RESET}{p.monthly_income}")
        if p.hardship:
            print(f"{_GREY}  hardship      {_RESET}{p.hardship}")
        print(f"{_GREY}  (use these values when the agent asks; deviate freely otherwise){_RESET}\n")

    def reply(
        self,
        history: list[dict],
        conversation_id: Optional[str] = None,
        iteration_id: Optional[int] = None,
    ) -> str:
        if self._first_reply and self._show_profile:
            self._print_profile_card()
            self._first_reply = False

        # Display the agent's last message so the user can respond to it.
        if history and history[-1]["role"] == "assistant":
            print(f"{_BLUE}{_BOLD}agent:{_RESET} {history[-1]['content']}\n")

        try:
            line = input(f"{_BOLD}you:{_RESET} ").strip()
        except EOFError:
            line = ""
        except KeyboardInterrupt:
            print("\n(interrupted — ending conversation)")
            return "I have to go now. Stop calling me."

        if not line:
            return "(no response)"
        return line
