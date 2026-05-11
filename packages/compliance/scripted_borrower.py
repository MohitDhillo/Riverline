"""Deterministic borrower for compliance probes.

Plays a fixed list of borrower utterances in sequence. No LLM, no variability —
makes probe outcomes reproducible.
"""

from __future__ import annotations

from typing import Optional


class ScriptedBorrower:
    """Drop-in replacement for BorrowerSimulator for compliance probes."""

    class _Profile:
        def __init__(self, id: str, persona: str) -> None:
            self.id = id
            self.persona = persona

    def __init__(self, turns: list[str], probe_id: str) -> None:
        self._turns = list(turns)
        self._idx = 0
        self.profile = self._Profile(
            id="00000000-0000-0000-0000-000000000000",
            persona=f"probe_{probe_id}",
        )

    def reply(
        self,
        history: list[dict],
        conversation_id: Optional[str] = None,
        iteration_id: Optional[int] = None,
    ) -> str:
        if self._idx < len(self._turns):
            msg = self._turns[self._idx]
            self._idx += 1
            return msg
        # ran out of scripted turns — end with a neutral acknowledgement
        return "(no further input)"
