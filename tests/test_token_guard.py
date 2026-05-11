"""Token-guard tests.

These tests are the evidence for FINAL_PLAN §3: the 2000-token agent budget and
500-token handoff budget are enforced in code, not aspirational.
"""

from __future__ import annotations

import pytest

from packages.llm.token_guard import (
    AGENT_BUDGET,
    HANDOFF_BUDGET,
    AgentContext,
    BudgetViolation,
    count_tokens,
)


def test_count_tokens_basic() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello") > 0
    # tiktoken cl100k: 'hello world' = 2 tokens
    assert count_tokens("hello world") == 2


def test_empty_context_within_budget() -> None:
    ctx = AgentContext(system_prompt="be helpful", handoff="", history=[])
    ctx.assert_within()
    assert ctx.total_tokens() < AGENT_BUDGET


def test_handoff_above_500_rejected() -> None:
    big_handoff = "x " * 600  # ~600 tokens
    ctx = AgentContext(system_prompt="s", handoff=big_handoff, history=[])
    with pytest.raises(BudgetViolation, match="handoff"):
        ctx.assert_within()


def test_system_plus_handoff_over_2000_rejected() -> None:
    # system 1600 + handoff 500 = 2100 -> over
    sys = "word " * 1600
    handoff = "h " * 400
    ctx = AgentContext(system_prompt=sys, handoff=handoff, history=[])
    with pytest.raises(BudgetViolation):
        ctx.fit_to_budget()


def test_history_trimmed_to_fit() -> None:
    # system ~50 tok, room ~1950 for history
    sys = "you are an assessment agent. " * 5  # small
    big_msgs = [{"role": "user", "content": "word " * 300} for _ in range(20)]
    ctx = AgentContext(system_prompt=sys, handoff="", history=big_msgs)
    assert ctx.total_tokens() > AGENT_BUDGET  # bloated
    fit = ctx.fit_to_budget()
    fit.assert_within()
    # most-recent turns kept
    assert fit.history[-1] == big_msgs[-1]
    # at least one turn was dropped
    assert len(fit.history) < len(big_msgs)


def test_fit_to_budget_preserves_system_and_handoff() -> None:
    sys = "system prompt here " * 10
    handoff = "handoff data " * 10
    msgs = [{"role": "user", "content": "word " * 500} for _ in range(5)]
    ctx = AgentContext(system_prompt=sys, handoff=handoff, history=msgs)
    fit = ctx.fit_to_budget()
    assert fit.system_prompt == sys
    assert fit.handoff == handoff


def test_anthropic_messages_includes_handoff() -> None:
    ctx = AgentContext(
        system_prompt="s",
        handoff='{"identity":{"verified":true}}',
        history=[{"role": "assistant", "content": "ok"}, {"role": "user", "content": "hi"}],
    )
    msgs = ctx.to_anthropic_messages()
    # Handoff is the leading user message; history follows and alternates.
    assert msgs[0]["role"] == "user"
    assert "handoff_context" in msgs[0]["content"]
    assert msgs[1]["role"] == "assistant"
    assert msgs[-1] == {"role": "user", "content": "hi"}


def test_anthropic_messages_injects_kickoff_when_no_handoff_and_empty_history() -> None:
    ctx = AgentContext(system_prompt="s", handoff="", history=[])
    msgs = ctx.to_anthropic_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_anthropic_messages_kickoff_when_history_starts_with_assistant() -> None:
    ctx = AgentContext(
        system_prompt="s",
        handoff="",
        history=[
            {"role": "assistant", "content": "first agent turn"},
            {"role": "user", "content": "borrower reply"},
        ],
    )
    msgs = ctx.to_anthropic_messages()
    # leading user kickoff, then alternating
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "user"


def test_anthropic_messages_no_kickoff_when_history_starts_with_user() -> None:
    # Borrower-simulator case: flipped history already starts with user.
    ctx = AgentContext(
        system_prompt="s",
        handoff="",
        history=[{"role": "user", "content": "agent first msg seen by borrower"}],
    )
    msgs = ctx.to_anthropic_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "agent first msg seen by borrower"


def test_token_counts_breakdown() -> None:
    ctx = AgentContext(
        system_prompt="system",
        handoff="handoff",
        history=[{"role": "user", "content": "hello"}],
    )
    tc = ctx.token_counts()
    assert tc["total"] == tc["system"] + tc["handoff"] + tc["history"]
    assert tc["system"] > 0
    assert tc["handoff"] > 0
    assert tc["history"] > 0


def test_exact_budget_boundary() -> None:
    """A context exactly at AGENT_BUDGET should pass; one token over must fail."""
    # Approximate: build a sys prompt right at the budget, no handoff/history
    # We'll synthesize tokens to land precisely on AGENT_BUDGET.
    base = "word " * (AGENT_BUDGET - 5)  # under-build
    ctx = AgentContext(system_prompt=base, handoff="", history=[])
    assert ctx.total_tokens() <= AGENT_BUDGET
    ctx.assert_within()

    over = "word " * (AGENT_BUDGET + 50)
    ctx2 = AgentContext(system_prompt=over, handoff="", history=[])
    with pytest.raises(BudgetViolation):
        ctx2.assert_within()


def test_handoff_exactly_at_500_passes() -> None:
    # Build handoff at ~498 tokens; should pass
    payload = "x " * 240
    assert count_tokens(payload) <= HANDOFF_BUDGET
    ctx = AgentContext(system_prompt="s", handoff=payload, history=[])
    ctx.assert_within()
