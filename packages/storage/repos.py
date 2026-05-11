"""Thin repository helpers around the SQLAlchemy models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from packages.llm.budget_tracker import CostRecord, budget
from packages.storage.db import session_scope
from packages.storage.models import (
    ActivePrompt,
    Conversation,
    CostLedgerEntry,
    Handoff,
    PromptVersion,
    Turn,
)


def get_active_prompt(agent_id: str) -> PromptVersion:
    with session_scope() as s:
        ap = s.execute(
            select(ActivePrompt).where(ActivePrompt.agent_id == agent_id)
        ).scalar_one_or_none()
        if not ap:
            raise RuntimeError(f"no active prompt for {agent_id} — run seed_db.py first")
        pv = s.get(PromptVersion, ap.version_id)
        s.expunge(pv)
        return pv


def upsert_prompt_version(
    agent_id: str,
    version: int,
    prompt_text: str,
    prompt_tokens: int,
    status: str = "active",
    parent_version: Optional[int] = None,
    adoption_data: Optional[dict] = None,
) -> int:
    with session_scope() as s:
        existing = s.execute(
            select(PromptVersion)
            .where(PromptVersion.agent_id == agent_id, PromptVersion.version == version)
        ).scalar_one_or_none()
        if existing:
            existing.prompt_text = prompt_text
            existing.prompt_tokens = prompt_tokens
            existing.status = status
            if adoption_data is not None:
                existing.adoption_data = adoption_data
            s.flush()
            return existing.id
        pv = PromptVersion(
            agent_id=agent_id,
            version=version,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            status=status,
            parent_version=parent_version,
            adoption_data=adoption_data,
            activated_at=datetime.utcnow() if status == "active" else None,
        )
        s.add(pv)
        s.flush()
        return pv.id


def set_active_prompt(agent_id: str, version_id: int) -> None:
    with session_scope() as s:
        ap = s.get(ActivePrompt, agent_id)
        if ap:
            ap.version_id = version_id
        else:
            s.add(ActivePrompt(agent_id=agent_id, version_id=version_id))


def create_conversation(
    borrower_id: uuid.UUID,
    workflow_id: Optional[str] = None,
    persona: Optional[str] = None,
    iteration_id: Optional[int] = None,
    agent_versions: Optional[dict] = None,
) -> uuid.UUID:
    with session_scope() as s:
        c = Conversation(
            borrower_id=borrower_id,
            workflow_id=workflow_id,
            persona=persona,
            iteration_id=iteration_id,
            agent_versions=agent_versions,
        )
        s.add(c)
        s.flush()
        return c.id


def add_turn(
    conversation_id: uuid.UUID,
    seq: int,
    agent_id: str,
    role: str,
    content: str,
    token_counts: Optional[dict] = None,
    tool_calls: Optional[dict] = None,
) -> None:
    with session_scope() as s:
        s.add(Turn(
            conversation_id=conversation_id,
            seq=seq,
            agent_id=agent_id,
            role=role,
            content=content,
            token_counts=token_counts,
            tool_calls=tool_calls,
        ))


def load_turns(conversation_id: uuid.UUID) -> list[dict]:
    """Return [{agent_id, role, content, seq}] in order."""
    with session_scope() as s:
        rows = s.execute(
            select(Turn).where(Turn.conversation_id == conversation_id).order_by(Turn.seq)
        ).scalars().all()
        return [
            {"agent_id": t.agent_id, "role": t.role, "content": t.content, "seq": t.seq}
            for t in rows
        ]


def end_conversation(conversation_id: uuid.UUID, outcome: str) -> None:
    with session_scope() as s:
        c = s.get(Conversation, conversation_id)
        if c:
            c.ended_at = datetime.utcnow()
            c.outcome = outcome


def record_handoff(
    conversation_id: uuid.UUID,
    from_agent: str,
    to_agent: str,
    payload: dict,
    payload_tokens: int,
    trimmed_fields: Optional[dict] = None,
) -> int:
    with session_scope() as s:
        h = Handoff(
            conversation_id=conversation_id,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            payload_tokens=payload_tokens,
            trimmed_fields=trimmed_fields,
        )
        s.add(h)
        s.flush()
        return h.id


def persist_cost_record(rec: CostRecord) -> None:
    with session_scope() as s:
        s.add(CostLedgerEntry(
            timestamp=rec.timestamp,
            provider=rec.provider,
            model=rec.model,
            purpose=rec.purpose,
            input_tokens=rec.input_tokens,
            output_tokens=rec.output_tokens,
            cache_write_tokens=rec.cache_write_tokens,
            cache_read_tokens=rec.cache_read_tokens,
            cost_usd=rec.cost_usd,
            iteration_id=rec.iteration_id,
            conversation_id=uuid.UUID(rec.conversation_id) if rec.conversation_id else None,
        ))


def install_cost_persistence() -> None:
    budget().install_persist_hook(persist_cost_record)
