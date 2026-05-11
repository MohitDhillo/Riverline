"""SQLAlchemy models for FINAL_PLAN §7.

Tables:
    prompt_versions       — versioned prompts per agent (incl. judge & simulator)
    active_prompt         — single-row-per-agent pointer to the current live version
    conversations         — one row per simulated or production conversation
    turns                 — every message + per-turn token counts (evidence!)
    handoffs              — the 500-token payloads (with trim audit)
    evaluations           — judge scores per conversation per agent
    compliance_checks     — per-rule pass/fail records
    cost_ledger           — every LLM call, the spend deliverable
    meta_eval_findings    — Darwin-Gödel layer's catches
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String(40), index=True)
    version: Mapped[int] = mapped_column(Integer)
    prompt_text: Mapped[str] = mapped_column(Text)
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    parent_version: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("prompt_versions.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), default="candidate")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    activated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    retired_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    adoption_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ActivePrompt(Base):
    __tablename__ = "active_prompt"

    agent_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    version_id: Mapped[int] = mapped_column(Integer, ForeignKey("prompt_versions.id"))


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    borrower_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    workflow_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    iteration_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    persona: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    agent_versions: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    outcome: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)

    turns: Mapped[list["Turn"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Turn.seq"
    )


class Turn(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    agent_id: Mapped[str] = mapped_column(String(40))  # 'agent_1' | 'agent_2' | 'agent_3' | 'borrower'
    role: Mapped[str] = mapped_column(String(20))  # 'user' | 'assistant' | 'tool'
    content: Mapped[str] = mapped_column(Text)
    token_counts: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    tool_calls: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="turns")


class Handoff(Base):
    __tablename__ = "handoffs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    from_agent: Mapped[str] = mapped_column(String(40))
    to_agent: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSONB)
    payload_tokens: Mapped[int] = mapped_column(Integer)
    trimmed_fields: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    agent_id: Mapped[str] = mapped_column(String(40))
    judge_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rubric: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    outcome_metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    compliance: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ComplianceCheck(Base):
    __tablename__ = "compliance_checks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    prompt_version_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("prompt_versions.id"), nullable=True
    )
    rule_id: Mapped[str] = mapped_column(String(40))
    passed: Mapped[bool] = mapped_column(Boolean)
    evidence: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CostLedgerEntry(Base):
    __tablename__ = "cost_ledger"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    provider: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(60))
    purpose: Mapped[str] = mapped_column(String(40), index=True)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 8))
    iteration_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)


class MetaEvalFinding(Base):
    __tablename__ = "meta_eval_findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    iteration_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    finding_type: Mapped[str] = mapped_column(String(60))
    description: Mapped[str] = mapped_column(Text)
    evidence: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    proposed_fix: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    applied: Mapped[bool] = mapped_column(Boolean, default=False)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
