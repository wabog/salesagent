from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from sales_agent.domain.phones import normalize_phone_number


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Channel(StrEnum):
    WHATSAPP = "whatsapp"


class Direction(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class PromptMode(StrEnum):
    PUBLISHED = "published"
    DRAFT = "draft"


class ActionType(StrEnum):
    UPDATE_STAGE = "update_stage"
    UPDATE_CONTACT_FIELDS = "update_contact_fields"
    APPEND_NOTE = "append_note"
    CREATE_FOLLOWUP = "create_followup"
    HANDOFF_HUMAN = "handoff_human"


class InboundMessage(BaseModel):
    message_id: str
    conversation_id: str
    phone_number: str
    text: str
    timestamp: datetime
    raw_payload: dict[str, Any]
    channel: Channel = Channel.WHATSAPP
    provider: str = "kapso"
    contact_name: str | None = None
    prompt_mode: PromptMode = PromptMode.PUBLISHED

    @field_validator("phone_number", mode="before")
    @classmethod
    def _normalize_phone_number(cls, value: str) -> str:
        return normalize_phone_number(value)


class OutboundMessage(BaseModel):
    conversation_id: str
    phone_number: str
    text: str
    channel: Channel = Channel.WHATSAPP
    provider: str = "kapso"

    @field_validator("phone_number", mode="before")
    @classmethod
    def _normalize_phone_number(cls, value: str) -> str:
        return normalize_phone_number(value)


class CRMContact(BaseModel):
    external_id: str
    phone_number: str
    full_name: str | None = None
    stage: str | None = None
    email: str | None = None
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("phone_number", mode="before")
    @classmethod
    def _normalize_phone_number(cls, value: str) -> str:
        return normalize_phone_number(value)


class ConversationMessage(BaseModel):
    message_id: str
    conversation_id: str
    phone_number: str
    direction: Direction
    text: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("phone_number", mode="before")
    @classmethod
    def _normalize_phone_number(cls, value: str) -> str:
        return normalize_phone_number(value)


class ProposedAction(BaseModel):
    type: ActionType
    reason: str
    args: dict[str, Any] = Field(default_factory=dict)


class PlanningResult(BaseModel):
    intent: str
    confidence: float = 0.0
    response_text: str
    actions: list[ProposedAction] = Field(default_factory=list)
    should_reply: bool = True


class ToolExecutionResult(BaseModel):
    action: ProposedAction
    success: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class AgentRunResult(BaseModel):
    run_id: str
    duplicate: bool = False
    intent: str
    response_text: str
    tool_results: list[ToolExecutionResult] = Field(default_factory=list)
    contact: CRMContact | None = None


class InboundProcessingResult(BaseModel):
    accepted: bool = True
    queued: bool = False
    duplicate: bool = False
    render_reply: bool = True
    aggregated_messages: int = 1
    run_id: str | None = None
    intent: str | None = None
    response_text: str = ""
    tool_results: list[ToolExecutionResult] = Field(default_factory=list)
    contact: CRMContact | None = None
