from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Channel(StrEnum):
    WHATSAPP = "whatsapp"


class Direction(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class ActionType(StrEnum):
    CREATE_CONTACT = "create_contact"
    UPDATE_STAGE = "update_stage"
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


class OutboundMessage(BaseModel):
    conversation_id: str
    phone_number: str
    text: str
    channel: Channel = Channel.WHATSAPP
    provider: str = "kapso"


class CRMContact(BaseModel):
    external_id: str
    phone_number: str
    full_name: str | None = None
    stage: str | None = None
    email: str | None = None
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    message_id: str
    conversation_id: str
    phone_number: str
    direction: Direction
    text: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
