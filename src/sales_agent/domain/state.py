from __future__ import annotations

from typing import NotRequired, TypedDict

from sales_agent.domain.models import CRMContact, InboundMessage, PlanningResult, ToolExecutionResult


class AgentState(TypedDict):
    event: InboundMessage
    run_id: str
    duplicate: bool
    contact: CRMContact | None
    recent_messages: list[dict]
    semantic_memories: list[str]
    planning: PlanningResult
    tool_results: list[ToolExecutionResult]
    response_text: str
    send_reply: bool
    handoff_requested: bool
    persisted: bool
    errors: NotRequired[list[str]]
