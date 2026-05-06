from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from langgraph.graph import END, StateGraph

from sales_agent.domain.models import (
    ActionType,
    AgentRunResult,
    ConversationMessage,
    Direction,
    OutboundMessage,
    ToolExecutionResult,
)
from sales_agent.domain.state import AgentState
from sales_agent.services.calendar_sync import enrich_contact_with_calendar, merge_contact_with_shadow
from sales_agent.services.lead_scope import LeadScopedCRMTools
from sales_agent.services.name_validation import get_effective_contact_name
from sales_agent.services.planner import AgentPlanner
from sales_agent.services.policy import ToolExecutionPolicy


def _merge_unique_messages(*message_groups: list[ConversationMessage], limit: int) -> list[ConversationMessage]:
    deduped: dict[str, ConversationMessage] = {}
    for group in message_groups:
        for message in group:
            deduped[message.message_id] = message
    merged = sorted(deduped.values(), key=lambda item: item.created_at)
    return merged[-limit:]


def _merge_unique_texts(*text_groups: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in text_groups:
        for text in group:
            normalized = text.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged[-limit:]


def _ensure_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class SalesAgentWorkflow:
    def __init__(
        self,
        crm_adapter,
        memory_store,
        channel_adapter,
        planner: AgentPlanner,
        policy: ToolExecutionPolicy,
        calendar_adapter=None,
    ):
        self.crm = crm_adapter
        self.memory = memory_store
        self.channel = channel_adapter
        self.planner = planner
        self.policy = policy
        self.calendar = calendar_adapter
        self.graph = self._build_graph().compile()

    async def run(self, event) -> AgentRunResult:
        initial_state: AgentState = {
            "event": event,
            "run_id": uuid4().hex,
            "duplicate": False,
            "current_lead": None,
            "lead_created": False,
            "recent_messages": [],
            "semantic_memories": [],
            "planning": None,
            "tool_results": [],
            "response_text": "",
            "send_reply": False,
            "handoff_requested": False,
            "persisted": False,
            "errors": [],
        }
        final_state = await self.graph.ainvoke(initial_state)
        return AgentRunResult(
            run_id=final_state["run_id"],
            duplicate=final_state["duplicate"],
            intent=final_state["planning"].intent if final_state["planning"] else "duplicate",
            response_text=final_state["response_text"],
            tool_results=final_state["tool_results"],
            contact=final_state["current_lead"],
        )

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("dedupe_and_lock", self._dedupe_and_lock)
        graph.add_node("resolve_current_lead", self._resolve_current_lead)
        graph.add_node("load_conversation_context", self._load_conversation_context)
        graph.add_node("classify_and_plan", self._classify_and_plan)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("persist_state", self._persist_state)
        graph.add_node("send_reply", self._send_reply)

        graph.set_entry_point("dedupe_and_lock")
        graph.add_conditional_edges(
            "dedupe_and_lock",
            self._route_duplicate,
            {
                "duplicate_end": END,
                "resolve_current_lead": "resolve_current_lead",
            },
        )
        graph.add_edge("resolve_current_lead", "load_conversation_context")
        graph.add_edge("load_conversation_context", "classify_and_plan")
        graph.add_edge("classify_and_plan", "execute_tools")
        graph.add_edge("execute_tools", "persist_state")
        graph.add_conditional_edges(
            "persist_state",
            self._route_reply,
            {
                "send_reply": "send_reply",
                "end": END,
            },
        )
        graph.add_edge("send_reply", END)
        return graph

    async def _dedupe_and_lock(self, state: AgentState) -> AgentState:
        if await self.memory.has_message(state["event"].message_id):
            state["duplicate"] = True
            state["response_text"] = ""
            state["send_reply"] = False
        return state

    def _route_duplicate(self, state: AgentState) -> str:
        return "duplicate_end" if state["duplicate"] else "resolve_current_lead"

    async def _resolve_current_lead(self, state: AgentState) -> AgentState:
        contact = await self.crm.find_contact_by_phone(state["event"].phone_number)
        shadow = await self.memory.get_contact_shadow(state["event"].phone_number)
        if contact is None:
            contact = await self.crm.create_contact(
                phone_number=state["event"].phone_number,
                full_name=get_effective_contact_name(shadow),
            )
            state["lead_created"] = True
        if shadow is not None:
            contact = merge_contact_with_shadow(contact, shadow)
        contact = await enrich_contact_with_calendar(
            contact,
            self.calendar,
            getattr(self.planner, "_settings", None).google_calendar_self_schedule_url if getattr(self.planner, "_settings", None) else None,
        )
        state["current_lead"] = contact
        return state

    async def _load_conversation_context(self, state: AgentState) -> AgentState:
        context_limit = getattr(getattr(self.planner, "_settings", None), "recent_message_context_limit", 24)
        semantic_limit = getattr(getattr(self.planner, "_settings", None), "semantic_memory_limit", 10)
        phone_context_max_age_days = getattr(getattr(self.planner, "_settings", None), "phone_context_max_age_days", 14)
        conversation_recent = await self.memory.get_recent_messages(state["event"].conversation_id, limit=context_limit)
        lead_recent = await self.memory.get_recent_messages_by_phone(
            state["event"].phone_number,
            limit=max(context_limit * 2, context_limit),
        )
        lead_recent = [
            message
            for message in lead_recent
            if _ensure_utc_datetime(message.created_at)
            >= _ensure_utc_datetime(state["event"].timestamp) - timedelta(days=phone_context_max_age_days)
        ]
        recent = _merge_unique_messages(conversation_recent, lead_recent, limit=context_limit)
        memories = _merge_unique_texts(
            await self.memory.search_memories(state["event"].conversation_id, state["event"].text, limit=semantic_limit),
            *(
                [await self.memory.search_memories_by_phone(state["event"].phone_number, state["event"].text, limit=semantic_limit)]
                if lead_recent
                else []
            ),
            limit=semantic_limit,
        )
        state["recent_messages"] = [message.model_dump(mode="json") for message in recent]
        state["semantic_memories"] = memories
        return state

    async def _classify_and_plan(self, state: AgentState) -> AgentState:
        planning = await self.planner.plan(
            text=state["event"].text,
            contact=state["current_lead"],
            recent_messages=[message["text"] for message in state["recent_messages"]],
            semantic_memories=state["semantic_memories"],
        )
        state["planning"] = planning
        state["response_text"] = planning.response_text.strip()
        state["send_reply"] = planning.should_reply
        return state

    async def _execute_tools(self, state: AgentState) -> AgentState:
        planning = state["planning"]
        current_lead = state["current_lead"]
        scoped_tools = LeadScopedCRMTools(self.crm, current_lead, calendar_adapter=self.calendar)
        results: list[ToolExecutionResult] = []
        for action in planning.actions:
            try:
                if action.type == ActionType.UPDATE_STAGE and not action.args.get("stage"):
                    continue
                self.policy.validate(action)
                if action.type == ActionType.UPDATE_STAGE:
                    current_lead = await scoped_tools.update_stage(action.args["stage"])
                    payload = current_lead.model_dump(mode="json")
                elif action.type == ActionType.UPDATE_CONTACT_FIELDS:
                    current_lead = await scoped_tools.update_fields(action.args["fields"])
                    payload = current_lead.model_dump(mode="json")
                elif action.type == ActionType.APPEND_NOTE:
                    current_lead = await scoped_tools.add_note(action.args["note"])
                    payload = current_lead.model_dump(mode="json")
                elif action.type == ActionType.CREATE_FOLLOWUP:
                    payload = await scoped_tools.create_followup(
                        action.args["summary"],
                        due_date=action.args.get("due_date"),
                    )
                    current_lead = scoped_tools.current_lead
                elif action.type == ActionType.COMPLETE_FOLLOWUP:
                    payload = await scoped_tools.complete_followup(
                        outcome=action.args.get("outcome"),
                    )
                    current_lead = scoped_tools.current_lead
                elif action.type == ActionType.CREATE_MEETING:
                    payload = await scoped_tools.create_meeting(
                        start_iso=action.args["start_iso"],
                        duration_minutes=int(action.args.get("duration_minutes", 30)),
                        title=action.args["title"],
                        description=action.args["description"],
                    )
                    current_lead = scoped_tools.current_lead
                elif action.type == ActionType.DELETE_MEETING:
                    payload = await scoped_tools.delete_meeting(event_id=action.args["event_id"])
                    current_lead = scoped_tools.current_lead
                elif action.type == ActionType.HANDOFF_HUMAN:
                    state["handoff_requested"] = True
                    payload = {"status": "requested"}
                else:
                    payload = {}
                results.append(ToolExecutionResult(action=action, success=True, payload=payload))
            except Exception as exc:  # noqa: BLE001
                results.append(ToolExecutionResult(action=action, success=False, error=str(exc)))
                state["errors"].append(str(exc))
        state["tool_results"] = results
        state["current_lead"] = current_lead
        created_meeting = next(
            (
                result.payload
                for result in results
                if result.success and result.action.type == ActionType.CREATE_MEETING
            ),
            None,
        )
        if created_meeting:
            lines = [f"Listo. Ya te dejé la demo agendada para {created_meeting.get('start_iso')}."]
            if created_meeting.get("meet_link"):
                lines.append(f"Link de Meet: {created_meeting['meet_link']}")
            elif created_meeting.get("html_link"):
                lines.append(f"Detalle del evento: {created_meeting['html_link']}")
            state["response_text"] = "\n".join(lines)
        return state

    async def _persist_state(self, state: AgentState) -> AgentState:
        inbound = ConversationMessage(
            message_id=state["event"].message_id,
            conversation_id=state["event"].conversation_id,
            phone_number=state["event"].phone_number,
            direction=Direction.INBOUND,
            text=state["event"].text,
            created_at=state["event"].timestamp,
            metadata={"provider": state["event"].provider},
        )
        await self.memory.append_message(inbound)
        if state["current_lead"] is not None:
            await self.memory.remember_contact(state["current_lead"])
        await self.memory.save_run(
            run_id=state["run_id"],
            event=state["event"],
            intent=state["planning"].intent,
            response_text=state["response_text"],
            tool_results=state["tool_results"],
            knowledge_lookups=state["planning"].knowledge_lookups,
        )
        state["persisted"] = True
        return state

    def _route_reply(self, state: AgentState) -> str:
        return "send_reply" if state["send_reply"] and state["response_text"] else "end"

    async def _send_reply(self, state: AgentState) -> AgentState:
        outbound = OutboundMessage(
            conversation_id=state["event"].conversation_id,
            phone_number=state["event"].phone_number,
            text=state["response_text"],
        )
        await self.channel.send_text(outbound)
        await self.memory.append_message(
            ConversationMessage(
                message_id=f"{state['run_id']}-out",
                conversation_id=outbound.conversation_id,
                phone_number=outbound.phone_number,
                direction=Direction.OUTBOUND,
                text=outbound.text,
                metadata={"run_id": state["run_id"]},
            )
        )
        return state
