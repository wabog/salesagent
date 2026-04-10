from __future__ import annotations

from uuid import uuid4

from langgraph.graph import END, StateGraph

from sales_agent.domain.models import (
    ActionType,
    AgentRunResult,
    CRMContact,
    ConversationMessage,
    Direction,
    OutboundMessage,
    ToolExecutionResult,
)
from sales_agent.domain.state import AgentState
from sales_agent.services.planner import AgentPlanner
from sales_agent.services.policy import ToolExecutionPolicy


class SalesAgentWorkflow:
    def __init__(self, crm_adapter, memory_store, channel_adapter, planner: AgentPlanner, policy: ToolExecutionPolicy):
        self.crm = crm_adapter
        self.memory = memory_store
        self.channel = channel_adapter
        self.planner = planner
        self.policy = policy
        self.graph = self._build_graph().compile()

    async def run(self, event) -> AgentRunResult:
        initial_state: AgentState = {
            "event": event,
            "run_id": uuid4().hex,
            "duplicate": False,
            "contact": None,
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
            contact=final_state["contact"],
        )

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("dedupe_and_lock", self._dedupe_and_lock)
        graph.add_node("load_contact_and_context", self._load_contact_and_context)
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
                "load_contact_and_context": "load_contact_and_context",
            },
        )
        graph.add_edge("load_contact_and_context", "classify_and_plan")
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
        return "duplicate_end" if state["duplicate"] else "load_contact_and_context"

    async def _load_contact_and_context(self, state: AgentState) -> AgentState:
        contact = await self.crm.find_contact_by_phone(state["event"].phone_number)
        if contact is None:
            contact = await self.memory.get_contact_shadow(state["event"].phone_number)
        recent = await self.memory.get_recent_messages(state["event"].conversation_id)
        memories = await self.memory.search_memories(state["event"].conversation_id, state["event"].text)
        state["contact"] = contact
        state["recent_messages"] = [message.model_dump(mode="json") for message in recent]
        state["semantic_memories"] = memories
        return state

    async def _classify_and_plan(self, state: AgentState) -> AgentState:
        planning = await self.planner.plan(
            text=state["event"].text,
            contact=state["contact"],
            recent_messages=[message["text"] for message in state["recent_messages"]],
            semantic_memories=state["semantic_memories"],
        )
        state["planning"] = planning
        state["response_text"] = planning.response_text
        state["send_reply"] = planning.should_reply
        return state

    async def _execute_tools(self, state: AgentState) -> AgentState:
        planning = state["planning"]
        contact = state["contact"]
        results: list[ToolExecutionResult] = []
        for action in planning.actions:
            try:
                self.policy.validate(action)
                if action.type == ActionType.CREATE_CONTACT:
                    contact = await self.crm.create_contact(
                        phone_number=state["event"].phone_number,
                        full_name=state["event"].contact_name,
                    )
                    payload = contact.model_dump(mode="json")
                elif action.type == ActionType.UPDATE_STAGE:
                    if contact is None:
                        contact = await self.crm.create_contact(state["event"].phone_number, state["event"].contact_name)
                    contact = await self.crm.change_stage(contact.external_id, action.args["stage"])
                    payload = contact.model_dump(mode="json")
                elif action.type == ActionType.APPEND_NOTE:
                    if contact is None:
                        contact = await self.crm.create_contact(state["event"].phone_number, state["event"].contact_name)
                    contact = await self.crm.append_note(contact.external_id, action.args["note"])
                    payload = contact.model_dump(mode="json")
                elif action.type == ActionType.CREATE_FOLLOWUP:
                    if contact is None:
                        contact = await self.crm.create_contact(state["event"].phone_number, state["event"].contact_name)
                    payload = await self.crm.create_followup(contact.external_id, action.args["summary"])
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
        state["contact"] = contact
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
        if state["contact"] is not None:
            await self.memory.remember_contact(state["contact"])
        await self.memory.save_run(
            run_id=state["run_id"],
            event=state["event"],
            intent=state["planning"].intent,
            response_text=state["response_text"],
            tool_results=state["tool_results"],
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
