from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sales_agent.adapters.channel import ConsoleChannelAdapter, KapsoWhatsAppAdapter
from sales_agent.adapters.crm_memory import InMemoryCRMAdapter
from sales_agent.adapters.crm_notion import NotionCRMAdapter
from sales_agent.adapters.memory_sql import SqlAlchemyMemoryStore
from sales_agent.core.config import Settings
from sales_agent.core.db import build_engine, build_session_factory, init_db
from sales_agent.domain.models import (
    ActionType,
    AgentRunResult,
    CRMContact,
    ConversationMessage,
    Direction,
    InboundMessage,
    InboundProcessingResult,
    OutboundMessage,
    PlanningResult,
    PromptMode,
    ToolExecutionResult,
)
from sales_agent.graph.workflow import SalesAgentWorkflow
from sales_agent.services.inbound_processor import DebouncedInboundProcessor
from sales_agent.services.lead_scope import LeadScopedCRMTools
from sales_agent.services.planner import AgentPlanner
from sales_agent.services.policy import ToolExecutionPolicy
from sales_agent.services.prompt_store import PromptConfigStore


@dataclass
class PreparedBatchRun:
    run_id: str
    events: list[InboundMessage]
    current_lead: CRMContact | None
    lead_created: bool
    recent_messages: list[ConversationMessage]
    response_text: str
    planning: PlanningResult


class SalesAgentApplication:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = build_engine(settings.database_url)
        self.session_factory: async_sessionmaker[AsyncSession] = build_session_factory(self.engine)
        self.memory_store = SqlAlchemyMemoryStore(self.session_factory)
        self.prompt_store = PromptConfigStore(self.session_factory)
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self.crm_adapter = self._build_crm_adapter()
        self.channel_adapter = self._build_channel_adapter()
        self.workflow = SalesAgentWorkflow(
            crm_adapter=self.crm_adapter,
            memory_store=self.memory_store,
            channel_adapter=self.channel_adapter,
            planner=AgentPlanner(settings, prompt_store=self.prompt_store),
            policy=ToolExecutionPolicy(),
        )
        self.inbound_processor = DebouncedInboundProcessor(
            debounce_seconds=settings.message_batch_window_seconds,
            prepare_batch=self.prepare_batched_run,
            commit_batch=self.commit_batched_run,
        )

    async def startup(self) -> None:
        await init_db(self.engine)

    async def shutdown(self) -> None:
        await self.inbound_processor.shutdown()
        await self.engine.dispose()

    async def handle_event(self, event: InboundMessage, *, wait_for_response: bool) -> InboundProcessingResult:
        conversation_lock = self._conversation_locks.setdefault(event.conversation_id, asyncio.Lock())
        async with conversation_lock:
            if await self.memory_store.has_message(event.message_id):
                return InboundProcessingResult(duplicate=True, render_reply=False)

            await self.memory_store.append_message(
                ConversationMessage(
                    message_id=event.message_id,
                    conversation_id=event.conversation_id,
                    phone_number=event.phone_number,
                    direction=Direction.INBOUND,
                    text=event.text,
                    created_at=event.timestamp,
                    metadata={"provider": event.provider},
                )
            )

        return await self.inbound_processor.submit(event, wait_for_result=wait_for_response)

    async def prepare_batched_run(self, events: list[InboundMessage]) -> PreparedBatchRun:
        anchor_event = events[-1]
        batch_message_ids = {event.message_id for event in events}
        combined_text = "\n".join(part for part in (event.text.strip() for event in events) if part).strip() or anchor_event.text

        contact = await self.crm_adapter.find_contact_by_phone(anchor_event.phone_number)
        lead_created = False
        if contact is None:
            shadow = await self.memory_store.get_contact_shadow(anchor_event.phone_number)
            contact = await self.crm_adapter.create_contact(
                phone_number=anchor_event.phone_number,
                full_name=anchor_event.contact_name or (shadow.full_name if shadow else None),
            )
            lead_created = True

        recent_messages = await self.memory_store.get_recent_messages(anchor_event.conversation_id, limit=12)
        recent_messages = [message for message in recent_messages if message.message_id not in batch_message_ids]
        semantic_memories = await self.memory_store.search_memories(anchor_event.conversation_id, combined_text)
        planning = await self.workflow.planner.plan(
            text=combined_text,
            contact=contact,
            recent_messages=[message.text for message in recent_messages],
            semantic_memories=semantic_memories,
            prompt_mode=anchor_event.prompt_mode,
        )
        response_text = planning.response_text.strip()

        return PreparedBatchRun(
            run_id=uuid4().hex,
            events=events,
            current_lead=contact,
            lead_created=lead_created,
            recent_messages=recent_messages,
            response_text=response_text,
            planning=planning,
        )

    async def commit_batched_run(self, prepared: PreparedBatchRun) -> InboundProcessingResult:
        anchor_event = prepared.events[-1]
        current_lead = prepared.current_lead
        planning = prepared.planning
        recent_texts = [message.text for message in prepared.recent_messages]
        combined_text = "\n".join(part for part in (event.text.strip() for event in prepared.events) if part).strip() or anchor_event.text
        tool_results: list[ToolExecutionResult] = []
        scoped_tools = LeadScopedCRMTools(self.crm_adapter, current_lead) if current_lead is not None else None

        for action in planning.actions:
            try:
                if action.type == ActionType.UPDATE_STAGE and not action.args.get("stage"):
                    inferred_stage = self.workflow.planner._infer_stage_from_text(  # noqa: SLF001
                        combined_text,
                        current_lead.stage if current_lead and current_lead.stage else "Prospecto",
                        recent_texts,
                    )
                    if inferred_stage:
                        action = action.model_copy(update={"args": {**action.args, "stage": inferred_stage}})
                self.workflow.policy.validate(action)
                if action.type == ActionType.UPDATE_STAGE and scoped_tools is not None:
                    current_lead = await scoped_tools.update_stage(action.args["stage"])
                    payload = current_lead.model_dump(mode="json")
                elif action.type == ActionType.UPDATE_CONTACT_FIELDS and scoped_tools is not None:
                    current_lead = await scoped_tools.update_fields(action.args["fields"])
                    payload = current_lead.model_dump(mode="json")
                elif action.type == ActionType.APPEND_NOTE and scoped_tools is not None:
                    current_lead = await scoped_tools.add_note(action.args["note"])
                    payload = current_lead.model_dump(mode="json")
                elif action.type == ActionType.CREATE_FOLLOWUP and scoped_tools is not None:
                    payload = await scoped_tools.create_followup(action.args["summary"])
                elif action.type == ActionType.HANDOFF_HUMAN:
                    payload = {"status": "requested"}
                else:
                    payload = {}
                tool_results.append(ToolExecutionResult(action=action, success=True, payload=payload))
            except Exception as exc:  # noqa: BLE001
                tool_results.append(ToolExecutionResult(action=action, success=False, error=str(exc)))

        if current_lead is not None:
            await self.memory_store.remember_contact(current_lead)

        await self.memory_store.save_run(
            run_id=prepared.run_id,
            event=anchor_event,
            intent=planning.intent,
            response_text=prepared.response_text,
            tool_results=tool_results,
        )

        if planning.should_reply and prepared.response_text:
            outbound = OutboundMessage(
                conversation_id=anchor_event.conversation_id,
                phone_number=anchor_event.phone_number,
                text=prepared.response_text,
            )
            await self.channel_adapter.send_text(outbound)
            await self.memory_store.append_message(
                ConversationMessage(
                    message_id=f"{prepared.run_id}-out",
                    conversation_id=outbound.conversation_id,
                    phone_number=outbound.phone_number,
                    direction=Direction.OUTBOUND,
                    text=outbound.text,
                    metadata={"run_id": prepared.run_id},
                )
            )

        result = AgentRunResult(
            run_id=prepared.run_id,
            duplicate=False,
            intent=planning.intent,
            response_text=prepared.response_text,
            tool_results=tool_results,
            contact=current_lead,
        )
        return InboundProcessingResult(
            queued=False,
            duplicate=False,
            render_reply=True,
            aggregated_messages=len(prepared.events),
            run_id=result.run_id,
            intent=result.intent,
            response_text=result.response_text,
            tool_results=result.tool_results,
            contact=result.contact,
        )

    async def get_prompt_editor_state(self, mode: PromptMode = PromptMode.PUBLISHED) -> dict:
        state = await self.prompt_store.get_editor_state(mode=mode)
        state["core_prompt"] = self.workflow.planner.get_prompt_scaffold()
        return state

    async def save_prompt_draft(self, business_prompt: str) -> dict:
        state = await self.prompt_store.save_draft(business_prompt)
        state["core_prompt"] = self.workflow.planner.get_prompt_scaffold()
        return state

    async def publish_prompt_draft(self) -> dict:
        state = await self.prompt_store.publish_draft()
        state["core_prompt"] = self.workflow.planner.get_prompt_scaffold()
        return state

    async def reset_prompt_draft(self) -> dict:
        state = await self.prompt_store.reset_draft()
        state["core_prompt"] = self.workflow.planner.get_prompt_scaffold()
        return state

    def _build_crm_adapter(self):
        if self.settings.crm_backend == "notion":
            return NotionCRMAdapter(self.settings)
        return InMemoryCRMAdapter()

    def _build_channel_adapter(self):
        if self.settings.kapso_api_token and self.settings.kapso_phone_number_id:
            return KapsoWhatsAppAdapter(self.settings)
        return ConsoleChannelAdapter()
