from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sales_agent.adapters.channel import ConsoleChannelAdapter, KapsoWhatsAppAdapter
from sales_agent.adapters.crm_memory import InMemoryCRMAdapter
from sales_agent.adapters.crm_notion import NotionCRMAdapter
from sales_agent.adapters.google_calendar import GoogleCalendarAdapter
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
    ToolExecutionResult,
)
from sales_agent.graph.workflow import SalesAgentWorkflow
from sales_agent.services.inbound_processor import DebouncedInboundProcessor
from sales_agent.services.calendar_sync import enrich_contact_with_calendar, merge_contact_with_shadow
from sales_agent.services.lead_scope import LeadScopedCRMTools
from sales_agent.services.media_preprocessor import InboundMediaPreprocessor
from sales_agent.services.name_validation import (
    ContactNameValidator,
    apply_name_validation_metadata,
    build_trusted_name_assessment,
    contact_has_reliable_name,
    get_effective_contact_name,
)
from sales_agent.services.planner import AgentPlanner
from sales_agent.services.policy import ToolExecutionPolicy
from sales_agent.services.prompt_library import PromptLibrary


logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


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


def _is_playground_event(event: InboundMessage) -> bool:
    return event.provider == "local-playground"


@dataclass
class PreparedBatchRun:
    run_id: str
    events: list[InboundMessage]
    current_lead: CRMContact | None
    lead_created: bool
    recent_messages: list[ConversationMessage]
    response_text: str
    planning: PlanningResult
    typing_stop_event: asyncio.Event | None = None
    typing_task: asyncio.Task | None = None


class SalesAgentApplication:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = build_engine(settings.database_url)
        self.session_factory: async_sessionmaker[AsyncSession] = build_session_factory(self.engine)
        self.memory_store = SqlAlchemyMemoryStore(self.session_factory)
        self.prompt_library = PromptLibrary()
        self.name_validator = ContactNameValidator(settings)
        self.media_preprocessor = InboundMediaPreprocessor(settings)
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self.crm_adapter = self._build_crm_adapter()
        self.calendar_adapter = self._build_calendar_adapter()
        self.channel_adapter = self._build_channel_adapter()
        self.workflow = SalesAgentWorkflow(
            crm_adapter=self.crm_adapter,
            memory_store=self.memory_store,
            channel_adapter=self.channel_adapter,
            calendar_adapter=self.calendar_adapter,
            planner=AgentPlanner(settings, prompt_library=self.prompt_library),
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
        typing_stop_event, typing_task = self._start_typing_loop(anchor_event)

        try:
            combined_text, planning_override = await self._build_combined_text(events)

            provider_name_assessment = await self.name_validator.assess_provider_name(anchor_event.contact_name)
            contact = await self.crm_adapter.find_contact_by_phone(anchor_event.phone_number)
            shadow = await self.memory_store.get_contact_shadow(anchor_event.phone_number)
            lead_created = False
            if contact is None:
                contact = await self.crm_adapter.create_contact(
                    phone_number=anchor_event.phone_number,
                    full_name=get_effective_contact_name(shadow),
                )
                lead_created = True
            if shadow is not None:
                contact = merge_contact_with_shadow(contact, shadow)
            if provider_name_assessment.status != "rejected":
                contact = apply_name_validation_metadata(contact, provider_name_assessment)
            contact = await enrich_contact_with_calendar(
                contact,
                self.calendar_adapter,
                self.settings.google_calendar_self_schedule_url,
            )

            conversation_recent = await self.memory_store.get_recent_messages(
                anchor_event.conversation_id,
                limit=self.settings.recent_message_context_limit,
            )
            conversation_recent = [
                message
                for message in conversation_recent
                if message.message_id not in batch_message_ids and message.phone_number == anchor_event.phone_number
            ]
            lead_recent: list[ConversationMessage] = []
            semantic_memory_groups: list[list[str]] = [
                await self.memory_store.search_memories(
                    anchor_event.conversation_id,
                    combined_text,
                    limit=self.settings.semantic_memory_limit,
                )
            ]
            if not _is_playground_event(anchor_event):
                lead_recent = await self.memory_store.get_recent_messages_by_phone(
                    anchor_event.phone_number,
                    limit=max(self.settings.recent_message_context_limit * 2, self.settings.recent_message_context_limit),
                )
                lead_recent = [
                    message
                    for message in lead_recent
                    if message.message_id not in batch_message_ids
                    and _ensure_utc_datetime(message.created_at)
                    >= _ensure_utc_datetime(anchor_event.timestamp) - timedelta(days=self.settings.phone_context_max_age_days)
                ]
                if lead_recent:
                    semantic_memory_groups.append(
                        await self.memory_store.search_memories_by_phone(
                            anchor_event.phone_number,
                            combined_text,
                            limit=self.settings.semantic_memory_limit,
                        )
                    )
            recent_messages = _merge_unique_messages(
                conversation_recent,
                lead_recent,
                limit=self.settings.recent_message_context_limit,
            )
            semantic_memories = _merge_unique_texts(*semantic_memory_groups, limit=self.settings.semantic_memory_limit)
            planning = planning_override or await self.workflow.planner.plan(
                text=combined_text,
                contact=contact,
                recent_messages=[message.text for message in recent_messages],
                semantic_memories=semantic_memories,
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
                typing_stop_event=typing_stop_event,
                typing_task=typing_task,
            )
        except Exception:
            await self._stop_typing_loop(typing_stop_event, typing_task)
            raise

    async def commit_batched_run(self, prepared: PreparedBatchRun) -> InboundProcessingResult:
        try:
            anchor_event = prepared.events[-1]
            current_lead = prepared.current_lead
            planning = prepared.planning
            recent_texts = [message.text for message in prepared.recent_messages]
            combined_text = "\n".join(part for part in (event.text.strip() for event in prepared.events) if part).strip() or anchor_event.text
            tool_results: list[ToolExecutionResult] = []
            scoped_tools = (
                LeadScopedCRMTools(self.crm_adapter, current_lead, calendar_adapter=self.calendar_adapter)
                if current_lead is not None
                else None
            )

            for action in planning.actions:
                try:
                    if action.type == ActionType.UPDATE_STAGE and not action.args.get("stage"):
                        continue
                    self.workflow.policy.validate(action)
                    if action.type == ActionType.UPDATE_STAGE and scoped_tools is not None:
                        current_lead = await scoped_tools.update_stage(action.args["stage"])
                        payload = current_lead.model_dump(mode="json")
                    elif action.type == ActionType.UPDATE_CONTACT_FIELDS and scoped_tools is not None:
                        if "full_name" in action.args.get("fields", {}) and current_lead is not None:
                            current_lead = apply_name_validation_metadata(
                                current_lead,
                                build_trusted_name_assessment(
                                    action.args["fields"]["full_name"],
                                    source="user_message",
                                ),
                            )
                            scoped_tools.current_lead = current_lead
                        current_lead = await scoped_tools.update_fields(action.args["fields"])
                        payload = current_lead.model_dump(mode="json")
                    elif action.type == ActionType.APPEND_NOTE and scoped_tools is not None:
                        current_lead = await scoped_tools.add_note(action.args["note"])
                        payload = current_lead.model_dump(mode="json")
                    elif action.type == ActionType.CREATE_FOLLOWUP and scoped_tools is not None:
                        payload = await scoped_tools.create_followup(
                            action.args["summary"],
                            due_date=action.args.get("due_date"),
                        )
                        current_lead = scoped_tools.current_lead
                    elif action.type == ActionType.COMPLETE_FOLLOWUP and scoped_tools is not None:
                        payload = await scoped_tools.complete_followup(
                            outcome=action.args.get("outcome"),
                        )
                        current_lead = scoped_tools.current_lead
                    elif action.type == ActionType.CREATE_MEETING and scoped_tools is not None:
                        payload = await scoped_tools.create_meeting(
                            start_iso=action.args["start_iso"],
                            duration_minutes=int(action.args.get("duration_minutes", self.settings.google_calendar_default_meeting_minutes)),
                            title=action.args["title"],
                            description=action.args["description"],
                        )
                        current_lead = scoped_tools.current_lead
                        current_lead = await self._sync_crm_after_meeting_created(scoped_tools, current_lead, payload)
                    elif action.type == ActionType.HANDOFF_HUMAN:
                        payload = {"status": "requested"}
                    else:
                        payload = {}
                    tool_results.append(ToolExecutionResult(action=action, success=True, payload=payload))
                except Exception as exc:  # noqa: BLE001
                    tool_results.append(ToolExecutionResult(action=action, success=False, error=str(exc)))

            prepared.response_text = self._finalize_response_text(prepared.response_text, tool_results, current_lead)
            if planning.knowledge_lookups:
                logger.info(
                    "run_knowledge_lookup run_id=%s sections=%s",
                    prepared.run_id,
                    [lookup.section for lookup in planning.knowledge_lookups],
                )

            if current_lead is not None:
                current_lead = self._refresh_live_context(
                    current_lead,
                    combined_text,
                    recent_texts,
                    tool_results,
                )
                await self.memory_store.remember_contact(current_lead)

            await self.memory_store.save_run(
                run_id=prepared.run_id,
                event=anchor_event,
                intent=planning.intent,
                response_text=prepared.response_text,
                tool_results=tool_results,
                knowledge_lookups=planning.knowledge_lookups,
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
        finally:
            await self._stop_typing_loop(prepared.typing_stop_event, prepared.typing_task)

    async def _build_combined_text(self, events: list[InboundMessage]) -> tuple[str, PlanningResult | None]:
        parts: list[str] = []
        document_requested = False
        for event in events:
            result = await self.media_preprocessor.preprocess_event(event)
            if result.bypass_intent == "document_unsupported":
                document_requested = True
                continue
            text = result.text.strip()
            if text:
                parts.append(text)
        combined_text = "\n".join(parts).strip()
        if combined_text:
            return combined_text, None
        if document_requested:
            return "", PlanningResult(
                intent="document_unsupported",
                confidence=1.0,
                response_text=(
                    "Recibí el archivo, pero por ahora no puedo procesar documentos directamente por este canal. "
                    "Si quieres, cuéntame qué necesitas revisar y te ayudo por aquí."
                ),
                actions=[],
                should_reply=True,
            )
        return "", PlanningResult(
            intent="non_text_ignored",
            confidence=1.0,
            response_text="",
            actions=[],
            should_reply=False,
        )

    def _start_typing_loop(self, event: InboundMessage) -> tuple[asyncio.Event | None, asyncio.Task | None]:
        if _is_playground_event(event) or event.provider != "kapso":
            return None, None
        stop_event = asyncio.Event()
        task = asyncio.create_task(self._typing_loop(event, stop_event))
        return stop_event, task

    async def _typing_loop(self, event: InboundMessage, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self.channel_adapter.send_typing_indicator(event)
            except Exception:  # noqa: BLE001
                return
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                continue

    async def _stop_typing_loop(
        self,
        stop_event: asyncio.Event | None,
        task: asyncio.Task | None,
    ) -> None:
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            try:
                await task
            except Exception:  # noqa: BLE001
                pass

    async def get_playground_context(self) -> dict:
        state = self.prompt_library.get_playground_context()
        state["planner_scaffold"] = self.workflow.planner.get_prompt_scaffold()
        return state

    def _build_crm_adapter(self):
        if self.settings.crm_backend == "notion":
            return NotionCRMAdapter(self.settings)
        return InMemoryCRMAdapter()

    def _build_channel_adapter(self):
        if self.settings.kapso_api_token and self.settings.kapso_phone_number_id:
            return KapsoWhatsAppAdapter(self.settings)
        return ConsoleChannelAdapter()

    def _build_calendar_adapter(self):
        if (
            self.settings.google_client_id
            and self.settings.google_client_secret
            and self.settings.google_refresh_token
        ):
            return GoogleCalendarAdapter(self.settings)
        return None

    def _finalize_response_text(
        self,
        response_text: str,
        tool_results: list[ToolExecutionResult],
        contact: CRMContact | None,
    ) -> str:
        final_text = response_text.strip().replace("https://wabog.com/signup", "https://wabog.com")
        failed_meeting = next(
            (
                result
                for result in tool_results
                if not result.success and result.action.type == ActionType.CREATE_MEETING
            ),
            None,
        )
        created_meeting = next(
            (
                result.payload
                for result in tool_results
                if result.success and result.action.type == ActionType.CREATE_MEETING
            ),
            None,
        )
        if created_meeting:
            start_iso = created_meeting.get("start_iso")
            confirmation_lines = [
                f"Listo. Ya te dejé la demo agendada para {start_iso}.",
            ]
            if created_meeting.get("meet_link"):
                confirmation_lines.append(f"Link de Meet: {created_meeting['meet_link']}")
            elif created_meeting.get("html_link"):
                confirmation_lines.append(f"Detalle del evento: {created_meeting['html_link']}")
            return "\n".join(confirmation_lines)

        if failed_meeting is not None:
            missing_fields: list[str] = []
            if contact is None or not contact_has_reliable_name(contact):
                missing_fields.append("nombre completo")
            if contact is None or not (contact.email or "").strip():
                missing_fields.append("correo")
            if missing_fields:
                joined = " y ".join(missing_fields)
                return (
                    "Todavía no la dejo agendada. "
                    f"Antes necesito tu {joined} para enviarte bien la invitación."
                )
            if self.settings.google_calendar_self_schedule_url:
                return (
                    "Todavía no pude dejar la demo creada en calendario. "
                    f"Si quieres, compárteme de nuevo el horario o usa este link: {self.settings.google_calendar_self_schedule_url}"
                )
            return "Todavía no pude dejar la demo creada en calendario. Si quieres, confirmamos de nuevo el horario."

        upcoming_event = (((contact.metadata if contact else {}) or {}).get("calendar") or {}).get("upcoming_event")
        if upcoming_event and not final_text:
            return f"Veo una demo futura agendada para {upcoming_event.get('start_iso')}."
        return final_text

    async def _sync_crm_after_meeting_created(
        self,
        scoped_tools: LeadScopedCRMTools,
        current_lead: CRMContact | None,
        meeting_payload: dict,
    ) -> CRMContact | None:
        if current_lead is None:
            return None

        start_iso = str(meeting_payload.get("start_iso", "")).strip()
        due_date = datetime.fromisoformat(start_iso).date().isoformat() if start_iso else None

        if current_lead.stage != "Demo agendada":
            current_lead = await scoped_tools.update_stage("Demo agendada")

        note_parts = [f"Demo agendada para {start_iso}." if start_iso else "Demo agendada."]
        if meeting_payload.get("meet_link"):
            note_parts.append(f"Meet: {meeting_payload['meet_link']}.")
        elif meeting_payload.get("html_link"):
            note_parts.append(f"Evento: {meeting_payload['html_link']}.")
        current_lead = await scoped_tools.add_note(" ".join(note_parts))

        summary = (
            f"Asistir a la demo agendada para {start_iso} y continuar seguimiento comercial."
            if start_iso
            else "Asistir a la demo agendada y continuar seguimiento comercial."
        )
        await scoped_tools.create_followup(summary, due_date=due_date)
        return scoped_tools.current_lead

    def _refresh_live_context(
        self,
        contact: CRMContact,
        combined_text: str,
        recent_texts: list[str],
        tool_results: list[ToolExecutionResult],
    ) -> CRMContact:
        metadata = dict(contact.metadata or {})
        live_context = dict((metadata.get("live_context") or {}))
        has_successful_meeting = any(
            result.success and result.action.type == ActionType.CREATE_MEETING
            for result in tool_results
        )
        has_upcoming_event = bool((((contact.metadata or {}).get("calendar") or {}).get("upcoming_event")))
        if has_successful_meeting or has_upcoming_event:
            live_context.pop("pending_booking_start_iso", None)
        else:
            pending_start = self.workflow.planner._extract_requested_meeting_start(  # noqa: SLF001
                combined_text,
                recent_texts,
            )
            if pending_start is not None:
                live_context["pending_booking_start_iso"] = pending_start.isoformat()
        if live_context:
            metadata["live_context"] = live_context
        else:
            metadata.pop("live_context", None)
        return contact.model_copy(update={"metadata": metadata})
