from datetime import date, datetime, timezone

import pytest

from sales_agent.core.config import Settings
from sales_agent.domain.models import (
    ActionType,
    CRMContact,
    InboundMessage,
    PlanningResult,
    ProposedAction,
    ToolExecutionResult,
)
from sales_agent.services.name_validation import NameCandidateAssessment, NameConfirmationDecision, apply_name_validation_metadata
from sales_agent.services.application import (
    PreparedBatchRun,
    SalesAgentApplication,
    _should_keep_text_for_contact_context,
)


class _StubCalendarAdapter:
    def __init__(self) -> None:
        self.deleted_event_ids: list[str] = []
        self.created_event_ids: list[str] = []

    async def find_upcoming_meeting(self, contact):  # noqa: ANN001
        return None

    async def create_meeting(self, contact, *, start_iso, duration_minutes, title, description):  # noqa: ANN001
        event_id = f"evt-{len(self.created_event_ids) + 1}"
        self.created_event_ids.append(event_id)
        return {
            "id": event_id,
            "summary": title,
            "description": description,
            "start_iso": start_iso,
            "end_iso": start_iso,
            "html_link": f"https://calendar.google.com/event?eid={event_id}",
            "meet_link": "https://meet.google.com/test-meet",
            "status": "confirmed",
            "attendees": [contact.email] if contact.email else [],
            "source": "agent_created",
        }

    async def delete_meeting(self, event_id):  # noqa: ANN001
        self.deleted_event_ids.append(event_id)
        return {"id": event_id, "status": "cancelled", "source": "agent_deleted"}


class _CapturingChannelAdapter:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.typing_events: list[str] = []

    async def send_text(self, outbound):  # noqa: ANN001
        self.messages.append(outbound.text)

    async def send_typing_indicator(self, event):  # noqa: ANN001
        self.typing_events.append(event.message_id)
        return {"ok": True}


def build_application() -> SalesAgentApplication:
    return SalesAgentApplication(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            GOOGLE_CALENDAR_SELF_SCHEDULE_URL="https://calendar.app.google/9KpgMYTW4nA17LV27",
        )
    )


def test_finalize_response_text_degrades_failed_meeting_when_contact_data_missing():
    app = build_application()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name=None,
        email=None,
    )
    tool_results = [
        ToolExecutionResult(
            action=ProposedAction(
                type=ActionType.CREATE_MEETING,
                reason="Intentar crear demo.",
                args={},
            ),
            success=False,
            error="Meeting creation requires start_iso.",
        )
    ]

    final_text = app._finalize_response_text(  # noqa: SLF001
        "Perfecto, agendamos la demo y te envío la invitación.",
        tool_results,
        contact,
        should_reply=True,
    )

    assert "todavía no la dejo agendada" in final_text.lower()
    assert "nombre completo" in final_text.lower()
    assert "correo" in final_text.lower()


def test_finalize_response_text_normalizes_unpublished_wabog_signup_route():
    app = build_application()

    final_text = app._finalize_response_text(  # noqa: SLF001
        "Te dejo el link para probarlo: https://wabog.com/signup",
        [],
        None,
        should_reply=True,
    )

    assert "https://wabog.com/signup" not in final_text
    assert "https://wabog.com" in final_text


def test_finalize_response_text_keeps_non_reply_runs_empty_even_with_upcoming_event():
    app = build_application()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Fabian Cuero",
        email="fabian@example.com",
        metadata={
            "calendar": {
                "upcoming_event": {"id": "evt-1", "start_iso": "2026-05-07T15:00:00-05:00"},
            }
        },
    )

    final_text = app._finalize_response_text(  # noqa: SLF001
        "",
        [],
        contact,
        should_reply=False,
    )

    assert final_text == ""


def test_finalize_response_text_ignores_past_cached_upcoming_event():
    app = build_application()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Fabian Cuero",
        email="fabian@example.com",
        metadata={
            "calendar": {
                "upcoming_event": {
                    "id": "evt-old",
                    "start_iso": "2026-05-07T15:00:00-05:00",
                    "end_iso": "2026-05-07T15:30:00-05:00",
                },
            }
        },
    )

    final_text = app._finalize_response_text(  # noqa: SLF001
        "",
        [],
        contact,
        should_reply=True,
    )

    assert final_text == ""


def test_contact_context_filters_existing_demo_claim_when_no_upcoming_event():
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        metadata={"calendar": {"upcoming_event": None}},
    )

    assert not _should_keep_text_for_contact_context(
        "¿Quieres que te recuerde el enlace de la demo agendada para el 7 de mayo?",
        contact,
    )
    assert _should_keep_text_for_contact_context("Hola, ¿en qué puedo ayudarte con Wabog?", contact)


@pytest.mark.asyncio
async def test_commit_batched_run_syncs_crm_after_meeting_creation():
    app = build_application()
    app.calendar_adapter = _StubCalendarAdapter()
    await app.startup()
    try:
        created = await app.crm_adapter.create_contact("573001111222", "Paula Diaz")
        current_lead = await app.crm_adapter.update_contact_fields(
            created.external_id,
            {"email": "paula@example.com"},
        )
        event = InboundMessage(
            message_id="meeting-app-1",
            conversation_id="conv-app-meeting",
            phone_number="573001111222",
            text="mañana a las 10 am me sirve",
            timestamp=datetime.now(timezone.utc),
            raw_payload={},
            provider="local-playground",
        )
        prepared = PreparedBatchRun(
            run_id="run-meeting-app-1",
            events=[event],
            current_lead=current_lead,
            lead_created=False,
            recent_messages=[],
            response_text="Perfecto. Procedo a agendarla.",
            planning=PlanningResult(
                intent="schedule_demo",
                confidence=0.9,
                response_text="Perfecto. Procedo a agendarla.",
                actions=[
                    ProposedAction(
                        type=ActionType.CREATE_MEETING,
                        reason="Lead confirmó horario.",
                        args={
                            "start_iso": "2026-05-02T10:00:00-05:00",
                            "duration_minutes": 30,
                            "title": "Demo Wabog - Paula Diaz",
                            "description": "Demo comercial creada por el agente de ventas de Wabog.",
                        },
                    )
                ],
            ),
        )

        result = await app.commit_batched_run(prepared)
    finally:
        await app.shutdown()

    assert result.contact is not None
    assert result.contact.stage == "Demo agendada"
    assert result.contact.followup_summary is not None
    assert result.contact.followup_due_date == date(2026, 5, 2)
    assert any("Demo agendada para 2026-05-02T10:00:00-05:00." in note for note in result.contact.notes)
    assert any(item.action.type == ActionType.CREATE_MEETING and item.success for item in result.tool_results)


@pytest.mark.asyncio
async def test_commit_batched_run_recreates_meeting_and_deletes_previous_event():
    app = build_application()
    stub_calendar = _StubCalendarAdapter()
    app.calendar_adapter = stub_calendar
    await app.startup()
    try:
        created = await app.crm_adapter.create_contact("573001111223", "Paula Diaz")
        current_lead = await app.crm_adapter.update_contact_fields(
            created.external_id,
            {"email": "paula@example.com"},
        )
        current_lead = current_lead.model_copy(
            update={
                "stage": "Demo agendada",
                "metadata": {
                    "name_validation": {"status": "trusted", "source": "user_message"},
                    "calendar": {
                        "connected": True,
                        "upcoming_event": {
                            "id": "evt-old",
                            "start_iso": "2026-05-02T10:00:00-05:00",
                        },
                    },
                },
            }
        )
        event = InboundMessage(
            message_id="meeting-app-2",
            conversation_id="conv-app-meeting-2",
            phone_number="573001111223",
            text="mañana a las 11 am me sirve",
            timestamp=datetime.now(timezone.utc),
            raw_payload={},
            provider="local-playground",
        )
        prepared = PreparedBatchRun(
            run_id="run-meeting-app-2",
            events=[event],
            current_lead=current_lead,
            lead_created=False,
            recent_messages=[],
            response_text="Voy a mover la demo.",
            planning=PlanningResult(
                intent="reschedule_demo",
                confidence=0.92,
                response_text="Voy a mover la demo.",
                actions=[
                    ProposedAction(
                        type=ActionType.CREATE_MEETING,
                        reason="Mover al nuevo horario.",
                        args={
                            "start_iso": "2026-05-02T11:00:00-05:00",
                            "duration_minutes": 30,
                            "title": "Demo Wabog - Paula Diaz",
                            "description": "Demo comercial creada por el agente de ventas de Wabog.",
                        },
                    ),
                    ProposedAction(
                        type=ActionType.DELETE_MEETING,
                        reason="Eliminar el evento anterior.",
                        args={"event_id": "evt-old"},
                    ),
                ],
            ),
        )

        result = await app.commit_batched_run(prepared)
    finally:
        await app.shutdown()

    assert stub_calendar.deleted_event_ids == ["evt-old"]
    assert any(item.action.type == ActionType.CREATE_MEETING and item.success for item in result.tool_results)
    assert any(item.action.type == ActionType.DELETE_MEETING and item.success for item in result.tool_results)
    assert "2026-05-02T11:00:00-05:00" in result.response_text


@pytest.mark.asyncio
async def test_handle_event_e2e_confirms_candidate_name_by_context_and_books_demo():
    app = build_application()
    app.calendar_adapter = _StubCalendarAdapter()
    app.channel_adapter = _CapturingChannelAdapter()
    await app.startup()
    try:
        created = await app.crm_adapter.create_contact("573156832405", "Juan Perez")
        created = await app.crm_adapter.change_stage(created.external_id, "Primer contacto")
        await app.crm_adapter.create_followup(
            created.external_id,
            "Coordinar fecha y hora para demo comercial.",
            due_date="2026-05-05",
        )
        current = await app.crm_adapter.find_contact_by_phone("573156832405")
        assert current is not None
        shadow = apply_name_validation_metadata(
            current.model_copy(
                update={
                    "followup_summary": "Coordinar fecha y hora para demo comercial.",
                    "followup_due_date": date(2026, 5, 5),
                }
            ),
            NameCandidateAssessment(
                status="needs_confirmation",
                confidence=0.95,
                normalized_name="Fabian C Villegas",
                candidate_name="Fabian C Villegas",
                source="provider_llm",
            ),
        )
        await app.memory_store.remember_contact(shadow)

        async def fake_resolve(text, contact, recent_messages):  # noqa: ANN001
            if text == "Ese es mi nombre agendala.":
                return NameConfirmationDecision(
                    status="confirmed_candidate_name",
                    confidence=0.99,
                    resolved_name="Fabian C Villegas",
                )
            return None

        def fake_plan_with_rules(text, contact):  # noqa: ANN001
            if text == "mañana a las 10am mi correo es fabiancuerov@gmail.com":
                return PlanningResult(
                    intent="Confirmación de fecha y correo para demo",
                    confidence=0.88,
                    response_text="Perfecto. Antes de agendarla, ¿tu nombre es Fabian C Villegas? Si no, compárteme tu nombre completo.",
                    actions=[
                        ProposedAction(
                            type=ActionType.UPDATE_CONTACT_FIELDS,
                            reason="El usuario confirma correo para contacto y demo",
                            args={"fields": {"email": "fabiancuerov@gmail.com"}},
                        )
                    ],
                )
            if text == "Ese es mi nombre agendala.":
                return PlanningResult(
                    intent="Confirmar nombre y agendar demo",
                    confidence=0.89,
                    response_text="Perfecto. Antes de agendarla, ¿tu nombre es Fabian C Villegas? Si no, compárteme tu nombre completo.",
                    actions=[
                        ProposedAction(
                            type=ActionType.UPDATE_CONTACT_FIELDS,
                            reason="Confirmación de nombre y correo para agendar demo",
                            args={"fields": {}},
                        ),
                        ProposedAction(
                            type=ActionType.COMPLETE_FOLLOWUP,
                            reason="Demo confirmada para mañana a las 10am",
                            args={"outcome": "Ese es mi nombre agendala."},
                        ),
                    ],
                )
            raise AssertionError(f"Unexpected text in test: {text}")

        app.workflow.planner._name_confirmation_resolver.resolve = fake_resolve  # noqa: SLF001
        app.workflow.planner._plan_with_rules = fake_plan_with_rules  # noqa: SLF001

        first = await app.handle_event(
            InboundMessage(
                message_id="prod-loop-1",
                conversation_id="conv-prod-loop",
                phone_number="573156832405",
                text="mañana a las 10am mi correo es fabiancuerov@gmail.com",
                timestamp=datetime.now(timezone.utc),
                raw_payload={},
                provider="local-playground",
            ),
            wait_for_response=True,
        )
        second = await app.handle_event(
            InboundMessage(
                message_id="prod-loop-2",
                conversation_id="conv-prod-loop",
                phone_number="573156832405",
                text="Ese es mi nombre agendala.",
                timestamp=datetime.now(timezone.utc),
                raw_payload={},
                provider="local-playground",
            ),
            wait_for_response=True,
        )
    finally:
        await app.shutdown()

    assert "compárteme tu nombre completo" in first.response_text.lower()
    assert "ya te dejé la demo agendada" in second.response_text.lower()
    assert "compárteme tu nombre completo" not in second.response_text.lower()
    assert any(item.action.type == ActionType.CREATE_MEETING and item.success for item in second.tool_results)
    assert second.contact is not None
    assert second.contact.full_name == "Fabian C Villegas"
    assert second.contact.stage == "Demo agendada"


@pytest.mark.asyncio
async def test_handle_event_e2e_prod_conversation_books_and_reprograms_without_name_pollution():
    app = build_application()
    stub_calendar = _StubCalendarAdapter()
    app.calendar_adapter = stub_calendar
    app.channel_adapter = _CapturingChannelAdapter()
    await app.startup()
    try:
        created = await app.crm_adapter.create_contact("573156832405", "Lead Inicial")
        current = await app.crm_adapter.find_contact_by_phone("573156832405")
        assert current is not None
        shadow = apply_name_validation_metadata(
            current,
            NameCandidateAssessment(
                status="needs_confirmation",
                confidence=0.95,
                normalized_name="Fabian C Villegas",
                candidate_name="Fabian C Villegas",
                source="provider_llm",
            ),
        )
        await app.memory_store.remember_contact(shadow)

        async def fake_resolve(text, contact, recent_messages):  # noqa: ANN001
            if text == "Fabian Cuero\nfabiancuerov@gmail.com":
                return NameConfirmationDecision(
                    status="provided_new_name",
                    confidence=0.99,
                    resolved_name="Fabian Cuero",
                )
            return None

        def fake_plan_with_rules(text, contact):  # noqa: ANN001
            if text == "Hola":
                return PlanningResult(
                    intent="greeting",
                    confidence=0.8,
                    response_text="Hola, ¿en qué puedo ayudarte hoy con Wabog?",
                    actions=[],
                )
            if text == "Llevo 300 casos al mes y hago el seguimiento manualmente entrando a la rama y los llevo en un excel los casos.":
                return PlanningResult(
                    intent="qualification",
                    confidence=0.84,
                    response_text="Con ese volumen, Wabog puede ayudarte a automatizar el seguimiento. ¿Quieres que agendemos una demo?",
                    actions=[],
                )
            if text == "Cuánto cuesta ?":
                return PlanningResult(
                    intent="pricing",
                    confidence=0.85,
                    response_text="Para ese volumen, lo ideal es revisar Enterprise en una demo.",
                    actions=[],
                )
            if text == "Si agéndala":
                return PlanningResult(
                    intent="schedule_interest",
                    confidence=0.9,
                    response_text="Perfecto. Para enviarte la invitación, compárteme también tu correo.",
                    actions=[],
                )
            if text == "Fabian Cuero\nfabiancuerov@gmail.com":
                return PlanningResult(
                    intent="collect_contact_data",
                    confidence=0.92,
                    response_text="Perfecto. Ya tengo tus datos. ¿Qué día y hora te sirve para la demo?",
                    actions=[],
                )
            if text == "Mañana a las 10am":
                return PlanningResult(
                    intent="schedule_demo",
                    confidence=0.93,
                    response_text="Perfecto, reviso ese horario.",
                    actions=[],
                )
            if text == "Ya no puedo a las 10 am mañana podemos re programar a las 11 am ?":
                return PlanningResult(
                    intent="reschedule_demo",
                    confidence=0.94,
                    response_text="Perfecto, voy a mover la demo al nuevo horario.",
                    actions=[],
                )
            raise AssertionError(f"Unexpected text in test: {text}")

        app.workflow.planner._name_confirmation_resolver.resolve = fake_resolve  # noqa: SLF001
        app.workflow.planner._plan_with_rules = fake_plan_with_rules  # noqa: SLF001

        turns = [
            ("prod-sim-1", "Hola"),
            ("prod-sim-2", "Llevo 300 casos al mes y hago el seguimiento manualmente entrando a la rama y los llevo en un excel los casos."),
            ("prod-sim-3", "Cuánto cuesta ?"),
            ("prod-sim-4", "Si agéndala"),
            ("prod-sim-5", "Fabian Cuero\nfabiancuerov@gmail.com"),
            ("prod-sim-6", "Mañana a las 10am"),
            ("prod-sim-7", "Ya no puedo a las 10 am mañana podemos re programar a las 11 am ?"),
        ]
        results = []
        for message_id, text in turns:
            results.append(
                await app.handle_event(
                    InboundMessage(
                        message_id=message_id,
                        conversation_id="conv-prod-sim",
                        phone_number="573156832405",
                        text=text,
                        timestamp=datetime.now(timezone.utc),
                        raw_payload={},
                        provider="local-playground",
                    ),
                    wait_for_response=True,
                )
            )
    finally:
        await app.shutdown()

    pending_name_reply = results[0].response_text
    assert "tu nombre es fabian c villegas" in pending_name_reply.lower()

    contact_after_name = results[4].contact
    assert contact_after_name is not None
    assert contact_after_name.full_name == "Fabian Cuero"
    assert contact_after_name.email == "fabiancuerov@gmail.com"

    booking_result = results[5]
    assert any(item.action.type == ActionType.CREATE_MEETING and item.success for item in booking_result.tool_results)
    first_event_id = next(
        item.payload["id"]
        for item in booking_result.tool_results
        if item.action.type == ActionType.CREATE_MEETING and item.success
    )
    assert booking_result.contact is not None
    assert booking_result.contact.full_name == "Fabian Cuero"
    assert booking_result.contact.metadata["name_validation"]["normalized_name"] == "Fabian Cuero"

    reschedule_result = results[6]
    assert any(item.action.type == ActionType.CREATE_MEETING and item.success for item in reschedule_result.tool_results)
    assert any(item.action.type == ActionType.DELETE_MEETING and item.success for item in reschedule_result.tool_results)
    assert first_event_id in stub_calendar.deleted_event_ids
    assert "11:00:00-05:00" in reschedule_result.response_text
    assert reschedule_result.contact is not None
    assert reschedule_result.contact.full_name == "Fabian Cuero"
    assert reschedule_result.contact.metadata["calendar"]["upcoming_event"]["start_iso"].endswith("11:00:00-05:00")
