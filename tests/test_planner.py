import pytest
from datetime import date, timedelta

from sales_agent.core.config import Settings
from sales_agent.domain.models import ActionType, CRMContact, PlanningResult, ProposedAction
from sales_agent.services.planner import AgentPlanner


def build_planner() -> AgentPlanner:
    return AgentPlanner(Settings(OPENAI_API_KEY=""))


def build_planner_with_calendar_link() -> AgentPlanner:
    return AgentPlanner(
        Settings(
            OPENAI_API_KEY="",
            GOOGLE_CALENDAR_SELF_SCHEDULE_URL="https://calendar.app.google/9KpgMYTW4nA17LV27",
        )
    )


def test_plan_with_rules_does_not_make_commercial_decisions_from_keywords():
    planner = build_planner()

    result = planner._plan_with_rules(  # noqa: SLF001
        "Hola, soy abogada independiente y quiero probar Wabog para mi firma.",
        None,
    )

    assert result.actions == []
    assert result.intent == "generic_reply"


def test_repair_actions_does_not_guess_stage_when_llm_omits_it():
    planner = build_planner()
    result = PlanningResult(
        intent="demo_confirmation",
        confidence=0.73,
        response_text="Perfecto, coordinemos.",
        actions=[
            ProposedAction(
                type=ActionType.UPDATE_STAGE,
                reason="El lead confirmó que quiere avanzar.",
                args={},
            )
        ],
    )

    repaired = planner._repair_actions(  # noqa: SLF001
        result,
        "si claro",
        None,
        ["¿Les gustaría agendar una demo para mostrarles cómo funciona Wabog?"],
    )

    stage_action = next(action for action in repaired.actions if action.type == ActionType.UPDATE_STAGE)
    assert stage_action.args == {}


def test_repair_actions_preserves_note_without_keyword_rewrite():
    planner = build_planner()
    result = PlanningResult(
        intent="demo_interest",
        confidence=0.8,
        response_text="Agendemos la demo.",
        actions=[
            ProposedAction(
                type=ActionType.APPEND_NOTE,
                reason="El modelo dejó la nota cruda.",
                args={"note": "si mi despacho se llama JD y manejamos como 200 procesos al mes"},
            )
        ],
    )

    repaired = planner._repair_actions(  # noqa: SLF001
        result,
        "si mi despacho se llama JD y manejamos como 200 procesos al mes",
        None,
        [],
    )

    note_action = next(action for action in repaired.actions if action.type == ActionType.APPEND_NOTE)
    assert note_action.args["note"] == "si mi despacho se llama JD y manejamos como 200 procesos al mes"


def test_build_sales_note_summarizes_tool_pain_and_next_step_naturally():
    planner = build_planner()

    note = planner._build_sales_note(  # noqa: SLF001
        "si usamos icarus una app, es una herramienta vieja y aveces no nos notifica bien ademas quisieramo poder recibir en whatsapp las notificaciones",
        [],
    )

    assert "Icarus" in note
    assert "notificaciones" in note
    assert "WhatsApp" in note


def test_repair_actions_preserves_followup_summary_without_keyword_rewrite():
    planner = build_planner()
    result = PlanningResult(
        intent="demo_interest",
        confidence=0.8,
        response_text="Agendemos.",
        actions=[
            ProposedAction(
                type=ActionType.CREATE_FOLLOWUP,
                reason="Resumen crudo del modelo.",
                args={"summary": "si me pareceria bien agendar una demo"},
            )
        ],
    )

    repaired = planner._repair_actions(  # noqa: SLF001
        result,
        "si me pareceria bien agendar una demo",
        None,
        [],
    )

    followup_action = next(action for action in repaired.actions if action.type == ActionType.CREATE_FOLLOWUP)
    assert followup_action.args["summary"] == "si me pareceria bien agendar una demo"
    assert "due_date" not in followup_action.args


def test_build_meeting_payload_for_explicit_demo_time():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Demo",
        email="lead@example.com",
    )

    payload = planner._build_meeting_payload(  # noqa: SLF001
        "Agendemos la demo mañana a las 3 pm",
        contact,
    )

    assert payload is not None
    assert payload["duration_minutes"] == planner._settings.google_calendar_default_meeting_minutes  # noqa: SLF001
    assert payload["title"] == "Demo Wabog - Lead Demo"
    assert "lead@example.com" in payload["description"]


def test_build_meeting_payload_uses_recent_context_for_day_and_hour():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Demo",
        email="lead@example.com",
    )

    payload = planner._build_meeting_payload(  # noqa: SLF001
        "9 am me sirve",
        contact,
        ["Podemos agendar una demo el martes en la mañana."],
    )

    assert payload is not None
    assert "T09:00:00" in payload["start_iso"]


def test_build_meeting_payload_uses_recent_afternoon_context_for_bare_hour():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Demo",
        email="lead@example.com",
    )

    payload = planner._build_meeting_payload(  # noqa: SLF001
        "a las 6",
        contact,
        ["Podemos agendar una demo el miércoles que viene en la tarde."],
    )

    assert payload is not None
    assert "T18:00:00" in payload["start_iso"]


def test_planning_guardrail_requests_name_and_email_before_creating_meeting():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name=None,
        email=None,
        stage="Primer contacto",
    )
    result = PlanningResult(
        intent="demo_interest",
        confidence=0.82,
        response_text="Perfecto, te la agendo.",
        actions=[
            ProposedAction(
                type=ActionType.CREATE_MEETING,
                reason="El modelo decidió agendar la demo.",
                args={},
            )
        ],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "Agendemos la demo mañana a las 3 pm",
        contact,
        [],
    )

    assert all(action.type != ActionType.CREATE_MEETING for action in guarded.actions)
    assert "nombre completo" in guarded.response_text.lower()
    assert "correo" in guarded.response_text.lower()


def test_planning_guardrail_does_not_create_meeting_from_recent_context_when_llm_did_not_request_it():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Demo",
        email="lead@example.com",
        stage="Primer contacto",
        followup_summary="Coordinar fecha y hora para demo comercial.",
    )
    result = PlanningResult(
        intent="confirm_demo",
        confidence=0.79,
        response_text="Gracias por confirmar la demo. Procederé a enviar la invitación.",
        actions=[
            ProposedAction(
                type=ActionType.UPDATE_STAGE,
                reason="Demo confirmada para martes a las 9 am.",
                args={"stage": "Primer contacto"},
            ),
            ProposedAction(
                type=ActionType.COMPLETE_FOLLOWUP,
                reason="La demo quedó lista.",
                args={"outcome": "demo confirmada"},
            ),
        ],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "si, para la demo del martes a las 9",
        contact,
        [
            "Perfecto. Para enviarte la invitación de la demo, compárteme tu correo.",
            "mi correo es lead@example.com",
            "Perfecto, ya tengo tu correo de contacto.",
            "martes 9 am",
        ],
    )

    action_types = [action.type for action in guarded.actions]
    assert ActionType.CREATE_MEETING not in action_types
    assert ActionType.COMPLETE_FOLLOWUP in action_types
    assert "procederé a enviar la invitación" not in guarded.response_text.lower()


def test_planning_guardrail_removes_false_booking_claim_without_meeting_action():
    planner = build_planner()
    result = PlanningResult(
        intent="confirm_demo",
        confidence=0.75,
        response_text="Gracias por confirmar la demo para el martes. Procederé a enviar la invitación.",
        actions=[],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "si",
        None,
        ["martes 9 am"],
    )

    assert "procederé a enviar la invitación" not in guarded.response_text.lower()
    assert "correctamente agendada" in guarded.response_text.lower()


def test_planning_guardrail_drops_empty_append_note_and_followup_actions():
    planner = build_planner()
    result = PlanningResult(
        intent="generic_reply",
        confidence=0.75,
        response_text="Respuesta del modelo.",
        actions=[
            ProposedAction(type=ActionType.APPEND_NOTE, reason="Vacía.", args={}),
            ProposedAction(type=ActionType.CREATE_FOLLOWUP, reason="Vacío.", args={}),
        ],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "no quiero agendar demo",
        None,
        [],
    )

    assert guarded.actions == []


def test_planning_guardrail_does_not_project_stage_or_meeting_from_pending_contact_fields():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Carlos Ruiz",
        stage="Primer contacto",
    )
    result = PlanningResult(
        intent="provide_email",
        confidence=0.8,
        response_text="Perfecto, ya casi queda.",
        actions=[
            ProposedAction(
                type=ActionType.UPDATE_CONTACT_FIELDS,
                reason="El lead compartió su correo.",
                args={"fields": {"email": "carlos@example.com"}},
            ),
        ],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "mi correo es carlos@example.com",
        contact,
        ["puede ser una demo", "martes 9 am"],
    )

    assert all(action.type != ActionType.UPDATE_STAGE for action in guarded.actions)
    assert all(action.type != ActionType.CREATE_MEETING for action in guarded.actions)


def test_plan_with_rules_does_not_append_self_schedule_link_from_demo_keywords():
    planner = build_planner_with_calendar_link()
    result = PlanningResult(
        intent="demo_interest",
        confidence=0.8,
        response_text="Perfecto. Coordinemos la demo.",
        actions=[],
    )

    planned = planner._plan_with_rules(  # noqa: SLF001
        "Quiero agendar una demo",
        None,
    )

    assert "calendar.app.google" not in planned.response_text


def test_plan_with_rules_creates_contact_update_action_for_email_and_name():
    planner = build_planner()

    result = planner._plan_with_rules(  # noqa: SLF001
        "Hola, mi nombre es juan perez y mi correo es juan@example.com",
        None,
    )

    update_action = next(action for action in result.actions if action.type == ActionType.UPDATE_CONTACT_FIELDS)
    assert update_action.args["fields"] == {
        "full_name": "Juan Perez",
        "email": "juan@example.com",
    }
    assert "correo" in result.response_text.lower()


@pytest.mark.asyncio
async def test_hard_rule_answers_name_from_current_contact():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Carlos Ruiz",
    )

    result = await planner.plan("como me llamo?", contact, [], [])

    assert result.intent == "ask_name"
    assert result.actions == []
    assert "Carlos Ruiz" in result.response_text


@pytest.mark.asyncio
async def test_hard_rule_asks_for_name_when_contact_name_is_missing_or_placeholder():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="~",
    )

    result = await planner.plan("como me llamo", contact, [], [])

    assert result.intent == "ask_name"
    assert result.actions == []
    assert "todavía no tengo tu nombre" in result.response_text.lower()


@pytest.mark.asyncio
async def test_hard_rule_asks_to_confirm_candidate_name_when_it_is_not_trusted_yet():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name=None,
        metadata={
            "name_validation": {
                "status": "needs_confirmation",
                "candidate_name": "Juan",
                "normalized_name": "Juan",
                "confidence": 0.62,
                "source": "provider",
            }
        },
    )

    result = await planner.plan("como me llamo", contact, [], [])

    assert result.intent == "ask_name"
    assert "juan" in result.response_text.lower()
    assert "confirmarlo" in result.response_text.lower()


@pytest.mark.asyncio
async def test_hard_rule_does_not_treat_phone_number_as_contact_name():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="3150000000",
    )

    result = await planner.plan("como me llamo", contact, [], [])

    assert result.intent == "ask_name"
    assert "todavía no tengo tu nombre" in result.response_text.lower()


@pytest.mark.asyncio
async def test_hard_rule_distinguishes_contact_source_questions():
    planner = build_planner()

    result = await planner.plan("de donde sacaron mi numero?", None, [], [])

    assert result.intent == "ask_contact_source"
    assert result.actions == []
    assert "te contactamos" in result.response_text.lower()


@pytest.mark.asyncio
async def test_hard_rule_returns_real_demo_link_when_upcoming_event_exists():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Carlos Ruiz",
        metadata={
            "calendar": {
                "upcoming_event": {
                    "id": "evt-1",
                    "start_iso": "2026-04-21T18:00:00-05:00",
                    "meet_link": "https://meet.google.com/abc-defg-hij",
                }
            }
        },
    )

    result = await planner.plan("mandame el link por aqui", contact, [], [])

    assert result.intent == "request_demo_link"
    assert result.actions == []
    assert "https://meet.google.com/abc-defg-hij" in result.response_text


def test_append_calendar_confirmation_removes_unsupported_reminder_promise():
    planner = build_planner()

    response_text = planner._append_calendar_confirmation(  # noqa: SLF001
        "Perfecto, Carlos. Ya tienes la demo agendada para el martes 21 de abril a las 9:00 AM. Te enviaré un recordatorio antes de la fecha.",
        {"start_iso": "2026-04-21T09:00:00-05:00"},
    )

    assert "recordatorio" not in response_text.lower()
    assert "veo en calendario una demo futura" in response_text.lower()


def test_missing_meeting_fields_response_confirms_candidate_name_before_booking():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        email="lead@example.com",
        metadata={
            "name_validation": {
                "status": "needs_confirmation",
                "candidate_name": "Juan",
                "normalized_name": "Juan",
                "confidence": 0.62,
                "source": "provider",
            }
        },
    )

    response = planner._build_missing_meeting_fields_response(["full_name"], contact)  # noqa: SLF001

    assert "tu nombre es Juan" in response


@pytest.mark.asyncio
async def test_plan_with_llm_does_not_add_meeting_or_followup_when_user_rejects_demo():
    planner = build_planner_with_calendar_link()
    planner._llm = _FakeStructuredOutput()  # type: ignore[assignment]  # noqa: SLF001
    planner._knowledge_selector_llm = _FakeKnowledgeSelector()  # type: ignore[assignment]  # noqa: SLF001

    result = await planner.plan(
        "mm no quiero agendar demo",
        CRMContact(external_id="lead-1", phone_number="3150000000", full_name="Juan Perez"),
        [],
        [],
    )

    assert all(action.type != ActionType.CREATE_MEETING for action in result.actions)
    assert all(action.type != ActionType.CREATE_FOLLOWUP for action in result.actions)
    assert "calendar.app.google" not in result.response_text


class _FakeStructuredOutput:
    def __init__(self) -> None:
        self.last_prompt = ""

    async def ainvoke(self, prompt: str):
        self.last_prompt = prompt

        class _Response:
            def model_dump(self_nonlocal):  # noqa: ANN001
                return {
                    "intent": "generic_reply",
                    "confidence": 0.75,
                    "response_text": "Respuesta del modelo.",
                    "should_reply": True,
                    "actions": [],
                }

        return _Response()


class _FakeKnowledgeSelector:
    async def ainvoke(self, prompt: str):
        class _Response:
            def model_dump(self_nonlocal):  # noqa: ANN001
                return {
                    "section_names": ["wabog_pricing"],
                    "reason": "Pricing question.",
                }

        return _Response()


@pytest.mark.asyncio
async def test_planner_loads_selected_knowledge_section_into_prompt():
    planner = AgentPlanner(Settings(OPENAI_API_KEY=""))
    planner._llm = _FakeStructuredOutput()  # noqa: SLF001
    planner._knowledge_selector_llm = _FakeKnowledgeSelector()  # noqa: SLF001

    result = await planner.plan("y eso cuanto vale", None, [], [])
    assert "Wabog Pricing" in planner._llm.last_prompt  # noqa: SLF001
    assert any(item.section == "wabog_pricing" for item in result.knowledge_lookups)
