import pytest
from datetime import date, timedelta

from sales_agent.core.config import Settings
from sales_agent.domain.models import CRMContact, PlanningResult, PromptMode, ProposedAction
from sales_agent.domain.models import ActionType
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


def test_sales_policy_infers_trial_stage_note_and_followup_for_new_lead():
    planner = build_planner()
    result = PlanningResult(
        intent="trial_interest",
        confidence=0.81,
        response_text="Te ayudo a revisar si Wabog encaja.",
        actions=[],
    )

    enforced = planner._enforce_sales_policy(  # noqa: SLF001
        result,
        "Hola, soy abogada independiente y quiero probar Wabog para mi firma.",
        None,
        [],
    )

    action_types = [action.type for action in enforced.actions]

    assert ActionType.UPDATE_STAGE in action_types
    assert ActionType.APPEND_NOTE in action_types
    assert ActionType.CREATE_FOLLOWUP in action_types
    stage_action = next(action for action in enforced.actions if action.type == ActionType.UPDATE_STAGE)
    assert stage_action.args["stage"] == "Prueba / Trial"


def test_sales_policy_advances_existing_lead_to_negotiation():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Demo",
        stage="Propuesta enviada",
    )
    result = PlanningResult(
        intent="generic_reply",
        confidence=0.7,
        response_text="Revisemos tu caso.",
        actions=[
            ProposedAction(
                type=ActionType.APPEND_NOTE,
                reason="El lead pidió propuesta.",
                args={"note": "Perfecto, envíame una propuesta y luego revisamos negociación."},
            )
        ],
    )

    enforced = planner._enforce_sales_policy(  # noqa: SLF001
        result,
        "Perfecto, envíame una propuesta y luego revisamos negociación.",
        contact,
        [],
    )

    stage_action = next(action for action in enforced.actions if action.type == ActionType.UPDATE_STAGE)
    assert stage_action.args["stage"] == "Negociación"


def test_sales_policy_does_not_duplicate_existing_note_action():
    planner = build_planner()
    result = PlanningResult(
        intent="demo_interest",
        confidence=0.8,
        response_text="Agendemos la demo.",
        actions=[
            ProposedAction(
                type=ActionType.APPEND_NOTE,
                reason="El modelo ya capturó la intención comercial.",
                args={"note": "Lead quiere demo para su despacho."},
            )
        ],
    )

    enforced = planner._enforce_sales_policy(  # noqa: SLF001
        result,
        "Quiero una demo de Wabog para mi despacho.",
        None,
        [],
    )

    note_actions = [action for action in enforced.actions if action.type == ActionType.APPEND_NOTE]
    assert len(note_actions) == 1


def test_repair_actions_infers_demo_stage_from_affirmative_reply_and_recent_context():
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
    assert stage_action.args["stage"] == "Primer contacto"


def test_repair_actions_infers_first_contact_for_buying_intent_without_stage():
    planner = build_planner()
    result = PlanningResult(
        intent="buying_intent",
        confidence=0.71,
        response_text="Cuéntame más sobre tu despacho.",
        actions=[
            ProposedAction(
                type=ActionType.UPDATE_STAGE,
                reason="El lead quiere adquirir Wabog.",
                args={},
            )
        ],
    )

    repaired = planner._repair_actions(  # noqa: SLF001
        result,
        "quisiera adquirir wabog",
        None,
        [],
    )

    stage_action = next(action for action in repaired.actions if action.type == ActionType.UPDATE_STAGE)
    assert stage_action.args["stage"] == "Primer contacto"


def test_repair_actions_rewrites_raw_note_into_natural_crm_summary():
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
    assert "Despacho JD" in note_action.args["note"]
    assert "200 procesos al mes" in note_action.args["note"]
    assert note_action.args["note"] != "si mi despacho se llama JD y manejamos como 200 procesos al mes"


def test_build_sales_note_summarizes_tool_pain_and_next_step_naturally():
    planner = build_planner()

    note = planner._build_sales_note(  # noqa: SLF001
        "si usamos icarus una app, es una herramienta vieja y aveces no nos notifica bien ademas quisieramo poder recibir en whatsapp las notificaciones",
        [],
    )

    assert "Icarus" in note
    assert "notificaciones" in note
    assert "WhatsApp" in note


def test_repair_actions_rewrites_followup_summary_when_it_is_raw_text():
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
    assert followup_action.args["summary"] == "Coordinar fecha y hora para demo comercial."
    assert followup_action.args["due_date"] == (date.today() + timedelta(days=1)).isoformat()


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


def test_sales_policy_requests_name_and_email_before_creating_meeting():
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
        actions=[],
    )

    enforced = planner._enforce_sales_policy(  # noqa: SLF001
        result,
        "Agendemos la demo mañana a las 3 pm",
        contact,
        [],
    )

    assert all(action.type != ActionType.CREATE_MEETING for action in enforced.actions)
    assert any(action.type == ActionType.CREATE_FOLLOWUP for action in enforced.actions)
    assert "nombre completo" in enforced.response_text.lower()
    assert "correo" in enforced.response_text.lower()


def test_infer_stage_keeps_demo_interest_in_first_contact_without_exact_slot():
    planner = build_planner()

    inferred_stage = planner._infer_stage_from_text(  # noqa: SLF001
        "puede ser una demo la otra semana tal vez",
        "Prospecto",
        [],
    )

    assert inferred_stage == "Primer contacto"


def test_sales_policy_appends_self_schedule_link_for_demo_interest():
    planner = build_planner_with_calendar_link()
    result = PlanningResult(
        intent="demo_interest",
        confidence=0.8,
        response_text="Perfecto. Coordinemos la demo.",
        actions=[],
    )

    enforced = planner._enforce_sales_policy(  # noqa: SLF001
        result,
        "Quiero agendar una demo",
        None,
        [],
    )

    assert "calendar.app.google" in enforced.response_text


def test_sales_policy_uses_upcoming_calendar_event_to_complete_followup():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Demo",
        stage="Primer contacto",
        followup_summary="Coordinar fecha y hora para demo comercial.",
        followup_due_date=date(2026, 4, 17),
        metadata={
            "calendar": {
                "upcoming_event": {
                    "id": "evt-1",
                    "start_iso": "2026-04-17T15:00:00-05:00",
                }
            }
        },
    )
    result = PlanningResult(
        intent="generic_reply",
        confidence=0.7,
        response_text="Perfecto.",
        actions=[],
    )

    enforced = planner._enforce_sales_policy(  # noqa: SLF001
        result,
        "hola",
        contact,
        [],
    )

    completion_action = next(action for action in enforced.actions if action.type == ActionType.COMPLETE_FOLLOWUP)
    assert "2026-04-17T15:00:00-05:00" in completion_action.args["outcome"]
    assert "demo futura" in enforced.response_text.lower()


def test_sales_policy_does_not_complete_followup_twice_when_calendar_event_exists_but_followup_is_cleared():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Demo",
        stage="Demo agendada",
        followup_summary=None,
        metadata={
            "calendar": {
                "upcoming_event": {
                    "id": "evt-1",
                    "start_iso": "2026-04-17T15:00:00-05:00",
                }
            }
        },
    )
    result = PlanningResult(
        intent="generic_reply",
        confidence=0.7,
        response_text="Perfecto.",
        actions=[],
    )

    enforced = planner._enforce_sales_policy(  # noqa: SLF001
        result,
        "hola",
        contact,
        [],
    )

    assert all(action.type != ActionType.COMPLETE_FOLLOWUP for action in enforced.actions)
    assert "demo futura" in enforced.response_text.lower()


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


def test_sales_policy_marks_active_followup_as_completed_when_user_confirms_done():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Demo",
        stage="Propuesta enviada",
        followup_summary="Enviar propuesta comercial.",
        followup_due_date=date(2026, 4, 16),
    )
    result = PlanningResult(
        intent="generic_reply",
        confidence=0.7,
        response_text="Perfecto.",
        actions=[],
    )

    enforced = planner._enforce_sales_policy(  # noqa: SLF001
        result,
        "Listo, ya envié la propuesta al cliente",
        contact,
        [],
    )

    completion_action = next(action for action in enforced.actions if action.type == ActionType.COMPLETE_FOLLOWUP)
    assert "propuesta" in completion_action.args["outcome"].lower()
    assert all(action.type != ActionType.CREATE_FOLLOWUP for action in enforced.actions)


def test_sales_policy_does_not_complete_followup_for_future_reminder_request():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Demo",
        stage="Propuesta enviada",
        followup_summary="Enviar propuesta comercial.",
        followup_due_date=date(2026, 4, 16),
    )
    result = PlanningResult(
        intent="generic_reply",
        confidence=0.7,
        response_text="Perfecto.",
        actions=[],
    )

    enforced = planner._enforce_sales_policy(  # noqa: SLF001
        result,
        "recordarme mañana enviar la propuesta",
        contact,
        [],
    )

    assert all(action.type != ActionType.COMPLETE_FOLLOWUP for action in enforced.actions)


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


class _FakePromptStore:
    async def get_business_prompt(self, mode: PromptMode) -> str:
        return "DRAFT brief." if mode == PromptMode.DRAFT else "PUBLISHED brief."


@pytest.mark.asyncio
async def test_planner_uses_prompt_mode_to_load_editable_business_prompt():
    planner = AgentPlanner(Settings(OPENAI_API_KEY=""), prompt_store=_FakePromptStore())
    planner._llm = _FakeStructuredOutput()  # noqa: SLF001

    await planner.plan("hola", None, [], [], prompt_mode=PromptMode.DRAFT)
    assert "DRAFT brief." in planner._llm.last_prompt  # noqa: SLF001

    await planner.plan("hola", None, [], [], prompt_mode=PromptMode.PUBLISHED)
    assert "PUBLISHED brief." in planner._llm.last_prompt  # noqa: SLF001
