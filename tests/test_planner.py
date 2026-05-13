import pytest
from datetime import date, timedelta

from sales_agent.core.config import Settings
from sales_agent.domain.models import ActionType, CRMContact, PlanningResult, ProposedAction
from sales_agent.services.planner import AgentPlanner, SemanticGuardrailDecision
from sales_agent.services.name_validation import NameConfirmationDecision


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


def test_serialize_contact_for_prompt_uses_curated_crm_context():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Fabian Cuero Villegas",
        email="fabian@example.com",
        stage="Primer contacto",
        followup_summary="Coordinar fecha y hora para demo comercial.",
        followup_due_date=date.today() + timedelta(days=1),
        notes=[
            "2026-05-01 - Lead confirmó volumen alto.",
            "2026-05-02 - Usa Excel para seguimiento.",
        ],
        metadata={
            "raw_properties": {"Nombre": "ruido"},
            "name_validation": {"status": "trusted"},
            "calendar": {"connected": True, "upcoming_event": None},
        },
    )

    rendered = planner._serialize_contact_for_prompt(contact)  # noqa: SLF001

    assert "Coordinar fecha y hora para demo comercial." in rendered
    assert "2026-05-02 - Usa Excel para seguimiento." in rendered
    assert "raw_properties" not in rendered
    assert "Fabian Cuero Villegas" in rendered


def test_serialize_contact_for_prompt_omits_past_followup():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Fabian Cuero Villegas",
        email="fabian@example.com",
        stage="Primer contacto",
        followup_summary="Asistir a la demo agendada para 2026-05-07T15:00:00-05:00 y continuar seguimiento comercial.",
        followup_due_date=date(2026, 5, 7),
        metadata={"name_validation": {"status": "trusted", "source": "user_message"}},
    )

    rendered = planner._serialize_contact_for_prompt(contact)  # noqa: SLF001

    assert '"followup_summary": null' in rendered
    assert '"followup_due_date": null' in rendered
    assert "2026-05-07T15:00:00-05:00" not in rendered


def test_serialize_contact_for_prompt_omits_past_cached_upcoming_event():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Fabian Cuero Villegas",
        email="fabian@example.com",
        stage="Demo agendada",
        metadata={
            "name_validation": {"status": "trusted", "source": "user_message"},
            "calendar": {
                "connected": True,
                "available": True,
                "upcoming_event": {
                    "id": "past-event",
                    "start_iso": "2026-05-07T15:00:00-05:00",
                    "end_iso": "2026-05-07T15:30:00-05:00",
                },
                "just_booked": True,
            },
        },
    )

    rendered = planner._serialize_contact_for_prompt(contact)  # noqa: SLF001

    assert '"upcoming_event": null' in rendered
    assert '"has_future_meeting": false' in rendered
    assert '"just_booked": false' in rendered


def test_compose_llm_prompt_includes_current_datetime():
    planner = build_planner()

    prompt = planner._compose_llm_prompt(  # noqa: SLF001
        contact_json="None",
        recent_messages="[]",
        semantic_memories="[]",
        knowledge_context="",
        text="hola",
    )

    assert "Current datetime (America/Bogota):" in prompt
    assert "Recent messages and semantic memories are historical conversation text" in prompt
    assert "If Contact.metadata.calendar.upcoming_event is null" in prompt


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


def test_planning_guardrail_requests_email_before_creating_meeting():
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
    assert "nombre completo" not in guarded.response_text.lower()
    assert "correo" in guarded.response_text.lower()


def test_repair_actions_uses_contextual_name_confirmation_to_fill_full_name():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Juan Perez",
        email="fabian@example.com",
        metadata={
            "name_validation": {
                "status": "needs_confirmation",
                "candidate_name": "Fabian C Villegas",
                "normalized_name": "Fabian C Villegas",
                "source": "provider_llm",
            }
        },
    )
    result = PlanningResult(
        intent="confirm_name",
        confidence=0.81,
        response_text="Perfecto. Antes de agendarla, ¿tu nombre es Fabian C Villegas?",
        actions=[
            ProposedAction(
                type=ActionType.UPDATE_CONTACT_FIELDS,
                reason="Confirmación pendiente del nombre.",
                args={"fields": {}},
            )
        ],
    )

    repaired = planner._repair_actions(  # noqa: SLF001
        result,
        "Ese es mi nombre agendala.",
        contact,
        [],
        NameConfirmationDecision(
            status="confirmed_candidate_name",
            confidence=0.97,
            resolved_name="Fabian C Villegas",
        ),
    )

    update_action = next(action for action in repaired.actions if action.type == ActionType.UPDATE_CONTACT_FIELDS)
    assert update_action.args["fields"]["full_name"] == "Fabian C Villegas"


def test_repair_actions_appends_explicit_contact_fields_even_when_llm_omits_update_action():
    planner = build_planner()
    result = PlanningResult(
        intent="collect_contact_data",
        confidence=0.8,
        response_text="Gracias, sigo contigo.",
        actions=[],
    )

    repaired = planner._repair_actions(  # noqa: SLF001
        result,
        "prueba.calendar.qa@example.com",
        None,
        [],
        NameConfirmationDecision(
            status="provided_new_name",
            confidence=0.98,
            resolved_name="Prueba Calendar Qa",
        ),
    )

    update_action = next(action for action in repaired.actions if action.type == ActionType.UPDATE_CONTACT_FIELDS)
    assert update_action.args["fields"] == {
        "full_name": "Prueba Calendar Qa",
        "email": "prueba.calendar.qa@example.com",
    }


def test_repair_actions_uses_contextual_resolver_for_standalone_full_name_message():
    planner = build_planner()
    result = PlanningResult(
        intent="name_correction",
        confidence=0.8,
        response_text="Gracias por la corrección.",
        actions=[],
    )

    repaired = planner._repair_actions(  # noqa: SLF001
        result,
        "Fabian Cuero Villegas",
        None,
        ["Tu nombre actual figura como Juan Perez. Si quieres, compárteme tu nombre completo correcto."],
        NameConfirmationDecision(
            status="provided_new_name",
            confidence=0.99,
            resolved_name="Fabian Cuero Villegas",
        ),
    )

    update_action = next(action for action in repaired.actions if action.type == ActionType.UPDATE_CONTACT_FIELDS)
    assert update_action.args["fields"] == {"full_name": "Fabian Cuero Villegas"}


def test_planning_guardrail_recreates_existing_meeting_when_new_time_is_requested():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Fabian Cuero",
        email="fabian@example.com",
        stage="Demo agendada",
        metadata={
            "name_validation": {"status": "trusted", "source": "user_message"},
            "calendar": {
                "connected": True,
                "upcoming_event": {
                    "id": "evt-old",
                    "start_iso": "2030-05-07T10:00:00-05:00",
                },
            },
        },
    )
    result = PlanningResult(
        intent="reprogram_demo",
        confidence=0.87,
        response_text="Voy a revisar la reprogramación.",
        actions=[],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "mañana a las 11 am",
        contact,
        ["La demo quedó agendada para mañana a las 10 am."],
    )

    assert [action.type for action in guarded.actions] == [
        ActionType.CREATE_MEETING,
        ActionType.DELETE_MEETING,
    ]
    assert guarded.actions[0].args["start_iso"].endswith("11:00:00-05:00")
    assert guarded.actions[1].args["event_id"] == "evt-old"


def test_planning_guardrail_recreates_existing_meeting_when_only_new_hour_is_requested():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Fabian Cuero",
        email="fabian@example.com",
        stage="Demo agendada",
        metadata={
            "name_validation": {"status": "trusted", "source": "user_message"},
            "calendar": {
                "connected": True,
                "upcoming_event": {
                    "id": "evt-old",
                    "start_iso": "2030-05-07T11:00:00-05:00",
                },
            },
        },
    )
    result = PlanningResult(
        intent="reprogram_demo",
        confidence=0.87,
        response_text="Voy a revisar la reprogramación.",
        actions=[],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "Puedes re programarla para las 2 pm ?",
        contact,
        [],
    )

    assert [action.type for action in guarded.actions] == [
        ActionType.CREATE_MEETING,
        ActionType.DELETE_MEETING,
    ]
    assert guarded.actions[0].args["start_iso"] == "2030-05-07T14:00:00-05:00"
    assert guarded.actions[1].args["event_id"] == "evt-old"


def test_planning_guardrail_creates_meeting_from_context_day_and_current_hour():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Fabian Cuero",
        email="fabian@example.com",
        stage="Primer contacto",
        metadata={"name_validation": {"status": "trusted", "source": "user_message"}},
    )
    result = PlanningResult(
        intent="reprogram_demo",
        confidence=0.87,
        response_text="Voy a revisar la reprogramación.",
        actions=[],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "Puedes re programarla para las 2 pm ?",
        contact,
        ["¿Quieres que te ayude con la demo que tienes agendada para mañana a las 11 am?"],
    )

    assert [action.type for action in guarded.actions] == [ActionType.CREATE_MEETING]
    assert guarded.actions[0].args["start_iso"].endswith("T14:00:00-05:00")


def test_planning_guardrail_does_not_repeat_pending_name_prompt_after_it_was_asked():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Inicial",
        email="lead@example.com",
        metadata={
            "name_validation": {
                "status": "needs_confirmation",
                "candidate_name": "Fabian C Villegas",
                "normalized_name": "Fabian C Villegas",
                "source": "provider_llm",
            }
        },
    )
    result = PlanningResult(
        intent="consult_price",
        confidence=0.8,
        response_text="El plan Enterprise se revisa con ventas según volumen.",
        actions=[],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "¿cuánto cuesta?",
        contact,
        [
            "Buenos días, ¿cómo puedo ayudarte hoy con Wabog? Antes de seguir, ¿tu nombre es Fabian C Villegas? Si no, compárteme tu nombre completo.",
            "Si Fabian",
            "Tengo 99 casos",
        ],
    )

    assert "enterprise" in guarded.response_text.lower()
    assert "tu nombre es fabian c villegas" not in guarded.response_text.lower()
    assert "compárteme tu nombre completo" not in guarded.response_text.lower()


def test_planning_guardrail_does_not_append_pending_name_prompt_during_qualification():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="En la área de rescate jurídico",
        email=None,
        metadata={
            "name_validation": {
                "status": "needs_confirmation",
                "candidate_name": "En La Área De Rescate Jurídico",
                "normalized_name": "En La Área De Rescate Jurídico",
                "source": "provider",
            }
        },
    )
    result = PlanningResult(
        intent="qualify_lead",
        confidence=0.84,
        response_text=(
            "Perfecto, 50 procesos ya me da una buena idea. "
            "¿Hoy los siguen manualmente o usan algún sistema de alertas?"
        ),
        actions=[],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "50",
        contact,
        ["Gracias por comunicarte con Rescate jurídico. ¿Cómo podemos ayudarte?"],
    )

    assert "rescate jurídico" not in guarded.response_text.lower()
    assert "nombre" not in guarded.response_text.lower()
    assert guarded.response_text == result.response_text


def test_planning_guardrail_strips_llm_repeated_pending_name_prompt_after_it_was_asked():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Inicial",
        email="lead@example.com",
        metadata={
            "name_validation": {
                "status": "needs_confirmation",
                "candidate_name": "Fabian C Villegas",
                "normalized_name": "Fabian C Villegas",
                "source": "provider_llm",
            }
        },
    )
    result = PlanningResult(
        intent="consult_price",
        confidence=0.8,
        response_text=(
            "El plan Enterprise se revisa con ventas según volumen. "
            "Antes de seguir, ¿tu nombre es Fabian C Villegas? Si no, compárteme tu nombre completo."
        ),
        actions=[],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "Quiero saber cómo funciona y cuál es el costo",
        contact,
        [
            "Buenos días, ¿cómo puedo ayudarte hoy con Wabog? Antes de seguir, ¿tu nombre es Fabian C Villegas? Si no, compárteme tu nombre completo.",
            "Si Fabian",
            "Tengo 99 casos",
        ],
    )

    assert guarded.response_text == "El plan Enterprise se revisa con ventas según volumen."


def test_repair_actions_allows_replacing_a_previously_confirmed_name():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Fabian Cuero",
        email="fabian@example.com",
        metadata={"name_validation": {"status": "trusted", "source": "user_message"}},
    )
    result = PlanningResult(
        intent="name_correction",
        confidence=0.84,
        response_text="Gracias por la aclaración.",
        actions=[],
    )

    repaired = planner._repair_actions(  # noqa: SLF001
        result,
        "Mi apellido completo también es Villegas",
        contact,
        ["Tu nombre actual es Fabian Cuero."],
        NameConfirmationDecision(
            status="provided_new_name",
            confidence=0.99,
            resolved_name="Fabian Cuero Villegas",
        ),
    )

    update_action = next(action for action in repaired.actions if action.type == ActionType.UPDATE_CONTACT_FIELDS)
    assert update_action.args["fields"] == {"full_name": "Fabian Cuero Villegas"}


def test_planning_guardrail_creates_meeting_after_contextual_name_confirmation():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Juan Perez",
        email="fabiancuerov@gmail.com",
        stage="Primer contacto",
        followup_summary="Coordinar fecha y hora para demo comercial.",
        metadata={
            "name_validation": {
                "status": "needs_confirmation",
                "candidate_name": "Fabian C Villegas",
                "normalized_name": "Fabian C Villegas",
                "source": "provider_llm",
            }
        },
    )
    result = PlanningResult(
        intent="confirm_demo",
        confidence=0.79,
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

    repaired = planner._repair_actions(  # noqa: SLF001
        result,
        "Ese es mi nombre agendala.",
        contact,
        [
            "Perfecto, Juan. Para agendar la demo necesito confirmar la fecha y hora que te convienen.",
            "mañana a las 10am mi correo es fabiancuerov@gmail.com",
            "Perfecto. Antes de agendarla, ¿tu nombre es Fabian C Villegas? Si no, compárteme tu nombre completo.",
        ],
        NameConfirmationDecision(
            status="confirmed_candidate_name",
            confidence=0.97,
            resolved_name="Fabian C Villegas",
        ),
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        repaired,
        "Ese es mi nombre agendala.",
        contact,
        [
            "Perfecto, Juan. Para agendar la demo necesito confirmar la fecha y hora que te convienen.",
            "mañana a las 10am mi correo es fabiancuerov@gmail.com",
            "Perfecto. Antes de agendarla, ¿tu nombre es Fabian C Villegas? Si no, compárteme tu nombre completo.",
        ],
        NameConfirmationDecision(
            status="confirmed_candidate_name",
            confidence=0.97,
            resolved_name="Fabian C Villegas",
        ),
    )

    action_types = [action.type for action in guarded.actions]
    assert ActionType.UPDATE_CONTACT_FIELDS in action_types
    assert ActionType.CREATE_MEETING in action_types


def test_planning_guardrail_does_not_retry_booking_on_unrelated_price_turn():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Laura Gómez",
        email="laura@example.com",
        stage="Primer contacto",
    )
    result = PlanningResult(
        intent="consulta precio",
        confidence=0.82,
        response_text="El plan Enterprise se revisa con ventas según volumen y operación.",
        actions=[],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "y eso cuanto vale",
        contact,
        [
            "Perfecto, agendemos la demo para mañana a las 10 am. Antes de confirmar, necesito que me confirmes tu nombre completo y correo electrónico.",
            "mi nombre es Laura Gómez",
            "mi correo es laura@example.com",
        ],
    )

    assert all(action.type != ActionType.CREATE_MEETING for action in guarded.actions)
    assert "enterprise" in guarded.response_text.lower()


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


def test_planning_guardrail_synthesizes_trial_stage_when_llm_omits_it():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Lead Demo",
        email=None,
        stage=None,
    )
    result = PlanningResult(
        intent="user wants to try Wabog",
        confidence=0.72,
        response_text="Cuéntame un poco más y te paso el link.",
        actions=[],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "quiero probarlo",
        contact,
        ["manejamos 5 procesos al mes", "cuanto cuesta"],
        semantic_guardrail=SemanticGuardrailDecision(
            inferred_stage="Prueba / Trial",
            stage_confidence=0.91,
        ),
    )

    stage_action = next(action for action in guarded.actions if action.type == ActionType.UPDATE_STAGE)
    assert stage_action.args["stage"] == "Prueba / Trial"


def test_planning_guardrail_normalizes_trial_response_to_public_wabog_link():
    planner = build_planner()
    result = PlanningResult(
        intent="user wants to try Wabog",
        confidence=0.78,
        response_text=(
            "Para probar Wabog, puedes iniciar una prueba gratuita directamente en nuestra plataforma: "
            "https://wabog.com/signup"
        ),
        actions=[],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "como lo pruebo",
        None,
        ["manejo 5 procesos al mes"],
        semantic_guardrail=SemanticGuardrailDecision(
            should_offer_trial_response=True,
            trial_confidence=0.93,
        ),
    )

    assert "https://wabog.com" in guarded.response_text
    assert "https://app.wabog.com" in guarded.response_text
    assert "https://wabog.com/signup" not in guarded.response_text
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


def test_planning_guardrail_synthesizes_meeting_when_slot_is_concrete_and_contact_is_complete():
    planner = build_planner()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Laura Gomez",
        email="laura@example.com",
        stage="Primer contacto",
    )
    result = PlanningResult(
        intent="schedule_demo",
        confidence=0.9,
        response_text="Perfecto, Laura. Procederé a crear la invitación para la demo.",
        actions=[],
    )

    guarded = planner._apply_planning_guardrail(  # noqa: SLF001
        result,
        "mañana a las 3 pm me sirve",
        contact,
        [
            "si me interesa ver una demo",
            "mi nombre es Laura Gomez",
            "mi correo es laura@example.com",
        ],
    )

    meeting_actions = [action for action in guarded.actions if action.type == ActionType.CREATE_MEETING]
    assert len(meeting_actions) == 1
    assert meeting_actions[0].args["title"] == "Demo Wabog - Laura Gomez"
    assert "T15:00:00" in meeting_actions[0].args["start_iso"]


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


def test_plan_with_rules_creates_contact_update_action_for_email_only():
    planner = build_planner()

    result = planner._plan_with_rules(  # noqa: SLF001
        "Hola, mi correo es juan@example.com",
        None,
    )

    update_action = next(action for action in result.actions if action.type == ActionType.UPDATE_CONTACT_FIELDS)
    assert update_action.args["fields"] == {
        "email": "juan@example.com",
    }
    assert "correo" in result.response_text.lower()


@pytest.mark.asyncio
async def test_contextual_special_case_answers_name_from_current_contact():
    planner = build_planner()
    planner._semantic_guardrail_llm = _FakeSemanticGuardrailLLM(  # noqa: SLF001
        special_case_intent="ask_name",
        special_case_confidence=0.97,
    )
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
async def test_contextual_special_case_asks_for_name_when_contact_name_is_missing_or_placeholder():
    planner = build_planner()
    planner._semantic_guardrail_llm = _FakeSemanticGuardrailLLM(  # noqa: SLF001
        special_case_intent="ask_name",
        special_case_confidence=0.97,
    )
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="~",
    )

    result = await planner.plan("como me llamo", contact, [], [])

    assert result.intent == "ask_name"
    assert result.actions == []
    assert "todavía no tengo un nombre confiable registrado" in result.response_text.lower()


@pytest.mark.asyncio
async def test_contextual_special_case_does_not_ask_to_confirm_candidate_name_when_it_is_not_trusted_yet():
    planner = build_planner()
    planner._semantic_guardrail_llm = _FakeSemanticGuardrailLLM(  # noqa: SLF001
        special_case_intent="ask_name",
        special_case_confidence=0.97,
    )
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
    assert "juan" not in result.response_text.lower()
    assert "confirmarlo" not in result.response_text.lower()
    assert "nombre confiable" in result.response_text.lower()


@pytest.mark.asyncio
async def test_contextual_special_case_does_not_treat_phone_number_as_contact_name():
    planner = build_planner()
    planner._semantic_guardrail_llm = _FakeSemanticGuardrailLLM(  # noqa: SLF001
        special_case_intent="ask_name",
        special_case_confidence=0.97,
    )
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="3150000000",
    )

    result = await planner.plan("como me llamo", contact, [], [])

    assert result.intent == "ask_name"
    assert "todavía no tengo un nombre confiable registrado" in result.response_text.lower()


@pytest.mark.asyncio
async def test_contextual_special_case_distinguishes_contact_source_questions():
    planner = build_planner()
    planner._semantic_guardrail_llm = _FakeSemanticGuardrailLLM(  # noqa: SLF001
        special_case_intent="ask_contact_source",
        special_case_confidence=0.96,
    )

    result = await planner.plan("de donde sacaron mi numero?", None, [], [])

    assert result.intent == "ask_contact_source"
    assert result.actions == []
    assert "te contactamos" in result.response_text.lower()


def test_append_calendar_confirmation_removes_unsupported_reminder_promise():
    planner = build_planner()

    response_text = planner._append_calendar_confirmation(  # noqa: SLF001
        "Perfecto, Carlos. Ya tienes la demo agendada para el martes 21 de abril a las 9:00 AM. Te enviaré un recordatorio antes de la fecha.",
        {"start_iso": "2026-04-21T09:00:00-05:00"},
    )

    assert "recordatorio" not in response_text.lower()
    assert "veo en calendario una demo futura" in response_text.lower()


def test_missing_meeting_fields_response_requests_only_email_before_booking():
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

    response = planner._build_missing_meeting_fields_response(["email"], contact)  # noqa: SLF001

    assert "correo" in response.lower()
    assert "nombre" not in response.lower()


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


class _FakeSemanticGuardrailLLM:
    def __init__(
        self,
        *,
        special_case_intent: str = "none",
        special_case_confidence: float = 0.0,
        inferred_stage: str | None = None,
        stage_confidence: float = 0.0,
        should_complete_followup: bool = False,
        complete_followup_confidence: float = 0.0,
        should_handoff_human: bool = False,
        handoff_confidence: float = 0.0,
        should_offer_trial_response: bool = False,
        trial_confidence: float = 0.0,
    ) -> None:
        self.payload = {
            "special_case_intent": special_case_intent,
            "special_case_confidence": special_case_confidence,
            "inferred_stage": inferred_stage,
            "stage_confidence": stage_confidence,
            "should_complete_followup": should_complete_followup,
            "complete_followup_confidence": complete_followup_confidence,
            "should_handoff_human": should_handoff_human,
            "handoff_confidence": handoff_confidence,
            "should_offer_trial_response": should_offer_trial_response,
            "trial_confidence": trial_confidence,
        }

    async def ainvoke(self, prompt: str):
        payload = dict(self.payload)

        class _Response:
            def model_dump(self_nonlocal):  # noqa: ANN001
                return payload

        return _Response()


@pytest.mark.asyncio
async def test_planner_loads_selected_knowledge_section_into_prompt():
    planner = AgentPlanner(Settings(OPENAI_API_KEY=""))
    planner._llm = _FakeStructuredOutput()  # noqa: SLF001
    planner._knowledge_selector_llm = _FakeKnowledgeSelector()  # noqa: SLF001

    result = await planner.plan("y eso cuanto vale", None, [], [])
    assert "Wabog Pricing" in planner._llm.last_prompt  # noqa: SLF001
    assert any(item.section == "wabog_pricing" for item in result.knowledge_lookups)
