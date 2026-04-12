from sales_agent.core.config import Settings
from sales_agent.domain.models import CRMContact, PlanningResult, ProposedAction
from sales_agent.domain.models import ActionType
from sales_agent.services.planner import AgentPlanner


def build_planner() -> AgentPlanner:
    return AgentPlanner(Settings(OPENAI_API_KEY=""))


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
    assert stage_action.args["stage"] == "Demo agendada"


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
