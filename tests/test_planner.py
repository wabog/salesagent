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
    )

    note_actions = [action for action in enforced.actions if action.type == ActionType.APPEND_NOTE]
    assert len(note_actions) == 1
