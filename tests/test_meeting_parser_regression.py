from sales_agent.core.config import Settings
from sales_agent.services.planner import AgentPlanner


def build_planner() -> AgentPlanner:
    return AgentPlanner(Settings(OPENAI_API_KEY=""))


def test_extract_requested_meeting_start_does_not_cross_match_old_today_with_case_volume():
    planner = build_planner()

    value = planner._extract_requested_meeting_start(  # noqa: SLF001
        "si me interesa ver una demo",
        [
            "Hola, ¿cómo puedo ayudarte hoy con la gestión de tus procesos judiciales? ¿Cuántos casos activos manejas actualmente?",
            "manejamos como 80 procesos",
        ],
    )

    assert value is None
