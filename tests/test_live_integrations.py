import pytest

from sales_agent.adapters.crm_notion import NotionCRMAdapter
from sales_agent.core.config import Settings
from sales_agent.services.planner import AgentPlanner


pytestmark = pytest.mark.integration


def get_live_settings() -> Settings:
    settings = Settings()
    if settings.crm_backend != "notion":
        pytest.skip("Live Notion integration test requires CRM_BACKEND=notion.")
    if not settings.notion_api_key or not settings.notion_data_source_id:
        pytest.skip("Live Notion integration test requires Notion credentials.")
    if not settings.openai_api_key:
        pytest.skip("Live OpenAI integration test requires OPENAI_API_KEY.")
    return settings


@pytest.mark.asyncio
async def test_notion_live_query_smoke():
    settings = get_live_settings()
    adapter = NotionCRMAdapter(settings)

    data = await adapter._request(  # noqa: SLF001
        "POST",
        f"/data_sources/{settings.notion_data_source_id}/query",
        json={"page_size": 1},
    )

    assert "results" in data
    assert isinstance(data["results"], list)


@pytest.mark.asyncio
async def test_openai_live_planner_smoke():
    settings = get_live_settings()
    planner = AgentPlanner(settings)
    if planner._llm is None:  # noqa: SLF001
        pytest.skip("Planner did not initialize a live OpenAI client.")

    result = await planner._plan_with_llm(  # noqa: SLF001
        text="El usuario dice hola y quiere una respuesta breve.",
        contact=None,
        recent_messages=[],
        semantic_memories=[],
    )

    assert result.intent
    assert result.response_text
