from datetime import datetime, timezone

import pytest

from sales_agent.adapters.channel import ConsoleChannelAdapter
from sales_agent.adapters.crm_memory import InMemoryCRMAdapter
from sales_agent.adapters.memory_sql import SqlAlchemyMemoryStore
from sales_agent.core.config import Settings
from sales_agent.core.db import build_engine, build_session_factory, init_db
from sales_agent.domain.models import InboundMessage
from sales_agent.graph.workflow import SalesAgentWorkflow
from sales_agent.services.planner import AgentPlanner
from sales_agent.services.policy import ToolExecutionPolicy


async def build_workflow():
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    session_factory = build_session_factory(engine)
    memory = SqlAlchemyMemoryStore(session_factory)
    crm = InMemoryCRMAdapter()
    workflow = SalesAgentWorkflow(
        crm_adapter=crm,
        memory_store=memory,
        channel_adapter=ConsoleChannelAdapter(),
        planner=AgentPlanner(Settings()),
        policy=ToolExecutionPolicy(),
    )
    return workflow, crm, memory


@pytest.mark.asyncio
async def test_workflow_creates_contact_and_updates_stage():
    workflow, crm, memory = await build_workflow()
    event = InboundMessage(
        message_id="msg-1",
        conversation_id="conv-1",
        phone_number="573001112233",
        text="por favor cambia etapa a demo",
        timestamp=datetime.now(timezone.utc),
        raw_payload={},
    )

    result = await workflow.run(event)
    contact = await crm.find_contact_by_phone("573001112233")
    recent = await memory.get_recent_messages("conv-1", limit=10)

    assert result.duplicate is False
    assert result.intent == "update_stage"
    assert contact is not None
    assert contact.stage == "Demo"
    assert len(recent) == 2
    assert recent[-1].text == result.response_text


@pytest.mark.asyncio
async def test_workflow_is_idempotent_on_duplicate_message():
    workflow, _, _ = await build_workflow()
    event = InboundMessage(
        message_id="dup-1",
        conversation_id="conv-dup",
        phone_number="573001112244",
        text="nota: lead pidió llamada mañana",
        timestamp=datetime.now(timezone.utc),
        raw_payload={},
    )

    first = await workflow.run(event)
    second = await workflow.run(event)

    assert first.duplicate is False
    assert second.duplicate is True
    assert second.response_text == ""
