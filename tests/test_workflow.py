from datetime import date, datetime, timedelta, timezone

import pytest

from sales_agent.adapters.channel import ConsoleChannelAdapter
from sales_agent.adapters.crm_memory import InMemoryCRMAdapter
from sales_agent.adapters.memory_sql import SqlAlchemyMemoryStore
from sales_agent.core.config import Settings
from sales_agent.core.db import build_engine, build_session_factory, init_db
from sales_agent.domain.models import CRMContact, InboundMessage
from sales_agent.domain.models import ActionType, PlanningResult, ProposedAction
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
        planner=AgentPlanner(Settings(OPENAI_API_KEY="")),
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
    contact = await crm.find_contact_by_phone("+573001112233")
    recent = await memory.get_recent_messages("conv-1", limit=10)

    assert result.duplicate is False
    assert result.intent == "update_stage"
    assert contact is not None
    assert contact.stage == "Demo"
    assert len(recent) == 2
    assert "Ya registré este número como lead en el CRM." not in result.response_text
    assert result.response_text.startswith("Actualicé la etapa del lead a Demo.")
    assert recent[-1].text == result.response_text
    assert all(tool_result.action.type != "create_contact" for tool_result in result.tool_results)


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


@pytest.mark.asyncio
async def test_existing_lead_is_reused_without_creation_side_effect():
    workflow, crm, _ = await build_workflow()
    existing = await crm.create_contact("573009998877", "Lead Existente")

    event = InboundMessage(
        message_id="existing-1",
        conversation_id="conv-existing",
        phone_number="573009998877",
        text="hola",
        timestamp=datetime.now(timezone.utc),
        raw_payload={},
    )

    result = await workflow.run(event)
    contact = await crm.find_contact_by_phone("+573009998877")

    assert result.duplicate is False
    assert contact is not None
    assert contact.external_id == existing.external_id
    assert "Ya registré este número como lead en el CRM." not in result.response_text


@pytest.mark.asyncio
async def test_workflow_does_not_reuse_stale_shadow_when_crm_contact_is_missing():
    workflow, crm, memory = await build_workflow()
    await memory.remember_contact(
        CRMContact(
            external_id="stale-shadow-id",
            phone_number="573001230000",
            full_name="Lead Stale",
            stage="Demo agendada",
        )
    )

    event = InboundMessage(
        message_id="stale-shadow-1",
        conversation_id="conv-shadow",
        phone_number="573001230000",
        text="hola",
        timestamp=datetime.now(timezone.utc),
        raw_payload={},
        contact_name="Lead Nuevo",
    )

    result = await workflow.run(event)
    contact = await crm.find_contact_by_phone("+573001230000")

    assert result.duplicate is False
    assert contact is not None
    assert contact.external_id != "stale-shadow-id"
    assert result.contact is not None
    assert result.contact.external_id == contact.external_id


@pytest.mark.asyncio
async def test_workflow_updates_contact_fields_from_explicit_user_data():
    workflow, crm, _ = await build_workflow()
    existing = await crm.create_contact("573001119900", "Lead Inicial")

    event = InboundMessage(
        message_id="contact-update-1",
        conversation_id="conv-contact-update",
        phone_number="573001119900",
        text="mi nombre es juan perez y mi correo es juan@example.com",
        timestamp=datetime.now(timezone.utc),
        raw_payload={},
    )

    result = await workflow.run(event)
    contact = await crm.find_contact_by_phone("+573001119900")

    assert existing.external_id == contact.external_id
    assert contact is not None
    assert contact.full_name == "Juan Perez"
    assert contact.email == "juan@example.com"
    update_action = next(action for action in result.tool_results if action.action.type == ActionType.UPDATE_CONTACT_FIELDS)
    assert update_action.success is True
    assert "correo" in result.response_text.lower()


@pytest.mark.asyncio
async def test_workflow_persists_followup_summary_and_due_date():
    workflow, crm, _ = await build_workflow()
    await crm.create_contact("573001119901", "Lead Followup")

    event = InboundMessage(
        message_id="followup-1",
        conversation_id="conv-followup",
        phone_number="573001119901",
        text="recordarme mañana enviar la propuesta",
        timestamp=datetime.now(timezone.utc),
        raw_payload={},
    )

    result = await workflow.run(event)
    contact = await crm.find_contact_by_phone("+573001119901")

    assert contact is not None
    assert contact.followup_summary == "recordarme mañana enviar la propuesta"
    assert contact.followup_due_date == date.today() + timedelta(days=1)
    assert contact.notes[-1] == (
        f"{date.today().isoformat()} - Seguimiento creado: recordarme mañana enviar la propuesta. "
        f"Vence: {(date.today() + timedelta(days=1)).isoformat()}."
    )
    followup_action = next(action for action in result.tool_results if action.action.type == ActionType.CREATE_FOLLOWUP)
    assert followup_action.success is True
    assert followup_action.payload["due_date"] == (date.today() + timedelta(days=1)).isoformat()


@pytest.mark.asyncio
async def test_workflow_completes_followup_and_logs_timeline_note():
    workflow, crm, _ = await build_workflow()
    created = await crm.create_contact("573001119902", "Lead Followup")
    await crm.create_followup(created.external_id, "Enviar propuesta comercial.", due_date="2026-04-16")

    event = InboundMessage(
        message_id="followup-complete-1",
        conversation_id="conv-followup-complete",
        phone_number="573001119902",
        text="listo, ya envié la propuesta",
        timestamp=datetime.now(timezone.utc),
        raw_payload={},
    )

    result = await workflow.run(event)
    contact = await crm.find_contact_by_phone("+573001119902")

    assert contact is not None
    assert contact.followup_summary is None
    assert contact.followup_due_date is None
    completion_action = next(action for action in result.tool_results if action.action.type == ActionType.COMPLETE_FOLLOWUP)
    assert completion_action.success is True
    assert contact.notes[-1] == (
        f"{date.today().isoformat()} - Seguimiento completado: Enviar propuesta comercial. "
        "Resultado: listo, ya envié la propuesta."
    )


@pytest.mark.asyncio
async def test_workflow_repairs_missing_stage_during_tool_execution():
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    session_factory = build_session_factory(engine)
    memory = SqlAlchemyMemoryStore(session_factory)
    crm = InMemoryCRMAdapter()

    class StubPlanner:
        async def plan(self, text, contact, recent_messages, semantic_memories, prompt_mode=None):  # noqa: ANN001
            return PlanningResult(
                intent="demo_confirmation",
                confidence=0.8,
                response_text="Perfecto, agendemos.",
                actions=[
                    ProposedAction(
                        type=ActionType.UPDATE_STAGE,
                        reason="El lead confirmó que quiere avanzar.",
                        args={},
                    )
                ],
            )

        def _infer_stage_from_text(self, text, current_stage, recent_messages):  # noqa: ANN001
            return "Demo agendada"

    workflow = SalesAgentWorkflow(
        crm_adapter=crm,
        memory_store=memory,
        channel_adapter=ConsoleChannelAdapter(),
        planner=StubPlanner(),
        policy=ToolExecutionPolicy(),
    )

    event = InboundMessage(
        message_id="repair-stage-1",
        conversation_id="conv-repair-stage",
        phone_number="573001240000",
        text="si claro",
        timestamp=datetime.now(timezone.utc),
        raw_payload={},
    )

    result = await workflow.run(event)
    contact = await crm.find_contact_by_phone("+573001240000")

    assert contact is not None
    assert contact.stage == "Demo agendada"
    assert result.tool_results[0].success is True
