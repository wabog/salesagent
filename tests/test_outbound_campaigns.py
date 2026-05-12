from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from sales_agent.adapters.memory_sql import SqlAlchemyMemoryStore
from sales_agent.core.db import OutboundRecipientRecord, build_engine, build_session_factory, init_db
from sales_agent.domain.models import CRMContact, OutboundTemplateMessage
from sales_agent.services.name_validation import NameCandidateAssessment
from sales_agent.services.outbound_campaigns import (
    AfterSendConfig,
    CRMFilterCondition,
    CRMFilterConfig,
    OutboundCampaignConfig,
    OutboundCampaignService,
    ScheduleConfig,
    TemplateConfig,
    TemplateParameterConfig,
)


class FakeCRM:
    def __init__(self, contacts: list[CRMContact]) -> None:
        self.contacts_by_phone = {contact.phone_number: contact for contact in contacts}
        self.conditions: list[dict[str, str]] | None = None
        self.notes: list[str] = []

    async def find_contact_by_phone(self, phone_number: str) -> CRMContact | None:
        return self.contacts_by_phone.get(phone_number)

    async def list_contacts_by_property_filters(self, conditions: list[dict[str, str]]) -> list[CRMContact]:
        self.conditions = conditions
        return list(self.contacts_by_phone.values())

    async def change_stage(self, external_id: str, stage: str) -> CRMContact:
        for phone, contact in self.contacts_by_phone.items():
            if contact.external_id == external_id:
                updated = contact.model_copy(update={"stage": stage})
                self.contacts_by_phone[phone] = updated
                return updated
        raise KeyError(external_id)

    async def append_note(self, external_id: str, note: str) -> CRMContact:
        self.notes.append(note)
        for contact in self.contacts_by_phone.values():
            if contact.external_id == external_id:
                return contact.model_copy(update={"notes": [*contact.notes, note]})
        raise KeyError(external_id)


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[OutboundTemplateMessage] = []

    async def send_template(self, message: OutboundTemplateMessage) -> dict:
        self.sent.append(message)
        return {"messages": [{"id": "wamid.test"}]}


class FakeNameValidator:
    def __init__(self, assessment: NameCandidateAssessment) -> None:
        self.assessment = assessment
        self.candidates: list[str | None] = []

    async def assess_outbound_greeting_name(self, candidate: str | None) -> NameCandidateAssessment:
        self.candidates.append(candidate)
        return self.assessment


def _config() -> OutboundCampaignConfig:
    return OutboundCampaignConfig(
        id="test_campaign",
        name="Test campaign",
        crm_filter=CRMFilterConfig(
            conditions=[
                CRMFilterCondition(property="Fuente", type="select", equals="facebook"),
                CRMFilterCondition(property="Etapa", type="status", equals="Nuevo Lead"),
            ]
        ),
        template=TemplateConfig(
            name="wabog_gentle_reactivation",
            language="es",
            body_text=(
                "Hola {{1}}, soy {{2}} de Wabog. Queria saber si aun te interesa conocer la plataforma. "
                "Si quieres, te cuento por aqui."
            ),
            parameters=[
                TemplateParameterConfig(type="lead_name", fallback="buen día"),
                TemplateParameterConfig(type="static", value="Fabian"),
            ],
        ),
        after_send=AfterSendConfig(
            stage="Primer contacto",
            note="Primer contacto outbound enviado por campaña {campaign_id} con {template_name}.",
        ),
        schedule=ScheduleConfig(
            mode="once_at",
            timezone="America/Bogota",
            send_at="2026-05-13 09:00",
            max_per_run=10,
        ),
    )


@pytest.mark.asyncio
async def test_outbound_campaign_sends_template_updates_crm_and_memory(tmp_path):
    engine = build_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await init_db(engine)
    session_factory = build_session_factory(engine)
    memory = SqlAlchemyMemoryStore(session_factory)
    crm = FakeCRM(
        [
            CRMContact(
                external_id="lead-1",
                phone_number="+573001112233",
                full_name="CRIMENYCARCEL",
                stage="Nuevo Lead",
            )
        ]
    )
    channel = FakeChannel()
    service = OutboundCampaignService(
        session_factory=session_factory,
        crm_adapter=crm,
        memory_store=memory,
        channel_adapter=channel,
    )
    config = _config()
    now = datetime(2026, 5, 13, 14, 0, tzinfo=UTC)

    seed = await service.seed_campaign(config, scheduled_at=now)
    run = await service.run_due(config, now=now)

    assert seed.scheduled_recipients == 1
    assert run.sent == 1
    assert channel.sent[0].body_parameters == [
        {"type": "text", "text": "buen día"},
        {"type": "text", "text": "Fabian"},
    ]
    assert channel.sent[0].rendered_text.startswith("Hola buen día, soy Fabian de Wabog.")
    assert crm.contacts_by_phone["+573001112233"].stage == "Primer contacto"
    assert crm.notes == [
        "Primer contacto outbound enviado por campaña test_campaign con wabog_gentle_reactivation."
    ]

    recent = await memory.get_recent_messages_by_phone("+573001112233", limit=5)
    assert [message.text for message in recent] == [channel.sent[0].rendered_text]
    assert recent[0].metadata["campaign_id"] == "test_campaign"

    async with session_factory() as session:
        result = await session.execute(select(OutboundRecipientRecord))
        recipient = result.scalar_one()
    assert recipient.status == "sent"
    assert recipient.provider_message_id == "wamid.test"
    await engine.dispose()


@pytest.mark.asyncio
async def test_outbound_campaign_uses_ai_validated_lead_name(tmp_path):
    engine = build_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await init_db(engine)
    session_factory = build_session_factory(engine)
    memory = SqlAlchemyMemoryStore(session_factory)
    crm = FakeCRM(
        [
            CRMContact(
                external_id="lead-1",
                phone_number="+573001112233",
                full_name="camila perez",
                stage="Nuevo Lead",
            )
        ]
    )
    channel = FakeChannel()
    name_validator = FakeNameValidator(
        NameCandidateAssessment(
            status="trusted",
            confidence=0.96,
            normalized_name="Camila Perez",
            candidate_name="Camila Perez",
            source="outbound_greeting_llm",
        )
    )
    service = OutboundCampaignService(
        session_factory=session_factory,
        crm_adapter=crm,
        memory_store=memory,
        channel_adapter=channel,
        name_validator=name_validator,
    )
    config = _config()
    now = datetime(2026, 5, 13, 14, 0, tzinfo=UTC)

    await service.seed_campaign(config, scheduled_at=now)
    await service.run_due(config, now=now)

    assert name_validator.candidates == ["camila perez"]
    assert channel.sent[0].body_parameters[0] == {"type": "text", "text": "Camila Perez"}
    assert channel.sent[0].rendered_text.startswith("Hola Camila Perez, soy Fabian de Wabog.")
    await engine.dispose()


@pytest.mark.asyncio
async def test_outbound_campaign_falls_back_when_ai_rejects_name(tmp_path):
    engine = build_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await init_db(engine)
    session_factory = build_session_factory(engine)
    memory = SqlAlchemyMemoryStore(session_factory)
    crm = FakeCRM(
        [
            CRMContact(
                external_id="lead-1",
                phone_number="+573001112233",
                full_name="CRIMENYCARCEL",
                stage="Nuevo Lead",
            )
        ]
    )
    channel = FakeChannel()
    name_validator = FakeNameValidator(
        NameCandidateAssessment(
            status="rejected",
            confidence=0.99,
            normalized_name=None,
            candidate_name="Crimenycarcel",
            source="outbound_greeting_llm",
        )
    )
    service = OutboundCampaignService(
        session_factory=session_factory,
        crm_adapter=crm,
        memory_store=memory,
        channel_adapter=channel,
        name_validator=name_validator,
    )
    config = _config()
    now = datetime(2026, 5, 13, 14, 0, tzinfo=UTC)

    await service.seed_campaign(config, scheduled_at=now)
    await service.run_due(config, now=now)

    assert name_validator.candidates == ["CRIMENYCARCEL"]
    assert channel.sent[0].body_parameters[0] == {"type": "text", "text": "buen día"}
    assert channel.sent[0].rendered_text.startswith("Hola buen día, soy Fabian de Wabog.")
    await engine.dispose()


@pytest.mark.asyncio
async def test_scheduler_tick_seeds_recurring_campaign_once_per_due_slot(tmp_path):
    engine = build_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await init_db(engine)
    session_factory = build_session_factory(engine)
    memory = SqlAlchemyMemoryStore(session_factory)
    crm = FakeCRM(
        [
            CRMContact(
                external_id="lead-1",
                phone_number="+573001112233",
                full_name=None,
                stage="Nuevo Lead",
            )
        ]
    )
    channel = FakeChannel()
    service = OutboundCampaignService(
        session_factory=session_factory,
        crm_adapter=crm,
        memory_store=memory,
        channel_adapter=channel,
    )
    config = _config().model_copy(
        update={
            "schedule": ScheduleConfig(
                mode="recurring",
                timezone="America/Bogota",
                days=["wed"],
                times=["09:00"],
                max_per_run=10,
            )
        }
    )
    now = datetime(2026, 5, 13, 14, 0, tzinfo=UTC)

    first_seed, first_run = await service.scheduler_tick(config, now=now)
    second_seed, second_run = await service.scheduler_tick(config, now=now)

    assert first_seed is not None
    assert first_seed.scheduled_recipients == 1
    assert first_run.sent == 1
    assert second_seed is None
    assert second_run.attempted == 0
    assert len(channel.sent) == 1
    await engine.dispose()
