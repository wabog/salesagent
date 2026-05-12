from __future__ import annotations

import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from sales_agent.core.db import OutboundCampaignRecord, OutboundRecipientRecord
from sales_agent.domain.models import CRMContact, ConversationMessage, Direction, OutboundTemplateMessage
from sales_agent.services.name_validation import NameCandidateAssessment, is_specific_person_name, normalize_person_name


class QueryableCRMAdapter(Protocol):
    async def find_contact_by_phone(self, phone_number: str) -> CRMContact | None: ...

    async def list_contacts_by_property_filters(self, conditions: list[dict[str, str]]) -> list[CRMContact]: ...

    async def change_stage(self, external_id: str, stage: str) -> CRMContact: ...

    async def append_note(self, external_id: str, note: str) -> CRMContact: ...


class OutboundMemoryStore(Protocol):
    async def append_message(self, message: ConversationMessage) -> None: ...

    async def remember_contact(self, contact: CRMContact) -> None: ...


class TemplateChannelAdapter(Protocol):
    async def send_template(self, message: OutboundTemplateMessage) -> dict: ...


class OutboundNameValidator(Protocol):
    async def assess_outbound_greeting_name(self, candidate: str | None) -> NameCandidateAssessment: ...


class CRMFilterCondition(BaseModel):
    property: str
    equals: str
    type: Literal["select", "status", "rich_text", "title", "email", "phone_number"] = "select"


class CRMFilterConfig(BaseModel):
    conditions: list[CRMFilterCondition] = Field(default_factory=list)


class TemplateParameterConfig(BaseModel):
    type: Literal["lead_name", "static"]
    value: str | None = None
    fallback: str | None = None
    name: str | None = None


class TemplateConfig(BaseModel):
    name: str
    language: str
    body_text: str
    parameter_format: Literal["POSITIONAL", "NAMED"] = "POSITIONAL"
    phone_number_id: str | None = None
    parameters: list[TemplateParameterConfig] = Field(default_factory=list)


class AfterSendConfig(BaseModel):
    stage: str | None = None
    note: str | None = None


class ScheduleConfig(BaseModel):
    mode: Literal["once_at", "recurring"]
    timezone: str = "America/Bogota"
    send_at: str | None = None
    days: list[str] = Field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    times: list[str] = Field(default_factory=list)
    max_per_run: int = 25
    jitter_seconds: int = 0


class OutboundCampaignConfig(BaseModel):
    id: str
    name: str
    crm_filter: CRMFilterConfig
    template: TemplateConfig
    after_send: AfterSendConfig = Field(default_factory=AfterSendConfig)
    schedule: ScheduleConfig
    status: Literal["active", "paused"] = "active"


class OutboundSeedResult(BaseModel):
    campaign_id: str
    matched_contacts: int
    scheduled_recipients: int
    skipped_existing: int
    scheduled_at: datetime


class OutboundRunResult(BaseModel):
    campaign_id: str
    attempted: int
    sent: int
    failed: int
    skipped: int


_DAY_INDEX = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def load_campaign_config(path: str | Path) -> OutboundCampaignConfig:
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        data = json.loads(raw)
    elif source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("YAML campaign files require PyYAML. Use JSON or install PyYAML.") from exc
        data = yaml.safe_load(raw)
    else:
        raise ValueError("Campaign config must be a .json, .yaml, or .yml file.")
    return OutboundCampaignConfig.model_validate(data)


def load_campaign_configs(paths: list[str | Path]) -> list[OutboundCampaignConfig]:
    configs: list[OutboundCampaignConfig] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in {".json", ".yaml", ".yml"}:
                    configs.append(load_campaign_config(child))
            continue
        configs.append(load_campaign_config(path))
    return configs


class OutboundCampaignService:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        crm_adapter: QueryableCRMAdapter,
        memory_store: OutboundMemoryStore,
        channel_adapter: TemplateChannelAdapter,
        name_validator: OutboundNameValidator | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._crm_adapter = crm_adapter
        self._memory_store = memory_store
        self._channel_adapter = channel_adapter
        self._name_validator = name_validator

    async def ensure_campaign(self, config: OutboundCampaignConfig) -> None:
        async with self._session_factory() as session:
            record = await session.get(OutboundCampaignRecord, config.id)
            payload = config.model_dump(mode="json")
            if record is None:
                session.add(
                    OutboundCampaignRecord(
                        campaign_id=config.id,
                        name=config.name,
                        status=config.status,
                        config_json=payload,
                    )
                )
            else:
                record.name = config.name
                record.status = config.status
                record.config_json = payload
            await session.commit()

    async def seed_campaign(
        self,
        config: OutboundCampaignConfig,
        *,
        scheduled_at: datetime | None = None,
    ) -> OutboundSeedResult:
        await self.ensure_campaign(config)
        target_scheduled_at = _ensure_utc(scheduled_at or _schedule_anchor(config.schedule, datetime.now(UTC)))
        contacts = await self._crm_adapter.list_contacts_by_property_filters(
            [condition.model_dump() for condition in config.crm_filter.conditions]
        )
        scheduled = 0
        skipped_existing = 0
        for index, contact in enumerate(contacts):
            if not contact.phone_number:
                continue
            recipient_scheduled_at = target_scheduled_at + timedelta(seconds=config.schedule.jitter_seconds * index)
            created = await self._create_recipient(config, contact, recipient_scheduled_at)
            if created:
                scheduled += 1
            else:
                skipped_existing += 1
        return OutboundSeedResult(
            campaign_id=config.id,
            matched_contacts=len(contacts),
            scheduled_recipients=scheduled,
            skipped_existing=skipped_existing,
            scheduled_at=target_scheduled_at,
        )

    async def run_due(
        self,
        config: OutboundCampaignConfig,
        *,
        now: datetime | None = None,
    ) -> OutboundRunResult:
        await self.ensure_campaign(config)
        current_time = _ensure_utc(now or datetime.now(UTC))
        async with self._session_factory() as session:
            result = await session.execute(
                select(OutboundRecipientRecord)
                .where(
                    OutboundRecipientRecord.campaign_id == config.id,
                    OutboundRecipientRecord.status == "scheduled",
                    OutboundRecipientRecord.scheduled_at <= current_time,
                )
                .order_by(OutboundRecipientRecord.scheduled_at)
                .limit(config.schedule.max_per_run)
            )
            recipients = list(result.scalars())
            campaign = await session.get(OutboundCampaignRecord, config.id)
            if campaign is not None:
                campaign.last_run_at = current_time
            await session.commit()

        sent = 0
        failed = 0
        skipped = 0
        for recipient in recipients:
            outcome = await self._send_recipient(config, recipient.recipient_id, current_time)
            if outcome == "sent":
                sent += 1
            elif outcome == "skipped":
                skipped += 1
            else:
                failed += 1

        return OutboundRunResult(
            campaign_id=config.id,
            attempted=len(recipients),
            sent=sent,
            failed=failed,
            skipped=skipped,
        )

    async def scheduler_tick(
        self,
        config: OutboundCampaignConfig,
        *,
        now: datetime | None = None,
    ) -> tuple[OutboundSeedResult | None, OutboundRunResult]:
        await self.ensure_campaign(config)
        current_time = _ensure_utc(now or datetime.now(UTC))
        seed_result: OutboundSeedResult | None = None
        due_at = _latest_due_schedule(config.schedule, current_time)
        if config.status == "active" and due_at is not None:
            async with self._session_factory() as session:
                campaign = await session.get(OutboundCampaignRecord, config.id)
                last_seeded_at = _ensure_utc(campaign.last_seeded_at) if campaign and campaign.last_seeded_at else None
            if last_seeded_at is None or last_seeded_at < due_at:
                seed_result = await self.seed_campaign(config, scheduled_at=due_at)
                async with self._session_factory() as session:
                    campaign = await session.get(OutboundCampaignRecord, config.id)
                    if campaign is not None:
                        campaign.last_seeded_at = due_at
                    await session.commit()
        run_result = await self.run_due(config, now=current_time)
        return seed_result, run_result

    async def _create_recipient(
        self,
        config: OutboundCampaignConfig,
        contact: CRMContact,
        scheduled_at: datetime,
    ) -> bool:
        async with self._session_factory() as session:
            session.add(
                OutboundRecipientRecord(
                    recipient_id=uuid4().hex,
                    campaign_id=config.id,
                    phone_number=contact.phone_number,
                    external_id=contact.external_id,
                    conversation_id=f"outbound:{config.id}:{contact.phone_number}",
                    scheduled_at=scheduled_at,
                    metadata_json={"seeded_from_stage": contact.stage, "full_name": contact.full_name},
                )
            )
            try:
                await session.commit()
                return True
            except IntegrityError:
                await session.rollback()
                return False

    async def _send_recipient(
        self,
        config: OutboundCampaignConfig,
        recipient_id: str,
        now: datetime,
    ) -> Literal["sent", "failed", "skipped"]:
        async with self._session_factory() as session:
            recipient = await session.get(OutboundRecipientRecord, recipient_id)
            if recipient is None or recipient.status != "scheduled":
                return "skipped"
            recipient.status = "sending"
            recipient.attempts += 1
            await session.commit()

        try:
            contact = await self._crm_adapter.find_contact_by_phone(recipient.phone_number)
            if contact is None:
                await self._mark_recipient(recipient_id, "skipped", error="Contact no longer exists in CRM.")
                return "skipped"
            if not _contact_still_matches_stage_filter(contact, config):
                await self._mark_recipient(recipient_id, "skipped", error="Contact no longer matches campaign stage.")
                return "skipped"

            rendered_text, body_parameters = await _render_template(
                config.template,
                contact,
                name_validator=self._name_validator,
            )
            response = await self._channel_adapter.send_template(
                OutboundTemplateMessage(
                    conversation_id=recipient.conversation_id,
                    phone_number=recipient.phone_number,
                    template_name=config.template.name,
                    language_code=config.template.language,
                    phone_number_id=config.template.phone_number_id,
                    body_parameters=body_parameters,
                    rendered_text=rendered_text,
                    callback_data=f"campaign={config.id};recipient={recipient_id}",
                )
            )
            if not _send_was_accepted(response):
                await self._mark_recipient(
                    recipient_id,
                    "failed",
                    rendered_text=rendered_text,
                    error=f"Provider did not accept send: {response}",
                )
                return "failed"

            updated_contact = contact
            if config.after_send.stage:
                updated_contact = await self._crm_adapter.change_stage(contact.external_id, config.after_send.stage)
            if config.after_send.note:
                updated_contact = await self._crm_adapter.append_note(
                    updated_contact.external_id,
                    _render_after_send_note(config, now),
                )
            await self._memory_store.remember_contact(updated_contact)
            await self._memory_store.append_message(
                ConversationMessage(
                    message_id=f"outbound:{recipient_id}",
                    conversation_id=recipient.conversation_id,
                    phone_number=recipient.phone_number,
                    direction=Direction.OUTBOUND,
                    text=rendered_text,
                    created_at=now,
                    metadata={
                        "provider": "kapso",
                        "campaign_id": config.id,
                        "template_name": config.template.name,
                        "first_touch": True,
                    },
                )
            )
            await self._mark_recipient(
                recipient_id,
                "sent",
                rendered_text=rendered_text,
                sent_at=now,
                provider_message_id=_extract_provider_message_id(response),
            )
            return "sent"
        except Exception as exc:  # noqa: BLE001
            await self._mark_recipient(recipient_id, "failed", error=str(exc))
            return "failed"

    async def _mark_recipient(
        self,
        recipient_id: str,
        status: str,
        *,
        rendered_text: str | None = None,
        sent_at: datetime | None = None,
        provider_message_id: str | None = None,
        error: str | None = None,
    ) -> None:
        async with self._session_factory() as session:
            recipient = await session.get(OutboundRecipientRecord, recipient_id)
            if recipient is None:
                return
            recipient.status = status
            if rendered_text is not None:
                recipient.rendered_text = rendered_text
            if sent_at is not None:
                recipient.sent_at = sent_at
            if provider_message_id is not None:
                recipient.provider_message_id = provider_message_id
            recipient.error = error
            await session.commit()


async def _render_template(
    template: TemplateConfig,
    contact: CRMContact,
    *,
    name_validator: OutboundNameValidator | None = None,
) -> tuple[str, list[dict[str, str]]]:
    values = [
        await _resolve_parameter(parameter, contact, name_validator=name_validator)
        for parameter in template.parameters
    ]
    rendered = template.body_text
    for index, value in enumerate(values, start=1):
        rendered = rendered.replace(f"{{{{{index}}}}}", value)

    if template.parameter_format == "NAMED":
        for parameter, value in zip(template.parameters, values, strict=True):
            if parameter.name:
                rendered = rendered.replace(f"{{{{{parameter.name}}}}}", value)
        body_parameters = [
            {"type": "text", "text": value, "parameter_name": parameter.name or str(index)}
            for index, (parameter, value) in enumerate(zip(template.parameters, values, strict=True), start=1)
        ]
    else:
        body_parameters = [{"type": "text", "text": value} for value in values]
    return rendered, body_parameters


async def _resolve_parameter(
    parameter: TemplateParameterConfig,
    contact: CRMContact,
    *,
    name_validator: OutboundNameValidator | None = None,
) -> str:
    if parameter.type == "static":
        return parameter.value or ""
    if name_validator is not None:
        assessment = await name_validator.assess_outbound_greeting_name(contact.full_name)
        if assessment.status == "trusted" and assessment.normalized_name:
            return assessment.normalized_name
        return parameter.fallback or ""
    if is_specific_person_name(contact.full_name):
        return normalize_person_name(contact.full_name) or (contact.full_name or "").strip()
    return parameter.fallback or ""


def _render_after_send_note(config: OutboundCampaignConfig, sent_at: datetime) -> str:
    note = config.after_send.note or ""
    return (
        note.replace("{campaign_id}", config.id)
        .replace("{template_name}", config.template.name)
        .replace("{sent_at}", sent_at.isoformat())
    )


def _contact_still_matches_stage_filter(contact: CRMContact, config: OutboundCampaignConfig) -> bool:
    for condition in config.crm_filter.conditions:
        if condition.type == "status" and condition.property.lower() in {"etapa", "stage"}:
            return contact.stage == condition.equals
    return True


def _send_was_accepted(response: dict) -> bool:
    status = str(response.get("status") or "").strip().lower()
    if status in {"disabled", "skipped", "failed", "error"}:
        return False
    if response.get("error") or response.get("errors"):
        return False
    return True


def _extract_provider_message_id(response: dict) -> str | None:
    messages = response.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        value = messages[0].get("id") or messages[0].get("message_id")
        return str(value) if value else None
    value = response.get("id") or response.get("message_id")
    return str(value) if value else None


def _latest_due_schedule(schedule: ScheduleConfig, now: datetime) -> datetime | None:
    current_time = _ensure_utc(now)
    if schedule.mode == "once_at":
        if not schedule.send_at:
            raise ValueError("once_at schedules require send_at.")
        send_at = _parse_local_datetime(schedule.send_at, ZoneInfo(schedule.timezone))
        return send_at if send_at <= current_time else None

    local_now = current_time.astimezone(ZoneInfo(schedule.timezone))
    allowed_days = {_DAY_INDEX[day] for day in schedule.days}
    candidates: list[datetime] = []
    for day_offset in range(8):
        local_day = (local_now - timedelta(days=day_offset)).date()
        if local_day.weekday() not in allowed_days:
            continue
        for time_value in schedule.times:
            candidate_time = _parse_time(time_value)
            candidate = datetime.combine(local_day, candidate_time, tzinfo=ZoneInfo(schedule.timezone)).astimezone(UTC)
            if candidate <= current_time:
                candidates.append(candidate)
    return max(candidates) if candidates else None


def _schedule_anchor(schedule: ScheduleConfig, now: datetime) -> datetime:
    due_at = _latest_due_schedule(schedule, now)
    if due_at is not None:
        return due_at
    if schedule.mode == "once_at" and schedule.send_at:
        return _parse_local_datetime(schedule.send_at, ZoneInfo(schedule.timezone))
    return _next_recurring_schedule(schedule, now)


def _next_recurring_schedule(schedule: ScheduleConfig, now: datetime) -> datetime:
    local_now = _ensure_utc(now).astimezone(ZoneInfo(schedule.timezone))
    allowed_days = {_DAY_INDEX[day] for day in schedule.days}
    for day_offset in range(8):
        local_day = (local_now + timedelta(days=day_offset)).date()
        if local_day.weekday() not in allowed_days:
            continue
        for time_value in sorted(schedule.times):
            candidate = datetime.combine(local_day, _parse_time(time_value), tzinfo=ZoneInfo(schedule.timezone))
            if candidate.astimezone(UTC) > _ensure_utc(now):
                return candidate.astimezone(UTC)
    raise ValueError("No future recurring schedule found.")


def _parse_local_datetime(value: str, timezone: ZoneInfo) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone)
    return parsed.astimezone(UTC)


def _parse_time(value: str) -> time:
    return time.fromisoformat(value)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
