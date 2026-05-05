from __future__ import annotations

from datetime import date

from sales_agent.domain.models import CRMContact
from sales_agent.services.name_validation import contact_has_reliable_name


class LeadScopedCRMTools:
    def __init__(self, crm_adapter, current_lead: CRMContact, calendar_adapter=None) -> None:
        self.crm = crm_adapter
        self.current_lead = current_lead
        self.calendar = calendar_adapter

    async def get_current(self) -> CRMContact:
        return self.current_lead

    async def update_stage(self, stage: str) -> CRMContact:
        self.current_lead = self._merge_local_metadata(
            await self.crm.change_stage(self.current_lead.external_id, stage)
        )
        return self.current_lead

    async def update_fields(self, fields: dict) -> CRMContact:
        self.current_lead = self._merge_local_metadata(
            await self.crm.update_contact_fields(self.current_lead.external_id, fields)
        )
        return self.current_lead

    async def add_note(self, note: str) -> CRMContact:
        self.current_lead = self._merge_local_metadata(
            await self.crm.append_note(
                self.current_lead.external_id,
                self._format_timeline_note(note),
            )
        )
        return self.current_lead

    def _format_timeline_note(self, message: str) -> str:
        normalized = " ".join(message.strip().split())
        if not normalized:
            return normalized
        today = date.today().isoformat()
        if normalized.startswith(today):
            return normalized
        return f"{today} - {normalized}"

    def _normalize_sentence(self, text: str) -> str:
        normalized = " ".join(text.strip().split()).rstrip(". ")
        return normalized

    async def create_followup(self, summary: str, due_date: str | None = None) -> dict:
        payload = await self.crm.create_followup(self.current_lead.external_id, summary, due_date=due_date)
        due_suffix = f" Vence: {payload['due_date']}." if payload.get("due_date") else ""
        await self.crm.append_note(
            self.current_lead.external_id,
            self._format_timeline_note(f"Seguimiento creado: {self._normalize_sentence(summary)}.{due_suffix}"),
        )
        refreshed = await self.crm.find_contact_by_phone(self.current_lead.phone_number)
        if refreshed is not None:
            self.current_lead = self._merge_local_metadata(refreshed)
        return payload

    async def complete_followup(self, outcome: str | None = None) -> dict:
        payload = await self.crm.complete_followup(self.current_lead.external_id, outcome=outcome)
        summary = self._normalize_sentence(payload.get("cleared_summary") or "Seguimiento activo")
        normalized_outcome = self._normalize_sentence(outcome or "")
        outcome_suffix = f" Resultado: {normalized_outcome}." if normalized_outcome else ""
        await self.crm.append_note(
            self.current_lead.external_id,
            self._format_timeline_note(f"Seguimiento completado: {summary}.{outcome_suffix}"),
        )
        refreshed = await self.crm.find_contact_by_phone(self.current_lead.phone_number)
        if refreshed is not None:
            self.current_lead = self._merge_local_metadata(refreshed)
        return payload

    async def create_meeting(
        self,
        *,
        start_iso: str,
        duration_minutes: int,
        title: str,
        description: str,
    ) -> dict:
        if self.calendar is None:
            raise ValueError("Calendar integration is not configured.")
        payload = await self.calendar.create_meeting(
            self.current_lead,
            start_iso=start_iso,
            duration_minutes=duration_minutes,
            title=title,
            description=description,
        )
        metadata = dict(self.current_lead.metadata or {})
        calendar_state = dict((metadata.get("calendar") or {}))
        calendar_state["upcoming_event"] = payload
        calendar_state["just_booked"] = True
        metadata["calendar"] = calendar_state
        self.current_lead = self.current_lead.model_copy(update={"metadata": metadata})
        return payload

    def _merge_local_metadata(self, refreshed: CRMContact) -> CRMContact:
        local_metadata = dict(self.current_lead.metadata or {})
        refreshed_metadata = dict(refreshed.metadata or {})
        merged_metadata = {**refreshed_metadata, **local_metadata}
        resolved_name = refreshed.full_name
        if contact_has_reliable_name(self.current_lead):
            resolved_name = self.current_lead.full_name
        elif not resolved_name:
            resolved_name = self.current_lead.full_name
        return refreshed.model_copy(update={"full_name": resolved_name, "metadata": merged_metadata})
