from __future__ import annotations

from datetime import date

from sales_agent.domain.models import CRMContact


class LeadScopedCRMTools:
    def __init__(self, crm_adapter, current_lead: CRMContact) -> None:
        self.crm = crm_adapter
        self.current_lead = current_lead

    async def get_current(self) -> CRMContact:
        return self.current_lead

    async def update_stage(self, stage: str) -> CRMContact:
        self.current_lead = await self.crm.change_stage(self.current_lead.external_id, stage)
        return self.current_lead

    async def update_fields(self, fields: dict) -> CRMContact:
        self.current_lead = await self.crm.update_contact_fields(self.current_lead.external_id, fields)
        return self.current_lead

    async def add_note(self, note: str) -> CRMContact:
        self.current_lead = await self.crm.append_note(
            self.current_lead.external_id,
            self._format_timeline_note(note),
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
            self.current_lead = refreshed
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
            self.current_lead = refreshed
        return payload
