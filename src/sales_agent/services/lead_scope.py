from __future__ import annotations

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
        self.current_lead = await self.crm.append_note(self.current_lead.external_id, note)
        return self.current_lead

    async def create_followup(self, summary: str) -> dict:
        return await self.crm.create_followup(self.current_lead.external_id, summary)
