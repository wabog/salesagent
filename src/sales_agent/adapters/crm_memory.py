from __future__ import annotations

from datetime import date, timedelta
from uuid import uuid4

from sales_agent.domain.phones import normalize_phone_number
from sales_agent.domain.models import CRMContact


class InMemoryCRMAdapter:
    def __init__(self) -> None:
        self._contacts_by_phone: dict[str, CRMContact] = {}
        self._contacts_by_id: dict[str, CRMContact] = {}
        self.followups: list[dict] = []

    async def find_contact_by_phone(self, phone_number: str) -> CRMContact | None:
        return self._contacts_by_phone.get(normalize_phone_number(phone_number))

    async def create_contact(self, phone_number: str, full_name: str | None = None) -> CRMContact:
        phone_number = normalize_phone_number(phone_number)
        existing = self._contacts_by_phone.get(phone_number)
        if existing:
            return existing
        contact = CRMContact(
            external_id=f"mem-{uuid4().hex[:10]}",
            phone_number=phone_number,
            full_name=full_name,
            stage="new",
        )
        self._contacts_by_phone[phone_number] = contact
        self._contacts_by_id[contact.external_id] = contact
        return contact

    async def update_contact_fields(self, external_id: str, fields: dict) -> CRMContact:
        contact = self._contacts_by_id[external_id]
        updated = contact.model_copy(update=fields)
        self._contacts_by_id[external_id] = updated
        self._contacts_by_phone[updated.phone_number] = updated
        return updated

    async def append_note(self, external_id: str, note: str) -> CRMContact:
        contact = self._contacts_by_id[external_id]
        updated = contact.model_copy(update={"notes": [*contact.notes, note]})
        self._contacts_by_id[external_id] = updated
        self._contacts_by_phone[updated.phone_number] = updated
        return updated

    async def change_stage(self, external_id: str, stage: str) -> CRMContact:
        return await self.update_contact_fields(external_id, {"stage": stage})

    async def create_followup(self, external_id: str, summary: str, due_date: str | None = None) -> dict:
        target_due_date = due_date or (date.today() + timedelta(days=1)).isoformat()
        contact = self._contacts_by_id[external_id]
        updated = contact.model_copy(
            update={
                "followup_summary": summary,
                "followup_due_date": date.fromisoformat(target_due_date),
            }
        )
        self._contacts_by_id[external_id] = updated
        self._contacts_by_phone[updated.phone_number] = updated
        followup = {
            "id": f"followup-{uuid4().hex[:8]}",
            "external_id": external_id,
            "summary": summary,
            "due_date": target_due_date,
            "status": "pending",
        }
        self.followups.append(followup)
        return followup

    async def complete_followup(self, external_id: str, outcome: str | None = None) -> dict:
        contact = self._contacts_by_id[external_id]
        cleared_summary = contact.followup_summary
        cleared_due_date = contact.followup_due_date.isoformat() if contact.followup_due_date else None
        updated = contact.model_copy(
            update={
                "followup_summary": None,
                "followup_due_date": None,
            }
        )
        self._contacts_by_id[external_id] = updated
        self._contacts_by_phone[updated.phone_number] = updated
        for followup in reversed(self.followups):
            if followup.get("external_id") != external_id or followup.get("status") != "pending":
                continue
            followup["status"] = "completed"
            followup["completed_at"] = date.today().isoformat()
            if outcome:
                followup["outcome"] = outcome
            break
        return {
            "external_id": external_id,
            "status": "completed",
            "cleared_summary": cleared_summary,
            "cleared_due_date": cleared_due_date,
            "outcome": outcome,
        }
