import pytest
from datetime import date, timedelta

from sales_agent.adapters.crm_memory import InMemoryCRMAdapter


@pytest.mark.asyncio
async def test_in_memory_crm_lifecycle():
    crm = InMemoryCRMAdapter()

    created = await crm.create_contact("573001112233", "Fabian")
    updated = await crm.change_stage(created.external_id, "Qualified")
    noted = await crm.append_note(created.external_id, "Pidió una demo")
    followup = await crm.create_followup(created.external_id, "Enviar propuesta mañana")

    assert created.phone_number == "+573001112233"
    assert updated.stage == "Qualified"
    assert noted.notes[-1] == "Pidió una demo"
    assert followup["summary"] == "Enviar propuesta mañana"
    assert followup["due_date"] == (date.today() + timedelta(days=1)).isoformat()
    refreshed = await crm.find_contact_by_phone("573001112233")
    assert refreshed is not None
    assert refreshed.followup_summary == "Enviar propuesta mañana"
    assert refreshed.followup_due_date == date.today() + timedelta(days=1)


@pytest.mark.asyncio
async def test_in_memory_crm_complete_followup_clears_active_summary():
    crm = InMemoryCRMAdapter()

    created = await crm.create_contact("573001112233", "Fabian")
    await crm.create_followup(created.external_id, "Enviar propuesta mañana", due_date="2026-04-16")

    completion = await crm.complete_followup(created.external_id, outcome="Propuesta enviada al lead")
    refreshed = await crm.find_contact_by_phone("573001112233")

    assert completion["status"] == "completed"
    assert completion["cleared_summary"] == "Enviar propuesta mañana"
    assert refreshed is not None
    assert refreshed.followup_summary is None
    assert refreshed.followup_due_date is None
    assert crm.followups[-1]["status"] == "completed"
    assert crm.followups[-1]["outcome"] == "Propuesta enviada al lead"
