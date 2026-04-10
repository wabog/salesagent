import pytest

from sales_agent.adapters.crm_memory import InMemoryCRMAdapter


@pytest.mark.asyncio
async def test_in_memory_crm_lifecycle():
    crm = InMemoryCRMAdapter()

    created = await crm.create_contact("573001112233", "Fabian")
    updated = await crm.change_stage(created.external_id, "Qualified")
    noted = await crm.append_note(created.external_id, "Pidió una demo")
    followup = await crm.create_followup(created.external_id, "Enviar propuesta mañana")

    assert created.phone_number == "573001112233"
    assert updated.stage == "Qualified"
    assert noted.notes[-1] == "Pidió una demo"
    assert followup["summary"] == "Enviar propuesta mañana"
