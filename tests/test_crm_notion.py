import pytest
from datetime import date

from sales_agent.adapters.crm_notion import NotionCRMAdapter
from sales_agent.core.config import Settings


def test_pick_active_page_skips_trashed_results():
    trashed = {"id": "lead-trashed", "archived": True, "in_trash": True}
    active = {"id": "lead-active", "archived": False, "in_trash": False}

    selected = NotionCRMAdapter._pick_active_page([trashed, active])  # noqa: SLF001

    assert selected == active


def test_pick_active_page_returns_none_when_all_results_are_trashed():
    selected = NotionCRMAdapter._pick_active_page(  # noqa: SLF001
        [
            {"id": "lead-1", "archived": True, "in_trash": True},
            {"id": "lead-2", "archived": False, "in_trash": True},
        ]
    )

    assert selected is None


@pytest.mark.asyncio
async def test_find_contact_by_phone_falls_back_to_normalized_scan(monkeypatch):
    adapter = NotionCRMAdapter(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="notion",
            NOTION_API_KEY="test-token",
            NOTION_DATA_SOURCE_ID="test-ds",
        )
    )

    async def fake_exact_lookup(phone_number):  # noqa: ANN001
        assert phone_number in {"+573137661093", "573137661093", "3137661093"}
        return None

    async def fake_scan(normalized_phone):  # noqa: ANN001
        assert normalized_phone == "+573137661093"
        return {
            "id": "lead-1",
            "properties": {
                "Nombre": {"title": [{"plain_text": "Fabian"}]},
                "Etapa": {"status": {"name": "Prospecto"}},
                "Telefono": {"phone_number": "313 7661093"},
            },
        }

    monkeypatch.setattr(adapter, "_query_by_exact_phone", fake_exact_lookup)
    monkeypatch.setattr(adapter, "_scan_by_normalized_phone", fake_scan)

    contact = await adapter.find_contact_by_phone("3137661093")

    assert contact is not None
    assert contact.phone_number == "+573137661093"
    assert contact.full_name == "Fabian"


def test_to_contact_reads_followup_summary_and_due_date():
    adapter = NotionCRMAdapter(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="notion",
            NOTION_API_KEY="test-token",
            NOTION_DATA_SOURCE_ID="test-ds",
        )
    )

    contact = adapter._to_contact(  # noqa: SLF001
        {
            "id": "lead-1",
            "properties": {
                "Nombre": {"title": [{"plain_text": "Fabian"}]},
                "Etapa": {"status": {"name": "Prospecto"}},
                "Telefono": {"phone_number": "313 7661093"},
                "Resumen seguimiento": {"rich_text": [{"plain_text": "Enviar propuesta comercial"}]},
                "Próxima acción": {"date": {"start": "2026-04-20"}},
            },
        }
    )

    assert contact.followup_summary == "Enviar propuesta comercial"
    assert contact.followup_due_date == date(2026, 4, 20)


def test_to_contact_reads_notes_as_timeline_lines():
    adapter = NotionCRMAdapter(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="notion",
            NOTION_API_KEY="test-token",
            NOTION_DATA_SOURCE_ID="test-ds",
        )
    )

    contact = adapter._to_contact(  # noqa: SLF001
        {
            "id": "lead-1",
            "properties": {
                "Nombre": {"title": [{"plain_text": "Fabian"}]},
                "Etapa": {"status": {"name": "Prospecto"}},
                "Telefono": {"phone_number": "313 7661093"},
                "Notas": {
                    "rich_text": [
                        {"plain_text": "2026-04-15 - Seguimiento creado: Enviar propuesta.\n2026-04-16 - Seguimiento completado: Enviar propuesta."}
                    ]
                },
            },
        }
    )

    assert contact.notes == [
        "2026-04-15 - Seguimiento creado: Enviar propuesta.",
        "2026-04-16 - Seguimiento completado: Enviar propuesta.",
    ]
