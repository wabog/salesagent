import pytest

from sales_agent.domain.models import CRMContact
from sales_agent.services.calendar_sync import enrich_contact_with_calendar


class FailingCalendarAdapter:
    async def find_upcoming_meeting(self, contact: CRMContact):
        raise RuntimeError("calendar token refresh failed")


@pytest.mark.asyncio
async def test_enrich_contact_with_calendar_degrades_when_lookup_fails():
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Ana Perez",
        email="ana@example.com",
    )

    enriched = await enrich_contact_with_calendar(
        contact,
        FailingCalendarAdapter(),
        "https://calendar.app.google/demo",
    )

    calendar = enriched.metadata["calendar"]

    assert calendar["connected"] is True
    assert calendar["available"] is False
    assert calendar["self_schedule_url"] == "https://calendar.app.google/demo"
    assert "token refresh failed" in calendar["error"]
