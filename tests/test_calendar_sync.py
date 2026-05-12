import pytest
from datetime import datetime, timedelta, timezone

from sales_agent.domain.models import CRMContact
from sales_agent.services.calendar_sync import enrich_contact_with_calendar


class FailingCalendarAdapter:
    async def find_upcoming_meeting(self, contact: CRMContact):
        raise RuntimeError("calendar token refresh failed")


class EmptyCalendarAdapter:
    async def find_upcoming_meeting(self, contact: CRMContact):
        return None


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


@pytest.mark.asyncio
async def test_enrich_contact_with_calendar_does_not_keep_past_cached_event():
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Ana Perez",
        email="ana@example.com",
        metadata={
            "calendar": {
                "upcoming_event": {
                    "id": "past-event",
                    "start_iso": "2026-05-07T15:00:00-05:00",
                    "end_iso": "2026-05-07T15:30:00-05:00",
                }
            }
        },
    )

    enriched = await enrich_contact_with_calendar(
        contact,
        EmptyCalendarAdapter(),
        "https://calendar.app.google/demo",
    )

    assert enriched.metadata["calendar"]["upcoming_event"] is None


@pytest.mark.asyncio
async def test_enrich_contact_with_calendar_preserves_future_cached_event_when_lookup_is_empty():
    future_start = datetime.now(timezone.utc) + timedelta(days=2)
    future_end = future_start + timedelta(minutes=30)
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Ana Perez",
        email="ana@example.com",
        metadata={
            "calendar": {
                "upcoming_event": {
                    "id": "future-event",
                    "start_iso": future_start.isoformat(),
                    "end_iso": future_end.isoformat(),
                }
            }
        },
    )

    enriched = await enrich_contact_with_calendar(
        contact,
        EmptyCalendarAdapter(),
        "https://calendar.app.google/demo",
    )

    assert enriched.metadata["calendar"]["upcoming_event"]["id"] == "future-event"
