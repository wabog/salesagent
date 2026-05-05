from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sales_agent.domain.models import CRMContact
from sales_agent.services.name_validation import contact_has_reliable_name


logger = logging.getLogger("uvicorn.error")


def merge_contact_with_shadow(contact: CRMContact, shadow: CRMContact | None) -> CRMContact:
    if shadow is None:
        return contact
    merged_metadata = {**(shadow.metadata or {}), **(contact.metadata or {})}
    if contact_has_reliable_name(shadow):
        resolved_name = shadow.full_name
    elif contact_has_reliable_name(contact):
        resolved_name = contact.full_name
    else:
        resolved_name = contact.full_name or shadow.full_name
    return contact.model_copy(
        update={
            "full_name": resolved_name,
            "stage": contact.stage or shadow.stage,
            "email": contact.email or shadow.email,
            "notes": contact.notes or shadow.notes,
            "metadata": merged_metadata,
            "followup_summary": contact.followup_summary or shadow.followup_summary,
            "followup_due_date": contact.followup_due_date or shadow.followup_due_date,
        }
    )


async def enrich_contact_with_calendar(contact: CRMContact, calendar_adapter, self_schedule_url: str | None) -> CRMContact:
    metadata = dict(contact.metadata or {})
    previous_calendar = dict((metadata.get("calendar") or {}))
    calendar_state: dict[str, Any] = {
        "connected": bool(calendar_adapter),
        "self_schedule_url": self_schedule_url,
        "last_checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if calendar_adapter is None:
        metadata["calendar"] = calendar_state
        return contact.model_copy(update={"metadata": metadata})

    try:
        upcoming_event = await calendar_adapter.find_upcoming_meeting(contact)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "calendar_lookup_failed contact_id=%s error=%s",
            contact.external_id,
            exc,
        )
        calendar_state["available"] = False
        calendar_state["error"] = str(exc)
        metadata["calendar"] = calendar_state
        return contact.model_copy(update={"metadata": metadata})

    previous_event_id = ((previous_calendar.get("upcoming_event") or {}).get("id"))
    current_event_id = (upcoming_event or {}).get("id")
    if upcoming_event is None and previous_calendar.get("upcoming_event"):
        upcoming_event = previous_calendar.get("upcoming_event")
        current_event_id = (upcoming_event or {}).get("id")
    calendar_state["available"] = True
    calendar_state["upcoming_event"] = upcoming_event
    calendar_state["just_booked"] = bool(current_event_id and current_event_id != previous_event_id)
    metadata["calendar"] = calendar_state
    return contact.model_copy(update={"metadata": metadata})
