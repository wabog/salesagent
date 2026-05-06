from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import httpx

from sales_agent.core.config import Settings
from sales_agent.domain.models import CRMContact


class GoogleCalendarAdapter:
    _GENERIC_CONTACT_NAMES = {
        "playground user",
        "lead",
        "lead demo",
        "usuario",
        "user",
    }

    def __init__(self, settings: Settings) -> None:
        if not settings.google_client_id or not settings.google_client_secret or not settings.google_refresh_token:
            raise ValueError("Google Calendar OAuth settings are incomplete.")
        self._settings = settings
        self._token_url = "https://oauth2.googleapis.com/token"
        self._base_url = "https://www.googleapis.com/calendar/v3"

    async def create_meeting(
        self,
        contact: CRMContact,
        *,
        start_iso: str,
        duration_minutes: int,
        title: str,
        description: str,
    ) -> dict[str, Any]:
        start_at = self._coerce_datetime(start_iso)
        end_at = start_at + timedelta(minutes=duration_minutes)
        conference_request_id = uuid4().hex
        body: dict[str, Any] = {
            "summary": title,
            "description": description,
            "start": {
                "dateTime": start_at.isoformat(),
                "timeZone": self._settings.google_calendar_timezone,
            },
            "end": {
                "dateTime": end_at.isoformat(),
                "timeZone": self._settings.google_calendar_timezone,
            },
        }
        if contact.email:
            body["attendees"] = [{"email": contact.email, "displayName": contact.full_name or None}]
        if self._settings.google_calendar_create_meet:
            body["conferenceData"] = {
                "createRequest": {
                    "requestId": conference_request_id,
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }
        access_token = await self._refresh_access_token()
        query_params = {"sendUpdates": "all"}
        if self._settings.google_calendar_create_meet:
            query_params["conferenceDataVersion"] = "1"
        async with httpx.AsyncClient(base_url=self._base_url, timeout=30.0) as client:
            response = await client.post(
                f"/calendars/{self._settings.google_calendar_id}/events",
                headers={"Authorization": f"Bearer {access_token}"},
                params=query_params,
                json=body,
            )
            response.raise_for_status()
            payload = response.json()
        return self._normalize_event(payload, source="agent_created")

    async def delete_meeting(self, event_id: str) -> dict[str, Any]:
        access_token = await self._refresh_access_token()
        async with httpx.AsyncClient(base_url=self._base_url, timeout=30.0) as client:
            response = await client.delete(
                f"/calendars/{self._settings.google_calendar_id}/events/{event_id}",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"sendUpdates": "all"},
            )
            response.raise_for_status()
        return {"id": event_id, "status": "cancelled", "source": "agent_deleted"}

    async def find_upcoming_meeting(self, contact: CRMContact, *, lookahead_days: int = 45) -> dict[str, Any] | None:
        if not contact.email and not contact.full_name:
            return None
        access_token = await self._refresh_access_token()
        now = datetime.now(timezone.utc)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=30.0) as client:
            response = await client.get(
                f"/calendars/{self._settings.google_calendar_id}/events",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "singleEvents": "true",
                    "orderBy": "startTime",
                    "timeMin": now.isoformat(),
                    "timeMax": (now + timedelta(days=lookahead_days)).isoformat(),
                    "maxResults": "100",
                },
            )
            response.raise_for_status()
            data = response.json()
        events = data.get("items", [])
        matching_events = [
            event for event in events if self._event_matches_contact(event, contact) and event.get("status") != "cancelled"
        ]
        if not matching_events:
            return None
        matching_events.sort(key=lambda item: ((item.get("start") or {}).get("dateTime") or ""))
        return self._normalize_event(matching_events[0], source="calendar_lookup")

    async def _refresh_access_token(self) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self._token_url,
                data={
                    "client_id": self._settings.google_client_id,
                    "client_secret": self._settings.google_client_secret,
                    "refresh_token": self._settings.google_refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()
            payload = response.json()
        return payload["access_token"]

    def _event_matches_contact(self, event: dict[str, Any], contact: CRMContact) -> bool:
        email = (contact.email or "").strip().lower()
        full_name = (contact.full_name or "").strip().lower()
        summary = str(event.get("summary") or "").lower()
        description = str(event.get("description") or "").lower()
        attendees = event.get("attendees") or []
        if email and any(str(item.get("email") or "").strip().lower() == email for item in attendees):
            return True
        phone_number = (contact.phone_number or "").strip()
        if phone_number and phone_number in description:
            return True
        if not phone_number and self._is_specific_contact_name(full_name) and (full_name in summary or full_name in description):
            return True
        return False

    def _is_specific_contact_name(self, full_name: str) -> bool:
        normalized = " ".join(full_name.strip().lower().split())
        if not normalized:
            return False
        if normalized in self._GENERIC_CONTACT_NAMES:
            return False
        return len(normalized) >= 6

    def _normalize_event(self, payload: dict[str, Any], *, source: str) -> dict[str, Any]:
        conference = payload.get("conferenceData") or {}
        entry_points = conference.get("entryPoints") or []
        meet_link = payload.get("hangoutLink")
        if not meet_link:
            for entry in entry_points:
                if entry.get("entryPointType") == "video":
                    meet_link = entry.get("uri")
                    if meet_link:
                        break
        return {
            "id": payload.get("id"),
            "summary": payload.get("summary"),
            "description": payload.get("description"),
            "start_iso": ((payload.get("start") or {}).get("dateTime") or (payload.get("start") or {}).get("date")),
            "end_iso": ((payload.get("end") or {}).get("dateTime") or (payload.get("end") or {}).get("date")),
            "html_link": payload.get("htmlLink"),
            "meet_link": meet_link,
            "status": payload.get("status"),
            "attendees": [item.get("email") for item in (payload.get("attendees") or []) if item.get("email")],
            "source": source,
        }

    def _coerce_datetime(self, value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
