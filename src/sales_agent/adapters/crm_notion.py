from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import httpx

from sales_agent.core.config import Settings
from sales_agent.domain.phones import build_legacy_phone_candidates, normalize_phone_number, phone_to_provider_digits
from sales_agent.domain.models import CRMContact


class NotionCRMAdapter:
    def __init__(self, settings: Settings) -> None:
        if not settings.notion_api_key or not settings.notion_data_source_id:
            raise ValueError("Notion settings are incomplete.")
        self._settings = settings
        self._base_headers = {
            "Authorization": f"Bearer {settings.notion_api_key}",
            "Notion-Version": settings.notion_version,
            "Content-Type": "application/json",
        }
        self._base_url = "https://api.notion.com/v1"

    async def find_contact_by_phone(self, phone_number: str) -> CRMContact | None:
        normalized_phone = normalize_phone_number(
            phone_number,
            default_country_code=self._settings.phone_default_country_code,
        )

        for candidate in build_legacy_phone_candidates(
            normalized_phone,
            default_country_code=self._settings.phone_default_country_code,
        ):
            page = await self._query_by_exact_phone(candidate)
            if page is not None:
                return self._to_contact(page)

        page = await self._scan_by_normalized_phone(normalized_phone)
        if page is None:
            return None
        return self._to_contact(page)

    async def create_contact(self, phone_number: str, full_name: str | None = None) -> CRMContact:
        normalized_phone = normalize_phone_number(
            phone_number,
            default_country_code=self._settings.phone_default_country_code,
        )
        properties = {
            self._settings.notion_phone_property: {
                "phone_number": phone_to_provider_digits(
                    normalized_phone,
                    default_country_code=self._settings.phone_default_country_code,
                )
            },
            self._settings.notion_name_property: {
                "title": [{"text": {"content": full_name or phone_to_provider_digits(normalized_phone)}}]
            },
        }
        data = await self._request("POST", "/pages", json={"parent": {"data_source_id": self._settings.notion_data_source_id}, "properties": properties})
        return self._to_contact(data)

    async def update_contact_fields(self, external_id: str, fields: dict) -> CRMContact:
        notion_properties: dict[str, Any] = {}
        if "full_name" in fields and fields["full_name"]:
            notion_properties[self._settings.notion_name_property] = {
                "title": [{"text": {"content": fields["full_name"]}}]
            }
        if "stage" in fields and fields["stage"]:
            notion_properties[self._settings.notion_stage_property] = {"status": {"name": fields["stage"]}}
        if "email" in fields and fields["email"]:
            notion_properties[self._settings.notion_email_property] = {"email": fields["email"]}
        if "phone_number" in fields and fields["phone_number"]:
            notion_properties[self._settings.notion_phone_property] = {
                "phone_number": phone_to_provider_digits(
                    fields["phone_number"],
                    default_country_code=self._settings.phone_default_country_code,
                )
            }
        data = await self._request("PATCH", f"/pages/{external_id}", json={"properties": notion_properties})
        return self._to_contact(data)

    async def append_note(self, external_id: str, note: str) -> CRMContact:
        page = await self._request("GET", f"/pages/{external_id}")
        properties = page.get("properties", {})
        notes_prop = properties.get(self._settings.notion_notes_property, {})
        current_notes = notes_prop.get("rich_text", [])
        if not current_notes:
            current_notes = notes_prop.get("text", [])
        existing_text = "".join(chunk.get("plain_text", "") for chunk in current_notes).strip()
        new_value = f"{existing_text}\n{note}".strip() if existing_text else note
        data = await self._request(
            "PATCH",
            f"/pages/{external_id}",
            json={
                "properties": {
                    self._settings.notion_notes_property: {
                        "rich_text": [{"text": {"content": new_value[:1900]}}]
                    }
                }
            },
        )
        return self._to_contact(data)

    async def change_stage(self, external_id: str, stage: str) -> CRMContact:
        return await self.update_contact_fields(external_id, {"stage": stage})

    async def create_followup(self, external_id: str, summary: str, due_date: str | None = None) -> dict:
        target_due_date = due_date or (date.today() + timedelta(days=1)).isoformat()
        await self._request(
            "PATCH",
            f"/pages/{external_id}",
            json={
                "properties": {
                    self._settings.notion_followup_summary_property: {
                        "rich_text": [{"text": {"content": summary[:1900]}}]
                    },
                    self._settings.notion_next_action_property: {
                        "date": {"start": target_due_date}
                    },
                    self._settings.notion_last_contact_property: {
                        "date": {"start": date.today().isoformat()}
                    },
                }
            },
        )
        return {
            "external_id": external_id,
            "summary": summary,
            "due_date": target_due_date,
            "provider": "notion",
            "status": "recorded",
        }

    async def complete_followup(self, external_id: str, outcome: str | None = None) -> dict:
        page = await self._request("GET", f"/pages/{external_id}")
        contact = self._to_contact(page)
        await self._request(
            "PATCH",
            f"/pages/{external_id}",
            json={
                "properties": {
                    self._settings.notion_followup_summary_property: {
                        "rich_text": []
                    },
                    self._settings.notion_next_action_property: {
                        "date": None
                    },
                    self._settings.notion_last_contact_property: {
                        "date": {"start": date.today().isoformat()}
                    },
                }
            },
        )
        return {
            "external_id": external_id,
            "status": "completed",
            "cleared_summary": contact.followup_summary,
            "cleared_due_date": contact.followup_due_date.isoformat() if contact.followup_due_date else None,
            "outcome": outcome,
            "provider": "notion",
        }

    async def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=20.0) as client:
            response = await client.request(method, path, headers=self._base_headers, json=json)
            response.raise_for_status()
            return response.json()

    async def _query_by_exact_phone(self, phone_number: str) -> dict[str, Any] | None:
        payload = {
            "filter": {
                "property": self._settings.notion_phone_property,
                "phone_number": {"equals": phone_number},
            }
        }
        data = await self._request(
            "POST",
            f"/data_sources/{self._settings.notion_data_source_id}/query",
            json=payload,
        )
        return self._pick_active_page(data.get("results", []))

    async def _scan_by_normalized_phone(self, normalized_phone: str) -> dict[str, Any] | None:
        start_cursor: str | None = None
        while True:
            payload: dict[str, Any] = {"page_size": 100}
            if start_cursor:
                payload["start_cursor"] = start_cursor
            data = await self._request(
                "POST",
                f"/data_sources/{self._settings.notion_data_source_id}/query",
                json=payload,
            )
            matched_pages = [
                page
                for page in data.get("results", [])
                if not page.get("archived")
                and not page.get("in_trash")
                and normalize_phone_number(
                    ((page.get("properties", {}).get(self._settings.notion_phone_property) or {}).get("phone_number")),
                    default_country_code=self._settings.phone_default_country_code,
                )
                == normalized_phone
            ]
            page = self._pick_active_page(matched_pages)
            if page is not None:
                return page

            if not data.get("has_more"):
                return None
            start_cursor = data.get("next_cursor")

    def _to_contact(self, payload: dict) -> CRMContact:
        properties = payload.get("properties", {})
        name_prop = properties.get(self._settings.notion_name_property, {})
        stage_prop = properties.get(self._settings.notion_stage_property, {})
        phone_prop = properties.get(self._settings.notion_phone_property, {})
        email_prop = properties.get(self._settings.notion_email_property, {})
        notes_prop = properties.get(self._settings.notion_notes_property, {})
        followup_prop = properties.get(self._settings.notion_followup_summary_property, {})
        next_action_prop = properties.get(self._settings.notion_next_action_property, {})
        name_chunks = name_prop.get("title", [])
        full_name = "".join(chunk.get("plain_text", "") for chunk in name_chunks) or None
        stage_data = stage_prop.get("status") or {}
        stage = stage_data.get("name")
        phone_number = phone_prop.get("phone_number")
        email = email_prop.get("email")
        note_chunks = notes_prop.get("rich_text", []) or notes_prop.get("text", [])
        note_text = "".join(chunk.get("plain_text", "") for chunk in note_chunks).strip()
        notes = [line.strip() for line in note_text.splitlines() if line.strip()]
        followup_chunks = followup_prop.get("rich_text", []) or followup_prop.get("text", [])
        followup_summary = "".join(chunk.get("plain_text", "") for chunk in followup_chunks).strip() or None
        next_action_date = (next_action_prop.get("date") or {}).get("start")
        return CRMContact(
            external_id=payload["id"],
            phone_number=phone_number,
            full_name=full_name,
            stage=stage,
            email=email,
            followup_summary=followup_summary,
            followup_due_date=date.fromisoformat(next_action_date) if next_action_date else None,
            notes=notes,
            metadata={"raw_properties": properties},
        )

    @staticmethod
    def _pick_active_page(results: list[dict[str, Any]]) -> dict[str, Any] | None:
        for page in results:
            if page.get("archived") or page.get("in_trash"):
                continue
            return page
        return None
