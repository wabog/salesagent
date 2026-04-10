from __future__ import annotations

from typing import Any

import httpx

from sales_agent.core.config import Settings
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
        results = data.get("results", [])
        if not results:
            return None
        return self._to_contact(results[0])

    async def create_contact(self, phone_number: str, full_name: str | None = None) -> CRMContact:
        properties = {
            self._settings.notion_phone_property: {"phone_number": phone_number},
            self._settings.notion_name_property: {
                "title": [{"text": {"content": full_name or phone_number}}]
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
            notion_properties["Email"] = {"email": fields["email"]}
        data = await self._request("PATCH", f"/pages/{external_id}", json={"properties": notion_properties})
        return self._to_contact(data)

    async def append_note(self, external_id: str, note: str) -> CRMContact:
        await self._request(
            "PATCH",
            f"/blocks/{external_id}/children",
            json={"children": [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": note}}]}}]},
        )
        contact = await self._request("GET", f"/pages/{external_id}")
        return self._to_contact(contact)

    async def change_stage(self, external_id: str, stage: str) -> CRMContact:
        return await self.update_contact_fields(external_id, {"stage": stage})

    async def create_followup(self, external_id: str, summary: str) -> dict:
        return {"external_id": external_id, "summary": summary, "provider": "notion", "status": "recorded_manually"}

    async def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=20.0) as client:
            response = await client.request(method, path, headers=self._base_headers, json=json)
            response.raise_for_status()
            return response.json()

    def _to_contact(self, payload: dict) -> CRMContact:
        properties = payload.get("properties", {})
        name_prop = properties.get(self._settings.notion_name_property, {})
        stage_prop = properties.get(self._settings.notion_stage_property, {})
        phone_prop = properties.get(self._settings.notion_phone_property, {})
        name_chunks = name_prop.get("title", [])
        full_name = "".join(chunk.get("plain_text", "") for chunk in name_chunks) or None
        stage_data = stage_prop.get("status") or {}
        stage = stage_data.get("name")
        phone_number = phone_prop.get("phone_number")
        return CRMContact(
            external_id=payload["id"],
            phone_number=phone_number,
            full_name=full_name,
            stage=stage,
            metadata={"raw_properties": properties},
        )
