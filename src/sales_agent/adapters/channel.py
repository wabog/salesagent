from __future__ import annotations

import httpx

from sales_agent.core.config import Settings
from sales_agent.domain.models import OutboundMessage


class ConsoleChannelAdapter:
    async def send_text(self, message: OutboundMessage) -> dict:
        return {"provider": "console", "status": "skipped", "text": message.text}


class KapsoWhatsAppAdapter:
    def __init__(self, settings: Settings) -> None:
        if not settings.kapso_api_token or not settings.kapso_phone_number_id:
            raise ValueError("Kapso settings are incomplete.")
        self._settings = settings

    async def send_text(self, message: OutboundMessage) -> dict:
        if not self._settings.whatsapp_send_enabled:
            return {"provider": "kapso", "status": "disabled", "text": message.text}
        url = f"{self._settings.kapso_base_url}/meta/whatsapp/v24.0/{self._settings.kapso_phone_number_id}/messages"
        body = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": message.phone_number,
            "type": "text",
            "text": {"body": message.text},
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                url,
                headers={"X-API-Key": self._settings.kapso_api_token},
                json=body,
            )
            response.raise_for_status()
            return response.json()
