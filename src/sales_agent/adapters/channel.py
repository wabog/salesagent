from __future__ import annotations

import httpx

from sales_agent.core.config import Settings
from sales_agent.domain.phones import phone_to_provider_digits
from sales_agent.domain.models import InboundMessage, OutboundMessage, OutboundTemplateMessage


class ConsoleChannelAdapter:
    async def send_text(self, message: OutboundMessage) -> dict:
        return {"provider": "console", "status": "skipped", "text": message.text}

    async def send_template(self, message: OutboundTemplateMessage) -> dict:
        return {
            "provider": "console",
            "status": "skipped",
            "template_name": message.template_name,
            "text": message.rendered_text,
        }

    async def send_typing_indicator(self, event: InboundMessage) -> dict:
        return {"provider": "console", "status": "skipped", "message_id": event.message_id}


class KapsoWhatsAppAdapter:
    def __init__(self, settings: Settings) -> None:
        if not settings.kapso_api_token or not settings.kapso_phone_number_id:
            raise ValueError("Kapso settings are incomplete.")
        self._settings = settings

    async def send_text(self, message: OutboundMessage) -> dict:
        if not self._settings.whatsapp_send_enabled:
            return {"provider": "kapso", "status": "disabled", "text": message.text}
        url = self._messages_url()
        body = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": phone_to_provider_digits(
                message.phone_number,
                default_country_code=self._settings.phone_default_country_code,
            ),
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

    async def send_template(self, message: OutboundTemplateMessage) -> dict:
        if not self._settings.whatsapp_send_enabled:
            return {
                "provider": "kapso",
                "status": "disabled",
                "template_name": message.template_name,
                "text": message.rendered_text,
            }
        body = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": phone_to_provider_digits(
                message.phone_number,
                default_country_code=self._settings.phone_default_country_code,
            ),
            "type": "template",
            "template": {
                "name": message.template_name,
                "language": {"code": message.language_code},
                "components": [
                    {
                        "type": "body",
                        "parameters": message.body_parameters,
                    }
                ],
            },
        }
        if message.callback_data:
            body["biz_opaque_callback_data"] = message.callback_data
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                self._messages_url(message.phone_number_id),
                headers={"X-API-Key": self._settings.kapso_api_token},
                json=body,
            )
            response.raise_for_status()
            return response.json()

    async def send_typing_indicator(self, event: InboundMessage) -> dict:
        if not self._settings.whatsapp_send_enabled:
            return {"provider": "kapso", "status": "disabled", "message_id": event.message_id}
        url = self._messages_url()
        body = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": event.message_id,
            "typing_indicator": {"type": "text"},
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                url,
                headers={"X-API-Key": self._settings.kapso_api_token},
                json=body,
            )
            response.raise_for_status()
            return response.json()

    def _messages_url(self, phone_number_id: str | None = None) -> str:
        resolved_phone_number_id = phone_number_id or self._settings.kapso_phone_number_id
        return f"{self._settings.kapso_base_url}/meta/whatsapp/v24.0/{resolved_phone_number_id}/messages"
