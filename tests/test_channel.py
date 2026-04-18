import httpx
import pytest
from datetime import datetime, timezone

from sales_agent.adapters.channel import KapsoWhatsAppAdapter
from sales_agent.core.config import Settings
from sales_agent.domain.models import InboundMessage, OutboundMessage


@pytest.mark.asyncio
async def test_kapso_adapter_uses_x_api_key(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    adapter = KapsoWhatsAppAdapter(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            KAPSO_BASE_URL="https://api.kapso.ai",
            KAPSO_PHONE_NUMBER_ID="1101272743059797",
            KAPSO_API_TOKEN="kapso-token",
            WHATSAPP_SEND_ENABLED="true",
        )
    )

    result = await adapter.send_text(
        OutboundMessage(
            conversation_id="conv-1",
            phone_number="+57 300 1112233",
            text="hola",
        )
    )

    assert result == {"ok": True}
    assert captured["url"] == "https://api.kapso.ai/meta/whatsapp/v24.0/1101272743059797/messages"
    assert captured["headers"] == {"X-API-Key": "kapso-token"}
    assert captured["json"]["to"] == "573001112233"


@pytest.mark.asyncio
async def test_kapso_adapter_typing_indicator_uses_mark_read_payload(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    adapter = KapsoWhatsAppAdapter(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            KAPSO_BASE_URL="https://api.kapso.ai",
            KAPSO_PHONE_NUMBER_ID="1101272743059797",
            KAPSO_API_TOKEN="kapso-token",
            WHATSAPP_SEND_ENABLED="true",
        )
    )

    result = await adapter.send_typing_indicator(
        InboundMessage(
            message_id="wamid.123",
            conversation_id="conv-1",
            phone_number="+57 300 1112233",
            text="hola",
            timestamp=datetime.now(timezone.utc),
            raw_payload={},
        )
    )

    assert result == {"ok": True}
    assert captured["headers"] == {"X-API-Key": "kapso-token"}
    assert captured["json"] == {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": "wamid.123",
        "typing_indicator": {"type": "text"},
    }
