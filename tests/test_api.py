import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from sales_agent.api.app import create_app
from sales_agent.core.config import Settings


@pytest.mark.asyncio
async def test_healthz_and_webhook_flow():
    app = create_app(Settings(DATABASE_URL="sqlite+aiosqlite:///:memory:", OPENAI_API_KEY=""))
    transport = ASGITransport(app=app)
    payload = json.loads(Path("fixtures/kapso_webhook.json").read_text())

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            health = await client.get("/healthz")
            webhook = await client.post("/webhooks/whatsapp/kapso", json=payload)

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert webhook.status_code == 200
    assert webhook.json()["duplicate"] is False


@pytest.mark.asyncio
async def test_local_chat_playground_flow():
    app = create_app(Settings(DATABASE_URL="sqlite+aiosqlite:///:memory:", OPENAI_API_KEY=""))
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            page = await client.get("/playground")
            chat = await client.post(
                "/chat/local",
                json={
                    "text": "hola",
                    "phone_number": "3156832405",
                    "conversation_id": "playground-test",
                },
            )

    assert page.status_code == 200
    assert "Sales Agent Playground" in page.text
    assert chat.status_code == 200
    assert chat.json()["duplicate"] is False
