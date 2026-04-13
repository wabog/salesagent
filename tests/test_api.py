import asyncio
import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from sales_agent.api.app import create_app
from sales_agent.core.config import Settings
from sales_agent.domain.models import Direction


@pytest.mark.asyncio
async def test_healthz_and_webhook_flow():
    app = create_app(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            MESSAGE_BATCH_WINDOW_SECONDS=0.05,
        )
    )
    transport = ASGITransport(app=app)
    payload = json.loads(Path("fixtures/kapso_webhook.json").read_text())

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            health = await client.get("/healthz")
            webhook = await client.post("/webhooks/whatsapp/kapso", json=payload)

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert webhook.status_code == 200
    assert webhook.json()["accepted"] is True
    assert webhook.json()["queued"] is True
    assert webhook.json()["duplicate"] is False


@pytest.mark.asyncio
async def test_webhook_ignores_non_message_events():
    app = create_app(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            MESSAGE_BATCH_WINDOW_SECONDS=0.05,
        )
    )
    transport = ASGITransport(app=app)
    payload = {
        "event": "message.status",
        "data": {
            "message": {
                "id": "msg-status",
                "timestamp": "1775704462",
            }
        },
    }

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            webhook = await client.post("/webhooks/whatsapp/kapso", json=payload)

    assert webhook.status_code == 200
    assert webhook.json()["accepted"] is False
    assert webhook.json()["render_reply"] is False
    assert "text content" in webhook.json()["ignored_reason"]


@pytest.mark.asyncio
async def test_local_chat_playground_flow():
    app = create_app(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            MESSAGE_BATCH_WINDOW_SECONDS=0.05,
        )
    )
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
    assert chat.json()["render_reply"] is True


@pytest.mark.asyncio
async def test_playground_disabled_in_production():
    app = create_app(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            APP_ENV="production",
            MESSAGE_BATCH_WINDOW_SECONDS=0.05,
        )
    )
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            page = await client.get("/playground")
            chat = await client.post(
                "/chat/local",
                json={"text": "hola", "phone_number": "3156832405", "conversation_id": "prod-test"},
            )

    assert page.status_code == 404
    assert chat.status_code == 404


@pytest.mark.asyncio
async def test_playground_token_protection():
    app = create_app(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            PLAYGROUND_ENABLED="true",
            PLAYGROUND_TOKEN="secret-token",
            MESSAGE_BATCH_WINDOW_SECONDS=0.05,
        )
    )
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            forbidden_page = await client.get("/playground")
            forbidden_chat = await client.post(
                "/chat/local",
                json={"text": "hola", "phone_number": "3156832405", "conversation_id": "protected-test"},
            )
            allowed_page = await client.get("/playground", params={"token": "secret-token"})
            allowed_chat = await client.post(
                "/chat/local",
                headers={"X-Playground-Token": "secret-token"},
                json={"text": "hola", "phone_number": "3156832405", "conversation_id": "protected-test"},
            )

    assert forbidden_page.status_code == 401
    assert forbidden_chat.status_code == 401
    assert allowed_page.status_code == 200
    assert allowed_chat.status_code == 200


@pytest.mark.asyncio
async def test_local_chat_batches_quick_messages_into_one_reply():
    app = create_app(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            MESSAGE_BATCH_WINDOW_SECONDS=0.05,
        )
    )
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            payloads = [
                {"text": "hola", "phone_number": "3156832405", "conversation_id": "batch-test"},
                {"text": "como", "phone_number": "3156832405", "conversation_id": "batch-test"},
                {"text": "vas", "phone_number": "3156832405", "conversation_id": "batch-test"},
            ]
            responses = await asyncio.gather(
                *(client.post("/chat/local", json=payload) for payload in payloads)
            )

        recent_messages = await app.state.sales_agent.memory_store.get_recent_messages("batch-test", limit=10)

    body = [response.json() for response in responses]
    renderable = [item for item in body if item["render_reply"]]
    suppressed = [item for item in body if not item["render_reply"]]
    outbound_messages = [message for message in recent_messages if message.direction == Direction.OUTBOUND]
    inbound_messages = [message for message in recent_messages if message.direction == Direction.INBOUND]

    assert len(renderable) == 1
    assert len(suppressed) == 2
    assert renderable[0]["aggregated_messages"] == 3
    assert len(outbound_messages) == 1
    assert len(inbound_messages) == 3
