import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from sales_agent.api.app import create_app
from sales_agent.core.config import Settings
from sales_agent.core.db import AgentRunRecord
from sales_agent.domain.models import ConversationMessage, Direction, InboundMessage
from sales_agent.services.media_preprocessor import MediaPreprocessingResult


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
    assert "supported inbound content" in webhook.json()["ignored_reason"]


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
async def test_local_chat_playground_does_not_force_default_contact_name():
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
            chat = await client.post(
                "/chat/local",
                json={
                    "text": "hola",
                    "phone_number": "3156832500",
                    "conversation_id": "playground-no-name",
                },
            )
        shadow = await app.state.sales_agent.memory_store.get_contact_shadow("+573156832500")

    assert chat.status_code == 200
    assert shadow is not None
    assert shadow.full_name is None


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


@pytest.mark.asyncio
async def test_playground_agent_context_exposes_repo_backed_prompt_files():
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
            context = await client.get("/playground/agent-context")

    assert context.status_code == 200
    payload = context.json()
    assert "planner_scaffold" in payload
    assert "Knowledge Context" in payload["planner_scaffold"]
    knowledge_names = [item["name"] for item in payload["knowledge_sections"]]
    assert "wabog_company" in knowledge_names
    assert "wabog_pricing" in knowledge_names


@pytest.mark.asyncio
async def test_prepare_batched_run_reuses_recent_messages_from_same_phone_across_conversations():
    app = create_app(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            MESSAGE_BATCH_WINDOW_SECONDS=0.05,
        )
    )

    async with app.router.lifespan_context(app):
        memory_store = app.state.sales_agent.memory_store
        await memory_store.append_message(
            ConversationMessage(
                message_id="prev-in-1",
                conversation_id="conv-old",
                phone_number="3156832405",
                direction=Direction.INBOUND,
                text="Necesito revisar una propuesta de Wabog.",
            )
        )
        await memory_store.append_message(
            ConversationMessage(
                message_id="prev-out-1",
                conversation_id="conv-old",
                phone_number="3156832405",
                direction=Direction.OUTBOUND,
                text="Perfecto, te envio la propuesta y manana hacemos seguimiento.",
            )
        )

        prepared = await app.state.sales_agent.prepare_batched_run(
            [
                InboundMessage(
                    message_id="new-conv-1",
                    conversation_id="conv-new",
                    phone_number="3156832405",
                    text="listo",
                    timestamp=datetime.now(timezone.utc),
                    raw_payload={},
                    provider="kapso",
                )
            ]
        )

    recent_texts = [message.text for message in prepared.recent_messages]
    assert "Necesito revisar una propuesta de Wabog." in recent_texts
    assert "Perfecto, te envio la propuesta y manana hacemos seguimiento." in recent_texts


@pytest.mark.asyncio
async def test_prepare_batched_run_playground_keeps_context_isolated_per_conversation():
    app = create_app(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            MESSAGE_BATCH_WINDOW_SECONDS=0.05,
        )
    )

    async with app.router.lifespan_context(app):
        memory_store = app.state.sales_agent.memory_store
        await memory_store.append_message(
            ConversationMessage(
                message_id="prev-in-playground",
                conversation_id="conv-old",
                phone_number="3156832405",
                direction=Direction.INBOUND,
                text="Necesito revisar una propuesta de Wabog.",
            )
        )
        await memory_store.append_message(
            ConversationMessage(
                message_id="prev-out-playground",
                conversation_id="conv-old",
                phone_number="3156832405",
                direction=Direction.OUTBOUND,
                text="Perfecto, te envío una propuesta y hacemos seguimiento.",
            )
        )

        prepared = await app.state.sales_agent.prepare_batched_run(
            [
                InboundMessage(
                    message_id="new-playground-1",
                    conversation_id="conv-new",
                    phone_number="3156832405",
                    text="hola",
                    timestamp=datetime.now(timezone.utc),
                    raw_payload={},
                    provider="local-playground",
                )
            ]
        )

    recent_texts = [message.text for message in prepared.recent_messages]
    assert "Necesito revisar una propuesta de Wabog." not in recent_texts
    assert "Perfecto, te envío una propuesta y hacemos seguimiento." not in recent_texts


@pytest.mark.asyncio
async def test_prepare_batched_run_ignores_other_phone_messages_inside_same_conversation():
    app = create_app(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            MESSAGE_BATCH_WINDOW_SECONDS=0.05,
        )
    )

    async with app.router.lifespan_context(app):
        memory_store = app.state.sales_agent.memory_store
        await memory_store.append_message(
            ConversationMessage(
                message_id="conv-shared-old",
                conversation_id="conv-shared",
                phone_number="3156832405",
                direction=Direction.INBOUND,
                text="Necesito una demo para mi despacho.",
            )
        )

        prepared = await app.state.sales_agent.prepare_batched_run(
            [
                InboundMessage(
                    message_id="conv-shared-new",
                    conversation_id="conv-shared",
                    phone_number="3156832500",
                    text="hola",
                    timestamp=datetime.now(timezone.utc),
                    raw_payload={},
                    provider="local-playground",
                )
            ]
        )

    recent_texts = [message.text for message in prepared.recent_messages]
    assert "Necesito una demo para mi despacho." not in recent_texts


@pytest.mark.asyncio
async def test_prepare_batched_run_document_only_uses_canned_reply(monkeypatch):
    app = create_app(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            MESSAGE_BATCH_WINDOW_SECONDS=0.05,
        )
    )

    async def fake_preprocess(event: InboundMessage) -> MediaPreprocessingResult:
        return MediaPreprocessingResult(
            bypass_intent="document_unsupported",
            bypass_response_text="document unsupported",
        )

    async with app.router.lifespan_context(app):
        monkeypatch.setattr(app.state.sales_agent.media_preprocessor, "preprocess_event", fake_preprocess)
        prepared = await app.state.sales_agent.prepare_batched_run(
            [
                InboundMessage(
                    message_id="doc-1",
                    conversation_id="conv-doc",
                    phone_number="3156832600",
                    text="",
                    timestamp=datetime.now(timezone.utc),
                    raw_payload={},
                    provider="local-playground",
                    message_type="document",
                )
            ]
        )

    assert prepared.planning.intent == "document_unsupported"
    assert prepared.planning.should_reply is True
    assert "no puedo procesar documentos" in prepared.response_text.lower()


@pytest.mark.asyncio
async def test_prepare_batched_run_sticker_only_skips_reply(monkeypatch):
    app = create_app(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            MESSAGE_BATCH_WINDOW_SECONDS=0.05,
        )
    )

    async def fake_preprocess(event: InboundMessage) -> MediaPreprocessingResult:
        return MediaPreprocessingResult(should_reply=False)

    async with app.router.lifespan_context(app):
        monkeypatch.setattr(app.state.sales_agent.media_preprocessor, "preprocess_event", fake_preprocess)
        prepared = await app.state.sales_agent.prepare_batched_run(
            [
                InboundMessage(
                    message_id="sticker-1",
                    conversation_id="conv-sticker",
                    phone_number="3156832601",
                    text="",
                    timestamp=datetime.now(timezone.utc),
                    raw_payload={},
                    provider="local-playground",
                    message_type="sticker",
                )
            ]
        )

    assert prepared.planning.intent == "non_text_ignored"
    assert prepared.planning.should_reply is False
    assert prepared.response_text == ""


@pytest.mark.asyncio
async def test_local_chat_persists_knowledge_lookup_in_run_logs():
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
        planner = app.state.sales_agent.workflow.planner

        async def fake_select_knowledge_sections(text, contact, recent_messages):  # noqa: ANN001
            from sales_agent.services.planner import KnowledgeSelectionOutput
            return KnowledgeSelectionOutput(
                section_names=["wabog_pricing"],
                reason="The user asked for pricing information.",
            )

        async def fake_plan_with_llm(text, contact, recent_messages, semantic_memories, knowledge_sections):  # noqa: ANN001
            from sales_agent.domain.models import PlanningResult
            return PlanningResult(
                intent="ask_pricing",
                confidence=0.9,
                response_text="El precio depende del volumen y del tipo de operación que tengan.",
                actions=[],
            )

        planner._llm = object()  # noqa: SLF001
        planner._select_knowledge_sections = fake_select_knowledge_sections  # type: ignore[method-assign]  # noqa: SLF001
        planner._plan_with_llm = fake_plan_with_llm  # type: ignore[method-assign]  # noqa: SLF001

        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/chat/local",
                json={
                    "text": "y eso cuanto vale",
                    "phone_number": "3156832800",
                    "conversation_id": "pricing-lookup-test",
                },
            )

        async with app.state.sales_agent.session_factory() as session:
            records = (await session.execute(select(AgentRunRecord))).scalars().all()

    assert response.status_code == 200
    assert len(records) == 1
    assert records[0].knowledge_lookups_json == [
        {
            "section": "wabog_pricing",
            "reason": "The user asked for pricing information.",
        }
    ]
