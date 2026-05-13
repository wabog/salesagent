from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from sales_agent.adapters.memory_sql import SqlAlchemyMemoryStore
from sales_agent.core.db import AgentRunRecord, ConversationThreadRecord, build_engine, build_session_factory, init_db
from sales_agent.domain.models import InboundMessage


@pytest.mark.asyncio
async def test_init_db_adds_missing_knowledge_lookups_column_for_legacy_sqlite(tmp_path: Path):
    database_path = tmp_path / "legacy.db"
    engine = build_engine(f"sqlite+aiosqlite:///{database_path}")

    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            """
            CREATE TABLE agent_runs (
                run_id VARCHAR(64) PRIMARY KEY,
                conversation_id VARCHAR(255) NOT NULL,
                message_id VARCHAR(255) NOT NULL,
                phone_number VARCHAR(64) NOT NULL,
                intent VARCHAR(128) NOT NULL,
                response_text TEXT NOT NULL,
                tool_results_json JSON NOT NULL DEFAULT '[]',
                created_at DATETIME NOT NULL
            )
            """
        )

    await init_db(engine)

    session_factory = build_session_factory(engine)
    async with session_factory() as session:
        session.add(
            AgentRunRecord(
                run_id="run-1",
                conversation_id="conv-1",
                message_id="msg-1",
                phone_number="+573001112233",
                intent="greeting",
                response_text="Hola",
                tool_results_json=[],
                knowledge_lookups_json=[{"section": "wabog_pricing", "reason": "Asked for price"}],
            )
        )
        await session.commit()

    async with session_factory() as session:
        records = (await session.execute(select(AgentRunRecord))).scalars().all()

    assert len(records) == 1
    assert records[0].knowledge_lookups_json == [
        {"section": "wabog_pricing", "reason": "Asked for price"}
    ]

    await engine.dispose()


@pytest.mark.asyncio
async def test_memory_store_bounds_long_intent_before_saving_run():
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    session_factory = build_session_factory(engine)
    memory = SqlAlchemyMemoryStore(session_factory)
    long_intent = (
        "Responder pregunta comercial sobre encaje del producto en el flujo actual y avanzar "
        "hacia un siguiente paso de diagnóstico/demostración ligera."
    )

    await memory.save_run(
        run_id="run-long-intent",
        event=InboundMessage(
            message_id="msg-long-intent",
            conversation_id="conv-long-intent",
            phone_number="+573001112233",
            text="Como encajaría en mi flujo actual",
            timestamp=datetime.now(timezone.utc),
            raw_payload={},
        ),
        intent=long_intent,
        response_text="Sí, encaja bastante bien con ese flujo.",
        tool_results=[],
        knowledge_lookups=[],
    )

    async with session_factory() as session:
        record = await session.get(AgentRunRecord, "run-long-intent")
        thread = await session.get(ConversationThreadRecord, "conv-long-intent")

    assert record is not None
    assert thread is not None
    assert len(record.intent) == 128
    assert record.intent == long_intent[:128]
    assert thread.last_intent == record.intent

    await engine.dispose()
