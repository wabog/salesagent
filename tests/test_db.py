from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from sales_agent.core.db import AgentRunRecord, build_engine, build_session_factory, init_db


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
