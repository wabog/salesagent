from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, String, Text, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class MessageRecord(Base):
    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(255), index=True)
    phone_number: Mapped[str] = mapped_column(String(64), index=True)
    direction: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ConversationThreadRecord(Base):
    __tablename__ = "conversation_threads"

    conversation_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    phone_number: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[str | None] = mapped_column(Text(), nullable=True)
    last_intent: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class AgentRunRecord(Base):
    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(255), index=True)
    message_id: Mapped[str] = mapped_column(String(255), index=True)
    phone_number: Mapped[str] = mapped_column(String(64), index=True)
    intent: Mapped[str] = mapped_column(String(128))
    response_text: Mapped[str] = mapped_column(Text())
    tool_results_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    knowledge_lookups_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)


class ContactShadowRecord(Base):
    __tablename__ = "contacts_shadow"

    phone_number: Mapped[str] = mapped_column(String(64), primary_key=True)
    external_id: Mapped[str] = mapped_column(String(255), index=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stage: Mapped[str | None] = mapped_column(String(128), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


def build_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, future=True)


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _run_additive_migrations(conn)


async def _run_additive_migrations(conn) -> None:
    columns = await _get_table_columns(conn, "agent_runs")
    if "knowledge_lookups_json" in columns:
        return

    if conn.dialect.name == "postgresql":
        await conn.exec_driver_sql(
            "ALTER TABLE agent_runs "
            "ADD COLUMN IF NOT EXISTS knowledge_lookups_json JSON NOT NULL DEFAULT '[]'"
        )
        return

    await conn.exec_driver_sql(
        "ALTER TABLE agent_runs "
        "ADD COLUMN knowledge_lookups_json JSON NOT NULL DEFAULT '[]'"
    )


async def _get_table_columns(conn, table_name: str) -> set[str]:
    if conn.dialect.name == "sqlite":
        return {
            row[1]
            for row in (
                await conn.exec_driver_sql(f"PRAGMA table_info({table_name})")
            ).fetchall()
        }

    if conn.dialect.name == "postgresql":
        result = await conn.execute(
            text(
                "SELECT column_name "
                "FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = :table_name"
            ),
            {"table_name": table_name},
        )
        return {row[0] for row in result.fetchall()}

    return set()


async def message_exists(session: AsyncSession, message_id: str) -> bool:
    result = await session.execute(select(MessageRecord.message_id).where(MessageRecord.message_id == message_id))
    return result.scalar_one_or_none() is not None
