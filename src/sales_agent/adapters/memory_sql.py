from __future__ import annotations

from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from sales_agent.core.db import AgentRunRecord, ContactShadowRecord, ConversationThreadRecord, MessageRecord
from sales_agent.domain.models import CRMContact, ConversationMessage, InboundMessage, ToolExecutionResult


class SqlAlchemyMemoryStore:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._session_factory = session_factory

    async def has_message(self, message_id: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                select(MessageRecord.message_id).where(MessageRecord.message_id == message_id)
            )
            return result.scalar_one_or_none() is not None

    async def append_message(self, message: ConversationMessage) -> None:
        for _ in range(2):
            async with self._session_factory() as session:
                session.add(
                    MessageRecord(
                        message_id=message.message_id,
                        conversation_id=message.conversation_id,
                        phone_number=message.phone_number,
                        direction=message.direction.value,
                        text=message.text,
                        created_at=message.created_at,
                        metadata_json=message.metadata,
                    )
                )
                thread = await session.get(ConversationThreadRecord, message.conversation_id)
                if thread is None:
                    thread = ConversationThreadRecord(
                        conversation_id=message.conversation_id,
                        phone_number=message.phone_number,
                    )
                    session.add(thread)
                try:
                    await session.commit()
                    return
                except IntegrityError:
                    await session.rollback()
        raise RuntimeError("Failed to append message after retry.")

    async def get_recent_messages(self, conversation_id: str, limit: int = 8) -> list[ConversationMessage]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == conversation_id)
                .order_by(desc(MessageRecord.created_at))
                .limit(limit)
            )
            items = list(result.scalars())
            items.reverse()
            return [
                ConversationMessage(
                    message_id=item.message_id,
                    conversation_id=item.conversation_id,
                    phone_number=item.phone_number,
                    direction=item.direction,
                    text=item.text,
                    created_at=item.created_at,
                    metadata=item.metadata_json,
                )
                for item in items
            ]

    async def remember_contact(self, contact: CRMContact) -> None:
        async with self._session_factory() as session:
            existing = await session.get(ContactShadowRecord, contact.phone_number)
            payload = {
                "phone_number": contact.phone_number,
                "external_id": contact.external_id,
                "full_name": contact.full_name,
                "stage": contact.stage,
                "email": contact.email,
                "notes_json": contact.notes,
                "metadata_json": contact.metadata,
            }
            if existing is None:
                session.add(ContactShadowRecord(**payload))
            else:
                for key, value in payload.items():
                    setattr(existing, key, value)
            await session.commit()

    async def get_contact_shadow(self, phone_number: str) -> CRMContact | None:
        async with self._session_factory() as session:
            record = await session.get(ContactShadowRecord, phone_number)
            if record is None:
                return None
            return CRMContact(
                external_id=record.external_id,
                phone_number=record.phone_number,
                full_name=record.full_name,
                stage=record.stage,
                email=record.email,
                notes=record.notes_json,
                metadata=record.metadata_json,
            )

    async def search_memories(self, conversation_id: str, query: str, limit: int = 3) -> list[str]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        messages = await self.get_recent_messages(conversation_id, limit=20)
        scored: list[tuple[int, str]] = []
        for item in messages:
            text = item.text.strip()
            if not text:
                continue
            score = sum(1 for term in query_terms if term in text.lower())
            if score:
                scored.append((score, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:limit]]

    async def save_run(
        self,
        run_id: str,
        event: InboundMessage,
        intent: str,
        response_text: str,
        tool_results: list[ToolExecutionResult],
    ) -> None:
        for _ in range(2):
            async with self._session_factory() as session:
                session.add(
                    AgentRunRecord(
                        run_id=run_id,
                        conversation_id=event.conversation_id,
                        message_id=event.message_id,
                        phone_number=event.phone_number,
                        intent=intent,
                        response_text=response_text,
                        tool_results_json=[result.model_dump(mode="json") for result in tool_results],
                    )
                )
                thread = await session.get(ConversationThreadRecord, event.conversation_id)
                if thread is None:
                    thread = ConversationThreadRecord(
                        conversation_id=event.conversation_id,
                        phone_number=event.phone_number,
                    )
                    session.add(thread)
                thread.last_intent = intent
                try:
                    await session.commit()
                    return
                except IntegrityError:
                    await session.rollback()
        raise RuntimeError("Failed to save run after retry.")
