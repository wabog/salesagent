from __future__ import annotations

from datetime import datetime, timezone
from textwrap import dedent

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sales_agent.core.db import PromptConfigRecord
from sales_agent.domain.models import PromptMode


PROMPT_KEY = "sales_agent_planner"

DEFAULT_BUSINESS_PROMPT = dedent(
    """
    - Wabog sells software and operational tooling for lawyers, law firms, and legal teams.
    - Be concise, warm, and operational.
    - Always answer as a commercial rep for Wabog.
    - If the user shows interest, qualify pain, process volume, current workflow, and buying context.
    - If the user asks for price, demo, trial, implementation, integrations, or wants to know how it works,
      move the lead forward in the pipeline when justified.
    """
).strip()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PromptConfigStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_business_prompt(self, mode: PromptMode = PromptMode.PUBLISHED) -> str:
        record = await self._get_record()
        if record is None:
            return DEFAULT_BUSINESS_PROMPT
        if mode == PromptMode.DRAFT:
            return record.draft_business_prompt
        return record.published_business_prompt

    async def get_editor_state(self, mode: PromptMode = PromptMode.PUBLISHED) -> dict:
        record = await self._get_record()
        draft_prompt = record.draft_business_prompt if record is not None else DEFAULT_BUSINESS_PROMPT
        published_prompt = record.published_business_prompt if record is not None else DEFAULT_BUSINESS_PROMPT
        return {
            "prompt_key": PROMPT_KEY,
            "draft_business_prompt": draft_prompt,
            "published_business_prompt": published_prompt,
            "active_business_prompt": draft_prompt if mode == PromptMode.DRAFT else published_prompt,
            "active_mode": mode,
            "updated_at": record.updated_at if record is not None else None,
            "published_at": record.published_at if record is not None else None,
        }

    async def save_draft(self, business_prompt: str) -> dict:
        normalized = self._normalize_prompt(business_prompt)
        async with self._session_factory() as session:
            record = await session.get(PromptConfigRecord, PROMPT_KEY)
            if record is None:
                record = PromptConfigRecord(
                    prompt_key=PROMPT_KEY,
                    draft_business_prompt=normalized,
                    published_business_prompt=DEFAULT_BUSINESS_PROMPT,
                    updated_at=_utc_now(),
                )
                session.add(record)
            else:
                record.draft_business_prompt = normalized
                record.updated_at = _utc_now()
            await session.commit()
        return await self.get_editor_state(mode=PromptMode.DRAFT)

    async def publish_draft(self) -> dict:
        async with self._session_factory() as session:
            record = await session.get(PromptConfigRecord, PROMPT_KEY)
            if record is None:
                now = _utc_now()
                record = PromptConfigRecord(
                    prompt_key=PROMPT_KEY,
                    draft_business_prompt=DEFAULT_BUSINESS_PROMPT,
                    published_business_prompt=DEFAULT_BUSINESS_PROMPT,
                    updated_at=now,
                    published_at=now,
                )
                session.add(record)
            else:
                record.published_business_prompt = record.draft_business_prompt
                record.updated_at = _utc_now()
                record.published_at = _utc_now()
            await session.commit()
        return await self.get_editor_state(mode=PromptMode.PUBLISHED)

    async def reset_draft(self) -> dict:
        async with self._session_factory() as session:
            record = await session.get(PromptConfigRecord, PROMPT_KEY)
            if record is None:
                now = _utc_now()
                record = PromptConfigRecord(
                    prompt_key=PROMPT_KEY,
                    draft_business_prompt=DEFAULT_BUSINESS_PROMPT,
                    published_business_prompt=DEFAULT_BUSINESS_PROMPT,
                    updated_at=now,
                    published_at=now,
                )
                session.add(record)
            else:
                record.draft_business_prompt = record.published_business_prompt
                record.updated_at = _utc_now()
            await session.commit()
        return await self.get_editor_state(mode=PromptMode.DRAFT)

    async def _get_record(self) -> PromptConfigRecord | None:
        async with self._session_factory() as session:
            return await session.get(PromptConfigRecord, PROMPT_KEY)

    def _normalize_prompt(self, value: str) -> str:
        normalized = value.strip()
        return normalized or DEFAULT_BUSINESS_PROMPT
