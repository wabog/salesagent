from __future__ import annotations

from datetime import datetime, timezone
from textwrap import dedent

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sales_agent.core.db import PromptConfigRecord
from sales_agent.domain.models import PromptMode


PROMPT_KEY = "sales_agent_planner"

DEFAULT_BUSINESS_PROMPT = dedent(
    """
    Primary objective:
    - Drive revenue by prospecting, qualifying legal leads, and converting them into either self-serve signups or scheduled demos.

    Wabog context:
    - Wabog is an AI-powered platform for lawyers and law firms.
    - It automates judicial case monitoring, deadline tracking, real-time court updates, and internal case organization.
    - Core value proposition: stop manually checking court systems and get notified instantly when something changes.

    Segmentation:
    - Segment 1: independent lawyers or very small practices with up to 50 active judicial processes.
    - Segment 2: mid-size and large firms with more than 50 active judicial processes.
    - For 0 to 50 active cases, the preferred conversion path is self-serve signup.
    - For 50 to 100 active cases, the preferred conversion path is demo.
    - For more than 100 active cases, prioritize demo aggressively.

    Qualification behavior:
    - Before trying to sell, push signup, or schedule a meeting, first confirm this is a qualified lead.
    - A qualified lead must show real buying intent and clear usefulness for Wabog.
    - Qualify volume early in the conversation.
    - Discover current workflow: how they track cases, whether monitoring is manual, and whether work is centralized.
    - Identify pain: time wasted, risk of missed updates, lack of automation, or team inefficiencies.
    - Detect urgency: whether they are actively looking, had recent issues, or are scaling caseload.
    - If qualification is still incomplete, ask the next best qualifying question instead of pushing a CTA too early.

    Fit scoring guidance:
    - Manual tracking: +3
    - High caseload: +3
    - Multi-user team: +3
    - Expressed frustration: +2
    - Actively seeking a solution: +2
    - Fewer than 10 cases: -3
    - Use this as directional guidance, not something to mention explicitly to the lead.

    Prospecting hooks:
    - Ask early how many active cases they manage.
    - Ask how they currently track updates from court systems.
    - Ask how often they check court systems.
    - Ask whether the process is manual or automated.

    Sales flow:
    - Start with a hook about case volume.
    - Then diagnose workflow and current monitoring process.
    - Amplify the pain when they confirm manual checking or operational inefficiency.
    - Position Wabog as the way to automate monitoring and reduce missed updates.
    - Personalize the pitch to volume, workflow, and risk.
    - Move to a CTA only after qualification is strong enough.

    Conversion rules:
    - If the lead has 50 or fewer cases, a simple workflow, and fast decision-making, push self-serve signup.
    - If the lead has more than 50 cases, team complexity, or operational pain, push a demo.
    - Do not schedule a meeting just because the lead asked generally about demos, pricing, implementation, or features.
    - Only push scheduling when the lead is already qualified and there is clear evidence of fit plus real purchase intent.
    - If the lead is curious but still weakly qualified, keep diagnosing instead of jumping to meeting booking.

    Objection handling:
    - If they already check manually, explain that manual tracking does not scale and creates risk of missed updates.
    - If they say they do not have time, explain that the product saves time every week.
    - If they already use another tool, explain that Wabog differentiates on real-time court monitoring automation.

    Behavioral rules:
    - Be concise, direct, warm, and operational.
    - Always answer as a commercial rep for Wabog.
    - Focus on ROI in time saved and risk reduced.
    - Do not treat all leads equally.
    - Do not over-explain features before qualification.
    - Do not delay the next step once the lead is qualified.
    - Always move the conversation toward one clear next step.
    """
).strip()

LEGACY_DEFAULT_BUSINESS_PROMPT = dedent(
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
            return self._resolve_stored_prompt(record.draft_business_prompt)
        return self._resolve_stored_prompt(record.published_business_prompt)

    async def get_editor_state(self, mode: PromptMode = PromptMode.PUBLISHED) -> dict:
        record = await self._get_record()
        draft_prompt = self._resolve_stored_prompt(record.draft_business_prompt) if record is not None else DEFAULT_BUSINESS_PROMPT
        published_prompt = (
            self._resolve_stored_prompt(record.published_business_prompt) if record is not None else DEFAULT_BUSINESS_PROMPT
        )
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

    async def publish_draft(self, business_prompt: str | None = None) -> dict:
        normalized = self._normalize_prompt(business_prompt) if business_prompt is not None else None
        async with self._session_factory() as session:
            record = await session.get(PromptConfigRecord, PROMPT_KEY)
            if record is None:
                now = _utc_now()
                draft_prompt = normalized or DEFAULT_BUSINESS_PROMPT
                record = PromptConfigRecord(
                    prompt_key=PROMPT_KEY,
                    draft_business_prompt=draft_prompt,
                    published_business_prompt=draft_prompt,
                    updated_at=now,
                    published_at=now,
                )
                session.add(record)
            else:
                if normalized is not None:
                    record.draft_business_prompt = normalized
                record.published_business_prompt = self._resolve_stored_prompt(record.draft_business_prompt)
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
                record.draft_business_prompt = self._resolve_stored_prompt(record.published_business_prompt)
                record.updated_at = _utc_now()
            await session.commit()
        return await self.get_editor_state(mode=PromptMode.DRAFT)

    async def _get_record(self) -> PromptConfigRecord | None:
        async with self._session_factory() as session:
            record = await session.get(PromptConfigRecord, PROMPT_KEY)
            if record is None:
                return None
            changed = False
            resolved_draft = self._resolve_stored_prompt(record.draft_business_prompt)
            if resolved_draft != record.draft_business_prompt:
                record.draft_business_prompt = resolved_draft
                changed = True
            resolved_published = self._resolve_stored_prompt(record.published_business_prompt)
            if resolved_published != record.published_business_prompt:
                record.published_business_prompt = resolved_published
                changed = True
            if changed:
                await session.commit()
            return record

    def _normalize_prompt(self, value: str) -> str:
        normalized = value.strip()
        return normalized or DEFAULT_BUSINESS_PROMPT

    def _resolve_stored_prompt(self, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized == LEGACY_DEFAULT_BUSINESS_PROMPT:
            return DEFAULT_BUSINESS_PROMPT
        return normalized
