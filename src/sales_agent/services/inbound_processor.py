from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from sales_agent.domain.models import InboundMessage, InboundProcessingResult


PrepareBatch = Callable[[list[InboundMessage]], Awaitable[object]]
CommitBatch = Callable[[object], Awaitable[InboundProcessingResult]]


@dataclass
class PendingConversationState:
    events: list[InboundMessage] = field(default_factory=list)
    waiters: list[tuple[str, asyncio.Future[InboundProcessingResult]]] = field(default_factory=list)
    timer_task: asyncio.Task | None = None
    processing: bool = False
    retry_requested: bool = False
    version: int = 0


class DebouncedInboundProcessor:
    def __init__(
        self,
        *,
        debounce_seconds: float,
        prepare_batch: PrepareBatch,
        commit_batch: CommitBatch,
    ) -> None:
        self._debounce_seconds = debounce_seconds
        self._prepare_batch = prepare_batch
        self._commit_batch = commit_batch
        self._states: dict[str, PendingConversationState] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def submit(self, event: InboundMessage, *, wait_for_result: bool) -> InboundProcessingResult:
        if self._closed:
            raise RuntimeError("Inbound processor is closed.")

        future: asyncio.Future[InboundProcessingResult] | None = None
        async with self._lock:
            state = self._states.setdefault(event.conversation_id, PendingConversationState())
            state.events.append(event)
            state.version += 1
            generation = state.version
            if wait_for_result:
                future = asyncio.get_running_loop().create_future()
                state.waiters.append((event.message_id, future))
            self._schedule_timer_locked(event.conversation_id, generation, self._debounce_seconds)

        if future is None:
            return InboundProcessingResult(queued=True, render_reply=False)
        return await future

    async def shutdown(self) -> None:
        async with self._lock:
            self._closed = True
            states = list(self._states.values())
            self._states.clear()

        for state in states:
            if state.timer_task is not None:
                state.timer_task.cancel()
            for _, future in state.waiters:
                if not future.done():
                    future.cancel()

    def _schedule_timer_locked(self, conversation_id: str, generation: int, delay: float) -> None:
        state = self._states[conversation_id]
        if state.timer_task is not None and not state.timer_task.done():
            state.timer_task.cancel()
        state.timer_task = asyncio.create_task(self._debounce_then_process(conversation_id, generation, delay))

    async def _debounce_then_process(self, conversation_id: str, generation: int, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            await self._process_if_ready(conversation_id, generation)
        except asyncio.CancelledError:
            return

    async def _process_if_ready(self, conversation_id: str, generation: int) -> None:
        async with self._lock:
            state = self._states.get(conversation_id)
            if state is None or state.version != generation:
                return
            if state.processing:
                state.retry_requested = True
                return
            if not state.events:
                self._cleanup_locked(conversation_id)
                return

            state.processing = True
            events_snapshot = list(state.events)
            waiters_snapshot = list(state.waiters)
            state.events.clear()
            state.waiters.clear()

        try:
            prepared = await self._prepare_batch(events_snapshot)

            async with self._lock:
                state = self._states.get(conversation_id)
                if state is None:
                    return
                stale = state.version != generation or bool(state.events)
                if stale:
                    state.events = events_snapshot + state.events
                    state.waiters = waiters_snapshot + state.waiters
                    return

            result = await self._commit_batch(prepared)
            latest_message_id = events_snapshot[-1].message_id
            for message_id, future in waiters_snapshot:
                if future.done():
                    continue
                payload = result.model_copy(
                    update={
                        "render_reply": message_id == latest_message_id,
                        "aggregated_messages": len(events_snapshot),
                    }
                )
                if not payload.render_reply:
                    payload = payload.model_copy(update={"response_text": "", "tool_results": []})
                future.set_result(payload)
        except Exception as exc:  # noqa: BLE001
            for _, future in waiters_snapshot:
                if not future.done():
                    future.set_exception(exc)
        finally:
            async with self._lock:
                state = self._states.get(conversation_id)
                if state is None:
                    return
                state.processing = False
                if state.events and (state.retry_requested or state.timer_task is None or state.timer_task.done()):
                    state.retry_requested = False
                    self._schedule_timer_locked(conversation_id, state.version, 0.0)
                else:
                    state.retry_requested = False
                    self._cleanup_locked(conversation_id)

    def _cleanup_locked(self, conversation_id: str) -> None:
        state = self._states.get(conversation_id)
        if state is None:
            return
        if state.processing or state.events or state.waiters:
            return
        if state.timer_task is not None and not state.timer_task.done():
            return
        self._states.pop(conversation_id, None)
