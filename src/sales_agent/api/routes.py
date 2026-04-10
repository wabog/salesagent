from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from sales_agent.adapters.whatsapp import normalize_kapso_payload
from sales_agent.domain.models import InboundMessage


router = APIRouter()


class ReplayPayload(BaseModel):
    event: InboundMessage | None = None
    webhook_payload: dict | None = None


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.post("/webhooks/whatsapp/kapso")
async def kapso_webhook(request: Request) -> dict:
    payload = await request.json()
    event = normalize_kapso_payload(payload)
    result = await request.app.state.sales_agent.workflow.run(event)
    return result.model_dump(mode="json")


@router.post("/internal/replay")
async def replay(payload: ReplayPayload, request: Request) -> dict:
    event = payload.event
    if event is None:
        event = normalize_kapso_payload(payload.webhook_payload or {})
    result = await request.app.state.sales_agent.workflow.run(event)
    return result.model_dump(mode="json")
