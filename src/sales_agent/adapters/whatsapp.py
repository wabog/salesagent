from __future__ import annotations

from datetime import datetime, timezone

from sales_agent.domain.models import InboundMessage


class KapsoPayloadError(ValueError):
    pass


def normalize_kapso_payload(payload: dict) -> InboundMessage:
    try:
        message = payload["body"]["message"]
        conversation = payload["body"]["conversation"]
        timestamp_raw = message["timestamp"]
        text = message.get("kapso", {}).get("content") or message.get("text", {}).get("body") or ""
        timestamp = datetime.fromtimestamp(int(timestamp_raw), tz=timezone.utc)
        return InboundMessage(
            message_id=message["id"],
            conversation_id=conversation["id"],
            phone_number=conversation["phone_number"],
            text=text.strip(),
            timestamp=timestamp,
            raw_payload=payload,
            contact_name=conversation.get("contact_name"),
        )
    except KeyError as exc:
        raise KapsoPayloadError(f"Kapso payload missing field: {exc}") from exc
