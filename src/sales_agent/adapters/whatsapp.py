from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sales_agent.domain.models import InboundMessage


class KapsoPayloadError(ValueError):
    pass


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _pick_first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _pick_message(container: dict[str, Any]) -> dict[str, Any]:
    direct = _as_dict(container.get("message"))
    if direct:
        return direct
    messages = container.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict):
                return item
    return {}


def _pick_conversation(container: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    direct = _pick_first_dict(container.get("conversation"), container.get("chat"), container.get("contact"))
    if direct:
        return direct
    return _pick_first_dict(message.get("conversation"), message.get("chat"), message.get("contact"))


def _parse_timestamp(raw: Any) -> datetime:
    if raw is None:
        return datetime.now(timezone.utc)
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return datetime.now(timezone.utc)
        if value.isdigit():
            return datetime.fromtimestamp(int(value), tz=timezone.utc)
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


def _extract_text(message: dict[str, Any], conversation: dict[str, Any]) -> str:
    text = (
        _as_dict(message.get("kapso")).get("content")
        or _as_dict(message.get("text")).get("body")
        or message.get("content")
        or message.get("body")
        or _as_dict(conversation.get("last_message")).get("content")
        or ""
    )
    return str(text).strip()


def _extract_media_url(message: dict[str, Any], message_type: str) -> str | None:
    kapso = _as_dict(message.get("kapso"))
    media_data = _pick_first_dict(kapso.get("media_data"), kapso.get("message_type_data"))
    type_payload = _as_dict(message.get(message_type))
    value = (
        kapso.get("media_url")
        or media_data.get("download_url")
        or media_data.get("url")
        or type_payload.get("link")
        or type_payload.get("url")
        or message.get("media_url")
    )
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _extract_media_filename(message: dict[str, Any], message_type: str) -> str | None:
    kapso = _as_dict(message.get("kapso"))
    media_data = _pick_first_dict(kapso.get("media_data"), kapso.get("message_type_data"))
    type_payload = _as_dict(message.get(message_type))
    value = type_payload.get("filename") or media_data.get("filename") or media_data.get("name")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _extract_media_content_type(message: dict[str, Any], message_type: str) -> str | None:
    kapso = _as_dict(message.get("kapso"))
    media_data = _pick_first_dict(kapso.get("media_data"), kapso.get("message_type_data"))
    type_payload = _as_dict(message.get(message_type))
    value = type_payload.get("mime_type") or media_data.get("mime_type") or media_data.get("content_type")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _extract_media_caption(message: dict[str, Any], message_type: str) -> str | None:
    kapso = _as_dict(message.get("kapso"))
    type_payload = _as_dict(message.get(message_type))
    value = type_payload.get("caption") or kapso.get("caption")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _extract_transcript(message: dict[str, Any]) -> str | None:
    kapso = _as_dict(message.get("kapso"))
    transcript = kapso.get("transcript")
    if isinstance(transcript, dict):
        value = transcript.get("text")
    else:
        value = transcript
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_kapso_payload(payload: dict) -> InboundMessage:
    body = _as_dict(payload.get("body"))
    data = _as_dict(payload.get("data"))
    root_message = _pick_message(payload)
    root_conversation = _pick_first_dict(payload.get("conversation"), payload.get("chat"), payload.get("contact"))

    message = _pick_first_dict(
        _pick_message(body),
        _pick_message(data),
        root_message,
    )
    conversation = _pick_first_dict(
        _pick_conversation(body, message),
        _pick_conversation(data, message),
        root_conversation,
    )

    if not message:
        raise KapsoPayloadError("Kapso payload does not contain a supported message object.")

    message_type = str(message.get("type") or _as_dict(message.get("kapso")).get("message_type") or "text").strip().lower()
    text = _extract_text(message, conversation)
    media_url = _extract_media_url(message, message_type)
    media_filename = _extract_media_filename(message, message_type)
    media_content_type = _extract_media_content_type(message, message_type)
    media_caption = _extract_media_caption(message, message_type)
    media_transcript = _extract_transcript(message)
    if not text and not any((media_url, media_caption, media_transcript)) and message_type == "text":
        raise KapsoPayloadError("Kapso payload does not contain supported inbound content.")
    if not text and message_type not in {"audio", "image", "document", "sticker", "video"}:
        raise KapsoPayloadError("Kapso payload does not contain supported inbound content.")

    message_id = (
        message.get("id")
        or message.get("message_id")
        or message.get("wamid")
    )
    if not message_id:
        raise KapsoPayloadError("Kapso payload missing field: 'message.id'")

    conversation_id = (
        conversation.get("id")
        or message.get("conversation_id")
        or message.get("chat_id")
        or message.get("from")
    )
    if not conversation_id:
        raise KapsoPayloadError("Kapso payload missing field: 'conversation.id'")

    phone_number = (
        conversation.get("phone_number")
        or conversation.get("wa_id")
        or conversation.get("phone")
        or message.get("from")
        or _as_dict(message.get("from")).get("phone_number")
    )
    if not phone_number:
        raise KapsoPayloadError("Kapso payload missing field: 'conversation.phone_number'")

    timestamp = _parse_timestamp(message.get("timestamp") or payload.get("timestamp"))

    return InboundMessage(
        message_id=str(message_id),
        conversation_id=str(conversation_id),
        phone_number=str(phone_number),
        text=text,
        timestamp=timestamp,
        raw_payload=payload,
        message_type=message_type,
        media_url=media_url,
        media_content_type=media_content_type,
        media_filename=media_filename,
        media_caption=media_caption,
        media_transcript=media_transcript,
        contact_name=conversation.get("contact_name") or conversation.get("name"),
    )
