import json
from pathlib import Path

import pytest

from sales_agent.adapters.whatsapp import KapsoPayloadError, normalize_kapso_payload


def test_normalize_kapso_payload():
    payload = json.loads(Path("fixtures/kapso_webhook.json").read_text())
    event = normalize_kapso_payload(payload)

    assert event.phone_number == "+573016803866"
    assert event.conversation_id == "52db4929-5965-45d1-bb4a-b225ec6ad8b8"
    assert event.text == "hola puedes revisar notion dime que personas hay en el pipeline"
    assert event.provider == "kapso"


def test_normalize_kapso_payload_from_data_wrapper():
    payload = {
        "event": "message.received",
        "data": {
            "message": {
                "id": "msg-123",
                "from": "573001112233",
                "timestamp": "1775704462",
                "content": "hola desde data",
            },
            "conversation": {
                "id": "conv-123",
                "phone_number": "573001112233",
                "contact_name": "Fabian",
            },
        },
    }

    event = normalize_kapso_payload(payload)

    assert event.message_id == "msg-123"
    assert event.conversation_id == "conv-123"
    assert event.phone_number == "+573001112233"
    assert event.text == "hola desde data"
    assert event.contact_name == "Fabian"


def test_normalize_kapso_payload_audio_extracts_media_metadata():
    payload = {
        "event": "message.received",
        "body": {
            "message": {
                "id": "msg-audio",
                "from": "573001112233",
                "timestamp": "1775704462",
                "type": "audio",
                "kapso": {
                    "content": "[Audio attached]",
                    "media_url": "https://api.kapso.ai/media/audio-1",
                    "transcript": {"text": "Necesito una demo"},
                },
                "audio": {"mime_type": "audio/ogg"},
            },
            "conversation": {
                "id": "conv-audio",
                "phone_number": "573001112233",
            },
        },
    }

    event = normalize_kapso_payload(payload)

    assert event.message_type == "audio"
    assert event.media_url == "https://api.kapso.ai/media/audio-1"
    assert event.media_content_type == "audio/ogg"
    assert event.media_transcript == "Necesito una demo"


def test_normalize_kapso_payload_rejects_unsupported_payload():
    payload = {
        "event": "message.status",
        "data": {
            "message": {
                "id": "msg-status",
                "timestamp": "1775704462",
            }
        },
    }

    with pytest.raises(KapsoPayloadError, match="supported inbound content"):
        normalize_kapso_payload(payload)
