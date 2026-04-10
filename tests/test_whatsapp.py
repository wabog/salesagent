import json
from pathlib import Path

from sales_agent.adapters.whatsapp import normalize_kapso_payload


def test_normalize_kapso_payload():
    payload = json.loads(Path("fixtures/kapso_webhook.json").read_text())
    event = normalize_kapso_payload(payload)

    assert event.phone_number == "573016803866"
    assert event.conversation_id == "52db4929-5965-45d1-bb4a-b225ec6ad8b8"
    assert event.text == "hola puedes revisar notion dime que personas hay en el pipeline"
    assert event.provider == "kapso"
