from sales_agent.adapters.google_calendar import GoogleCalendarAdapter
from sales_agent.core.config import Settings
from sales_agent.domain.models import CRMContact


def build_adapter() -> GoogleCalendarAdapter:
    return GoogleCalendarAdapter(
        Settings(
            OPENAI_API_KEY="",
            GOOGLE_CLIENT_ID="client-id",
            GOOGLE_CLIENT_SECRET="client-secret",
            GOOGLE_REFRESH_TOKEN="refresh-token",
        )
    )


def test_event_match_ignores_generic_playground_name():
    adapter = build_adapter()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Playground User",
        email=None,
    )
    event = {
        "summary": "Demo Wabog - Playground User",
        "description": "Lead: Playground User.",
        "attendees": [],
        "status": "confirmed",
    }

    assert adapter._event_matches_contact(event, contact) is False  # noqa: SLF001


def test_event_match_prefers_email_for_specific_contact():
    adapter = build_adapter()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name="Carlos Ruiz",
        email="carlos@example.com",
    )
    event = {
        "summary": "Demo Wabog - Otra Persona",
        "description": "Evento comercial.",
        "attendees": [{"email": "carlos@example.com"}],
        "status": "confirmed",
    }

    assert adapter._event_matches_contact(event, contact) is True  # noqa: SLF001


def test_event_match_does_not_match_only_on_name_when_phone_differs():
    adapter = build_adapter()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="+573150000001",
        full_name="Carlos Ruiz",
        email=None,
    )
    event = {
        "summary": "Demo Wabog - Carlos Ruiz",
        "description": "Demo comercial creada por el agente. Lead: Carlos Ruiz. Telefono: +573150000000.",
        "attendees": [],
        "status": "confirmed",
    }

    assert adapter._event_matches_contact(event, contact) is False  # noqa: SLF001
