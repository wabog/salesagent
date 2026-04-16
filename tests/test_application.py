from sales_agent.core.config import Settings
from sales_agent.domain.models import ActionType, CRMContact, ProposedAction, ToolExecutionResult
from sales_agent.services.application import SalesAgentApplication


def build_application() -> SalesAgentApplication:
    return SalesAgentApplication(
        Settings(
            DATABASE_URL="sqlite+aiosqlite:///:memory:",
            OPENAI_API_KEY="",
            CRM_BACKEND="memory",
            GOOGLE_CALENDAR_SELF_SCHEDULE_URL="https://calendar.app.google/9KpgMYTW4nA17LV27",
        )
    )


def test_finalize_response_text_degrades_failed_meeting_when_contact_data_missing():
    app = build_application()
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name=None,
        email=None,
    )
    tool_results = [
        ToolExecutionResult(
            action=ProposedAction(
                type=ActionType.CREATE_MEETING,
                reason="Intentar crear demo.",
                args={},
            ),
            success=False,
            error="Meeting creation requires start_iso.",
        )
    ]

    final_text = app._finalize_response_text(  # noqa: SLF001
        "Perfecto, agendamos la demo y te envío la invitación.",
        tool_results,
        contact,
    )

    assert "todavía no la dejo agendada" in final_text.lower()
    assert "nombre completo" in final_text.lower()
    assert "correo" in final_text.lower()
