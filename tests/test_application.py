from datetime import date, datetime, timezone

import pytest

from sales_agent.core.config import Settings
from sales_agent.domain.models import (
    ActionType,
    CRMContact,
    InboundMessage,
    PlanningResult,
    ProposedAction,
    ToolExecutionResult,
)
from sales_agent.services.application import PreparedBatchRun, SalesAgentApplication


class _StubCalendarAdapter:
    async def create_meeting(self, contact, *, start_iso, duration_minutes, title, description):  # noqa: ANN001
        return {
            "id": "evt-1",
            "summary": title,
            "description": description,
            "start_iso": start_iso,
            "end_iso": start_iso,
            "html_link": "https://calendar.google.com/event?eid=evt-1",
            "meet_link": "https://meet.google.com/test-meet",
            "status": "confirmed",
            "attendees": [contact.email] if contact.email else [],
            "source": "agent_created",
        }


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


def test_finalize_response_text_normalizes_unpublished_wabog_signup_route():
    app = build_application()

    final_text = app._finalize_response_text(  # noqa: SLF001
        "Te dejo el link para probarlo: https://wabog.com/signup",
        [],
        None,
    )

    assert "https://wabog.com/signup" not in final_text
    assert "https://wabog.com" in final_text


@pytest.mark.asyncio
async def test_commit_batched_run_syncs_crm_after_meeting_creation():
    app = build_application()
    app.calendar_adapter = _StubCalendarAdapter()
    await app.startup()
    try:
        created = await app.crm_adapter.create_contact("573001111222", "Paula Diaz")
        current_lead = await app.crm_adapter.update_contact_fields(
            created.external_id,
            {"email": "paula@example.com"},
        )
        event = InboundMessage(
            message_id="meeting-app-1",
            conversation_id="conv-app-meeting",
            phone_number="573001111222",
            text="mañana a las 10 am me sirve",
            timestamp=datetime.now(timezone.utc),
            raw_payload={},
            provider="local-playground",
        )
        prepared = PreparedBatchRun(
            run_id="run-meeting-app-1",
            events=[event],
            current_lead=current_lead,
            lead_created=False,
            recent_messages=[],
            response_text="Perfecto. Procedo a agendarla.",
            planning=PlanningResult(
                intent="schedule_demo",
                confidence=0.9,
                response_text="Perfecto. Procedo a agendarla.",
                actions=[
                    ProposedAction(
                        type=ActionType.CREATE_MEETING,
                        reason="Lead confirmó horario.",
                        args={
                            "start_iso": "2026-05-02T10:00:00-05:00",
                            "duration_minutes": 30,
                            "title": "Demo Wabog - Paula Diaz",
                            "description": "Demo comercial creada por el agente de ventas de Wabog.",
                        },
                    )
                ],
            ),
        )

        result = await app.commit_batched_run(prepared)
    finally:
        await app.shutdown()

    assert result.contact is not None
    assert result.contact.stage == "Demo agendada"
    assert result.contact.followup_summary is not None
    assert result.contact.followup_due_date == date(2026, 5, 2)
    assert any("Demo agendada para 2026-05-02T10:00:00-05:00." in note for note in result.contact.notes)
    assert any(item.action.type == ActionType.CREATE_MEETING and item.success for item in result.tool_results)
