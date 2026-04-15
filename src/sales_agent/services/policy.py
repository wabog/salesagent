from __future__ import annotations

import re

from sales_agent.domain.models import ActionType, ProposedAction


class ToolExecutionPolicy:
    def validate(self, action: ProposedAction) -> None:
        if action.type == ActionType.UPDATE_STAGE:
            stage = action.args.get("stage", "").strip()
            if not stage:
                raise ValueError("Stage update requires a target stage.")
        if action.type == ActionType.UPDATE_CONTACT_FIELDS:
            fields = action.args.get("fields")
            if not isinstance(fields, dict) or not fields:
                raise ValueError("Contact update requires a non-empty fields object.")
            allowed_fields = {"full_name", "email"}
            sanitized = {key: str(value).strip() for key, value in fields.items() if key in allowed_fields and str(value).strip()}
            if not sanitized:
                raise ValueError("Contact update requires at least one supported field.")
            email = sanitized.get("email")
            if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                raise ValueError("Contact update email must be valid.")
        if action.type == ActionType.APPEND_NOTE:
            note = action.args.get("note", "").strip()
            if not note:
                raise ValueError("Appending a note requires non-empty content.")
        if action.type == ActionType.CREATE_FOLLOWUP:
            summary = action.args.get("summary", "").strip()
            if not summary:
                raise ValueError("Follow-up creation requires a summary.")
            due_date = str(action.args.get("due_date", "")).strip()
            if due_date:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", due_date):
                    raise ValueError("Follow-up due_date must use YYYY-MM-DD format.")
        if action.type == ActionType.COMPLETE_FOLLOWUP:
            outcome = str(action.args.get("outcome", "")).strip()
            if outcome and len(outcome) > 500:
                raise ValueError("Follow-up completion outcome is too long.")
