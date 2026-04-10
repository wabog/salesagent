from __future__ import annotations

from sales_agent.domain.models import ActionType, ProposedAction


class ToolExecutionPolicy:
    def validate(self, action: ProposedAction) -> None:
        if action.type == ActionType.UPDATE_STAGE:
            stage = action.args.get("stage", "").strip()
            if not stage:
                raise ValueError("Stage update requires a target stage.")
        if action.type == ActionType.APPEND_NOTE:
            note = action.args.get("note", "").strip()
            if not note:
                raise ValueError("Appending a note requires non-empty content.")
        if action.type == ActionType.CREATE_FOLLOWUP:
            summary = action.args.get("summary", "").strip()
            if not summary:
                raise ValueError("Follow-up creation requires a summary.")
