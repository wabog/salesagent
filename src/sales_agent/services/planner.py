from __future__ import annotations

import re
from textwrap import dedent

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from sales_agent.core.config import Settings
from sales_agent.domain.models import CRMContact, PlanningResult, ProposedAction
from sales_agent.domain.models import ActionType


class PlannerOutput(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    response_text: str
    should_reply: bool = True
    actions: list[ProposedAction] = Field(default_factory=list)


class AgentPlanner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = None
        if settings.openai_api_key:
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=0.2,
            ).with_structured_output(PlannerOutput)

    async def plan(
        self,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str],
        semantic_memories: list[str],
    ) -> PlanningResult:
        if self._llm is not None:
            return await self._plan_with_llm(text, contact, recent_messages, semantic_memories)
        return self._plan_with_rules(text, contact)

    async def _plan_with_llm(
        self,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str],
        semantic_memories: list[str],
    ) -> PlanningResult:
        prompt = dedent(
            f"""
            You are a sales copilot for inbound WhatsApp conversations.
            Decide the user intent, whether to reply, and which CRM actions to take.

            Contact:
            {contact.model_dump_json(indent=2) if contact else "None"}

            Recent messages:
            {recent_messages}

            Semantic memories:
            {semantic_memories}

            User message:
            {text}
            """
        ).strip()
        output = await self._llm.ainvoke(prompt)
        return PlanningResult(**output.model_dump())

    def _plan_with_rules(self, text: str, contact: CRMContact | None) -> PlanningResult:
        lowered = text.lower()
        actions: list[ProposedAction] = []
        response_lines: list[str] = []
        intent = "generic_reply"
        confidence = 0.55

        if contact is None:
            actions.append(
                ProposedAction(
                    type=ActionType.CREATE_CONTACT,
                    reason="No existe lead previo para este número.",
                    args={},
                )
            )
            response_lines.append("Ya tomé tu contacto en el pipeline para seguir la conversación.")

        if match := re.search(r"etapa(?:\s+a)?\s+([a-zA-ZáéíóúÁÉÍÓÚ ]+)", lowered):
            stage = match.group(1).strip().title()
            actions.append(
                ProposedAction(
                    type=ActionType.UPDATE_STAGE,
                    reason="El mensaje pide un cambio de etapa.",
                    args={"stage": stage},
                )
            )
            intent = "update_stage"
            confidence = 0.9
            response_lines.append(f"Actualicé la etapa del lead a {stage}.")

        if "nota:" in lowered:
            note = text.split("nota:", 1)[1].strip()
            actions.append(
                ProposedAction(
                    type=ActionType.APPEND_NOTE,
                    reason="El mensaje contiene una nota explícita.",
                    args={"note": note},
                )
            )
            intent = "append_note"
            confidence = 0.9
            response_lines.append("Registré la nota en el CRM.")

        if any(token in lowered for token in ("seguimiento", "follow up", "follow-up", "recordarme")):
            actions.append(
                ProposedAction(
                    type=ActionType.CREATE_FOLLOWUP,
                    reason="El mensaje sugiere programar seguimiento.",
                    args={"summary": text.strip()},
                )
            )
            intent = "followup"
            confidence = max(confidence, 0.8)
            response_lines.append("Dejé creado un seguimiento para este lead.")

        if any(token in lowered for token in ("asesor", "humano", "llámame", "llamame")):
            actions.append(
                ProposedAction(
                    type=ActionType.HANDOFF_HUMAN,
                    reason="El usuario pidió intervención humana.",
                    args={},
                )
            )
            intent = "handoff"
            confidence = 0.95
            response_lines.append("Voy a escalar esta conversación al equipo comercial.")

        if any(token in lowered for token in ("precio", "planes", "demo", "cotización", "cotizacion")):
            if intent == "generic_reply":
                intent = "sales_interest"
                confidence = 0.75
            stage = contact.stage if contact and contact.stage else "nuevo"
            response_lines.append(
                f"Entendido. Veo el lead en etapa {stage} y sigo con la conversación comercial."
            )

        if not response_lines:
            response_lines.append("Recibido. Ya tengo el contexto del lead y sigo con esta conversación.")

        return PlanningResult(
            intent=intent,
            confidence=confidence,
            response_text=" ".join(response_lines),
            actions=actions,
            should_reply=True,
        )
