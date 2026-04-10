from __future__ import annotations

import re
from textwrap import dedent

from openai import OpenAIError
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
    _STAGE_ORDER = {
        "Prospecto": 0,
        "Primer contacto": 1,
        "Demo agendada": 2,
        "Demo realizada": 3,
        "Propuesta enviada": 4,
        "Negociación": 5,
        "Prueba / Trial": 6,
        "Cliente": 7,
        "Perdido": -1,
        "No califica": -1,
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = None
        if settings.openai_api_key:
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=0.2,
            ).with_structured_output(PlannerOutput, method="function_calling")

    async def plan(
        self,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str],
        semantic_memories: list[str],
    ) -> PlanningResult:
        if self._llm is not None:
            try:
                result = await self._plan_with_llm(text, contact, recent_messages, semantic_memories)
                return self._enforce_sales_policy(self._repair_actions(result, text, contact), text, contact)
            except OpenAIError:
                return self._plan_with_rules(text, contact)
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
            You are a senior inbound sales agent for Wabog.com.
            Wabog sells software and operational tooling for lawyers, law firms, and legal teams.
            Your job is to qualify interest, move the lead to the correct pipeline stage, capture commercial notes,
            propose demos or trials when appropriate, and push the conversation toward the next concrete step.

            The orchestrator has already resolved the current lead from the sender phone number.
            You can only act on that current lead. Never assume access to other CRM records.
            Decide the user intent, whether to reply, and which actions to take on the current lead.

            Sales rules:
            - Be concise, warm, and operational.
            - Always answer as a commercial rep for Wabog.
            - If the user shows interest, qualify pain, process volume, current workflow, and buying context.
            - If the user asks for price, demo, trial, implementation, integrations, or wants to know how it works,
              move the lead forward in the pipeline when justified.
            - Use only these stage transitions when justified by the message:
              Prospecto -> Primer contacto
              Primer contacto -> Demo agendada
              Demo agendada -> Demo realizada
              Demo realizada -> Propuesta enviada
              Propuesta enviada -> Negociación
              Trial or active evaluation -> Prueba / Trial
              Closed won -> Cliente
              Disqualified -> No califica
              Lost -> Perdido
            - Add notes when the user reveals buying intent, objections, current process, or next-step commitments.
            - Create a follow-up when a concrete next step or reminder is needed.
            - Never invent CRM data you do not have.

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

    def _repair_actions(self, result: PlanningResult, text: str, contact: CRMContact | None) -> PlanningResult:
        current_stage = contact.stage if contact and contact.stage else "Prospecto"
        repaired_actions: list[ProposedAction] = []
        for action in result.actions:
            args = dict(action.args)
            if action.type == ActionType.UPDATE_STAGE and not args.get("stage"):
                inferred = self._infer_stage_from_text(text, current_stage)
                if inferred:
                    args["stage"] = inferred
            if action.type == ActionType.APPEND_NOTE and not args.get("note"):
                args["note"] = text.strip()
            if action.type == ActionType.CREATE_FOLLOWUP and not args.get("summary"):
                args["summary"] = text.strip()
            repaired_actions.append(action.model_copy(update={"args": args}))
        return result.model_copy(update={"actions": repaired_actions})

    def _enforce_sales_policy(self, result: PlanningResult, text: str, contact: CRMContact | None) -> PlanningResult:
        lowered = text.lower()
        current_stage = contact.stage if contact and contact.stage else "Prospecto"
        actions = [action.model_copy(deep=True) for action in result.actions]
        notes = {action.args.get("note", "").strip() for action in actions if action.type == ActionType.APPEND_NOTE}
        has_followup = any(action.type == ActionType.CREATE_FOLLOWUP for action in actions)
        stage_index = next(
            (index for index, action in enumerate(actions) if action.type == ActionType.UPDATE_STAGE),
            None,
        )
        inferred_stage = self._infer_stage_from_text(text, current_stage)
        if inferred_stage and self._should_update_stage(current_stage, inferred_stage):
            stage_action = ProposedAction(
                type=ActionType.UPDATE_STAGE,
                reason="La política comercial infirió una transición de etapa por la intención del lead.",
                args={"stage": inferred_stage},
            )
            if stage_index is None:
                actions.append(stage_action)
            else:
                actions[stage_index] = stage_action

        if self._should_add_sales_note(lowered, notes):
            actions.append(
                ProposedAction(
                    type=ActionType.APPEND_NOTE,
                    reason="La política comercial registra el interés o contexto revelado por el lead.",
                    args={"note": text.strip()},
                )
            )

        if self._should_create_followup(lowered) and not has_followup:
            actions.append(
                ProposedAction(
                    type=ActionType.CREATE_FOLLOWUP,
                    reason="La política comercial requiere dejar el siguiente paso explícito.",
                    args={"summary": text.strip()},
                )
            )

        return result.model_copy(update={"actions": actions})

    def _infer_stage_from_text(self, text: str, current_stage: str) -> str | None:
        lowered = text.lower()
        if any(token in lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")):
            return "Demo agendada"
        if any(token in lowered for token in ("trial", "prueba", "probar", "testear")):
            return "Prueba / Trial"
        if any(token in lowered for token in ("negociación", "negociacion", "negociar", "descuento")):
            return "Negociación"
        if any(token in lowered for token in ("propuesta", "cotización", "cotizacion")):
            return "Propuesta enviada"
        if any(token in lowered for token in ("no me interesa", "no estoy interesado", "no aplica")):
            return "No califica"
        if any(token in lowered for token in ("ya compré", "ya compre", "arranquemos", "vamos a trabajar")):
            return "Cliente"
        if any(token in lowered for token in ("precio", "planes", "cuanto cuesta", "costos")) and current_stage == "Prospecto":
            return "Primer contacto"
        return None

    def _should_update_stage(self, current_stage: str, inferred_stage: str) -> bool:
        current_rank = self._STAGE_ORDER.get(current_stage, 0)
        inferred_rank = self._STAGE_ORDER.get(inferred_stage, current_rank)
        if inferred_stage in {"No califica", "Perdido", "Cliente"}:
            return inferred_stage != current_stage
        return inferred_rank > current_rank

    def _should_add_sales_note(self, lowered_text: str, existing_notes: set[str]) -> bool:
        if existing_notes:
            return False
        note_signals = (
            "precio",
            "planes",
            "cotización",
            "cotizacion",
            "demo",
            "reunión",
            "reunion",
            "trial",
            "prueba",
            "probar",
            "negociación",
            "negociacion",
            "propuesta",
            "despacho",
            "firma",
            "abogada",
            "abogado",
        )
        return any(token in lowered_text for token in note_signals)

    def _should_create_followup(self, lowered_text: str) -> bool:
        followup_signals = (
            "demo",
            "agendar",
            "reunión",
            "reunion",
            "llamada",
            "trial",
            "prueba",
            "probar",
            "propuesta",
            "negociación",
            "negociacion",
            "seguimiento",
        )
        return any(token in lowered_text for token in followup_signals)

    def _plan_with_rules(self, text: str, contact: CRMContact | None) -> PlanningResult:
        lowered = text.lower()
        actions: list[ProposedAction] = []
        response_lines: list[str] = []
        intent = "generic_reply"
        confidence = 0.55
        current_stage = contact.stage if contact and contact.stage else "Prospecto"

        def add_stage_change(stage: str, reason: str) -> None:
            nonlocal intent, confidence
            if any(action.type == ActionType.UPDATE_STAGE for action in actions):
                return
            actions.append(
                ProposedAction(
                    type=ActionType.UPDATE_STAGE,
                    reason=reason,
                    args={"stage": stage},
                )
            )
            intent = "update_stage"
            confidence = max(confidence, 0.82)

        if match := re.search(r"etapa(?:\s+a)?\s+([a-zA-ZáéíóúÁÉÍÓÚ ]+)", lowered):
            stage = match.group(1).strip().title()
            add_stage_change(stage, "El mensaje pide un cambio de etapa.")
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

        if any(token in lowered for token in ("precio", "planes", "cotización", "cotizacion", "cuanto cuesta", "costos")):
            if intent == "generic_reply":
                intent = "pricing_interest"
                confidence = 0.8
            if current_stage == "Prospecto":
                add_stage_change("Primer contacto", "El lead preguntó por precio o planes.")
            actions.append(
                ProposedAction(
                    type=ActionType.APPEND_NOTE,
                    reason="El lead preguntó por precio o planes.",
                    args={"note": f"Interés comercial en pricing/planes: {text.strip()}"},
                )
            )
            response_lines.append(
                "Claro. Wabog está pensado para abogados y despachos que quieren operar mejor sus procesos. "
                "Para orientarte bien, te ayudo a revisar tu caso y te propongo el siguiente paso comercial."
            )

        if any(token in lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")):
            if intent == "generic_reply":
                intent = "demo_interest"
                confidence = 0.85
            add_stage_change("Demo agendada", "El lead pidió demo o reunión comercial.")
            actions.append(
                ProposedAction(
                    type=ActionType.APPEND_NOTE,
                    reason="El lead pidió demo o reunión.",
                    args={"note": f"Solicitud de demo/reunión: {text.strip()}"},
                )
            )
            actions.append(
                ProposedAction(
                    type=ActionType.CREATE_FOLLOWUP,
                    reason="Hay que dar siguiente paso comercial después de pedir demo.",
                    args={"summary": f"Coordinar demo para lead actual. Contexto: {text.strip()}"},
                )
            )
            response_lines.append(
                "Perfecto. Tiene sentido coordinar una demo para mostrarte cómo Wabog puede ayudarte a gestionar y vender mejor tus servicios legales."
            )

        if any(token in lowered for token in ("trial", "prueba", "probar", "testear")):
            if intent == "generic_reply":
                intent = "trial_interest"
                confidence = 0.86
            add_stage_change("Prueba / Trial", "El lead quiere probar la solución.")
            actions.append(
                ProposedAction(
                    type=ActionType.APPEND_NOTE,
                    reason="El lead expresó interés en trial.",
                    args={"note": f"Interés en trial/prueba: {text.strip()}"},
                )
            )
            response_lines.append(
                "Buen siguiente paso. Si estás evaluando Wabog, te acompaño para que la prueba tenga un objetivo comercial claro y veas rápido si encaja contigo."
            )

        if any(token in lowered for token in ("no me interesa", "no estoy interesado", "no aplica", "no aplica para mi")):
            if intent == "generic_reply":
                intent = "disqualified"
                confidence = 0.88
            add_stage_change("No califica", "El lead indicó que no aplica o no tiene interés.")
            response_lines.append("Entendido. Dejo registrado que por ahora no hay encaje comercial.")

        if any(token in lowered for token in ("ya compré", "ya compre", "vamos a trabajar", "arranquemos", "listo hagámosle", "listo hagamosle")):
            if intent == "generic_reply":
                intent = "closed_won"
                confidence = 0.9
            add_stage_change("Cliente", "El lead indicó cierre o arranque.")
            response_lines.append("Perfecto. Dejo el lead como cliente y registramos el siguiente paso operativo.")

        if not response_lines:
            response_lines.append(
                "Perfecto. Soy el agente comercial de Wabog para abogados. Cuéntame cómo estás manejando hoy tus procesos o qué quieres mejorar, y te guío al siguiente paso."
            )

        return PlanningResult(
            intent=intent,
            confidence=confidence,
            response_text=" ".join(response_lines),
            actions=actions,
            should_reply=True,
        )
