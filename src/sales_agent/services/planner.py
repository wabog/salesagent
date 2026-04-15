from __future__ import annotations

import re
from datetime import date, timedelta
from textwrap import dedent

from openai import OpenAIError
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from sales_agent.core.config import Settings
from sales_agent.domain.models import ActionType
from sales_agent.domain.models import CRMContact, PlanningResult, PromptMode, ProposedAction
from sales_agent.services.prompt_store import DEFAULT_BUSINESS_PROMPT


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

    def __init__(self, settings: Settings, prompt_store=None) -> None:
        self._settings = settings
        self._prompt_store = prompt_store
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
        prompt_mode: PromptMode = PromptMode.PUBLISHED,
    ) -> PlanningResult:
        if self._llm is not None:
            try:
                result = await self._plan_with_llm(text, contact, recent_messages, semantic_memories, prompt_mode)
                repaired = self._repair_actions(result, text, contact, recent_messages)
                return self._enforce_sales_policy(repaired, text, contact, recent_messages)
            except OpenAIError:
                return self._plan_with_rules(text, contact)
        return self._plan_with_rules(text, contact)

    async def _plan_with_llm(
        self,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str],
        semantic_memories: list[str],
        prompt_mode: PromptMode,
    ) -> PlanningResult:
        business_prompt = DEFAULT_BUSINESS_PROMPT
        if self._prompt_store is not None:
            business_prompt = await self._prompt_store.get_business_prompt(prompt_mode)
        prompt = self._compose_llm_prompt(
            business_prompt=business_prompt,
            contact_json=contact.model_dump_json(indent=2) if contact else "None",
            recent_messages=str(recent_messages),
            semantic_memories=str(semantic_memories),
            text=text,
        )
        output = await self._llm.ainvoke(prompt)
        return PlanningResult(**output.model_dump())

    def get_prompt_scaffold(self) -> str:
        return self._compose_llm_prompt(
            business_prompt="{{BUSINESS_PROMPT}}",
            contact_json="{{CONTACT}}",
            recent_messages="{{RECENT_MESSAGES}}",
            semantic_memories="{{SEMANTIC_MEMORIES}}",
            text="{{USER_MESSAGE}}",
        )

    def _compose_llm_prompt(
        self,
        *,
        business_prompt: str,
        contact_json: str,
        recent_messages: str,
        semantic_memories: str,
        text: str,
    ) -> str:
        return dedent(
            f"""
            You are a senior inbound sales agent for Wabog.com.
            Your job is to qualify interest, move the lead to the correct pipeline stage, capture commercial notes,
            propose demos or trials when appropriate, and push the conversation toward the next concrete step.

            Commercial brief:
            {business_prompt}

            The orchestrator has already resolved the current lead from the sender phone number.
            You can only act on that current lead. Never assume access to other CRM records.
            Decide the user intent, whether to reply, and which actions to take on the current lead.

            Sales rules:
            - If the user explicitly shares or corrects contact data for the current lead, use UPDATE_CONTACT_FIELDS.
            - UPDATE_CONTACT_FIELDS can persist `full_name` and `email` for the current lead.
            - If the user confirms that the current follow-up or promised next step was already completed, use COMPLETE_FOLLOWUP.
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
            - APPEND_NOTE entries must be CRM-ready summaries in Spanish, written naturally for a future sales rep.
            - Notes must capture durable facts and next steps, not copy-paste the user's message.
            - Prefer 1 to 3 short sentences such as company context, current tool, pain points, urgency, and agreed next step.
            - Create a follow-up when a concrete next step or reminder is needed.
            - CREATE_FOLLOWUP summaries must be short operational reminders, not the raw user message.
            - Never invent CRM data you do not have.

            Contact:
            {contact_json}

            Recent messages:
            {recent_messages}

            Semantic memories:
            {semantic_memories}

            User message:
            {text}
            """
        ).strip()

    def _repair_actions(
        self,
        result: PlanningResult,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str],
    ) -> PlanningResult:
        current_stage = contact.stage if contact and contact.stage else "Prospecto"
        repaired_actions: list[ProposedAction] = []
        for action in result.actions:
            args = dict(action.args)
            if action.type == ActionType.UPDATE_STAGE and not args.get("stage"):
                inferred = self._infer_stage_from_text(text, current_stage, recent_messages)
                if inferred:
                    args["stage"] = inferred
            if action.type == ActionType.UPDATE_CONTACT_FIELDS:
                fields = args.get("fields")
                normalized_fields = dict(fields) if isinstance(fields, dict) else {}
                inferred_email = self._extract_email(text)
                inferred_name = self._extract_explicit_name(text)
                if inferred_email and not normalized_fields.get("email"):
                    normalized_fields["email"] = inferred_email
                if inferred_name and not normalized_fields.get("full_name"):
                    normalized_fields["full_name"] = inferred_name
                args["fields"] = {
                    key: str(value).strip()
                    for key, value in normalized_fields.items()
                    if key in {"email", "full_name"} and str(value).strip()
                }
            if action.type == ActionType.APPEND_NOTE:
                existing_note = str(args.get("note", "")).strip()
                if not existing_note or self._should_rewrite_note(existing_note, text):
                    generated_note = self._build_sales_note(text, recent_messages)
                    if generated_note:
                        args["note"] = generated_note
            if action.type == ActionType.CREATE_FOLLOWUP:
                existing_summary = str(args.get("summary", "")).strip()
                if not existing_summary or self._should_rewrite_followup_summary(existing_summary, text):
                    args["summary"] = self._build_followup_summary(text, recent_messages)
                if not str(args.get("due_date", "")).strip():
                    args["due_date"] = self._infer_followup_due_date(text, recent_messages)
            if action.type == ActionType.COMPLETE_FOLLOWUP:
                existing_outcome = str(args.get("outcome", "")).strip()
                if not existing_outcome:
                    args["outcome"] = self._build_followup_completion_outcome(text)
            repaired_actions.append(action.model_copy(update={"args": args}))
        return result.model_copy(update={"actions": repaired_actions})

    def _enforce_sales_policy(
        self,
        result: PlanningResult,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str] | None = None,
    ) -> PlanningResult:
        lowered = text.lower()
        current_stage = contact.stage if contact and contact.stage else "Prospecto"
        actions = [action.model_copy(deep=True) for action in result.actions]
        notes = {action.args.get("note", "").strip() for action in actions if action.type == ActionType.APPEND_NOTE}
        has_followup = any(action.type == ActionType.CREATE_FOLLOWUP for action in actions)
        has_followup_completion = any(action.type == ActionType.COMPLETE_FOLLOWUP for action in actions)
        completion_detected = self._should_complete_followup(lowered, contact)
        stage_index = next(
            (index for index, action in enumerate(actions) if action.type == ActionType.UPDATE_STAGE),
            None,
        )
        inferred_stage = self._infer_stage_from_text(text, current_stage, recent_messages or [])
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
                    args={"note": self._build_sales_note(text, recent_messages or [])},
                )
            )

        if completion_detected and not has_followup_completion:
            actions.append(
                ProposedAction(
                    type=ActionType.COMPLETE_FOLLOWUP,
                    reason="La política comercial detectó que el siguiente paso vigente ya fue cumplido.",
                    args={"outcome": self._build_followup_completion_outcome(text)},
                )
            )
            has_followup_completion = True

        if self._should_create_followup(lowered) and not has_followup and not completion_detected:
            actions.append(
                ProposedAction(
                    type=ActionType.CREATE_FOLLOWUP,
                    reason="La política comercial requiere dejar el siguiente paso explícito.",
                    args={
                        "summary": self._build_followup_summary(text, recent_messages or []),
                        "due_date": self._infer_followup_due_date(text, recent_messages or []),
                    },
                )
            )

        return result.model_copy(update={"actions": actions})

    def _infer_stage_from_text(self, text: str, current_stage: str, recent_messages: list[str] | None = None) -> str | None:
        lowered = text.lower()
        recent_lowered = " ".join(message.lower() for message in (recent_messages or [])[-3:])
        affirmative_reply = lowered.strip() in {
            "si",
            "sí",
            "si claro",
            "sí claro",
            "claro",
            "de una",
            "dale",
            "hagámosle",
            "hagamosle",
            "ok",
            "vale",
            "me sirve",
        }
        if affirmative_reply:
            if any(token in recent_lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")):
                return "Demo agendada"
            if any(token in recent_lowered for token in ("trial", "prueba", "probar", "testear")):
                return "Prueba / Trial"
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
        if current_stage == "Prospecto" and any(
            token in lowered
            for token in ("adquirir", "comprar", "quisiera", "quiero", "me interesa", "interesado", "interesada")
        ):
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

    def _should_rewrite_note(self, note: str, text: str) -> bool:
        normalized_note = self._normalize_text(note)
        normalized_text = self._normalize_text(text)
        if not normalized_note:
            return True
        if normalized_note == normalized_text:
            return True
        if normalized_text and normalized_text in normalized_note and len(normalized_note) <= len(normalized_text) + 40:
            return True
        return normalized_note.startswith(
            (
                "solicitud de demo",
                "interés comercial en pricing",
                "interes comercial en pricing",
                "interés en trial",
                "interes en trial",
            )
        )

    def _should_rewrite_followup_summary(self, summary: str, text: str) -> bool:
        normalized_summary = self._normalize_text(summary)
        normalized_text = self._normalize_text(text)
        if not normalized_summary:
            return True
        if normalized_summary == normalized_text:
            return True
        return normalized_summary.startswith(("coordinar demo para lead actual", "contexto:"))

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    def _build_sales_note(self, text: str, recent_messages: list[str]) -> str:
        lowered = text.lower()
        recent_lowered = " ".join(message.lower() for message in recent_messages[-3:])
        sentences: list[str] = []

        firm_name_match = re.search(
            r"(?:despacho|firma)(?:\s+(?:llamado|llamada|que se llama|se llama))?\s+([A-Za-z0-9][A-Za-z0-9_-]*)",
            text,
            re.IGNORECASE,
        )
        lawyers_match = re.search(r"(\d+)\s+abogad", lowered)
        processes_match = re.search(r"(\d+)\s+proces", lowered)
        current_tool = self._extract_current_tool(text)
        if current_tool is None:
            for previous_message in reversed(recent_messages[-4:]):
                current_tool = self._extract_current_tool(previous_message)
                if current_tool:
                    break

        context_bits: list[str] = []
        if firm_name_match:
            context_bits.append(f"Despacho {firm_name_match.group(1)}")
        elif "despacho" in lowered or "firma" in lowered:
            context_bits.append("Despacho del lead")
        if lawyers_match:
            context_bits.append(f"{lawyers_match.group(1)} abogados")
        if processes_match:
            context_bits.append(f"aprox. {processes_match.group(1)} procesos al mes")
        if context_bits:
            if len(context_bits) == 1:
                sentences.append(f"{context_bits[0]}.")
            else:
                sentences.append(f"{context_bits[0]} con {', '.join(context_bits[1:])}.")

        tool_and_pain_parts: list[str] = []
        if current_tool:
            tool_and_pain_parts.append(f"Hoy usan {current_tool}")
        if any(token in lowered for token in ("problema", "problemas", "seguimiento", "llevar", "llevando")) and "seguimiento" in lowered:
            tool_and_pain_parts.append("reportan dolor en el seguimiento de procesos")
        if any(token in lowered for token in ("no notifica", "notifica bien", "notificaciones", "notificar")):
            tool_and_pain_parts.append("indican fallas de notificaciones")
        if "whatsapp" in lowered and any(token in lowered for token in ("notifica", "notificaciones", "notificar")):
            tool_and_pain_parts.append("quieren recibir alertas por WhatsApp")
        if tool_and_pain_parts:
            sentences.append(self._capitalize_first(self._join_note_parts(tool_and_pain_parts)) + ".")

        commercial_parts: list[str] = []
        if any(token in lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")):
            if any(token in lowered for token in ("si", "sí", "me parecería", "me pareceria", "claro", "vale", "ok", "de una")):
                commercial_parts.append("Confirman disposición para agendar una demo")
            else:
                commercial_parts.append("Muestran interés en agendar una demo")
        elif lowered.strip() in {"si", "sí", "si claro", "sí claro", "claro", "de una", "dale", "ok", "vale", "me sirve"} and any(
            token in recent_lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")
        ):
            commercial_parts.append("Confirman disposición para agendar una demo")
        if any(token in lowered for token in ("trial", "prueba", "probar", "testear")):
            commercial_parts.append("Quieren evaluar Wabog en prueba o trial")
        if any(token in lowered for token in ("precio", "planes", "cotización", "cotizacion", "cuanto cuesta", "costos")):
            commercial_parts.append("Consultan por pricing o propuesta comercial")
        if any(token in lowered for token in ("lo antes posible", "pronto", "urgente", "rápido", "rapido")):
            commercial_parts.append("Buscan avanzar pronto")
        if commercial_parts:
            sentences.append(self._capitalize_first(self._join_note_parts(commercial_parts)) + ".")

        if not sentences:
            return "Lead comparte contexto comercial relevante y conviene profundizar en necesidades y siguiente paso."
        return " ".join(sentences[:3])

    def _build_followup_summary(self, text: str, recent_messages: list[str]) -> str:
        lowered = text.lower()
        recent_lowered = " ".join(message.lower() for message in recent_messages[-3:])
        if any(token in lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")):
            return "Coordinar fecha y hora para demo comercial."
        if lowered.strip() in {"si", "sí", "si claro", "sí claro", "claro", "de una", "dale", "ok", "vale", "me sirve"} and any(
            token in recent_lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")
        ):
            return "Confirmar fecha y hora para la demo acordada."
        if any(token in lowered for token in ("trial", "prueba", "probar", "testear")):
            return "Definir siguiente paso para trial o prueba guiada."
        if any(token in lowered for token in ("precio", "planes", "cotización", "cotizacion", "propuesta")):
            return "Responder pricing o enviar propuesta comercial."
        return "Dar seguimiento comercial al lead."

    def _infer_followup_due_date(self, text: str, recent_messages: list[str]) -> str:
        lowered = text.lower()
        recent_lowered = " ".join(message.lower() for message in recent_messages[-3:])
        base_date = date.today()
        if any(token in lowered for token in ("hoy", "esta tarde", "este rato")):
            return base_date.isoformat()
        if any(token in lowered for token in ("mañana", "manana")):
            return (base_date + timedelta(days=1)).isoformat()
        if "próxima semana" in lowered or "proxima semana" in lowered:
            return (base_date + timedelta(days=7)).isoformat()
        if any(token in lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")):
            return (base_date + timedelta(days=1)).isoformat()
        if lowered.strip() in {"si", "sí", "si claro", "sí claro", "claro", "de una", "dale", "ok", "vale", "me sirve"} and any(
            token in recent_lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")
        ):
            return (base_date + timedelta(days=1)).isoformat()
        return (base_date + timedelta(days=1)).isoformat()

    def _extract_current_tool(self, text: str) -> str | None:
        patterns = (
            r"(?:herramienta|tool|app)\s+llamada\s+([A-Za-z0-9][A-Za-z0-9_-]*)",
            r"(?:usamos|usan|actualmente usan|hoy usan)\s+([A-Za-z0-9][A-Za-z0-9_-]*)",
        )
        for pattern in patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                return match.group(1)[0].upper() + match.group(1)[1:]
        return None

    def _join_note_parts(self, parts: list[str]) -> str:
        unique_parts: list[str] = []
        seen: set[str] = set()
        for part in parts:
            normalized = self._normalize_text(part)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_parts.append(part)
        if not unique_parts:
            return ""
        if len(unique_parts) == 1:
            return unique_parts[0]
        if len(unique_parts) == 2:
            return f"{unique_parts[0]} y {unique_parts[1]}"
        return ", ".join(unique_parts[:-1]) + f" y {unique_parts[-1]}"

    def _capitalize_first(self, text: str) -> str:
        if not text:
            return text
        return text[0].upper() + text[1:]

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

    def _should_complete_followup(self, lowered_text: str, contact: CRMContact | None) -> bool:
        if contact is None or not contact.followup_summary:
            return False
        explicit_completion_phrases = (
            "ya envié",
            "ya envie",
            "ya agendé",
            "ya agende",
            "ya coordiné",
            "ya coordine",
            "ya realicé",
            "ya realice",
            "ya quedó",
            "ya quedo",
            "ya se envió",
            "ya se envio",
            "ya se agendó",
            "ya se agendo",
            "ya está hecho",
            "ya esta hecho",
            "seguimiento completado",
            "seguimiento cumplido",
        )
        if any(phrase in lowered_text for phrase in explicit_completion_phrases):
            return True
        completion_context_markers = (
            "ya",
            "listo",
            "hecho",
            "complet",
            "cumpl",
            "resuelto",
        )
        completion_action_markers = (
            "envié",
            "envie",
            "agendé",
            "agende",
            "realicé",
            "realice",
            "coordiné",
            "coordine",
            "cerré",
            "cerre",
        )
        return any(token in lowered_text for token in completion_context_markers) and any(
            token in lowered_text for token in completion_action_markers
        )

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

        explicit_email = self._extract_email(text)
        explicit_name = self._extract_explicit_name(text)
        contact_fields: dict[str, str] = {}
        current_email = (contact.email or "").strip() if contact else ""
        current_name = (contact.full_name or "").strip() if contact else ""
        if explicit_email and explicit_email != current_email:
            contact_fields["email"] = explicit_email
        if explicit_name and explicit_name != current_name:
            contact_fields["full_name"] = explicit_name
        if contact_fields:
            actions.append(
                ProposedAction(
                    type=ActionType.UPDATE_CONTACT_FIELDS,
                    reason="El lead compartió datos de contacto explícitos.",
                    args={"fields": contact_fields},
                )
            )
            confidence = max(confidence, 0.78)
            if set(contact_fields) == {"email", "full_name"}:
                response_lines.append("Perfecto, ya tengo tu nombre y tu correo para seguir con la conversación.")
            elif "email" in contact_fields:
                response_lines.append("Perfecto, ya tengo tu correo de contacto.")
            elif "full_name" in contact_fields:
                response_lines.append("Perfecto, ya tengo tu nombre para continuar.")

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
                    args={
                        "summary": text.strip(),
                        "due_date": self._infer_followup_due_date(text, []),
                    },
                )
            )
            intent = "followup"
            confidence = max(confidence, 0.8)
            response_lines.append("Dejé creado un seguimiento para este lead.")

        if self._should_complete_followup(lowered, contact):
            actions.append(
                ProposedAction(
                    type=ActionType.COMPLETE_FOLLOWUP,
                    reason="El lead indicó que ya se cumplió el siguiente paso pendiente.",
                    args={"outcome": self._build_followup_completion_outcome(text)},
                )
            )
            if intent == "generic_reply":
                intent = "followup_completed"
                confidence = max(confidence, 0.78)
            response_lines.append("Perfecto. Marco como completado el seguimiento activo.")

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
                    args={"note": self._build_sales_note(text, [])},
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
                    args={"note": self._build_sales_note(text, [])},
                )
            )
            actions.append(
                ProposedAction(
                    type=ActionType.CREATE_FOLLOWUP,
                    reason="Hay que dar siguiente paso comercial después de pedir demo.",
                    args={
                        "summary": self._build_followup_summary(text, []),
                        "due_date": self._infer_followup_due_date(text, []),
                    },
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
                    args={"note": self._build_sales_note(text, [])},
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

    def _extract_email(self, text: str) -> str | None:
        match = re.search(r"\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", text, re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip().lower()

    def _extract_explicit_name(self, text: str) -> str | None:
        patterns = (
            r"\bmi nombre es\s+([A-Za-zÁÉÍÓÚÑáéíóúñ][A-Za-zÁÉÍÓÚÑáéíóúñ' -]{1,60}?)(?=\s+(?:y\s+(?:mi\s+)?correo)\b|[,.!?]|\n|$)",
            r"\bme llamo\s+([A-Za-zÁÉÍÓÚÑáéíóúñ][A-Za-zÁÉÍÓÚÑáéíóúñ' -]{1,60}?)(?=\s+(?:y\s+(?:mi\s+)?correo)\b|[,.!?]|\n|$)",
        )
        for pattern in patterns:
            if not (match := re.search(pattern, text, re.IGNORECASE)):
                continue
            candidate = re.split(r"[,.!?\n]", match.group(1).strip(), maxsplit=1)[0].strip(" -")
            candidate = re.sub(r"\s+", " ", candidate)
            if not candidate:
                return None
            parts = candidate.split(" ")
            if len(parts) > 5:
                return None
            return " ".join(part.capitalize() for part in parts)
        return None

    def _build_followup_completion_outcome(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip())
        if not cleaned:
            return "Lead confirmó que el seguimiento se completó."
        if cleaned.endswith((".", "!", "?")):
            return cleaned
        return f"{cleaned}."
