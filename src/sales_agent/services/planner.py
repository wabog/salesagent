from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from textwrap import dedent
from zoneinfo import ZoneInfo

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
        hard_rule_result = self._apply_hard_rules(text, contact)
        if hard_rule_result is not None:
            return hard_rule_result
        if self._llm is not None:
            try:
                result = await self._plan_with_llm(text, contact, recent_messages, semantic_memories, prompt_mode)
                repaired = self._repair_actions(result, text, contact, recent_messages)
                enforced = self._enforce_sales_policy(repaired, text, contact, recent_messages)
                return self._apply_planning_guardrail(enforced, text, contact, recent_messages)
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
            - Never claim that a meeting, demo, invite, or calendar booking is confirmed unless CREATE_MEETING succeeded.
            - Interest in a demo does not mean the demo is scheduled yet.
            - Move a lead to `Demo agendada` only when there is a concrete date and time or the contact already has an upcoming calendar event.
            - If the lead wants a demo but there is no exact slot yet, keep pushing to the next step without pretending it is booked.
            - Before creating a meeting, make sure you have the lead full name and email if those fields are missing.
            - If date and time are already defined but name or email are still missing, ask for the missing fields instead of saying the invite was sent.
            - Reuse recent conversation context aggressively. If the last user message only confirms something like "si, agenda", "martes a las 9", or "dale", combine it with prior turns before deciding actions.
            - If a prior turn already gave the day or hour for a demo and the current turn confirms it, create the meeting immediately when contact data is sufficient.
            - Do not ask again for data that the current lead already has in Contact.
            - Distinguish identity questions from contact-source questions. "como me llamo", "cual es mi nombre", and "como me tienes guardado" ask about the lead name, not about how Wabog contacted them.
            - For identity questions, answer from the current Contact. If the current lead has no reliable name yet, say that plainly and ask for it.
            - Questions like "de donde sacaron mi numero", "como consiguieron mi contacto", or "como me llamaron" are about contact source, not the lead name.
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
            - Use CREATE_MEETING only when the lead already confirmed a specific day and time for the demo or call.
            - CREATE_MEETING must include `start_iso`, `duration_minutes`, `title`, and `description`.
            - If there is a self-scheduling link available and the lead wants a demo but has not fixed a time, offer that link naturally.
            - If the contact metadata already includes an upcoming calendar event, treat the demo as scheduled.
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

    def _apply_hard_rules(self, text: str, contact: CRMContact | None) -> PlanningResult | None:
        normalized = self._normalize_lookup_text(text)
        if normalized in {
            "como me llamo",
            "cual es mi nombre",
            "que nombre tienes mio",
            "que nombre tienes de mi",
            "como me tienes guardado",
        }:
            full_name = (contact.full_name or "").strip() if contact else ""
            if self._is_specific_contact_name(full_name, contact.phone_number if contact else None):
                response_text = f"Te tengo registrado como {full_name}."
            else:
                response_text = "Todavía no tengo tu nombre registrado. Si quieres, compártemelo y lo guardo."
            return PlanningResult(
                intent="ask_name",
                confidence=0.99,
                response_text=response_text,
                actions=[],
                should_reply=True,
            )

        if normalized in {
            "de donde sacaron mi numero",
            "de donde sacaste mi numero",
            "como consiguieron mi numero",
            "como consiguieron mi contacto",
            "como me llamaron",
            "como me llamaste",
            "como me contacto wabog",
            "porque me escribieron",
        }:
            return PlanningResult(
                intent="ask_contact_source",
                confidence=0.96,
                response_text=(
                    "Te contactamos porque tu número quedó registrado como posible interesado en soluciones de Wabog "
                    "para abogados y equipos legales. Si prefieres, también te cuento brevemente qué hacemos."
                ),
                actions=[],
                should_reply=True,
            )
        return None

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
            if action.type == ActionType.CREATE_MEETING:
                meeting_payload = self._build_meeting_payload(text, contact, recent_messages)
                if meeting_payload:
                    merged_payload = dict(meeting_payload)
                    merged_payload.update({key: value for key, value in args.items() if value})
                    args = merged_payload
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
        has_meeting_creation = any(action.type == ActionType.CREATE_MEETING for action in actions)
        upcoming_event = self._get_upcoming_calendar_event(contact)
        completion_detected = self._should_complete_followup(lowered, contact) or (
            upcoming_event is not None and bool(contact and contact.followup_summary)
        )
        stage_index = next(
            (index for index, action in enumerate(actions) if action.type == ActionType.UPDATE_STAGE),
            None,
        )
        inferred_stage = self._infer_stage_from_text(text, current_stage, recent_messages or [])
        if upcoming_event is not None:
            inferred_stage = "Demo agendada"
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

        if upcoming_event is not None and not notes:
            actions.append(
                ProposedAction(
                    type=ActionType.APPEND_NOTE,
                    reason="El calendario ya muestra una demo futura para este lead.",
                    args={"note": self._build_calendar_note(upcoming_event)},
                )
            )

        if completion_detected and not has_followup_completion:
            actions.append(
                ProposedAction(
                    type=ActionType.COMPLETE_FOLLOWUP,
                    reason="La política comercial detectó que el siguiente paso vigente ya fue cumplido.",
                    args={"outcome": self._build_followup_completion_outcome(text, upcoming_event=upcoming_event)},
                )
            )
            has_followup_completion = True

        meeting_payload = self._build_meeting_payload(text, contact, recent_messages or [])
        missing_meeting_fields = self._missing_contact_fields_for_meeting(contact)
        if (
            meeting_payload
            and not has_meeting_creation
            and upcoming_event is None
            and not missing_meeting_fields
        ):
            actions.append(
                ProposedAction(
                    type=ActionType.CREATE_MEETING,
                    reason="El lead ya dio una fecha y hora concreta para la demo.",
                    args=meeting_payload,
                )
            )
            has_meeting_creation = True

        if self._should_create_followup(lowered) and not has_followup and not completion_detected and not has_meeting_creation:
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

        response_text = result.response_text
        if meeting_payload and missing_meeting_fields and upcoming_event is None:
            response_text = self._build_missing_meeting_fields_response(missing_meeting_fields)
        if self._should_offer_self_schedule_link(lowered, upcoming_event) and self._settings.google_calendar_self_schedule_url:
            response_text = self._append_self_schedule_link(response_text)
        if upcoming_event is not None:
            response_text = self._append_calendar_confirmation(response_text, upcoming_event)

        return result.model_copy(update={"actions": actions, "response_text": response_text})

    def _apply_planning_guardrail(
        self,
        result: PlanningResult,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str] | None = None,
    ) -> PlanningResult:
        actions = [action.model_copy(deep=True) for action in result.actions]
        recent_messages = recent_messages or []
        projected_contact = self._project_contact_for_actions(contact, actions, text)
        meeting_payload = self._build_meeting_payload(text, projected_contact, recent_messages)
        missing_meeting_fields = self._missing_contact_fields_for_meeting(projected_contact)
        upcoming_event = self._get_upcoming_calendar_event(contact)

        if meeting_payload and upcoming_event is None:
            actions = [
                action
                for action in actions
                if action.type not in {ActionType.COMPLETE_FOLLOWUP, ActionType.CREATE_FOLLOWUP}
            ]

            stage_updated = False
            meeting_present = False
            normalized_actions: list[ProposedAction] = []
            for action in actions:
                if action.type == ActionType.UPDATE_STAGE:
                    stage_updated = True
                    normalized_actions.append(
                        action.model_copy(
                            update={
                                "reason": "El guardrail detectó que ya existe contexto suficiente para tratar la demo como agendada.",
                                "args": {"stage": "Demo agendada"} if not missing_meeting_fields else {"stage": "Primer contacto"},
                            }
                        )
                    )
                    continue
                if action.type == ActionType.CREATE_MEETING:
                    meeting_present = True
                    normalized_actions.append(
                        action.model_copy(
                            update={
                                "reason": "El guardrail reconstruyó el slot de agenda usando el contexto reciente.",
                                "args": meeting_payload,
                            }
                        )
                    )
                    continue
                normalized_actions.append(action)

            if not stage_updated:
                normalized_actions.append(
                    ProposedAction(
                        type=ActionType.UPDATE_STAGE,
                        reason="El guardrail fija la etapa comercial según el contexto actual de agenda.",
                        args={"stage": "Demo agendada"} if not missing_meeting_fields else {"stage": "Primer contacto"},
                    )
                )
            if not missing_meeting_fields and not meeting_present:
                normalized_actions.append(
                    ProposedAction(
                        type=ActionType.CREATE_MEETING,
                        reason="El guardrail detectó fecha y hora concretas para crear la demo sin esperar otra confirmación.",
                        args=meeting_payload,
                    )
                )
            actions = normalized_actions

        response_text = result.response_text
        if meeting_payload and not missing_meeting_fields and upcoming_event is None:
            response_text = "Perfecto. Voy a agendar la demo ahora mismo."
        elif meeting_payload and missing_meeting_fields and upcoming_event is None:
            response_text = self._build_missing_meeting_fields_response(missing_meeting_fields)

        if upcoming_event is None and not any(action.type == ActionType.CREATE_MEETING for action in actions):
            response_text = self._strip_false_booking_claims(response_text)

        return result.model_copy(update={"actions": actions, "response_text": response_text})

    def _infer_stage_from_text(self, text: str, current_stage: str, recent_messages: list[str] | None = None) -> str | None:
        lowered = text.lower()
        recent_lowered = " ".join(message.lower() for message in (recent_messages or [])[-3:])
        has_concrete_demo_slot = self._extract_requested_meeting_start(text, recent_messages) is not None
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
                return "Demo agendada" if has_concrete_demo_slot else "Primer contacto"
            if any(token in recent_lowered for token in ("trial", "prueba", "probar", "testear")):
                return "Prueba / Trial"
        if any(token in lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")):
            return "Demo agendada" if has_concrete_demo_slot else "Primer contacto"
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

    def _normalize_lookup_text(self, value: str) -> str:
        normalized = self._normalize_text(value)
        normalized = normalized.translate(str.maketrans("áéíóúü", "aeiouu"))
        normalized = re.sub(r"[^a-z0-9ñ\s]", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _is_specific_contact_name(self, value: str | None, phone_number: str | None = None) -> bool:
        normalized = self._normalize_lookup_text(value or "")
        if not normalized:
            return False
        phone_digits = re.sub(r"\D", "", phone_number or "")
        value_digits = re.sub(r"\D", "", value or "")
        if phone_digits and value_digits and (phone_digits.endswith(value_digits) or value_digits.endswith(phone_digits)):
            return False
        return normalized not in {"~", "-", "lead", "usuario", "user", "playground user", "sin nombre"}

    def _project_contact_for_actions(
        self,
        contact: CRMContact | None,
        actions: list[ProposedAction],
        text: str,
    ) -> CRMContact | None:
        if contact is None:
            return None
        projected = contact.model_copy(deep=True)
        explicit_email = self._extract_email(text)
        explicit_name = self._extract_explicit_name(text)
        if explicit_email:
            projected.email = explicit_email
        if explicit_name:
            projected.full_name = explicit_name
        for action in actions:
            if action.type != ActionType.UPDATE_CONTACT_FIELDS:
                continue
            fields = action.args.get("fields")
            if not isinstance(fields, dict):
                continue
            if str(fields.get("email", "")).strip():
                projected.email = str(fields["email"]).strip()
            if str(fields.get("full_name", "")).strip():
                projected.full_name = str(fields["full_name"]).strip()
        return projected

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
        processes_match = re.search(r"(\d+)(?:\s*[-a]\s*(\d+))?\s+proces", lowered)
        current_tool = self._extract_current_tool(text)
        if current_tool is None:
            for previous_message in reversed(recent_messages[-4:]):
                current_tool = self._extract_current_tool(previous_message)
                if current_tool:
                    break

        context_bits: list[str] = []
        if firm_name_match:
            context_bits.append(f"Despacho {firm_name_match.group(1)}")
        elif any(token in lowered for token in ("abogado solo", "abogada sola", "trabajo solo", "trabajo sola", "yo solo", "yo sola")):
            context_bits.append("Abogado independiente")
        elif "despacho" in lowered or "firma" in lowered:
            context_bits.append("Despacho del lead")
        if lawyers_match:
            context_bits.append(f"{lawyers_match.group(1)} abogados")
        if processes_match:
            if processes_match.group(2):
                context_bits.append(f"aprox. {processes_match.group(1)}-{processes_match.group(2)} procesos al mes")
            else:
                context_bits.append(f"aprox. {processes_match.group(1)} procesos al mes")
        if context_bits:
            if len(context_bits) == 1:
                sentences.append(f"{context_bits[0]}.")
            else:
                sentences.append(f"{context_bits[0]} con {', '.join(context_bits[1:])}.")

        tool_and_pain_parts: list[str] = []
        if current_tool:
            tool_and_pain_parts.append(f"Hoy usan {current_tool}")
        if "excel" in lowered:
            tool_and_pain_parts.append("apoyan parte del proceso en Excel")
        if "whatsapp" in lowered:
            tool_and_pain_parts.append("usan WhatsApp en la operación diaria")
        if any(token in lowered for token in ("problema", "problemas", "seguimiento", "llevar", "llevando")) and "seguimiento" in lowered:
            tool_and_pain_parts.append("reportan dolor en el seguimiento de procesos")
        if any(token in lowered for token in ("no notifica", "notifica bien", "notificaciones", "notificar")):
            tool_and_pain_parts.append("indican fallas de notificaciones")
        if any(token in lowered for token in ("se me pierden cosas", "se pierden cosas", "se me pasan cosas", "se me escapan cosas")):
            tool_and_pain_parts.append("se le pierden tareas o pendientes en el flujo actual")
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

    def _build_meeting_payload(
        self,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str] | None = None,
    ) -> dict[str, str | int] | None:
        start_at = self._extract_requested_meeting_start(text, recent_messages)
        if start_at is None:
            return None
        lead_label = (contact.full_name or "Lead").strip() if contact else "Lead"
        description_parts = [
            "Demo comercial creada por el agente de ventas de Wabog.",
            f"Lead: {lead_label}.",
        ]
        if contact and contact.phone_number:
            description_parts.append(f"Telefono: {contact.phone_number}.")
        if contact and contact.email:
            description_parts.append(f"Email: {contact.email}.")
        return {
            "start_iso": start_at.isoformat(),
            "duration_minutes": self._settings.google_calendar_default_meeting_minutes,
            "title": f"Demo Wabog - {lead_label}",
            "description": " ".join(description_parts),
        }

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

    def _missing_contact_fields_for_meeting(self, contact: CRMContact | None) -> list[str]:
        if contact is None:
            return ["full_name", "email"]
        missing: list[str] = []
        if not (contact.full_name or "").strip():
            missing.append("full_name")
        if not (contact.email or "").strip():
            missing.append("email")
        return missing

    def _build_missing_meeting_fields_response(self, missing_fields: list[str]) -> str:
        if missing_fields == ["full_name", "email"]:
            return (
                "Perfecto. Para dejarte la demo agendada y enviarte la invitación, "
                "compárteme tu nombre completo y tu correo."
            )
        if missing_fields == ["email"]:
            return "Perfecto. Para enviarte la invitación de la demo, compárteme tu correo."
        if missing_fields == ["full_name"]:
            return "Perfecto. Antes de agendarla, compárteme tu nombre completo."
        return (
            "Perfecto. Antes de dejar la demo agendada, compárteme los datos que me faltan "
            "para enviarte la invitación."
        )

    def _strip_false_booking_claims(self, response_text: str) -> str:
        lowered = response_text.lower()
        false_booking_signals = (
            "ya te deje la demo agendada",
            "ya te dejé la demo agendada",
            "ya quedo agendada",
            "ya quedó agendada",
            "te enviare la invitacion",
            "te enviaré la invitación",
            "procedere a enviar la invitacion",
            "procederé a enviar la invitación",
            "voy a enviar la invitacion",
            "voy a enviar la invitación",
        )
        if not any(signal in lowered for signal in false_booking_signals):
            return response_text
        return "Perfecto. Revisemos los datos y el horario para dejar la demo correctamente agendada."

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
        upcoming_event = self._get_upcoming_calendar_event(contact)

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
                    args={"outcome": self._build_followup_completion_outcome(text, upcoming_event=upcoming_event)},
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
            meeting_payload = self._build_meeting_payload(text, contact, [])
            missing_meeting_fields = self._missing_contact_fields_for_meeting(contact)
            target_stage = "Demo agendada" if meeting_payload and not missing_meeting_fields else "Primer contacto"
            add_stage_change(target_stage, "El lead pidió demo o reunión comercial.")
            actions.append(
                ProposedAction(
                    type=ActionType.APPEND_NOTE,
                    reason="El lead pidió demo o reunión.",
                    args={"note": self._build_sales_note(text, [])},
                )
            )
            if meeting_payload and upcoming_event is None and not missing_meeting_fields:
                actions.append(
                    ProposedAction(
                        type=ActionType.CREATE_MEETING,
                        reason="El lead ya definió una fecha y hora concreta para la demo.",
                        args=meeting_payload,
                    )
                )
                response_lines.append("Perfecto. Voy a dejar esa demo creada de una vez en el calendario.")
            elif meeting_payload and missing_meeting_fields:
                response_lines.append(self._build_missing_meeting_fields_response(missing_meeting_fields))
            elif upcoming_event is None:
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

        response_text = " ".join(response_lines)
        if self._should_offer_self_schedule_link(lowered, upcoming_event) and self._settings.google_calendar_self_schedule_url:
            response_text = self._append_self_schedule_link(response_text)
        if upcoming_event is not None:
            response_text = self._append_calendar_confirmation(response_text, upcoming_event)

        return PlanningResult(
            intent=intent,
            confidence=confidence,
            response_text=response_text,
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

    def _build_followup_completion_outcome(self, text: str, upcoming_event: dict | None = None) -> str:
        if upcoming_event is not None:
            return f"El lead ya tiene una demo agendada para {upcoming_event.get('start_iso')}."
        cleaned = re.sub(r"\s+", " ", text.strip())
        if not cleaned:
            return "Lead confirmó que el seguimiento se completó."
        if cleaned.endswith((".", "!", "?")):
            return cleaned
        return f"{cleaned}."

    def _get_upcoming_calendar_event(self, contact: CRMContact | None) -> dict | None:
        if contact is None:
            return None
        return ((contact.metadata or {}).get("calendar") or {}).get("upcoming_event")

    def _build_calendar_note(self, upcoming_event: dict) -> str:
        return f"Ya existe una demo futura en calendario para el lead. Inicio: {upcoming_event.get('start_iso')}."

    def _append_self_schedule_link(self, response_text: str) -> str:
        if not self._settings.google_calendar_self_schedule_url:
            return response_text
        if self._settings.google_calendar_self_schedule_url in response_text:
            return response_text
        link_line = f"Si te sirve, también puedes agendar directamente aquí: {self._settings.google_calendar_self_schedule_url}"
        if link_line in response_text:
            return response_text
        return f"{response_text} {link_line}".strip()

    def _append_calendar_confirmation(self, response_text: str, upcoming_event: dict) -> str:
        response_text = self._strip_unsupported_reminder_promises(response_text)
        confirmation = f"Veo en calendario una demo futura para {upcoming_event.get('start_iso')}."
        if confirmation in response_text:
            return response_text
        return f"{response_text} {confirmation}".strip()

    def _strip_unsupported_reminder_promises(self, response_text: str) -> str:
        cleaned = re.sub(
            r"\s*Te enviar[eé] un recordatorio antes de la fecha\.?",
            "",
            response_text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*Te voy a recordar antes de la fecha\.?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", cleaned).strip()

    def _should_offer_self_schedule_link(self, lowered_text: str, upcoming_event: dict | None) -> bool:
        if upcoming_event is not None:
            return False
        if self._extract_requested_meeting_start(lowered_text, []) is not None:
            return False
        return any(
            token in lowered_text
            for token in (
                "demo",
                "agendar",
                "agenda",
                "reunión",
                "reunion",
                "llamada",
                "calendario",
                "link",
                "horario",
                "disponibilidad",
            )
        )

    def _extract_requested_meeting_start(self, text: str, recent_messages: list[str] | None = None) -> datetime | None:
        context_messages = [message.strip() for message in (recent_messages or [])[-6:] if message.strip()]
        combined_text = " ".join(context_messages + [text.strip()]).strip()
        lowered = combined_text.lower()
        if not any(token in lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")):
            return None
        tz = ZoneInfo(self._settings.google_calendar_timezone)
        base_date = datetime.now(tz).date()

        relative_match = re.search(
            r"\b(hoy|mañana|manana)\b(?:.*?)(?:a\s+las\s+|a\s+la\s+|tipo\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            lowered,
        )
        if relative_match:
            day_token, hour_raw, minute_raw, meridiem = relative_match.groups()
            target_date = base_date if day_token == "hoy" else base_date + timedelta(days=1)
            hour = int(hour_raw)
            minute = int(minute_raw or "0")
            if meridiem == "pm" and hour < 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            elif meridiem is None and 1 <= hour <= 7:
                hour += 12
            return datetime(target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=tz)

        isoish_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})[ t](\d{1,2}):(\d{2})\b", lowered)
        if isoish_match:
            date_raw, hour_raw, minute_raw = isoish_match.groups()
            target_date = date.fromisoformat(date_raw)
            return datetime(target_date.year, target_date.month, target_date.day, int(hour_raw), int(minute_raw), tzinfo=tz)

        weekday_map = {
            "lunes": 0,
            "martes": 1,
            "miercoles": 2,
            "miércoles": 2,
            "jueves": 3,
            "viernes": 4,
            "sabado": 5,
            "sábado": 5,
            "domingo": 6,
        }
        weekday_match = re.search(r"\b(lunes|martes|miercoles|miércoles|jueves|viernes|sabado|sábado|domingo)\b", lowered)
        hour_match = re.search(r"(?:a\s+las\s+|a\s+la\s+|tipo\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lowered)
        if weekday_match and hour_match:
            weekday_name = weekday_match.group(1)
            weekday = weekday_map[weekday_name]
            days_ahead = (weekday - base_date.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = base_date + timedelta(days=days_ahead)
            hour_raw, minute_raw, meridiem = hour_match.groups()
            hour = int(hour_raw)
            minute = int(minute_raw or "0")
            if meridiem == "pm" and hour < 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            return datetime(target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=tz)
        return None
