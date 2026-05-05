from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from openai import OpenAIError
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from sales_agent.core.config import Settings
from sales_agent.domain.models import ActionType
from sales_agent.domain.models import CRMContact, KnowledgeLookup, PlanningResult, ProposedAction
from sales_agent.services.name_validation import (
    ContextualNameConfirmationResolver,
    NameConfirmationDecision,
    apply_name_validation_metadata,
    build_trusted_name_assessment,
    contact_has_reliable_name,
    get_name_confirmation_candidate,
    is_specific_person_name,
)
from sales_agent.services.prompt_library import PromptLibrary, PromptSection


logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


class PlannerOutput(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    response_text: str
    should_reply: bool = True
    actions: list[ProposedAction] = Field(default_factory=list)


class KnowledgeSelectionOutput(BaseModel):
    section_names: list[str] = Field(default_factory=list)
    reason: str = ""


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

    def __init__(self, settings: Settings, prompt_library: PromptLibrary | None = None) -> None:
        self._settings = settings
        self._prompt_library = prompt_library or PromptLibrary()
        self._name_confirmation_resolver = ContextualNameConfirmationResolver(settings)
        self._llm = None
        self._knowledge_selector_llm = None
        if settings.openai_api_key:
            base_llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=0.2,
            )
            self._llm = base_llm.with_structured_output(PlannerOutput, method="function_calling")
            self._knowledge_selector_llm = base_llm.with_structured_output(
                KnowledgeSelectionOutput,
                method="function_calling",
            )

    async def plan(
        self,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str],
        semantic_memories: list[str],
    ) -> PlanningResult:
        hard_rule_result = self._apply_hard_rules(text, contact)
        if hard_rule_result is not None:
            return hard_rule_result
        if self._llm is not None:
            try:
                selection = await self._select_knowledge_sections(text, contact, recent_messages)
                knowledge_sections = self._prompt_library.get_sections_by_name(selection.section_names)
                if knowledge_sections:
                    logger.info(
                        "knowledge_lookup sections=%s reason=%s",
                        [section.name for section in knowledge_sections],
                        selection.reason.strip() or "unspecified",
                    )
                result = await self._plan_with_llm(text, contact, recent_messages, semantic_memories, knowledge_sections)
                if knowledge_sections:
                    result = result.model_copy(
                        update={
                            "knowledge_lookups": [
                                KnowledgeLookup(
                                    section=section.name,
                                    reason=selection.reason.strip() or "Selected by planner before final response.",
                                )
                                for section in knowledge_sections
                            ]
                        }
                    )
                name_confirmation = await self._resolve_name_confirmation(text, contact, recent_messages)
                repaired = self._repair_actions(result, text, contact, recent_messages, name_confirmation)
                return self._apply_planning_guardrail(repaired, text, contact, recent_messages, name_confirmation)
            except OpenAIError:
                return self._plan_with_rules(text, contact)
        result = self._plan_with_rules(text, contact)
        name_confirmation = await self._resolve_name_confirmation(text, contact, recent_messages)
        repaired = self._repair_actions(result, text, contact, recent_messages, name_confirmation)
        return self._apply_planning_guardrail(repaired, text, contact, recent_messages, name_confirmation)

    async def _plan_with_llm(
        self,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str],
        semantic_memories: list[str],
        knowledge_sections: list[PromptSection] | None = None,
    ) -> PlanningResult:
        prompt = self._compose_llm_prompt(
            contact_json=contact.model_dump_json(indent=2) if contact else "None",
            recent_messages=str(recent_messages),
            semantic_memories=str(semantic_memories),
            knowledge_context=self._render_knowledge_context(knowledge_sections or []),
            text=text,
        )
        output = await self._llm.ainvoke(prompt)
        return PlanningResult(**output.model_dump())

    def get_prompt_scaffold(self) -> str:
        return self._prompt_library.render_prompt_scaffold()

    def _compose_llm_prompt(
        self,
        *,
        contact_json: str,
        recent_messages: str,
        semantic_memories: str,
        knowledge_context: str,
        text: str,
    ) -> str:
        parts = [
            "# Core Agent",
            self._prompt_library.read_section("core_agent"),
            "",
            "# Business Rules",
            self._prompt_library.read_section("business_rules"),
            "",
            "# Knowledge Context",
            knowledge_context or "No additional knowledge loaded for this turn.",
            "",
            "Contact:",
            contact_json,
            "",
            "Recent messages:",
            recent_messages,
            "",
            "Semantic memories:",
            semantic_memories,
            "",
            "User message:",
            text,
        ]
        return "\n".join(parts).strip()

    async def _select_knowledge_sections(
        self,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str],
    ) -> KnowledgeSelectionOutput:
        if self._knowledge_selector_llm is None:
            sections = self._prompt_library.search_sections(" ".join([text, *recent_messages[-3:]]))
            return KnowledgeSelectionOutput(
                section_names=[section.name for section in sections],
                reason="Keyword fallback matched the user message and recent context.",
            )

        available = "\n".join(
            f"- {section.name}: {section.description} Tags: {', '.join(section.tags)}"
            for section in self._prompt_library.get_knowledge_sections()
        )
        selector_prompt = "\n".join(
            [
                "You decide whether the agent needs extra Wabog knowledge before responding.",
                "Select zero to three sections only when the user is asking for product facts, pricing, plans, integrations, implementation, FAQ, or other factual business information.",
                "Do not select sections for generic qualification, greeting, or scheduling-only turns.",
                "Return only section names from the available list.",
                "",
                "Available sections:",
                available,
                "",
                "Current contact:",
                contact.model_dump_json(indent=2) if contact else "None",
                "",
                "Recent messages:",
                str(recent_messages[-4:]),
                "",
                "User message:",
                text,
            ]
        )
        output = await self._knowledge_selector_llm.ainvoke(selector_prompt)
        return KnowledgeSelectionOutput(**output.model_dump())

    def _render_knowledge_context(self, sections: list[PromptSection]) -> str:
        if not sections:
            return ""
        rendered: list[str] = []
        for section in sections:
            rendered.extend(
                [
                    f"## {section.title}",
                    self._prompt_library.read_section(section.name),
                    "",
                ]
            )
        return "\n".join(rendered).strip()

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
            candidate_name = get_name_confirmation_candidate(contact)
            if contact_has_reliable_name(contact):
                response_text = f"Te tengo registrado como {full_name}."
            elif candidate_name:
                response_text = f"Te tengo como {candidate_name}, pero prefiero confirmarlo contigo. ¿Ese es tu nombre?"
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
        name_confirmation: NameConfirmationDecision | None = None,
    ) -> PlanningResult:
        repaired_actions: list[ProposedAction] = []
        contextual_name_applied = False
        for action in result.actions:
            args = dict(action.args)
            if action.type == ActionType.UPDATE_CONTACT_FIELDS:
                fields = args.get("fields")
                normalized_fields = dict(fields) if isinstance(fields, dict) else {}
                inferred_email = self._extract_email(text)
                inferred_name = self._extract_explicit_name(text)
                if inferred_email and not normalized_fields.get("email"):
                    normalized_fields["email"] = inferred_email
                if inferred_name and not normalized_fields.get("full_name"):
                    normalized_fields["full_name"] = inferred_name
                if (
                    name_confirmation is not None
                    and name_confirmation.status in {"confirmed_candidate_name", "provided_new_name"}
                    and name_confirmation.resolved_name
                    and not normalized_fields.get("full_name")
                ):
                    normalized_fields["full_name"] = name_confirmation.resolved_name
                    contextual_name_applied = True
                args["fields"] = {
                    key: str(value).strip()
                    for key, value in normalized_fields.items()
                    if key in {"email", "full_name"} and str(value).strip()
                }
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
        if (
            name_confirmation is not None
            and name_confirmation.status in {"confirmed_candidate_name", "provided_new_name"}
            and name_confirmation.resolved_name
            and not contextual_name_applied
            and (contact is None or (contact.full_name or "").strip() != name_confirmation.resolved_name)
        ):
            repaired_actions.append(
                ProposedAction(
                    type=ActionType.UPDATE_CONTACT_FIELDS,
                    reason="Validador contextual confirmó el nombre del lead.",
                    args={"fields": {"full_name": name_confirmation.resolved_name}},
                )
            )
        return result.model_copy(update={"actions": repaired_actions})

    def _apply_planning_guardrail(
        self,
        result: PlanningResult,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str] | None = None,
        name_confirmation: NameConfirmationDecision | None = None,
    ) -> PlanningResult:
        actions: list[ProposedAction] = []
        for action in result.actions:
            if action.type == ActionType.UPDATE_STAGE and not action.args.get("stage"):
                continue
            if action.type == ActionType.APPEND_NOTE and not str(action.args.get("note", "")).strip():
                continue
            if action.type == ActionType.CREATE_FOLLOWUP and not str(action.args.get("summary", "")).strip():
                continue
            actions.append(action.model_copy(deep=True))
        recent_messages = recent_messages or []
        projected_contact = self._project_contact_for_actions(contact, actions, text)
        meeting_payload = self._build_meeting_payload(text, projected_contact, recent_messages)
        missing_meeting_fields = self._missing_contact_fields_for_meeting(projected_contact)
        upcoming_event = self._get_upcoming_calendar_event(contact)

        response_text = result.response_text
        normalized_actions: list[ProposedAction] = []
        for action in actions:
            if action.type != ActionType.CREATE_MEETING:
                normalized_actions.append(action)
                continue

            merged_args = dict(action.args)
            if meeting_payload is not None:
                merged_args = dict(meeting_payload) | {key: value for key, value in merged_args.items() if value}

            if missing_meeting_fields:
                response_text = self._build_missing_meeting_fields_response(missing_meeting_fields, projected_contact)
                continue

            normalized_actions.append(action.model_copy(update={"args": merged_args}))

        actions = normalized_actions

        if (
            meeting_payload is not None
            and not missing_meeting_fields
            and upcoming_event is None
            and not any(action.type == ActionType.CREATE_MEETING for action in actions)
            and not actions
        ):
            actions.append(
                ProposedAction(
                    type=ActionType.CREATE_MEETING,
                    reason="El lead confirmó un horario concreto para la demo y ya tiene datos completos.",
                    args=dict(meeting_payload),
                )
            )

        if (
            meeting_payload is not None
            and not missing_meeting_fields
            and upcoming_event is None
            and not any(action.type == ActionType.CREATE_MEETING for action in actions)
            and name_confirmation is not None
            and name_confirmation.status in {"confirmed_candidate_name", "provided_new_name"}
        ):
            actions.append(
                ProposedAction(
                    type=ActionType.CREATE_MEETING,
                    reason="El validador contextual confirmó la identidad pendiente y el lead pidió agendar con horario concreto.",
                    args=dict(meeting_payload),
                )
            )

        if upcoming_event is None and not any(action.type == ActionType.CREATE_MEETING for action in actions):
            response_text = self._strip_false_booking_claims(response_text)

        inferred_stage = self._infer_stage_from_text(
            text,
            contact.stage if contact and contact.stage else "Prospecto",
            recent_messages,
        )
        negative_stage_signal = any(
            phrase in text.lower()
            for phrase in (
                "no quiero",
                "no me interesa",
                "no aplica",
                "no gracias",
                "despues",
                "después",
            )
        )
        if (
            inferred_stage
            and self._should_update_stage(contact.stage if contact and contact.stage else "Prospecto", inferred_stage)
            and not negative_stage_signal
            and not any(action.type == ActionType.UPDATE_STAGE for action in actions)
        ):
            actions.append(
                ProposedAction(
                    type=ActionType.UPDATE_STAGE,
                    reason="Se infiere una etapa comercial válida a partir del mensaje actual y el contexto reciente.",
                    args={"stage": inferred_stage},
                )
            )

        response_text = self._normalize_trial_response(text, response_text)

        return result.model_copy(update={"actions": actions, "response_text": response_text})

    async def _resolve_name_confirmation(
        self,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str],
    ) -> NameConfirmationDecision | None:
        return await self._name_confirmation_resolver.resolve(text, contact, recent_messages)

    def _normalize_trial_response(self, text: str, response_text: str) -> str:
        lowered = text.lower()
        if not any(token in lowered for token in ("prueba", "pruebo", "probar", "probarlo", "testear")):
            return self._normalize_wabog_urls(response_text)
        normalized = self._normalize_wabog_urls(response_text)
        wabog_url = "https://wabog.com"
        app_url = "https://app.wabog.com"
        return (
            f"Puedes probar Wabog desde {wabog_url}. "
            f"Funciona muy bien a través de WhatsApp para monitoreo y seguimiento. "
            f"Si prefieres una app más profesional para gestión de procesos, también tenemos {app_url}."
        )

    def _normalize_wabog_urls(self, response_text: str) -> str:
        return re.sub(r"https://wabog\.com/[^\s)\]]+", "https://wabog.com", response_text)

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
        return is_specific_person_name(value)

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
            projected = apply_name_validation_metadata(
                projected,
                build_trusted_name_assessment(explicit_name, source="user_message"),
            )
        for action in actions:
            if action.type != ActionType.UPDATE_CONTACT_FIELDS:
                continue
            fields = action.args.get("fields")
            if not isinstance(fields, dict):
                continue
            if str(fields.get("email", "")).strip():
                projected.email = str(fields["email"]).strip()
            if str(fields.get("full_name", "")).strip():
                resolved_name = str(fields["full_name"]).strip()
                projected.full_name = resolved_name
                projected = apply_name_validation_metadata(
                    projected,
                    build_trusted_name_assessment(resolved_name, source="user_message"),
                )
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
        lead_label = (contact.full_name or "Lead").strip() if contact and contact_has_reliable_name(contact) else "Lead"
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
        if not contact_has_reliable_name(contact):
            missing.append("full_name")
        if not (contact.email or "").strip():
            missing.append("email")
        return missing

    def _build_missing_meeting_fields_response(self, missing_fields: list[str], contact: CRMContact | None = None) -> str:
        candidate_name = get_name_confirmation_candidate(contact)
        if missing_fields == ["full_name", "email"]:
            if candidate_name:
                return (
                    f"Perfecto. Antes de dejarte la demo agendada, ¿tu nombre es {candidate_name}? "
                    "Y de una vez compárteme también tu correo para enviarte la invitación."
                )
            return (
                "Perfecto. Para dejarte la demo agendada y enviarte la invitación, "
                "compárteme tu nombre completo y tu correo."
            )
        if missing_fields == ["email"]:
            return "Perfecto. Para enviarte la invitación de la demo, compárteme tu correo."
        if missing_fields == ["full_name"]:
            if candidate_name:
                return f"Perfecto. Antes de agendarla, ¿tu nombre es {candidate_name}? Si no, compárteme tu nombre completo."
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

        if not response_lines:
            response_lines.append(
                "Perfecto. Soy el agente comercial de Wabog para abogados. Cuéntame cómo estás manejando hoy tus procesos o qué quieres mejorar, y te guío al siguiente paso."
            )

        response_text = " ".join(response_lines)
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
        current_text = text.strip()
        lowered_text = current_text.lower()
        lowered_context_messages = [message.lower() for message in context_messages]
        combined_lowered = " ".join(lowered_context_messages + [lowered_text]).strip()
        if not any(token in combined_lowered for token in ("demo", "agendar", "reunión", "reunion", "llamada", "presentación", "presentacion")):
            return None
        tz = ZoneInfo(self._settings.google_calendar_timezone)
        base_date = datetime.now(tz).date()

        relative_match = re.search(
            r"\b(hoy|mañana|manana)\b(?:.*?)(?:a\s+las\s+|a\s+la\s+|tipo\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            lowered_text,
        )
        if relative_match:
            return self._build_relative_datetime(relative_match.groups(), base_date, tz)

        for previous_message in reversed(lowered_context_messages):
            relative_match = re.search(
                r"\b(hoy|mañana|manana)\b(?:.*?)(?:a\s+las\s+|a\s+la\s+|tipo\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
                previous_message,
            )
            if relative_match:
                return self._build_relative_datetime(relative_match.groups(), base_date, tz)

        isoish_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})[ t](\d{1,2}):(\d{2})\b", lowered_text)
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
        current_weekday_match = re.search(r"\b(lunes|martes|miercoles|miércoles|jueves|viernes|sabado|sábado|domingo)\b", lowered_text)
        context_weekday_name = None
        for message in reversed(lowered_context_messages):
            if match := re.search(r"\b(lunes|martes|miercoles|miércoles|jueves|viernes|sabado|sábado|domingo)\b", message):
                context_weekday_name = match.group(1)
                break

        weekday_name = current_weekday_match.group(1) if current_weekday_match else context_weekday_name
        hour_match = re.search(r"(?:a\s+las\s+|a\s+la\s+|tipo\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lowered_text)
        if weekday_name and hour_match:
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

        bare_hour_match = re.search(r"(?:a\s+las\s+|a\s+la\s+|tipo\s+)?(\d{1,2})(?::(\d{2}))?\b", lowered_text)
        if weekday_name and bare_hour_match:
            weekday = weekday_map[weekday_name]
            days_ahead = (weekday - base_date.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = base_date + timedelta(days=days_ahead)
            hour_raw, minute_raw = bare_hour_match.groups()
            hour = int(hour_raw)
            minute = int(minute_raw or "0")
            period_context = " ".join(lowered_context_messages[-2:] + [lowered_text]).strip()
            if any(token in period_context for token in ("tarde", "pm", "noche")) and 1 <= hour <= 11:
                hour += 12
            elif "mañana" in period_context and hour == 12:
                hour = 0
            elif hour == 12 and "mediodia" not in period_context and "medio dia" not in period_context and "tarde" not in period_context:
                hour = 12
            return datetime(target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=tz)
        return None

    def _build_relative_datetime(
        self,
        match_groups: tuple[str, str, str | None, str | None],
        base_date: date,
        tz: ZoneInfo,
    ) -> datetime:
        day_token, hour_raw, minute_raw, meridiem = match_groups
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
