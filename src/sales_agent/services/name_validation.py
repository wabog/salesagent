from __future__ import annotations

import re
from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAIError
from pydantic import BaseModel, Field

from sales_agent.core.config import Settings
from sales_agent.domain.models import CRMContact


class NameValidationOutput(BaseModel):
    looks_like_real_name: bool
    confidence: float = Field(ge=0.0, le=1.0)
    normalized_name: str | None = None


class NameCandidateAssessment(BaseModel):
    status: str
    confidence: float = 0.0
    normalized_name: str | None = None
    candidate_name: str | None = None
    source: str = "provider"


class NameConfirmationDecision(BaseModel):
    status: str = "unclear"
    confidence: float = 0.0
    resolved_name: str | None = None
    source: str = "contextual_validator"


class NameConfirmationOutput(BaseModel):
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolved_name: str | None = None


_GENERIC_NAMES = {
    "",
    "~",
    "-",
    "user",
    "usuario",
    "lead",
    "unknown",
    "sin nombre",
    "playground user",
}

_NON_NAME_TOKENS = {
    "abogado",
    "abogada",
    "doctor",
    "doctora",
    "dr",
    "dra",
    "lic",
    "cliente",
    "prospecto",
    "contacto",
    "equipo",
    "legal",
    "wabog",
}

_PROVIDER_NAME_SOURCES = {"provider", "provider_llm"}


def normalize_person_name(value: str | None) -> str | None:
    cleaned = " ".join((value or "").strip().split())
    if not cleaned:
        return None
    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ'`-]+", cleaned)
    if not tokens:
        return None
    return " ".join(token[:1].upper() + token[1:].lower() for token in tokens)


def _normalize_lookup(value: str | None) -> str:
    normalized = " ".join((value or "").strip().lower().split())
    normalized = normalized.translate(str.maketrans("áéíóúü", "aeiouu"))
    normalized = re.sub(r"[^a-z0-9ñ\\s'-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def is_specific_person_name(value: str | None) -> bool:
    normalized = _normalize_lookup(value)
    if not normalized or normalized in _GENERIC_NAMES:
        return False
    if re.fullmatch(r"\+?\d[\d\s-]{5,}", normalized):
        return False
    tokens = [token for token in re.split(r"\s+", normalized) if token]
    if not tokens:
        return False
    if any(token in _NON_NAME_TOKENS for token in tokens):
        return False
    if any(not re.fullmatch(r"[a-zñ'-]{1,30}", token) for token in tokens):
        return False
    long_tokens = [token for token in tokens if re.fullmatch(r"[a-zñ'-]{2,30}", token)]
    return len(long_tokens) >= 2


def contact_has_reliable_name(contact: CRMContact | None) -> bool:
    if contact is None or not is_specific_person_name(contact.full_name):
        return False
    validation = ((contact.metadata or {}).get("name_validation") or {})
    status = str(validation.get("status") or "").strip().lower()
    source = str(validation.get("source") or "").strip().lower()
    if not status:
        return True
    if source in _PROVIDER_NAME_SOURCES:
        return False
    return status in {"trusted", "confirmed", "legacy"}


def get_name_confirmation_candidate(contact: CRMContact | None) -> str | None:
    if contact is None:
        return None
    validation = ((contact.metadata or {}).get("name_validation") or {})
    if str(validation.get("status") or "").strip().lower() != "needs_confirmation":
        return None
    candidate = str(validation.get("candidate_name") or validation.get("normalized_name") or "").strip()
    return candidate or None


def get_effective_contact_name(contact: CRMContact | None) -> str | None:
    if contact is None:
        return None
    if contact_has_reliable_name(contact):
        return (contact.full_name or "").strip() or None
    return None


def apply_name_validation_metadata(
    contact: CRMContact,
    assessment: NameCandidateAssessment | None,
) -> CRMContact:
    if assessment is None:
        return contact
    metadata = dict(contact.metadata or {})
    provider_name = (metadata.get("provider_name") or "").strip() or None
    confirmed_name = (metadata.get("confirmed_name") or "").strip() or None
    if assessment.source in _PROVIDER_NAME_SOURCES and assessment.normalized_name:
        provider_name = assessment.normalized_name
        metadata["provider_name"] = provider_name
    if assessment.source in _PROVIDER_NAME_SOURCES and confirmed_name:
        if provider_name and confirmed_name and provider_name == confirmed_name:
            metadata.pop("provider_name", None)
        return contact.model_copy(update={"metadata": metadata})
    metadata["name_validation"] = {
        "status": assessment.status,
        "confidence": assessment.confidence,
        "normalized_name": assessment.normalized_name,
        "candidate_name": assessment.candidate_name,
        "source": assessment.source,
    }
    updated_name = contact.full_name
    if assessment.source not in _PROVIDER_NAME_SOURCES and assessment.status in {"trusted", "confirmed", "legacy"} and assessment.normalized_name:
        updated_name = assessment.normalized_name
        confirmed_name = assessment.normalized_name
        metadata["confirmed_name"] = confirmed_name
    elif confirmed_name:
        metadata["confirmed_name"] = confirmed_name
    if provider_name and confirmed_name and provider_name == confirmed_name:
        metadata.pop("provider_name", None)
    return contact.model_copy(update={"full_name": updated_name, "metadata": metadata})


def build_trusted_name_assessment(full_name: str, source: str) -> NameCandidateAssessment:
    normalized_name = normalize_person_name(full_name) or full_name.strip()
    return NameCandidateAssessment(
        status="trusted",
        confidence=0.99,
        normalized_name=normalized_name,
        candidate_name=normalized_name,
        source=source,
    )


class ContactNameValidator:
    def __init__(self, settings: Settings) -> None:
        self._llm = None
        if settings.openai_api_key:
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=0.0,
            ).with_structured_output(NameValidationOutput, method="function_calling")

    async def assess_provider_name(self, candidate: str | None) -> NameCandidateAssessment:
        normalized_name = normalize_person_name(candidate)
        normalized_lookup = _normalize_lookup(candidate)
        if not normalized_name or normalized_lookup in _GENERIC_NAMES:
            return NameCandidateAssessment(status="rejected", confidence=0.0, source="provider")
        if re.fullmatch(r"\+?\d[\d\s-]{5,}", normalized_lookup):
            return NameCandidateAssessment(status="rejected", confidence=0.0, source="provider")

        tokens = [token for token in normalized_lookup.split() if token]
        if any(token in _NON_NAME_TOKENS for token in tokens):
            return NameCandidateAssessment(
                status="rejected",
                confidence=0.05,
                normalized_name=normalized_name,
                candidate_name=normalized_name,
                source="provider",
            )

        if len(tokens) >= 2 and is_specific_person_name(normalized_name):
            return NameCandidateAssessment(
                status="needs_confirmation",
                confidence=0.96,
                normalized_name=normalized_name,
                candidate_name=normalized_name,
                source="provider",
            )

        if self._llm is not None:
            llm_assessment = await self._assess_with_llm(candidate, normalized_name)
            if llm_assessment is not None:
                return llm_assessment

        return NameCandidateAssessment(
            status="needs_confirmation",
            confidence=0.55,
            normalized_name=normalized_name,
            candidate_name=normalized_name,
            source="provider",
        )

    async def _assess_with_llm(
        self,
        candidate: str,
        normalized_name: str | None,
    ) -> NameCandidateAssessment | None:
        try:
            output = await self._llm.ainvoke(
                "\n".join(
                    [
                        "Decide if this text looks like a real person's name for a sales CRM.",
                        "Be conservative.",
                        "Reject placeholders, roles, product names, nicknames, and generic labels.",
                        f"Candidate: {candidate}",
                        f"Normalized: {normalized_name or 'None'}",
                    ]
                )
            )
        except OpenAIError:
            return None

        if output.looks_like_real_name and output.confidence >= 0.9 and is_specific_person_name(output.normalized_name or normalized_name):
            trusted_name = normalize_person_name(output.normalized_name or normalized_name)
            return NameCandidateAssessment(
                status="needs_confirmation",
                confidence=output.confidence,
                normalized_name=trusted_name,
                candidate_name=trusted_name,
                source="provider_llm",
            )
        if output.looks_like_real_name and output.confidence >= 0.55:
            candidate_name = normalize_person_name(output.normalized_name or normalized_name)
            return NameCandidateAssessment(
                status="needs_confirmation",
                confidence=output.confidence,
                normalized_name=candidate_name,
                candidate_name=candidate_name,
                source="provider_llm",
            )
        return NameCandidateAssessment(
            status="rejected",
            confidence=output.confidence,
            normalized_name=normalize_person_name(output.normalized_name or normalized_name),
            candidate_name=normalize_person_name(output.normalized_name or normalized_name),
            source="provider_llm",
        )


class ContextualNameConfirmationResolver:
    def __init__(self, settings: Settings) -> None:
        self._llm = None
        if settings.openai_api_key:
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=0.0,
            ).with_structured_output(NameConfirmationOutput, method="function_calling")

    async def resolve(
        self,
        text: str,
        contact: CRMContact | None,
        recent_messages: list[str],
    ) -> NameConfirmationDecision | None:
        candidate_name = get_name_confirmation_candidate(contact)
        if contact is None or self._llm is None:
            return None

        prompt = "\n".join(
            [
                "You are deciding whether the latest user turn establishes a reliable lead name for CRM use.",
                "Return one decision only:",
                "- confirmed_candidate_name",
                "- provided_new_name",
                "- rejected_candidate_name",
                "- unclear",
                "Use context, not keywords.",
                "Use provided_new_name only when the lead clearly established a real person full name that should replace or fill the CRM name.",
                "Use confirmed_candidate_name only when the lead confirmed the pending candidate name from context.",
                "If the recent agent turn asked whether the candidate name is correct and the latest user turn naturally confirms it, return confirmed_candidate_name even when the same turn also includes qualification details, scheduling details, or another answer.",
                "Use rejected_candidate_name only when the lead clearly rejects the pending candidate name.",
                "If the latest user message is only scheduling, pricing, thanks, email, operational details, or anything else that does not establish a reliable person name, return unclear.",
                "Continuing the booking flow is not enough by itself to confirm a name.",
                "Be conservative. If there is any real doubt, return unclear.",
                "",
                f"Current CRM full_name: {(contact.full_name or '').strip() or 'None'}",
                f"Candidate name pending confirmation: {candidate_name or 'None'}",
                f"Recent conversation: {recent_messages[-6:]}",
                f"Latest user message: {text}",
            ]
        )
        try:
            output = await self._llm.ainvoke(prompt)
        except OpenAIError:
            return None

        decision = str(output.decision or "").strip().lower() or "unclear"
        resolved_name = normalize_person_name(output.resolved_name)
        if decision == "confirmed_candidate_name":
            resolved_name = candidate_name
        if decision != "provided_new_name":
            resolved_name = resolved_name if decision == "confirmed_candidate_name" else None
        return NameConfirmationDecision(
            status=decision,
            confidence=output.confidence,
            resolved_name=resolved_name,
            source="contextual_validator",
        )
