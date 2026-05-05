import pytest

from sales_agent.core.config import Settings
from sales_agent.domain.models import CRMContact
from sales_agent.services.name_validation import (
    ContactNameValidator,
    NameCandidateAssessment,
    apply_name_validation_metadata,
    contact_has_reliable_name,
    get_name_confirmation_candidate,
)


@pytest.mark.asyncio
async def test_name_validator_rejects_generic_provider_name():
    validator = ContactNameValidator(Settings(OPENAI_API_KEY=""))

    result = await validator.assess_provider_name("user")

    assert result.status == "rejected"


@pytest.mark.asyncio
async def test_name_validator_marks_single_token_name_for_confirmation():
    validator = ContactNameValidator(Settings(OPENAI_API_KEY=""))

    result = await validator.assess_provider_name("juan")

    assert result.status == "needs_confirmation"
    assert result.normalized_name == "Juan"


@pytest.mark.asyncio
async def test_name_validator_trusts_multi_token_human_name():
    validator = ContactNameValidator(Settings(OPENAI_API_KEY=""))

    result = await validator.assess_provider_name("juan david perez")

    assert result.status == "trusted"
    assert result.normalized_name == "Juan David Perez"


@pytest.mark.asyncio
async def test_name_validator_trusts_name_with_middle_initial():
    validator = ContactNameValidator(Settings(OPENAI_API_KEY=""))

    result = await validator.assess_provider_name("fabian c villegas")

    assert result.status == "trusted"
    assert result.normalized_name == "Fabian C Villegas"


def test_contact_name_helpers_use_persisted_validation_status():
    contact = CRMContact(
        external_id="lead-1",
        phone_number="3150000000",
        full_name=None,
    )
    confirmed = apply_name_validation_metadata(
        contact,
        NameCandidateAssessment(
            status="needs_confirmation",
            confidence=0.62,
            normalized_name="Juan",
            candidate_name="Juan",
            source="provider",
        ),
    )
    assert not contact_has_reliable_name(confirmed)
    assert get_name_confirmation_candidate(confirmed) == "Juan"

    trusted = apply_name_validation_metadata(
        confirmed,
        NameCandidateAssessment(
            status="trusted",
            confidence=0.99,
            normalized_name="Juan David Perez",
            candidate_name="Juan David Perez",
            source="user_message",
        ),
    )
    assert trusted.full_name == "Juan David Perez"
    assert contact_has_reliable_name(trusted)
