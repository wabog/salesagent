# Agent Behavior And Guardrail Design

This file captures non-negotiable design rules for the Wabog sales agent and should be read before changing prompts, planner logic, validators, or booking guardrails.

## Non-negotiable rule

- Guardrails must not depend on fixed keyword lists or exact phrase matching when the decision depends on meaning.
- This applies especially to identity confirmation, consent, completion, schedule confirmation, follow-up completion, and similar conversational states.
- Those decisions must be resolved by contextual reasoning, ideally through a dedicated small validator or mini-agent with scoped input and structured output.
- Hardcoded keywords may still be acceptable for narrow deterministic parsing such as dates, emails, or phone normalization, but not for semantic confirmation of user intent.

## Why this exists

- We already saw a production loop where the lead confirmed the suggested name naturally, but the system repeated the same validation question.
- The failure mode came from mixing a contextual business decision with rigid phrase extraction.
- When a guardrail blocks a critical action like `CREATE_MEETING`, any semantic ambiguity must be resolved contextually, not through a fixed phrase whitelist.

## Candidate name flow

`candidate_name` is the provisional name the system stores when a provider-supplied contact name looks plausible but is not trusted enough to be used as final CRM identity.

Current flow in code:

1. WhatsApp/Kapso inbound payload provides `conversation.contact_name` or `conversation.name`.
2. That value enters `InboundMessage.contact_name`.
3. `ContactNameValidator.assess_provider_name(...)` evaluates it.
4. The validator returns one of these statuses:
   - `trusted`: looks like a sufficiently reliable real person name
   - `needs_confirmation`: plausible name, but not reliable enough to treat as final
   - `rejected`: placeholder, role, phone-like string, generic label, or clearly invalid
5. `apply_name_validation_metadata(...)` stores the assessment in `contact.metadata["name_validation"]`.
6. If status is `needs_confirmation`, the system exposes `candidate_name` through `get_name_confirmation_candidate(...)`.
7. Booking guardrails currently treat a `needs_confirmation` name as missing until it becomes trusted or confirmed.

## How a name becomes doubtful

The current validator does two layers:

1. Deterministic filters
   - Rejects generic placeholders like `user`, `usuario`, `lead`, `unknown`
   - Rejects role-like tokens such as `abogado`, `doctor`, `cliente`, `wabog`
   - Rejects phone-like strings
   - Trusts multi-token names that already look specific enough
2. LLM fallback
   - If the provider name is not obviously trusted or rejected, the validator asks a model whether it looks like a real person name for CRM use
   - If confidence is high enough, it returns `trusted`
   - If it looks plausible but not fully reliable, it returns `needs_confirmation`

This is why `candidate_name` exists: it is not a confirmed identity, only a plausible provider-origin hint.

## Architectural rule for future changes

- Never solve semantic confirmation problems by adding more fixed expressions such as `si`, `correcto`, `ese es mi nombre`, or similar.
- If the product needs to understand whether the user confirmed a suggested identity, use a small contextual validator that sees:
  - latest user turn
  - latest agent question
  - current candidate name
  - recent conversation context
- That validator should return a structured decision such as:
  - `confirmed_candidate_name`
  - `provided_new_name`
  - `rejected_candidate_name`
  - `unclear`
- Only then should the planner or application decide whether to unblock contact updates or meeting creation.

## Files involved today

- `src/sales_agent/adapters/whatsapp.py`
- `src/sales_agent/services/name_validation.py`
- `src/sales_agent/services/planner.py`
- `src/sales_agent/services/application.py`

## Change policy

- Before editing guardrails, check this file first.
- If a guardrail is about semantics, context, confirmation, or human intent, prefer a scoped validator or mini-agent over hardcoded word matching.
