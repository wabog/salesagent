# Outbound Campaigns

This document describes the reusable outbound campaign infrastructure for Wabog's sales agent.
It documents the code that exists today; no production campaign config has been created yet.

## Purpose

Outbound campaigns let Wabog send approved WhatsApp template messages to CRM leads while preserving agent context.
The first outbound message is saved in the same message memory used by inbound processing, so when the lead replies the normal inbound agent can continue with the correct context.

The implementation is generic: future campaigns should be added as config files instead of adding campaign-specific Python code.

## Current Components

- `src/sales_agent/services/outbound_campaigns.py`
  - Loads campaign configs from JSON or YAML.
  - Seeds eligible CRM contacts into `outbound_recipients`.
  - Runs due sends.
  - Supports one-time and recurring schedules.
  - Renders template parameters.
  - Writes outbound memory and post-send CRM updates.
- `src/sales_agent/outbound_cli.py`
  - CLI entrypoint for seeding, running, and scheduler ticks.
- `src/sales_agent/adapters/channel.py`
  - Adds `send_template(...)` for Kapso WhatsApp templates.
- `src/sales_agent/adapters/crm_notion.py`
  - Adds generic Notion property filters for campaign seeding.
- `src/sales_agent/core/db.py`
  - Adds `outbound_campaigns` and `outbound_recipients`.
- `src/sales_agent/services/name_validation.py`
  - Adds AI-based outbound greeting name validation.

## Data Model

`outbound_campaigns` stores the current campaign definition and scheduler state:

- `campaign_id`
- `name`
- `status`
- `config_json`
- `last_seeded_at`
- `last_run_at`

`outbound_recipients` stores recipient-level state:

- `recipient_id`
- `campaign_id`
- `phone_number`
- `external_id`
- `conversation_id`
- `status`
- `scheduled_at`
- `sent_at`
- `attempts`
- `provider_message_id`
- `rendered_text`
- `error`
- `metadata_json`

There is a unique constraint on `campaign_id + phone_number` to prevent duplicate sends inside the same campaign.

## Send Flow

1. `seed` or `scheduler-tick` loads a campaign config.
2. The service queries Notion using `crm_filter.conditions`.
3. Matching contacts become `outbound_recipients` with `status = scheduled`.
4. `run-once` or `scheduler-tick` selects due recipients.
5. Before sending, the service re-checks the current contact by phone.
6. If the contact no longer matches the campaign stage filter, the recipient is skipped.
7. Template parameters are rendered.
8. Kapso sends the approved WhatsApp template.
9. If Kapso accepts the send:
   - the rendered outbound text is saved to `messages` with `direction = OUTBOUND`
   - CRM stage and note are updated if configured
   - the contact shadow is refreshed
   - the recipient is marked `sent`
10. If send fails, the recipient is marked `failed` with the error.

The inbound agent flow is unchanged. When a lead replies, the existing webhook path loads recent messages by phone and sees the saved outbound message.

## Greeting Name Validation

Outbound greeting names are validated with AI when `OPENAI_API_KEY` is configured.

The service calls `ContactNameValidator.assess_outbound_greeting_name(...)` for `lead_name` parameters. The name is used only when the validator returns `trusted` with a confidence of at least `0.8`.

If the AI rejects the value, is uncertain, or errors, the parameter falls back to the configured fallback value.

For local development without OpenAI credentials, the code has a conservative heuristic fallback so tests and dry development can run. Production should rely on the AI validator.

## Scheduling

Two schedule modes are supported.

### One-Time

Use `once_at` for a fixed campaign send time:

```json
{
  "schedule": {
    "mode": "once_at",
    "timezone": "America/Bogota",
    "send_at": "2026-05-13 09:00",
    "max_per_run": 25,
    "jitter_seconds": 120
  }
}
```

### Recurring

Use `recurring` when new matching leads should be picked up at specific times:

```json
{
  "schedule": {
    "mode": "recurring",
    "timezone": "America/Bogota",
    "days": ["mon", "tue", "wed", "thu", "fri"],
    "times": ["09:00"],
    "max_per_run": 25,
    "jitter_seconds": 120
  }
}
```

The scheduler tracks `last_seeded_at`, so the same due schedule is not seeded repeatedly.
Recipient-level dedupe prevents the same phone from receiving the same campaign twice.

## CLI

Run with the configured app environment:

```bash
python -m sales_agent.outbound_cli seed path/to/campaign.json
python -m sales_agent.outbound_cli run-once path/to/campaign.json
python -m sales_agent.outbound_cli scheduler-tick path/to/campaign.json
python -m sales_agent.outbound_cli scheduler-tick path/to/campaigns-dir
```

Recommended production pattern:

```bash
python -m sales_agent.outbound_cli scheduler-tick path/to/campaigns-dir
```

Run that command from cron or the deployment scheduler every minute or every five minutes.

## Config Format

JSON works without additional dependencies. YAML is supported only if `PyYAML` is installed in the environment.

Example shape:

```json
{
  "id": "example_campaign",
  "name": "Example campaign",
  "status": "active",
  "crm_filter": {
    "conditions": [
      { "property": "Fuente", "type": "select", "equals": "facebook" },
      { "property": "Etapa", "type": "status", "equals": "Nuevo Lead" }
    ]
  },
  "template": {
    "name": "wabog_gentle_reactivation",
    "language": "es",
    "body_text": "Hola {{1}}, soy {{2}} de Wabog. Queria saber si aun te interesa conocer la plataforma. Si quieres, te cuento por aqui.",
    "parameter_format": "POSITIONAL",
    "phone_number_id": "1101272743059797",
    "parameters": [
      { "type": "lead_name", "fallback": "buen día" },
      { "type": "static", "value": "Fabian" }
    ]
  },
  "after_send": {
    "stage": "Primer contacto",
    "note": "Primer contacto outbound enviado por campaña {campaign_id} con {template_name} el {sent_at}."
  },
  "schedule": {
    "mode": "recurring",
    "timezone": "America/Bogota",
    "days": ["mon", "tue", "wed", "thu", "fri"],
    "times": ["09:00"],
    "max_per_run": 25,
    "jitter_seconds": 120
  }
}
```

## Current Decisions

- Do not move a lead to `Primer contacto` before Kapso accepts the outbound send.
- Store the rendered text in memory, not only the template name.
- Keep outbound orchestration outside the planner. The normal inbound agent takes over only after the lead replies.
- Use `campaign_id + phone_number` dedupe for safety.
- Use AI validation for greeting names and fail closed to the configured fallback.

## What Is Still Missing Before First Campaign

- Create the actual campaign config file.
- Choose one-time or recurring schedule.
- Confirm production env vars:
  - `CRM_BACKEND=notion`
  - `NOTION_API_KEY`
  - `NOTION_DATA_SOURCE_ID`
  - `KAPSO_PHONE_NUMBER_ID`
  - `KAPSO_API_TOKEN`
  - `WHATSAPP_SEND_ENABLED=true`
  - `OPENAI_API_KEY`
- Add the `scheduler-tick` command to the production scheduler.
- Run an initial dry review of matching Notion leads before enabling sends.

## Verification

Automated coverage exists in:

- `tests/test_channel.py`
- `tests/test_outbound_campaigns.py`
- `tests/test_name_validation.py`

Latest local run after implementation:

```text
107 passed
```
