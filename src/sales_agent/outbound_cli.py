from __future__ import annotations

import argparse
import asyncio
import json

from sales_agent.adapters.channel import ConsoleChannelAdapter, KapsoWhatsAppAdapter
from sales_agent.adapters.crm_notion import NotionCRMAdapter
from sales_agent.adapters.memory_sql import SqlAlchemyMemoryStore
from sales_agent.core.config import Settings
from sales_agent.core.db import build_engine, build_session_factory, init_db
from sales_agent.services.name_validation import ContactNameValidator
from sales_agent.services.outbound_campaigns import (
    OutboundCampaignService,
    load_campaign_config,
    load_campaign_configs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage outbound WhatsApp campaign jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed", help="Create scheduled recipients from a campaign config.")
    seed.add_argument("config", help="Campaign config path (.json, .yaml, or .yml).")

    run_once = subparsers.add_parser("run-once", help="Send due scheduled recipients for a campaign.")
    run_once.add_argument("config", help="Campaign config path (.json, .yaml, or .yml).")

    tick = subparsers.add_parser("scheduler-tick", help="Seed and send due recipients for one or more configs.")
    tick.add_argument("configs", nargs="+", help="Campaign config file(s) or directories.")

    return parser


async def _build_service(settings: Settings) -> tuple[OutboundCampaignService, object]:
    engine = build_engine(settings.database_url)
    await init_db(engine)
    session_factory = build_session_factory(engine)
    if settings.crm_backend != "notion":
        raise RuntimeError("Outbound campaigns require CRM_BACKEND=notion.")
    crm_adapter = NotionCRMAdapter(settings)
    memory_store = SqlAlchemyMemoryStore(session_factory)
    channel_adapter = (
        KapsoWhatsAppAdapter(settings)
        if settings.kapso_api_token and settings.kapso_phone_number_id
        else ConsoleChannelAdapter()
    )
    service = OutboundCampaignService(
        session_factory=session_factory,
        crm_adapter=crm_adapter,
        memory_store=memory_store,
        channel_adapter=channel_adapter,
        name_validator=ContactNameValidator(settings),
    )
    return service, engine


async def main_async(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service, engine = await _build_service(Settings())
    try:
        if args.command == "seed":
            result = await service.seed_campaign(load_campaign_config(args.config))
            print(result.model_dump_json(indent=2))
            return 0
        if args.command == "run-once":
            result = await service.run_due(load_campaign_config(args.config))
            print(result.model_dump_json(indent=2))
            return 0
        if args.command == "scheduler-tick":
            payload = []
            for config in load_campaign_configs(args.configs):
                seed_result, run_result = await service.scheduler_tick(config)
                payload.append(
                    {
                        "campaign_id": config.id,
                        "seed": seed_result.model_dump(mode="json") if seed_result else None,
                        "run": run_result.model_dump(mode="json"),
                    }
                )
            print(json.dumps(payload, indent=2, default=str))
            return 0
        raise RuntimeError(f"Unsupported command: {args.command}")
    finally:
        await engine.dispose()


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
