from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sales_agent.adapters.channel import ConsoleChannelAdapter, KapsoWhatsAppAdapter
from sales_agent.adapters.crm_memory import InMemoryCRMAdapter
from sales_agent.adapters.crm_notion import NotionCRMAdapter
from sales_agent.adapters.memory_sql import SqlAlchemyMemoryStore
from sales_agent.core.config import Settings
from sales_agent.core.db import build_engine, build_session_factory, init_db
from sales_agent.graph.workflow import SalesAgentWorkflow
from sales_agent.services.planner import AgentPlanner
from sales_agent.services.policy import ToolExecutionPolicy


class SalesAgentApplication:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = build_engine(settings.database_url)
        self.session_factory: async_sessionmaker[AsyncSession] = build_session_factory(self.engine)
        self.memory_store = SqlAlchemyMemoryStore(self.session_factory)
        self.crm_adapter = self._build_crm_adapter()
        self.channel_adapter = self._build_channel_adapter()
        self.workflow = SalesAgentWorkflow(
            crm_adapter=self.crm_adapter,
            memory_store=self.memory_store,
            channel_adapter=self.channel_adapter,
            planner=AgentPlanner(settings),
            policy=ToolExecutionPolicy(),
        )

    async def startup(self) -> None:
        await init_db(self.engine)

    def _build_crm_adapter(self):
        if self.settings.crm_backend == "notion":
            return NotionCRMAdapter(self.settings)
        return InMemoryCRMAdapter()

    def _build_channel_adapter(self):
        if self.settings.kapso_api_token and self.settings.kapso_phone_number_id:
            return KapsoWhatsAppAdapter(self.settings)
        return ConsoleChannelAdapter()
