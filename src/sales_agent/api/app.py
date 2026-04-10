from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from sales_agent.api.routes import router
from sales_agent.core.config import Settings, get_settings
from sales_agent.services.application import SalesAgentApplication


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    application = SalesAgentApplication(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await application.startup()
        yield

    app = FastAPI(title="Sales Agent", lifespan=lifespan)
    app.state.sales_agent = application
    app.include_router(router)
    return app
