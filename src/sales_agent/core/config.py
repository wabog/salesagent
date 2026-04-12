from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = Field(default="development", alias="APP_ENV")
    database_url: str = Field(default="sqlite+aiosqlite:///./sales_agent.db", alias="DATABASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    crm_backend: str = Field(default="memory", alias="CRM_BACKEND")
    notion_api_key: str | None = Field(default=None, alias="NOTION_API_KEY")
    notion_data_source_id: str | None = Field(default=None, alias="NOTION_DATA_SOURCE_ID")
    notion_version: str = Field(default="2025-09-03", alias="NOTION_VERSION")
    notion_phone_property: str = Field(default="Telefono", alias="NOTION_PHONE_PROPERTY")
    notion_name_property: str = Field(default="Nombre", alias="NOTION_NAME_PROPERTY")
    notion_stage_property: str = Field(default="Etapa", alias="NOTION_STAGE_PROPERTY")
    notion_notes_property: str = Field(default="Notas", alias="NOTION_NOTES_PROPERTY")
    notion_next_action_property: str = Field(default="Próxima acción", alias="NOTION_NEXT_ACTION_PROPERTY")
    notion_last_contact_property: str = Field(default="Último contacto", alias="NOTION_LAST_CONTACT_PROPERTY")
    kapso_base_url: str = Field(default="https://api.kapso.ai", alias="KAPSO_BASE_URL")
    kapso_phone_number_id: str | None = Field(default=None, alias="KAPSO_PHONE_NUMBER_ID")
    kapso_api_token: str | None = Field(default=None, alias="KAPSO_API_TOKEN")
    whatsapp_send_enabled: bool = Field(default=False, alias="WHATSAPP_SEND_ENABLED")
    message_batch_window_seconds: float = Field(default=3.0, alias="MESSAGE_BATCH_WINDOW_SECONDS")
    playground_enabled: bool | None = Field(default=None, alias="PLAYGROUND_ENABLED")
    playground_token: str | None = Field(default=None, alias="PLAYGROUND_TOKEN")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def is_playground_enabled(self) -> bool:
        if self.playground_enabled is not None:
            return self.playground_enabled
        return self.app_env != "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
