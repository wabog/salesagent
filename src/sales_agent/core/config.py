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
    notion_email_property: str = Field(default="Email", alias="NOTION_EMAIL_PROPERTY")
    notion_followup_summary_property: str = Field(default="Resumen seguimiento", alias="NOTION_FOLLOWUP_SUMMARY_PROPERTY")
    notion_next_action_property: str = Field(default="Próxima acción", alias="NOTION_NEXT_ACTION_PROPERTY")
    notion_last_contact_property: str = Field(default="Último contacto", alias="NOTION_LAST_CONTACT_PROPERTY")
    kapso_base_url: str = Field(default="https://api.kapso.ai", alias="KAPSO_BASE_URL")
    kapso_phone_number_id: str | None = Field(default=None, alias="KAPSO_PHONE_NUMBER_ID")
    kapso_api_token: str | None = Field(default=None, alias="KAPSO_API_TOKEN")
    whatsapp_send_enabled: bool = Field(default=False, alias="WHATSAPP_SEND_ENABLED")
    google_client_id: str | None = Field(default=None, alias="GOOGLE_CLIENT_ID")
    google_client_secret: str | None = Field(default=None, alias="GOOGLE_CLIENT_SECRET")
    google_refresh_token: str | None = Field(default=None, alias="GOOGLE_REFRESH_TOKEN")
    google_calendar_id: str = Field(default="primary", alias="GOOGLE_CALENDAR_ID")
    google_calendar_timezone: str = Field(default="America/Bogota", alias="GOOGLE_CALENDAR_TIMEZONE")
    google_calendar_default_meeting_minutes: int = Field(default=30, alias="GOOGLE_CALENDAR_DEFAULT_MEETING_MINUTES")
    google_calendar_create_meet: bool = Field(default=True, alias="GOOGLE_CALENDAR_CREATE_MEET")
    google_calendar_self_schedule_url: str | None = Field(default=None, alias="GOOGLE_CALENDAR_SELF_SCHEDULE_URL")
    message_batch_window_seconds: float = Field(default=3.0, alias="MESSAGE_BATCH_WINDOW_SECONDS")
    recent_message_context_limit: int = Field(default=24, alias="RECENT_MESSAGE_CONTEXT_LIMIT")
    semantic_memory_limit: int = Field(default=10, alias="SEMANTIC_MEMORY_LIMIT")
    phone_context_max_age_days: int = Field(default=14, alias="PHONE_CONTEXT_MAX_AGE_DAYS")
    crm_notes_context_limit: int = Field(default=8, alias="CRM_NOTES_CONTEXT_LIMIT")
    playground_enabled: bool | None = Field(default=None, alias="PLAYGROUND_ENABLED")
    playground_token: str | None = Field(default=None, alias="PLAYGROUND_TOKEN")
    phone_default_country_code: str = Field(default="57", alias="PHONE_DEFAULT_COUNTRY_CODE")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def is_playground_enabled(self) -> bool:
        if self.playground_enabled is not None:
            return self.playground_enabled
        return self.app_env != "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
