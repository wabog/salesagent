from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

from sales_agent.core.config import Settings
from sales_agent.domain.models import InboundMessage


DOCUMENT_UNSUPPORTED_REPLY = (
    "Recibí el archivo, pero por ahora no puedo procesar documentos directamente por este canal. "
    "Si quieres, cuéntame qué necesitas revisar y te ayudo por aquí."
)


@dataclass
class MediaPreprocessingResult:
    text: str = ""
    should_reply: bool = True
    bypass_intent: str | None = None
    bypass_response_text: str | None = None


class InboundMediaPreprocessor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    async def preprocess_event(self, event: InboundMessage) -> MediaPreprocessingResult:
        message_type = (event.message_type or "text").lower()
        if message_type == "text":
            return MediaPreprocessingResult(text=event.text.strip())
        if message_type == "audio":
            return MediaPreprocessingResult(text=(await self._transcribe_audio(event)).strip())
        if message_type == "image":
            return MediaPreprocessingResult(text=(await self._describe_image(event)).strip())
        if message_type == "document":
            return MediaPreprocessingResult(
                bypass_intent="document_unsupported",
                bypass_response_text=DOCUMENT_UNSUPPORTED_REPLY,
            )
        if message_type == "sticker":
            return MediaPreprocessingResult(should_reply=False)
        fallback_text = event.text.strip()
        if fallback_text:
            return MediaPreprocessingResult(text=fallback_text)
        return MediaPreprocessingResult(should_reply=False)

    async def _transcribe_audio(self, event: InboundMessage) -> str:
        if self._client is not None and event.media_url:
            try:
                audio_bytes = await self._download_media(event.media_url)
                buffer = io.BytesIO(audio_bytes)
                buffer.name = event.media_filename or "audio.ogg"
                transcript = await self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=buffer,
                )
                text = str(getattr(transcript, "text", "")).strip()
                if text:
                    return text
            except Exception:
                pass
        if event.media_transcript:
            return event.media_transcript.strip()
        return event.text.strip()

    async def _describe_image(self, event: InboundMessage) -> str:
        prompt = (
            "Resume solo el contexto util para una conversacion comercial de WhatsApp. "
            "No describas detalles visuales irrelevantes. Extrae si hay texto legible, herramientas, procesos, "
            "dolores operativos o solicitudes claras del usuario. Responde en espanol neutro en 1 a 3 frases."
        )
        if self._client is not None and event.media_url:
            try:
                image_bytes = await self._download_media(event.media_url)
                media_type = event.media_content_type or "image/jpeg"
                data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
                response = await self._client.responses.create(
                    model=self._settings.openai_model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": data_url},
                            ],
                        }
                    ],
                )
                text = str(getattr(response, "output_text", "")).strip()
                if text:
                    return text
            except Exception:
                pass
        return event.text.strip()

    async def _download_media(self, media_url: str) -> bytes:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(media_url)
            response.raise_for_status()
            return response.content
