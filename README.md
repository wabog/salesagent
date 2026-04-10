# Sales Agent

Servicio propio para un agente comercial que recibe mensajes de WhatsApp, recupera contexto del lead, ejecuta tools controladas sobre el CRM y responde usando un flujo orquestado con LangGraph.

## Qué incluye

- `FastAPI` como API de entrada.
- `LangGraph` para el flujo del agente.
- `CRMAdapter` desacoplado para poder usar Notion hoy y otro CRM mañana.
- `SQLAlchemy` para persistencia de mensajes, runs y shadow cache del contacto.
- Modo heurístico local si no hay modelo configurado.

## Ejecutar localmente

Camino recomendado con `uv`:

```bash
cp .env.example .env
uv sync --group dev
uv run uvicorn sales_agent.main:app --reload
```

Atajos opcionales:

```bash
make setup
make dev
make test
```

Si no tienes Python `3.13` instalado, `uv` lo resuelve con:

```bash
uv python install 3.13
```

La ruta antigua con `venv + pip` sigue funcionando, pero ya no es el flujo principal.

## Variables de entorno

Revisa [`.env.example`](/home/gidiom/Wabog/salesAgent/.env.example). Para desarrollo, el servicio funciona con SQLite y un CRM en memoria. Para producción, usa `Postgres` en `DATABASE_URL` y configura `CRM_BACKEND=notion`.

## Endpoints

- `POST /webhooks/whatsapp/kapso`
- `POST /internal/replay`
- `GET /healthz`

## Flujo

1. Normaliza el payload de WhatsApp.
2. Aplica idempotencia por `message_id`.
3. Carga contacto, estado conversacional e historial.
4. Clasifica intención y planea acciones.
5. Ejecuta tools permitidas.
6. Persiste run y mensajes.
7. Envía respuesta por el canal configurado.
