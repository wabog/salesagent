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

## Tests

Suite local, sin depender de proveedores externos:

```bash
uv run pytest -q
```

Smoke tests reales contra Notion y OpenAI:

```bash
uv run pytest -q -m integration tests/test_live_integrations.py
```

Esos tests usan la configuración activa del entorno y validan:
- que Notion responde una consulta real sobre el CRM configurado
- que OpenAI responde una llamada real del planner

## Variables de entorno

Revisa [`.env.example`](/home/gidiom/Wabog/salesAgent/.env.example). Para desarrollo, el servicio funciona con SQLite y un CRM en memoria. Para producción, usa `Postgres` en `DATABASE_URL` y configura `CRM_BACKEND=notion`.

## Endpoints

- `POST /webhooks/whatsapp/kapso`
- `POST /internal/replay`
- `POST /chat/local`
- `GET /playground`
- `GET /healthz`

## Probar desde chat local

Con la app levantada, abre:

```text
http://127.0.0.1:8000/playground
```

Ese playground usa el endpoint `POST /chat/local` para hablar con el agente sin pasar por Kapso. Sirve para probar Notion, OpenAI y memoria conversacional local antes de conectar WhatsApp real.

## Regla central del CRM

Cada conversación queda anclada a un solo lead:

1. entra un mensaje con `phone_number`
2. el orquestador busca ese número en el CRM
3. si no existe, crea el lead
4. desde ahí el workflow solo opera sobre ese `current_lead`

El agente no tiene queries abiertas sobre el CRM. Las actions de negocio se ejecutan únicamente a través de tools acotadas al lead resuelto por teléfono.

## Flujo

1. Normaliza el payload de WhatsApp.
2. Aplica idempotencia por `message_id`.
3. Resuelve el `current_lead` por teléfono o lo crea si no existe.
4. Carga historial y memorias de esa conversación.
5. Clasifica intención y planea acciones sobre ese lead únicamente.
6. Ejecuta tools limitadas al `current_lead`.
7. Persiste run y mensajes.
8. Envía respuesta por el canal configurado.
