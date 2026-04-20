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

Variables útiles para producción:

- `DATABASE_URL=postgresql+asyncpg://usuario:password@host:5432/sales_agent`
- `APP_ENV=production`
- `MESSAGE_BATCH_WINDOW_SECONDS=3` para agrupar ráfagas cortas del usuario antes de responder
- `PLAYGROUND_ENABLED=false` para no exponer pruebas manuales
- `PLAYGROUND_ENABLED=true` junto con `PLAYGROUND_TOKEN=un-secreto-largo` si quieres habilitarlo temporalmente

## Endpoints

- `POST /webhooks/whatsapp/kapso`
- `POST /internal/replay`
- `POST /chat/local`
- `GET /playground`
- `GET /playground/agent-context`
- `GET /healthz`

## Probar desde chat local

Con la app levantada, abre:

```text
http://127.0.0.1:8000/playground
```

Ese playground usa el endpoint `POST /chat/local` para hablar con el agente sin pasar por Kapso. Sirve para probar Notion, OpenAI y memoria conversacional local antes de conectar WhatsApp real.

Los prompts y el knowledge del agente viven en archivos `.md` dentro del repo y se cargan desde código:

- `core prompt`: instrucciones base y reglas operativas del planner.
- `business rules`: brief comercial, segmentación, calificación y CTA.
- `knowledge`: secciones consultables por el agente bajo demanda, por ejemplo compañía, pricing, FAQ o integraciones.

El playground ya no permite editar prompts. Solo muestra en lectura el scaffold del planner y las secciones de knowledge cargadas desde archivos versionados.

En `production` el playground queda deshabilitado por defecto. Si lo habilitas con `PLAYGROUND_ENABLED=true` y defines `PLAYGROUND_TOKEN`, accede con:

```text
https://tu-dominio/playground?token=TU_TOKEN
```

La página reutiliza ese token para llamar a `POST /chat/local`.

## Despliegue en EasyPanel

El repo ya incluye `Dockerfile` para desplegar como servicio web.

Pasos recomendados:

1. Crea un servicio `App` desde Git y apunta al repo.
2. Usa el `Dockerfile` del proyecto.
3. Expón el puerto `8000` o mapea el puerto público al `8000` interno.
4. Define variables de entorno en EasyPanel:
   - `APP_ENV=production`
   - `DATABASE_URL=postgresql+asyncpg://...`
   - `CRM_BACKEND=notion`
   - `OPENAI_API_KEY=...`
   - `NOTION_API_KEY=...`
   - `NOTION_DATA_SOURCE_ID=...`
   - `KAPSO_PHONE_NUMBER_ID=...`
   - `KAPSO_API_TOKEN=...`
   - `WHATSAPP_SEND_ENABLED=false` mientras pruebas
5. Configura healthcheck contra `GET /healthz`.
6. Si vas a usar SQLite por alguna razón, monta un volumen persistente; para producción real conviene Postgres.

Notas prácticas:

- No subas `.env` al repo; EasyPanel debe inyectar secretos desde su panel.
- Si el repo es privado, EasyPanel puede desplegarlo igual usando integración con GitHub/GitLab o un token de acceso.
- Si publicas el repo, el código del playground será público, pero no los secretos si los mantienes fuera del repo. Lo que sí debes evitar es dejar `/playground` abierto en producción.

## Regla central del CRM

Cada conversación queda anclada a un solo lead:

1. entra un mensaje con `phone_number`
2. el orquestador busca ese número en el CRM
3. si no existe, crea el lead
4. desde ahí el workflow solo opera sobre ese `current_lead`

El agente no tiene queries abiertas sobre el CRM. Las actions de negocio se ejecutan únicamente a través de tools acotadas al lead resuelto por teléfono.

## Flujo

1. Normaliza el payload de WhatsApp.
2. Aplica idempotencia por `message_id` y persiste el inbound.
3. Agrupa mensajes consecutivos de la misma conversación durante una ventana corta (`MESSAGE_BATCH_WINDOW_SECONDS`).
4. Cuando se cierra la ventana, revisa si entró algo nuevo antes del commit; si entró, recompone el lote y reprocesa.
5. Resuelve el `current_lead` por teléfono o lo crea si no existe.
6. Carga historial y memorias de esa conversación.
7. Clasifica intención y planea acciones sobre ese lead únicamente.
8. Ejecuta tools limitadas al `current_lead`.
9. Persiste el run y envía una sola respuesta para el lote.
