from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sales_agent.adapters.whatsapp import KapsoPayloadError, normalize_kapso_payload
from sales_agent.domain.models import InboundMessage


router = APIRouter()


class ReplayPayload(BaseModel):
    event: InboundMessage | None = None
    webhook_payload: dict | None = None


class ChatPayload(BaseModel):
    text: str
    phone_number: str = "3156832405"
    conversation_id: str = "playground-conv"
    contact_name: str | None = None


def _require_playground_access(request: Request, provided_token: str | None = None) -> None:
    settings = request.app.state.sales_agent.settings
    if not settings.is_playground_enabled():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Playground disabled.",
        )
    expected_token = settings.playground_token
    if expected_token and provided_token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid playground token.",
        )


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/playground", response_class=HTMLResponse)
async def playground(request: Request, token: str | None = None) -> str:
    _require_playground_access(request, token)
    return """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Sales Agent Playground</title>
    <style>
      :root {
        --bg: #f4efe6;
        --panel: #fffaf2;
        --ink: #1c1c1a;
        --muted: #6e675e;
        --accent: #0f6c5c;
        --accent-2: #d96f32;
        --border: #d8cfc1;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        background: radial-gradient(circle at top left, #fff7eb, var(--bg) 55%);
        color: var(--ink);
      }
      .wrap {
        max-width: 960px;
        margin: 32px auto;
        padding: 0 20px;
      }
      .panel {
        background: color-mix(in srgb, var(--panel) 92%, white);
        border: 1px solid var(--border);
        border-radius: 20px;
        box-shadow: 0 16px 50px rgba(44, 35, 17, 0.08);
        overflow: hidden;
      }
      .head {
        padding: 20px 24px;
        border-bottom: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
      }
      .title {
        margin: 0;
        font-size: 24px;
        line-height: 1.1;
      }
      .sub {
        margin: 6px 0 0;
        color: var(--muted);
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        align-items: end;
      }
      .prompt-editor {
        padding: 20px 24px;
        border-bottom: 1px solid var(--border);
        display: grid;
        gap: 16px;
        background: linear-gradient(180deg, rgba(255, 250, 242, 0.95), rgba(255, 250, 242, 0.82));
      }
      .stack {
        display: grid;
        gap: 10px;
      }
      label {
        display: block;
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 6px;
      }
      input, textarea, button, select {
        font: inherit;
      }
      input, textarea, select {
        width: 100%;
        padding: 12px 14px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: white;
      }
      textarea {
        min-height: 110px;
        resize: vertical;
      }
      .chat {
        padding: 24px;
        min-height: 420px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        background:
          linear-gradient(180deg, rgba(255,255,255,0.3), rgba(255,255,255,0)),
          repeating-linear-gradient(180deg, transparent 0, transparent 27px, rgba(216, 207, 193, 0.15) 28px);
      }
      .msg {
        max-width: 78%;
        padding: 14px 16px;
        border-radius: 18px;
        line-height: 1.45;
        white-space: pre-wrap;
      }
      .msg.user {
        align-self: flex-end;
        background: var(--accent);
        color: white;
        border-bottom-right-radius: 6px;
      }
      .msg.agent {
        align-self: flex-start;
        background: white;
        border: 1px solid var(--border);
        border-bottom-left-radius: 6px;
      }
      .msg.pending {
        opacity: 0.72;
      }
      .meta {
        display: block;
        margin-top: 8px;
        font-size: 12px;
        color: var(--muted);
      }
      .composer {
        padding: 20px 24px 24px;
        border-top: 1px solid var(--border);
        display: grid;
        gap: 12px;
      }
      .actions {
        display: flex;
        gap: 12px;
        justify-content: space-between;
        flex-wrap: wrap;
      }
      .toolbar {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
      }
      button {
        border: 0;
        border-radius: 999px;
        padding: 12px 18px;
        cursor: pointer;
      }
      .primary {
        background: var(--accent-2);
        color: white;
      }
      .secondary {
        background: #efe5d6;
        color: var(--ink);
      }
      .status {
        color: var(--muted);
        font-size: 13px;
        min-height: 18px;
      }
      .hint {
        color: var(--muted);
        font-size: 13px;
        margin: 0;
      }
      details {
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.72);
        padding: 12px 14px;
      }
      summary {
        cursor: pointer;
        font-weight: 600;
      }
      pre {
        margin: 10px 0 0;
        white-space: pre-wrap;
        word-break: break-word;
        font: 12px/1.45 "IBM Plex Mono", "SFMono-Regular", monospace;
        color: var(--ink);
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="panel">
        <div class="head">
          <div>
            <h1 class="title">Sales Agent Playground</h1>
            <p class="sub">Prueba el agente localmente con Notion y OpenAI antes de conectar WhatsApp.</p>
          </div>
          <div class="grid">
            <div>
              <label for="phone">Teléfono</label>
              <input id="phone" value="3156832405" />
            </div>
            <div>
              <label for="conversation">Conversation ID</label>
              <input id="conversation" value="playground-conv" />
            </div>
          </div>
        </div>
        <div class="prompt-editor">
          <p class="hint">Los prompts y el knowledge del agente viven en archivos `.md` versionados en el repo. El playground solo los muestra en lectura para inspección.</p>
          <div class="status" id="promptStatus"></div>
          <details>
            <summary>Planner scaffold</summary>
            <pre id="corePrompt"></pre>
          </details>
          <details>
            <summary>Knowledge sections</summary>
            <pre id="knowledgeSections"></pre>
          </details>
        </div>
        <div id="chat" class="chat"></div>
        <div class="composer">
          <textarea id="text" placeholder="Escribe un mensaje para el agente..."></textarea>
          <div class="actions">
            <div class="status" id="status"></div>
            <div>
              <button class="secondary" id="reset" type="button">Reset chat</button>
              <button class="primary" id="send" type="button">Enviar</button>
            </div>
          </div>
        </div>
      </div>
    </div>
    <script>
      const params = new URLSearchParams(window.location.search);
      const playgroundToken = params.get("token") || "";
      const chat = document.getElementById("chat");
      const text = document.getElementById("text");
      const phone = document.getElementById("phone");
      const conversation = document.getElementById("conversation");
      const status = document.getElementById("status");
      const promptStatus = document.getElementById("promptStatus");
      const corePrompt = document.getElementById("corePrompt");
      const knowledgeSections = document.getElementById("knowledgeSections");
      function currentChatKey() {
        const normalizedPhone = (phone.value || "3156832405").trim() || "3156832405";
        const normalizedConversation = (conversation.value || "playground-conv").trim() || "playground-conv";
        return `sales-agent-playground:${normalizedPhone}:${normalizedConversation}`;
      }

      function loadState() {
        const raw = localStorage.getItem(currentChatKey());
        if (!raw) return [];
        try { return JSON.parse(raw); } catch { return []; }
      }

      function saveState(messages) {
        localStorage.setItem(currentChatKey(), JSON.stringify(messages));
      }

      function nextId(prefix) {
        if (window.crypto && typeof window.crypto.randomUUID === "function") {
          return `${prefix}-${window.crypto.randomUUID()}`;
        }
        return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      }

      function appendMessage(item) {
        const messages = loadState();
        messages.push(item);
        saveState(messages);
        render(messages);
      }

      function updateMessage(messageId, updater) {
        const messages = loadState();
        const index = messages.findIndex((item) => item.id === messageId);
        if (index === -1) return;
        messages[index] = updater(messages[index]);
        saveState(messages);
        render(messages);
      }

      function removeMessage(messageId) {
        const messages = loadState().filter((item) => item.id !== messageId);
        saveState(messages);
        render(messages);
      }

      function render(messages) {
        chat.innerHTML = "";
        for (const item of messages) {
          const el = document.createElement("div");
          el.className = `msg ${item.role}`;
          if (item.pending) {
            el.classList.add("pending");
          }
          el.textContent = item.text;
          if (item.meta) {
            const meta = document.createElement("span");
            meta.className = "meta";
            meta.textContent = item.meta;
            el.appendChild(meta);
          }
          chat.appendChild(el);
        }
        chat.scrollTop = chat.scrollHeight;
      }

      async function fetchAgentContext() {
        const response = await fetch(`/playground/agent-context`, {
          headers: {
            ...(playgroundToken ? { "X-Playground-Token": playgroundToken } : {})
          }
        });
        if (!response.ok) {
          throw new Error("No se pudo cargar el contexto del agente.");
        }
        return await response.json();
      }

      async function loadAgentContext() {
        promptStatus.textContent = "Cargando contexto del agente...";
        try {
          const data = await fetchAgentContext();
          corePrompt.textContent = data.planner_scaffold || "";
          const renderedKnowledge = (data.knowledge_sections || []).map((item) => {
            return `# ${item.title} (${item.name})\n\n${item.content || ""}`;
          }).join("\\n\\n---\\n\\n");
          knowledgeSections.textContent = renderedKnowledge;
          promptStatus.textContent = "Contexto cargado desde archivos .md del repo.";
        } catch (error) {
          promptStatus.textContent = error instanceof Error ? error.message : "Error al cargar contexto.";
        }
      }

      async function sendMessage() {
        const value = text.value.trim();
        if (!value) return;
        const requestId = nextId("playground");
        const replyId = `reply-${requestId}`;
        appendMessage({ id: `user-${requestId}`, role: "user", text: value });
        appendMessage({
          id: replyId,
          role: "agent",
          text: "Pensando...",
          meta: "Procesando mensaje",
          pending: true,
        });
        text.value = "";
        status.textContent = "Consultando al agente...";

        try {
          const response = await fetch("/chat/local", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...(playgroundToken ? { "X-Playground-Token": playgroundToken } : {})
            },
            body: JSON.stringify({
              text: value,
              phone_number: phone.value.trim() || "3156832405",
              conversation_id: conversation.value.trim() || "playground-conv"
            })
          });
          const data = await response.json();
          const lead = data.contact?.full_name || data.contact?.phone_number || "sin lead";
          const stage = data.contact?.stage || "sin etapa";
          const tools = (data.tool_results || []).map(item => item.action?.type).filter(Boolean).join(", ") || "sin tools";
          const meta = `intent=${data.intent} | lead=${lead} | etapa=${stage} | tools=${tools}`;
          if (!data.render_reply) {
            removeMessage(replyId);
            status.textContent = response.ok ? "Agrupado con mensajes posteriores" : "Error";
            return;
          }
          updateMessage(replyId, (current) => ({
            ...current,
            text: data.response_text || "Sin respuesta",
            meta,
            pending: false,
          }));
          status.textContent = response.ok ? "OK" : "Error";
        } catch (error) {
          updateMessage(replyId, (current) => ({
            ...current,
            text: "No se pudo obtener respuesta del agente.",
            meta: error instanceof Error ? error.message : "Error inesperado",
            pending: false,
          }));
          status.textContent = "Error";
        }
      }

      document.getElementById("send").addEventListener("click", sendMessage);
      text.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          sendMessage();
        }
      });
      document.getElementById("reset").addEventListener("click", () => {
        saveState([]);
        render([]);
        status.textContent = "Conversación reiniciada.";
      });
      for (const field of [phone, conversation]) {
        field.addEventListener("change", () => {
          render(loadState());
          status.textContent = "Contexto del playground actualizado.";
        });
      }
      render(loadState());
      loadAgentContext();
    </script>
  </body>
</html>
"""


@router.get("/playground/agent-context")
async def get_agent_context(
    request: Request,
    x_playground_token: str | None = Header(default=None, alias="X-Playground-Token"),
) -> dict:
    _require_playground_access(request, x_playground_token)
    return await request.app.state.sales_agent.get_playground_context()


@router.post("/webhooks/whatsapp/kapso")
async def kapso_webhook(request: Request) -> dict:
    payload = await request.json()
    try:
        event = normalize_kapso_payload(payload)
    except KapsoPayloadError as exc:
        return {
            "accepted": False,
            "queued": False,
            "duplicate": False,
            "render_reply": False,
            "aggregated_messages": 0,
            "response_text": "",
            "ignored_reason": str(exc),
        }
    result = await request.app.state.sales_agent.handle_event(event, wait_for_response=False)
    return result.model_dump(mode="json")


@router.post("/internal/replay")
async def replay(payload: ReplayPayload, request: Request) -> dict:
    event = payload.event
    if event is None:
        event = normalize_kapso_payload(payload.webhook_payload or {})
    result = await request.app.state.sales_agent.handle_event(event, wait_for_response=True)
    return result.model_dump(mode="json")


@router.post("/chat/local")
async def local_chat(
    payload: ChatPayload,
    request: Request,
    x_playground_token: str | None = Header(default=None, alias="X-Playground-Token"),
) -> dict:
    _require_playground_access(request, x_playground_token)
    event = InboundMessage(
        message_id=f"local-{uuid4().hex}",
        conversation_id=payload.conversation_id,
        phone_number=payload.phone_number,
        text=payload.text,
        timestamp=datetime.now(timezone.utc),
        raw_payload={"source": "playground"},
        provider="local-playground",
        contact_name=payload.contact_name,
    )
    result = await request.app.state.sales_agent.handle_event(event, wait_for_response=True)
    return result.model_dump(mode="json")
