from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sales_agent.adapters.whatsapp import normalize_kapso_payload
from sales_agent.domain.models import InboundMessage


router = APIRouter()


class ReplayPayload(BaseModel):
    event: InboundMessage | None = None
    webhook_payload: dict | None = None


class ChatPayload(BaseModel):
    text: str
    phone_number: str = "3156832405"
    conversation_id: str = "playground-conv"
    contact_name: str | None = "Playground User"


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/playground", response_class=HTMLResponse)
async def playground() -> str:
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
      label {
        display: block;
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 6px;
      }
      input, textarea, button {
        font: inherit;
      }
      input, textarea {
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
      const chat = document.getElementById("chat");
      const text = document.getElementById("text");
      const phone = document.getElementById("phone");
      const conversation = document.getElementById("conversation");
      const status = document.getElementById("status");
      const storeKey = "sales-agent-playground";

      function loadState() {
        const raw = localStorage.getItem(storeKey);
        if (!raw) return [];
        try { return JSON.parse(raw); } catch { return []; }
      }

      function saveState(messages) {
        localStorage.setItem(storeKey, JSON.stringify(messages));
      }

      function render(messages) {
        chat.innerHTML = "";
        for (const item of messages) {
          const el = document.createElement("div");
          el.className = `msg ${item.role}`;
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

      async function sendMessage() {
        const value = text.value.trim();
        if (!value) return;
        const messages = loadState();
        messages.push({ role: "user", text: value });
        saveState(messages);
        render(messages);
        text.value = "";
        status.textContent = "Consultando al agente...";

        const response = await fetch("/chat/local", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
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
        messages.push({ role: "agent", text: data.response_text, meta });
        saveState(messages);
        render(messages);
        status.textContent = response.ok ? "OK" : "Error";
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
      render(loadState());
    </script>
  </body>
</html>
"""


@router.post("/webhooks/whatsapp/kapso")
async def kapso_webhook(request: Request) -> dict:
    payload = await request.json()
    event = normalize_kapso_payload(payload)
    result = await request.app.state.sales_agent.workflow.run(event)
    return result.model_dump(mode="json")


@router.post("/internal/replay")
async def replay(payload: ReplayPayload, request: Request) -> dict:
    event = payload.event
    if event is None:
        event = normalize_kapso_payload(payload.webhook_payload or {})
    result = await request.app.state.sales_agent.workflow.run(event)
    return result.model_dump(mode="json")


@router.post("/chat/local")
async def local_chat(payload: ChatPayload, request: Request) -> dict:
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
    result = await request.app.state.sales_agent.workflow.run(event)
    return result.model_dump(mode="json")
