from __future__ import annotations

import json
import os
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx


AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPES = ["https://www.googleapis.com/auth/calendar"]
REDIRECT_HOST = "127.0.0.1"


def pick_redirect_port() -> int:
    preferred = os.getenv("GOOGLE_OAUTH_REDIRECT_PORT")
    if preferred:
        return int(preferred)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((REDIRECT_HOST, 0))
        return int(sock.getsockname()[1])


def load_env_file() -> dict[str, str]:
    env_path = Path(".env")
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class CallbackState:
    def __init__(self) -> None:
        self.code: str | None = None
        self.error: str | None = None
        self.event = threading.Event()


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    state: CallbackState

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        self.state.code = query.get("code", [None])[0]
        self.state.error = query.get("error", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        if self.state.code:
            self.wfile.write(b"<h1>Autorizacion completada</h1><p>Vuelve a la terminal.</p>")
        else:
            self.wfile.write(b"<h1>Error</h1><p>Revisa la terminal.</p>")
        self.state.event.set()

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    env_values = load_env_file()
    client_id = os.getenv("GOOGLE_CLIENT_ID") or env_values.get("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET") or env_values.get("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit("Faltan GOOGLE_CLIENT_ID y/o GOOGLE_CLIENT_SECRET en .env o variables de entorno.")

    redirect_port = pick_redirect_port()
    redirect_uri = f"http://{REDIRECT_HOST}:{redirect_port}/callback"

    state = CallbackState()
    OAuthCallbackHandler.state = state
    server = HTTPServer((REDIRECT_HOST, redirect_port), OAuthCallbackHandler)
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"{AUTH_URL}?{urlencode(params)}"
    print("Abre esta URL si tu navegador no se abre solo:\n")
    print(auth_url)
    print("\nEsperando autorizacion en el navegador...")
    webbrowser.open(auth_url)

    state.event.wait()
    server.server_close()

    if state.error:
        raise SystemExit(f"Google devolvio un error: {state.error}")
    if not state.code:
        raise SystemExit("No se recibio authorization code.")

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": state.code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
        )
        response.raise_for_status()
        token_payload = response.json()

    refresh_token = token_payload.get("refresh_token")
    if not refresh_token:
        print(json.dumps(token_payload, indent=2, ensure_ascii=True))
        raise SystemExit(
            "Google no devolvio refresh_token. Revoca el acceso de la app en tu cuenta Google y vuelve a correr el script."
        )

    print("\nRefresh token generado. Agrega esto a tu .env del VPS:\n")
    print(f"GOOGLE_REFRESH_TOKEN={refresh_token}")


if __name__ == "__main__":
    main()
