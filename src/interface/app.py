"""Unified infinite-dnd browser app: World and Logs as tabs under one shell."""

from __future__ import annotations

import argparse
import json
import socket
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from src.interface.assets import read_asset, static_asset
from src.engine.state import WorldState
from src.interface.session_log import DEFAULT_LOG_DIR, list_logs, load_session
from src.interface.web_player import PlayBroker
from src.interface.world_state import DEFAULT_STATE_DIR, list_state_runs, load_series, load_view

WORLD_HTML = read_asset("templates/world.html")
LOGS_HTML = read_asset("templates/logs.html")

def _with_nav(html: str, active: str) -> str:
    world_current = ' aria-current="page"' if active == "world" else ""
    logs_current = ' aria-current="page"' if active == "logs" else ""
    tabs = (
        '<nav class="tabs" aria-label="Primary">'
        f'<a href="/" class="{"active" if active == "world" else ""}"'
        f'{world_current}>World</a>'
        f'<a href="/logs" class="{"active" if active == "logs" else ""}"'
        f'{logs_current}>Logs</a>'
        "</nav>"
    )
    html = html.replace('<div class="topbar-main">', f'<div class="topbar-main">{tabs}', 1)
    html = html.replace("</head>", '  <link rel="stylesheet" href="/static/studio.css">\n</head>', 1)
    html = html.replace("<body>", f'<body class="view-{active}">', 1)
    html = html.replace(
        '<div class="brand"><span class="brand-mark"></span>infinite-dnd</div>',
        '<div class="brand"><span class="brand-mark"></span>Infinite D&amp;D <span class="brand-sub">Observer</span></div>',
        1,
    )
    return html


def _build_handler(
    state_dir: Path = DEFAULT_STATE_DIR,
    log_dir: Path = DEFAULT_LOG_DIR,
    play_broker: PlayBroker | None = None,
) -> type[BaseHTTPRequestHandler]:
    broker = play_broker or PlayBroker()

    class StudioHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_with_nav(WORLD_HTML, "world"))
            elif parsed.path == "/logs":
                self._send_html(_with_nav(LOGS_HTML, "logs"))
            elif parsed.path.startswith("/static/"):
                self._send_static(parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/runs":
                self._send_json(list_state_runs(state_dir))
            elif parsed.path == "/api/files":
                self._send_json(list_logs(log_dir))
            elif parsed.path == "/api/play/status":
                self._send_json(broker.status())
            elif parsed.path.startswith("/api/play/request/"):
                request_id = unquote(parsed.path.removeprefix("/api/play/request/"))
                self._send_json(broker.status(request_id))
            elif parsed.path.startswith("/api/state/"):
                parts = self._parse_run_path(parsed.path, "/api/state/")
                if parts is None:
                    self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                    return
                scenario, run_id = parts
                tick_values = parse_qs(parsed.query).get("tick")
                try:
                    tick = int(tick_values[0]) if tick_values else None
                    self._require_run(scenario, run_id)
                    self._send_json(load_view(state_dir, scenario, run_id, tick))
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except FileNotFoundError as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            elif parsed.path.startswith("/api/series/"):
                parts = self._parse_run_path(parsed.path, "/api/series/")
                if parts is None:
                    self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                    return
                scenario, run_id = parts
                try:
                    self._require_run(scenario, run_id)
                    self._send_json(load_series(state_dir, scenario, run_id))
                except FileNotFoundError as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            elif parsed.path.startswith("/api/logs/"):
                name = unquote(parsed.path.removeprefix("/api/logs/"))
                path = next((Path(item["path"]) for item in list_logs(log_dir) if item["name"] == name), None)
                if path:
                    self._send_json(load_session(path))
                else:
                    self._send_json({"error": f"Log not found: {name}"}, HTTPStatus.NOT_FOUND)
            else:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/api/play/request":
                    actor_id = payload.get("actor_id")
                    if not isinstance(actor_id, str) or not actor_id:
                        raise ValueError("actor_id is required.")
                    request_id = broker.begin(actor_id, WorldState.model_validate(payload.get("state")))
                    self._send_json({"status": "waiting", "request_id": request_id}, HTTPStatus.CREATED)
                elif parsed.path == "/api/play/action":
                    request_id, line = payload.get("request_id"), payload.get("line")
                    if not isinstance(request_id, str) or not isinstance(line, str):
                        raise ValueError("request_id and line are required.")
                    self._send_json(broker.submit(request_id, line))
                elif parsed.path == "/api/play/finish":
                    request_id = payload.get("request_id")
                    if not isinstance(request_id, str):
                        raise ValueError("request_id is required.")
                    broker.finish(request_id)
                    self._send_json({"status": "idle"})
                else:
                    self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except LookupError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)

        def _read_json(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Invalid Content-Length.") from exc
            if length <= 0 or length > 5_000_000:
                raise ValueError("A JSON body is required (maximum 5 MB).")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
            return payload

        def _parse_run_path(self, path: str, prefix: str) -> tuple[str, str] | None:
            remainder = unquote(path.removeprefix(prefix))
            parts = remainder.split("/")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                return None
            return parts[0], parts[1]

        def _require_run(self, scenario: object, run_id: object) -> None:
            if (
                not isinstance(scenario, str)
                or not isinstance(run_id, str)
                or not any(
                    meta["scenario"] == scenario and meta["run_id"] == run_id
                    for meta in list_state_runs(state_dir)
                )
            ):
                raise FileNotFoundError(f"Run not found: {scenario}/{run_id}")

        def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            self._send(html.encode("utf-8"), "text/html; charset=utf-8")

        def _send_static(self, name: str) -> None:
            asset = static_asset(name)
            if asset is None:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send(*asset)

        def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return StudioHandler


def _pick_port(host: str, preferred: int) -> int:
    for candidate in [preferred, *range(preferred + 1, preferred + 10), 0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
            return sock.getsockname()[1]
    raise RuntimeError("Could not find an available port.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the infinite-dnd app (World + Logs tabs).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(
        (args.host, _pick_port(args.host, args.port)),
        _build_handler(args.state_dir, args.log_dir),
    )
    url = f"http://{args.host}:{server.server_port}"
    print(f"Serving infinite-dnd at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping infinite-dnd.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
