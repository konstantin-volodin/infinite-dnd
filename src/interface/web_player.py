"""Small HTTP bridge between the dashboard and a human-controlled game process."""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from src.agents.character.tools import CharacterTool
from src.engine.state import WorldState
from src.interface.player import _InputError, _describe_situation, _parse_intent


_TOOL_ADAPTER = TypeAdapter(CharacterTool)


@dataclass
class _Turn:
    id: str
    actor_id: str
    state: WorldState
    action: CharacterTool | None = None


class PlayBroker:
    """Thread-safe single-game turn inbox owned by the dashboard server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turn: _Turn | None = None

    def begin(self, actor_id: str, state: WorldState) -> str:
        if actor_id not in state.characters:
            raise ValueError(f"Unknown acting character '{actor_id}'.")
        with self._lock:
            if self._turn is not None and self._turn.action is None:
                raise RuntimeError("Another player turn is already waiting for input.")
            request_id = uuid4().hex
            self._turn = _Turn(request_id, actor_id, state)
            return request_id

    def status(self, request_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            turn = self._turn
            if turn is None or (request_id is not None and turn.id != request_id):
                return {"status": "idle"}
            if turn.action is not None:
                return {"status": "submitted", "request_id": turn.id, "action": turn.action.model_dump()}
            return {
                "status": "waiting",
                "request_id": turn.id,
                "actor_id": turn.actor_id,
                "actor_name": turn.actor_id,
                "situation": _describe_situation(turn.actor_id, turn.state, include_help=False).strip(),
            }

    def submit(self, request_id: str, line: str) -> dict[str, Any]:
        with self._lock:
            turn = self._turn
            if turn is None or turn.id != request_id:
                raise LookupError("That turn is no longer waiting for input.")
            if turn.action is not None:
                raise RuntimeError("An action was already submitted for this turn.")
            try:
                turn.action = _parse_intent(turn.actor_id, turn.state, line)
            except _InputError as exc:
                raise ValueError(str(exc)) from exc
            return {"status": "submitted", "action": turn.action.model_dump()}

    def finish(self, request_id: str) -> None:
        with self._lock:
            if self._turn is not None and self._turn.id == request_id:
                self._turn = None


def _json_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Cannot reach the web dashboard at {url}: {exc.reason}") from exc


def make_web_pc_controller(base_url: str = "http://127.0.0.1:8765"):
    """Create a runtime-compatible PC controller backed by a dashboard server."""
    base_url = base_url.rstrip("/")

    async def controller(actor_id: str, state: WorldState) -> CharacterTool:
        started = await asyncio.to_thread(
            _json_request,
            f"{base_url}/api/play/request",
            {"actor_id": actor_id, "state": state.model_dump(mode="json")},
        )
        request_id = started["request_id"]
        try:
            while True:
                result = await asyncio.to_thread(_json_request, f"{base_url}/api/play/request/{request_id}")
                if result.get("status") == "submitted":
                    return _TOOL_ADAPTER.validate_python(result["action"])
                if result.get("status") != "waiting":
                    raise RuntimeError("The dashboard discarded the pending player turn.")
                await asyncio.sleep(0.35)
        finally:
            try:
                await asyncio.to_thread(_json_request, f"{base_url}/api/play/finish", {"request_id": request_id})
            except ConnectionError:
                pass

    return controller
