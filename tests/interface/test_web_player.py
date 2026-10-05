import asyncio
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from src.agents.character.tools import Action, Travel
from src.engine.state.models import Character, Location, WorldState
from src.interface.app import _build_handler
from src.interface.web_player import PlayBroker, make_web_pc_controller


def _state() -> WorldState:
    return WorldState(
        locations={
            "inn": Location(id="inn", connections=["road"]),
            "road": Location(id="road", connections=["inn"]),
        },
        characters={"hero": Character(id="hero", location="inn")},
    )


def _post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def test_broker_exposes_situation_and_parses_commands():
    broker = PlayBroker()
    request_id = broker.begin("hero", _state())

    waiting = broker.status()
    assert waiting["status"] == "waiting"
    assert waiting["request_id"] == request_id
    assert "Your turn: hero" in waiting["situation"]
    assert "Connections: road" in waiting["situation"]
    assert "Type a plain sentence" not in waiting["situation"]

    submitted = broker.submit(request_id, "/travel road")
    assert submitted["action"] == Travel(actor="hero", destination="road").model_dump()
    assert broker.status(request_id)["status"] == "submitted"


def test_broker_rejects_invalid_action_without_consuming_turn():
    broker = PlayBroker()
    request_id = broker.begin("hero", _state())

    with pytest.raises(ValueError, match="Unknown destination"):
        broker.submit(request_id, "/travel moon")

    assert broker.status()["status"] == "waiting"


def test_remote_controller_round_trip_through_dashboard(tmp_path):
    broker = PlayBroker()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler(tmp_path, tmp_path, broker))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    def submit_when_ready() -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with urllib.request.urlopen(f"{base_url}/api/play/status") as response:
                status = json.load(response)
            if status["status"] == "waiting":
                _post(
                    f"{base_url}/api/play/action",
                    {"request_id": status["request_id"], "line": "inspect the hearth"},
                )
                return
            time.sleep(0.02)
        raise AssertionError("controller did not request a turn")

    submitter = threading.Thread(target=submit_when_ready)
    submitter.start()
    try:
        action = asyncio.run(make_web_pc_controller(base_url)("hero", _state()))
    finally:
        submitter.join()
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert action == Action(actor="hero", description="inspect the hearth")
    assert broker.status()["status"] == "idle"
