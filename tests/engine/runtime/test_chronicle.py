import asyncio
from contextlib import contextmanager
from typing import Any, Iterator

from pydantic_ai.exceptions import UsageLimitExceeded

from src.engine.runtime.chronicle import (
    HISTORY_COMPACT_THRESHOLD,
    HISTORY_KEEP_RECENT,
    compact_history,
    digest,
)
from src.engine.runtime.replay import ReplayTape
from src.engine.state.models import HistoryEvent, WorldState
from src.interface.session_log import Logger


class _Logger(Logger):
    """No-I/O Logger test double; nominal subtype preserves runtime signatures."""

    def __init__(self) -> None:
        pass

    @contextmanager
    def run(self, label: str) -> Iterator[None]:
        yield

    def log_event(self, event: str, **kwargs: Any) -> None:
        pass

    def log_messages(self, label: str, messages: list[Any]) -> None:
        pass


def _events(n: int) -> list[HistoryEvent]:
    return [HistoryEvent(text=f"event {i}", location="tavern") for i in range(n)]


# ============ digest() fallback ============

def test_digest_joins_event_texts():
    events = [HistoryEvent(text="hero enters", location="tavern"), HistoryEvent(text="hero waits", location="tavern")]
    assert digest(events) == "hero enters hero waits"


def test_digest_truncates_long_output():
    events = [HistoryEvent(text="x" * 1000, location="tavern")]
    result = digest(events)
    assert len(result) == 600
    assert result.endswith("...")


# ============ compact_history() trigger/selection ============

def test_below_threshold_does_not_compact():
    state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD))
    asyncio.run(compact_history(state, _Logger()))
    assert len(state.history) == HISTORY_COMPACT_THRESHOLD
    assert state.chronicle == []


def test_above_threshold_archives_oldest_and_keeps_recent_tail(monkeypatch):
    state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD + 1))

    async def fake_run(_prompt, deps, usage_limits):
        class _Result:
            output = f"summary of {len(deps.events)} events"
            def all_messages(self):
                return []
        return _Result()

    monkeypatch.setattr("src.engine.runtime.chronicle.chronicler_agent.run", fake_run)
    asyncio.run(compact_history(state, _Logger()))

    assert len(state.history) == HISTORY_KEEP_RECENT
    assert [e.text for e in state.history] == [f"event {i}" for i in range(HISTORY_COMPACT_THRESHOLD + 1 - HISTORY_KEEP_RECENT, HISTORY_COMPACT_THRESHOLD + 1)]
    archived_count = HISTORY_COMPACT_THRESHOLD + 1 - HISTORY_KEEP_RECENT
    assert state.chronicle == [f"summary of {archived_count} events"]


def test_usage_limit_exceeded_falls_back_to_digest(monkeypatch):
    state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD + 1))

    async def fake_run(*_args, **_kwargs):
        raise UsageLimitExceeded("too many requests")

    monkeypatch.setattr("src.engine.runtime.chronicle.chronicler_agent.run", fake_run)
    asyncio.run(compact_history(state, _Logger()))

    archived = _events(HISTORY_COMPACT_THRESHOLD + 1 - HISTORY_KEEP_RECENT)
    assert state.chronicle == [digest(archived)]
    assert len(state.history) == HISTORY_KEEP_RECENT


# ============ replay integration ============

def test_compact_history_playback_never_calls_chronicler_agent(tmp_path, monkeypatch):
    path = tmp_path / "run.jsonl"
    ReplayTape.recording(path).chronicle("The hero spent months rebuilding the docks.")

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("the chronicler agent was called during replay")

    monkeypatch.setattr("src.engine.runtime.chronicle.chronicler_agent.run", unexpected)
    state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD + 1))
    playback = ReplayTape.playback(path)
    asyncio.run(compact_history(state, _Logger(), playback))

    assert state.chronicle == ["The hero spent months rebuilding the docks."]
    assert len(state.history) == HISTORY_KEEP_RECENT
    playback.assert_consumed()


def test_compact_history_records_chronicle_entry_for_replay(tmp_path, monkeypatch):
    path = tmp_path / "run.jsonl"
    recording = ReplayTape.recording(path)

    async def fake_run(_prompt, deps, usage_limits):
        class _Result:
            output = "a recorded summary"
            def all_messages(self):
                return []
        return _Result()

    monkeypatch.setattr("src.engine.runtime.chronicle.chronicler_agent.run", fake_run)
    state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD + 1))
    asyncio.run(compact_history(state, _Logger(), recording))

    playback = ReplayTape.playback(path)
    assert playback.chronicle() == "a recorded summary"
