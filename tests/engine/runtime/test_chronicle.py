import asyncio
import json
from contextlib import contextmanager
from typing import Any, Iterator

from pydantic_ai.exceptions import ModelAPIError, UsageLimitExceeded

from src.agents.dm.chronicler import SAGA_INSTRUCTIONS
from src.engine.runtime import chronicle
from src.engine.runtime.chronicle import (
    CHRONICLE_RECENT,
    HISTORY_COMPACT_THRESHOLD,
    HISTORY_KEEP_RECENT,
    SAGA_MAX_CHARS,
    chapter_end,
    compact_history,
    digest,
)
from src.engine.runtime.replay import ReplayTape
from src.engine.state.models import HistoryEvent, WorldState
from src.interface.session_log import Logger

PROSE = "Brenna walked the dock at dusk. " * 12
REPLY = f"TITLE: The Dock\nPROSE:\n{PROSE}\n\n{PROSE}\nSUMMARY: Brenna walked the dock."


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


def _events(n: int, start: int = 0) -> list[HistoryEvent]:
    return [
        HistoryEvent(
            text=f"event {i}", location="tavern", characters=["hero"], minutes_elapsed=5
        )
        for i in range(start, start + n)
    ]


def _fake_agent(monkeypatch, replies: list[str] | None = None) -> list[dict]:
    """Patch the chronicler; returns the list of calls it received."""
    calls: list[dict] = []

    async def fake_run(prompt, instructions, usage_limits, model_settings):
        calls.append({"prompt": prompt, "instructions": instructions})

        class _Result:
            output = (
                replies.pop(0)
                if replies
                else ("the saga" if instructions == SAGA_INSTRUCTIONS else REPLY)
            )

            def all_messages(self):
                return []

        return _Result()

    monkeypatch.setattr("src.engine.runtime.chronicle.chronicler_agent.run", fake_run)
    return calls


def _compact(state, tmp_path, replay=None):
    asyncio.run(compact_history(state, _Logger(), replay, tmp_path, "hero"))


def _chapter(tmp_path, number: int) -> dict:
    path = tmp_path / "chapters" / f"chapter_{number:04d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ============ digest() fallback ============


def test_digest_joins_event_texts():
    events = [
        HistoryEvent(text="hero enters", location="tavern"),
        HistoryEvent(text="hero waits", location="tavern"),
    ]
    assert digest(events) == "hero enters hero waits"


def test_digest_truncates_long_output():
    events = [HistoryEvent(text="word " * 300, location="tavern")]
    result = digest(events)
    assert len(result) <= 600
    assert result.endswith("…")


# ============ chapter selection ============


def test_below_threshold_does_not_compact(tmp_path):
    state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD))
    _compact(state, tmp_path)
    assert len(state.history) == HISTORY_COMPACT_THRESHOLD
    assert state.chronicle == []
    assert not (tmp_path / "chapters").exists()


def test_chapter_end_defaults_to_keeping_the_recent_tail():
    history = _events(HISTORY_COMPACT_THRESHOLD + 1)
    assert chapter_end(history) == len(history) - HISTORY_KEEP_RECENT


def test_chapter_end_moves_to_the_next_change_of_scene():
    history = _events(HISTORY_COMPACT_THRESHOLD + 1)
    scene_change = len(history) - HISTORY_KEEP_RECENT + 3
    for event in history[scene_change:]:
        event.location = "forest"
    assert chapter_end(history) == scene_change


# ============ chapter files ============


def test_compaction_writes_chapter_file_with_its_raw_events(tmp_path, monkeypatch):
    calls = _fake_agent(monkeypatch)
    state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD + 1))
    archived = state.history[:-HISTORY_KEEP_RECENT]

    _compact(state, tmp_path)

    chapter = _chapter(tmp_path, 1)
    assert set(chapter) == {"number", "title", "prose", "summary", "events"}
    assert chapter["number"] == 1
    assert chapter["title"] == "The Dock"
    assert chapter["prose"] == f"{PROSE.strip()}\n\n{PROSE.strip()}"
    assert chapter["summary"] == "Brenna walked the dock."
    assert chapter["events"] == [
        {
            "text": e.text,
            "location": "tavern",
            "characters": ["hero"],
            "minutes_elapsed": 5,
        }
        for e in archived
    ]
    assert [e.text for e in state.history] == [
        f"event {i}" for i in range(len(archived), HISTORY_COMPACT_THRESHOLD + 1)
    ]
    assert state.chronicle == ["Brenna walked the dock."]
    assert "event 0" in calls[0]["prompt"] and "first chapter" in calls[0]["prompt"]


def test_next_chapter_is_numbered_on_and_continues_from_the_last_ending(
    tmp_path, monkeypatch
):
    calls = _fake_agent(monkeypatch)
    state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD + 1))
    _compact(state, tmp_path)
    state.history += _events(HISTORY_COMPACT_THRESHOLD, start=100)
    _compact(state, tmp_path)

    assert _chapter(tmp_path, 2)["number"] == 2
    assert "how chapter 1 ended" in calls[1]["prompt"]
    assert PROSE.strip() in calls[1]["prompt"]
    assert "Brenna walked the dock." in calls[1]["prompt"]  # chronicle as memory


def _assert_fallback_chapter(tmp_path, state, archived):
    chapter = _chapter(tmp_path, 1)
    assert chapter["title"] == "Chapter 1"
    assert chapter["prose"] == " ".join(e.text for e in archived)
    assert chapter["summary"] == digest(archived)
    assert len(chapter["events"]) == len(archived)
    assert state.chronicle == [digest(archived)]
    assert len(state.history) == HISTORY_KEEP_RECENT


def test_failed_narrator_call_still_archives_a_fallback_chapter(tmp_path, monkeypatch):
    for error in (
        UsageLimitExceeded("too many requests"),
        ModelAPIError("ollama", "Connection error"),
    ):

        async def fake_run(*_args, error=error, **_kwargs):
            raise error

        monkeypatch.setattr(
            "src.engine.runtime.chronicle.chronicler_agent.run", fake_run
        )
        run_dir = tmp_path / type(error).__name__
        state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD + 1))
        archived = state.history[:-HISTORY_KEEP_RECENT]
        _compact(state, run_dir)
        _assert_fallback_chapter(run_dir, state, archived)


def test_junk_reply_falls_back_per_field(tmp_path, monkeypatch):
    _fake_agent(monkeypatch, ["Sure! Here is the chapter."])
    state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD + 1))
    archived = state.history[:-HISTORY_KEEP_RECENT]
    _compact(state, tmp_path)
    _assert_fallback_chapter(tmp_path, state, archived)


# ============ bounded chronicle ============


def test_chronicle_folds_old_summaries_into_one_saga(tmp_path, monkeypatch):
    calls = _fake_agent(monkeypatch)
    state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD + 1))
    state.chronicle = [f"summary {i}" for i in range(CHRONICLE_RECENT + 1)]

    _compact(state, tmp_path)

    saga_call = calls[-1]
    assert saga_call["instructions"] == SAGA_INSTRUCTIONS
    assert saga_call["prompt"] == "summary 0\n\nsummary 1"
    assert state.chronicle == [
        "the saga",
        *[f"summary {i}" for i in range(2, CHRONICLE_RECENT + 1)],
        "Brenna walked the dock.",
    ]


def test_chronicle_size_stays_flat_over_many_chapters(tmp_path, monkeypatch):
    long_summary = "x " * 400
    _fake_agent(monkeypatch)
    monkeypatch.setattr(
        chronicle, "parse_chapter", lambda _r: ("T", long_summary, PROSE)
    )
    state = WorldState()
    sizes = []
    for n in range(40):
        state.history += _events(HISTORY_COMPACT_THRESHOLD + 1 - len(state.history))
        _compact(state, tmp_path)
        sizes.append(sum(map(len, state.chronicle)))
    assert len(state.chronicle) == CHRONICLE_RECENT + 1
    assert max(sizes) <= SAGA_MAX_CHARS + CHRONICLE_RECENT * 600
    assert len(list((tmp_path / "chapters").iterdir())) == 40


def test_failed_fold_keeps_the_origin_and_the_latest_bounded(tmp_path, monkeypatch):
    _fake_agent(monkeypatch, [REPLY, ""])
    state = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD + 1))
    state.chronicle = ["ORIGIN " + "a " * 600, "b " * 600 + " LATEST", "c", "d"]
    _compact(state, tmp_path)
    saga = state.chronicle[0]
    assert len(saga) <= SAGA_MAX_CHARS
    assert saga.startswith("ORIGIN") and saga.endswith("LATEST")
    assert state.chronicle[1:] == ["c", "d", "Brenna walked the dock."]


# ============ replay ============


def test_playback_reproduces_chapters_without_calling_the_agent(tmp_path, monkeypatch):
    _fake_agent(monkeypatch)
    tape = tmp_path / "run.jsonl"
    recorded = WorldState(history=_events(HISTORY_COMPACT_THRESHOLD + 1))
    recorded.chronicle = ["a", "b", "c", "d"]  # forces a saga fold too
    played = recorded.model_copy(deep=True)
    _compact(recorded, tmp_path / "live", ReplayTape.recording(tape))

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("the chronicler agent was called during replay")

    monkeypatch.setattr("src.engine.runtime.chronicle.chronicler_agent.run", unexpected)
    playback = ReplayTape.playback(tape)
    _compact(played, tmp_path / "replay", playback)

    playback.assert_consumed()
    assert played == recorded
    assert _chapter(tmp_path / "replay", 1) == _chapter(tmp_path / "live", 1)
