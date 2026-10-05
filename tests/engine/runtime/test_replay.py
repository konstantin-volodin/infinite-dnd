import asyncio
from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from src.agents.character.tools import Action, Check, Speak, Travel, Wait
from src.agents.dm.agent import DMResult
from src.agents.dm.director import DirectorResult
from src.agents.dm.tools import Create, Modify
from src.engine.runtime.loop import flow_agent_turn, flow_director, flow_dm, tick
from src.engine.runtime.replay import ReplayError, ReplayTape
from src.engine.state.models import Character, HistoryEvent, Location, Quest, WorldState
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


def _state() -> WorldState:
    return WorldState(
        locations={"tavern": Location(id="tavern")},
        characters={"hero": Character(id="hero", role="warrior", location="tavern")},
        quests={"q1": Quest(id="q1", title="A Quest", description="", owner="hero")},
    )


def test_tape_round_trips_all_structured_output_types(tmp_path):
    path = tmp_path / "run.jsonl"
    recording = ReplayTape.recording(path)
    recording.bind_context("crossroads", "hero")
    recording.character("hero", Speak(actor="hero", target="innkeeper", message="Hello"))
    recording.dm(DMResult(
        creates=[Create(type="item", name="key", location="tavern")],
        modifies=[Modify(action="update_quest", target_id="q1", advance=True)],
        minutes=[3],
    ))
    recording.director(DirectorResult(
        event="The bell begins to toll.",
        create=Create(type="npc", name="Bailiff", location="tavern"),
        quest_id="q1",
    ))
    recording.chronicle("The hero spent a week rebuilding the tavern's trade.")

    playback = ReplayTape.playback(path)
    assert playback.resolve_context(None, None) == ("crossroads", "hero")
    assert playback.character("hero") == Speak(actor="hero", target="innkeeper", message="Hello")
    assert playback.dm() == DMResult(
        creates=[Create(type="item", name="key", location="tavern")],
        modifies=[Modify(action="update_quest", target_id="q1", advance=True)],
        minutes=[3],
    )
    assert playback.director() == DirectorResult(
        event="The bell begins to toll.",
        create=Create(type="npc", name="Bailiff", location="tavern"),
        quest_id="q1",
    )
    assert playback.chronicle() == "The hero spent a week rebuilding the tavern's trade."
    playback.assert_consumed()


def test_replay_context_rejects_conflicting_cli_selection(tmp_path):
    path = tmp_path / "run.jsonl"
    ReplayTape.recording(path).bind_context("crossroads", "hero")
    playback = ReplayTape.playback(path)

    with pytest.raises(ReplayError, match="Replay scenario is 'crossroads'"):
        playback.resolve_context("other-world", None)


def test_replay_fails_loudly_when_scheduler_does_not_match(tmp_path):
    path = tmp_path / "run.jsonl"
    ReplayTape.recording(path).character("hero", Wait(actor="hero"))
    playback = ReplayTape.playback(path)

    with pytest.raises(ReplayError, match="expected dm:dm, found character:hero"):
        playback.dm()


def test_tick_playback_never_calls_character_or_dm_agents(tmp_path, monkeypatch):
    path = tmp_path / "run.jsonl"
    tape = ReplayTape.recording(path)
    tape.character("hero", Wait(actor="hero"))
    tape.dm(DMResult(creates=[], modifies=[], minutes=[4]))

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("an LLM agent was called during replay")

    monkeypatch.setattr("src.engine.runtime.loop.character_agent.run", unexpected)
    monkeypatch.setattr("src.engine.runtime.loop.dm_agent.run", unexpected)
    state = _state()
    asyncio.run(tick("hero", state, 0, _Logger(), replay=ReplayTape.playback(path)))

    assert state.history == [
        HistoryEvent(text="hero waits.", location="tavern", characters=["hero"], minutes_elapsed=5)
    ]
    assert state.minutes_elapsed == 5


def test_flow_playback_never_calls_director_agent(tmp_path, monkeypatch):
    path = tmp_path / "run.jsonl"
    ReplayTape.recording(path).director(DirectorResult(event="A horn sounds.", quest_id="q1"))

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("the director agent was called during replay")

    monkeypatch.setattr("src.engine.runtime.loop.director_agent.run", unexpected)
    state = _state()
    asyncio.run(flow_director(state, "tavern", _Logger(), ReplayTape.playback(path)))

    assert state.history[-1].text == "A horn sounds."
    assert state.director_interventions == {"q1": 1}


def test_action_playback_restores_effects_without_resolver_agent(tmp_path, monkeypatch):
    path = tmp_path / "run.jsonl"
    recorded_state = _state()
    recorded_state.characters["hero"].knowledge.append("The loose brick hides a key.")
    recorded_state.history.append(HistoryEvent(
        text="hero finds a key behind the loose brick.",
        location="tavern",
        characters=["hero"],
    ))
    tape = ReplayTape.recording(path)
    tape.character("hero", Action(actor="hero", description="Search the loose brick"))
    tape.action_resolution("hero", recorded_state, "A hidden key is found.")
    tape.dm(DMResult(creates=[], modifies=[], minutes=[2]))

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("the action resolver agent was called during replay")

    monkeypatch.setattr("src.agents.action_resolver.agent.agent.run", unexpected)
    state = _state()
    playback = ReplayTape.playback(path)
    asyncio.run(tick("hero", state, 0, _Logger(), replay=playback))

    assert state.characters["hero"].knowledge == ["The loose brick hides a key."]
    assert state.history[-1].text == "hero finds a key behind the loose brick."
    assert state.history[-1].minutes_elapsed == 5
    playback.assert_consumed()


def test_empty_event_dm_flow_does_not_consume_tape(tmp_path):
    path = tmp_path / "run.jsonl"
    recording = ReplayTape.recording(path)
    recording.character("hero", Wait(actor="hero"))
    playback = ReplayTape.playback(path)

    assert asyncio.run(flow_dm(_state(), [], _Logger(), playback)) == DMResult([], [], [])
    assert asyncio.run(flow_agent_turn("hero", _state(), _Logger(), replay=playback)) == Wait(actor="hero")
    playback.assert_consumed()


def test_check_round_trips_and_playback_restores_recorded_roll(tmp_path):
    path = tmp_path / "check.jsonl"
    check = Check(actor="hero", ability="wisdom", description="notices the tripwire", difficulty=16)
    recorded_state = _state()
    recorded_state.history.append(HistoryEvent(
        text="hero succeeds: notices the tripwire [wisdom; 18+0=18 vs DC 16].",
        location="tavern", characters=["hero"],
    ))
    tape = ReplayTape.recording(path)
    tape.character("hero", check)
    tape.action_resolution("hero", recorded_state, recorded_state.history[-1].text)
    tape.dm(DMResult(creates=[], modifies=[], minutes=[1]))

    state = _state()
    playback = ReplayTape.playback(path)
    asyncio.run(tick("hero", state, 0, _Logger(), replay=playback))
    assert state.history[-1].text == recorded_state.history[-1].text
    assert state.history[-1].minutes_elapsed == 1
    playback.assert_consumed()


def test_travel_time_floor_applies_when_recorded_dm_underestimates(tmp_path):
    path = tmp_path / "travel.jsonl"
    recording = ReplayTape.recording(path)
    recording.character("hero", Travel(actor="hero", destination="road"))
    recording.dm(DMResult(creates=[], modifies=[], minutes=[0]))

    state = _state()
    state.locations["tavern"].connections.append("road")
    state.locations["road"] = Location(id="road", connections=["tavern"])
    playback = ReplayTape.playback(path)

    asyncio.run(tick("hero", state, 0, _Logger(), replay=playback))

    assert state.history[-1].minutes_elapsed == 10
    assert state.minutes_elapsed == 10
    playback.assert_consumed()
