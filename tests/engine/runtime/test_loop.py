import asyncio

from pydantic_ai.exceptions import ModelAPIError

from src.agents.character.tools import Action, Attack, Speak, Travel, Wait
from src.engine.runtime.loop import (
    _STALL_QUIET_TICKS,
    _advance_grounded_objective,
    _can_advance_final_objective,
    _campaign_outcome,
    _is_stalled,
    _minimum_action_minutes,
    _record_intervention,
    _record_resolution_if_needed,
    _run_game,
    tick,
)
from src.agents.dm.agent import DMResult
from src.agents.dm.tools import Modify
from src.engine.state.models import Character, HistoryEvent, Location, Quest, WorldState
from src.engine.state.operations import WorldOperations


def _state() -> WorldState:
    return WorldState(
        locations={"tavern": Location(id="tavern")},
        characters={"hero": Character(id="hero", role="warrior", location="tavern")},
        quests={"q1": Quest(id="q1", title="Clear the Cave", description="", owner="hero", plan=["scout", "clear"])},
    )


def _event(text: str) -> HistoryEvent:
    return HistoryEvent(text=text, location="tavern", characters=["hero"])


# ============ STALL DETECTION ============

def test_fresh_state_not_stalled():
    assert not _is_stalled(_state())


def test_quiet_ticks_trigger_stall():
    state = _state()
    state.time = _STALL_QUIET_TICKS
    assert _is_stalled(state)


def test_quest_advancement_resets_quiet_clock():
    state = _state()
    state.time = _STALL_QUIET_TICKS
    WorldOperations(state).advance_quest("q1", advance=True)
    assert state.last_quest_advance_time == state.time
    assert not _is_stalled(state)


def test_idle_chatter_triggers_stall():
    state = _state()
    for text in ('hero says: "hm"', "hero waits.", 'hero says to bob: "well?"', "hero waits.", 'hero says: "so..."'):
        state.history.append(_event(text))
    assert _is_stalled(state)


def test_mixed_recent_events_not_stalled():
    state = _state()
    for text in ('hero says: "hm"', "hero waits.", "hero picks up 'sword'.", "hero waits.", 'hero says: "so..."'):
        state.history.append(_event(text))
    assert not _is_stalled(state)


def test_fewer_than_window_idle_events_not_stalled():
    state = _state()
    for text in ("hero waits.", "hero waits."):
        state.history.append(_event(text))
    assert not _is_stalled(state)


def test_director_reset_suppresses_consecutive_fire():
    state = _state()
    state.time = _STALL_QUIET_TICKS
    assert _is_stalled(state)
    # what tick() does after a director beat: reset the clock; the beat lands a non-idle event
    state.last_quest_advance_time = state.time
    state.history.append(_event("A stranger bursts in demanding payment."))
    assert not _is_stalled(state)


# ============ TURN FEEDBACK ============

def test_resolution_without_state_event_is_recorded_for_player_feedback():
    state = _state()

    events = _record_resolution_if_needed(state, "hero", 0, "  Nothing useful is hidden here.  ")

    assert [event.text for event in events] == ["Nothing useful is hidden here."]
    assert events[0].location == "tavern"
    assert events[0].characters == ["hero"]


def test_existing_resolution_events_are_not_duplicated():
    state = _state()
    state.history.append(_event("hero finds a brass key."))

    events = _record_resolution_if_needed(state, "hero", 0, "found a brass key")

    assert [event.text for event in events] == ["hero finds a brass key."]


def test_direct_actions_have_deterministic_time_floors():
    assert _minimum_action_minutes(Travel(actor="hero", destination="road")) == 10
    assert _minimum_action_minutes(Wait(actor="hero")) == 5
    assert _minimum_action_minutes(Speak(actor="hero", message="hello")) == 1
    assert _minimum_action_minutes(Attack(actor="hero", target="bandit")) == 1
    assert _minimum_action_minutes(Action(actor="hero", description="search the archive")) == 5
    assert _minimum_action_minutes(Action(actor="hero", description="open the door")) == 1


def test_final_objective_requires_an_active_resolution_attempt():
    quest = _state().quests["q1"]
    quest.current_step = len(quest.plan) - 1

    assert not _can_advance_final_objective(quest, Travel(actor="hero", destination="cave"))
    assert not _can_advance_final_objective(quest, Wait(actor="hero"))
    assert not _can_advance_final_objective(quest, Action(actor="ally", description="clear the cave"))
    assert not _can_advance_final_objective(quest, Action(actor="hero", description="clear the cave"))
    assert _can_advance_final_objective(
        quest, Action(actor="hero", description="clear the cave"), [_event("hero cleared the cave.")]
    )
    quest.plan[-1] = "report the smugglers"
    assert _can_advance_final_objective(
        quest, Speak(actor="hero", message="I report the smugglers."), [_event("hero reported the smugglers.")]
    )


def test_final_objective_rejects_clue_discovery_without_resolution_evidence():
    state = _state()
    quest = state.quests["q1"]
    quest.plan[-1] = "recover the lost amulet"
    quest.current_step = len(quest.plan) - 1

    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="search the cave for clues"),
        [_event("hero learns: someone recovered the lost amulet.")],
    )
    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="search the cave for clues"),
        [_event("hero discovers the bandit recovered the lost amulet.")],
    )
    assert _can_advance_final_objective(
        quest,
        Action(actor="hero", description="take the amulet"),
        [_event("hero picks up the lost amulet.")],
    )
    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="take the amulet"),
        [_event("hero recovered the lost map.")],
    )
    quest.plan[-1] = "recover the lost amulet before dawn"
    assert _can_advance_final_objective(
        quest,
        Action(actor="hero", description="take the amulet"),
        [_event("hero recovered the lost amulet.")],
    )

    quest.plan[-1] = "find the lost amulet"
    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="search the cave for clues"),
        [_event("hero found a clue about the lost amulet.")],
    )
    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="search the cave for clues"),
        [_event("hero found the location of the lost amulet.")],
    )
    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="search the cave for clues"),
        [_event("hero found a trail to the lost amulet.")],
    )
    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="search the cave for clues"),
        [_event("hero found a map leading to the lost amulet.")],
    )
    for indirect_event in (
        "hero found Kaelen's trail.",
        "hero found the trail of Kaelen.",
    ):
        assert not _can_advance_final_objective(
            quest,
            Action(actor="hero", description="search the cave for clues"),
            [_event(indirect_event)],
        )
    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="search the cave for clues"),
        [_event("hero found evidence about the lost amulet.")],
    )
    for indirect_event in (
        "hero found a theft report about the lost amulet.",
        "hero found a poster about the lost amulet.",
        "hero found a rumor about the lost amulet.",
        "hero found the cave where the lost amulet was last seen.",
        "hero found the place the lost amulet was last seen.",
    ):
        assert not _can_advance_final_objective(
            quest,
            Action(actor="hero", description="search the cave for clues"),
            [_event(indirect_event)],
        )
    quest.plan[-1] = "find Kaelen"
    for indirect_event in (
        "hero found Kaelen's house.",
        "hero found the home where Kaelen lives.",
        "hero found the cave containing Kaelen.",
        "hero found Kaelen's hiding place.",
        "hero found the hiding place of Kaelen.",
        "hero found where Kaelen is hiding.",
    ):
        assert not _can_advance_final_objective(
            quest,
            Action(actor="hero", description="search the cave for clues"),
            [_event(indirect_event)],
        )
    assert _can_advance_final_objective(
        quest,
        Action(actor="hero", description="find Kaelen"),
        [_event("hero found Kaelen at the cave.")],
    )
    quest.plan[-1] = "find the lost amulet"
    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="search the cave for clues"),
        [_event("hero found no trace of the lost amulet.")],
    )
    assert _can_advance_final_objective(
        quest,
        Action(actor="hero", description="take the amulet"),
        [_event("hero found the lost amulet.")],
    )
    assert _can_advance_final_objective(
        quest,
        Action(actor="hero", description="find the amulet"),
        [_event("hero found the lost amulet, last seen near the river.")],
    )


def test_final_track_objective_requires_direct_target_resolution():
    quest = _state().quests["q1"]
    quest.plan[-1] = "track down Kaelen"
    quest.current_step = len(quest.plan) - 1

    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="read the notice"),
        [_event("hero tracked down a clue about Kaelen.")],
    )
    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="follow the lead"),
        [_event("hero tracked down the location of Kaelen at the docks.")],
    )
    assert _can_advance_final_objective(
        quest,
        Action(actor="hero", description="find Kaelen"),
        [_event("hero tracked down Kaelen at the docks.")],
    )


def test_tick_rejects_direct_final_completion_from_clue_discovery(monkeypatch):
    state = _state()
    quest = state.quests["q1"]
    quest.plan[-1] = "find the lost amulet"
    quest.current_step = len(quest.plan) - 1

    async def choose_action(*_args):
        return Action(actor="hero", description="search the cave for clues")

    async def resolve_intent(tool, world, **_kwargs):
        if isinstance(tool, Action):
            event = _event("hero found a clue about the lost amulet.")
            world.history.append(event)
            return event.text
        raise AssertionError("the blocked completion must not reach the resolver")

    async def report_clue(*_args):
        return DMResult(
            creates=[],
            modifies=[Modify(action="update_quest", target_id="q1", status="completed")],
            minutes=[1],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_intent)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_clue)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert quest.status == "active"
    assert quest.current_step == len(quest.plan) - 1
    assert state.characters["hero"].stats.xp == 0


def test_compound_final_objective_accepts_either_resolution_branch():
    quest = _state().quests["q1"]
    quest.plan[-1] = "gather proof and confront or report the smugglers"
    quest.current_step = len(quest.plan) - 1

    for event_text in (
        "hero gathered proof and confronted the smugglers.",
        "hero confronted the smugglers with proof.",
        "hero reported the smugglers with proof.",
    ):
        assert _can_advance_final_objective(
            quest,
            Action(actor="hero", description="gather proof and resolve the smugglers"),
            [_event(event_text)],
        )
    assert not _can_advance_final_objective(
        quest,
        Action(actor="hero", description="gather proof"),
        [_event("hero gathered proof about the smugglers.")],
    )


def test_grounded_discovery_advances_search_objective_when_dm_does_not():
    state = _state()
    state.quests["q1"].plan = ["search tavern for clues", "clear the cave"]
    event = _event("hero learns: the bandits use the north tunnel.")
    state.history.append(event)

    results = _advance_grounded_objective(
        state,
        "hero",
        Action(actor="hero", description="search the bar for signs"),
        [event],
        {"q1": 0},
    )

    assert state.quests["q1"].current_step == 1
    assert state.characters["hero"].stats.xp == 10
    assert "Quest 'q1' updated" in results[0]


def test_grounded_objective_fallback_requires_new_evidence_and_matching_location():
    state = _state()
    state.quests["q1"].plan = ["search forest for clues"]

    assert _advance_grounded_objective(
        state,
        "hero",
        Action(actor="hero", description="search the tavern"),
        [_event("hero learns: the cellar is damp.")],
        {"q1": 0},
    ) == []
    state.quests["q1"].plan = ["search tavern for clues"]
    assert _advance_grounded_objective(
        state,
        "hero",
        Action(actor="hero", description="search the tavern"),
        [_event("hero searches but finds nothing useful.")],
        {"q1": 0},
    ) == []
    assert state.quests["q1"].current_step == 0


def test_grounded_objective_fallback_cannot_complete_from_final_clue_discovery():
    state = _state()
    state.quests["q1"].plan = ["find clues in tavern"]
    event = _event("hero finds a clue in the tavern.")

    assert _advance_grounded_objective(
        state,
        "hero",
        Action(actor="hero", description="search the tavern"),
        [event],
        {"q1": 0},
    ) == []
    assert state.quests["q1"].current_step == 0
    assert state.quests["q1"].status == "active"
    assert state.characters["hero"].stats.xp == 0


# ============ CAMPAIGN OUTCOME ============

def test_campaign_ends_in_victory_when_all_owned_quests_are_completed():
    state = _state()
    state.quests["q1"].status = "completed"
    state.quests["side"] = Quest(
        id="side",
        title="Return the Key",
        description="",
        owner="hero",
        status="Completed",
    )

    assert _campaign_outcome(state, "hero") == "completed"


def test_campaign_ends_in_defeat_when_resolved_owned_quest_failed():
    state = _state()
    state.quests["q1"].status = "failed"

    assert _campaign_outcome(state, "hero") == "failed"


def test_campaign_continues_with_active_owned_quest_or_no_owned_quests():
    state = _state()
    assert _campaign_outcome(state, "hero") is None

    state.quests["q1"].owner = "someone-else"
    assert _campaign_outcome(state, "hero") is None


def test_game_loop_stops_immediately_after_campaign_victory(monkeypatch):
    state = _state()
    ticks: list[int] = []
    logged_events: list[str] = []

    class Manager:
        run_id = "test-run"
        scenario = "demo"
        manifest = {"pc": "hero", "title": "Test Quest", "hook": "Finish it."}

        def __init__(self, **_kwargs): pass
        def init_state(self): return state
        def latest_snapshot_name(self): return None
        def save_state(self, _state): pass

    class Log:
        def __init__(self, **_kwargs): pass
        def log_event(self, event, **_kwargs): logged_events.append(event)
        def log_turn(self, _turn): pass
        def close(self, **_kwargs): pass

    async def finish_quest(_pc_id, world, tick_index, _logger, _controller, _replay):
        ticks.append(tick_index)
        world.quests["q1"].status = "completed"

    async def skip_compaction(*_args): pass

    monkeypatch.setattr("src.engine.runtime.loop.StateManager", Manager)
    monkeypatch.setattr("src.engine.runtime.loop.Logger", Log)
    monkeypatch.setattr("src.engine.runtime.loop.tick", finish_quest)
    monkeypatch.setattr("src.engine.runtime.loop.compact_history", skip_compaction)

    asyncio.run(_run_game("demo", "hero", max_turns=5, new_character=None))

    assert ticks == [0]
    assert "campaign_completed" in logged_events


def test_game_loop_records_provider_failure_without_persisting_partial_turn(monkeypatch):
    state = _state()
    logged_events: list[tuple[str, dict]] = []
    saves: list[int] = []
    closed: list[dict] = []

    class Manager:
        run_id = "test-run"
        scenario = "demo"
        manifest = {"pc": "hero", "title": "Test Quest", "hook": "Finish it."}

        def __init__(self, **_kwargs): pass
        def init_state(self): return state
        def latest_snapshot_name(self): return None
        def save_state(self, world): saves.append(world.time)

    class Log:
        def __init__(self, **_kwargs): pass
        def log_event(self, event, **kwargs): logged_events.append((event, kwargs))
        def log_turn(self, _turn): pass
        def close(self, **kwargs): closed.append(kwargs)

    async def fail_provider(*_args):
        raise ModelAPIError("anthropic", "Connection error")

    monkeypatch.setattr("src.engine.runtime.loop.StateManager", Manager)
    monkeypatch.setattr("src.engine.runtime.loop.Logger", Log)
    monkeypatch.setattr("src.engine.runtime.loop.tick", fail_provider)

    asyncio.run(_run_game("demo", "hero", max_turns=5, new_character=None))

    assert [event for event, _ in logged_events].count("run_error") == 1
    assert dict(logged_events)["run_error"] == {
        "error": "ModelAPIError",
        "message": "Connection error",
    }
    assert saves == [0]
    assert closed == [{"turns_completed": 0}]


# ============ ESCALATION COUNTER ============

def test_record_intervention_targets_known_quest():
    state = _state()
    assert _record_intervention(state, "q1") == "q1"
    assert _record_intervention(state, "q1") == "q1"
    assert state.director_interventions == {"q1": 2}


def test_record_intervention_falls_back_to_world():
    state = _state()
    assert _record_intervention(state, None) == "world"
    assert _record_intervention(state, "no-such-quest") == "world"
    assert state.director_interventions == {"world": 2}
