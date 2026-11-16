import asyncio
from unittest.mock import MagicMock as Mock

from src.agents.dm.storyteller import Arc, Person, Place
from src.engine.runtime import campaign
from src.engine.runtime.campaign import (
    FORCE_PROGRESS_TICKS,
    abandon_stalled_quests,
    campaign_upkeep,
    hand_over,
    open_arc,
)
from src.engine.runtime.replay import ReplayTape
from src.engine.state.models import (
    Character,
    CharacterStats,
    Location,
    Quest,
    WorldState,
)


def _state() -> WorldState:
    return WorldState(
        protagonist="hero",
        locations={
            "tavern": Location(id="tavern", connections=["road"]),
            "road": Location(id="road", connections=["tavern"]),
        },
        characters={
            "hero": Character(id="hero", location="tavern"),
            "barkeep": Character(id="barkeep", location="tavern"),
            "bram": Character(id="bram", location="road"),
        },
        quests={
            "q1": Quest(
                id="q1",
                title="Clear the Cave",
                owner="hero",
                plan=["scout the cave", "clear the cave"],
            )
        },
    )


def _arc() -> Arc:
    return Arc(
        hook="Word comes from Saltmere.",
        places=[Place(name="Saltmere", connects_to="road"), Place(name="Bell Tower")],
        people=[Person(name="Ivo", goal="hide debts", location="Saltmere")],
        quest_title="The Drowned Bells",
        quest_description="Learn who silenced the bells.",
        plan=["question Ivo at Saltmere", "climb the Bell Tower", "face the thief"],
    )


def _kill_hero(state: WorldState) -> None:
    state.characters["hero"].stats = CharacterStats(hp=0, max_hp=5)


# ============ SUCCESSOR ============


def test_living_hero_keeps_the_story():
    assert hand_over(_state()) is None


def test_successor_prefers_a_tied_character_and_inherits_the_quest():
    state = _state()
    state.characters["bram"].relationships["hero"] = "sworn friend"
    _kill_hero(state)

    assert hand_over(state) == ("hero", "bram")

    assert state.protagonist == "bram"
    assert state.quests["q1"].owner == "bram"
    assert "hero died pursuing 'Clear the Cave'." in state.characters["bram"].knowledge
    assert state.characters["bram"].goal == "Finish what hero began: Clear the Cave."
    assert state.history[-1].text == (
        "hero has fallen. bram vows to carry on and takes up 'Clear the Cave'."
    )


def test_successor_tie_breaks_on_the_heros_location():
    state = _state()
    state.characters["hero"].relationships = {"bram": "friend", "barkeep": "friend"}
    _kill_hero(state)

    assert hand_over(state) == ("hero", "barkeep")


def test_dead_or_untied_characters_cannot_succeed_so_a_newcomer_arrives():
    state = _state()
    state.characters["hero"].relationships["bram"] = "friend"
    state.characters["bram"].stats = CharacterStats(hp=0, max_hp=5)
    state.characters["wandering-sellsword"] = Character(
        id="wandering-sellsword", location="road"
    )
    _kill_hero(state)

    assert hand_over(state) == ("hero", "wandering-sellsword-2")

    heir = state.characters["wandering-sellsword-2"]
    assert (heir.location, heir.stats.hp) == ("tavern", 10)
    assert state.quests["q1"].owner == "wandering-sellsword-2"
    WorldState.model_validate(state.model_dump())


# ============ FORCED PROGRESS ============


def test_quest_is_abandoned_exactly_at_the_stall_bound():
    state = _state()
    state.time = FORCE_PROGRESS_TICKS - 1
    assert abandon_stalled_quests(state) == []

    state.time = FORCE_PROGRESS_TICKS
    assert abandon_stalled_quests(state) == ["q1"]

    quest = state.quests["q1"]
    assert quest.status == "failed"
    assert quest.steps[-1] == "abandoned, never managed to scout the cave"
    assert "The trail goes cold: 'Clear the Cave' is lost." in state.history[-1].text
    assert state.history[-1].characters == ["hero", "barkeep"]


def test_progress_resets_the_stall_bound():
    state = _state()
    state.time = FORCE_PROGRESS_TICKS
    state.quests["q1"].last_progress = 5

    assert abandon_stalled_quests(state) == []


# ============ ARCS ============


def _fake_storyteller(monkeypatch, output):
    class Result:
        def __init__(self):
            self.output = output

        def all_messages(self):
            return []

    async def run(*_args, **_kwargs):
        if isinstance(output, Exception):
            raise output
        return Result()

    monkeypatch.setattr(campaign.storyteller_agent, "run", run)


def test_open_arc_applies_the_storyteller_output(monkeypatch, tmp_path):
    state = _state()
    _fake_storyteller(monkeypatch, _arc())
    tape = ReplayTape.recording(tmp_path / "tape.jsonl")

    assert asyncio.run(open_arc(state, Mock(), tape)) == _arc()

    assert state.quests["the-drowned-bells"].owner == "hero"
    assert "saltmere" in state.locations["road"].connections

    replayed = _state()
    assert asyncio.run(open_arc(replayed, Mock(), ReplayTape.playback(tape.path)))
    assert replayed == state


def test_storyteller_failure_leaves_the_world_unchanged(monkeypatch):
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    state = _state()
    _fake_storyteller(monkeypatch, UnexpectedModelBehavior("bad output"))

    assert asyncio.run(open_arc(state, Mock())) is None
    assert state == _state()


def test_upkeep_opens_an_arc_only_when_the_hero_has_no_active_quest(monkeypatch):
    state = _state()
    _fake_storyteller(monkeypatch, _arc())
    logger = Mock()

    asyncio.run(campaign_upkeep(state, logger))
    assert "the-drowned-bells" not in state.quests

    state.quests["q1"].status = "failed"
    asyncio.run(campaign_upkeep(state, logger))
    assert state.quests["the-drowned-bells"].status == "active"
    logger.log_event.assert_any_call(
        "arc_opened",
        quest="The Drowned Bells",
        places=["saltmere", "bell-tower"],
        people=["ivo"],
        faction=None,
    )
