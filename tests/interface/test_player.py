import asyncio

import pytest

from src.agents.character.tools import Action, Attack, Check, Speak, Travel, Wait
from src.engine.state.models import (
    Character,
    Faction,
    HistoryEvent,
    Location,
    ProgressClock,
    Quest,
    WorldState,
)
from src.interface.player import _describe_situation, _parse_intent, _InputError, console_pc_controller


def _state() -> WorldState:
    return WorldState(
        locations={
            "tavern": Location(id="tavern", description="A dim tavern.", connections=["square"], items=["lantern"]),
            "square": Location(id="square", description="The town square."),
        },
        characters={
            "hero": Character(id="hero", role="warrior", location="tavern", inventory=["sword"]),
            "bob": Character(id="bob", role="bartender", location="tavern"),
        },
    )


# ============ free-text default ============

def test_plain_text_becomes_action():
    intent = _parse_intent("hero", _state(), "search the bar for clues")
    assert intent == Action(actor="hero", description="search the bar for clues")


def test_blank_input_reprompts():
    with pytest.raises(_InputError):
        _parse_intent("hero", _state(), "   ")


# ============ slash commands ============

def test_wait_command():
    assert _parse_intent("hero", _state(), "/wait") == Wait(actor="hero")


def test_travel_command_resolves_fuzzy_destination():
    intent = _parse_intent("hero", _state(), "/travel Square")
    assert intent == Travel(actor="hero", destination="square")


def test_travel_command_rejects_unknown_destination():
    with pytest.raises(_InputError, match="Unknown destination"):
        _parse_intent("hero", _state(), "/travel nowhere")


def test_travel_command_requires_argument():
    with pytest.raises(_InputError, match="Usage: /travel"):
        _parse_intent("hero", _state(), "/travel")


def test_attack_command_resolves_target():
    intent = _parse_intent("hero", _state(), "/attack bob")
    assert intent == Attack(actor="hero", target="bob")


def test_attack_command_rejects_unknown_target():
    with pytest.raises(_InputError, match="Unknown target"):
        _parse_intent("hero", _state(), "/attack ghost")


def test_check_command_with_dc():
    assert _parse_intent("hero", _state(), "/check dexterity 14 pick the lock") == Check(
        actor="hero", ability="dexterity", difficulty=14, description="pick the lock"
    )


def test_contested_check_command_resolves_opponent():
    assert _parse_intent("hero", _state(), "/check charisma 10 bluff convincingly vs bob") == Check(
        actor="hero", ability="charisma", difficulty=10,
        description="bluff convincingly", opponent="bob",
    )


def test_speak_command_with_resolved_target():
    intent = _parse_intent("hero", _state(), "/speak bob where's the innkeeper?")
    assert intent == Speak(actor="hero", message="where's the innkeeper?", target="bob")


def test_speak_command_without_target():
    intent = _parse_intent("hero", _state(), "/speak anyone listening?")
    assert intent == Speak(actor="hero", message="anyone listening?", target=None)


def test_speak_command_requires_argument():
    with pytest.raises(_InputError, match="Usage: /speak"):
        _parse_intent("hero", _state(), "/speak")


def test_unknown_command_reprompts():
    with pytest.raises(_InputError, match="Unknown command"):
        _parse_intent("hero", _state(), "/dance")


# ============ situation summary ============

def test_describe_situation_includes_key_facts():
    text = _describe_situation("hero", _state())
    assert "tavern" in text
    assert "sword" in text
    assert "bob" in text
    assert "square" in text


def test_describe_situation_frames_the_players_next_choice():
    state = _state()
    state.characters["hero"].goal = "Find who stole the lantern"
    state.quests["missing-light"] = Quest(
        id="missing-light",
        title="The Missing Light",
        description="Recover the tavern's lantern.",
        owner="hero",
        plan=["question the bartender", "search the square"],
    )
    state.history.append(HistoryEvent(
        text='bob whispers: "I saw someone run toward the square."',
        location="tavern",
        characters=["hero", "bob"],
    ))
    state.factions["night"] = Faction(
        id="night",
        name="The Coming Night",
        goal="Let the trail go cold",
        clocks=[ProgressClock(
            id="sunset",
            name="Sunset",
            consequence="The thief escapes with the lantern.",
            segments=4,
            progress=1,
            fail_quest_id="missing-light",
        )],
    )

    text = _describe_situation("hero", state)

    assert "Goal: Find who stole the lantern" in text
    assert "Current objective (The Missing Light): question the bartender" in text
    assert "Deadline (Sunset): 1/4 — The thief escapes with the lantern." in text
    assert 'Just happened: bob whispers: "I saw someone run toward the square."' in text


def test_describe_situation_hides_other_characters_quests_and_unwitnessed_events():
    state = _state()
    state.quests["bobs-business"] = Quest(
        id="bobs-business",
        title="Private Errand",
        description="Do not show this to the hero.",
        owner="bob",
    )
    state.history.append(HistoryEvent(
        text="bob hides a key in the cellar.",
        location="tavern",
        characters=["bob"],
    ))

    text = _describe_situation("hero", state)

    assert "Private Errand" not in text
    assert "bob hides a key" not in text


# ============ console controller loop ============

def test_console_controller_reprompts_on_invalid_input(monkeypatch, capsys):
    lines = iter(["", "/travel nowhere", "/wait"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))

    intent = asyncio.run(console_pc_controller("hero", _state()))

    assert intent == Wait(actor="hero")
    assert "Unknown destination" in capsys.readouterr().out


def test_console_controller_accepts_plain_text(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "grab the lantern")

    intent = asyncio.run(console_pc_controller("hero", _state()))

    assert intent == Action(actor="hero", description="grab the lantern")
