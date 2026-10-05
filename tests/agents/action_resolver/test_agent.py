import asyncio
from types import SimpleNamespace

from pydantic_ai.exceptions import UsageLimitExceeded

from src.agents.character.tools import Action
from src.engine.state.models import Character, Faction, Location, ProgressClock, Quest, WorldState
from src.agents.action_resolver.agent import ActionResolverDeps, remember, resolve
from src.agents.character.tools import Check, Speak, Wait
from src.agents.dm.tools import Modify


def _state() -> WorldState:
    return WorldState(
        locations={"tavern": Location(id="tavern", connections=[])},
        characters={
            "hero": Character(id="hero", role="warrior", location="tavern", goal="find the ale"),
            "merchant": Character(id="merchant", role="merchant", location="tavern"),
        },
    )


def test_remember_applied_deterministically():
    state = _state()
    tool = Wait(actor="hero", remember="the innkeeper is hiding something")
    asyncio.run(resolve(tool, state))
    assert "the innkeeper is hiding something" in state.characters["hero"].knowledge


def test_new_goal_applied_deterministically():
    state = _state()
    tool = Wait(actor="hero", new_goal="expose the innkeeper")
    asyncio.run(resolve(tool, state))
    assert state.characters["hero"].goal == "expose the innkeeper"
    # a private event should be logged, visible only to this character
    event = state.history[-1]
    assert "expose the innkeeper" in event.text
    assert event.characters == ["hero"]


def test_duplicate_new_goal_does_not_create_goal_churn():
    state = _state()

    asyncio.run(resolve(Wait(actor="hero", new_goal="  FIND   the ale  "), state))

    assert state.characters["hero"].goal == "find the ale"
    assert [event.text for event in state.history] == ["hero waits."]


def test_agent_cannot_narrow_active_quest_into_repetitive_search_goal():
    state = _state()
    state.quests["ale"] = Quest(
        id="ale",
        title="The Missing Ale",
        description="Find the missing ale.",
        owner="hero",
    )

    asyncio.run(resolve(Wait(
        actor="hero",
        new_goal="Find clues about the missing ale in the cellar",
    ), state))

    assert state.characters["hero"].goal == "find the ale"
    assert [event.text for event in state.history] == ["hero waits."]


def test_remember_and_new_goal_together_on_any_tool():
    state = _state()
    tool = Speak(actor="hero", message="I've had enough of this place.", remember="the barkeep flinched at the question", new_goal="leave town")
    asyncio.run(resolve(tool, state))
    hero = state.characters["hero"]
    assert "the barkeep flinched at the question" in hero.knowledge
    assert hero.goal == "leave town"


def test_no_self_updates_when_fields_absent():
    state = _state()
    history_before = len(state.history)
    tool = Wait(actor="hero")
    asyncio.run(resolve(tool, state))
    assert state.characters["hero"].goal == "find the ale"  # unchanged
    assert len(state.history) == history_before + 1  # only the wait event itself


def test_injured_character_can_wait_to_catch_their_breath():
    state = _state()
    state.characters["hero"].stats.hp = 3

    result = asyncio.run(resolve(Wait(actor="hero"), state))

    assert state.characters["hero"].stats.hp == 4
    assert result == "hero catches their breath and recovers 1 HP. HP: 4/5."
    assert state.history[-1].text == result


def test_wait_recovery_is_capped_at_max_hp():
    state = _state()
    state.characters["hero"].stats.hp = 4

    result = asyncio.run(resolve(Wait(actor="hero"), state))

    assert state.characters["hero"].stats.hp == 5
    assert "recovers 1 HP" in result


def test_full_health_wait_remains_unchanged():
    state = _state()

    result = asyncio.run(resolve(Wait(actor="hero"), state))

    assert result == "hero waits."
    assert state.characters["hero"].stats.hp == 5
    assert state.history[-1].text == result


def test_wait_cannot_revive_a_dead_character():
    state = _state()
    state.characters["hero"].stats.hp = 0

    result = asyncio.run(resolve(Wait(actor="hero"), state))

    assert result == "Cannot wait — 'hero' is dead."
    assert state.characters["hero"].stats.hp == 0
    assert state.history == []


def test_free_form_action_records_its_outcome_before_optional_self_updates(monkeypatch):
    state = _state()

    async def resolved_action(*_args, **_kwargs):
        return SimpleNamespace(output="A misaligned portrait reveals a hidden compartment.")

    monkeypatch.setattr("src.agents.action_resolver.agent.agent.run", resolved_action)

    result = asyncio.run(resolve(Action(
        actor="hero",
        description="inspect the portraits",
        new_goal="open the hidden compartment",
    ), state))

    assert result == "A misaligned portrait reveals a hidden compartment."
    assert [event.text for event in state.history] == [
        "A misaligned portrait reveals a hidden compartment.",
        "hero's goal is now: open the hidden compartment",
    ]


def test_action_resolver_keeps_only_one_decisive_knowledge_fact_per_action():
    state = _state()
    ctx = SimpleNamespace(deps=ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="read the genealogy",
    ))

    first = remember(ctx, "The circlet is hidden behind the chapel altar.")
    second = remember(ctx, "The altar has a secret compartment containing the circlet.")

    assert "learns" in first
    assert "Call done now" in second
    assert state.characters["hero"].knowledge == ["The circlet is hidden behind the chapel altar."]


def test_free_form_action_usage_limit_becomes_visible_outcome_instead_of_crash(monkeypatch):
    state = _state()

    async def exhausted(*_args, **_kwargs):
        raise UsageLimitExceeded("request limit reached")

    monkeypatch.setattr("src.agents.action_resolver.agent.agent.run", exhausted)

    result = asyncio.run(resolve(Action(actor="hero", description="search forever"), state))

    assert result == "hero makes no further progress on that action."
    assert state.history[-1].text == result


def test_modify_update_relationship_applied():
    state = _state()
    tool = Modify(action="update_relationship", target_id="hero", other_id="merchant", reason="grateful — she healed me")
    result = asyncio.run(resolve(tool, state))
    assert state.characters["hero"].relationships["merchant"] == "grateful — she healed me"
    assert "grateful" in result


def test_modify_update_relationship_unknown_target():
    state = _state()
    tool = Modify(action="update_relationship", target_id="ghost", other_id="merchant", reason="hostile")
    result = asyncio.run(resolve(tool, state))
    assert "Cannot update relationship" in result
    assert "ghost" not in state.characters


def test_modify_update_relationship_missing_fields():
    state = _state()
    tool = Modify(action="update_relationship", target_id="hero", other_id=None, reason=None)
    result = asyncio.run(resolve(tool, state))
    assert "Cannot update relationship" in result
    assert state.characters["hero"].relationships == {}


def test_modify_advance_faction_clock_applied():
    state = _state()
    state.factions["crew"] = Faction(
        id="crew",
        name="The Crew",
        goal="move the goods",
        clocks=[ProgressClock(id="haul", name="Haul the goods", segments=2, consequence="The goods are gone.")],
    )
    tool = Modify(action="advance_faction_clock", target_id="crew", other_id="haul")
    result = asyncio.run(resolve(tool, state))
    assert state.factions["crew"].clocks[0].progress == 1
    assert "1/2" in result


def test_modify_advance_faction_clock_missing_clock_id():
    state = _state()
    tool = Modify(action="advance_faction_clock", target_id="crew", other_id=None)
    result = asyncio.run(resolve(tool, state))
    assert "Cannot advance faction clock" in result


class _Rolls:
    def __init__(self, *values: int):
        self.values = iter(values)

    def randint(self, _low: int, _high: int) -> int:
        return next(self.values)


def test_check_resolution_records_concise_roll_history():
    state = _state()
    result = asyncio.run(resolve(
        Check(actor="hero", ability="dexterity", description="picks the cellar lock", difficulty=14, modifier=2),
        state,
        rng=_Rolls(12),
    ))
    assert result == "hero succeeds: picks the cellar lock [dexterity; 12+2=14 vs DC 14]."
    assert state.history[-1].text == result
    assert state.history[-1].characters == ["hero"]


def test_contested_check_records_both_characters():
    state = _state()
    result = asyncio.run(resolve(Check(
        actor="hero", ability="charisma", description="bluffs the merchant",
        opponent="merchant", modifier=1, opposing_modifier=2,
    ), state, rng=_Rolls(16, 10)))
    assert "16+1=17 vs merchant 10+2=12" in result
    assert state.history[-1].characters == ["hero", "merchant"]
