import asyncio
import time
from inspect import signature
from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from src.agents.character.tools import Action
from src.engine.state import WorldOperations
from src.engine.state.models import Character, Faction, Location, ProgressClock, Quest, WorldState
from src.agents.action_resolver.agent import (
    ActionResolverDeps,
    add_detail,
    agent,
    adjust_hp,
    change_item,
    create_item,
    create_npc,
    discover_exit,
    drop,
    remember,
    resolve,
    take,
)
from src.agents.character.tools import Attack, Check, Speak, Travel, Wait
from src.agents.dm.tools import Create, Modify


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


def test_travel_does_not_store_paraphrase_of_recent_knowledge():
    state = _state()
    state.locations["market-square"] = Location(id="market-square")
    state.locations["tavern"].connections = ["market-square"]
    hero = state.characters["hero"]
    original = (
        "Kaelen wore a guard-issue cloak the night he vanished—the fibers caught on the "
        "lantern post confirm he was dragged along the bridge railing."
    )
    hero.knowledge = [original]

    asyncio.run(resolve(Travel(
        actor="hero",
        destination="market-square",
        remember=(
            "Kaelen was dragged along valley-bridge—guard cloak fibers caught on the lantern "
            "post confirm violence, not voluntary movement."
        ),
    ), state))

    assert hero.knowledge == [original]
    assert [event.text for event in state.history] == ["hero moved to 'market-square'."]


def test_travel_keeps_related_but_distinct_new_knowledge():
    state = _state()
    state.locations["market-square"] = Location(id="market-square")
    state.locations["tavern"].connections = ["market-square"]
    hero = state.characters["hero"]
    hero.knowledge = [
        "Kaelen wore a guard-issue cloak when he vanished near valley-bridge.",
    ]
    new_fact = "Alan paid for an unconscious passenger to be transported downriver."

    asyncio.run(resolve(Travel(
        actor="hero",
        destination="market-square",
        remember=new_fact,
    ), state))

    assert hero.knowledge[-1] == new_fact


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


def test_supporting_npc_cannot_replace_goal_with_active_quest_step():
    state = _state()
    state.characters["merchant"].goal = (
        "Help young Hero recover the Oak Circlet while keeping old house secrets."
    )
    state.quests["circlet"] = Quest(
        id="circlet",
        title="The Oak Circlet",
        description="Recover the family Circlet before dawn.",
        owner="hero",
        plan=["search the upper-library reading table for the Circlet"],
    )

    asyncio.run(resolve(Speak(
        actor="merchant",
        target="hero",
        message="We should search the reading table before the trail grows cold.",
        new_goal="Search the upper-library reading table for the Circlet",
    ), state))

    assert state.characters["merchant"].goal == (
        "Help young Hero recover the Oak Circlet while keeping old house secrets."
    )
    assert [event.text for event in state.history] == [
        'merchant says to hero: "We should search the reading table before the trail grows cold."',
    ]


def test_supporting_npc_can_make_major_goal_change_about_active_quest():
    state = _state()
    state.characters["merchant"].goal = "Help Hero recover the Oak Circlet."
    state.quests["circlet"] = Quest(
        id="circlet",
        title="The Oak Circlet",
        owner="hero",
        plan=["search the upper-library reading table for the Circlet"],
    )

    asyncio.run(resolve(Speak(
        actor="merchant",
        target="hero",
        message="The Circlet belongs to me now.",
        new_goal="Betray Hero and steal the Oak Circlet for myself",
    ), state))

    assert state.characters["merchant"].goal == "Betray Hero and steal the Oak Circlet for myself"


def test_remember_and_new_goal_together_on_any_tool():
    state = _state()
    tool = Speak(actor="hero", message="I've had enough of this place.", remember="the barkeep flinched at the question", new_goal="leave town")
    asyncio.run(resolve(tool, state))
    hero = state.characters["hero"]
    assert "the barkeep flinched at the question" in hero.knowledge
    assert hero.goal == "leave town"


def test_current_action_note_is_not_stored_as_durable_knowledge():
    state = _state()

    asyncio.run(resolve(Speak(
        actor="hero",
        target="merchant",
        message="Did you see the missing guard near the market?",
        remember=(
            "Asking Merchant about the guard's last movements near the market—"
            "searching for witnesses to the disappearance."
        ),
    ), state))

    assert state.characters["hero"].knowledge == []
    assert [event.text for event in state.history] == [
        'hero says to merchant: "Did you see the missing guard near the market?"',
    ]


def test_subject_elided_completed_action_note_is_not_stored_as_durable_knowledge():
    state = _state()

    asyncio.run(resolve(Speak(
        actor="hero",
        target="merchant",
        message="I found your name in a smugglers' ledger. What are you buying?",
        remember=(
            "Confronted Merchant directly with the ledger evidence to gauge his "
            "response and involvement."
        ),
    ), state))

    assert state.characters["hero"].knowledge == []
    assert [event.text for event in state.history] == [
        'hero says to merchant: "I found your name in a smugglers\' ledger. What are you buying?"',
    ]


def test_subject_elided_action_with_discovery_is_stored_as_durable_knowledge():
    state = _state()
    fact = "Confronted Merchant and discovered the ledger code matches the Teal Moon."

    asyncio.run(resolve(Wait(actor="hero", remember=fact), state))

    assert state.characters["hero"].knowledge == [fact]


def test_embedded_confrontation_note_is_not_stored_as_durable_knowledge():
    state = _state()
    state.locations["market-square"] = Location(id="market-square")
    state.locations["tavern"].connections = ["market-square"]

    asyncio.run(resolve(Travel(
        actor="hero",
        destination="market-square",
        remember=(
            "Alan conducted an unscheduled transaction the day Kaelen vanished—"
            "confronting him directly about what happened."
        ),
    ), state))

    assert state.characters["hero"].knowledge == []
    assert [event.text for event in state.history] == ["hero moved to 'market-square'."]


def test_embedded_current_action_note_is_not_stored_as_durable_knowledge():
    state = _state()

    asyncio.run(resolve(Speak(
        actor="hero",
        target="merchant",
        message="Did you see Kaelen after his patrol near the river?",
        remember=(
            "Ronny-spice is a merchant in market-square; asking about Kaelen's "
            "disappearance and the torn fabric found at valley-bridge railing."
        ),
    ), state))

    assert state.characters["hero"].knowledge == []
    assert [event.text for event in state.history] == [
        'hero says to merchant: "Did you see Kaelen after his patrol near the river?"',
    ]


def test_embedded_completed_action_fact_is_stored_as_durable_knowledge():
    state = _state()
    fact = (
        "Ronny-spice is a merchant in market-square; asking about Kaelen revealed "
        "that Alan paid for an unconscious passenger's river passage."
    )

    asyncio.run(resolve(Wait(actor="hero", remember=fact), state))

    assert state.characters["hero"].knowledge == [fact]


def test_embedded_plan_is_not_stored_as_durable_knowledge():
    memories = [
        (
            "I have proof from the crate and ship's log. Now I need to get Old Keph "
            "and put this in front of the village."
        ),
        (
            "Old Keph is the village elder and knows every ship that passed the cove — "
            "he's my next step to confirm the Teal Moon connection."
        ),
        (
            "The Oak Circlet is missing from Thorne Manor — need to search for clues "
            "about where it's hidden or who took it."
        ),
        (
            "Crew manifest names and ship assignments from the cove—I need Keph to "
            "identify the black-hulled sloop's crew against these records."
        ),
        (
            "Marsh Venn captains the black-hulled sloop. I have his ship's log and crew "
            "manifest proving his smuggling operation. Time to report to Old Keph."
        ),
        (
            "The Teal Moon's log proves the crew connection to the crates—I have enough "
            "proof now to report to Old Keph and expose them before they move again."
        ),
        (
            "Blue fabric scrap confirms Kaelen was at valley-bridge; now heading to "
            "market-square to ask around about his last known movements."
        ),
        (
            "The harbormaster's notice confirms the black-hulled sloop and "
            "compass-rose-and-crescent sigil match what I found. Need to know who's "
            "running this operation and when they're coming back for that crate."
        ),
        (
            "Grun (porter) arrived 8 months ago—newest staff member. Tamín "
            "(stablemaster) joined 5 years ago. Should question them about the "
            "Circlet's whereabouts."
        ),
        (
            "Finnian warned me about Calla asking questions regarding a vault behind "
            "the cold hearth — this is worth investigating directly"
        ),
        (
            "Kaelen was heading to meet Dockmaster Alan at the trade-dock three weeks "
            "ago—this is my next lead to follow."
        ),
        (
            "The black-hulled sloop is my next target—that ship connects to the V.M. "
            "merchant mark and the smuggling operation."
        ),
        (
            "Someone is in the master bedroom right now moving a heavy chest—this is "
            "my best lead on the circlet's location."
        ),
        (
            "Ronny Spice is asking the docks crew about Kaelen—check back with her "
            "after visiting the guild-hall ledger."
        ),
    ]

    for memory in memories:
        state = _state()
        asyncio.run(resolve(Wait(actor="hero", remember=memory), state))

        assert state.characters["hero"].knowledge == []
        assert [event.text for event in state.history] == ["hero waits."]


def test_known_character_follow_up_plan_is_not_stored_as_durable_knowledge():
    state = _state()
    state.characters["old-keph"] = Character(
        id="old-keph",
        role="village elder",
        location="tavern",
    )
    fact_and_plan = (
        "I have the Teal Moon ship's log, crew manifest, and trading route map—solid "
        "proof of Captain V.K. and the black-hulled sloop's smuggling operation. "
        "Old Keph needs to see this before the crew strikes."
    )

    asyncio.run(resolve(Wait(actor="hero", remember=fact_and_plan), state))

    assert state.characters["hero"].knowledge == []
    assert [event.text for event in state.history] == ["hero waits."]


def test_known_character_pending_action_is_not_stored_as_durable_knowledge():
    state = _state()
    state.characters["elara-swift"] = Character(
        id="elara-swift",
        role="former guard",
        location="tavern",
    )
    pending_action = (
        "Elara is heading to find Alan at the trade-dock to ask about Kaelen's disappearance"
    )

    asyncio.run(resolve(Wait(actor="hero", remember=pending_action), state))

    assert state.characters["hero"].knowledge == []
    assert [event.text for event in state.history] == ["hero waits."]


@pytest.mark.parametrize(
    "fact",
    [
        "The smugglers need to unload the sloop before dawn.",
        "The smugglers had enough time to unload the sloop before dawn.",
        "The smugglers have enough proof now to accuse Old Keph.",
        "The staff handbook says servants should lock the manor doors after midnight.",
        "The stolen circlet is worth 500 gold to the right collector.",
        "Elara headed to the trade-dock and found Alan there.",
        "The current tide exposes a path to the sea cave.",
        "The black-hulled sloop was my next target before it sailed.",
        "The master bedroom was my best lead until Greta found the hidden vault.",
        "Ronny asked Elara to check back after visiting the guild-hall ledger.",
    ],
)
def test_external_requirement_is_still_stored_as_durable_knowledge(fact):
    state = _state()

    asyncio.run(resolve(Wait(actor="hero", remember=fact), state))

    assert state.characters["hero"].knowledge == [fact]


def test_goal_note_is_not_stored_as_durable_knowledge():
    state = _state()
    state.locations["forest"] = Location(id="forest")
    state.locations["tavern"].connections = ["forest"]

    asyncio.run(resolve(Travel(
        actor="hero",
        destination="forest",
        remember="Goal: search the forest for clues about the missing guard.",
    ), state))

    assert state.characters["hero"].knowledge == []
    assert [event.text for event in state.history] == ["hero moved to 'forest'."]


@pytest.mark.parametrize(
    "memory",
    [
        "Current objective: search the upper-library for clues about the Oak Circlet's whereabouts",
        "My current objective is to search the upper-library for clues about the Oak Circlet's whereabouts.",
    ],
)
def test_current_objective_note_is_not_stored_as_durable_knowledge(memory):
    state = _state()
    state.locations["upper-library"] = Location(id="upper-library")
    state.locations["tavern"].connections = ["upper-library"]

    asyncio.run(resolve(Travel(
        actor="hero",
        destination="upper-library",
        remember=memory,
    ), state))

    assert state.characters["hero"].knowledge == []
    assert [event.text for event in state.history] == ["hero moved to 'upper-library'."]


def test_past_fact_is_still_stored_when_the_character_takes_a_new_action():
    state = _state()
    state.locations["forest"] = Location(id="forest")
    state.locations["tavern"].connections = ["forest"]

    asyncio.run(resolve(Travel(
        actor="hero",
        destination="forest",
        remember="Searching the riverbank revealed torn cloth from the missing guard.",
    ), state))

    assert state.characters["hero"].knowledge == [
        "Searching the riverbank revealed torn cloth from the missing guard.",
    ]


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


def test_free_form_action_reports_actual_mutations_instead_of_contradictory_prose(monkeypatch):
    state = _state()

    async def resolved_action(*_args, **kwargs):
        ctx = SimpleNamespace(deps=kwargs["deps"])
        discover_exit(
            ctx,
            name="wine cellar",
            description="A cool stone cellar beneath the tavern.",
        )
        return SimpleNamespace(output="Descended into the wine cellar and searched it.")

    monkeypatch.setattr("src.agents.action_resolver.agent.agent.run", resolved_action)

    result = asyncio.run(resolve(Action(
        actor="hero",
        description="head down to the wine cellar and search it",
    ), state))

    assert result == "Location 'wine-cellar' added."
    assert state.characters["hero"].location == "tavern"
    assert state.locations["tavern"].connections == ["wine-cellar"]
    assert state.history[-1].text == result


def test_free_form_action_applies_dependent_tool_calls_in_model_order(monkeypatch):
    state = _state()
    calls = 0

    def model(_messages, _info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[
                ToolCallPart(
                    "discover_exit",
                    {"name": "river overlook", "description": "A bluff over the river."},
                    tool_call_id="discover",
                ),
                ToolCallPart(
                    "add_detail",
                    {"location": "river-overlook", "detail": "Drag marks end at the edge."},
                    tool_call_id="detail",
                ),
            ])
        return ModelResponse(parts=[
            ToolCallPart("done", {"response": "The tracks end at the river."}, tool_call_id="done"),
        ])

    original_add_location = WorldOperations.add_location

    def slow_add_location(self, *args, **kwargs):
        # Make the former parallel execution race deterministic: add_detail used
        # to run before this mutation completed and report the location missing.
        time.sleep(0.05)
        return original_add_location(self, *args, **kwargs)

    monkeypatch.setattr(WorldOperations, "add_location", slow_add_location)

    with agent.override(model=FunctionModel(model)):
        result = asyncio.run(resolve(Action(
            actor="hero",
            description="follow the tracks to the river and inspect where they end",
        ), state))

    assert result == "Location 'river-overlook' added. Location 'river-overlook' updated."
    assert state.locations["river-overlook"].features == ["Drag marks end at the edge."]
    assert [event.text for event in state.history] == [result]


def test_free_form_action_applies_mutations_sent_with_terminal_output():
    state = _state()

    def model(_messages, _info):
        return ModelResponse(parts=[
            ToolCallPart(
                "remember",
                {"knowledge": "Old Keph conducts business at lighthouse-base."},
                tool_call_id="remember",
            ),
            ToolCallPart(
                "add_detail",
                {"detail": "The notice post lists Old Keph at lighthouse-base."},
                tool_call_id="detail",
            ),
            ToolCallPart(
                "done",
                {"response": "The notice post confirms where Old Keph can be found."},
                tool_call_id="done",
            ),
        ])

    with agent.override(model=FunctionModel(model)):
        result = asyncio.run(resolve(Action(
            actor="hero",
            description="search the notice post for Old Keph's whereabouts",
        ), state))

    assert state.characters["hero"].knowledge == [
        "Old Keph conducts business at lighthouse-base.",
    ]
    assert state.locations["tavern"].features == [
        "The notice post lists Old Keph at lighthouse-base.",
    ]
    assert result == (
        "hero learns: Old Keph conducts business at lighthouse-base. "
        "Location 'tavern' updated."
    )
    assert [event.text for event in state.history] == [
        "hero learns: Old Keph conducts business at lighthouse-base.",
    ]


def test_free_form_action_cannot_interact_with_known_remote_location():
    state = _state()
    state.locations["upper-library"] = Location(id="upper-library")
    state.locations["tavern"].connections.append("upper-library")
    before = state.model_dump()

    result = asyncio.run(resolve(Action(
        actor="hero",
        description=(
            "Search the upper-library for clues and examine the reading table "
            "where the circlet was last seen."
        ),
    ), state))

    assert result == (
        "Cannot resolve action at 'upper-library' — 'hero' is at 'tavern'. "
        "Travel there first."
    )
    assert state.model_dump() == before


def test_free_form_action_cannot_mutate_quest_state_directly():
    state = _state()
    state.quests["ale"] = Quest(
        id="ale",
        title="The Missing Ale",
        description="Find and return the missing ale.",
        owner="hero",
        plan=["find the missing ale", "return the missing ale"],
    )

    assert "update_quest" not in agent._function_toolset.tools
    assert state.quests["ale"].status == "active"
    assert state.quests["ale"].current_step == 0


def test_free_form_action_keeps_resolver_memory_over_pre_resolution_memory(monkeypatch):
    state = _state()

    async def resolved_action(*_args, **kwargs):
        deps = kwargs["deps"]
        WorldOperations(deps.state).add_knowledge(
            deps.char.id,
            "The innkeeper left for the docks an hour ago.",
        )
        return SimpleNamespace(output="The innkeeper already left for the docks.")

    monkeypatch.setattr("src.agents.action_resolver.agent.agent.run", resolved_action)

    asyncio.run(resolve(Action(
        actor="hero",
        description="ask where the innkeeper went",
        remember="The innkeeper is waiting in the tavern.",
    ), state))

    assert state.characters["hero"].knowledge == [
        "The innkeeper left for the docks an hour ago.",
    ]


def test_free_form_action_uses_character_memory_when_resolver_records_none(monkeypatch):
    state = _state()

    async def resolved_action(*_args, **_kwargs):
        return SimpleNamespace(output="The loose brick bears the thieves' mark.")

    monkeypatch.setattr("src.agents.action_resolver.agent.agent.run", resolved_action)

    asyncio.run(resolve(Action(
        actor="hero",
        description="inspect the loose brick",
        remember="The loose brick bears the thieves' mark.",
    ), state))

    assert state.characters["hero"].knowledge == [
        "The loose brick bears the thieves' mark.",
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


def test_action_resolver_does_not_count_blank_knowledge_as_an_effect():
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="inspect the empty desk",
    )
    ctx = SimpleNamespace(deps=deps)
    history_before = list(state.history)

    result = remember(ctx, "  \t\n  ")

    assert result == "Cannot add knowledge — fact cannot be blank."
    assert state.characters["hero"].knowledge == []
    assert state.history == history_before
    assert deps.effects == []
    assert deps.remembered_this_action is False


def test_action_resolver_rejects_transient_self_action_memory_then_accepts_discovery():
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description=(
            "Move between the merchant and the tavern exit to block their escape "
            "while keeping them in conversation."
        ),
    )
    ctx = SimpleNamespace(deps=deps)

    transient = remember(
        ctx,
        (
            "Hero positioned himself between the merchant and the tavern exit to "
            "prevent their escape while questioning them."
        ),
    )
    corrected = remember(ctx, "The merchant admitted the stolen circlet is hidden at the mill.")

    assert transient == (
        "Cannot remember a transient action as durable knowledge. Record only a concrete "
        "discovery, then call done."
    )
    assert "learns" in corrected
    assert state.characters["hero"].knowledge == [
        "The merchant admitted the stolen circlet is hidden at the mill.",
    ]
    assert deps.remembered_this_action is True


def test_action_resolver_rejects_subject_elided_completed_action_memory():
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="confront the merchant with the seized ledger",
    )
    ctx = SimpleNamespace(deps=deps)

    result = remember(
        ctx,
        "Confronted Merchant directly with the ledger evidence to gauge his response.",
    )

    assert result == (
        "Cannot remember a transient action as durable knowledge. Record only a concrete "
        "discovery, then call done."
    )
    assert state.characters["hero"].knowledge == []
    assert deps.effects == []
    assert deps.remembered_this_action is False


def test_action_resolver_rejects_known_character_follow_up_plan():
    state = _state()
    state.characters["old-keph"] = Character(
        id="old-keph",
        role="village elder",
        location="tavern",
    )
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="review the seized records and decide who should receive them",
    )
    ctx = SimpleNamespace(deps=deps)

    result = remember(
        ctx,
        (
            "The seized records prove Captain V.K. runs the smuggling operation. "
            "Old Keph needs to see this before the crew strikes."
        ),
    )

    assert result == (
        "Cannot remember a pending plan as durable knowledge. Record only the concrete "
        "fact, then call done."
    )
    assert state.characters["hero"].knowledge == []
    assert deps.effects == []
    assert deps.remembered_this_action is False


def test_action_resolver_rejects_known_character_pending_action():
    state = _state()
    state.characters["elara-swift"] = Character(
        id="elara-swift",
        role="former guard",
        location="tavern",
    )
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="listen while Elara explains whom she plans to question next",
    )
    ctx = SimpleNamespace(deps=deps)

    result = remember(
        ctx,
        "Elara is heading to find Alan at the trade-dock to ask about Kaelen's disappearance",
    )

    assert result == (
        "Cannot remember a pending plan as durable knowledge. Record only the concrete "
        "fact, then call done."
    )
    assert state.characters["hero"].knowledge == []
    assert deps.effects == []
    assert deps.remembered_this_action is False


def test_action_resolver_rejects_subject_elided_follow_up_plan():
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="review which servants may know where the circlet went",
    )
    ctx = SimpleNamespace(deps=deps)

    result = remember(
        ctx,
        (
            "Grun (porter) arrived 8 months ago—newest staff member. Tamín "
            "(stablemaster) joined 5 years ago. Should question them about the "
            "Circlet's whereabouts."
        ),
    )

    assert result == (
        "Cannot remember a pending plan as durable knowledge. Record only the concrete "
        "fact, then call done."
    )
    assert state.characters["hero"].knowledge == []
    assert deps.effects == []
    assert deps.remembered_this_action is False


def test_action_resolver_rejects_deictic_worth_follow_up_plan():
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="follow Finnian's lead about Calla and the vault",
    )
    ctx = SimpleNamespace(deps=deps)

    result = remember(
        ctx,
        (
            "Finnian warned me about Calla asking questions regarding a vault behind "
            "the cold hearth — this is worth investigating directly"
        ),
    )

    assert result == (
        "Cannot remember a pending plan as durable knowledge. Record only the concrete "
        "fact, then call done."
    )
    assert state.characters["hero"].knowledge == []
    assert deps.effects == []
    assert deps.remembered_this_action is False


def test_action_resolver_rejects_deictic_next_lead_follow_up_plan():
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="review Kaelen's planned meeting with Dockmaster Alan",
    )
    ctx = SimpleNamespace(deps=deps)

    result = remember(
        ctx,
        (
            "Kaelen was heading to meet Dockmaster Alan at the trade-dock three weeks "
            "ago—this is my next lead to follow."
        ),
    )

    assert result == (
        "Cannot remember a pending plan as durable knowledge. Record only the concrete "
        "fact, then call done."
    )
    assert state.characters["hero"].knowledge == []
    assert deps.effects == []
    assert deps.remembered_this_action is False


@pytest.mark.parametrize(
    "knowledge",
    [
        (
            "The black-hulled sloop is my next target—that ship connects to the V.M. "
            "merchant mark and the smuggling operation."
        ),
        (
            "Someone is in the master bedroom right now moving a heavy chest—this is "
            "my best lead on the circlet's location."
        ),
    ],
)
def test_action_resolver_rejects_possessive_priority_plan(knowledge):
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="review the merchant mark's connection to the black-hulled sloop",
    )
    ctx = SimpleNamespace(deps=deps)

    result = remember(ctx, knowledge)

    assert result == (
        "Cannot remember a pending plan as durable knowledge. Record only the concrete "
        "fact, then call done."
    )
    assert state.characters["hero"].knowledge == []
    assert deps.effects == []
    assert deps.remembered_this_action is False


def test_action_resolver_rejects_embedded_check_back_instruction():
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="review Ronny's lead after searching the guild ledger",
    )
    ctx = SimpleNamespace(deps=deps)
    knowledge = (
        "Ronny Spice is asking the docks crew about Kaelen—check back with her "
        "after visiting the guild-hall ledger."
    )

    result = remember(ctx, knowledge)

    assert result == (
        "Cannot remember a pending plan as durable knowledge. Record only the concrete "
        "fact, then call done."
    )
    assert state.characters["hero"].knowledge == []
    assert deps.effects == []
    assert deps.remembered_this_action is False


def test_action_resolver_rejects_possessive_current_objective_note():
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="search the upper-library for clues about the Oak Circlet",
    )
    ctx = SimpleNamespace(deps=deps)

    result = remember(
        ctx,
        "My current objective is to search the upper-library for clues about the Oak Circlet's whereabouts.",
    )

    assert result == (
        "Cannot remember a pending plan as durable knowledge. Record only the concrete "
        "fact, then call done."
    )
    assert state.characters["hero"].knowledge == []
    assert deps.effects == []
    assert deps.remembered_this_action is False


def test_character_tool_does_not_store_transient_self_action_note():
    state = _state()

    asyncio.run(resolve(Wait(
        actor="hero",
        remember="Hero stood watch beside the tavern exit while waiting for the merchant.",
    ), state))

    assert state.characters["hero"].knowledge == []
    assert [event.text for event in state.history] == ["hero waits."]


def test_action_resolver_keeps_self_attributed_discovery_memory():
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="search the merchant's desk for evidence",
    )
    ctx = SimpleNamespace(deps=deps)

    result = remember(ctx, "Hero found the smuggler's manifest beneath the merchant's desk.")

    assert "learns" in result
    assert state.characters["hero"].knowledge == [
        "Hero found the smuggler's manifest beneath the merchant's desk.",
    ]


@pytest.mark.parametrize(
    ("first_fact", "character_id", "expected_failure"),
    [
        (
            "The circlet is hidden behind the chapel altar.",
            None,
            "already knows that",
        ),
        (
            "The chapel sexton hid the circlet behind the altar.",
            "missing-witness",
            "character 'missing-witness' not found",
        ),
    ],
)
def test_action_resolver_allows_corrected_memory_after_failed_attempt(
    first_fact,
    character_id,
    expected_failure,
):
    state = _state()
    state.characters["hero"].knowledge = [
        "The circlet is hidden behind the chapel altar.",
    ]
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="question the chapel sexton",
    )
    ctx = SimpleNamespace(deps=deps)

    first = remember(ctx, first_fact, character_id=character_id)
    corrected = remember(
        ctx,
        "The chapel sexton saw Lord Harwin enter the east wing after dinner.",
    )

    assert expected_failure in first
    assert "learns" in corrected
    assert deps.remembered_this_action is True
    assert state.characters["hero"].knowledge == [
        "The circlet is hidden behind the chapel altar.",
        "The chapel sexton saw Lord Harwin enter the east wing after dinner.",
    ]


def test_action_resolver_rejects_whereabouts_that_conflict_with_canonical_state():
    state = _state()
    state.locations.update({
        "lighthouse-base": Location(id="lighthouse-base"),
        "village-square": Location(id="village-square"),
    })
    state.characters["old-keph"] = Character(
        id="old-keph",
        role="village elder",
        location="lighthouse-base",
    )
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="ask where Old Keph is right now",
    )
    ctx = SimpleNamespace(deps=deps)

    result = remember(
        ctx,
        "Old-Keph was spotted heading toward the village-square a bell ago, likely still nearby.",
    )

    assert result == (
        "Cannot remember that old-keph is at 'village-square' — canonical state places them "
        "at 'lighthouse-base'. Use create_npc first if this action directly established their "
        "new location."
    )
    assert state.characters["hero"].knowledge == []
    assert deps.effects == []
    assert deps.remembered_this_action is False


def test_action_resolver_accepts_whereabouts_after_canonical_location_update():
    state = _state()
    state.locations["village-square"] = Location(id="village-square")
    state.characters["old-keph"] = Character(
        id="old-keph",
        role="village elder",
        location="tavern",
    )
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="find Old Keph in the village square",
    )
    ctx = SimpleNamespace(deps=deps)

    move_result = create_npc(ctx, "Old Keph", location="village-square")
    remember_result = remember(ctx, "Old Keph is at village-square.")

    assert move_result == "old-keph appears at 'village-square'."
    assert "learns" in remember_result
    assert state.characters["old-keph"].location == "village-square"
    assert state.characters["hero"].knowledge == ["Old Keph is at village-square."]


def test_character_tool_does_not_upgrade_workplace_to_current_whereabouts():
    state = _state()
    state.locations.update({
        "guild-hall": Location(id="guild-hall"),
        "trade-dock": Location(id="trade-dock"),
    })
    state.locations["tavern"].connections = ["guild-hall"]
    WorldOperations(state).speak(
        "merchant",
        "Alan works the trade-dock—you'll find him down by the water.",
        "hero",
    )

    asyncio.run(resolve(Travel(
        actor="hero",
        destination="guild-hall",
        remember=(
            "Alan the dockmaster is at the trade-dock; the merchant warned me he knows "
            "more than he lets on."
        ),
    ), state))

    assert state.characters["hero"].knowledge == []
    assert [event.text for event in state.history] == [
        'merchant says to hero: "Alan works the trade-dock—you\'ll find him down by the water."',
        "hero moved to 'guild-hall'.",
    ]


def test_character_tool_keeps_directly_reported_current_whereabouts():
    state = _state()
    state.locations.update({
        "guild-hall": Location(id="guild-hall"),
        "trade-dock": Location(id="trade-dock"),
    })
    state.locations["tavern"].connections = ["guild-hall"]
    WorldOperations(state).speak(
        "merchant",
        "Alan is waiting at the trade-dock for the harbor bell.",
        "hero",
    )

    fact = "Alan the dockmaster is at the trade-dock waiting for the harbor bell."
    asyncio.run(resolve(Travel(
        actor="hero",
        destination="guild-hall",
        remember=fact,
    ), state))

    assert state.characters["hero"].knowledge == [fact]


def test_character_tool_rejects_whereabouts_that_conflict_with_canonical_state():
    state = _state()
    state.locations.update({
        "guild-hall": Location(id="guild-hall"),
        "trade-dock": Location(id="trade-dock"),
    })
    state.locations["tavern"].connections = ["guild-hall"]
    state.characters["dockmaster-alan"] = Character(
        id="dockmaster-alan",
        role="dockmaster",
        location="trade-dock",
    )

    asyncio.run(resolve(Travel(
        actor="hero",
        destination="guild-hall",
        remember="Dockmaster Alan is at guild-hall.",
    ), state))

    assert state.characters["hero"].knowledge == []
    assert state.characters["dockmaster-alan"].location == "trade-dock"


def test_action_resolver_allows_known_npc_to_report_about_another_location():
    state = _state()
    state.locations["village-square"] = Location(id="village-square")
    state.characters["old-keph"] = Character(
        id="old-keph",
        role="village elder",
        location="tavern",
    )
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="ask Old Keph what he knows",
    )
    ctx = SimpleNamespace(deps=deps)

    result = remember(ctx, "Old Keph says the smugglers meet at village-square.")

    assert "learns" in result
    assert state.characters["hero"].knowledge == [
        "Old Keph says the smugglers meet at village-square.",
    ]


def test_action_resolver_does_not_record_zero_effect_hp_adjustment():
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="tend wounds that are already healed",
    )
    ctx = SimpleNamespace(deps=deps)

    result = adjust_hp(ctx, 3)

    assert "heals 0 HP" in result
    assert deps.effects == []
    assert state.history == []


def test_action_resolver_rejects_remote_hp_adjustment():
    state = _state()
    state.locations["harbor-dock"] = Location(id="harbor-dock")
    merchant = state.characters["merchant"]
    merchant.location = "harbor-dock"
    merchant.stats.hp = 1
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="finish the fleeing merchant",
    )
    ctx = SimpleNamespace(deps=deps)

    result = adjust_hp(ctx, -1, character_id="merchant")

    assert result == "Cannot adjust HP — 'hero' and 'merchant' are not in the same location."
    assert merchant.stats.hp == 1
    assert deps.effects == []
    assert state.history == []


def test_action_resolver_allows_colocated_hp_adjustment():
    state = _state()
    merchant = state.characters["merchant"]
    merchant.stats.hp = 3
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="bandage the merchant",
    )
    ctx = SimpleNamespace(deps=deps)

    result = adjust_hp(ctx, 2, character_id="merchant")

    assert result == "merchant heals 2 HP. HP: 5/5."
    assert merchant.stats.hp == 5
    assert deps.effects == [result]
    assert state.history[-1].text == result


def test_action_resolver_cannot_revive_a_dead_character():
    state = _state()
    merchant = state.characters["merchant"]
    merchant.stats.hp = 0
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="bandage the fallen merchant",
    )
    ctx = SimpleNamespace(deps=deps)

    result = adjust_hp(ctx, 2, character_id="merchant")

    assert result == "Cannot heal — 'merchant' is dead."
    assert merchant.stats.hp == 0
    assert deps.effects == []
    assert state.history == []


def test_action_resolver_rejects_remote_inventory_mutations():
    state = _state()
    state.locations["harbor-dock"] = Location(
        id="harbor-dock",
        items=["sealed ledger"],
    )
    merchant = state.characters["merchant"]
    merchant.location = "harbor-dock"
    merchant.inventory = ["brass key"]
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="make the distant merchant exchange evidence",
    )
    ctx = SimpleNamespace(deps=deps)

    take_result = take(ctx, "sealed ledger", character_id="merchant")
    drop_result = drop(ctx, "brass key", character_id="merchant")

    assert take_result == "Cannot take item — 'hero' and 'merchant' are not in the same location."
    assert drop_result == "Cannot drop item — 'hero' and 'merchant' are not in the same location."
    assert merchant.inventory == ["brass key"]
    assert state.locations["harbor-dock"].items == ["sealed ledger"]
    assert deps.effects == []
    assert state.history == []


def test_action_resolver_allows_colocated_character_to_take_item():
    state = _state()
    state.locations["tavern"].items = ["sealed ledger"]
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="hand the ledger to the merchant",
    )
    ctx = SimpleNamespace(deps=deps)

    result = take(ctx, "sealed ledger", character_id="merchant")

    assert result == "merchant picks up 'sealed ledger'."
    assert state.characters["merchant"].inventory == ["sealed ledger"]
    assert state.locations["tavern"].items == []
    assert deps.effects == [result]
    assert state.history[-1].characters == ["merchant"]


def test_action_resolver_cannot_recreate_item_held_by_another_character():
    state = _state()
    state.characters["hero"].inventory = ["sealed letter with black wax"]
    actor = state.characters["merchant"]
    deps = ActionResolverDeps(
        char=actor,
        state=state,
        description="pick up the sealed letter from the empty crate",
    )
    ctx = SimpleNamespace(deps=deps)

    first_take = take(ctx, "sealed letter with black wax")
    recreate = create_item(ctx, "sealed letter with black wax")
    second_take = take(ctx, "sealed letter with black wax")

    assert "it's not at 'tavern'" in first_take
    assert recreate == (
        "Cannot create item — 'sealed letter with black wax' already belongs to 'hero'."
    )
    assert "it's not at 'tavern'" in second_take
    assert state.characters["hero"].inventory == ["sealed letter with black wax"]
    assert actor.inventory == []
    assert state.locations["tavern"].items == []
    assert deps.effects == []
    assert state.history == []


def test_action_resolver_rejects_item_name_without_letters_or_numbers():
    state = _state()
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="find an unnamed object in the room",
    )
    ctx = SimpleNamespace(deps=deps)

    result = create_item(ctx, " - ")

    assert result == "Cannot create item — item name must contain a letter or number."
    assert state.locations["tavern"].items == []
    assert deps.effects == []
    assert state.history == []


def test_action_resolver_records_opened_item_state_and_blocks_reopening():
    state = _state()
    state.characters["hero"].inventory = ["sealed wax letter"]
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="open the sealed wax letter and read it",
    )
    ctx = SimpleNamespace(deps=deps)

    opened = change_item(ctx, "sealed wax letter", "opened wax letter")
    repeated = change_item(ctx, "sealed wax letter", "opened wax letter")

    assert opened == "hero changes 'sealed wax letter' into 'opened wax letter'."
    assert repeated == "Cannot change 'sealed wax letter' — hero cannot access it."
    assert state.characters["hero"].inventory == ["opened wax letter"]
    assert deps.effects == [opened]
    assert [event.text for event in state.history] == [opened]


def test_add_detail_reports_unchanged_for_an_existing_location_feature():
    state = _state()
    state.locations["tavern"].features = ["a chalk sigil behind the bar"]
    actor = state.characters["hero"]
    deps = ActionResolverDeps(
        char=actor,
        state=state,
        description="inspect the chalk sigil behind the bar",
    )
    ctx = SimpleNamespace(deps=deps)

    result = add_detail(ctx, "a chalk sigil behind the bar")

    assert result == "Location 'tavern' unchanged."
    assert deps.effects == []
    assert state.locations["tavern"].features == ["a chalk sigil behind the bar"]


def test_action_resolver_lethal_damage_records_defeat_and_awards_xp():
    state = _state()
    merchant = state.characters["merchant"]
    merchant.stats.hp = 1
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="finish the wounded merchant",
    )
    ctx = SimpleNamespace(deps=deps)

    result = adjust_hp(ctx, -1, character_id="merchant")

    assert merchant.stats.hp == 0
    assert state.characters["hero"].stats.xp == 25
    assert "defeating merchant" in result
    assert [event.text for event in state.history] == [
        "merchant takes 1 damage. HP: 0/5.",
        "merchant falls dead.",
        "hero earns 25 XP (defeating merchant).",
    ]
    assert [event.characters for event in state.history] == [
        ["hero", "merchant"],
        ["hero", "merchant"],
        ["hero"],
    ]
    assert deps.effects == [result]


def test_action_resolver_self_inflicted_lethal_damage_does_not_award_xp():
    state = _state()
    hero = state.characters["hero"]
    hero.stats.hp = 1
    deps = ActionResolverDeps(
        char=hero,
        state=state,
        description="drink the poisoned chalice",
    )
    ctx = SimpleNamespace(deps=deps)

    result = adjust_hp(ctx, -1)

    assert result == "hero takes 1 damage. HP: 0/5."
    assert hero.stats.hp == 0
    assert hero.stats.xp == 0
    assert [event.text for event in state.history] == [
        "hero takes 1 damage. HP: 0/5.",
        "hero falls dead.",
    ]
    assert deps.effects == [result]


def test_action_resolver_reveals_known_npc_without_duplicating_identity():
    state = _state()
    state.locations["forest-trail"] = Location(id="forest-trail")
    state.characters["dockmaster-alan"] = Character(
        id="dockmaster-alan",
        role="dockmaster",
        location="tavern",
    )
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="steady Alan after he emerges from the forest",
    )
    ctx = SimpleNamespace(deps=deps)

    result = create_npc(
        ctx,
        name="Alan",
        role="dockmaster",
        location="forest-trail",
    )

    assert result == "dockmaster-alan appears at 'forest-trail'."
    assert "alan" not in state.characters
    assert state.characters["dockmaster-alan"].location == "forest-trail"
    assert deps.effects == [result]
    assert state.history[-1].characters == ["dockmaster-alan"]


def test_action_resolver_preserves_deferred_npc_relationship_identity():
    state = _state()
    state.locations["trade-dock"] = Location(id="trade-dock")
    state.characters["merchant"].relationships = {"alan-dockmaster": "contact"}
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="search the docks for Alan",
    )
    ctx = SimpleNamespace(deps=deps)

    result = create_npc(
        ctx,
        name="Alan",
        role="dock-merchant",
        location="trade-dock",
    )

    assert result == "alan-dockmaster appears at 'trade-dock'."
    assert "alan" not in state.characters
    assert state.characters["merchant"].relationships == {
        "alan-dockmaster": "contact",
    }
    assert deps.effects == [result]


def test_action_resolver_cannot_invent_a_revealed_npc_goal():
    state = _state()
    state.locations["trade-dock"] = Location(id="trade-dock")
    deps = ActionResolverDeps(
        char=state.characters["hero"],
        state=state,
        description="search the docks for Alan after his suspicious withdrawal",
    )
    ctx = SimpleNamespace(deps=deps)

    assert "goal" not in signature(create_npc).parameters

    result = create_npc(
        ctx,
        name="Alan",
        role="dock-merchant",
        backstory="Made a large withdrawal before disappearing from guild records.",
        location="trade-dock",
    )

    assert result == "alan appears at 'trade-dock'."
    assert state.characters["alan"].goal == ""
    assert state.characters["alan"].backstory == (
        "Made a large withdrawal before disappearing from guild records."
    )


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


def test_modify_update_quest_normalizes_id():
    state = _state()
    state.quests["missing-brother"] = Quest(
        id="missing-brother",
        title="The Missing Brother",
        owner="hero",
    )

    result = asyncio.run(resolve(
        Modify(action="update_quest", target_id="MISSING BROTHER", advance=True),
        state,
    ))

    assert "updated" in result
    assert state.quests["missing-brother"].current_step == 1
    assert state.characters["hero"].stats.xp == 10


def test_modify_normalizes_character_and_location_targets():
    state = _state()

    location_result = asyncio.run(resolve(
        Modify(action="update_location", target_id="TAVERN", reason="The shutters are barred."),
        state,
    ))
    npc_result = asyncio.run(resolve(
        Modify(action="remove_npc", target_id="MERCHANT", reason="Leaves for the market."),
        state,
    ))

    assert "updated" in location_result
    assert state.locations["tavern"].description == "The shutters are barred."
    assert "merchant is gone" in npc_result
    assert "merchant" not in state.characters


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


def test_modify_does_not_report_completed_faction_clock_as_advanced():
    state = _state()
    state.factions["crew"] = Faction(
        id="crew",
        name="The Crew",
        goal="move the goods",
        clocks=[
            ProgressClock(
                id="haul",
                name="Haul the goods",
                progress=2,
                segments=2,
                consequence="The goods are gone.",
                consequence_triggered=True,
            )
        ],
    )

    result = asyncio.run(resolve(
        Modify(action="advance_faction_clock", target_id="crew", other_id="haul"),
        state,
    ))

    assert result == "Clock 'haul' no longer advances — it is already completed."
    assert state.factions["crew"].clocks[0].progress == 2
    assert state.history == []


def test_modify_advance_faction_clock_normalizes_ids():
    state = _state()
    state.factions["black-hull-crew"] = Faction(
        id="black-hull-crew",
        name="The Black Hull Crew",
        goal="move the goods",
        clocks=[ProgressClock(id="secret-haul", name="Haul the goods", segments=2, consequence="The goods are gone.")],
    )
    tool = Modify(action="advance_faction_clock", target_id="BLACK_HULL_CREW", other_id="SECRET HAUL")

    result = asyncio.run(resolve(tool, state))

    assert state.factions["black-hull-crew"].clocks[0].progress == 1
    assert "1/2" in result


def test_create_normalizes_known_location_ids():
    state = _state()

    asyncio.run(resolve(Create(type="item", name="Brass Key", location="TAVERN"), state))
    asyncio.run(resolve(Create(type="npc", name="Dock Guard", location="Tavern"), state))
    asyncio.run(resolve(Create(type="location", name="Back Room", location="TAVERN"), state))

    assert "Brass Key" in state.locations["tavern"].items
    assert state.characters["dock-guard"].location == "tavern"
    assert state.locations["back-room"].connections == ["tavern"]
    assert "back-room" in state.locations["tavern"].connections


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


def test_contested_check_rejects_self_opponent_alias_without_logging():
    state = _state()

    result = asyncio.run(resolve(Check(
        actor="hero", ability="charisma", description="argues with himself",
        opponent="HERO", modifier=1, opposing_modifier=2,
    ), state, rng=_Rolls(16, 10)))

    assert result == "Cannot resolve check — actor and opponent must be different characters."
    assert state.history == []


def test_contested_check_rejects_remote_opponent_without_logging():
    state = _state()
    state.locations["forest"] = Location(id="forest")
    state.characters["merchant"].location = "forest"

    result = asyncio.run(resolve(Check(
        actor="hero", ability="charisma", description="bluffs the merchant",
        opponent="merchant", modifier=1, opposing_modifier=2,
    ), state, rng=_Rolls(16, 10)))

    assert "not in the same location" in result
    assert state.history == []


def test_contested_check_rejects_dead_opponent_without_logging():
    state = _state()
    state.characters["merchant"].stats.hp = 0

    result = asyncio.run(resolve(Check(
        actor="hero", ability="charisma", description="intimidates the merchant",
        opponent="merchant", modifier=1, opposing_modifier=2,
    ), state, rng=_Rolls(16, 10)))

    assert result == "Cannot resolve check — opponent 'merchant' is dead."
    assert state.history == []


def test_dead_character_cannot_use_any_character_tool_or_self_update():
    state = _state()
    hero = state.characters["hero"]
    hero.stats.hp = 0
    tools = [
        Speak(actor="hero", message="I still live."),
        Travel(actor="hero", destination="tavern"),
        Wait(actor="hero"),
        Action(actor="hero", description="search the room"),
        Attack(actor="hero", target="merchant"),
        Check(actor="hero", ability="wisdom", description="notice a clue"),
    ]

    for tool in tools:
        result = asyncio.run(resolve(
            tool.model_copy(update={
                "remember": "a posthumous discovery",
                "new_goal": "haunt the innkeeper",
            }),
            state,
        ))
        assert "is dead" in result

    assert hero.location == "tavern"
    assert hero.knowledge == []
    assert hero.goal == "find the ale"
    assert state.history == []


def test_rejected_character_action_cannot_apply_optional_self_updates():
    state = _state()

    result = asyncio.run(resolve(
        Travel(
            actor="hero",
            destination="missing-road",
            remember="the missing road leads north",
            new_goal="follow the missing road",
        ),
        state,
    ))

    assert result == "Cannot move to 'missing-road' — location not found."
    assert state.characters["hero"].location == "tavern"
    assert state.characters["hero"].knowledge == []
    assert state.characters["hero"].goal == "find the ale"
    assert state.history == []


def test_attack_resolution_uses_injected_rng(monkeypatch):
    state = _state()
    monkeypatch.setattr("src.engine.rules.random.randint", lambda _low, _high: 1)

    result = asyncio.run(resolve(
        Attack(actor="hero", target="merchant"),
        state,
        rng=_Rolls(4),
    ))

    assert "for 4 damage" in result
    assert state.characters["merchant"].stats.hp == 1


def test_deterministic_dispatch_normalizes_character_ids_and_self_updates():
    state = _state()

    result = asyncio.run(resolve(
        Attack(
            actor="HERO",
            target="Merchant",
            remember="the merchant carries a silver key",
        ),
        state,
        rng=_Rolls(4),
    ))

    assert "hero attacks merchant for 4 damage" in result
    assert state.characters["merchant"].stats.hp == 1
    assert state.characters["hero"].knowledge == ["the merchant carries a silver key"]


def test_deterministic_dispatch_normalizes_travel_location_id():
    state = _state()
    state.locations["tavern"].connections.append("forest")
    state.locations["forest"] = Location(id="forest", connections=["tavern"])

    result = asyncio.run(resolve(Travel(actor="HERO", destination="Forest"), state))

    assert "hero moved to 'forest'" in result
    assert state.characters["hero"].location == "forest"
