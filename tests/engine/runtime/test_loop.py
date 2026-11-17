import asyncio
from unittest.mock import Mock
from src.interface.session_log import Logger

import pytest
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior

from src.agents.character.tools import Action, Attack, Check, Speak, Travel, Wait
from src.engine.runtime.loop import (
    MAX_FAILED_TICKS,
    MIN_TICKS_BETWEEN_ADVANCES,
    _STALL_QUIET_TICKS,
    _can_accelerate_faction_clock,
    _can_anchor_new_location,
    _can_anchor_new_npc,
    _is_stalled,
    _minimum_action_minutes,
    _pick_next_actor,
    _quest_veto,
    _record_intervention,
    _record_resolution_if_needed,
    _run_game,
    _scene_actors,
    tick,
)
from src.agents.dm.agent import DMResult
from src.agents.dm.tools import Create, Modify
from src.engine.state.models import (
    Character,
    Faction,
    HistoryEvent,
    Location,
    ProgressClock,
    Quest,
    WorldState,
)
from src.engine.state.operations import WorldOperations


def _state() -> WorldState:
    return WorldState(
        locations={"tavern": Location(id="tavern")},
        characters={"hero": Character(id="hero", role="warrior", location="tavern")},
        quests={
            "q1": Quest(
                id="q1",
                title="Clear the Cave",
                description="",
                owner="hero",
                plan=["scout", "clear"],
            )
        },
    )


def _event(text: str) -> HistoryEvent:
    return HistoryEvent(text=text, location="tavern", characters=["hero"])


def test_dm_npc_reveal_rejects_a_knowledge_subject_at_the_learners_location():
    create = Create(type="npc", name="Dockmaster Alan", location="tavern")

    assert not _can_anchor_new_npc(
        create,
        [_event("hero learns: Dockmaster Alan keeps the cargo ledger.")],
    )


def test_dm_npc_reveal_allows_an_explicit_remote_location():
    create = Create(type="npc", name="Dockmaster Alan", location="forest-trail")
    event = _event("hero learns: Dockmaster Alan is waiting at forest-trail.")

    assert _can_anchor_new_npc(create, [event])


def test_dm_npc_reveal_allows_title_and_name_in_generated_order():
    create = Create(type="npc", name="Dockmaster Alan", location="trade-dock")
    event = _event("hero learns: Alan the dockmaster works at trade-dock.")

    assert _can_anchor_new_npc(create, [event])


def test_dm_npc_reveal_does_not_borrow_a_rosters_location():
    create = Create(type="npc", name="Margot", location="servants-quarters")
    event = _event(
        "hero learns: The roster in servants-quarters covers winter 1287. "
        "Servants on duty then: Margot (head cook), Thomas, and Elwin."
    )

    assert not _can_anchor_new_npc(create, [event])


def test_dm_npc_reveal_rejects_a_prior_name_at_the_turns_location():
    create = Create(type="npc", name="Calla", location="servants-quarters")
    event = HistoryEvent(
        text="hero moved to 'servants-quarters'.",
        location="servants-quarters",
        characters=["hero"],
    )

    assert not _can_anchor_new_npc(create, [event])


def test_dm_npc_reveal_does_not_borrow_a_speakers_location():
    create = Create(type="npc", name="Calla", location="manor-foyer")
    event = HistoryEvent(
        text=(
            'finnian-grey says to hero: "Watch for that new scullion Calla '
            'asking questions about the vault."'
        ),
        location="manor-foyer",
        characters=["finnian-grey", "hero"],
    )

    assert not _can_anchor_new_npc(create, [event])


def test_dm_npc_reveal_can_use_a_named_participants_location():
    create = Create(type="npc", name="Calla", location="manor-foyer")
    event = HistoryEvent(
        text="Calla grabs hero's sleeve.",
        location="manor-foyer",
        characters=["calla", "hero"],
    )

    assert _can_anchor_new_npc(create, [event])


def test_dm_npc_reveal_rejects_an_organization_at_the_turns_location():
    create = Create(
        type="npc",
        name="Crimson Tide trading company",
        description="A smuggling faction operating out of the eastern ports.",
        location="tavern",
        role="smuggling faction",
    )

    assert not _can_anchor_new_npc(
        create,
        [_event("hero learns: The Crimson Tide trading company is using the cove.")],
    )


def test_dm_npc_reveal_allows_a_person_affiliated_with_an_organization():
    create = Create(
        type="npc",
        name="Mara Voss",
        description="An agent for the Crimson Tide trading company.",
        location="tavern",
        role="company agent",
    )

    assert _can_anchor_new_npc(
        create,
        [_event("hero learns: Mara Voss handles the cargo ledger at the tavern.")],
    )


def test_dm_location_reveal_allows_a_concrete_named_destination():
    create = Create(
        type="location",
        name="Blackglass Tower",
        description="A ruined watchtower above the river.",
        location="tavern",
    )

    assert _can_anchor_new_location(
        create,
        [_event("hero learns: The smugglers carried the cargo to Blackglass Tower.")],
    )


def test_tick_rejects_npc_reveal_at_an_unsupported_remote_location(monkeypatch):
    state = _state()
    state.locations["village-square"] = Location(id="village-square")
    logged_events = []

    class CapturingLogger(Logger):
        def __init__(self):
            pass

        def log_event(self, event, **kwargs):
            logged_events.append((event, kwargs))

    async def inspect_ship_log(*_args):
        return Action(actor="hero", description="inspect the ship log")

    async def resolve_log(tool, world, **_kwargs):
        if isinstance(tool, Action):
            event = _event(
                "hero learns: The ship log names no shore contact, only the Handler."
            )
            world.history.append(event)
            return event.text
        raise AssertionError("the unsupported NPC reveal must not reach resolution")

    async def invent_handler_location(*_args):
        return DMResult(
            creates=[
                Create(
                    type="npc",
                    name="the-handler",
                    description="An unknown shore contact.",
                    location="village-square",
                )
            ],
            modifies=[],
            minutes=[1],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", inspect_ship_log)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_log)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", invent_handler_location)

    asyncio.run(tick("hero", state, 0, logger=CapturingLogger()))

    assert "the-handler" not in state.characters
    assert [event.text for event in state.history] == [
        "hero learns: The ship log names no shore contact, only the Handler."
    ]
    assert logged_events == [
        (
            "world_update_rejected",
            {
                "update": "create",
                "entity_type": "npc",
                "name": "the-handler",
                "location": "village-square",
                "reason": "unsupported_npc_evidence",
            },
        )
    ]


def test_tick_does_not_turn_a_named_company_into_a_character(monkeypatch):
    state = _state()

    async def inspect_crate(*_args):
        return Action(actor="hero", description="inspect the smuggler crate")

    async def resolve_crate(tool, world, **_kwargs):
        if isinstance(tool, Action):
            event = _event(
                "hero learns: The Crimson Tide trading company is using the cove."
            )
            world.history.append(event)
            return event.text
        raise AssertionError("the company reveal must not reach resolution")

    async def misclassify_company(*_args):
        return DMResult(
            creates=[
                Create(
                    type="npc",
                    name="Crimson Tide trading company",
                    description="A smuggling faction operating out of the eastern ports.",
                    location="tavern",
                    role="smuggling faction",
                )
            ],
            modifies=[],
            minutes=[2],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", inspect_crate)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_crate)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", misclassify_company)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert "crimson-tide-trading-company" not in state.characters
    assert [event.text for event in state.history] == [
        "hero learns: The Crimson Tide trading company is using the cove."
    ]


def test_tick_does_not_materialize_a_vague_placeholder_location(monkeypatch):
    state = _state()

    async def inspect_manifest(*_args):
        return Action(actor="hero", description="inspect the shipping manifest")

    async def resolve_manifest(tool, world, **_kwargs):
        if isinstance(tool, Action):
            event = _event(
                "hero learns: The manifest says the cargo came from an upriver location."
            )
            world.history.append(event)
            return event.text
        if isinstance(tool, Create):
            return WorldOperations(world).add_location(
                tool.name,
                description=tool.description,
                connections=[tool.location] if tool.location else [],
            )
        raise AssertionError(f"unexpected resolution: {tool!r}")

    async def invent_placeholder(*_args):
        return DMResult(
            creates=[
                Create(
                    type="location",
                    name="upriver-location",
                    description="An unspecified upriver source of cargo.",
                    location="tavern",
                )
            ],
            modifies=[],
            minutes=[5],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", inspect_manifest)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_manifest)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", invent_placeholder)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert set(state.locations) == {"tavern"}
    assert [event.text for event in state.history] == [
        "hero learns: The manifest says the cargo came from an upriver location."
    ]


def test_tick_does_not_place_a_previously_mentioned_npc_after_travel(monkeypatch):
    state = _state()
    state.locations["tavern"].connections.append("servants-quarters")
    state.locations["servants-quarters"] = Location(id="servants-quarters")

    async def travel_to_servants_quarters(*_args):
        return Travel(actor="hero", destination="servants-quarters")

    async def reveal_calla_from_prior_context(*_args):
        return DMResult(
            creates=[
                Create(
                    type="npc",
                    name="Calla",
                    description="A scullion mentioned on an earlier turn.",
                    location="servants-quarters",
                )
            ],
            modifies=[],
            minutes=[10],
        )

    monkeypatch.setattr(
        "src.engine.runtime.loop.flow_agent_turn", travel_to_servants_quarters
    )
    monkeypatch.setattr(
        "src.engine.runtime.loop.flow_dm", reveal_calla_from_prior_context
    )

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert state.characters["hero"].location == "servants-quarters"
    assert "calla" not in state.characters
    assert [event.text for event in state.history] == [
        "hero moved to 'servants-quarters'."
    ]


def test_tick_does_not_place_a_remembered_npc_at_the_travel_destination(monkeypatch):
    state = _state()
    state.locations["tavern"].connections.append("manor-foyer")
    state.locations["manor-foyer"] = Location(id="manor-foyer")
    fact = (
        "The circlet was taken from the reading table between dinner and now. "
        "Calla, a new scullion, was asking about the vault."
    )

    async def travel_to_foyer(*_args):
        return Travel(actor="hero", destination="manor-foyer", remember=fact)

    async def place_calla_at_foyer(*_args):
        return DMResult(
            creates=[
                Create(
                    type="npc",
                    name="Calla",
                    description="A new scullion who was asking about the vault.",
                    location="manor-foyer",
                )
            ],
            modifies=[],
            minutes=[10, 1],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", travel_to_foyer)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", place_calla_at_foyer)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert state.characters["hero"].location == "manor-foyer"
    assert state.characters["hero"].knowledge == [fact]
    assert "calla" not in state.characters
    assert [event.text for event in state.history] == [
        "hero moved to 'manor-foyer'.",
        f"hero learns: {fact}",
    ]


def test_tick_does_not_materialize_historical_people_from_a_roster(monkeypatch):
    state = _state()
    state.locations["servants-quarters"] = Location(id="servants-quarters")
    roster_fact = (
        "The servants' roster in servants-quarters covers winter 1287. "
        "Servants on duty then: Margot (head cook), Thomas (groundskeeper), "
        "and Elwin (chamberlain). None of the current staff served then."
    )

    async def inspect_roster(*_args):
        return Action(actor="hero", description="inspect the servants' roster")

    async def resolve_roster(tool, world, **_kwargs):
        if isinstance(tool, Action):
            event = _event(f"hero learns: {roster_fact}")
            world.history.append(event)
            return event.text
        raise AssertionError("historical NPC reveals must not reach resolution")

    async def materialize_historical_staff(*_args):
        return DMResult(
            creates=[
                Create(
                    type="npc",
                    name=name,
                    role=role,
                    description=f"{role} on duty in winter 1287.",
                    location="servants-quarters",
                )
                for name, role in (
                    ("Margot", "head cook"),
                    ("Thomas", "groundskeeper"),
                    ("Elwin", "chamberlain"),
                )
            ],
            modifies=[],
            minutes=[15],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", inspect_roster)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_roster)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", materialize_historical_staff)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert set(state.characters) == {"hero"}
    assert [event.text for event in state.history] == [f"hero learns: {roster_fact}"]


def test_tick_skips_dm_item_already_created_by_the_action(monkeypatch):
    state = _state()

    async def search_office(*_args):
        return Action(actor="hero", description="search the dockmaster's office")

    async def create_journal(tool, world, **_kwargs):
        if isinstance(tool, Action):
            return WorldOperations(world).create_item(
                "Alan's leather journal", "tavern"
            )
        raise AssertionError(
            "DM enrichment must not recreate an item already in the world"
        )

    async def repeat_journal(*_args):
        return DMResult(
            creates=[
                Create(
                    type="item",
                    name="Alan's leather journal",
                    description="A journal revealed by the resolved search.",
                    location="tavern",
                )
            ],
            modifies=[],
            minutes=[5],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", search_office)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", create_journal)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", repeat_journal)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert state.locations["tavern"].items == ["Alan's leather journal"]
    assert [event.text for event in state.history] == [
        "'Alan's leather journal' appears at 'tavern'."
    ]


# ============ SCHEDULER ============


def test_scene_scheduler_skips_goal_less_unaddressed_npcs():
    state = _state()
    state.characters["dockmaster"] = Character(id="dockmaster", location="tavern")

    assert _scene_actors(state, "hero") == ["hero"]


def test_scene_scheduler_includes_goal_less_npc_after_direct_address():
    state = _state()
    state.characters["dockmaster"] = Character(id="dockmaster", location="tavern")
    WorldOperations(state).speak("hero", "Show me the cargo ledger.", "dockmaster")
    state.history.append(
        HistoryEvent(
            text="dockmaster's relationship with hero is now 'wary'.",
            location="tavern",
            characters=["dockmaster", "hero"],
        )
    )

    assert _scene_actors(state, "hero") == ["hero", "dockmaster"]

    WorldOperations(state).speak("dockmaster", "Here it is.", "hero")

    assert _scene_actors(state, "hero") == ["hero"]


def test_scene_scheduler_prioritizes_a_directly_addressed_npc_reply():
    state = _state()
    state.characters["calla"] = Character(id="calla", location="tavern")
    WorldOperations(state).speak("hero", "What did you find?", "calla")

    # Tick 8 would otherwise rotate back to the PC: 8 % 2 == 0.
    assert _pick_next_actor(state, "hero", 8) == "calla"

    WorldOperations(state).speak("calla", "The lock hides a passage.", "hero")

    assert _pick_next_actor(state, "hero", 9) == "hero"


def test_npcs_addressing_each_other_do_not_starve_the_hero_of_turns():
    """Seen live: two NPCs answered each other for six ticks straight."""
    state = _state()
    for npc in ("calla", "keph"):
        state.characters[npc] = Character(id=npc, location="tavern", goal="talk")
    WorldOperations(state).speak("calla", "This crate sails.", "keph")

    assert _pick_next_actor(state, "hero", 0) == "hero"


def test_hero_acts_every_other_tick_in_a_crowded_scene():
    """Seen live: four NPCs in the scene left the hero one turn in five."""
    state = _state()
    for npc in ("calla", "keph", "marin"):
        state.characters[npc] = Character(id=npc, location="tavern", goal="talk")

    turns = [_pick_next_actor(state, "hero", tick) for tick in range(6)]

    assert turns == ["hero", "calla", "hero", "keph", "hero", "marin"]


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
    for text in (
        'hero says: "hm"',
        "hero waits.",
        'hero says to bob: "well?"',
        "hero waits.",
        'hero says: "so..."',
    ):
        state.history.append(_event(text))
    assert _is_stalled(state)


def test_mixed_recent_events_not_stalled():
    state = _state()
    for text in (
        'hero says: "hm"',
        "hero waits.",
        "hero picks up 'sword'.",
        "hero waits.",
        'hero says: "so..."',
    ):
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

    events = _record_resolution_if_needed(
        state, "hero", 0, "  Nothing useful is hidden here.  "
    )

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
    assert (
        _minimum_action_minutes(Action(actor="hero", description="search the archive"))
        == 5
    )
    assert (
        _minimum_action_minutes(Action(actor="hero", description="open the door")) == 1
    )


def test_only_active_interactions_can_accelerate_faction_clocks():
    state = _state()
    state.characters["guild-guard"] = Character(
        id="guild-guard", role="guild sentry", location="tavern"
    )
    state.factions["guild"] = Faction(
        id="guild",
        name="The Guild",
        goal="Control the harbor",
        clocks=[
            ProgressClock(
                id="alarm",
                name="Raise the alarm",
                consequence="The gates close.",
                segments=4,
            )
        ],
    )

    detected = [_event("A guild lookout spots hero and warns the sentries.")]
    direct = [
        HistoryEvent(
            text='hero says to guild-guard: "Sound the alarm."',
            location="tavern",
            characters=["hero", "guild-guard"],
        )
    ]

    def can_accelerate(tool, events=detected):
        return _can_accelerate_faction_clock(state, tool, "guild", "alarm", events)

    assert can_accelerate(
        Speak(actor="hero", target="guild-guard", message="sound the alarm"), direct
    )
    assert can_accelerate(Action(actor="hero", description="burn the guild records"))
    assert can_accelerate(Attack(actor="hero", target="guild-guard"))
    assert not can_accelerate(Travel(actor="hero", destination="road"))
    assert not can_accelerate(Wait(actor="hero"))
    assert not can_accelerate(Action(actor="hero", description="inspect a crate"), [])


def test_finding_faction_evidence_does_not_accelerate_its_clock():
    state = _state()
    state.factions["black-hull-crew"] = Faction(
        id="black-hull-crew",
        name="The Black-Hull Crew",
        goal="Move contraband and silence witnesses",
        clocks=[
            ProgressClock(
                id="retaliation",
                name="Prepare retaliation",
                consequence="The crew attacks the village.",
                segments=8,
            )
        ],
    )
    event = _event(
        "hero learns: The crate bears three crossed anchors in black paint—symbol of the Deepwater Compact crew."
    )

    assert not _can_accelerate_faction_clock(
        state,
        Action(actor="hero", description="inspect the smuggler crate"),
        "black-hull-crew",
        "retaliation",
        [event],
    )


def test_notice_board_evidence_does_not_accelerate_faction_clock():
    state = _state()
    state.factions["black-hull-crew"] = Faction(
        id="black-hull-crew",
        name="The Black-Hull Crew",
        goal="Move contraband and silence witnesses",
        clocks=[
            ProgressClock(
                id="retaliation",
                name="Prepare retaliation",
                consequence="The crew attacks the village.",
                segments=8,
            )
        ],
    )
    event = _event(
        "hero learns: The harbormaster's notice board flags the black-hulled sloop "
        "with warnings about irregular docking and disputed cargo manifests."
    )

    assert not _can_accelerate_faction_clock(
        state,
        Action(actor="hero", description="search the harbormaster's notice board"),
        "black-hull-crew",
        "retaliation",
        [event],
    )


def test_faction_lookout_noticing_actor_can_accelerate_clock():
    state = _state()
    state.factions["black-hull-crew"] = Faction(
        id="black-hull-crew",
        name="The Black-Hull Crew",
        goal="Move contraband and silence witnesses",
        clocks=[
            ProgressClock(
                id="retaliation",
                name="Prepare retaliation",
                consequence="The crew attacks the village.",
                segments=8,
            )
        ],
    )
    event = _event("A Black-Hull Crew lookout notices hero searching the cave.")

    assert _can_accelerate_faction_clock(
        state,
        Action(actor="hero", description="search the smugglers' cave"),
        "black-hull-crew",
        "retaliation",
        [event],
    )


def test_tick_binds_tool_to_scheduled_actor(monkeypatch):
    state = _state()
    state.characters["ally"] = Character(id="ally", role="scout", location="tavern")

    async def choose_action(*_args):
        return Speak(
            actor="ally", message="The wrong actor must not control this turn."
        )

    async def report_turn(*_args):
        return DMResult(creates=[], modifies=[], minutes=[1])

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_turn)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert (
        state.history[-1].text
        == 'hero says: "The wrong actor must not control this turn."'
    )
    assert state.history[-1].characters == ["hero"]


def test_tick_ignores_minute_estimates_without_matching_events(monkeypatch):
    state = _state()

    async def choose_action(*_args):
        return Speak(actor="hero", message="One brief warning.")

    async def report_turn(*_args):
        return DMResult(creates=[], modifies=[], minutes=[2, 60])

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_turn)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert len(state.history) == 1
    assert state.history[0].minutes_elapsed == 2
    assert state.minutes_elapsed == 2


def test_tick_clamps_negative_minute_estimates_for_later_events(monkeypatch):
    state = _state()

    async def choose_action(*_args):
        return Speak(
            actor="hero",
            message="Remember the broken seal.",
            remember="The letter's seal was already broken.",
        )

    async def report_turn(*_args):
        return DMResult(creates=[], modifies=[], minutes=[2, -10])

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_turn)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert [event.minutes_elapsed for event in state.history] == [2, 0]
    assert state.minutes_elapsed == 2
    WorldState.model_validate(state.model_dump())


def test_tick_does_not_charge_time_for_later_knowledge_event(monkeypatch):
    state = _state()

    async def choose_action(*_args):
        return Speak(
            actor="hero",
            message="Remember the broken seal.",
            remember="The letter's seal was already broken.",
        )

    async def report_turn(*_args):
        return DMResult(creates=[], modifies=[], minutes=[2, 5])

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_turn)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert [event.minutes_elapsed for event in state.history] == [2, 0]
    assert state.minutes_elapsed == 2


def test_tick_rejects_dm_removal_of_active_player_character(monkeypatch):
    state = _state()

    async def choose_action(*_args):
        return Speak(actor="hero", message="I am still here.")

    async def remove_player(*_args):
        return DMResult(
            creates=[],
            modifies=[
                Modify(action="remove_npc", target_id="HERO", reason="Disappears.")
            ],
            minutes=[1],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", remove_player)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert "hero" in state.characters
    assert state.history[-1].text == 'hero says: "I am still here."'


def test_tick_rejects_faction_clock_acceleration_for_travel(monkeypatch):
    state = _state()
    state.locations["tavern"].connections.append("servants-quarters")
    state.locations["servants-quarters"] = Location(id="servants-quarters")
    state.factions["encroaching-dawn"] = Faction(
        id="encroaching-dawn",
        name="The Encroaching Dawn",
        goal="End the search before the curse can be broken",
        clocks=[
            ProgressClock(
                id="dawn-breaks",
                name="Dawn breaks",
                consequence="The curse becomes permanent.",
                segments=6,
            )
        ],
    )

    async def choose_travel(*_args):
        return Travel(actor="hero", destination="servants-quarters")

    async def accelerate_dawn(*_args):
        return DMResult(
            creates=[],
            modifies=[
                Modify(
                    action="advance_faction_clock",
                    target_id="encroaching-dawn",
                    other_id="dawn-breaks",
                )
            ],
            minutes=[10],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_travel)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", accelerate_dawn)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert state.characters["hero"].location == "servants-quarters"
    assert state.factions["encroaching-dawn"].clocks[0].progress == 0


def test_tick_rejects_event_acceleration_for_elapsed_time_deadline(monkeypatch):
    state = _state()
    state.factions["encroaching-dawn"] = Faction(
        id="encroaching-dawn",
        name="The Encroaching Dawn",
        goal="End the search before the curse can be broken",
        clocks=[
            ProgressClock(
                id="dawn-breaks",
                name="Dawn breaks",
                consequence="The curse becomes permanent.",
                segments=6,
                event_acceleration=False,
            )
        ],
    )

    async def discuss_search(*_args):
        return Speak(actor="hero", message="We should keep searching before dawn.")

    async def accelerate_dawn(*_args):
        return DMResult(
            creates=[],
            modifies=[
                Modify(
                    action="advance_faction_clock",
                    target_id="encroaching-dawn",
                    other_id="dawn-breaks",
                )
            ],
            minutes=[2],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", discuss_search)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", accelerate_dawn)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert state.minutes_elapsed == 2
    assert state.factions["encroaching-dawn"].clocks[0].progress == 0


def test_tick_allows_event_acceleration_for_action_driven_clock(monkeypatch):
    state = _state()
    state.characters["smuggler-guard"] = Character(
        id="smuggler-guard",
        role="smuggler crew lookout",
        location="tavern",
    )
    state.factions["crew"] = Faction(
        id="crew",
        name="The Smuggler Crew",
        goal="Hide the evidence",
        clocks=[
            ProgressClock(
                id="retaliation",
                name="Prepare retaliation",
                consequence="The crew attacks.",
                segments=4,
            )
        ],
    )

    async def confront_crew(*_args):
        return Speak(
            actor="hero",
            target="smuggler-guard",
            message="I know about your smuggling operation.",
        )

    async def accelerate_retaliation(*_args):
        return DMResult(
            creates=[],
            modifies=[
                Modify(
                    action="advance_faction_clock",
                    target_id="crew",
                    other_id="retaliation",
                )
            ],
            minutes=[2],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", confront_crew)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", accelerate_retaliation)

    asyncio.run(tick("hero", state, 0, logger=Mock(spec=Logger)))

    assert state.factions["crew"].clocks[0].progress == 1


def test_tick_rejects_faction_acceleration_for_unobserved_evidence_discovery(
    monkeypatch,
):
    state = _state()
    logged_events = []

    class CapturingLogger(Logger):
        def __init__(self):
            pass

        def log_event(self, event, **kwargs):
            logged_events.append((event, kwargs))

    state.factions["black-hull-crew"] = Faction(
        id="black-hull-crew",
        name="The Black-Hull Crew",
        goal="Move contraband and silence witnesses",
        clocks=[
            ProgressClock(
                id="retaliation",
                name="Prepare retaliation",
                consequence="The crew attacks the village.",
                segments=8,
            )
        ],
    )

    async def inspect_crate(*_args):
        return Action(actor="hero", description="inspect the smuggler crate")

    async def resolve_intent(tool, world, **_kwargs):
        if isinstance(tool, Action):
            event = _event(
                "hero learns: The crate bears three crossed anchors in black paint—symbol of the Deepwater Compact crew."
            )
            world.history.append(event)
            return event.text
        raise AssertionError("ungrounded clock acceleration must not reach resolution")

    async def accelerate_retaliation(*_args):
        return DMResult(
            creates=[],
            modifies=[
                Modify(
                    action="advance_faction_clock",
                    target_id="black-hull-crew",
                    other_id="retaliation",
                )
            ],
            minutes=[5],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", inspect_crate)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_intent)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", accelerate_retaliation)

    asyncio.run(tick("hero", state, 0, logger=CapturingLogger()))

    assert state.factions["black-hull-crew"].clocks[0].progress == 0
    assert logged_events == [
        (
            "world_update_rejected",
            {
                "update": "advance_faction_clock",
                "target_id": "black-hull-crew",
                "other_id": "retaliation",
                "reason": "unsupported_faction_acceleration",
            },
        )
    ]


# ============ QUEST PROGRESS ============


def _quest_world() -> WorldState:
    state = WorldState(
        time=10,
        last_quest_advance_time=10,
        locations={
            "harbor": Location(id="harbor", connections=["ruins"]),
            "ruins": Location(id="ruins", connections=["harbor"]),
        },
        characters={
            "hero": Character(id="hero", location="ruins"),
            "elara": Character(id="elara", location="ruins"),
            "varek": Character(id="varek", location="ruins"),
        },
        quests={
            "q1": Quest(
                id="q1",
                title="Debt",
                description="",
                owner="hero",
                plan=["interrogate Elara at the ruins", "settle the debt"],
            )
        },
    )
    return state


def _veto(state, tool, events=None, advanced=()):
    events = events if events is not None else [_event("something happened")]
    return _quest_veto(state, state.quests["q1"], tool, events, set(advanced))


def test_quest_progress_needs_a_real_attempt_by_the_owner():
    state = _quest_world()
    asked = [
        HistoryEvent(
            text='hero says to elara: "?"',
            location="ruins",
            characters=["hero", "elara"],
        )
    ]
    assert _veto(state, Speak(actor="hero", target="elara", message="?"), asked) is None
    assert _veto(state, Travel(actor="hero", destination="harbor")) == "not_an_attempt"
    assert _veto(state, Wait(actor="hero")) == "not_an_attempt"
    assert _veto(state, Action(actor="varek", description="search")) == "not_owner"


def test_quest_progress_accepts_a_reply_only_from_someone_the_objective_names():
    state = _quest_world()
    reply = [
        HistoryEvent(
            text='elara says to hero: "!"',
            location="ruins",
            characters=["elara", "hero"],
        )
    ]
    assert _veto(state, Speak(actor="elara", target="hero", message="!"), reply) is None
    other = [
        HistoryEvent(
            text='varek says to hero: "!"',
            location="ruins",
            characters=["varek", "hero"],
        )
    ]
    assert (
        _veto(state, Speak(actor="varek", target="hero", message="!"), other)
        == "not_owner"
    )


def test_quest_progress_needs_the_named_place_and_person():
    state = _quest_world()
    examine = Action(actor="hero", description="examine her ledger", target="elara")
    assert _veto(state, examine) is None
    # Seen live: "follow Glimmerfin's sigils" was vetoed with her at the hero's side.
    assert _veto(state, Action(actor="hero", description="search")) is None
    state.characters["elara"].location = "harbor"
    assert _veto(state, examine) is None  # named in passing: she is not at the ruins
    state.quests["q1"].plan[0] = "interrogate Elara"
    assert _veto(state, examine) == "named_person_absent"
    state.quests["q1"].plan[0] = "interrogate Elara at the ruins"
    state.characters["hero"].location = "harbor"
    assert _veto(state, examine) == "wrong_location"


def test_a_person_named_in_passing_elsewhere_is_not_required():
    """Seen live: 'check the well for supplies Jory guards' while Jory was not there."""
    state = _quest_world()
    state.characters["elara"].location = "harbor"
    assert _veto(state, Action(actor="hero", description="search")) is None


def test_failed_attempts_do_not_advance_quests():
    state = _quest_world()
    state.quests["q1"].plan = ["pick the lock", "settle the debt"]
    check = Check(actor="hero", ability="dexterity", description="pick the lock")
    failed = [_event("hero fails: pick the lock [dexterity; 3+0=3 vs DC 10].")]
    assert _veto(state, check, failed) == "failed_attempt"
    assert _veto(state, check, [_event("Cannot resolve check — x.")]) == (
        "failed_attempt"
    )
    assert _veto(state, check, [_event("hero succeeds: pick the lock.")]) is None


def test_quest_progress_is_paced_to_one_step_every_few_ticks():
    state = _quest_world()
    quest = state.quests["q1"]
    quest.current_step, quest.last_progress = 1, state.time - 1
    act = Action(actor="hero", description="pay varek")
    assert _veto(state, act) == "too_soon"
    quest.last_progress = state.time - MIN_TICKS_BETWEEN_ADVANCES
    assert _veto(state, act) is None
    assert _veto(state, act, advanced={"q1"}) == "one_step_per_tick"


def test_tick_advances_one_step_per_quest_even_when_the_dm_claims_more(monkeypatch):
    state = _quest_world()
    state.quests["q1"].plan = ["search the ruins", "read the ledger", "pay varek"]

    async def search(*_args):
        return Action(actor="hero", description="search the ruins")

    async def resolve_search(intent, world, **kwargs):
        if isinstance(intent, Modify):
            return WorldOperations(world).advance_quest(
                intent.target_id, advance=intent.advance
            )
        world.history.append(
            HistoryEvent(
                text="hero finds a ledger.", location="ruins", characters=["hero"]
            )
        )
        return "hero finds a ledger."

    async def claim_twice(*_args):
        claim = Modify(action="update_quest", target_id="Debt", advance=True)
        return DMResult(creates=[], modifies=[claim, claim], minutes=[5])

    monkeypatch.setattr("src.engine.runtime.loop._pick_next_actor", lambda *_: "hero")
    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", search)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_search)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", claim_twice)
    logger = Mock()

    asyncio.run(tick("hero", state, 0, logger))

    assert state.quests["q1"].current_step == 1
    vetoes = [
        c
        for c in logger.log_event.call_args_list
        if c.args[0] == "quest_advance_vetoed"
    ]
    assert [c.kwargs["guard"] for c in vetoes] == ["one_step_per_tick"]


def test_tick_charges_travel_time_per_leg_of_the_route(monkeypatch):
    state = _quest_world()
    state.locations["dock"] = Location(id="dock", connections=["harbor"])
    state.locations["harbor"].connections.append("dock")
    state.characters["hero"].location = "dock"

    async def go_far(*_args):
        return Travel(actor="hero", destination="ruins")

    async def no_review(*_args):
        return DMResult(creates=[], modifies=[], minutes=[1])

    monkeypatch.setattr("src.engine.runtime.loop._pick_next_actor", lambda *_: "hero")
    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", go_far)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", no_review)

    asyncio.run(tick("hero", state, 0, Mock()))

    assert state.characters["hero"].location == "ruins"
    assert state.history[-1].text == "hero travelled via harbor to 'ruins'."
    assert state.minutes_elapsed == 20


# ============ ENDLESS RUN ============


class _Run:
    """Fake persistence and logging around _run_game; records what the loop did."""

    def __init__(self, monkeypatch, state: WorldState, *, latest: str | None = None):
        self.events: list[tuple[str, dict]] = []
        self.saves: list[int] = []
        self.closed: list[dict] = []
        self.turns: list[int] = []
        run = self

        class Manager:
            run_id = "test-run"
            scenario = "demo"
            state_dir = None  # compaction is stubbed out
            manifest = {"pc": "hero", "title": "Test Quest", "hook": "Go."}

            def __init__(self, **_kwargs):
                pass

            def init_state(self):
                return state

            def latest_snapshot_name(self):
                return latest

            def load_state(self, snapshot):
                assert snapshot == latest
                return state

            def save_state(self, world):
                run.saves.append(world.time)

        class Log:
            def __init__(self, **_kwargs):
                pass

            def log_event(self, event, **kwargs):
                run.events.append((event, kwargs))

            def log_turn(self, turn):
                run.turns.append(turn)

            def close(self, **kwargs):
                run.closed.append(kwargs)

        async def skip(*_args):
            pass

        monkeypatch.setattr("src.engine.runtime.loop.StateManager", Manager)
        monkeypatch.setattr("src.engine.runtime.loop.Logger", Log)
        monkeypatch.setattr("src.engine.runtime.loop.compact_history", skip)
        monkeypatch.setattr("src.engine.runtime.loop.campaign_upkeep", skip)
        monkeypatch.setattr("src.engine.runtime.loop.FAILED_TICK_BACKOFF_S", 0)

    def names(self) -> list[str]:
        return [event for event, _ in self.events]


def test_resumed_game_logs_next_world_turn_instead_of_restarting_at_one(monkeypatch):
    state = _state()
    state.time = 7
    tick_indexes: list[int] = []
    run = _Run(monkeypatch, state, latest="world_state_7.json")

    async def complete_turn(_pc_id, _world, tick_index, *_args):
        tick_indexes.append(tick_index)
        return True

    monkeypatch.setattr("src.engine.runtime.loop.tick", complete_turn)
    asyncio.run(_run_game("demo", "hero", max_turns=1, new_character=None, resume=True))

    assert run.turns == [8]
    assert tick_indexes == [7]
    assert run.closed == [{"turns_completed": 8}]


def test_resume_continues_with_the_persisted_protagonist(monkeypatch):
    state = _state()
    state.characters["heir"] = Character(id="heir", location="tavern")
    state.protagonist = "heir"
    actors: list[str] = []
    _Run(monkeypatch, state, latest="world_state_0.json")

    async def complete_turn(pc_id, *_args):
        actors.append(pc_id)
        return True

    monkeypatch.setattr("src.engine.runtime.loop.tick", complete_turn)
    asyncio.run(_run_game("demo", None, max_turns=1, new_character=None, resume=True))

    assert actors == ["heir"]


def test_campaign_continues_after_every_quest_is_resolved(monkeypatch):
    state = _state()
    ticks: list[int] = []
    run = _Run(monkeypatch, state)

    async def finish_quest(_pc_id, world, tick_index, *_args):
        ticks.append(tick_index)
        world.quests["q1"].status = "failed"
        return True

    monkeypatch.setattr("src.engine.runtime.loop.tick", finish_quest)
    assert asyncio.run(_run_game("demo", "hero", max_turns=3, new_character=None))

    assert ticks == [0, 1, 2]
    assert run.saves == [0, 1, 2, 3]


def test_zero_turns_runs_until_stopped(monkeypatch):
    state = _state()
    run = _Run(monkeypatch, state)

    async def complete_turn(*_args):
        return True

    monkeypatch.setattr("src.engine.runtime.loop.tick", complete_turn)
    asyncio.run(
        _run_game(
            "demo",
            "hero",
            max_turns=0,
            new_character=None,
            stop_check=lambda: state.time >= 40,
        )
    )

    assert run.saves[-1] == 40
    assert ("run_stopped", {}) in run.events


def test_pc_death_hands_the_story_to_a_successor(monkeypatch):
    state = _state()
    state.characters["hero"].relationships["squire"] = "loyal squire"
    state.characters["squire"] = Character(id="squire", location="tavern")
    run = _Run(monkeypatch, state)

    async def lethal_turn(_pc_id, world, *_args):
        world.characters["hero"].stats.hp = 0
        return True

    monkeypatch.setattr("src.engine.runtime.loop.tick", lethal_turn)
    asyncio.run(_run_game("demo", "hero", max_turns=1, new_character=None))

    assert ("pc_death", {"pc": "hero"}) in run.events
    assert ("successor", {"pc": "squire", "previous": "hero"}) in run.events
    assert state.protagonist == "squire"
    assert state.quests["q1"].owner == "squire"
    assert run.closed == [{"turns_completed": 1}]


def test_failed_tick_is_rolled_back_and_retried_with_the_next_actor(monkeypatch):
    state = _state()
    tick_indexes: list[int] = []
    run = _Run(monkeypatch, state)

    async def flaky_turn(_pc_id, world, tick_index, *_args):
        tick_indexes.append(tick_index)
        world.characters["hero"].stats.gold += 100  # partial work of the tick
        if len(tick_indexes) == 1:
            raise UnexpectedModelBehavior("Exceeded maximum retries (1)")
        return True

    monkeypatch.setattr("src.engine.runtime.loop.tick", flaky_turn)
    assert asyncio.run(_run_game("demo", "hero", max_turns=1, new_character=None))

    assert tick_indexes == [0, 1]
    assert state.characters["hero"].stats.gold == 100  # first attempt rolled back
    assert run.saves == [0, 1]
    assert (
        "tick_failed",
        {
            "error": "UnexpectedModelBehavior",
            "message": "Exceeded maximum retries (1)",
            "consecutive": 1,
        },
    ) in run.events
    assert "run_error" not in run.names()


@pytest.mark.parametrize(
    "error",
    [
        ModelAPIError("ollama", "Connection error"),
        UnexpectedModelBehavior("Tool exceeded max retries count of 3"),
    ],
)
def test_game_gives_up_only_after_consecutive_failed_ticks(monkeypatch, error):
    state = _state()
    run = _Run(monkeypatch, state)

    async def failing_turn(*_args):
        raise error

    monkeypatch.setattr("src.engine.runtime.loop.tick", failing_turn)
    succeeded = asyncio.run(_run_game("demo", "hero", max_turns=50, new_character=None))

    assert succeeded is False
    assert run.names().count("tick_failed") == MAX_FAILED_TICKS
    assert run.names().count("run_error") == 1
    assert run.saves == [0]
    assert run.closed == [{"turns_completed": 0}]


def test_turn_without_character_action_is_a_failed_tick(monkeypatch):
    state = _state()
    run = _Run(monkeypatch, state)

    async def no_character_action(*_args):
        return None

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", no_character_action)
    asyncio.run(_run_game("demo", "hero", max_turns=5, new_character=None))

    assert run.names().count("tick_failed") == MAX_FAILED_TICKS
    assert run.saves == [0]


def test_game_loop_logs_and_reraises_unexpected_post_turn_failure(monkeypatch):
    state = _state()
    run = _Run(monkeypatch, state)

    async def complete_turn(*_args):
        return True

    async def fail_compaction(*_args):
        raise RuntimeError("chronicle storage unavailable")

    monkeypatch.setattr("src.engine.runtime.loop.tick", complete_turn)
    monkeypatch.setattr("src.engine.runtime.loop.compact_history", fail_compaction)

    with pytest.raises(RuntimeError, match="chronicle storage unavailable"):
        asyncio.run(_run_game("demo", "hero", max_turns=5, new_character=None))

    assert run.names().count("run_error") == 1
    assert dict(run.events)["run_error"] == {
        "error": "RuntimeError",
        "message": "chronicle storage unavailable",
    }
    assert run.saves == [0]
    assert run.closed == [{"turns_completed": 0}]


def test_dm_quest_claim_vetoed_by_a_guard_is_logged(monkeypatch):
    state = _state()
    logger = Mock()

    async def resolve_wait(_intent, world, **_kwargs):
        world.history.append(_event("hero waits."))
        return "hero waits."

    async def claim_progress(*_args):
        return DMResult(
            creates=[],
            modifies=[
                Modify(
                    action="update_quest", target_id="q1", advance=True, step="scouted"
                )
            ],
            minutes=[5],
        )

    async def wait_turn(*_args):
        return Wait(actor="hero")

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", wait_turn)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_wait)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", claim_progress)

    asyncio.run(tick("hero", state, 0, logger))

    assert state.quests["q1"].current_step == 0
    logger.log_event.assert_any_call(
        "quest_advance_vetoed",
        guard="not_an_attempt",
        quest="q1",
        objective="scout",
        claim="scouted",
        status=None,
        actor="hero",
        tool="wait",
        events=["hero waits."],
    )


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
