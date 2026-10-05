import asyncio

import pytest
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior

from src.agents.character.tools import Action, Attack, Speak, Travel, Wait
from src.engine.runtime.loop import (
    _STALL_QUIET_TICKS,
    _advance_grounded_objective,
    _can_accelerate_faction_clock,
    _can_anchor_new_location,
    _can_anchor_new_npc,
    _can_advance_final_objective,
    _can_advance_objective,
    _campaign_outcome,
    _complete_resolved_final_objective,
    _is_stalled,
    _minimum_action_minutes,
    _pick_next_actor,
    _record_intervention,
    _record_resolution_if_needed,
    _run_game,
    _scene_actors,
    tick,
)
from src.agents.dm.agent import DMResult
from src.agents.dm.tools import Create, Modify
from src.engine.state.models import Character, Faction, HistoryEvent, Location, ProgressClock, Quest, WorldState
from src.engine.state.operations import WorldOperations


def _state() -> WorldState:
    return WorldState(
        locations={"tavern": Location(id="tavern")},
        characters={"hero": Character(id="hero", role="warrior", location="tavern")},
        quests={"q1": Quest(id="q1", title="Clear the Cave", description="", owner="hero", plan=["scout", "clear"])},
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

    assert not _can_anchor_new_npc(create, [_event(
        "hero learns: The Crimson Tide trading company is using the cove."
    )])


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

    class CapturingLogger:
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
            creates=[Create(
                type="npc",
                name="the-handler",
                description="An unknown shore contact.",
                location="village-square",
            )],
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
            creates=[Create(
                type="npc",
                name="Crimson Tide trading company",
                description="A smuggling faction operating out of the eastern ports.",
                location="tavern",
                role="smuggling faction",
            )],
            modifies=[],
            minutes=[2],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", inspect_crate)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_crate)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", misclassify_company)

    asyncio.run(tick("hero", state, 0, logger=None))

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
            creates=[Create(
                type="location",
                name="upriver-location",
                description="An unspecified upriver source of cargo.",
                location="tavern",
            )],
            modifies=[],
            minutes=[5],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", inspect_manifest)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_manifest)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", invent_placeholder)

    asyncio.run(tick("hero", state, 0, logger=None))

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
            creates=[Create(
                type="npc",
                name="Calla",
                description="A scullion mentioned on an earlier turn.",
                location="servants-quarters",
            )],
            modifies=[],
            minutes=[10],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", travel_to_servants_quarters)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", reveal_calla_from_prior_context)

    asyncio.run(tick("hero", state, 0, logger=None))

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
            creates=[Create(
                type="npc",
                name="Calla",
                description="A new scullion who was asking about the vault.",
                location="manor-foyer",
            )],
            modifies=[],
            minutes=[10, 1],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", travel_to_foyer)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", place_calla_at_foyer)

    asyncio.run(tick("hero", state, 0, logger=None))

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

    asyncio.run(tick("hero", state, 0, logger=None))

    assert set(state.characters) == {"hero"}
    assert [event.text for event in state.history] == [f"hero learns: {roster_fact}"]


def test_tick_skips_dm_item_already_created_by_the_action(monkeypatch):
    state = _state()

    async def search_office(*_args):
        return Action(actor="hero", description="search the dockmaster's office")

    async def create_journal(tool, world, **_kwargs):
        if isinstance(tool, Action):
            return WorldOperations(world).create_item("Alan's leather journal", "tavern")
        raise AssertionError("DM enrichment must not recreate an item already in the world")

    async def repeat_journal(*_args):
        return DMResult(
            creates=[Create(
                type="item",
                name="Alan's leather journal",
                description="A journal revealed by the resolved search.",
                location="tavern",
            )],
            modifies=[],
            minutes=[5],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", search_office)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", create_journal)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", repeat_journal)

    asyncio.run(tick("hero", state, 0, logger=None))

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
    state.history.append(HistoryEvent(
        text="dockmaster's relationship with hero is now 'wary'.",
        location="tavern",
        characters=["dockmaster", "hero"],
    ))

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


def test_only_active_interactions_can_accelerate_faction_clocks():
    state = _state()
    state.characters["guild-guard"] = Character(id="guild-guard", role="guild sentry", location="tavern")
    state.factions["guild"] = Faction(
        id="guild",
        name="The Guild",
        goal="Control the harbor",
        clocks=[ProgressClock(
            id="alarm",
            name="Raise the alarm",
            consequence="The gates close.",
            segments=4,
        )],
    )

    detected = [_event("A guild lookout spots hero and warns the sentries.")]
    direct = [HistoryEvent(
        text='hero says to guild-guard: "Sound the alarm."',
        location="tavern",
        characters=["hero", "guild-guard"],
    )]

    def can_accelerate(tool, events=detected):
        return _can_accelerate_faction_clock(state, tool, "guild", "alarm", events)

    assert can_accelerate(Speak(actor="hero", target="guild-guard", message="sound the alarm"), direct)
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
        clocks=[ProgressClock(
            id="retaliation",
            name="Prepare retaliation",
            consequence="The crew attacks the village.",
            segments=8,
        )],
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
        clocks=[ProgressClock(
            id="retaliation",
            name="Prepare retaliation",
            consequence="The crew attacks the village.",
            segments=8,
        )],
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
        clocks=[ProgressClock(
            id="retaliation",
            name="Prepare retaliation",
            consequence="The crew attacks the village.",
            segments=8,
        )],
    )
    event = _event("A Black-Hull Crew lookout notices hero searching the cave.")

    assert _can_accelerate_faction_clock(
        state,
        Action(actor="hero", description="search the smugglers' cave"),
        "black-hull-crew",
        "retaliation",
        [event],
    )


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


def test_intermediate_objective_requires_relevant_owner_attempt():
    state = _state()
    quest = state.quests["q1"]
    quest.plan = ["question the tavern patrons about the cave", "clear the cave"]

    relevant_event = _event("hero learns: a patron saw torchlight near the cave.")
    assert _can_advance_objective(
        state,
        quest,
        Speak(actor="hero", target="patron", message="What did you see near the cave?"),
        [relevant_event],
    )
    assert not _can_advance_objective(
        state,
        quest,
        Travel(actor="hero", destination="cave"),
        [_event("hero moved to 'cave'.")],
    )
    assert not _can_advance_objective(
        state,
        quest,
        Action(actor="hero", description="search the cellar for supplies"),
        [_event("hero finds a coil of rope in the cellar.")],
    )
    assert not _can_advance_objective(
        state,
        quest,
        Speak(actor="hero", target="patron", message="What did you see near the cave?"),
        [_event("hero finds no useful information from the patron.")],
    )


def test_dialogue_objective_requires_a_dialogue_attempt():
    state = _state()
    quest = state.quests["q1"]
    quest.plan = ["question the servants about the circlet", "recover the circlet"]
    roster = _event("hero learns: The servants are Kae, Marta, Thorne, and Colm.")

    assert not _can_advance_objective(
        state,
        quest,
        Action(
            actor="hero",
            description="inspect the servants' roster to see who might know about the circlet",
        ),
        [roster],
    )
    assert _can_advance_objective(
        state,
        quest,
        Action(actor="hero", description="ask the gathered servants about the circlet"),
        [_event("hero learns: Marta saw the circlet in the upper library.")],
    )


def test_dialogue_objective_requires_the_message_to_name_a_quest_topic():
    state = _state()
    state.locations["servants-quarters"] = Location(id="servants-quarters")
    state.characters["hero"].location = "servants-quarters"
    state.characters["marta"] = Character(
        id="marta",
        role="laundress",
        location="servants-quarters",
    )
    quest = state.quests["q1"]
    quest.title = "The Oak Circlet"
    quest.description = "Recover the Thorne family's Oak Circlet before dawn."
    quest.plan = ["question the servants about the circlet", "recover the circlet"]

    def speech(message: str) -> tuple[Speak, list[HistoryEvent]]:
        return (
            Speak(actor="hero", target="marta", message=message),
            [HistoryEvent(
                text=f'hero says to marta: "{message}"',
                location="servants-quarters",
                characters=["hero", "marta"],
            )],
        )

    unrelated_tool, unrelated_events = speech("How is the weather?")
    assert not _can_advance_objective(state, quest, unrelated_tool, unrelated_events)

    relevant_tool, relevant_events = speech(
        "The Oak Circlet is missing. Have you noticed anything unusual?"
    )
    assert _can_advance_objective(state, quest, relevant_tool, relevant_events)


def test_relevant_npc_reply_can_complete_the_owners_dialogue_objective():
    state = _state()
    state.characters["calla"] = Character(
        id="calla",
        role="scullion",
        location="tavern",
    )
    quest = state.quests["q1"]
    quest.title = "Investigate the Vault"
    quest.description = "Learn what Calla knows about the vault behind the cold hearth."
    quest.plan = [
        "determine what Calla knows and her motives",
        "decide whether to investigate the vault",
    ]
    question = HistoryEvent(
        text=(
            'hero says to calla: "What do you know about the vault behind the '
            'cold hearth, and why are you looking for it?"'
        ),
        location="tavern",
        characters=["hero", "calla"],
    )
    reply = HistoryEvent(
        text=(
            'calla says to hero: "I have heard it is hidden behind the cold hearth. '
            'I want to learn what the vault contains."'
        ),
        location="tavern",
        characters=["calla", "hero"],
    )
    state.history = [question, reply]

    assert _can_advance_objective(
        state,
        quest,
        Speak(
            actor="calla",
            target="hero",
            message=(
                "I have heard it is hidden behind the cold hearth. "
                "I want to learn what the vault contains."
            ),
        ),
        [reply],
    )


def test_npc_reply_requires_a_relevant_owner_question_and_relevant_answer():
    state = _state()
    state.characters["calla"] = Character(id="calla", location="tavern")
    quest = state.quests["q1"]
    quest.title = "Investigate the Vault"
    quest.description = "Learn what Calla knows about the vault behind the cold hearth."
    quest.plan = [
        "determine what Calla knows and her motives",
        "decide whether to investigate the vault",
    ]
    reply_tool = Speak(
        actor="calla",
        target="hero",
        message="I heard the vault is hidden behind the cold hearth.",
    )
    reply = HistoryEvent(
        text='calla says to hero: "I heard the vault is hidden behind the cold hearth."',
        location="tavern",
        characters=["calla", "hero"],
    )
    unrelated_question = HistoryEvent(
        text='hero says to calla: "How is the weather?"',
        location="tavern",
        characters=["hero", "calla"],
    )
    state.history = [unrelated_question, reply]

    assert not _can_advance_objective(state, quest, reply_tool, [reply])

    relevant_question = HistoryEvent(
        text='hero says to calla: "What do you know about the vault behind the cold hearth?"',
        location="tavern",
        characters=["hero", "calla"],
    )
    unrelated_reply = HistoryEvent(
        text='calla says to hero: "The weather has been miserable all week."',
        location="tavern",
        characters=["calla", "hero"],
    )
    state.history = [relevant_question, unrelated_reply]

    assert not _can_advance_objective(
        state,
        quest,
        Speak(
            actor="calla",
            target="hero",
            message="The weather has been miserable all week.",
        ),
        [unrelated_reply],
    )

    progress = HistoryEvent(
        text="hero earns 10 XP (quest 'q1' progress).",
        location="tavern",
        characters=["hero"],
    )
    state.history = [relevant_question, progress, reply]

    assert not _can_advance_objective(state, quest, reply_tool, [reply])


def test_tick_advances_dialogue_objective_from_the_addressed_npcs_reply(monkeypatch):
    state = _state()
    state.characters["calla"] = Character(
        id="calla",
        role="scullion",
        location="tavern",
    )
    quest = state.quests["q1"]
    quest.title = "Investigate the Vault"
    quest.description = "Learn what Calla knows about the vault behind the cold hearth."
    quest.plan = [
        "determine what Calla knows and her motives",
        "decide whether to investigate the vault",
    ]
    state.history = [HistoryEvent(
        text=(
            'hero says to calla: "What do you know about the vault behind the '
            'cold hearth, and why are you looking for it?"'
        ),
        location="tavern",
        characters=["hero", "calla"],
    )]

    async def answer_question(*_args):
        return Speak(
            actor="calla",
            target="hero",
            message=(
                "I have heard it is hidden behind the cold hearth. "
                "I want to learn what the vault contains."
            ),
        )

    async def report_progress(*_args):
        return DMResult(
            creates=[],
            modifies=[Modify(
                action="update_quest",
                target_id="q1",
                advance=True,
                step="Calla revealed what she knows about the vault and why she is seeking it",
            )],
            minutes=[2],
        )

    monkeypatch.setattr("src.engine.runtime.loop._pick_next_actor", lambda *_args: "calla")
    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", answer_question)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_progress)

    asyncio.run(tick("hero", state, 1, logger=None))

    assert quest.current_step == 1
    assert quest.steps == [
        "determine what Calla knows and her motives — "
        "Calla revealed what she knows about the vault and why she is seeking it",
    ]
    assert state.characters["hero"].stats.xp == 10


def test_location_bound_objective_requires_evidence_at_a_named_location():
    state = _state()
    state.locations.update({
        "guild-hall": Location(id="guild-hall"),
        "trade-dock": Location(id="trade-dock"),
    })
    quest = state.quests["q1"]
    quest.plan = ["check the guild-hall ledger for anything unusual", "find the missing guard"]
    tool = Action(actor="hero", description="inspect the cargo ledger for unusual shipments")

    state.characters["hero"].location = "trade-dock"
    remote_evidence = HistoryEvent(
        text="hero learns: The cargo ledger lists an unusual private shipment.",
        location="trade-dock",
        characters=["hero"],
    )
    assert not _can_advance_objective(state, quest, tool, [remote_evidence])

    state.characters["hero"].location = "guild-hall"
    local_evidence = remote_evidence.model_copy(update={"location": "guild-hall"})
    assert _can_advance_objective(state, quest, tool, [local_evidence])


def test_referential_contact_objective_accepts_a_known_quest_link():
    state = _state()
    quest = state.quests["q1"]
    quest.title = "The Missing Brother"
    quest.description = "Find clues about the disappearance of Kaelen Swift."
    quest.plan = ["track down whoever saw him last", "find Kaelen Swift"]
    state.characters["hero"].knowledge = [
        "Dockmaster Alan met Kaelen at the bridge on the day he vanished."
    ]
    state.characters["dockmaster-alan"] = Character(id="dockmaster-alan", location="tavern")
    tool = Speak(
        actor="hero",
        target="dockmaster-alan",
        message="You met my brother Kaelen at the bridge. What happened to him?",
    )
    event = HistoryEvent(
        text='hero says to dockmaster-alan: "You met my brother Kaelen at the bridge. What happened?"',
        location="tavern",
        characters=["hero", "dockmaster-alan"],
    )

    assert _can_advance_objective(state, quest, tool, [event])


def test_referential_contact_objective_rejects_an_ungrounded_conversation():
    state = _state()
    quest = state.quests["q1"]
    quest.title = "The Missing Brother"
    quest.description = "Find clues about the disappearance of Kaelen Swift."
    quest.plan = ["track down whoever saw him last", "find Kaelen Swift"]
    state.characters["hero"].knowledge = [
        "Dockmaster Alan met Kaelen at the bridge on the day he vanished."
    ]
    state.characters["dockmaster-alan"] = Character(id="dockmaster-alan", location="tavern")
    event = HistoryEvent(
        text='hero says to dockmaster-alan: "How is the weather?"',
        location="tavern",
        characters=["hero", "dockmaster-alan"],
    )

    unrelated = Speak(actor="hero", target="dockmaster-alan", message="How is the weather?")
    assert not _can_advance_objective(state, quest, unrelated, [event])

    related = Speak(actor="hero", target="dockmaster-alan", message="What happened to Kaelen?")
    state.characters["hero"].knowledge = ["Dockmaster Alan handles the port's cargo."]
    assert not _can_advance_objective(state, quest, related, [event])


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


@pytest.mark.parametrize("quest_id", ["q1", "Q1"])
def test_tick_rejects_direct_final_completion_from_clue_discovery(monkeypatch, quest_id):
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
            modifies=[Modify(action="update_quest", target_id=quest_id, status="completed")],
            minutes=[1],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_intent)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_clue)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert quest.status == "active"
    assert quest.current_step == len(quest.plan) - 1
    assert state.characters["hero"].stats.xp == 0


def test_tick_rejects_intermediate_progress_from_unrelated_action(monkeypatch):
    state = _state()
    quest = state.quests["q1"]
    quest.plan = ["question the tavern patrons about the cave", "clear the cave"]

    async def choose_action(*_args):
        return Action(actor="hero", description="search the cellar for supplies")

    async def resolve_intent(tool, world, **_kwargs):
        if isinstance(tool, Action):
            event = _event("hero finds a coil of rope in the cellar.")
            world.history.append(event)
            return event.text
        raise AssertionError("the blocked quest update must not reach the resolver")

    async def report_progress(*_args):
        return DMResult(
            creates=[],
            modifies=[Modify(action="update_quest", target_id="q1", advance=True)],
            minutes=[5],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_intent)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_progress)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert quest.current_step == 0
    assert quest.steps == []
    assert state.characters["hero"].stats.xp == 0


def test_tick_rejects_dialogue_progress_from_inspecting_a_roster(monkeypatch):
    state = _state()
    quest = state.quests["q1"]
    quest.plan = ["question the servants about the circlet", "recover the circlet"]

    async def inspect_roster(*_args):
        return Action(
            actor="hero",
            description="inspect the servants' roster to see who might know about the circlet",
        )

    async def resolve_roster(tool, world, **_kwargs):
        if isinstance(tool, Action):
            event = _event("hero learns: The servants are Kae, Marta, Thorne, and Colm.")
            world.history.append(event)
            return event.text
        raise AssertionError("the blocked quest update must not reach the resolver")

    async def report_progress(*_args):
        return DMResult(
            creates=[],
            modifies=[Modify(
                action="update_quest",
                target_id="q1",
                advance=True,
                step="identified the servants in the manor",
            )],
            minutes=[3],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", inspect_roster)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_roster)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_progress)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert quest.current_step == 0
    assert quest.steps == []
    assert state.characters["hero"].stats.xp == 0


def test_tick_rejects_location_bound_progress_from_a_different_location(monkeypatch):
    state = _state()
    state.locations.update({
        "guild-hall": Location(id="guild-hall"),
        "trade-dock": Location(id="trade-dock"),
    })
    state.characters["hero"].location = "trade-dock"
    quest = state.quests["q1"]
    quest.plan = ["check the guild-hall ledger for anything unusual", "find the missing guard"]

    async def inspect_trade_dock_ledger(*_args):
        return Action(actor="hero", description="inspect the cargo ledger for unusual shipments")

    async def resolve_search(tool, world, **_kwargs):
        if isinstance(tool, Action):
            event = HistoryEvent(
                text="hero learns: The cargo ledger lists an unusual private shipment.",
                location="trade-dock",
                characters=["hero"],
            )
            world.history.append(event)
            return event.text
        raise AssertionError("the blocked quest update must not reach the resolver")

    async def report_progress(*_args):
        return DMResult(
            creates=[],
            modifies=[Modify(
                action="update_quest",
                target_id="q1",
                advance=True,
                step="found an unusual private shipment in the cargo ledger",
            )],
            minutes=[10],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", inspect_trade_dock_ledger)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_search)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_progress)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert quest.current_step == 0
    assert quest.steps == []
    assert state.characters["hero"].stats.xp == 0


@pytest.mark.parametrize(
    "negative_finding",
    [
        "contains no direct reference to the Oak Circlet or its location",
        "contains family lineage records but no direct information about the Oak Circlet's hiding place or history",
    ],
)
def test_tick_rejects_search_progress_from_negative_clue_result(monkeypatch, negative_finding):
    state = _state()
    quest = state.quests["q1"]
    quest.title = "The Oak Circlet"
    quest.description = "Recover the Thorne family's Oak Circlet before dawn."
    quest.plan = [
        "search the tavern for clues about the circlet's whereabouts",
        "recover the Oak Circlet before dawn",
    ]

    async def examine_genealogy(*_args):
        return Action(
            actor="hero",
            description="examine the leather-bound genealogy for the Oak Circlet's location",
        )

    async def resolve_search(tool, world, **_kwargs):
        if isinstance(tool, Action):
            event = _event(
                f"hero learns: The leather-bound genealogy {negative_finding}."
            )
            world.history.append(event)
            return event.text
        raise AssertionError("the rejected quest update must not reach the resolver")

    async def report_progress(*_args):
        return DMResult(
            creates=[],
            modifies=[Modify(
                action="update_quest",
                target_id="q1",
                advance=True,
                step=f"genealogy {negative_finding}",
            )],
            minutes=[15],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", examine_genealogy)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_search)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_progress)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert quest.current_step == 0
    assert quest.steps == []
    assert state.characters["hero"].stats.xp == 0


def test_tick_binds_tool_to_scheduled_actor(monkeypatch):
    state = _state()
    state.characters["ally"] = Character(id="ally", role="scout", location="tavern")

    async def choose_action(*_args):
        return Speak(actor="ally", message="The wrong actor must not control this turn.")

    async def report_turn(*_args):
        return DMResult(creates=[], modifies=[], minutes=[1])

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_turn)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert state.history[-1].text == 'hero says: "The wrong actor must not control this turn."'
    assert state.history[-1].characters == ["hero"]


def test_tick_ignores_minute_estimates_without_matching_events(monkeypatch):
    state = _state()

    async def choose_action(*_args):
        return Speak(actor="hero", message="One brief warning.")

    async def report_turn(*_args):
        return DMResult(creates=[], modifies=[], minutes=[2, 60])

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_turn)

    asyncio.run(tick("hero", state, 0, logger=None))

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

    asyncio.run(tick("hero", state, 0, logger=None))

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

    asyncio.run(tick("hero", state, 0, logger=None))

    assert [event.minutes_elapsed for event in state.history] == [2, 0]
    assert state.minutes_elapsed == 2


def test_tick_rejects_dm_removal_of_active_player_character(monkeypatch):
    state = _state()

    async def choose_action(*_args):
        return Speak(actor="hero", message="I am still here.")

    async def remove_player(*_args):
        return DMResult(
            creates=[],
            modifies=[Modify(action="remove_npc", target_id="HERO", reason="Disappears.")],
            minutes=[1],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", remove_player)

    asyncio.run(tick("hero", state, 0, logger=None))

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
        clocks=[ProgressClock(
            id="dawn-breaks",
            name="Dawn breaks",
            consequence="The curse becomes permanent.",
            segments=6,
        )],
    )

    async def choose_travel(*_args):
        return Travel(actor="hero", destination="servants-quarters")

    async def accelerate_dawn(*_args):
        return DMResult(
            creates=[],
            modifies=[Modify(
                action="advance_faction_clock",
                target_id="encroaching-dawn",
                other_id="dawn-breaks",
            )],
            minutes=[10],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_travel)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", accelerate_dawn)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert state.characters["hero"].location == "servants-quarters"
    assert state.factions["encroaching-dawn"].clocks[0].progress == 0


def test_tick_rejects_event_acceleration_for_elapsed_time_deadline(monkeypatch):
    state = _state()
    state.factions["encroaching-dawn"] = Faction(
        id="encroaching-dawn",
        name="The Encroaching Dawn",
        goal="End the search before the curse can be broken",
        clocks=[ProgressClock(
            id="dawn-breaks",
            name="Dawn breaks",
            consequence="The curse becomes permanent.",
            segments=6,
            event_acceleration=False,
        )],
    )

    async def discuss_search(*_args):
        return Speak(actor="hero", message="We should keep searching before dawn.")

    async def accelerate_dawn(*_args):
        return DMResult(
            creates=[],
            modifies=[Modify(
                action="advance_faction_clock",
                target_id="encroaching-dawn",
                other_id="dawn-breaks",
            )],
            minutes=[2],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", discuss_search)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", accelerate_dawn)

    asyncio.run(tick("hero", state, 0, logger=None))

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
        clocks=[ProgressClock(
            id="retaliation",
            name="Prepare retaliation",
            consequence="The crew attacks.",
            segments=4,
        )],
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
            modifies=[Modify(
                action="advance_faction_clock",
                target_id="crew",
                other_id="retaliation",
            )],
            minutes=[2],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", confront_crew)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", accelerate_retaliation)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert state.factions["crew"].clocks[0].progress == 1


def test_tick_rejects_faction_acceleration_for_unobserved_evidence_discovery(monkeypatch):
    state = _state()
    logged_events = []

    class CapturingLogger:
        def log_event(self, event, **kwargs):
            logged_events.append((event, kwargs))

    state.factions["black-hull-crew"] = Faction(
        id="black-hull-crew",
        name="The Black-Hull Crew",
        goal="Move contraband and silence witnesses",
        clocks=[ProgressClock(
            id="retaliation",
            name="Prepare retaliation",
            consequence="The crew attacks the village.",
            segments=8,
        )],
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
            modifies=[Modify(
                action="advance_faction_clock",
                target_id="black-hull-crew",
                other_id="retaliation",
            )],
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


def test_compound_final_objective_accepts_grounded_direct_confrontation():
    quest = _state().quests["q1"]
    quest.plan[-1] = "gather proof and confront or report the smugglers"
    quest.current_step = len(quest.plan) - 1
    tool = Speak(
        actor="hero",
        target="captain-vess-rann",
        message="I have your ship's log and the cave ledgers.",
    )
    event = HistoryEvent(
        text='hero says to captain-vess-rann: "I have your ship\'s log and the cave ledgers."',
        location="tavern",
        characters=["hero", "captain-vess-rann"],
    )

    assert _can_advance_final_objective(
        quest,
        tool,
        [event],
        "confronted Captain Vess Rann directly with proof from the ship's log and ledgers",
    )
    assert not _can_advance_final_objective(
        quest,
        tool,
        [event],
        "confronted Captain Vess Rann about the weather",
    )
    unrelated_tool = tool.model_copy(update={"message": "Fine weather today."})
    unrelated_event = event.model_copy(update={
        "text": 'hero says to captain-vess-rann: "Fine weather today."',
    })
    assert not _can_advance_final_objective(
        quest,
        unrelated_tool,
        [unrelated_event],
        "confronted Captain Vess Rann directly with proof from the ship's log and ledgers",
    )


def test_compound_final_objective_matches_punctuated_target_initials():
    quest = _state().quests["q1"]
    quest.plan[-1] = "gather proof and confront or report the smugglers"
    quest.current_step = len(quest.plan) - 1
    tool = Speak(
        actor="hero",
        target="vm",
        message="The ship's log proves you signed off on the stolen goods.",
    )
    event = HistoryEvent(
        text='hero says to vm: "The ship\'s log proves you signed off on the stolen goods."',
        location="tavern",
        characters=["hero", "vm"],
    )

    assert _can_advance_final_objective(
        quest,
        tool,
        [event],
        "confronted V.M. with the ship's log—proof of stolen goods signed by her",
    )
    assert not _can_advance_final_objective(
        quest,
        tool,
        [event],
        "confronted V.A. with the ship's log—proof of stolen goods signed by her",
    )


def test_tick_completes_final_confrontation_with_punctuated_target_initials(monkeypatch):
    state = _state()
    state.characters["vm"] = Character(id="vm", location="tavern")
    quest = state.quests["q1"]
    quest.plan = ["gather proof and confront or report the smugglers"]

    async def confront_smuggler(*_args):
        return Speak(
            actor="hero",
            target="vm",
            message="The ship's log proves you signed off on the stolen goods.",
        )

    async def report_confrontation(*_args):
        return DMResult(
            creates=[],
            modifies=[Modify(
                action="update_quest",
                target_id="q1",
                advance=True,
                step="confronted V.M. with the ship's log—proof of stolen goods signed by her",
            )],
            minutes=[2],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", confront_smuggler)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_confrontation)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert quest.status == "completed"
    assert quest.current_step == len(quest.plan)
    assert state.characters["hero"].stats.xp == 60


def test_tick_completes_speech_final_objective_from_grounded_dm_step(monkeypatch):
    state = _state()
    state.characters["captain-vess-rann"] = Character(id="captain-vess-rann", location="tavern")
    quest = state.quests["q1"]
    quest.plan = ["gather proof and confront or report the smugglers"]

    async def choose_action(*_args):
        return Speak(
            actor="hero",
            target="captain-vess-rann",
            message="I have your ship's log and the cave ledgers.",
        )

    async def report_confrontation(*_args):
        return DMResult(
            creates=[],
            modifies=[Modify(
                action="update_quest",
                target_id="q1",
                advance=True,
                step="confronted Captain Vess Rann directly with proof from the ship's log and ledgers",
            )],
            minutes=[2],
        )

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", report_confrontation)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert quest.status == "completed"
    assert quest.current_step == len(quest.plan)
    assert state.characters["hero"].stats.xp == 60


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


def test_direct_final_resolution_bypasses_intermediate_objectives_without_their_xp():
    state = _state()
    quest = state.quests["q1"]
    quest.title = "The Oak Circlet"
    quest.description = "Recover the Thorne family's Oak Circlet before dawn."
    quest.plan = [
        "search the upper-library for clues about the circlet's whereabouts",
        "question the servants about the circlet",
        "search the manor-foyer and family portraits for hidden clues",
        "recover the Oak Circlet before dawn",
    ]
    event = _event("hero picks up the Oak Circlet.")

    results = _complete_resolved_final_objective(
        state,
        "hero",
        Action(actor="hero", description="take the Oak Circlet"),
        [event],
    )

    assert quest.status == "completed"
    assert quest.current_step == len(quest.plan)
    assert quest.steps == [
        f"{objective} — bypassed by direct resolution"
        for objective in quest.plan[:-1]
    ] + ["recover the Oak Circlet before dawn — resolved directly"]
    assert state.characters["hero"].stats.xp == 60
    assert "+60 XP to hero" in results[0]
    WorldState.model_validate(state.model_dump())


def test_tick_completes_final_objective_when_dm_omits_quest_update(monkeypatch):
    state = _state()
    quest = state.quests["q1"]
    quest.plan = ["search the manor", "recover the Oak Circlet before dawn"]

    async def choose_action(*_args):
        return Action(actor="hero", description="take the Oak Circlet")

    async def resolve_intent(tool, world, **_kwargs):
        if isinstance(tool, Action):
            event = _event("hero picks up the Oak Circlet.")
            world.history.append(event)
            return event.text
        raise AssertionError("no DM updates should need resolution")

    async def omit_progress(*_args):
        return DMResult(creates=[], modifies=[], minutes=[1])

    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", choose_action)
    monkeypatch.setattr("src.engine.runtime.loop.resolve", resolve_intent)
    monkeypatch.setattr("src.engine.runtime.loop.flow_dm", omit_progress)

    asyncio.run(tick("hero", state, 0, logger=None))

    assert quest.status == "completed"
    assert quest.current_step == len(quest.plan)
    assert state.characters["hero"].stats.xp == 60


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


def test_resumed_game_logs_next_world_turn_instead_of_restarting_at_one(monkeypatch):
    state = _state()
    state.time = 7
    logged_turns: list[int] = []
    tick_indexes: list[int] = []
    closed: list[dict] = []

    class Manager:
        run_id = "resumed-run"
        scenario = "demo"
        manifest = {"pc": "hero", "title": "Test Quest", "hook": "Continue it."}

        def __init__(self, **_kwargs): pass
        def latest_snapshot_name(self): return "world_state_7.json"
        def load_state(self, snapshot):
            assert snapshot == "world_state_7.json"
            return state
        def save_state(self, _state): pass

    class Log:
        def __init__(self, **_kwargs): pass
        def log_event(self, _event, **_kwargs): pass
        def log_turn(self, turn): logged_turns.append(turn)
        def close(self, **kwargs): closed.append(kwargs)

    async def complete_turn(_pc_id, _world, tick_index, _logger, _controller, _replay):
        tick_indexes.append(tick_index)
        return True

    async def skip_compaction(*_args): pass

    monkeypatch.setattr("src.engine.runtime.loop.StateManager", Manager)
    monkeypatch.setattr("src.engine.runtime.loop.Logger", Log)
    monkeypatch.setattr("src.engine.runtime.loop.tick", complete_turn)
    monkeypatch.setattr("src.engine.runtime.loop.compact_history", skip_compaction)

    asyncio.run(_run_game("demo", "hero", max_turns=1, new_character=None, resume=True))

    assert logged_turns == [8]
    assert tick_indexes == [7]
    assert closed == [{"turns_completed": 8}]


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

    succeeded = asyncio.run(_run_game("demo", "hero", max_turns=5, new_character=None))

    assert succeeded is True
    assert ticks == [0]
    assert "campaign_completed" in logged_events


def test_game_loop_records_pc_death_on_final_turn(monkeypatch):
    state = _state()
    logged_events: list[tuple[str, dict]] = []
    closed: list[dict] = []

    class Manager:
        run_id = "test-run"
        scenario = "demo"
        manifest = {"pc": "hero", "title": "Test Quest", "hook": "Survive it."}

        def __init__(self, **_kwargs): pass
        def init_state(self): return state
        def latest_snapshot_name(self): return None
        def save_state(self, _state): pass

    class Log:
        def __init__(self, **_kwargs): pass
        def log_event(self, event, **kwargs): logged_events.append((event, kwargs))
        def log_turn(self, _turn): pass
        def close(self, **kwargs): closed.append(kwargs)

    async def lethal_turn(_pc_id, world, _tick_index, _logger, _controller, _replay):
        world.characters["hero"].stats.hp = 0
        return True

    async def skip_compaction(*_args): pass

    monkeypatch.setattr("src.engine.runtime.loop.StateManager", Manager)
    monkeypatch.setattr("src.engine.runtime.loop.Logger", Log)
    monkeypatch.setattr("src.engine.runtime.loop.tick", lethal_turn)
    monkeypatch.setattr("src.engine.runtime.loop.compact_history", skip_compaction)

    asyncio.run(_run_game("demo", "hero", max_turns=1, new_character=None))

    assert ("pc_death", {"pc": "hero"}) in logged_events
    assert closed == [{"turns_completed": 1}]


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

    succeeded = asyncio.run(_run_game("demo", "hero", max_turns=5, new_character=None))

    assert succeeded is False
    assert [event for event, _ in logged_events].count("run_error") == 1
    assert dict(logged_events)["run_error"] == {
        "error": "ModelAPIError",
        "message": "Connection error",
    }
    assert saves == [0]
    assert closed == [{"turns_completed": 0}]


def test_game_loop_records_retry_exhaustion_without_persisting_partial_turn(monkeypatch):
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

    async def exhaust_retries(*_args):
        raise UnexpectedModelBehavior("Tool exceeded max retries count of 3")

    monkeypatch.setattr("src.engine.runtime.loop.StateManager", Manager)
    monkeypatch.setattr("src.engine.runtime.loop.Logger", Log)
    monkeypatch.setattr("src.engine.runtime.loop.tick", exhaust_retries)

    succeeded = asyncio.run(_run_game("demo", "hero", max_turns=5, new_character=None))

    assert succeeded is False
    assert [event for event, _ in logged_events].count("run_error") == 1
    assert dict(logged_events)["run_error"] == {
        "error": "UnexpectedModelBehavior",
        "message": "Tool exceeded max retries count of 3",
    }
    assert saves == [0]
    assert closed == [{"turns_completed": 0}]


def test_game_loop_logs_and_reraises_unexpected_post_turn_failure(monkeypatch):
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

    async def complete_turn(*_args):
        return True

    async def fail_compaction(*_args):
        raise RuntimeError("chronicle storage unavailable")

    monkeypatch.setattr("src.engine.runtime.loop.StateManager", Manager)
    monkeypatch.setattr("src.engine.runtime.loop.Logger", Log)
    monkeypatch.setattr("src.engine.runtime.loop.tick", complete_turn)
    monkeypatch.setattr("src.engine.runtime.loop.compact_history", fail_compaction)

    with pytest.raises(RuntimeError, match="chronicle storage unavailable"):
        asyncio.run(_run_game("demo", "hero", max_turns=5, new_character=None))

    assert [event for event, _ in logged_events].count("run_error") == 1
    assert dict(logged_events)["run_error"] == {
        "error": "RuntimeError",
        "message": "chronicle storage unavailable",
    }
    assert saves == [0]
    assert closed == [{"turns_completed": 0}]


def test_game_loop_does_not_persist_turn_without_character_action(monkeypatch):
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

    async def no_character_action(*_args):
        return None

    async def unexpected_compaction(*_args):
        raise AssertionError("an incomplete turn must not reach compaction")

    monkeypatch.setattr("src.engine.runtime.loop.StateManager", Manager)
    monkeypatch.setattr("src.engine.runtime.loop.Logger", Log)
    monkeypatch.setattr("src.engine.runtime.loop.flow_agent_turn", no_character_action)
    monkeypatch.setattr("src.engine.runtime.loop.compact_history", unexpected_compaction)

    asyncio.run(_run_game("demo", "hero", max_turns=5, new_character=None))

    assert ("run_stopped", {"reason": "no_character_action"}) in logged_events
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
