from src.engine.state.models import Character, Faction, ProgressClock, Quest, WorldState
from src.engine.state.operations import WorldOperations


def _state(*, progress: int = 0, segments: int = 3) -> WorldState:
    faction = Faction(
        id="guild",
        name="The Ash Guild",
        goal="Control the harbor",
        clocks=[
            ProgressClock(
                id="seize-docks",
                name="Seize the docks",
                progress=progress,
                segments=segments,
                consequence="The guild closes the harbor gates.",
            )
        ],
    )
    return WorldState(factions={faction.id: faction})


def test_advance_clock_clamps_and_triggers_consequence_once():
    state = _state(progress=2, segments=3)
    operations = WorldOperations(state)

    result = operations.advance_faction_clock("guild", "seize-docks", amount=5)

    clock = state.factions["guild"].clocks[0]
    assert clock.progress == 3
    assert clock.consequence_triggered is True
    assert "Consequence triggered" in result
    assert [event.text for event in state.history] == [
        "The Ash Guild completes 'Seize the docks'. The guild closes the harbor gates."
    ]

    operations.advance_faction_clock("guild", "seize-docks")
    assert len(state.history) == 1


def test_advance_all_clocks_is_stable_and_skips_triggered_clocks():
    state = _state(progress=2, segments=3)
    state.factions["cabal"] = Faction(
        id="cabal",
        name="Cabal",
        goal="Hide the truth",
        clocks=[
            ProgressClock(
                id="erase-records",
                name="Erase records",
                consequence="The archive burns.",
                segments=1,
            )
        ],
    )

    WorldOperations(state).advance_faction_clocks()
    WorldOperations(state).advance_faction_clocks()

    assert [event.text for event in state.history] == [
        "Cabal completes 'Erase records'. The archive burns.",
        "The Ash Guild completes 'Seize the docks'. The guild closes the harbor gates.",
    ]


def test_hourly_advance_moves_one_segment_per_hour_crossed():
    state = _state(segments=5)
    operations = WorldOperations(state)

    assert operations.advance_faction_clocks_hourly(0, 59) == []
    assert state.factions["guild"].clocks[0].progress == 0

    operations.advance_faction_clocks_hourly(59, 61)
    assert state.factions["guild"].clocks[0].progress == 1

    operations.advance_faction_clocks_hourly(61, 240)  # a long rest crosses several hours at once
    assert state.factions["guild"].clocks[0].progress == 4


def test_missing_clock_does_not_mutate_state():
    state = _state()

    result = WorldOperations(state).advance_faction_clock("guild", "missing")

    assert "not found" in result
    assert state.factions["guild"].clocks[0].progress == 0
    assert state.history == []


def test_completed_deadline_fails_linked_active_quest_once_and_warns_everyone():
    state = _state(progress=2, segments=3)
    state.time = 7
    state.characters = {
        "hero": Character(id="hero"),
        "witness": Character(id="witness"),
    }
    state.quests["save-village"] = Quest(
        id="save-village",
        title="Save the Village",
        description="Stop the raid before dawn.",
    )
    state.factions["guild"].clocks[0].fail_quest_id = "save-village"

    result = WorldOperations(state).advance_faction_clock("guild", "seize-docks")

    assert state.quests["save-village"].status == "failed"
    assert state.last_quest_advance_time == 7
    assert "Quest 'save-village' failed" in result
    assert len(state.history) == 1
    assert state.history[0].text == (
        "The Ash Guild completes 'Seize the docks'. The guild closes the harbor gates. "
        "Quest 'Save the Village' failed."
    )
    assert state.history[0].characters == ["hero", "witness"]

    WorldOperations(state).advance_faction_clock("guild", "seize-docks")
    assert len(state.history) == 1


def test_hourly_deadline_triggers_at_exact_boundary():
    state = _state(segments=6)
    state.quests["save-village"] = Quest(
        id="save-village",
        title="Save the Village",
        description="Stop the raid before dawn.",
    )
    state.factions["guild"].clocks[0].fail_quest_id = "save-village"
    operations = WorldOperations(state)

    operations.advance_faction_clocks_hourly(0, 359)
    assert state.quests["save-village"].status == "active"
    assert state.factions["guild"].clocks[0].progress == 5

    operations.advance_faction_clocks_hourly(359, 360)
    assert state.quests["save-village"].status == "failed"
    assert state.factions["guild"].clocks[0].consequence_triggered is True
    assert len(state.history) == 1


def test_resolved_quest_stops_linked_deadline_without_overwriting_status():
    state = _state(progress=2, segments=3)
    state.quests["save-village"] = Quest(
        id="save-village",
        title="Save the Village",
        description="",
        status="completed",
    )
    clock = state.factions["guild"].clocks[0]
    clock.fail_quest_id = "save-village"

    assert WorldOperations(state).advance_faction_clocks() == []
    assert state.quests["save-village"].status == "completed"
    assert clock.progress == 2
    assert state.history == []
