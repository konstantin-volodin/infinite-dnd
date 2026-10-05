import pytest
from pydantic import ValidationError

from src.engine.state.models import (
    Character,
    CharacterStats,
    Faction,
    HistoryEvent,
    Location,
    ProgressClock,
    Quest,
    WorldState,
)


def test_character_stats():
    stats = CharacterStats(hp=20, max_hp=20, level=3)
    assert stats.hp == 20
    assert stats.max_hp == 20
    assert stats.level == 3


def test_location():
    loc = Location(id="forest-1", description="dark forest", connections=["village-1"], features=["cabin in the woods"], items=["sword"])
    assert loc.id == "forest-1"
    assert loc.connections == ["village-1"]
    assert loc.features == ["cabin in the woods"]
    assert loc.items == ["sword"]


def test_character():
    stats = CharacterStats(hp=20, max_hp=20, level=3)
    char = Character(id="hero-1", role="warrior", inventory=["sword", "shield"], goal="slay the dragon", stats=stats)
    assert char.id == "hero-1"
    assert char.role == "warrior"
    assert char.stats.hp == 20
    assert char.inventory == ["sword", "shield"]
    assert char.goal == "slay the dragon"


def test_quest():
    char = Character(id="hero-1", role="warrior")
    quest = Quest(id="q1", title="Slay Dragon", description="defeat the dragon", owner=char.id)
    assert quest.status == "active"
    assert quest.owner == "hero-1"
    assert quest.description == "defeat the dragon"
    # back-compat defaults — old snapshots/scenarios without plan still construct fine
    assert quest.plan == []
    assert quest.current_step == 0
    assert quest.steps == []


def test_quest_with_plan():
    quest = Quest(id="q2", title="Slay Dragon", description="defeat the dragon", plan=["find the lair", "slay the dragon"])
    assert quest.plan == ["find the lair", "slay the dragon"]
    assert quest.current_step == 0


def test_quest_normalizes_status_and_preserves_legacy_progress_text():
    completed = Quest(id="done", title="Done", status=" Completed ")
    legacy = Quest(
        id="legacy",
        title="Legacy",
        status="question the servants about the circlet",
        steps=["search the upper library"],
    )

    assert completed.status == "completed"
    assert completed.steps == []
    assert legacy.status == "active"
    assert legacy.steps == [
        "search the upper library",
        "question the servants about the circlet",
    ]


@pytest.mark.parametrize(
    ("current_step", "status", "message"),
    [
        (3, "completed", "current_step cannot exceed plan length"),
        (2, "active", "active quest cannot have an exhausted plan"),
        (1, "completed", "completed quest must exhaust its plan"),
    ],
)
def test_planned_quest_rejects_impossible_progress(current_step, status, message):
    with pytest.raises(ValidationError, match=message):
        Quest(
            id="rescue",
            title="Rescue the Scout",
            plan=["find the trail", "rescue the scout"],
            current_step=current_step,
            status=status,
        )


@pytest.mark.parametrize("blank_objective", ["", "   "])
def test_planned_quest_rejects_blank_objectives(blank_objective):
    with pytest.raises(ValidationError, match="quest plan objectives cannot be blank"):
        Quest(
            id="rescue",
            title="Rescue the Scout",
            plan=["find the trail", blank_objective, "rescue the scout"],
        )


def test_world_state():
    loc = Location(id="forest-1", description="dark forest", connections=["village-1"], features=["cabin in the woods"], items=["sword"])
    stats = CharacterStats(hp=20, max_hp=20, level=3)
    char = Character(id="hero-1", role="warrior", inventory=["sword", "shield"], goal="slay the dragon", stats=stats)
    quest = Quest(id="q1", title="Slay Dragon", description="defeat the dragon", owner=char.id)

    world = WorldState(
        time=0,
        locations={"forest-1": loc},
        characters={"hero-1": char},
        quests={"q1": quest},
        history=[HistoryEvent(text="Hero enters the forest", location="forest-1", characters=["hero-1"])]
    )
    assert world.time == 0
    assert world.locations == {"forest-1": loc}
    assert world.characters == {"hero-1": char}
    assert world.quests == {"q1": quest}
    assert world.history == [HistoryEvent(text="Hero enters the forest", location="forest-1", characters=["hero-1"])]


def test_world_state_chronicle_defaults_empty_and_round_trips():
    world = WorldState()
    assert world.chronicle == []

    world.chronicle.append("Long ago, the hero entered the forest.")
    assert WorldState.model_validate(world.model_dump()).chronicle == ["Long ago, the hero entered the forest."]


def test_faction_clock_round_trips_with_world_state():
    clock = ProgressClock(id="raid", name="Prepare raid", consequence="The village is attacked.", segments=4)
    faction = Faction(id="wolves", name="The Wolves", goal="Take the valley", clocks=[clock])
    world = WorldState(factions={faction.id: faction})

    loaded = WorldState.model_validate(world.model_dump())

    assert loaded.factions["wolves"].clocks[0].segments == 4
    assert loaded.factions["wolves"].clocks[0].progress == 0


def test_faction_clock_quest_link_is_optional_and_must_resolve():
    quest = Quest(id="defend", title="Defend the Village", description="")
    clock = ProgressClock(
        id="raid",
        name="Prepare raid",
        consequence="The village is attacked.",
        segments=4,
        fail_quest_id="defend",
    )
    faction = Faction(id="wolves", name="The Wolves", goal="Take the valley", clocks=[clock])

    world = WorldState(quests={quest.id: quest}, factions={faction.id: faction})
    assert world.factions["wolves"].clocks[0].fail_quest_id == "defend"

    with pytest.raises(ValidationError, match="links unknown quest 'defend'"):
        WorldState(factions={faction.id: faction})


@pytest.mark.parametrize("status", ["active", "completed"])
def test_triggered_faction_clock_requires_linked_quest_to_be_failed(status):
    quest = Quest(id="defend", title="Defend the Village", status=status)
    clock = ProgressClock(
        id="raid",
        name="Prepare raid",
        consequence="The village is attacked.",
        progress=4,
        segments=4,
        consequence_triggered=True,
        fail_quest_id=quest.id,
    )
    faction = Faction(id="wolves", name="The Wolves", goal="Take the valley", clocks=[clock])

    with pytest.raises(
        ValidationError,
        match="triggered faction clock 'wolves/raid' requires linked quest 'defend' to be failed",
    ):
        WorldState(quests={quest.id: quest}, factions={faction.id: faction})

    quest.status = "failed"
    world = WorldState(quests={quest.id: quest}, factions={faction.id: faction})
    assert world.quests[quest.id].status == "failed"


def test_world_state_rejects_orphaned_active_quest_owner():
    quest = Quest(id="rescue", title="Rescue the Scout", owner="missing-scout")

    with pytest.raises(
        ValidationError,
        match="active quest 'rescue' owner 'missing-scout' is not a known character",
    ):
        WorldState(quests={quest.id: quest})

    quest.status = "completed"
    world = WorldState(quests={quest.id: quest})
    assert world.quests[quest.id].owner == "missing-scout"


def test_world_state_rejects_character_at_unknown_location():
    character = Character(id="lost-scout", location="missing-camp")

    with pytest.raises(
        ValidationError,
        match="character 'lost-scout' location 'missing-camp' is not a known location",
    ):
        WorldState(characters={character.id: character})

    locationless_character = Character(id="unplaced-npc")
    world = WorldState(characters={locationless_character.id: locationless_character})
    assert world.characters[locationless_character.id].location == ""


def test_world_state_without_factions_is_backward_compatible():
    data = WorldState().model_dump()
    del data["factions"]

    assert WorldState.model_validate(data).factions == {}


def test_collection_defaults_are_not_shared():
    first = WorldState()
    second = WorldState()

    first.locations["forest"] = Location(id="forest")
    first.history.append(HistoryEvent(text="arrived", location="forest"))
    first.chronicle.append("a summary")

    assert second.locations == {}
    assert second.history == []
    assert second.chronicle == []

    first_character = Character(id="first")
    second_character = Character(id="second")
    first_character.inventory.append("torch")
    assert second_character.inventory == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("hp", -1), ("max_hp", -1), ("level", 0), ("xp", -1), ("gold", -1)],
)
def test_character_stats_reject_invalid_values(field, value):
    with pytest.raises(ValidationError):
        CharacterStats.model_validate({field: value})


def test_character_stats_reject_hp_above_maximum():
    with pytest.raises(ValidationError, match="hp cannot exceed max_hp"):
        CharacterStats(hp=6, max_hp=5)


@pytest.mark.parametrize("field", ["time", "minutes_elapsed", "last_quest_advance_time"])
def test_world_state_rejects_negative_clocks(field):
    with pytest.raises(ValidationError):
        WorldState.model_validate({field: -1})


def test_world_state_rejects_future_quest_advance_time():
    with pytest.raises(ValidationError, match="last_quest_advance_time cannot exceed time"):
        WorldState(time=4, last_quest_advance_time=5)


def test_events_and_quests_reject_negative_progress():
    with pytest.raises(ValidationError):
        HistoryEvent(text="impossible", location="void", minutes_elapsed=-1)
    with pytest.raises(ValidationError):
        Quest(id="q", title="Quest", description="", current_step=-1)


def test_faction_requires_clock_and_clock_rejects_invalid_progress():
    with pytest.raises(ValidationError):
        Faction(id="empty", name="Empty", goal="Nothing", clocks=[])
    with pytest.raises(ValidationError, match="progress cannot exceed segments"):
        ProgressClock(id="clock", name="Clock", consequence="Trouble.", progress=3, segments=2)
    with pytest.raises(ValidationError, match="completed clock must trigger its consequence"):
        ProgressClock(id="clock", name="Clock", consequence="Trouble.", progress=2, segments=2)
    with pytest.raises(ValidationError):
        ProgressClock(id="clock", name="Clock", consequence="", segments=2)
    with pytest.raises(ValidationError, match="clock ids must be unique after normalization"):
        Faction(
            id="duplicates",
            name="Duplicates",
            goal="Confuse everyone",
            clocks=[
                ProgressClock(id="same", name="First", consequence="One.", segments=2),
                ProgressClock(id="same", name="Second", consequence="Two.", segments=2),
            ],
        )


def test_faction_rejects_slug_equivalent_clock_ids():
    with pytest.raises(ValidationError, match="clock ids must be unique after normalization"):
        Faction(
            id="ambiguous",
            name="Ambiguous",
            goal="Confuse clock updates",
            clocks=[
                ProgressClock(id="secret-haul", name="First", consequence="One.", segments=2),
                ProgressClock(id="SECRET HAUL", name="Second", consequence="Two.", segments=2),
            ],
        )


def test_faction_rejects_clock_id_without_letters_or_numbers():
    with pytest.raises(ValidationError, match="clock ids must contain a letter or number"):
        Faction(
            id="valid-faction",
            name="Invalid Clock",
            goal="Become impossible to update",
            clocks=[
                ProgressClock(id="!!!", name="Broken", consequence="Trouble.", segments=2),
            ],
        )


@pytest.mark.parametrize(
    ("field", "entity_name", "entities"),
    [
        (
            "locations",
            "location",
            {
                "old-road": Location(id="old-road"),
                "OLD ROAD": Location(id="OLD ROAD"),
            },
        ),
        (
            "characters",
            "character",
            {
                "dock-master": Character(id="dock-master"),
                "DOCK MASTER": Character(id="DOCK MASTER"),
            },
        ),
        (
            "quests",
            "quest",
            {
                "lost-map": Quest(id="lost-map", title="First Map"),
                "LOST MAP": Quest(id="LOST MAP", title="Second Map"),
            },
        ),
    ],
)
def test_world_state_rejects_slug_equivalent_entity_ids(field, entity_name, entities):
    with pytest.raises(ValidationError, match=rf"{entity_name} ids must be unique after normalization"):
        WorldState(**{field: entities})


@pytest.mark.parametrize(
    ("field", "entity_name", "entity"),
    [
        ("locations", "location", Location(id="!!!")),
        ("characters", "character", Character(id="___")),
        ("quests", "quest", Quest(id="---", title="Invalid Quest")),
        (
            "factions",
            "faction",
            Faction(
                id="...",
                name="Invalid Faction",
                goal="Become impossible to update",
                clocks=[
                    ProgressClock(
                        id="valid-clock",
                        name="Valid Clock",
                        consequence="Trouble.",
                        segments=2,
                    )
                ],
            ),
        ),
    ],
)
def test_world_state_rejects_entity_ids_without_letters_or_numbers(field, entity_name, entity):
    with pytest.raises(ValidationError, match=rf"{entity_name} ids must contain a letter or number"):
        WorldState(**{field: {entity.id: entity}})


@pytest.mark.parametrize(
    ("field", "entity_name", "entity"),
    [
        ("locations", "location", Location(id="old-road")),
        ("characters", "character", Character(id="dock-master")),
        ("quests", "quest", Quest(id="lost-map", title="Lost Map")),
        (
            "factions",
            "faction",
            Faction(
                id="ash-guild",
                name="Ash Guild",
                goal="Control the harbor",
                clocks=[
                    ProgressClock(
                        id="raid",
                        name="Raid",
                        consequence="The harbor falls.",
                        segments=2,
                    )
                ],
            ),
        ),
    ],
)
def test_world_state_rejects_entity_map_key_mismatches(field, entity_name, entity):
    with pytest.raises(
        ValidationError,
        match=rf"{entity_name} map key 'alias' does not match entity id '{entity.id}'",
    ):
        WorldState(**{field: {"alias": entity}})


def test_world_state_rejects_slug_equivalent_faction_ids():
    factions = {
        "ash-guild": Faction(
            id="ash-guild",
            name="First Guild",
            goal="Control the harbor",
            clocks=[ProgressClock(id="raid", name="Raid", consequence="One.", segments=2)],
        ),
        "ASH GUILD": Faction(
            id="ASH GUILD",
            name="Second Guild",
            goal="Control the roads",
            clocks=[ProgressClock(id="blockade", name="Blockade", consequence="Two.", segments=2)],
        ),
    }

    with pytest.raises(ValidationError, match="faction ids must be unique after normalization"):
        WorldState(factions=factions)
