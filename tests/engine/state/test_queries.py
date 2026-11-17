import pytest

from src.engine.state.models import Character, Location, WorldState
from src.engine.state.queries import (
    characters_in_location,
    connected_location_ids,
    live_factions,
    named_ids,
    nearby_location_ids,
    resolve_character,
    resolve_location_id,
    route,
    slugify,
)


@pytest.fixture
def state() -> WorldState:
    return WorldState(
        locations={
            "tavern": Location(id="tavern", connections=["forest"]),
            "forest": Location(id="forest", connections=["tavern", "missing"]),
        },
        characters={
            "elara-swift": Character(
                id="elara-swift", role="ranger", location="forest"
            ),
            "bram-the-bold": Character(
                id="bram-the-bold", role="fighter", location="tavern"
            ),
        },
    )


def test_slugify():
    assert slugify("Hidden Cellar") == "hidden-cellar"
    assert slugify("  The_Ancient  Library  ") == "the-ancient-library"
    assert slugify("Elara Swift!") == "elara-swift"
    assert slugify("") == ""


def test_resolve_character(state):
    exact = resolve_character(state, "elara-swift")
    normalized = resolve_character(state, "Elara Swift")
    partial = resolve_character(state, "bram")
    assert exact is not None and exact.id == "elara-swift"
    assert normalized is not None and normalized.id == "elara-swift"
    assert partial is not None and partial.id == "bram-the-bold"
    assert resolve_character(state, None) is None
    assert resolve_character(state, "ghost") is None


def test_resolve_character_ignores_article_and_title_order(state):
    state.characters["alan-the-dockmaster"] = Character(
        id="alan-the-dockmaster",
        role="dockmaster",
        location="tavern",
    )

    for name in ("alan-dockmaster", "dockmaster-alan"):
        character = resolve_character(state, name)
        assert character is not None and character.id == "alan-the-dockmaster"


def test_resolve_character_rejects_ambiguous_identity_alias(state):
    state.characters["alan-the-dockmaster"] = Character(
        id="alan-the-dockmaster",
        role="dockmaster",
        location="tavern",
    )
    state.characters["dockmaster-alan"] = Character(
        id="dockmaster-alan",
        role="dockmaster",
        location="forest",
    )

    assert resolve_character(state, "alan-dockmaster") is None


def test_resolve_character_rejects_ambiguous_partial_match(state):
    state.characters["bram-the-brave"] = Character(
        id="bram-the-brave",
        role="guard",
        location="tavern",
    )

    assert resolve_character(state, "bram") is None
    character = resolve_character(state, "bram the bold")
    assert character is not None and character.id == "bram-the-bold"


def test_resolve_location_id(state):
    assert resolve_location_id(state, "tavern") == "tavern"
    assert resolve_location_id(state, "Tavern") == "tavern"
    assert resolve_location_id(state, "for") == "forest"
    assert resolve_location_id(state, None) is None
    assert resolve_location_id(state, "void") is None


def test_resolve_location_id_rejects_ambiguous_partial_match(state):
    state.locations["forest-edge"] = Location(id="forest-edge")

    assert resolve_location_id(state, "for") is None
    assert resolve_location_id(state, "forest") == "forest"


def test_characters_in_location(state):
    assert [c.id for c in characters_in_location(state, "tavern")] == ["bram-the-bold"]
    assert (
        characters_in_location(state, "tavern", exclude_character_id="bram-the-bold")
        == []
    )
    assert characters_in_location(state, "nowhere") == []


def test_connected_location_ids(state):
    # drops connections that point to missing locations
    assert connected_location_ids(state, "tavern") == ["forest"]
    assert connected_location_ids(state, "forest") == ["tavern"]
    assert connected_location_ids(state, "nowhere") == []


def test_nearby_location_ids_walks_a_bounded_number_of_hops():
    chain = ["a", "b", "c", "d"]
    state = WorldState(
        locations={
            lid: Location(
                id=lid,
                connections=[n for n in chain if abs(chain.index(n) - i) == 1],
            )
            for i, lid in enumerate(chain)
        }
    )

    assert nearby_location_ids(state, ["a"]) == ["a", "b", "c"]
    assert nearby_location_ids(state, ["a", "d"], hops=1) == ["a", "d", "b", "c"]
    assert nearby_location_ids(state, ["nowhere"]) == []


def test_route_is_the_shortest_known_path():
    chain = ["a", "b", "c", "d"]
    state = WorldState(
        locations={
            lid: Location(
                id=lid,
                connections=[n for n in chain if abs(chain.index(n) - i) == 1],
            )
            for i, lid in enumerate(chain)
        }
    )
    state.locations["a"].connections.append("d")  # a shortcut

    assert route(state, "a", "c") == ["b", "c"]
    assert route(state, "b", "d") in (["a", "d"], ["c", "d"])
    state.locations["e"] = Location(id="e")
    assert route(state, "a", "e") is None


def test_named_ids_finds_ids_spelled_out_in_prose():
    ids = ["iron-gate-ruins", "elara-scribe", "dock"]
    text = "interrogate Elara-scribe at Iron Gate Ruins"
    assert named_ids(text, ids) == {"iron-gate-ruins", "elara-scribe"}
    assert named_ids("search the docks", ids) == set()
    assert named_ids("guard Elara-Scribe’s ledger at the dock's end", ids) == {
        "elara-scribe",
        "dock",
    }


def test_live_factions_hide_finished_and_orphaned_agendas():
    from src.engine.state.models import Faction, ProgressClock, Quest

    def faction(fid, **clock):
        return Faction(
            id=fid,
            name=fid,
            goal="win",
            clocks=[
                ProgressClock(id="c", name="c", consequence="x", segments=2, **clock)
            ],
        )

    state = WorldState(
        quests={
            "open": Quest(id="open", title="Open"),
            "done": Quest(id="done", title="Done", status="completed"),
        },
        factions={
            "live": faction("live", fail_quest_id="open"),
            "fired": faction("fired", progress=2, consequence_triggered=True),
            "orphan": faction("orphan", fail_quest_id="done"),
            "free": faction("free"),
        },
    )

    assert [f.id for f in live_factions(state)] == ["free", "live"]
