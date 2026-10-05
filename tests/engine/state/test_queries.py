import pytest

from src.engine.state.models import Character, Location, WorldState
from src.engine.state.queries import (
    characters_in_location,
    connected_location_ids,
    resolve_character,
    resolve_location_id,
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
            "elara-swift": Character(id="elara-swift", role="ranger", location="forest"),
            "bram-the-bold": Character(id="bram-the-bold", role="fighter", location="tavern"),
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


def test_resolve_location_id(state):
    assert resolve_location_id(state, "tavern") == "tavern"
    assert resolve_location_id(state, "Tavern") == "tavern"
    assert resolve_location_id(state, "for") == "forest"
    assert resolve_location_id(state, None) is None
    assert resolve_location_id(state, "void") is None


def test_characters_in_location(state):
    assert [c.id for c in characters_in_location(state, "tavern")] == ["bram-the-bold"]
    assert characters_in_location(state, "tavern", exclude_character_id="bram-the-bold") == []
    assert characters_in_location(state, "nowhere") == []


def test_connected_location_ids(state):
    # drops connections that point to missing locations
    assert connected_location_ids(state, "tavern") == ["forest"]
    assert connected_location_ids(state, "forest") == ["tavern"]
    assert connected_location_ids(state, "nowhere") == []
