from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import ModelRetry

from src.agents.character.agent import CharacterDeps, travel_output
from src.engine.state.models import Character, Location, WorldState


def _context(*, connections: list[str]) -> Any:
    state = WorldState(
        locations={
            "tavern": Location(id="tavern", connections=connections),
            "forest": Location(id="forest"),
        },
        characters={"hero": Character(id="hero", location="tavern")},
    )
    return SimpleNamespace(deps=CharacterDeps(char=state.characters["hero"], state=state))


def test_travel_output_accepts_connected_existing_location():
    result = travel_output(_context(connections=["forest"]), "forest")

    assert result.actor == "hero"
    assert result.destination == "forest"


def test_travel_output_retries_unknown_or_unconnected_location():
    with pytest.raises(ModelRetry, match=r"Cannot travel to 'castle'.*Valid location ids: forest"):
        travel_output(_context(connections=["forest"]), "castle")


def test_travel_output_does_not_offer_dangling_connection():
    with pytest.raises(ModelRetry, match="No connected locations are available; use action to discover one"):
        travel_output(_context(connections=["missing-location"]), "missing-location")
