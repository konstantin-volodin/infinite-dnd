"""Shared fixtures for the pytest suite. Mirrors src/ layout under tests/."""

import pytest

from src.engine.state.models import Location


@pytest.fixture
def locations() -> dict[str, Location]:
    """The tavern/forest/cave graph reused across the state-operations tests."""
    return {
        "tavern": Location(id="tavern", connections=["forest"], items=["ale"]),
        "forest": Location(id="forest", connections=["tavern", "cave"]),
        "cave": Location(id="cave", connections=["forest"]),
    }
