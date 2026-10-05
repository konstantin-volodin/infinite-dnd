from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import ModelRetry

from src.agents.character.agent import CharacterDeps, action_output, travel_output
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


def test_action_output_retries_direct_interaction_with_remote_character_when_target_omitted():
    ctx = _context(connections=["forest"])
    ctx.deps.state.characters["eleanor"] = Character(id="eleanor", location="forest")

    with pytest.raises(ModelRetry, match=r"Cannot interact with 'eleanor'.*not in the same location"):
        action_output(ctx, "Find Eleanor and question her directly about where she moved the circlet.")


def test_action_output_retries_explicit_remote_character_target():
    ctx = _context(connections=["forest"])
    ctx.deps.state.characters["eleanor"] = Character(id="eleanor", location="forest")

    with pytest.raises(ModelRetry, match=r"Cannot interact with 'eleanor'.*not in the same location"):
        action_output(ctx, "Search her pockets for the circlet.", target="Eleanor")


def test_action_output_retries_movement_to_explicit_remote_location():
    ctx = _context(connections=["forest"])

    with pytest.raises(ModelRetry, match=r"Cannot move to 'forest' with action.*Use travel"):
        action_output(
            ctx,
            "Move toward the forest to search for the missing circlet.",
            target="forest",
        )


def test_action_output_retries_movement_to_remote_location_when_target_omitted():
    ctx = _context(connections=["forest"])

    with pytest.raises(ModelRetry, match=r"Cannot move to 'forest' with action.*Use travel"):
        action_output(ctx, "Head toward the forest and search for tracks.")

    with pytest.raises(ModelRetry, match=r"Cannot move to 'forest' with action.*Use travel"):
        action_output(ctx, "Chase Calla toward the forest exit and demand she explain the locked chest.")


def test_action_output_retries_interaction_with_remote_location_when_target_omitted():
    ctx = _context(connections=["forest"])

    with pytest.raises(
        ModelRetry,
        match=r"Cannot interact with 'forest' from 'tavern'.*Travel there",
    ):
        action_output(ctx, "Search the forest for clues and examine the old trail marker.")


def test_action_output_allows_remote_location_reference_without_movement():
    ctx = _context(connections=["forest"])

    result = action_output(ctx, "Search the ledger for shipments from the forest.", target="forest")
    gold_result = action_output(ctx, "Count the gold shipment from the forest.", target="forest")

    assert result.description == "Search the ledger for shipments from the forest."
    assert gold_result.description == "Count the gold shipment from the forest."


def test_action_output_allows_noninteractive_remote_reference_and_local_interaction():
    ctx = _context(connections=["forest"])
    ctx.deps.state.characters.update({
        "eleanor": Character(id="eleanor", location="forest"),
        "merchant": Character(id="merchant", location="tavern"),
    })

    research = action_output(ctx, "Search the ledger for references to Eleanor.")
    interaction = action_output(ctx, "Question the merchant about the ledger.")

    assert research.description == "Search the ledger for references to Eleanor."
    assert interaction.description == "Question the merchant about the ledger."
