"""Action resolver prompt builders - converts character actions into state changes."""

from src.engine.state import (
    Character,
    WorldState,
    connected_location_ids,
    resolve_character,
    resolve_location_id,
    slugify,
)
from src.engine.rules import get_health_status
from src.agents.utils import render


def _resolve_target_item(char: Character, state: WorldState, target: str | None) -> str | None:
    if not target:
        return None

    search_spaces: list[list[str]] = [char.inventory]
    loc = state.locations.get(char.location)
    if loc:
        search_spaces.append(loc.items)

    target_slug = slugify(target)
    for values in search_spaces:
        for value in values:
            if slugify(value) == target_slug:
                return value
    return None


def action_resolver_system() -> str:
    """Build the action resolver system prompt."""
    return render("action_resolver/identity.jinja")


def action_resolver_context(char: Character, state: WorldState, description: str, target: str | None = None) -> str:
    """Build the action resolver context."""

    loc = state.locations.get(char.location)
    if loc:
        loc = loc.model_copy(update={"connections": connected_location_ids(state, loc.id)})
    recent_events = [event.text for event in state.history[-8:]]
    others = [
        f"{other.id} ({other.role})" if other.role else other.id
        for other in state.characters.values()
        if other.location == char.location and other.id != char.id
    ]
    known_characters = [
        other
        for other in state.characters.values()
        if other.id != char.id
    ]
    quests = [
        quest
        for quest in state.quests.values()
        if str(getattr(quest, "status", "active")).lower() not in ("completed", "failed")
    ]
    target_character = resolve_character(state, target)
    target_location_id = resolve_location_id(state, target)
    target_location = state.locations.get(target_location_id or "") if target_location_id else None
    target_item = _resolve_target_item(char, state, target)

    return render(
        "action_resolver/state.jinja",
        char=char,
        description=description,
        target=target,
        target_character=target_character,
        target_location=target_location,
        target_item=target_item,
        loc=loc,
        others=others,
        known_characters=known_characters,
        quests=quests,
        chronicle=state.chronicle[-3:],
        recent_events=recent_events,
        health_status=get_health_status(char),
    )
