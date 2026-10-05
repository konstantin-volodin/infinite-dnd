"""Read-only lookups over WorldState. Fuzzy id resolution lives here."""

from src.engine.state.identifiers import slugify
from src.engine.state.models import Character, Faction, ProgressClock, WorldState

_CHARACTER_ID_ARTICLES = {"a", "an", "the"}


def character_ids_match(left: str, right: str) -> bool:
    """Return whether two character IDs differ only by word order or articles."""
    left_slug = slugify(left)
    right_slug = slugify(right)
    if left_slug == right_slug:
        return True
    left_words = sorted(
        word for word in left_slug.split("-") if word not in _CHARACTER_ID_ARTICLES
    )
    right_words = sorted(
        word for word in right_slug.split("-") if word not in _CHARACTER_ID_ARTICLES
    )
    return bool(left_words) and left_words == right_words


def _fuzzy_match(raw: str, ids: list[str]) -> str | None:
    """Exact slug match first, then a unique substring match."""
    needle = slugify(raw)
    for candidate in ids:
        if slugify(candidate) == needle:
            return candidate
    matches = [candidate for candidate in ids if needle and needle in slugify(candidate)]
    return matches[0] if len(matches) == 1 else None


def resolve_character(state: WorldState, raw: str | None) -> Character | None:
    if not raw:
        return None
    if raw in state.characters:
        return state.characters[raw]
    identity_matches = [
        candidate
        for candidate in state.characters
        if character_ids_match(raw, candidate)
    ]
    if len(identity_matches) == 1:
        return state.characters[identity_matches[0]]
    if len(identity_matches) > 1:
        return None
    match = _fuzzy_match(raw, list(state.characters))
    return state.characters[match] if match else None


def resolve_location_id(state: WorldState, raw: str | None) -> str | None:
    if not raw:
        return None
    if raw in state.locations:
        return raw
    return _fuzzy_match(raw, list(state.locations))


def characters_in_location(
    state: WorldState, location_id: str, *, exclude_character_id: str | None = None
) -> list[Character]:
    return [
        c for c in state.characters.values()
        if c.location == location_id and c.id != exclude_character_id
    ]


def connected_location_ids(state: WorldState, location_id: str) -> list[str]:
    loc = state.locations.get(location_id)
    if not loc:
        return []
    return [c for c in loc.connections if c in state.locations]


def quest_deadline_clocks(state: WorldState, quest_ids: set[str]) -> list[tuple[Faction, ProgressClock]]:
    """Unresolved faction clocks that would fail one of the given quests, in stable faction/clock order."""
    return [
        (faction, clock)
        for faction in (state.factions[fid] for fid in sorted(state.factions))
        for clock in sorted(faction.clocks, key=lambda candidate: candidate.id)
        if clock.fail_quest_id in quest_ids and not clock.consequence_triggered
    ]


def is_dialogue(text: str) -> bool:
    """Heuristic: does this event text read as speech — quoted or with a 'says' verb?"""
    return '"' in text or "says" in text.lower()
