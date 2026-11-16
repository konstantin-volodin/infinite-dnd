"""Read-only lookups over WorldState. Fuzzy id resolution lives here."""

from collections import deque

from src.engine.state.identifiers import slugify
from src.engine.state.models import Character, Faction, ProgressClock, Quest, WorldState

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
    matches = [
        candidate for candidate in ids if needle and needle in slugify(candidate)
    ]
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
        c
        for c in state.characters.values()
        if c.location == location_id and c.id != exclude_character_id
    ]


def connected_location_ids(state: WorldState, location_id: str) -> list[str]:
    loc = state.locations.get(location_id)
    if not loc:
        return []
    return [c for c in loc.connections if c in state.locations]


def route(state: WorldState, start: str, goal: str) -> list[str] | None:
    """Shortest path from start to goal as the places entered, in order. None if unreachable."""
    came_from = {start: start}
    queue = deque([start])
    while queue and goal not in came_from:
        here = queue.popleft()
        for neighbour in connected_location_ids(state, here):
            if neighbour not in came_from:
                came_from[neighbour] = here
                queue.append(neighbour)
    if goal not in came_from:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    return path[-2::-1]


def named_ids(text: str, ids) -> set[str]:
    """Known ids spelled out in free text, e.g. 'question Old Keph at harbor-dock'."""
    padded = f"-{slugify(text)}-"
    return {id_ for id_ in ids if f"-{slugify(id_)}-" in padded}


def objective_routes(state: WorldState, char: Character) -> list[tuple[str, list[str]]]:
    """Places the character's current objectives name, each with the route there."""
    routes = []
    for quest in state.quests.values():
        if quest.owner != char.id or quest.status != "active":
            continue
        if quest.current_step >= len(quest.plan):
            continue
        for place in sorted(named_ids(quest.plan[quest.current_step], state.locations)):
            if path := route(state, char.location, place):
                routes.append((place, path))
    return routes


def find_quest(state: WorldState, raw: str) -> Quest | None:
    """A quest by id, ignoring case and punctuation, or by a unique title."""
    needle = slugify(raw)
    for quest_id, quest in state.quests.items():
        if slugify(quest_id) == needle:
            return quest
    titled = [q for q in state.quests.values() if slugify(q.title) == needle]
    return titled[0] if len(titled) == 1 else None


def nearby_location_ids(
    state: WorldState, origins: list[str], hops: int = 2
) -> list[str]:
    """Locations within `hops` connections of any origin — keeps prompts bounded as the map grows."""
    seen = [origin for origin in dict.fromkeys(origins) if origin in state.locations]
    frontier = list(seen)
    for _ in range(hops):
        frontier = list(
            dict.fromkeys(
                neighbour
                for location_id in frontier
                for neighbour in connected_location_ids(state, location_id)
                if neighbour not in seen
            )
        )
        seen.extend(frontier)
    return seen


def live_factions(state: WorldState) -> list[Faction]:
    """Factions with a clock that can still advance, in stable id order."""
    return [
        faction
        for faction in (state.factions[fid] for fid in sorted(state.factions))
        if any(
            not clock.consequence_triggered
            and not (
                clock.fail_quest_id
                and state.quests[clock.fail_quest_id].status != "active"
            )
            for clock in faction.clocks
        )
    ]


def quest_deadline_clocks(
    state: WorldState, quest_ids: set[str]
) -> list[tuple[Faction, ProgressClock]]:
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
