"""DM prompt builders."""

from src.engine.state import WorldState, live_factions, nearby_location_ids
from src.agents.utils import render

_RECENT_ENDED_QUESTS = 5


def _scene_catalogue(state: WorldState) -> tuple[list[str], list[str], list[str]]:
    """Location, character, and item ids near the protagonist — bounded as the world grows."""
    hero = state.characters.get(state.protagonist)
    location_ids = (
        nearby_location_ids(state, [hero.location]) if hero else list(state.locations)
    )
    characters = [c for c in state.characters.values() if c.location in location_ids]
    items = {item for lid in location_ids for item in state.locations[lid].items} | {
        item for c in characters for item in c.inventory
    }
    return sorted(location_ids), sorted(c.id for c in characters), sorted(items)


def dm_system() -> str:
    return render("dm/identity.jinja")


def dm_context(deps) -> str:
    state: WorldState = deps.state
    location_ids, character_ids, item_ids = _scene_catalogue(state)
    ended = [qid for qid, q in state.quests.items() if q.status != "active"]
    quest_ids = sorted(
        [qid for qid, q in state.quests.items() if q.status == "active"]
        + ended[-_RECENT_ENDED_QUESTS:]
    )
    quests = [
        q
        for q in state.quests.values()
        if str(getattr(q, "status", "active")).lower() not in ("completed", "failed")
    ]
    return render(
        "dm/state.jinja",
        chronicle=state.chronicle[-3:],
        context_events=deps.context_events,
        new_events=deps.new_events,
        location_ids=location_ids,
        character_ids=character_ids,
        item_ids=item_ids,
        quest_ids=quest_ids,
        quests=quests,
        factions=live_factions(state),
    )


def director_system() -> str:
    return render("dm/director_identity.jinja")


def director_context(deps) -> str:
    state: WorldState = deps.state
    scene = state.locations.get(deps.location_id)
    recent_events = [e.text for e in state.history[-10:]]
    quests = [
        q
        for q in state.quests.values()
        if str(getattr(q, "status", "active")).lower() not in ("completed", "failed")
    ]
    present = sorted(
        (c for c in state.characters.values() if c.location == deps.location_id),
        key=lambda c: c.id,
    )
    return render(
        "dm/director_state.jinja",
        chronicle=state.chronicle[-3:],
        recent_events=recent_events,
        quests=quests,
        interventions=state.director_interventions,
        location_id=deps.location_id,
        scene=scene,
        present=present,
        factions=live_factions(state),
    )
