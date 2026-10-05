"""DM prompt builders."""

from src.engine.state import WorldState
from src.agents.utils import render


def dm_system() -> str:
    return render("dm/identity.jinja")


def dm_context(deps) -> str:
    state: WorldState = deps.state
    location_ids = sorted(state.locations.keys())
    character_ids = sorted(state.characters.keys())
    item_ids = sorted(
        {item for loc in state.locations.values() for item in loc.items}
        | {item for char in state.characters.values() for item in char.inventory}
    )
    quest_ids = sorted(state.quests.keys())
    quests = [
        q for q in state.quests.values()
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
        factions=[state.factions[fid] for fid in sorted(state.factions)],
    )


def director_system() -> str:
    return render("dm/director_identity.jinja")


def director_context(deps) -> str:
    state: WorldState = deps.state
    scene = state.locations.get(deps.location_id)
    recent_events = [e.text for e in state.history[-10:]]
    quests = [
        q for q in state.quests.values()
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
        factions=[state.factions[fid] for fid in sorted(state.factions)],
    )
