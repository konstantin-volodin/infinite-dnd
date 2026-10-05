"""Character prompt builders - system prompt and context."""

from src.engine.state import Character, WorldState, characters_in_location, is_dialogue, quest_deadline_clocks
from src.engine.rules import get_health_status
from src.agents.utils import render


def character_system(char: Character) -> str:
    """Build the character identity."""
    return render("character/identity.jinja", char=char)


def character_context(char: Character, state: WorldState) -> str:
    """Build the character context."""

    # internal status
    health_status = get_health_status(char)
    loc = state.locations.get(char.location)
    knowledge = char.knowledge[-5:] if char.knowledge else []

    # relevant quests
    quests = [
        q for q in state.quests.values()
        if q.owner == char.id and q.status.lower() not in ("completed", "failed")
    ]
    quest_ids = {quest.id for quest in quests}
    deadlines = [(faction.name, clock) for faction, clock in quest_deadline_clocks(state, quest_ids)]

    # Recent events (only ones this character witnessed)
    recent_events = [
        e.text for e in state.history
        if char.id in e.characters
    ][-20:]

    # Someone speaking to me?
    someone_speaking_to_me = False
    if recent_events:
        last = recent_events[-1]
        someone_speaking_to_me = is_dialogue(last) and not last.startswith(char.id)

    # Others present, with visible condition (role, health if not healthy)
    present_characters = characters_in_location(state, char.location, exclude_character_id=char.id)
    others = []
    for c in present_characters:
        status = get_health_status(c)
        tags = [t for t in (c.role, status if status != "healthy" else "") if t]
        if disposition := c.relationships.get(char.id):
            tags.append(f"thinks of me: {disposition}")
        others.append(f"{c.id} ({', '.join(tags)})" if tags else c.id)
    speak_targets = [c.id for c in present_characters if c.stats.hp > 0]
    # Warnings
    warnings: list[str] = []
    if sum(1 for e in recent_events[-5:] if is_dialogue(e)) >= 4:
        warnings.append("*Lots of talking. Maybe time for action.*")

    return render(
        "character/state.jinja",
        char=char,
        health_status=health_status,
        loc=loc,
        knowledge=knowledge,
        chronicle=state.chronicle[-3:],
        recent_events=recent_events,
        someone_speaking_to_me=someone_speaking_to_me,
        others=others,
        speak_targets=speak_targets,
        quests=quests,
        deadlines=deadlines,
        warnings=warnings,
    )
