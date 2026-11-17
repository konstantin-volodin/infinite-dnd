"""Campaign upkeep between ticks: the story never ends, it changes hands and moves on.

- a dead hero hands the story to a successor (deterministic)
- a quest stuck past a bound skips the stuck objective, or fails on its last one (deterministic)
- a hero with no active quest gets a new arc from the storyteller (one LLM call)
"""

from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.usage import UsageLimits

from src.agents.dm.storyteller import (
    Arc,
    StorytellerDeps,
    agent as storyteller_agent,
    apply_arc,
)
from src.engine.state import Character, WorldOperations, WorldState, slugify
from src.interface.session_log import Logger
from .replay import ReplayTape

# Ticks an active quest may go without its objective advancing before it is
# given up on. The director fires every few quiet ticks (see loop._is_stalled), so
# a quest reaching this bound has already had several interventions.
FORCE_PROGRESS_TICKS = 16

_ARC_USAGE = UsageLimits(request_limit=6)


def _active_quests(state: WorldState, owner: str):
    return [
        q for q in state.quests.values() if q.owner == owner and q.status == "active"
    ]


def _successor(state: WorldState, hero: Character) -> Character:
    """A living character tied to the hero or their quest; otherwise a newcomer."""
    quest_text = "-".join(
        slugify(f"{q.title} {q.description} {' '.join(q.plan)}")
        for q in _active_quests(state, hero.id)
    )
    tied = [
        c
        for c in state.characters.values()
        if c.id != hero.id
        and c.stats.hp > 0
        and (
            c.id in hero.relationships
            or hero.id in c.relationships
            or f"-{c.id}-" in f"-{quest_text}-"
        )
    ]
    if tied:
        return min(tied, key=lambda c: (c.location != hero.location, c.id))

    newcomer_id = "wandering-sellsword"
    suffix = 2
    while newcomer_id in state.characters:
        newcomer_id, suffix = f"wandering-sellsword-{suffix}", suffix + 1
    WorldOperations(state).spawn_character(
        newcomer_id,
        role="sellsword",
        location_id=hero.location,
        backstory=f"I arrived just after {hero.id} fell, and I owe the dead a debt.",
        goal=f"Learn what killed {hero.id} and finish what they started.",
    )
    newcomer = state.characters[newcomer_id]
    newcomer.stats.max_hp = newcomer.stats.hp = 10
    newcomer.relationships[hero.id] = "a stranger whose cause I took up"
    return newcomer


def hand_over(state: WorldState) -> tuple[str, str] | None:
    """If the protagonist is dead, make a successor the protagonist. Returns (dead, successor)."""
    hero = state.characters.get(state.protagonist)
    if hero is None or hero.stats.hp > 0:
        return None
    heir = _successor(state, hero)
    inherited = _active_quests(state, hero.id)
    for quest in inherited:
        quest.owner = heir.id
        quest.last_progress = state.time  # the successor gets a full bound
        heir.knowledge.append(f"{hero.id} died pursuing '{quest.title}'.")
    if inherited:
        heir.goal = f"Finish what {hero.id} began: {inherited[0].title}."
    cause = f" and takes up '{inherited[0].title}'" if inherited else ""
    WorldOperations(state).world_event(
        f"{hero.id} has fallen. {heir.id} vows to carry on{cause}.", heir.location
    )
    state.protagonist = heir.id
    return hero.id, heir.id


def move_stalled_quests_on(state: WorldState) -> list[tuple[str, str]]:
    """Keep a stuck quest moving, as a DM would: after FORCE_PROGRESS_TICKS without
    progress the hero gives up on the objective and presses on to the next one; a
    stuck final objective loses the quest. Returns (quest id, "skipped" | "abandoned").
    """
    moved: list[tuple[str, str]] = []
    for quest in [q for q in state.quests.values() if q.status == "active"]:
        if state.time - quest.last_progress < FORCE_PROGRESS_TICKS:
            continue
        owner = state.characters.get(quest.owner)
        place = owner.location if owner else ""
        who = quest.owner or "No one"
        ops = WorldOperations(state)
        if quest.current_step < len(quest.plan) - 1:
            objective = quest.plan[quest.current_step]
            quest.steps.append(f"gave up trying to {objective}")
            quest.current_step += 1
            quest.last_progress = state.time
            ops.world_event(
                f"Time is running out. {who} gives up trying to {objective} and "
                f"presses on to {quest.plan[quest.current_step]}, less prepared "
                "than they hoped.",
                place,
            )
            moved.append((quest.id, "skipped"))
            continue
        objective = quest.plan[-1] if quest.plan else quest.title
        ops.advance_quest(
            quest.id,
            new_status="failed",
            step=f"abandoned, never managed to {objective}",
        )
        ops.world_event(
            f"The trail goes cold: '{quest.title}' is lost. "
            f"{who} never managed to {objective}, "
            "and whoever stood to gain from that failure now moves freely.",
            place,
        )
        moved.append((quest.id, "abandoned"))
    return moved


async def open_arc(
    state: WorldState, logger: Logger, replay: ReplayTape | None = None
) -> Arc | None:
    """Ask the storyteller for the next arc and apply it. None when it could not."""
    if replay and replay.is_playback:
        arc = replay.arc()
    else:
        try:
            with logger.run("storyteller"):
                result = await storyteller_agent.run(
                    "The hero has no quest. Open the next arc.",
                    deps=StorytellerDeps(state=state),
                    usage_limits=_ARC_USAGE,
                )
                logger.log_messages("storyteller", result.all_messages())
            arc = result.output
        except AgentRunError as exc:
            print(f"  [fallback] storyteller failed: {exc}", flush=True)
            arc = None
        if replay:
            arc = replay.arc(arc)
    if arc is None:
        return None

    trial = state.model_copy(deep=True)
    if errors := apply_arc(trial, arc):  # validated in the tool; replays must agree
        logger.log_event("arc_rejected", errors=errors)
        return None
    WorldState.model_validate(trial.model_dump())
    for field_name in WorldState.model_fields:
        setattr(state, field_name, getattr(trial, field_name))
    return arc


def succeed_if_dead(state: WorldState, logger: Logger) -> None:
    """Hand the story over when the protagonist has died, and announce it."""
    if handed := hand_over(state):
        dead, heir = handed
        print(f"\n=== {dead} has died. {heir} carries the story on. ===")
        logger.log_event("pc_death", pc=dead)
        logger.log_event("successor", pc=heir, previous=dead)


async def campaign_upkeep(
    state: WorldState, logger: Logger, replay: ReplayTape | None = None
) -> None:
    """Run between ticks: successor on death, forced closure, then a new arc if needed."""
    succeed_if_dead(state, logger)
    for quest_id, outcome in move_stalled_quests_on(state):
        print(
            f"  [quest] {quest_id}: objective {outcome} after {FORCE_PROGRESS_TICKS} stalled ticks"
        )
        print(f"  {state.history[-1].text}")
        logger.log_event(
            f"quest_{outcome}", quest=quest_id, stalled_ticks=FORCE_PROGRESS_TICKS
        )
    if _active_quests(state, state.protagonist):
        return
    pre = len(state.history)
    arc = await open_arc(state, logger, replay)
    if arc is None:
        logger.log_event("arc_failed")
        return
    for event in state.history[pre:]:
        print(f"  {event.text}  (new arc)")
    logger.log_event(
        "arc_opened",
        quest=arc.quest_title,
        places=[slugify(place.name) for place in arc.places],
        people=[slugify(person.name) for person in arc.people],
        faction=arc.faction_name or None,
    )
