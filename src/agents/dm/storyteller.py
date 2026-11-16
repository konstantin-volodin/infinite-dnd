"""Storyteller: when the hero has no active quest, one LLM call opens the next arc.

An arc is a hook event, a small new region joined to the known map, a few people,
one planned quest, and optionally a faction racing the hero. `apply_arc` is the
only writer and goes through WorldOperations; the output tool dry-runs it on a
copy so the model gets the structural errors back as a retry.
"""

from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput

from src.agents.utils import OUTPUT_RETRIES, create_model, render
from src.engine.state import (
    WorldOperations,
    WorldState,
    connected_location_ids,
    resolve_character,
    resolve_location_id,
    slugify,
)

ARC_DEADLINE_HOURS = 12  # in-world hours before an arc's faction completes its plan
_CONTEXT_PEOPLE = 8


class Place(BaseModel):
    name: str
    description: str = ""
    connects_to: str = ""  # a known location id, or the name of an earlier new place


class Person(BaseModel):
    name: str
    role: str = ""
    goal: str = ""
    location: str = ""  # one of the new places


class Arc(BaseModel):
    hook: str
    places: list[Place]
    people: list[Person]
    quest_title: str
    quest_description: str
    plan: list[str]
    faction_name: str = ""
    faction_goal: str = ""
    deadline_consequence: str = ""


@dataclass
class StorytellerDeps:
    state: WorldState


def apply_arc(state: WorldState, arc: Arc) -> list[str]:
    """Write the arc into the world. Returns structural errors (empty on success)."""
    hero = state.characters[state.protagonist]
    ops = WorldOperations(state)
    plan = [objective.strip() for objective in arc.plan if objective.strip()]
    if not 3 <= len(plan) <= 5:
        return ["plan needs 3 to 5 objectives"]
    if not 2 <= len(arc.places) <= 4:
        return ["places needs 2 to 4 new places"]
    if not arc.people:
        return ["people needs at least one person"]

    known = set(state.locations)
    new_ids: list[str] = []
    for place in arc.places:
        place_id = slugify(place.name)
        anchor = resolve_location_id(state, place.connects_to)
        if not new_ids and anchor not in known:
            anchor = hero.location  # the region must join the map the hero knows
        elif anchor is None:
            anchor = new_ids[-1]
        result = ops.add_location(place_id, place.description, connections=[anchor])
        if not result.endswith("added."):
            return [f"place '{place.name}': {result} Invent a new place."]
        new_ids.append(place_id)

    if not any(
        f"-{place_id}-" in f"-{slugify(o)}-" for o in plan for place_id in new_ids
    ):
        return [f"name one of the new places ({', '.join(new_ids)}) in an objective"]

    ops.world_event(arc.hook.strip(), hero.location)
    for person in arc.people:
        existing = resolve_character(state, person.name)
        if existing and existing.id == hero.id:
            continue
        location = resolve_location_id(state, person.location)
        result = ops.reveal_character(
            existing.id if existing else slugify(person.name),
            role=person.role,
            location_id=location if location in new_ids else new_ids[0],
            goal=person.goal,
        )
        if result.startswith("Cannot"):
            return [f"person '{person.name}': {result}"]

    quest_id = slugify(arc.quest_title)
    result = ops.add_quest(
        quest_id, arc.quest_title, arc.quest_description, owner=hero.id, plan=plan
    )
    if not result.endswith("added."):
        return [f"quest: {result} Pick a different quest."]
    hero.goal = arc.quest_description

    if arc.faction_name.strip() and arc.deadline_consequence.strip():
        result = ops.add_faction(
            slugify(arc.faction_name),
            arc.faction_name,
            arc.faction_goal or arc.quest_description,
            arc.deadline_consequence,
            ARC_DEADLINE_HOURS,
            fail_quest_id=quest_id,
        )
        if not result.endswith("added."):
            return [f"faction: {result}"]
    return []


def report(
    ctx: RunContext[StorytellerDeps],
    hook: str,
    places: list[Place],
    people: list[Person],
    quest_title: str,
    quest_description: str,
    plan: list[str],
    faction_name: str = "",
    faction_goal: str = "",
    deadline_consequence: str = "",
) -> Arc:
    """open the next arc of the campaign."""
    arc = Arc(
        hook=hook,
        places=places,
        people=people,
        quest_title=quest_title,
        quest_description=quest_description,
        plan=plan,
        faction_name=faction_name,
        faction_goal=faction_goal,
        deadline_consequence=deadline_consequence,
    )
    trial = ctx.deps.state.model_copy(deep=True)
    errors = apply_arc(trial, arc)
    if errors:
        raise ModelRetry("; ".join(errors))
    return arc


def storyteller_context(state: WorldState) -> str:
    hero = state.characters[state.protagonist]
    ended = [q for q in state.quests.values() if q.status != "active"][-3:]
    people = sorted(
        (
            c
            for c in state.characters.values()
            if c.id != hero.id
            and c.stats.hp > 0
            and (
                c.id in hero.relationships
                or hero.id in c.relationships
                or c.location == hero.location
            )
        ),
        key=lambda c: (c.id not in hero.relationships, c.id),
    )[:_CONTEXT_PEOPLE]
    return render(
        "dm/storyteller_state.jinja",
        hero=hero,
        arc_number=len(ended) + 1,
        chronicle=state.chronicle,
        recent_events=[e.text for e in state.history[-12:]],
        ended=ended,
        people=people,
        gateways=[hero.location, *connected_location_ids(state, hero.location)][:6],
    )


agent: Agent[StorytellerDeps, Arc] = Agent(
    model=create_model(),
    deps_type=StorytellerDeps,
    output_type=ToolOutput(report, name="report"),
    output_retries=OUTPUT_RETRIES,
    instructions="You open the next arc of an ongoing D&D campaign. Call `report` once.",
)


@agent.system_prompt
def _identity(_: RunContext[StorytellerDeps]) -> str:
    return render("dm/storyteller_identity.jinja")


@agent.instructions
def _context(ctx: RunContext[StorytellerDeps]) -> str:
    return storyteller_context(ctx.deps.state)
