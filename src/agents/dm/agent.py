"""DM agent: one LLM call per tick that enriches the world, reviews quests, and estimates time."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ToolOutput

from src.engine.state import WorldState
from src.agents.utils import create_model
from .tools import Create, Modify
from .context import dm_context, dm_system


@dataclass
class DMDeps:
    state: WorldState
    context_events: list[str] = field(default_factory=list)
    new_events: list[str] = field(default_factory=list)


class NewEntity(BaseModel):
    type: Literal["location", "npc", "item", "quest"]
    name: str
    description: str = ""
    location: str | None = None
    role: str | None = None
    goal: str | None = None
    owner: str | None = None
    plan: list[str] | None = None  # quests only: 3-5 concrete, ordered objectives

    def to_create(self) -> Create:
        return Create(**self.model_dump())


class QuestUpdate(BaseModel):
    quest_id: str
    new_status: str | None = None
    step: str | None = None
    advance: bool = False  # the quest's CURRENT objective was clearly accomplished this turn


class RelationshipUpdate(BaseModel):
    character_id: str  # whose opinion this is
    target_id: str  # who the opinion is about
    relation: str  # free-form disposition, e.g. "hostile — saw me steal from the shrine"


class ClockAdvance(BaseModel):
    faction_id: str
    clock_id: str


@dataclass
class DMResult:
    creates: list[Create]
    modifies: list[Modify]
    minutes: list[int]


def dm_output(
    _: RunContext[DMDeps],
    entities: list[NewEntity],
    quest_updates: list[QuestUpdate],
    relationship_updates: list[RelationshipUpdate],
    clock_advances: list[ClockAdvance],
    minutes: list[int],
) -> DMResult:
    """register new entities, report quest, relationship, and faction-clock changes, and estimate minutes elapsed per new event."""
    creates = [e.to_create() for e in entities if e.name.strip()]
    quest_modifies = [
        Modify(action="update_quest", target_id=u.quest_id, status=u.new_status, step=u.step, advance=u.advance)
        for u in quest_updates
        if u.new_status or u.step or u.advance
    ]
    relationship_modifies = [
        Modify(action="update_relationship", target_id=u.character_id, other_id=u.target_id, reason=u.relation)
        for u in relationship_updates
        if u.relation.strip() and u.character_id != u.target_id
    ]
    clock_modifies = [
        Modify(action="advance_faction_clock", target_id=c.faction_id, other_id=c.clock_id)
        for c in clock_advances
        if c.faction_id.strip() and c.clock_id.strip()
    ]
    clamped_minutes = [max(0, min(int(m), 1440)) for m in minutes]
    return DMResult(creates=creates, modifies=quest_modifies + relationship_modifies + clock_modifies, minutes=clamped_minutes)


agent: Agent[DMDeps, DMResult] = Agent(
    model=create_model(),
    deps_type=DMDeps,
    output_type=ToolOutput(dm_output, name="report"),
    instructions="You maintain the world after each turn: register new entities, advance quests, adjust relationships, and estimate time elapsed.",
)


@agent.system_prompt
def _identity(_: RunContext[DMDeps]) -> str:
    return dm_system()


@agent.instructions
def _context(ctx: RunContext[DMDeps]) -> str:
    return dm_context(ctx.deps)
