"""Director agent: fires when the story stalls — one LLM call that introduces a single complication."""

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext, ToolOutput

from src.engine.state import WorldState
from src.agents.utils import create_model
from .agent import NewEntity
from .tools import Create
from .context import director_context, director_system


@dataclass
class DirectorDeps:
    state: WorldState
    location_id: str


@dataclass
class DirectorResult:
    event: str
    create: Create | None = None
    quest_id: str | None = None


def director_output(
    _: RunContext[DirectorDeps],
    event: str,
    entity: NewEntity | None = None,
    quest_id: str | None = None,
) -> DirectorResult:
    """introduce exactly one complication: a narrative event, optionally one required new entity, tied to a quest when possible."""
    create = entity.to_create() if entity and entity.name.strip() else None
    return DirectorResult(event=event.strip(), create=create, quest_id=quest_id)


agent: Agent[DirectorDeps, DirectorResult] = Agent(
    model=create_model(),
    deps_type=DirectorDeps,
    output_type=ToolOutput(director_output, name="report"),
    instructions="The story has stalled. You introduce exactly one concrete complication grounded in the existing world.",
)


@agent.system_prompt
def _identity(_: RunContext[DirectorDeps]) -> str:
    return director_system()


@agent.instructions
def _context(ctx: RunContext[DirectorDeps]) -> str:
    return director_context(ctx.deps)
