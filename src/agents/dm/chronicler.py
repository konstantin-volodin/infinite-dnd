"""Chronicler agent: one LLM call that compresses archived history into one chronicle entry."""

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from src.agents.utils import create_model


@dataclass
class ChroniclerDeps:
    events: list[str]


agent: Agent[ChroniclerDeps, str] = Agent(
    model=create_model(),
    deps_type=ChroniclerDeps,
    output_type=str,
    instructions=(
        "You compress a run of past D&D events into one short chronicle entry — the kind of "
        "backstory summary a storyteller recalls, not a blow-by-blow log. Preserve names, "
        "outcomes, and causal threads; drop dialogue and minor detail. Two to four sentences."
    ),
)


@agent.instructions
def _events(ctx: RunContext[ChroniclerDeps]) -> str:
    lines = "\n".join(f"- {event}" for event in ctx.deps.events)
    return f"## events to compress\n{lines}"
