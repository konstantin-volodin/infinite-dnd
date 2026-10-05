"""Character agent: emits typed intents. Does not mutate state."""

from dataclasses import dataclass, replace

from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput
from pydantic_ai.tools import ToolDefinition

from src.engine.state import Character, WorldState, characters_in_location, connected_location_ids
from src.agents.utils import create_model
from .context import character_context, character_system
from .tools import Ability, Action, Attack, CharacterTool, Check, Speak, Travel, Wait


@dataclass
class CharacterDeps:
    char: Character
    state: WorldState


def _living_targets(ctx: RunContext[CharacterDeps]) -> list[str]:
    """Living characters sharing the actor's location — valid speak/attack targets."""
    return [
        c.id for c in characters_in_location(ctx.deps.state, ctx.deps.char.location, exclude_character_id=ctx.deps.char.id)
        if c.stats.hp > 0
    ]


def _travel_options(ctx: RunContext[CharacterDeps]) -> list[str]:
    return connected_location_ids(ctx.deps.state, ctx.deps.char.location)


def _prepare_output_tools(
    ctx: RunContext[CharacterDeps],
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    targets = _living_targets(ctx)
    options = _travel_options(ctx)

    result: list[ToolDefinition] = []
    for td in tool_defs:
        if td.name == "speak":
            hint = f"Valid targets: {', '.join(targets)}." if targets else "No one else here — leave target empty."
            result.append(replace(td, description=f"say something short — 1-2 sentences, like real speech. {hint}"))
        elif td.name == "travel":
            if options:
                result.append(replace(td, description=f"travel to a connected location. Valid ids: {', '.join(options)}."))
            else:
                result.append(replace(td, description="no connected locations are available; use action to discover one."))
        elif td.name == "attack":
            if targets:
                result.append(replace(td, description=f"attack someone here. Valid targets: {', '.join(targets)}."))
        else:
            result.append(td)
    return result


def speak_output(
    ctx: RunContext[CharacterDeps],
    message: str,
    target: str | None = None,
    remember: str | None = None,
    new_goal: str | None = None,
) -> Speak:
    """say something short — 1-2 sentences, like real speech. can be targeted dialogue or thinking out loud."""
    targets = _living_targets(ctx)
    if target and target not in targets:
        hint = f"Valid targets: {', '.join(targets)}." if targets else "No one else is here."
        raise ModelRetry(f"Cannot speak to {target!r}. {hint}")
    return Speak(actor=ctx.deps.char.id, message=message, target=target, remember=remember, new_goal=new_goal)


def travel_output(
    ctx: RunContext[CharacterDeps],
    location: str,
    remember: str | None = None,
    new_goal: str | None = None,
) -> Travel:
    """travel to a connected location using its exact id."""
    options = _travel_options(ctx)
    if location not in options:
        hint = (
            f"Valid location ids: {', '.join(options)}."
            if options
            else "No connected locations are available; use action to discover one."
        )
        raise ModelRetry(f"Cannot travel to {location!r}. {hint}")
    return Travel(actor=ctx.deps.char.id, destination=location, remember=remember, new_goal=new_goal)


def wait_output(
    ctx: RunContext[CharacterDeps],
    remember: str | None = None,
    new_goal: str | None = None,
) -> Wait:
    """do nothing for now and wait to see what happens next."""
    return Wait(actor=ctx.deps.char.id, remember=remember, new_goal=new_goal)


def action_output(
    ctx: RunContext[CharacterDeps],
    description: str,
    target: str | None = None,
    remember: str | None = None,
    new_goal: str | None = None,
) -> Action:
    """take a concrete action — search, examine, attempt, interact, discover. describe what and how in one plain sentence — no flourishes. use this when you can make progress, not just when speak/travel don't fit."""
    return Action(actor=ctx.deps.char.id, description=description, target=target, remember=remember, new_goal=new_goal)


def attack_output(
    ctx: RunContext[CharacterDeps],
    target: str,
    remember: str | None = None,
    new_goal: str | None = None,
) -> Attack:
    """attack a character here. a serious, violent act with lasting consequences."""
    targets = _living_targets(ctx)
    if target not in targets:
        hint = f"Valid targets: {', '.join(targets)}." if targets else "No one here to attack."
        raise ModelRetry(f"Cannot attack {target!r}. {hint}")
    return Attack(actor=ctx.deps.char.id, target=target, remember=remember, new_goal=new_goal)


def check_output(
    ctx: RunContext[CharacterDeps],
    ability: Ability,
    description: str,
    difficulty: int = 10,
    modifier: int = 0,
    opponent: str | None = None,
    opposing_modifier: int = 0,
    remember: str | None = None,
    new_goal: str | None = None,
) -> Check:
    """attempt a risky action with a d20 check; use opponent only for a direct contest."""
    if opponent and opponent not in _living_targets(ctx):
        raise ModelRetry(f"Cannot contest {opponent!r}; they are not a living character here.")
    return Check(
        actor=ctx.deps.char.id, ability=ability, description=description, difficulty=difficulty,
        modifier=modifier, opponent=opponent, opposing_modifier=opposing_modifier,
        remember=remember, new_goal=new_goal,
    )


CHARACTER_RESPONSE_OUTPUTS = [
    ToolOutput(speak_output, name="speak"),
    ToolOutput(wait_output, name="wait"),
]

agent: Agent[CharacterDeps, CharacterTool] = Agent(
    model=create_model(),
    deps_type=CharacterDeps,
    output_type=[
        ToolOutput(action_output, name="action"),
        ToolOutput(check_output, name="check"),
        ToolOutput(speak_output, name="speak"),
        ToolOutput(travel_output, name="travel"),
        ToolOutput(attack_output, name="attack"),
        ToolOutput(wait_output, name="wait"),
    ],
    prepare_output_tools=_prepare_output_tools,
    instructions="You are a character in a D&D game.",
)


@agent.system_prompt
def identity(ctx: RunContext[CharacterDeps]) -> str:
    return character_system(ctx.deps.char)


@agent.instructions
def context(ctx: RunContext[CharacterDeps]) -> str:
    return character_context(ctx.deps.char, ctx.deps.state)
