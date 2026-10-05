"""Resolver: the sole writer to world state.

Public entry: `resolve(intent, state, usage=None) -> str`.
Deterministic dispatch for structured intents; an internal LLM sub-agent
(`agent`) handles free-form Action tool.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent, RunContext, ToolOutput
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits

from src.engine.state import (
    Character,
    HistoryEvent,
    WorldOperations,
    WorldState,
    slugify,
)
from src.agents.character.tools import Action, Attack, CharacterTool, Check, Speak, Travel, Wait
from src.engine.rules import DieRoller, resolve_check
from src.agents.dm.tools import Create, Modify
from src.agents.utils import create_model
from src.interface.session_log import Logger
from .context import action_resolver_context, action_resolver_system


AnyTool = CharacterTool | Create | Modify
_ACTION_USAGE = UsageLimits(request_limit=8)


# ============================================================
# Public entry
# ============================================================

async def resolve(
    tool: AnyTool,
    state: WorldState,
    usage: RunUsage | None = None,
    logger: Logger | None = None,
    rng: DieRoller | None = None,
) -> str:
    """Execute a tool call against world state. Single writer surface."""
    result = await _dispatch(tool, state, usage, logger, rng)
    _apply_self_updates(tool, state)
    if logger:
        subject = getattr(tool, "actor", None) or getattr(tool, "target_id", None) or getattr(tool, "name", None)
        logger.log_event("resolved", tool=type(tool).__name__, subject=subject, result=result)
    return result


def _apply_self_updates(tool: AnyTool, state: WorldState) -> None:
    """Deterministically apply a character tool's optional self-updates (remember/new_goal). No extra LLM call."""
    actor = getattr(tool, "actor", None)
    if not actor or actor not in state.characters:
        return
    char = state.characters[actor]
    ops = WorldOperations(state)
    if remember := getattr(tool, "remember", None):
        ops.add_knowledge(actor, remember)
    if new_goal := getattr(tool, "new_goal", None):
        current_verb = char.goal.strip().casefold().split(maxsplit=1)[0] if char.goal.strip() else ""
        proposed_verb = new_goal.strip().casefold().split(maxsplit=1)[0] if new_goal.strip() else ""
        quest_words = {
            word.strip(".,!?;:'\"")
            for quest in state.quests.values()
            if quest.owner == actor and quest.status.lower() not in {"completed", "failed"}
            for word in quest.title.casefold().split()
            if len(word.strip(".,!?;:'\"")) > 2
        }
        proposed_words = {word.strip(".,!?;:'\"") for word in new_goal.casefold().split()}
        restates_active_quest = current_verb == proposed_verb and bool(quest_words & proposed_words)
        if not restates_active_quest:
            ops.set_goal(actor, new_goal)


async def _dispatch(
    tool: AnyTool, state: WorldState, usage: RunUsage | None, logger: Logger | None, rng: DieRoller | None
) -> str:
    if isinstance(tool, Speak):
        return WorldOperations(state).speak(tool.actor, tool.message, tool.target)
    if isinstance(tool, Travel):
        return WorldOperations(state).move_character(tool.actor, tool.destination)
    if isinstance(tool, Wait):
        return _resolve_wait(tool, state)
    if isinstance(tool, Attack):
        return WorldOperations(state).attack(tool.actor, tool.target)
    if isinstance(tool, Check):
        return _resolve_check(tool, state, rng)
    if isinstance(tool, Create):
        return _resolve_create(tool, state)
    if isinstance(tool, Modify):
        return _resolve_modify(tool, state)
    if isinstance(tool, Action):
        return await _resolve_action(tool, state, usage, logger)
    raise TypeError(f"Unknown tool: {tool!r}")


# ============================================================
# Deterministic dispatch
# ============================================================

def _resolve_wait(tool: Wait, state: WorldState) -> str:
    actor = state.characters.get(tool.actor)
    if actor is None:
        return f"Cannot wait — character {tool.actor!r} not found."
    if actor.stats.hp <= 0:
        return f"Cannot wait — {tool.actor!r} is dead."

    recovered = max(0, min(1, actor.stats.max_hp - actor.stats.hp))
    if recovered:
        actor.stats.hp += recovered
        text = (
            f"{tool.actor} catches their breath and recovers {recovered} HP. "
            f"HP: {actor.stats.hp}/{actor.stats.max_hp}."
        )
    else:
        text = f"{tool.actor} waits."
    state.history.append(HistoryEvent(text=text, location=actor.location, characters=[tool.actor]))
    return text


def _resolve_check(tool: Check, state: WorldState, rng: DieRoller | None) -> str:
    actor = state.characters.get(tool.actor)
    if actor is None:
        return f"Cannot resolve check — character {tool.actor!r} not found."
    opponent = state.characters.get(tool.opponent) if tool.opponent else None
    if tool.opponent and opponent is None:
        return f"Cannot resolve check — opponent {tool.opponent!r} not found."

    result = resolve_check(
        tool.difficulty,
        tool.modifier,
        opposing_modifier=tool.opposing_modifier if opponent else None,
        rng=rng,
    )
    outcome = "succeeds" if result.success else "fails"
    detail = f"{result.roll}{result.modifier:+d}={result.total} vs DC {tool.difficulty}"
    characters = [tool.actor]
    if opponent:
        detail = (
            f"{result.roll}{result.modifier:+d}={result.total} vs "
            f"{opponent.id} {result.opposing_roll}{result.opposing_modifier:+d}={result.opposing_total}"
        )
        characters.append(opponent.id)
    text = f"{tool.actor} {outcome}: {tool.description} [{tool.ability}; {detail}]."
    state.history.append(HistoryEvent(text=text, location=actor.location, characters=characters))
    return text


def _resolve_create(tool: Create, state: WorldState) -> str:
    ops = WorldOperations(state)
    if tool.type == "location":
        connections = [tool.location] if tool.location else []
        return ops.add_location(slugify(tool.name), description=tool.description, connections=connections)
    if tool.type == "item":
        if not tool.location:
            return "Cannot create item — location is required."
        return ops.create_item(tool.name, tool.location)
    if tool.type == "npc":
        if not tool.location:
            return "Cannot create NPC — location is required."
        return ops.spawn_character(
            slugify(tool.name),
            role=tool.role or "",
            location_id=tool.location,
            backstory=tool.description,
            goal=tool.goal or "",
        )
    if tool.type == "quest":
        return ops.add_quest(
            slugify(tool.name),
            title=tool.name,
            description=tool.description,
            owner=tool.owner,
            plan=tool.plan,
        )
    return f"Unknown create type: {tool.type!r}."


def _resolve_modify(tool: Modify, state: WorldState) -> str:
    ops = WorldOperations(state)
    if tool.action == "update_quest":
        if not tool.status and not tool.step and not tool.advance:
            return "Cannot update a quest without status, step, or advance."
        return ops.advance_quest(tool.target_id, new_status=tool.status, step=tool.step, advance=tool.advance)
    if tool.action == "remove_npc":
        return ops.delete_npc(tool.target_id, reason=tool.reason or "")
    if tool.action == "update_location":
        return ops.modify_location(tool.target_id, description=tool.reason)
    if tool.action == "update_relationship":
        if not tool.other_id or not tool.reason:
            return "Cannot update relationship — other_id and reason are required."
        return ops.update_relationship(tool.target_id, tool.other_id, tool.reason)
    if tool.action == "advance_faction_clock":
        if not tool.other_id:
            return "Cannot advance faction clock — other_id (the clock id) is required."
        return ops.advance_faction_clock(tool.target_id, tool.other_id)
    return f"Unknown modify action: {tool.action!r}."


async def _resolve_action(tool: Action, state: WorldState, usage: RunUsage | None, logger: Logger | None) -> str:
    char = state.characters.get(tool.actor)
    if not char:
        return f"Cannot resolve action — character {tool.actor!r} not found."
    history_size = len(state.history)
    prompt = f"Resolve this action: {tool.description}"
    if tool.target:
        prompt += f" (target: {tool.target})"
    deps = ActionResolverDeps(char=char, state=state, description=tool.description, target=tool.target)
    try:
        if logger:
            with logger.run("action_resolver"):
                result = await agent.run(prompt, deps=deps, usage=usage, usage_limits=_ACTION_USAGE)
                logger.log_messages("action_resolver", result.all_messages())
        else:
            result = await agent.run(prompt, deps=deps, usage=usage, usage_limits=_ACTION_USAGE)
        output = result.output.strip()
    except UsageLimitExceeded:
        output = (
            f"{char.id}'s action produced the recorded change."
            if len(state.history) > history_size
            else f"{char.id} makes no further progress on that action."
        )
    if output and len(state.history) == history_size:
        state.history.append(HistoryEvent(
            text=output,
            location=char.location,
            characters=[char.id],
        ))
    return output


# ============================================================
# Internal LLM sub-agent for free-form Action tool
# ============================================================

@dataclass
class ActionResolverDeps:
    char: Character
    state: WorldState
    description: str
    target: str | None = None
    remembered_this_action: bool = False


agent: Agent[ActionResolverDeps, str] = Agent(
    model=create_model(),
    deps_type=ActionResolverDeps,
    output_type=ToolOutput(str, name="done"),
    instructions="Resolve exactly one character action into concrete state changes. Report the outcome in one short, plain sentence — no scene-setting or flourishes.",
)


@agent.system_prompt
def _identity(_: RunContext[ActionResolverDeps]) -> str:
    return action_resolver_system()


@agent.instructions
def _context(ctx: RunContext[ActionResolverDeps]) -> str:
    return action_resolver_context(
        ctx.deps.char,
        ctx.deps.state,
        description=ctx.deps.description,
        target=ctx.deps.target,
    )


def _ops(ctx: RunContext[ActionResolverDeps]) -> WorldOperations:
    return WorldOperations(ctx.deps.state)


@agent.tool
def remember(
    ctx: RunContext[ActionResolverDeps],
    knowledge: str,
    character_id: str | None = None,
) -> str:
    """add a concrete piece of knowledge to the acting character or another known character."""
    if ctx.deps.remembered_this_action:
        return "A decisive fact is already recorded from this action. Call done now without another remember call."
    ctx.deps.remembered_this_action = True
    return _ops(ctx).add_knowledge(character_id or ctx.deps.char.id, knowledge)


@agent.tool
def add_detail(
    ctx: RunContext[ActionResolverDeps],
    detail: str,
    location: str | None = None,
) -> str:
    """add a newly discovered concrete detail to the current location or another known location."""
    return _ops(ctx).modify_location(location or ctx.deps.char.location, add_feature=detail)


@agent.tool
def discover_exit(
    ctx: RunContext[ActionResolverDeps],
    name: str,
    description: str,
    location_id: str | None = None,
    anchor_location: str | None = None,
) -> str:
    """add a newly discovered reachable location and connect it to the current place."""
    anchor = anchor_location or ctx.deps.char.location
    return _ops(ctx).add_location(location_id or slugify(name), description=description, connections=[anchor])


@agent.tool
def adjust_hp(
    ctx: RunContext[ActionResolverDeps],
    delta: int,
    character_id: str | None = None,
    reason: str | None = None,
) -> str:
    """change a character's HP by a small signed amount when the action causes harm or recovery."""
    target = character_id or ctx.deps.char.id
    ops = _ops(ctx)
    return ops.heal(target, delta) if delta >= 0 else ops.damage(target, -delta)


@agent.tool
def update_quest(
    ctx: RunContext[ActionResolverDeps],
    quest_id: str,
    status: str,
) -> str:
    """update a quest status when the action clearly advances or completes it."""
    return _ops(ctx).advance_quest(quest_id, new_status=status)


@agent.tool
def take(
    ctx: RunContext[ActionResolverDeps],
    item_name: str,
    character_id: str | None = None,
) -> str:
    """move an item from a location into a character's inventory."""
    return _ops(ctx).take_item(character_id or ctx.deps.char.id, item_name)


@agent.tool
def drop(
    ctx: RunContext[ActionResolverDeps],
    item_name: str,
    character_id: str | None = None,
) -> str:
    """move an item from a character's inventory into a location."""
    return _ops(ctx).drop_item(character_id or ctx.deps.char.id, item_name)


@agent.tool
def create_item(
    ctx: RunContext[ActionResolverDeps],
    item_name: str,
    location: str | None = None,
) -> str:
    """place a new item in a location. use when the action reveals or produces a tangible object."""
    return _ops(ctx).create_item(item_name, location or ctx.deps.char.location)


@agent.tool
def give_gold(ctx: RunContext[ActionResolverDeps], amount: int, character_id: str) -> str:
    """give some of my gold to another character — payment, bribe, tip. no item involved."""
    return _ops(ctx).give_gold(ctx.deps.char.id, character_id, amount)


@agent.tool
def trade_item(
    ctx: RunContext[ActionResolverDeps],
    item_name: str,
    price: int,
    counterparty_id: str,
    role: Literal["buyer", "seller"] = "buyer",
) -> str:
    """exchange an item for gold with another character here. role='buyer': I pay for their item. role='seller': they pay for mine."""
    me = ctx.deps.char.id
    buyer, seller = (me, counterparty_id) if role == "buyer" else (counterparty_id, me)
    return _ops(ctx).trade_item(buyer, seller, item_name, price)


@agent.tool
def create_npc(
    ctx: RunContext[ActionResolverDeps],
    name: str,
    role: str = "",
    goal: str = "",
    backstory: str = "",
    location: str | None = None,
) -> str:
    """add a new NPC to the world. use when the action reveals or encounters a new character."""
    return _ops(ctx).spawn_character(
        slugify(name),
        role=role,
        location_id=location or ctx.deps.char.location,
        backstory=backstory,
        goal=goal,
    )
