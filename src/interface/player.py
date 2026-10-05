"""Console PC controller — a human plays the acting character's turn from stdin.

Drop-in for `pc_controller` in `src.engine.runtime.loop`: same signature, same
`CharacterTool` output, so the rest of the engine (resolver, DM, replay
recording) doesn't need to know a human is behind the wheel.
"""

import asyncio
from typing import cast

from src.agents.character.tools import Ability, Action, Attack, CharacterTool, Check, Speak, Travel, Wait
from src.engine.rules import get_health_status
from src.engine.state import (
    WorldState,
    characters_in_location,
    connected_location_ids,
    quest_deadline_clocks,
    resolve_character,
    resolve_location_id,
)

_HELP = "Type a plain sentence to act, or: /check <ability> <DC> <action> [vs <character>] | /speak ... | /travel ... | /attack ... | /wait"

_ABILITIES: set[str] = {"strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"}


class _InputError(ValueError):
    """A malformed or unresolvable command — caller prints it and reprompts."""


def _describe_situation(actor_id: str, state: WorldState, *, include_help: bool = True) -> str:
    char = state.characters[actor_id]
    loc = state.locations.get(char.location)
    lines = [f"\n=== Your turn: {actor_id} ==="]
    if loc:
        lines.append(f"Location: {loc.id} — {loc.description}" if loc.description else f"Location: {loc.id}")
        if loc.items:
            lines.append(f"Items here: {', '.join(loc.items)}")
    lines.append(
        f"HP: {char.stats.hp}/{char.stats.max_hp}  Gold: {char.stats.gold}  "
        f"Inventory: {', '.join(char.inventory) if char.inventory else 'none'}"
    )
    if char.goal:
        lines.append(f"Goal: {char.goal}")

    active_quests = [
        quest for quest in state.quests.values()
        if quest.owner == actor_id and quest.status.lower() not in {"completed", "failed"}
    ]
    for quest in active_quests:
        objective = (
            quest.plan[quest.current_step]
            if quest.current_step < len(quest.plan)
            else quest.description
        )
        if objective:
            lines.append(f"Current objective ({quest.title}): {objective}")

    active_quest_ids = {quest.id for quest in active_quests}
    for _, clock in quest_deadline_clocks(state, active_quest_ids):
        lines.append(
            f"Deadline ({clock.name}): {clock.progress}/{clock.segments} — "
            f"{clock.consequence}"
        )

    witnessed = [event.text for event in state.history if actor_id in event.characters]
    if witnessed:
        lines.append(f"Just happened: {witnessed[-1]}")

    present = characters_in_location(state, char.location, exclude_character_id=actor_id)
    if present:
        lines.append("Present: " + ", ".join(f"{c.id} ({get_health_status(c)})" for c in present))

    connections = connected_location_ids(state, char.location)
    if connections:
        lines.append("Connections: " + ", ".join(connections))

    if include_help:
        lines.append(f"\n{_HELP}")
    return "\n".join(lines)


def _split_speak_target(state: WorldState, rest: str) -> tuple[str | None, str]:
    """`/speak bob hello there` -> target 'bob', message 'hello there'; no match leaves it untargeted."""
    head, _, tail = rest.partition(" ")
    target = resolve_character(state, head)
    if target and tail:
        return target.id, tail
    return None, rest


def _parse_intent(actor_id: str, state: WorldState, line: str) -> CharacterTool:
    line = line.strip()
    if not line:
        raise _InputError(_HELP)
    if not line.startswith("/"):
        return Action(actor=actor_id, description=line)

    command, _, rest = line[1:].partition(" ")
    command, rest = command.lower(), rest.strip()

    if command == "wait":
        return Wait(actor=actor_id)
    if command == "travel":
        if not rest:
            raise _InputError("Usage: /travel <destination>")
        destination = resolve_location_id(state, rest)
        if destination is None:
            raise _InputError(f"Unknown destination '{rest}'.")
        return Travel(actor=actor_id, destination=destination)
    if command == "attack":
        if not rest:
            raise _InputError("Usage: /attack <target>")
        target = resolve_character(state, rest)
        if target is None:
            raise _InputError(f"Unknown target '{rest}'.")
        return Attack(actor=actor_id, target=target.id)
    if command == "check":
        parts = rest.split(maxsplit=2)
        if len(parts) < 3 or parts[0].lower() not in _ABILITIES:
            raise _InputError("Usage: /check <ability> <DC> <action> [vs <character>]")
        try:
            difficulty = int(parts[1])
        except ValueError as exc:
            raise _InputError("Check DC must be a number from 1 to 30.") from exc
        description = parts[2]
        opponent_id = None
        if " vs " in description:
            description, _, opponent_text = description.rpartition(" vs ")
            opponent = resolve_character(state, opponent_text)
            if opponent is None:
                raise _InputError(f"Unknown opponent '{opponent_text}'.")
            opponent_id = opponent.id
        try:
            return Check(
                actor=actor_id, ability=cast(Ability, parts[0].lower()), difficulty=difficulty,
                description=description.strip(), opponent=opponent_id,
            )
        except ValueError as exc:
            raise _InputError("Check DC must be a number from 1 to 30.") from exc
    if command == "speak":
        if not rest:
            raise _InputError("Usage: /speak [target] <message>")
        target_id, message = _split_speak_target(state, rest)
        return Speak(actor=actor_id, message=message, target=target_id)

    raise _InputError(f"Unknown command '/{command}'. {_HELP}")


async def console_pc_controller(actor_id: str, state: WorldState) -> CharacterTool:
    """Human-in-the-loop `pc_controller`: prints the situation, reprompts until input parses."""
    print(_describe_situation(actor_id, state), flush=True)
    while True:
        line = await asyncio.to_thread(input, "> ")
        try:
            return _parse_intent(actor_id, state, line)
        except _InputError as exc:
            print(f"  {exc}", flush=True)
