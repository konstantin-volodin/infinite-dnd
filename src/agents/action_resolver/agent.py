"""Resolver: the sole writer to world state.

Public entry: `resolve(intent, state, usage=None) -> str`.
Deterministic dispatch for structured intents; an internal LLM sub-agent
(`agent`) handles free-form Action tool.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai import Agent, RunContext, ToolOutput
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits

from src.engine.state import (
    Character,
    HistoryEvent,
    WorldOperations,
    WorldState,
    resolve_character,
    resolve_location_id,
    slugify,
)
from src.agents.character.agent import remote_action_location
from src.agents.character.tools import Action, Attack, CharacterTool, Check, Speak, Travel, Wait
from src.engine.rules import DieRoller, resolve_check
from src.agents.dm.tools import Create, Modify
from src.agents.utils import create_model
from src.interface.session_log import Logger
from .context import action_resolver_context, action_resolver_system


AnyTool = CharacterTool | Create | Modify
_ACTION_USAGE = UsageLimits(request_limit=8)
_TACTICAL_GOAL_VERBS = {
    "examine",
    "explore",
    "find",
    "follow",
    "go",
    "inspect",
    "investigate",
    "look",
    "question",
    "search",
    "speak",
    "talk",
    "travel",
    "visit",
}
_GOAL_STOP_WORDS = {"about", "before", "from", "into", "that", "the", "their", "this", "with"}
_KNOWLEDGE_STOP_WORDS = {
    "about",
    "after",
    "along",
    "and",
    "before",
    "both",
    "from",
    "into",
    "not",
    "that",
    "the",
    "their",
    "this",
    "was",
    "were",
    "with",
}
_PENDING_INTENT_PREFIXES = {
    "ask",
    "asking",
    "check",
    "checking",
    "find",
    "finding",
    "go",
    "goal",
    "going",
    "head",
    "heading",
    "investigate",
    "investigating",
    "look",
    "looking",
    "objective",
    "plan",
    "question",
    "questioning",
    "search",
    "searching",
    "speak",
    "speaking",
    "talk",
    "talking",
    "track",
    "tracking",
    "travel",
    "traveling",
    "travelling",
    "try",
    "trying",
    "wait",
    "waiting",
}
_PENDING_INTENT_PHRASES = {
    ("current", "objective"),
    ("my", "current", "objective"),
    ("need", "to"),
    ("our", "current", "objective"),
    ("plan", "to"),
    ("want", "to"),
}
_EMBEDDED_PENDING_INTENT_PHRASES = {
    ("i", "must"),
    ("i", "need"),
    ("i", "need", "to"),
    ("i", "plan", "to"),
    ("i", "should"),
    ("i", "want", "to"),
    ("i", "will"),
    ("my", "next", "step"),
    ("our", "next", "step"),
    ("we", "must"),
    ("we", "need"),
    ("we", "need", "to"),
    ("we", "plan", "to"),
    ("we", "should"),
    ("we", "want", "to"),
    ("we", "will"),
}
_EMBEDDED_PENDING_INTENT_CLAUSE = re.compile(
    r"(?:^|[.!?;:—–]|\s-\s)\s*(?:now\s+)?(?:"
    r"(?:i|we)\s+(?:must|need|plan|should|want|will)\b|"
    r"(?:i|we)\s+have\b[^.!?;:—–]*\bnow\s+to\b|"
    r"(?:going|heading|traveling|travelling)\b|"
    r"check\s+back\b|"
    r"[^.!?;:—–]*\bis\s+(?:my|our)\s+(?:(?:next\s+(?:lead|target))|(?:best\s+lead))\b|"
    r"(?:need|plan|time|want)\s+to\b|"
    r"(?:this|that|it)\s+is\s+(?:my|our)\s+next\s+lead\b|"
    r"(?:this|that|it)\s+(?:is|seems)\s+worth\s+(?:asking|checking|examining|following|"
    r"investigating|looking|pursuing|questioning|searching|tracking|visiting)\b|"
    r"should\b)",
    re.IGNORECASE,
)
_EMBEDDED_TRANSIENT_ACTION_CLAUSE = re.compile(
    r"(?:[.!?;:—–]|\s-\s)\s*(?P<clause>"
    r"(?:asking|checking|confronting|examining|guarding|investigating|looking|questioning|"
    r"searching|speaking|talking|tracking|waiting|watching)\b[^.!?;:—–]*)",
    re.IGNORECASE,
)
_OUTCOME_WORDS = {
    "confirmed",
    "discovered",
    "found",
    "learned",
    "proved",
    "revealed",
    "showed",
    "uncovered",
}
_ACTION_RESULT_WORDS = _OUTCOME_WORDS - {"found"}
_DEICTIC_PLAN_OBJECTS = {"it", "that", "them", "these", "this", "those"}
_TRANSIENT_SELF_ACTION_VERBS = {
    "acted",
    "acting",
    "asked",
    "asking",
    "blocked",
    "blocking",
    "confronted",
    "confronting",
    "examined",
    "examining",
    "guarded",
    "guarding",
    "looked",
    "looking",
    "moved",
    "moving",
    "positioned",
    "positioning",
    "questioned",
    "questioning",
    "searched",
    "searching",
    "stood",
    "standing",
    "waited",
    "waiting",
    "watched",
    "watching",
}


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
    tool = _normalize_tool_ids(tool, state)
    history_size = len(state.history)
    knowledge_size = sum(len(character.knowledge) for character in state.characters.values())
    blocked_result = _dead_actor_result(tool, state)
    result = blocked_result or await _dispatch(tool, state, usage, logger, rng)
    if blocked_result is None and len(state.history) > history_size:
        _apply_self_updates(
            tool,
            state,
            resolver_recorded_knowledge=(
                sum(len(character.knowledge) for character in state.characters.values())
                > knowledge_size
            ),
        )
    if logger:
        subject = getattr(tool, "actor", None) or getattr(tool, "target_id", None) or getattr(tool, "name", None)
        logger.log_event("resolved", tool=type(tool).__name__, subject=subject, result=result)
    return result


def _dead_actor_result(tool: AnyTool, state: WorldState) -> str | None:
    """Reject every character action consistently once its actor is dead."""
    if not isinstance(tool, (Speak, Travel, Wait, Action, Attack, Check)):
        return None
    actor = state.characters.get(tool.actor)
    if actor is None or actor.stats.hp > 0:
        return None
    verb = {
        Speak: "speak",
        Travel: "travel",
        Wait: "wait",
        Action: "act",
        Attack: "attack",
        Check: "make a check",
    }[type(tool)]
    return f"Cannot {verb} — {tool.actor!r} is dead."


def _normalize_tool_ids(tool: AnyTool, state: WorldState) -> AnyTool:
    """Canonicalize entity references before deterministic operations consume them."""
    if isinstance(tool, Modify):
        target_id: str | None = None
        if tool.action == "update_quest":
            target_id = next(
                (candidate for candidate in state.quests if slugify(candidate) == slugify(tool.target_id)),
                None,
            )
        elif tool.action == "remove_npc" and (target := resolve_character(state, tool.target_id)):
            target_id = target.id
        elif tool.action == "update_location":
            target_id = resolve_location_id(state, tool.target_id)
        return tool.model_copy(update={"target_id": target_id}) if target_id else tool

    if not isinstance(tool, (Speak, Travel, Wait, Action, Attack, Check)):
        return tool

    updates: dict[str, str] = {}
    if actor := resolve_character(state, tool.actor):
        updates["actor"] = actor.id

    if isinstance(tool, Speak) and (target := resolve_character(state, tool.target)):
        updates["target"] = target.id
    elif isinstance(tool, Travel) and (destination := resolve_location_id(state, tool.destination)):
        updates["destination"] = destination
    elif isinstance(tool, Attack) and (target := resolve_character(state, tool.target)):
        updates["target"] = target.id
    elif isinstance(tool, Check) and (opponent := resolve_character(state, tool.opponent)):
        updates["opponent"] = opponent.id

    return tool.model_copy(update=updates) if updates else tool


def _apply_self_updates(
    tool: AnyTool,
    state: WorldState,
    *,
    resolver_recorded_knowledge: bool = False,
) -> None:
    """Deterministically apply a character tool's optional self-updates (remember/new_goal). No extra LLM call."""
    actor = getattr(tool, "actor", None)
    if not actor or actor not in state.characters:
        return
    ops = WorldOperations(state)
    # An Action's resolver sees the outcome and may already persist its decisive
    # fact. In that case, the character's pre-resolution memory is redundant and
    # can even contradict what actually happened.
    if (
        (remember := getattr(tool, "remember", None))
        and not (isinstance(tool, Action) and resolver_recorded_knowledge)
        and not _describes_pending_intent(remember, state)
        and not _describes_transient_self_action(state.characters[actor], remember)
        and not _repeats_recent_knowledge(state.characters[actor], remember)
        and not _conflicting_current_whereabouts(state, remember)
        and not _upgrades_workplace_to_current_whereabouts(state, remember)
    ):
        ops.add_knowledge(actor, remember)
    if new_goal := getattr(tool, "new_goal", None):
        if not _narrows_active_quest_goal(actor, new_goal, state):
            ops.set_goal(actor, new_goal)


def _knowledge_words(text: str) -> set[str]:
    """Return stable content words for conservative memory deduplication."""
    return {
        word
        for word in slugify(text).split("-")
        if len(word) > 2 and word not in _KNOWLEDGE_STOP_WORDS
    }


def _describes_pending_intent(text: str, state: WorldState | None = None) -> bool:
    """Reject action-plan notes that are not durable facts about the world."""
    words = slugify(text).split("-")
    if not words:
        return False
    if state and _describes_known_character_follow_up(state, text):
        return True
    has_embedded_intent = (
        _EMBEDDED_PENDING_INTENT_CLAUSE.search(text) is not None
        or any(
            not (set(slugify(match.group("clause")).split("-")) & _ACTION_RESULT_WORDS)
            for match in _EMBEDDED_TRANSIENT_ACTION_CLAUSE.finditer(text)
        )
        or any(
            tuple(words[index:index + len(phrase)]) == phrase
            for phrase in _EMBEDDED_PENDING_INTENT_PHRASES
            for index in range(len(words) - len(phrase) + 1)
        )
    )
    if has_embedded_intent:
        return True
    if set(words) & _OUTCOME_WORDS:
        return False
    return (
        words[0] in _PENDING_INTENT_PREFIXES
        or any(
            tuple(words[:len(phrase)]) == phrase
            for phrase in _PENDING_INTENT_PHRASES
        )
        or words[0] in {"must", "should", "will"}
    )


def _describes_known_character_follow_up(state: WorldState, text: str) -> bool:
    """Detect pending activity or a hand-off plan assigned to a known character."""
    clauses = re.split(r"[.!?;:—–]|\s-\s", text)
    for clause_index, clause in enumerate(clauses):
        words = slugify(clause).split("-")
        if words and words[0] in {"now", "then"}:
            words = words[1:]
        for character_id in state.characters:
            subject = slugify(character_id).split("-")
            if words[:len(subject)] == subject:
                remainder = words[len(subject):]
            elif subject and words[:1] == subject[:1]:
                remainder = words[1:]
            else:
                remainder = []
            if tuple(remainder[:2]) in {
                ("is", "going"),
                ("is", "heading"),
                ("is", "traveling"),
                ("is", "travelling"),
            }:
                return True
            if (
                clause_index > 0
                and tuple(remainder[:2]) in {("need", "to"), ("needs", "to")}
                and set(remainder[2:]) & _DEICTIC_PLAN_OBJECTS
            ):
                return True
    return False


def _describes_transient_self_action(character: Character, text: str) -> bool:
    """Reject self-action restatements that belong in history, not durable knowledge."""
    words = slugify(text).split("-")
    if not words or set(words) & _OUTCOME_WORDS:
        return False

    character_words = slugify(character.id).split("-")
    if words[0] in _TRANSIENT_SELF_ACTION_VERBS:
        subject_end = 0
    elif words[0] == "i":
        subject_end = 1
    elif character_words and words[:len(character_words)] == character_words:
        subject_end = len(character_words)
    elif character_words and words[0] == character_words[0]:
        subject_end = 1
    else:
        return False

    return bool(set(words[subject_end:subject_end + 5]) & _TRANSIENT_SELF_ACTION_VERBS)


def _repeats_recent_knowledge(character: Character, proposed: str) -> bool:
    """Reject a self-authored paraphrase of a fact already visible in recent context."""
    proposed_words = _knowledge_words(proposed)
    if len(proposed_words) < 6:
        return False
    for known in character.knowledge[-5:]:
        known_words = _knowledge_words(known)
        shared = len(proposed_words & known_words)
        if shared >= 6 and shared / min(len(proposed_words), len(known_words)) >= 0.5:
            return True
    return False


def _goal_words(text: str) -> set[str]:
    """Return meaningful normalized words for conservative goal comparisons."""
    return {
        word
        for word in slugify(text).split("-")
        if len(word) > 2 and word not in _GOAL_STOP_WORDS
    }


def _goal_verb(text: str) -> str:
    words = slugify(text).split("-")
    return words[0] if words else ""


def _narrows_active_quest_goal(actor: str, proposed_goal: str, state: WorldState) -> bool:
    """Reject a tactical quest step masquerading as a durable character goal."""
    character = state.characters[actor]
    current_words = _goal_words(character.goal)
    proposed_words = _goal_words(proposed_goal)
    current_verb = _goal_verb(character.goal)
    proposed_verb = _goal_verb(proposed_goal)

    for quest in state.quests.values():
        if quest.status.lower() in {"completed", "failed"}:
            continue
        topic_words = _goal_words(quest.title)
        proposed_mentions_quest = bool(topic_words & proposed_words)

        # Preserve the original owner guard for unplanned and legacy quests.
        if quest.owner == actor and current_verb == proposed_verb and proposed_mentions_quest:
            return True

        # Supporting NPCs are relevant only when their existing durable goal
        # already aligns with this quest. A genuine change such as betrayal is
        # not a tactical verb and remains allowed.
        supports_quest = quest.owner == actor or bool(topic_words & current_words)
        if not supports_quest or not proposed_mentions_quest or proposed_verb not in _TACTICAL_GOAL_VERBS:
            continue
        if any(len(proposed_words & _goal_words(objective)) >= 2 for objective in quest.plan):
            return True

    return False


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
        return WorldOperations(state).attack(tool.actor, tool.target, rng=rng)
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
    if opponent and opponent.id == actor.id:
        return "Cannot resolve check — actor and opponent must be different characters."
    if opponent and opponent.stats.hp <= 0:
        return f"Cannot resolve check — opponent {tool.opponent!r} is dead."
    if opponent and opponent.location != actor.location:
        return f"Cannot resolve check — {tool.actor!r} and {tool.opponent!r} are not in the same location."

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
    location = resolve_location_id(state, tool.location) or tool.location
    if tool.type == "location":
        connections = [location] if location else []
        return ops.add_location(slugify(tool.name), description=tool.description, connections=connections)
    if tool.type == "item":
        if not location:
            return "Cannot create item — location is required."
        return ops.create_item(tool.name, location)
    if tool.type == "npc":
        if not location:
            return "Cannot create NPC — location is required."
        return ops.reveal_character(
            slugify(tool.name),
            role=tool.role or "",
            location_id=location,
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
    if remote_location := remote_action_location(state, char, tool.description, tool.target):
        location_id, _ = remote_location
        return (
            f"Cannot resolve action at {location_id!r} — {tool.actor!r} is at "
            f"{char.location!r}. Travel there first."
        )
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
        output = " ".join(deps.effects) if deps.effects else result.output.strip()
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
    effects: list[str] = field(default_factory=list)


agent: Agent[ActionResolverDeps, str] = Agent(
    model=create_model(),
    deps_type=ActionResolverDeps,
    output_type=ToolOutput(str, name="done"),
    end_strategy="exhaustive",
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


def _apply_effect(ctx: RunContext[ActionResolverDeps], operation: Callable[[], str]) -> str:
    """Record only operation results that correspond to a real state mutation."""
    before = ctx.deps.state.model_dump()
    result = operation()
    if ctx.deps.state.model_dump() != before:
        ctx.deps.effects.append(result)
    return result


def _phrase_positions(words: list[str], phrase: list[str]) -> list[int]:
    """Return each position where a token phrase occurs contiguously."""
    if not phrase:
        return []
    return [
        index
        for index in range(len(words) - len(phrase) + 1)
        if words[index:index + len(phrase)] == phrase
    ]


def _conflicting_current_whereabouts(
    state: WorldState,
    knowledge: str,
) -> tuple[Character, str] | None:
    """Find a durable whereabouts claim that contradicts canonical character state."""
    words = slugify(knowledge).split("-")
    location_verbs = {"is", "located", "remains", "stays", "stands", "waits"}
    movement_verbs = {"heads", "heading", "moved", "moves", "traveled", "travels", "went"}
    location_links = {"at", "in", "inside", "near", "outside"}
    movement_links = {"at", "for", "into", "to", "toward", "towards"}
    current_markers = {"currently", "likely", "nearby", "now", "still"}

    for character in state.characters.values():
        character_words = slugify(character.id).split("-")
        for character_position in _phrase_positions(words, character_words):
            for location_id in state.locations:
                if location_id == character.location:
                    continue
                location_words = slugify(location_id).split("-")
                for location_position in _phrase_positions(words, location_words):
                    if location_position <= character_position:
                        continue
                    bridge = words[
                        character_position + len(character_words):location_position
                    ]
                    after_location = words[location_position + len(location_words):]
                    direct_location_claim = (
                        bool(bridge)
                        and len(bridge) <= 4
                        and (
                            bridge[0] in location_verbs
                            and bool(location_links & set(bridge[1:]))
                            or bridge[0] in movement_verbs
                            and bool(movement_links & set(bridge[1:]))
                            or bridge[:3] == ["can", "be", "found"]
                        )
                    )
                    uncertain_current_claim = (
                        bridge[:3] == ["was", "spotted", "heading"]
                        and bool(current_markers & set(after_location[:10]))
                    )
                    if direct_location_claim or uncertain_current_claim:
                        return character, location_id
    return None


def _current_location_subject_words(words: list[str], location_position: int) -> set[str]:
    """Return the nearby subject words when a phrase makes a current-location claim."""
    location_verbs = {"is", "located", "remains", "stays", "stands", "waits"}
    movement_verbs = {"heads", "heading", "moved", "moves", "traveled", "travels", "went"}
    location_links = {"at", "in", "inside", "near", "outside"}
    movement_links = {"at", "for", "into", "to", "toward", "towards"}
    start = max(0, location_position - 7)
    for verb_position in range(location_position - 1, start - 1, -1):
        bridge = words[verb_position:location_position]
        if "not" in bridge:
            continue
        direct_claim = (
            bridge[0] in location_verbs
            and bool(location_links & set(bridge[1:]))
            or bridge[0] in movement_verbs
            and bool(movement_links & set(bridge[1:]))
            or bridge[:3] == ["can", "be", "found"]
        )
        if direct_claim:
            return {
                word
                for word in words[max(0, verb_position - 3):verb_position]
                if word not in {"a", "an", "the"}
            }
    return set()


def _upgrades_workplace_to_current_whereabouts(state: WorldState, knowledge: str) -> bool:
    """Reject current presence inferred only from a recent workplace association."""
    proposed_words = re.findall(r"[a-z0-9]+", knowledge.casefold())
    workplace_verbs = {"employed", "work", "worked", "works"}
    ignored_subject_words = {"at", "in", "says", "the", "to"}

    for location_id in state.locations:
        location_words = re.findall(r"[a-z0-9]+", location_id.casefold())
        proposed_positions = _phrase_positions(proposed_words, location_words)
        if not proposed_positions:
            continue
        for proposed_position in proposed_positions:
            proposed_subject = _current_location_subject_words(
                proposed_words,
                proposed_position,
            )
            if not proposed_subject:
                continue
            for event in state.history[-12:]:
                source_words = re.findall(r"[a-z0-9]+", event.text.casefold())
                for source_position in _phrase_positions(source_words, location_words):
                    source_start = max(0, source_position - 7)
                    verb_positions = [
                        position
                        for position in range(source_start, source_position)
                        if source_words[position] in workplace_verbs
                    ]
                    if not verb_positions:
                        continue
                    workplace_position = verb_positions[-1]
                    source_subject = {
                        word
                        for word in source_words[
                            max(0, workplace_position - 3):workplace_position
                        ]
                        if word not in ignored_subject_words
                    }
                    if proposed_subject & source_subject:
                        return True
    return False


@agent.tool(sequential=True)
def remember(
    ctx: RunContext[ActionResolverDeps],
    knowledge: str,
    character_id: str | None = None,
) -> str:
    """add a concrete piece of knowledge to the acting character or another known character."""
    if ctx.deps.remembered_this_action:
        return "A decisive fact is already recorded from this action. Call done now without another remember call."
    target = ctx.deps.state.characters.get(character_id or ctx.deps.char.id)
    if _describes_pending_intent(knowledge, ctx.deps.state):
        return (
            "Cannot remember a pending plan as durable knowledge. Record only the concrete "
            "fact, then call done."
        )
    if target and _describes_transient_self_action(target, knowledge):
        return (
            "Cannot remember a transient action as durable knowledge. Record only a concrete "
            "discovery, then call done."
        )
    if conflict := _conflicting_current_whereabouts(ctx.deps.state, knowledge):
        character, claimed_location = conflict
        return (
            f"Cannot remember that {character.id} is at '{claimed_location}' — canonical state "
            f"places them at '{character.location}'. Use create_npc first if this action "
            "directly established their new location."
        )
    if _upgrades_workplace_to_current_whereabouts(ctx.deps.state, knowledge):
        return (
            "Cannot remember current whereabouts based only on a workplace association. "
            "Record the workplace fact without claiming the person is currently there."
        )
    effects_before = len(ctx.deps.effects)
    result = _apply_effect(
        ctx,
        lambda: _ops(ctx).add_knowledge(character_id or ctx.deps.char.id, knowledge),
    )
    if len(ctx.deps.effects) > effects_before:
        ctx.deps.remembered_this_action = True
    return result


@agent.tool(sequential=True)
def add_detail(
    ctx: RunContext[ActionResolverDeps],
    detail: str,
    location: str | None = None,
) -> str:
    """add a newly discovered concrete detail to the current location or another known location."""
    return _apply_effect(
        ctx,
        lambda: _ops(ctx).modify_location(location or ctx.deps.char.location, add_feature=detail),
    )


@agent.tool(sequential=True)
def discover_exit(
    ctx: RunContext[ActionResolverDeps],
    name: str,
    description: str,
    location_id: str | None = None,
    anchor_location: str | None = None,
) -> str:
    """add a newly discovered reachable location and connect it to the current place."""
    anchor = anchor_location or ctx.deps.char.location
    return _apply_effect(
        ctx,
        lambda: _ops(ctx).add_location(
            location_id or slugify(name),
            description=description,
            connections=[anchor],
        ),
    )


@agent.tool(sequential=True)
def adjust_hp(
    ctx: RunContext[ActionResolverDeps],
    delta: int,
    character_id: str | None = None,
    reason: str | None = None,
) -> str:
    """change a character's HP by a small signed amount when the action causes harm or recovery."""
    target = character_id or ctx.deps.char.id
    target_character = ctx.deps.state.characters.get(target)
    if (
        target_character
        and target_character.id != ctx.deps.char.id
        and target_character.location != ctx.deps.char.location
    ):
        return (
            f"Cannot adjust HP — '{ctx.deps.char.id}' and '{target}' "
            "are not in the same location."
        )
    ops = _ops(ctx)
    return _apply_effect(
        ctx,
        lambda: (
            ops.heal(target, delta)
            if delta >= 0
            else ops.damage(target, -delta, source_character_id=ctx.deps.char.id)
        ),
    )


@agent.tool(sequential=True)
def take(
    ctx: RunContext[ActionResolverDeps],
    item_name: str,
    character_id: str | None = None,
) -> str:
    """move an item from a location into a character's inventory."""
    target = character_id or ctx.deps.char.id
    target_character = ctx.deps.state.characters.get(target)
    if (
        target_character
        and target_character.id != ctx.deps.char.id
        and target_character.location != ctx.deps.char.location
    ):
        return (
            f"Cannot take item — '{ctx.deps.char.id}' and '{target}' "
            "are not in the same location."
        )
    return _apply_effect(
        ctx,
        lambda: _ops(ctx).take_item(target, item_name),
    )


@agent.tool(sequential=True)
def drop(
    ctx: RunContext[ActionResolverDeps],
    item_name: str,
    character_id: str | None = None,
) -> str:
    """move an item from a character's inventory into a location."""
    target = character_id or ctx.deps.char.id
    target_character = ctx.deps.state.characters.get(target)
    if (
        target_character
        and target_character.id != ctx.deps.char.id
        and target_character.location != ctx.deps.char.location
    ):
        return (
            f"Cannot drop item — '{ctx.deps.char.id}' and '{target}' "
            "are not in the same location."
        )
    return _apply_effect(
        ctx,
        lambda: _ops(ctx).drop_item(target, item_name),
    )


@agent.tool(sequential=True)
def change_item(
    ctx: RunContext[ActionResolverDeps],
    item_name: str,
    new_name: str,
) -> str:
    """rename an accessible item when the action durably changes its state, such as opening, breaking, or repairing it."""
    return _apply_effect(
        ctx,
        lambda: _ops(ctx).rename_item(ctx.deps.char.id, item_name, new_name),
    )


@agent.tool(sequential=True)
def create_item(
    ctx: RunContext[ActionResolverDeps],
    item_name: str,
    location: str | None = None,
) -> str:
    """place a new item in a location. use when the action reveals or produces a tangible object."""
    return _apply_effect(
        ctx,
        lambda: _ops(ctx).create_item(item_name, location or ctx.deps.char.location),
    )


@agent.tool(sequential=True)
def give_gold(ctx: RunContext[ActionResolverDeps], amount: int, character_id: str) -> str:
    """give some of my gold to another character — payment, bribe, tip. no item involved."""
    return _apply_effect(
        ctx,
        lambda: _ops(ctx).give_gold(ctx.deps.char.id, character_id, amount),
    )


@agent.tool(sequential=True)
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
    return _apply_effect(
        ctx,
        lambda: _ops(ctx).trade_item(buyer, seller, item_name, price),
    )


@agent.tool(sequential=True)
def create_npc(
    ctx: RunContext[ActionResolverDeps],
    name: str,
    role: str = "",
    backstory: str = "",
    location: str | None = None,
) -> str:
    """reveal an encountered NPC without inventing a durable motive for them."""
    return _apply_effect(
        ctx,
        lambda: _ops(ctx).reveal_character(
            slugify(name),
            role=role,
            location_id=location or ctx.deps.char.location,
            backstory=backstory,
        ),
    )
