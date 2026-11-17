"""Game loop. Each tick: one actor takes a turn → resolve → stamp elapsed time → enrich world → review quests."""

import asyncio
import os
import re
import sys
from collections.abc import Awaitable, Callable
from io import TextIOWrapper

from pydantic_ai.exceptions import AgentRunError, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from src.engine.state import (
    StateManager,
    WorldOperations,
    WorldState,
    is_dialogue,
    find_quest,
    named_ids,
    resolve_character,
    route,
    slugify,
)
from src.engine.state.models import HistoryEvent, Quest
from src.agents.character.agent import CharacterDeps, agent as character_agent
from src.agents.character.tools import (
    Action,
    Attack,
    CharacterTool,
    Check,
    Speak,
    Travel,
    Wait,
)
from src.agents.dm.agent import DMDeps, DMResult, agent as dm_agent
from src.agents.dm.tools import Create, Modify
from src.agents.dm.director import DirectorDeps, agent as director_agent
from src.agents.action_resolver.agent import resolve
from src.agents.server import read_metrics
from src.interface.session_log import Logger
from src.interface.world_state import format_clock
from .campaign import campaign_upkeep, succeed_if_dead
from .chronicle import compact_history
from .replay import ReplayTape


_AGENT_USAGE = UsageLimits(request_limit=12)
_DM_USAGE = UsageLimits(request_limit=16)

# A failed tick (agent error, no character action) is rolled back and retried;
# the campaign only gives up after this many in a row, backing off between them.
MAX_FAILED_TICKS = 5
FAILED_TICK_BACKOFF_S = 10


class _SkippedTick(Exception):
    """A tick that produced nothing and is rolled back like a failed one."""


# Locations must resolve before NPCs/items that anchor to them. Quests last (anchor to PCs).
_CREATE_ORDER = {"location": 0, "npc": 1, "item": 2, "quest": 3}

# Stall detection — director fires when no quest advanced for this many ticks,
# or the last few events are all talk/waiting.
_STALL_QUIET_TICKS = 6
_STALL_IDLE_EVENTS = 5

# Ticks that must pass between two accepted steps of one quest, so an over-eager
# DM cannot burn through a plan on consecutive turns.
MIN_TICKS_BETWEEN_ADVANCES = 2

_FACTION_ACCELERATION_CAUSE_PATTERN = re.compile(
    r"\b(?:alerts?|alerted|burns?|burned|catches?|caught|confronts?|confronted|"
    r"destroys?|destroyed|discovers?|discovered|escapes?|escaped|informs?|informed|"
    r"notices?(?![\s-]+boards?\b)|noticed|reports?|reported|sabotages?|sabotaged|sees|saw|signals?|signaled|"
    r"spots?|spotted|tells?|told|warns?|warned|witnesses?|witnessed)\b"
)
_FACTION_ACCELERATION_STOP_WORDS = {
    "a",
    "an",
    "and",
    "anyone",
    "at",
    "before",
    "can",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
}
_NON_PERSON_NPC_NAME_PATTERN = re.compile(
    r"\b(?:army|cartel|clan|company|consortium|corporation|crew|cult|faction|gang|guild|"
    r"militia|navy|organi[sz]ation|society|syndicate|tribe)\b",
    re.IGNORECASE,
)
_VAGUE_LOCATION_WORDS = {
    "area",
    "destination",
    "downriver",
    "elsewhere",
    "location",
    "nearby",
    "place",
    "region",
    "site",
    "somewhere",
    "unknown",
    "unnamed",
    "upriver",
}
_INDEFINITE_LOCATION_WORDS = {"a", "an", "another", "some"}


# ─── Scheduler ────────────────────────────────────────────────


def _awaiting_response(
    state: WorldState, character_id: str, speaker: str | None = None
) -> bool:
    """Whether the latest speech involving the character directly addressed them (by `speaker`, if given)."""
    for event in reversed(state.history):
        if (
            character_id not in event.characters
            or not is_dialogue(event.text)
            or not event.text.startswith(f"{event.characters[0]} says")
        ):
            continue
        return (
            len(event.characters) > 1
            and event.characters[-1] == character_id
            and speaker in (None, event.characters[0])
        )
    return False


def _scene_actors(state: WorldState, active_pc_id: str) -> list[str]:
    """Living, motivated scene participants. PC first, then others by id."""
    scene = state.characters[active_pc_id].location
    active_quest_owners = {
        quest.owner
        for quest in state.quests.values()
        if quest.status.casefold() not in {"completed", "failed"}
    }
    present = [
        cid
        for cid, character in state.characters.items()
        if character.location == scene
        and character.stats.hp > 0
        and (
            cid == active_pc_id
            or character.goal.strip()
            or cid in active_quest_owners
            or _awaiting_response(state, cid)
        )
    ]
    present.sort(key=lambda cid: (cid != active_pc_id, cid))
    return present


def _pick_next_actor(state: WorldState, active_pc_id: str, tick: int) -> str:
    actors = _scene_actors(state, active_pc_id)
    pending_responders = [
        character_id
        for character_id in actors
        if character_id != active_pc_id
        and _awaiting_response(state, character_id, speaker=active_pc_id)
    ]
    if pending_responders:
        # The hero's direct question gets a reply before round-robin rotation
        # gives the hero another turn. NPCs addressing each other do not jump
        # the queue, or their exchange starves the hero of turns.
        return pending_responders[0]
    # It is the hero's story: they act every other tick however crowded the
    # scene is, and the others share the turns in between.
    others = actors[1:]
    if tick % 2 == 0 or not others:
        return active_pc_id
    return others[tick // 2 % len(others)]


def _describe_tool(tool: CharacterTool) -> str:
    """One-line preview of the acting character's chosen step."""
    if isinstance(tool, Speak):
        head = f"speak → {tool.target}" if tool.target else "speak"
        return f'{head}: "{tool.message}"'
    if isinstance(tool, Travel):
        return f"travel → {tool.destination}"
    if isinstance(tool, Wait):
        return "wait"
    if isinstance(tool, Attack):
        return f"attack → {tool.target}"
    if isinstance(tool, Check):
        target = (
            f" vs {tool.opponent}" if tool.opponent else f" vs DC {tool.difficulty}"
        )
        return f"check ({tool.ability}{target}): {tool.description}"
    if isinstance(tool, Action):
        suffix = f" ({tool.target})" if tool.target else ""
        return f"action{suffix}: {tool.description}"
    return repr(tool)


def _mentions_npc_name(text: str, name: str) -> bool:
    """Match all meaningful name words, allowing generated title/name reordering."""
    name_words = set(name.split("-")) - {"a", "an", "the"}
    return bool(name_words) and name_words <= set(slugify(text).split("-"))


def _can_anchor_new_npc(create: Create, new_events: list[HistoryEvent]) -> bool:
    """Require current-turn identity and location evidence for a new NPC reveal."""
    if create.type != "npc":
        return True
    if _NON_PERSON_NPC_NAME_PATTERN.search(create.name):
        return False

    name = slugify(create.name)
    named_events = [
        event for event in new_events if _mentions_npc_name(event.text, name)
    ]
    if not named_events:
        return False
    if not create.location:
        return True

    anchor = slugify(create.location)
    for event in named_events:
        named_segments = (
            segment
            for segment in re.split(r"(?<=[.!?])\s+|[\r\n]+", event.text)
            if _mentions_npc_name(segment, name)
        )
        explicit_anchor = any(
            re.search(
                rf"(?:^|-)(?:at|in|inside|near|outside)-(?:the-)?{re.escape(anchor)}(?:-|$)",
                slugify(segment),
            )
            for segment in named_segments
        )
        if explicit_anchor:
            return True
        # Event locations describe their participants, not every person they
        # mention. Do not materialize someone merely named in dialogue or
        # narration at the speaker's location.
        named_participant = any(
            _mentions_npc_name(character_id, name) for character_id in event.characters
        )
        if named_participant and slugify(event.location) == anchor:
            return True
    return False


def _can_anchor_new_location(create: Create, new_events: list[HistoryEvent]) -> bool:
    """Require a concrete location name stated in a current-turn event."""
    if create.type != "location":
        return True

    name = slugify(create.name)
    name_words = name.split("-")
    if not name_words or all(word in _VAGUE_LOCATION_WORDS for word in name_words):
        return False

    mention_pattern = re.compile(rf"(?:^|-){re.escape(name)}(?:-|$)")
    for event in new_events:
        event_slug = slugify(event.text)
        for mention in mention_pattern.finditer(event_slug):
            prefix_words = event_slug[: mention.start()].strip("-").split("-")
            if prefix_words and prefix_words[-1] in _INDEFINITE_LOCATION_WORDS:
                continue
            return True
    return False


def _item_create_already_materialized(state: WorldState, create: Create) -> bool:
    """Whether DM enrichment is repeating an item created earlier this turn."""
    if create.type != "item":
        return False

    normalized_name = slugify(create.name)
    return any(
        slugify(item) == normalized_name
        for location in state.locations.values()
        for item in location.items
    ) or any(
        slugify(item) == normalized_name
        for character in state.characters.values()
        for item in character.inventory
    )


def _log_rejected_world_update(
    logger: Logger | None,
    create: Create,
    reason: str,
) -> None:
    """Persist guarded DM enrichment attempts for post-run auditing."""
    if logger is None:
        return
    logger.log_event(
        "world_update_rejected",
        update="create",
        entity_type=create.type,
        name=create.name,
        location=create.location,
        reason=reason,
    )


def _log_rejected_world_modification(
    logger: Logger | None,
    modify: Modify,
    reason: str,
) -> None:
    """Persist guarded DM modification attempts for post-run auditing."""
    if logger is None:
        return
    details = {
        "update": modify.action,
        "target_id": modify.target_id,
        "reason": reason,
    }
    if modify.other_id:
        details["other_id"] = modify.other_id
    logger.log_event("world_update_rejected", **details)


# ─── Stall detection ──────────────────────────────────────────


def _minimum_action_minutes(tool: CharacterTool, legs: int = 1) -> int:
    """Keep direct actions consequential when the DM underestimates their duration."""
    if isinstance(tool, Travel):
        return 10 * max(1, legs)
    if isinstance(tool, Wait):
        return 5
    if isinstance(tool, Action) and any(
        verb in tool.description.lower()
        for verb in ("search", "examine", "inspect", "investigate")
    ):
        return 5
    return 1


def _can_accelerate_faction_clock(
    state: WorldState,
    tool: CharacterTool,
    faction_id: str,
    clock_id: str | None,
    events: list[HistoryEvent],
) -> bool:
    """Require current-event evidence that an interaction affected the faction."""
    if not isinstance(tool, (Action, Attack, Check, Speak)) or not clock_id:
        return False
    faction = next(
        (
            candidate
            for candidate_id, candidate in state.factions.items()
            if slugify(candidate_id) == slugify(faction_id)
        ),
        None,
    )
    if not faction:
        return False
    clock = next(
        (
            candidate
            for candidate in faction.clocks
            if slugify(candidate.id) == slugify(clock_id)
        ),
        None,
    )
    if not clock or not clock.event_acceleration or not events:
        return False

    faction_words = _word_roots(f"{faction.id} {faction.name} {faction.goal}")
    clock_words = _word_roots(f"{clock.id} {clock.name} {clock.consequence}")
    anchors = (faction_words | clock_words) - _FACTION_ACCELERATION_STOP_WORDS
    event_text = " ".join(event.text for event in events)
    if not anchors & _word_roots(event_text):
        return False

    # Directly addressing or attacking a character whose identity/role ties them
    # to the faction is itself notice. Other actions need an explicit causal event;
    # merely finding evidence about a faction must not accelerate its plans.
    if isinstance(tool, (Attack, Speak)) and tool.target:
        target = resolve_character(state, tool.target)
        event_characters = {
            character_id for event in events for character_id in event.characters
        }
        if target and target.id in event_characters:
            target_words = _word_roots(f"{target.id} {target.role}")
            if target_words & faction_words:
                return True

    return bool(_FACTION_ACCELERATION_CAUSE_PATTERN.search(event_text.casefold()))


def _word_roots(text: str) -> set[str]:
    """Normalize simple plurals so objectives and generated prose can be compared."""
    words = set(re.findall(r"[a-z]+", text.casefold().replace("-", " ")))
    return {
        word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words
    }


def _quest_veto(
    state: WorldState,
    quest: Quest,
    tool: CharacterTool,
    events: list[HistoryEvent],
    advanced: set[str],
) -> str | None:
    """Accept a DM progress claim only on facts the engine can check in state.

    The DM judges *what* happened; the engine checks that it could have: a real
    attempt by the owner (or a reply to them from someone the objective names),
    at the place and with the people the objective names, that did not fail,
    and not faster than one step per MIN_TICKS_BETWEEN_ADVANCES. Returns the
    broken rule's name, or None to accept.
    """
    owner = state.characters.get(quest.owner)
    if owner is None:
        return "no_owner"
    if not isinstance(tool, (Action, Attack, Check, Speak)):
        return "not_an_attempt"
    objective = (
        quest.plan[quest.current_step] if quest.current_step < len(quest.plan) else ""
    )
    people = named_ids(objective, state.characters) - {owner.id}
    reply = isinstance(tool, Speak) and tool.target == owner.id
    if tool.actor != owner.id and not (reply and tool.actor in people):
        return "not_owner"
    if any(
        event.text.startswith(("Cannot ", f"{tool.actor} fails:")) for event in events
    ):
        return "failed_attempt"
    places = named_ids(objective, state.locations)
    if places and owner.location not in places:
        return "wrong_location"
    # A person the objective names must be on hand, not necessarily acting:
    # "follow Glimmerfin's sigils" is done beside her, not to her. Someone named
    # only in passing ("supplies Jory guards") who is elsewhere is not required.
    expected = {
        p for p in people if not places or state.characters[p].location in places
    }
    if expected and not any(
        state.characters[p].location == owner.location for p in expected
    ):
        return "named_person_absent"
    if quest.id in advanced:
        return "one_step_per_tick"
    if (
        quest.current_step
        and state.time - quest.last_progress < MIN_TICKS_BETWEEN_ADVANCES
    ):
        return "too_soon"
    return None


def _record_resolution_if_needed(
    state: WorldState,
    actor_id: str,
    history_size: int,
    resolution: str,
) -> list[HistoryEvent]:
    """Keep a resolved turn visible when it produced no state-operation event."""
    if len(state.history) == history_size and resolution.strip():
        actor = state.characters.get(actor_id)
        state.history.append(
            HistoryEvent(
                text=resolution.strip(),
                location=actor.location if actor else "",
                characters=[actor_id],
            )
        )
    return state.history[history_size:]


def _is_idle_event(text: str) -> bool:
    """Speak/wait-style event."""
    return is_dialogue(text) or "waits" in text.lower()


def _is_stalled(state: WorldState) -> bool:
    """True when no quest has advanced for a while, or recent events are all chatter/waiting."""
    if state.time - state.last_quest_advance_time >= _STALL_QUIET_TICKS:
        return True
    tail = state.history[-_STALL_IDLE_EVENTS:]
    return len(tail) == _STALL_IDLE_EVENTS and all(_is_idle_event(e.text) for e in tail)


def _record_intervention(state: WorldState, quest_id: str | None) -> str:
    """Bump the escalation counter for the targeted quest, or 'world' if untargeted/unknown."""
    key = quest_id if quest_id and quest_id in state.quests else "world"
    state.director_interventions[key] = state.director_interventions.get(key, 0) + 1
    return key


# ─── Agent flows ──────────────────────────────────────────────


async def flow_agent_turn(
    actor_id: str,
    state: WorldState,
    logger: Logger,
    pc_controller: Callable[[str, WorldState], Awaitable[CharacterTool]] | None = None,
    replay: ReplayTape | None = None,
) -> CharacterTool | None:
    """Ask the acting character to pick one next step."""
    char = state.characters[actor_id]
    print(f"  [{actor_id}]", flush=True)
    if replay and replay.is_playback:
        return replay.character(actor_id)
    if pc_controller is not None:
        intent = await pc_controller(actor_id, state)
        logger.log_event(
            "player_action",
            actor=actor_id,
            tool=intent.kind,
            action=intent.model_dump(mode="json"),
        )
        return replay.character(actor_id, intent) if replay else intent
    label = f"character:{actor_id}"
    try:
        with logger.run(label):
            result = await character_agent.run(
                "Choose exactly ONE next step by calling action, check, speak, travel, attack, or wait.",
                deps=CharacterDeps(char=char, state=state),
                usage_limits=_AGENT_USAGE,
            )
            logger.log_messages(label, result.all_messages())
    except UsageLimitExceeded as exc:
        print(f"  [limit] {actor_id}'s turn ended: {exc}", flush=True)
        return replay.character(actor_id, None) if replay else None
    output = result.output
    return replay.character(actor_id, output) if replay else output


async def flow_dm(
    state: WorldState,
    new_events: list[HistoryEvent],
    logger: Logger,
    replay: ReplayTape | None = None,
) -> DMResult:
    """One LLM call: enrich the world, review quests, and estimate time for this tick's new events."""
    if not new_events:
        return DMResult(creates=[], modifies=[], minutes=[])
    if replay and replay.is_playback:
        return replay.dm()
    new_texts = [e.text for e in new_events]
    tail = state.history[
        : -len(new_events)
    ]  # new_events is non-empty here (checked above)
    context_events = [e.text for e in tail[-10:]]
    label = "dm"
    try:
        with logger.run(label):
            result = await dm_agent.run(
                "Register any new entities, report quest progress, and estimate minutes elapsed for each new event.",
                deps=DMDeps(
                    state=state, context_events=context_events, new_events=new_texts
                ),
                usage_limits=_DM_USAGE,
            )
            logger.log_messages(label, result.all_messages())
    except AgentRunError as exc:
        # Bookkeeping is optional: a failed review costs this tick's enrichment only.
        print(f"  [fallback] dm review failed: {exc}", flush=True)
        fallback = DMResult(creates=[], modifies=[], minutes=[1] * len(new_events))
        return replay.dm(fallback) if replay else fallback
    dm_result = result.output
    if not dm_result.minutes or len(dm_result.minutes) != len(new_events):
        dm_result.minutes = [1] * len(new_events)
    dm_result.creates = sorted(
        dm_result.creates, key=lambda i: _CREATE_ORDER.get(i.type, 3)
    )
    return replay.dm(dm_result) if replay else dm_result


async def flow_director(
    state: WorldState,
    location_id: str,
    logger: Logger,
    replay: ReplayTape | None = None,
) -> None:
    """Stall-breaker: one LLM call that introduces a single complication grounded in existing state."""
    label = "director"
    if replay and replay.is_playback:
        beat = replay.director()
    else:
        try:
            with logger.run(label):
                result = await director_agent.run(
                    "The story has stalled. Introduce exactly one grounded complication.",
                    deps=DirectorDeps(state=state, location_id=location_id),
                    usage_limits=_DM_USAGE,
                )
                logger.log_messages(label, result.all_messages())
        except AgentRunError as exc:
            print(f"  [fallback] director beat failed: {exc}", flush=True)
            if replay:
                replay.director(None)
            return
        beat = result.output
        if replay:
            beat = replay.director(beat)

    if beat is None:
        return

    # Director events are retcon-style arrivals: logged, printed, 0 minutes.
    pre = len(state.history)
    # A crew or gang is not a character: left in, it takes turns and speaks as one person.
    if beat.create and not (
        beat.create.type == "npc"
        and _NON_PERSON_NPC_NAME_PATTERN.search(beat.create.name)
    ):
        await resolve(beat.create, state, logger=logger)
    if beat.event:
        WorldOperations(state).world_event(beat.event, location_id)
    for event in state.history[pre:]:
        print(f"  {event.text}  (director)")

    key = _record_intervention(state, beat.quest_id)
    logger.log_event("director_beat", quest=key, text=beat.event)


# ─── Tick composition ────────────────────────────────────────


async def tick(
    active_pc_id: str,
    state: WorldState,
    tick_index: int,
    logger: Logger,
    pc_controller: Callable[[str, WorldState], Awaitable[CharacterTool]] | None = None,
    replay: ReplayTape | None = None,
) -> bool:
    actor_id = _pick_next_actor(state, active_pc_id, tick_index)

    controller = pc_controller if actor_id == active_pc_id else None
    intent = await flow_agent_turn(actor_id, state, logger, controller, replay)
    if intent is None:
        return False
    if intent.actor != actor_id:
        # A turn may only mutate state on behalf of the character selected by the scheduler.
        intent = intent.model_copy(update={"actor": actor_id})
    print(f"    ↳ {_describe_tool(intent)}")

    # 1. Resolve the turn.
    pre = len(state.history)
    legs = (
        len(route(state, state.characters[actor_id].location, intent.destination) or [])
        if isinstance(intent, Travel)
        else 1
    )
    if replay and replay.is_playback and isinstance(intent, (Action, Check)):
        resolution = replay.action_resolution(actor_id, state)
    else:
        resolution = await resolve(intent, state, logger=logger)
        if replay and isinstance(intent, (Action, Check)):
            replay.action_resolution(actor_id, state, resolution)
    new_events = _record_resolution_if_needed(state, actor_id, pre, resolution)

    # 2. One DM call: time estimates, new entities, and quest progress.
    dm = await flow_dm(state, new_events, logger, replay)
    event_minutes = [max(0, minutes) for minutes in dm.minutes[: len(new_events)]]
    event_minutes.extend([0] * (len(new_events) - len(event_minutes)))
    if event_minutes:
        event_minutes[0] = max(event_minutes[0], _minimum_action_minutes(intent, legs))
    for index, event in enumerate(new_events[1:], start=1):
        # Later knowledge events describe the result of the primary action or
        # its optional memory update; they must not charge for that time twice.
        if " learns:" in event.text.casefold():
            event_minutes[index] = 0

    # 2a. Stamp elapsed time onto the resulting events and advance the clock.
    for event, mins in zip(new_events, event_minutes):
        event.minutes_elapsed = mins
    state.minutes_elapsed += sum(event_minutes)
    for event in new_events:
        print(f"  {event.text}  (+{event.minutes_elapsed}m)")
    if new_events:
        print(f"  [clock: {format_clock(state.minutes_elapsed)}]")

    # 2b. Enrich the world. Revealed entities are retcons — logged but not time-stamped.
    pre_enrich = len(state.history)
    for create_intent in dm.creates:
        if _item_create_already_materialized(state, create_intent):
            print(f"  [world] {create_intent.name}: item is already materialized.")
            _log_rejected_world_update(logger, create_intent, "already_materialized")
            continue
        if not _can_anchor_new_location(create_intent, new_events):
            print(
                f"  [world] {create_intent.name}: cannot reveal location without "
                "a concrete name in the current events."
            )
            _log_rejected_world_update(
                logger, create_intent, "unsupported_location_evidence"
            )
            continue
        if not _can_anchor_new_npc(create_intent, new_events):
            print(
                f"  [world] {create_intent.name}: cannot reveal NPC at "
                f"{create_intent.location!r} without current-event identity and location evidence."
            )
            _log_rejected_world_update(
                logger, create_intent, "unsupported_npc_evidence"
            )
            continue
        await resolve(create_intent, state, logger=logger)
    for event in state.history[pre_enrich:]:
        print(f"  {event.text}  (revealed)")

    # 2c. Review quests. Step-level progress; may append XP events via advance_quest.
    pre_quest = len(state.history)
    advanced: set[str] = set()
    for update in dm.modifies:
        if (
            update.action == "advance_faction_clock"
            and not _can_accelerate_faction_clock(
                state,
                intent,
                update.target_id,
                update.other_id,
                new_events,
            )
        ):
            print(
                f"  [faction] {update.target_id}: this turn cannot accelerate that clock."
            )
            _log_rejected_world_modification(
                logger,
                update,
                "unsupported_faction_acceleration",
            )
            continue
        if (
            update.action == "remove_npc"
            and (target := resolve_character(state, update.target_id))
            and target.id == active_pc_id
        ):
            print(
                f"  [world] {update.target_id}: cannot remove the active player character."
            )
            continue
        quest = (
            find_quest(state, update.target_id)
            if update.action == "update_quest"
            else None
        )
        claims_progress = update.advance or update.status == "completed"
        if (
            quest
            and quest.status == "active"
            and claims_progress
            and (guard := _quest_veto(state, quest, intent, new_events, advanced))
        ):
            print(f"  [quest] {quest.id}: vetoed by guard '{guard}'.")
            logger.log_event(
                "quest_advance_vetoed",
                guard=guard,
                quest=quest.id,
                objective=quest.plan[quest.current_step]
                if quest.current_step < len(quest.plan)
                else None,
                claim=update.step,
                status=update.status,
                actor=intent.actor,
                tool=intent.kind,
                events=[event.text for event in new_events],
            )
            continue
        msg = await resolve(update, state, logger=logger)
        if quest and claims_progress:
            advanced.add(quest.id)
        print(f"  [quest] {update.target_id}: {msg}")
    for event in state.history[pre_quest:]:
        print(f"  {event.text}")

    # 3. Director beat: when the story stalls, one proactive complication breaks it.
    if _is_stalled(state):
        await flow_director(
            state, state.characters[active_pc_id].location, logger, replay
        )
        state.last_quest_advance_time = state.time  # suppress consecutive fires
    return True


# ============================================================
# Entry point
# ============================================================


def _snapshot(state: WorldState) -> dict:
    """World/quest/xp counters for scorecard.py — logged before and after every tick."""
    quests = state.quests.values()
    return {
        "quests_active": sum(
            1 for q in quests if q.status.lower() not in {"completed", "failed"}
        ),
        "quests_completed": sum(1 for q in quests if q.status.lower() == "completed"),
        "quests_failed": sum(1 for q in quests if q.status.lower() == "failed"),
        "locations": len(state.locations),
        "characters": len(state.characters),
        "total_xp": sum(c.stats.xp for c in state.characters.values()),
        "minutes_elapsed": state.minutes_elapsed,
    }


async def _run_game(
    scenario: str | None,
    character_id: str | None,
    max_turns: int,
    new_character: dict | None,
    *,
    resume: bool = False,
    stop_check: Callable[[], bool] | None = None,
    pc_controller: Callable[[str, WorldState], Awaitable[CharacterTool]] | None = None,
    progress_callback: Callable[[WorldState], None] | None = None,
    replay: ReplayTape | None = None,
) -> bool:
    """Run ticks until `max_turns` complete (0 = until interrupted). False if it gave up."""
    if replay:
        scenario, character_id = replay.resolve_context(scenario, character_id)
    manager = StateManager(scenario=scenario, resume=resume)
    session_id = manager.run_id
    latest = manager.latest_snapshot_name() if resume else None
    state = manager.load_state(latest) if latest else manager.init_state()

    if new_character:
        result = WorldOperations(state).spawn_character(**new_character)
        print(result)
        if result.startswith("Cannot spawn"):
            raise RuntimeError(result)
        state.protagonist = new_character["character_id"]
    elif not state.protagonist:
        state.protagonist = character_id or manager.manifest["pc"]
    pc_id = state.protagonist
    if pc_id not in state.characters:
        raise RuntimeError(f"PC '{pc_id}' not in scenario '{manager.scenario}'")

    if replay:
        replay.bind_context(manager.scenario, pc_id)

    print(f"\n=== {manager.manifest.get('title', manager.scenario)} ===")
    print(manager.manifest.get("hook", ""))
    print(f"Playing as: {pc_id}\n")

    logger = Logger(
        character_id=pc_id,
        max_turns=max_turns,
        scenario=manager.scenario,
        scenario_title=manager.manifest.get("title", manager.scenario),
        session_id=session_id,
        append=resume,
        # Prefill/decode split only exists for the local llama-server.
        metrics_reader=read_metrics
        if os.getenv("LLM_PROVIDER", "local") == "local"
        else None,
    )
    run_failed = False
    failures = 0
    turns = 0
    mark = replay.mark() if replay else 0
    try:
        manager.save_state(state)
        if progress_callback:
            progress_callback(state)
        logger.log_event("world_snapshot", **_snapshot(state))
        while not max_turns or turns < max_turns:
            if stop_check and stop_check():
                logger.log_event("run_stopped")
                break
            print(f"\n--- Tick {state.time + 1} ---")
            logger.log_turn(state.time + 1)
            pre_minutes = state.minutes_elapsed
            before = state.model_copy(deep=True)
            mark = replay.mark() if replay else 0
            try:
                if replay and replay.skipped():
                    raise _SkippedTick("recorded as skipped")
                await campaign_upkeep(state, logger, replay)
                # A retried tick rotates to the next actor instead of repeating a failure.
                if not await tick(
                    state.protagonist,
                    state,
                    state.time + failures,
                    logger,
                    pc_controller,
                    replay,
                ):
                    raise _SkippedTick("no character action")
                succeed_if_dead(state, logger)
            except (AgentRunError, _SkippedTick) as exc:
                for field_name in WorldState.model_fields:
                    setattr(state, field_name, getattr(before, field_name))
                if replay:
                    replay.skip(mark)
                failures += 1
                logger.log_event(
                    "tick_failed",
                    error=type(exc).__name__,
                    message=str(exc),
                    consecutive=failures,
                )
                print(
                    f"  [error] tick skipped ({failures} in a row): {exc}", flush=True
                )
                if failures >= MAX_FAILED_TICKS:
                    logger.log_event(
                        "run_error", error=type(exc).__name__, message=str(exc)
                    )
                    print("  [error] campaign stopped: too many failed ticks.")
                    run_failed = True
                    break
                if not (replay and replay.is_playback):
                    await asyncio.sleep(FAILED_TICK_BACKOFF_S * failures)
                continue
            failures = 0
            turns += 1
            # Off-screen agendas march with in-world time, not turn count.
            WorldOperations(state).advance_faction_clocks_hourly(
                pre_minutes, state.minutes_elapsed
            )
            await compact_history(
                state, logger, replay, manager.state_dir, state.protagonist
            )  # between ticks only — never mid-tick
            logger.log_event("world_snapshot", **_snapshot(state))
            state.time += 1
            manager.save_state(state)
            if progress_callback:
                progress_callback(state)
        if replay:
            replay.assert_consumed()
    except asyncio.CancelledError:
        # Ctrl-C: the last completed tick is already saved; the partial one is dropped.
        logger.log_event("run_stopped", reason="interrupted")
        if replay:
            replay.truncate(mark)
        print(
            f"\n=== Interrupted. Tick {state.time} is saved; continue with "
            f"--resume --scenario {manager.scenario} ===",
            flush=True,
        )
        raise
    except Exception as exc:
        logger.log_event("run_error", error=type(exc).__name__, message=str(exc))
        print(f"  [error] campaign crashed: {exc}", flush=True)
        raise
    finally:
        logger.close(turns_completed=state.time)
    return not run_failed


def run_game(
    character_id: str | None = None,
    max_turns: int = 50,
    scenario: str | None = None,
    new_character: dict | None = None,
    *,
    resume: bool = False,
    replay: ReplayTape | None = None,
    pc_controller: Callable[[str, WorldState], Awaitable[CharacterTool]] | None = None,
) -> bool:
    # Narrative text (arrows, em-dashes) doesn't fit Windows' default console codepage.
    if isinstance(sys.stdout, TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        return asyncio.run(
            _run_game(
                scenario,
                character_id,
                max_turns,
                new_character,
                resume=resume,
                replay=replay,
                pc_controller=pc_controller,
            )
        )
    except KeyboardInterrupt:
        return True
