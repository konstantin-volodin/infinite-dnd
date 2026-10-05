"""Game loop. Each tick: one actor takes a turn → resolve → stamp elapsed time → enrich world → review quests."""

import asyncio
import os
import re
import sys
from collections.abc import Awaitable, Callable
from io import TextIOWrapper

from pydantic_ai.exceptions import AgentRunError, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from src.engine.state import StateManager, WorldOperations, WorldState, is_dialogue, resolve_character, slugify
from src.engine.state.models import HistoryEvent, Quest
from src.agents.character.agent import CharacterDeps, agent as character_agent
from src.agents.character.tools import Action, Attack, CharacterTool, Check, Speak, Travel, Wait
from src.agents.dm.agent import DMDeps, DMResult, agent as dm_agent
from src.agents.dm.tools import Create, Modify
from src.agents.dm.director import DirectorDeps, agent as director_agent
from src.agents.action_resolver.agent import resolve
from src.agents.server import read_metrics
from src.interface.session_log import Logger
from src.interface.world_state import format_clock
from .chronicle import compact_history
from .replay import ReplayTape


_AGENT_USAGE = UsageLimits(request_limit=12)
_DM_USAGE = UsageLimits(request_limit=16)

# Locations must resolve before NPCs/items that anchor to them. Quests last (anchor to PCs).
_CREATE_ORDER = {"location": 0, "npc": 1, "item": 2, "quest": 3}

# Stall detection — director fires when no quest advanced for this many ticks,
# or the last few events are all talk/waiting.
_STALL_QUIET_TICKS = 6
_STALL_IDLE_EVENTS = 5

_FINAL_OBJECTIVE_VERB_FORMS = {
    "clear": {"clear", "clears", "cleared"},
    "confront": {"confront", "confronts", "confronted"},
    "defeat": {"defeat", "defeats", "defeated"},
    "find": {"find", "finds", "found"},
    "gather": {"gather", "gathers", "gathered"},
    "recover": {"recover", "recovers", "recovered"},
    "report": {"report", "reports", "reported"},
    "track": {"track", "tracks", "tracked"},
}
_FINAL_OBJECTIVE_STOP_WORDS = {
    "a", "an", "and", "at", "before", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with",
}
_FINAL_OBJECTIVE_DISCOVERY_PATTERN = re.compile(
    r"\b(?:clue|clues|lead|whereabouts|destination)\b"
    r"|\blocation(?:\s+(?:of|for))?\b"
    r"|\b(?:map|path|route|trail)\s+(?:about|for|of|to|toward|towards|leading|pointing)\b"
    r"|\b[a-z]+['’]s\s+(?:map|path|route|trail)\b"
    r"|\b[a-z]+['’]s\s+(?:house|home|hideout|camp|cabin|room|cave|cellar|quarters)\b"
    r"|\b(?:house|home|hideout|camp|cabin|room|cave|cellar|quarters)\b[^.!?]{0,80}\b(?:of|where|containing|holding)\b"
    r"|\b(?:hiding\s+place|hiding\s+spot)\b"
    r"|\bwhere\b[^.!?]{0,80}\b(?:is|was|are|were)\s+(?:hiding|located|staying|living)\b"
    r"|\b(?:where|place|spot|site)\b[^.!?]{0,80}\b(?:last\s+seen|last\s+known|was\s+seen|were\s+seen)\b"
)
_FINAL_OBJECTIVE_INDIRECT_FIND_PATTERN = re.compile(
    r"\b(?:description|descriptions|evidence|information|portrait|portraits|proof|record|records|"
    r"report|reports|poster|posters|rumor|rumors|rumour|rumours|sign|signs|testimony|trace|traces)\b"
)
_FINAL_OBJECTIVE_CONSTRAINT_PATTERN = re.compile(r"\b(?:after|at|before|by|during|until)\b")
_OBJECTIVE_GENERIC_WORDS = {
    "about", "around", "clue", "clues", "evidence", "hidden", "information", "known",
    "last", "movement", "movements", "proof", "sign", "signs", "unusual", "whereabouts",
}
_REFERENTIAL_CONTACT_OBJECTIVE_PATTERN = re.compile(
    r"^(?:find|locate|track\s+down)\s+(?:the\s+)?(?:person|somebody|someone|who|whoever|witness)\b"
)
_DIALOGUE_OBJECTIVE_PATTERN = re.compile(
    r"^(?:ask|interview|question|speak(?:\s+to)?|talk(?:\s+to)?)\b"
)
_DIALOGUE_ATTEMPT_PATTERN = re.compile(
    r"^(?:ask|interview|question|speak(?:\s+to)?|talk(?:\s+to)?)\b"
)
_OBJECTIVE_ATTEMPT_TYPES = (Action, Attack, Check, Speak)
_FAILED_ATTEMPT_PATTERN = re.compile(
    r"\b(?:cannot|failed|fails|nothing)\b"
    r"|\bno\s+(?:clue|clues|evidence|new|sign|signs|useful)\b"
    r"|\bno\s+(?:direct\s+)?information\b"
    r"|\bno\s+(?:direct\s+)?references?\b"
)
_FACTION_ACCELERATION_CAUSE_PATTERN = re.compile(
    r"\b(?:alerts?|alerted|burns?|burned|catches?|caught|confronts?|confronted|"
    r"destroys?|destroyed|discovers?|discovered|escapes?|escaped|informs?|informed|"
    r"notices?(?![\s-]+boards?\b)|noticed|reports?|reported|sabotages?|sabotaged|sees|saw|signals?|signaled|"
    r"spots?|spotted|tells?|told|warns?|warned|witnesses?|witnessed)\b"
)
_FACTION_ACCELERATION_STOP_WORDS = {
    "a", "an", "and", "anyone", "at", "before", "can", "for", "from", "in", "of", "on", "or", "the", "to",
}
_NON_PERSON_NPC_NAME_PATTERN = re.compile(
    r"\b(?:army|cartel|clan|company|consortium|corporation|crew|cult|faction|gang|guild|"
    r"militia|navy|organi[sz]ation|society|syndicate|tribe)\b",
    re.IGNORECASE,
)
_VAGUE_LOCATION_WORDS = {
    "area", "destination", "downriver", "elsewhere", "location", "nearby",
    "place", "region", "site", "somewhere", "unknown", "unnamed", "upriver",
}
_INDEFINITE_LOCATION_WORDS = {"a", "an", "another", "some"}


# ─── Scheduler ────────────────────────────────────────────────

def _awaiting_response(state: WorldState, character_id: str) -> bool:
    """Whether the latest speech involving the character directly addressed them."""
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
        if character_id != active_pc_id and _awaiting_response(state, character_id)
    ]
    if pending_responders:
        # A direct question should produce a reply before round-robin rotation
        # gives the speaker another turn and encourages duplicate dialogue.
        return pending_responders[0]
    return actors[tick % len(actors)]


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
        target = f" vs {tool.opponent}" if tool.opponent else f" vs DC {tool.difficulty}"
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
        event for event in new_events
        if _mentions_npc_name(event.text, name)
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
            _mentions_npc_name(character_id, name)
            for character_id in event.characters
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
            prefix_words = event_slug[:mention.start()].strip("-").split("-")
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

def _minimum_action_minutes(tool: CharacterTool) -> int:
    """Keep direct actions consequential when the DM underestimates their duration."""
    if isinstance(tool, Travel):
        return 10
    if isinstance(tool, Wait):
        return 5
    if isinstance(tool, Action) and any(
        verb in tool.description.lower() for verb in ("search", "examine", "inspect", "investigate")
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
        (candidate for candidate in faction.clocks if slugify(candidate.id) == slugify(clock_id)),
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
        event_characters = {character_id for event in events for character_id in event.characters}
        if target and target.id in event_characters:
            target_words = _word_roots(f"{target.id} {target.role}")
            if target_words & faction_words:
                return True

    return bool(_FACTION_ACCELERATION_CAUSE_PATTERN.search(event_text.casefold()))


def _claim_mentions_target(claim: str, target_id: str) -> bool:
    """Match canonical IDs in prose, including punctuated initialisms like ``V.M.``."""
    target_words = re.findall(r"[a-z0-9]+", target_id.casefold())
    claim_words = re.findall(r"[a-z0-9]+", claim.casefold())
    if set(target_words) <= set(claim_words):
        return True

    # A short canonical ID may be rendered as separated initials by the DM.
    # Only join one-character tokens, so ordinary phrases cannot match ``vm``.
    if len(target_words) != 1 or not 1 < len(target_words[0]) <= 4:
        return False
    initials = target_words[0]
    width = len(initials)
    return any(
        all(len(word) == 1 for word in window) and "".join(window) == initials
        for index in range(len(claim_words) - width + 1)
        if (window := claim_words[index:index + width])
    )


def _can_advance_final_objective(
    quest,
    tool: CharacterTool,
    events: list[HistoryEvent] | None = None,
    claimed_resolution: str | None = None,
) -> bool:
    """Require owner-attributed evidence that the final planned objective happened."""
    if not quest.plan or quest.current_step != len(quest.plan) - 1:
        return True
    if tool.actor != quest.owner or not isinstance(tool, (Action, Attack, Check, Speak)):
        return False

    objective_verb = quest.plan[quest.current_step].split(maxsplit=1)[0].casefold()
    verb_forms = _FINAL_OBJECTIVE_VERB_FORMS.get(objective_verb, {objective_verb})
    if objective_verb == "recover":
        verb_forms |= {"pick", "picks", "picked", "take", "takes", "took", "receive", "receives", "received"}
    objective_text = quest.plan[quest.current_step].casefold()
    target_text = _FINAL_OBJECTIVE_CONSTRAINT_PATTERN.split(objective_text, maxsplit=1)[0]
    objective_words = set(re.findall(r"[a-z]+", target_text))
    objective_verbs = {verb for verb in _FINAL_OBJECTIVE_VERB_FORMS if verb in objective_words}
    alternative_verbs = objective_verbs - {objective_verb} if re.search(r"\bor\b", target_text) else set()
    target_words = objective_words - _FINAL_OBJECTIVE_STOP_WORDS - objective_verbs
    alternative_forms = {
        form for verb in alternative_verbs for form in _FINAL_OBJECTIVE_VERB_FORMS[verb]
    }
    resolution_forms = verb_forms | alternative_forms
    resolution_pattern = re.compile(
        rf"^{re.escape(quest.owner.casefold())}\s+(?:has\s+|have\s+)?(?:{'|'.join(resolution_forms)})\b"
    )

    for event in events or []:
        text = event.text.casefold()
        words = set(re.findall(r"[a-z]+", text))
        if quest.owner not in event.characters or not resolution_pattern.match(text):
            continue
        if _FINAL_OBJECTIVE_DISCOVERY_PATTERN.search(text) or (
            objective_verb in {"find", "recover"} and _FINAL_OBJECTIVE_INDIRECT_FIND_PATTERN.search(text)
        ):
            continue
        if resolution_forms & words and target_words <= words and (
            not alternative_verbs
            or any(_FINAL_OBJECTIVE_VERB_FORMS[verb] & words for verb in alternative_verbs)
        ):
            return True

    # Speech is durably recorded as ``owner says to target``, not as the
    # resolution verb used by objectives such as "confront" or "report".
    # Accept the DM's step classification only when the underlying turn is a
    # confirmed direct address and the claim is anchored to both the objective
    # and the addressed character. This keeps unrelated dialogue from closing
    # a quest while allowing real speech events to satisfy speech objectives.
    if not isinstance(tool, Speak) or not tool.target or not claimed_resolution:
        return False
    direct_speech = any(
        event.characters[:1] == [quest.owner]
        and tool.target in event.characters[1:]
        and event.text.casefold().startswith(f"{quest.owner.casefold()} says")
        for event in events or []
    )
    if not direct_speech:
        return False

    claim = claimed_resolution.casefold()
    claim_words = set(re.findall(r"[a-z]+", claim))
    claim_pattern = re.compile(
        rf"^(?:{re.escape(quest.owner.casefold())}\s+)?"
        rf"(?:has\s+|have\s+)?(?:{'|'.join(resolution_forms)})\b"
    )
    if not claim_pattern.match(claim) or _FINAL_OBJECTIVE_DISCOVERY_PATTERN.search(claim):
        return False
    if alternative_verbs and not any(
        _FINAL_OBJECTIVE_VERB_FORMS[verb] & claim_words for verb in alternative_verbs
    ):
        return False

    target_id_words = set(re.findall(r"[a-z]+", tool.target.casefold()))
    message_words = set(re.findall(r"[a-z]+", tool.message.casefold()))
    claim_details = (
        claim_words
        - resolution_forms
        - target_id_words
        - set(re.findall(r"[a-z]+", quest.owner.casefold()))
        - _FINAL_OBJECTIVE_STOP_WORDS
    )
    if not message_words & claim_details:
        return False
    objective_anchor = target_words & claim_words
    return target_words <= claim_words or bool(
        objective_anchor and target_id_words and _claim_mentions_target(claim, tool.target)
    )


def _word_roots(text: str) -> set[str]:
    """Normalize simple plurals so objectives and generated prose can be compared."""
    words = set(re.findall(r"[a-z]+", text.casefold().replace("-", " ")))
    return {word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words}


def _named_objective_locations(state: WorldState, objective: str) -> set[str]:
    """Return known location IDs explicitly named by an objective."""
    objective_id = f"-{slugify(objective)}-"
    matches: set[str] = set()
    for location_id, location in state.locations.items():
        aliases = {slugify(location_id), slugify(location.id)} - {""}
        if any(f"-{alias}-" in objective_id for alias in aliases):
            matches.add(slugify(location.id))
    return matches


def _is_grounded_referential_contact(
    state: WorldState,
    quest: Quest,
    tool: CharacterTool,
    owner_events: list[HistoryEvent],
) -> bool:
    """Recognize direct contact with an unnamed person already tied to the quest."""
    objective = quest.plan[quest.current_step]
    if (
        not isinstance(tool, Speak)
        or not tool.target
        or not _REFERENTIAL_CONTACT_OBJECTIVE_PATTERN.search(objective.casefold())
        or not any(tool.target in event.characters for event in owner_events)
    ):
        return False

    actor = state.characters.get(quest.owner)
    if not actor:
        return False

    context_words = _word_roots(f"{quest.title} {quest.description}")
    context_words -= _word_roots(" ".join((*_OBJECTIVE_GENERIC_WORDS, *_FINAL_OBJECTIVE_STOP_WORDS)))
    context_words -= {
        form
        for forms in _FINAL_OBJECTIVE_VERB_FORMS.values()
        for form in _word_roots(" ".join(forms))
    }
    target_words = _word_roots(tool.target)
    message_words = _word_roots(tool.message)
    if not context_words & message_words:
        return False

    return any(
        target_words & (fact_words := _word_roots(fact)) and context_words & fact_words
        for fact in actor.knowledge
    )


def _dialogue_attempt_mentions_quest_topic(quest: Quest, tool: CharacterTool) -> bool:
    """Require dialogue content, not just its target/location, to concern the quest."""
    if isinstance(tool, Speak):
        attempt_text = tool.message
    elif isinstance(tool, (Action, Check)):
        attempt_text = tool.description
    else:
        return False

    objective = quest.plan[quest.current_step]
    topic_words = _word_roots(f"{objective} {quest.title} {quest.description}")
    topic_words -= _word_roots(" ".join(_OBJECTIVE_GENERIC_WORDS))
    topic_words -= _word_roots("ask interview question speak talk")
    topic_words -= {
        form
        for forms in _FINAL_OBJECTIVE_VERB_FORMS.values()
        for form in _word_roots(" ".join(forms))
    }
    topic_words -= _FINAL_OBJECTIVE_STOP_WORDS | {"s"}
    return not topic_words or bool(topic_words & _word_roots(attempt_text))


def _is_grounded_dialogue_reply(
    state: WorldState,
    quest: Quest,
    tool: CharacterTool,
    events: list[HistoryEvent],
) -> bool:
    """Treat a relevant NPC answer as the outcome of its owner's preceding question."""
    if (
        not isinstance(tool, Speak)
        or tool.actor == quest.owner
        or tool.target != quest.owner
        or not events
        or not _dialogue_attempt_mentions_quest_topic(quest, tool)
    ):
        return False

    direct_replies = [
        event
        for event in events
        if event.characters[:1] == [tool.actor]
        and quest.owner in event.characters[1:]
        and event.text.casefold().startswith(
            f"{tool.actor.casefold()} says to {quest.owner.casefold()}:"
        )
    ]
    if not direct_replies or any(
        _FAILED_ATTEMPT_PATTERN.search(event.text.casefold())
        for event in direct_replies
    ):
        return False

    prior_history = state.history
    if len(state.history) >= len(events) and state.history[-len(events):] == events:
        prior_history = state.history[:-len(events)]
    prior_exchange = next(
        (
            (index, event)
            for index, event in reversed(list(enumerate(prior_history)))
            if quest.owner in event.characters and tool.actor in event.characters
        ),
        None,
    )
    if not prior_exchange or not prior_exchange[1].text.casefold().startswith(
        f"{quest.owner.casefold()} says to {tool.actor.casefold()}:"
    ):
        return False
    exchange_index, exchange_event = prior_exchange
    progress_marker = f"quest '{quest.id.casefold()}' progress"
    if any(
        progress_marker in event.text.casefold()
        for event in prior_history[exchange_index + 1:]
    ):
        return False

    question = Speak(
        actor=quest.owner,
        target=tool.actor,
        message=exchange_event.text.split(":", 1)[-1].strip(),
    )
    return _dialogue_attempt_mentions_quest_topic(quest, question)


def _can_advance_objective(
    state: WorldState,
    quest: Quest,
    tool: CharacterTool,
    events: list[HistoryEvent],
    claimed_resolution: str | None = None,
) -> bool:
    """Require an owner-controlled attempt grounded in the current objective."""
    if quest.plan and quest.current_step == len(quest.plan) - 1:
        return _can_advance_final_objective(quest, tool, events, claimed_resolution)
    if _is_grounded_dialogue_reply(state, quest, tool, events):
        return True
    if tool.actor != quest.owner or not isinstance(tool, _OBJECTIVE_ATTEMPT_TYPES):
        return False

    objective = quest.plan[quest.current_step].casefold()
    if _DIALOGUE_OBJECTIVE_PATTERN.match(objective) and not (
        isinstance(tool, Speak)
        or isinstance(tool, (Action, Check))
        and _DIALOGUE_ATTEMPT_PATTERN.match(tool.description.casefold())
    ):
        return False
    if (
        _DIALOGUE_OBJECTIVE_PATTERN.match(objective)
        and not _dialogue_attempt_mentions_quest_topic(quest, tool)
    ):
        return False

    owner_events = [event for event in events if quest.owner in event.characters]
    if not owner_events or any(_FAILED_ATTEMPT_PATTERN.search(event.text.casefold()) for event in owner_events):
        return False

    objective_locations = _named_objective_locations(state, quest.plan[quest.current_step])
    if objective_locations and not any(slugify(event.location) in objective_locations for event in owner_events):
        return False

    actor = state.characters.get(quest.owner)
    attempt_parts = [actor.location if actor else ""]
    if isinstance(tool, (Action, Check)):
        attempt_parts.extend((tool.description, tool.target if isinstance(tool, Action) and tool.target else ""))
    elif isinstance(tool, Speak):
        attempt_parts.extend((tool.message, tool.target or ""))
    elif isinstance(tool, Attack):
        attempt_parts.append(tool.target)
    evidence_words = _word_roots(" ".join((*attempt_parts, *(event.text for event in owner_events))))

    objective_words = _word_roots(quest.plan[quest.current_step])
    context_words = _word_roots(f"{quest.title} {quest.description}")
    generic_words = _word_roots(" ".join(_OBJECTIVE_GENERIC_WORDS))
    verb_words = {
        form
        for forms in _FINAL_OBJECTIVE_VERB_FORMS.values()
        for form in _word_roots(" ".join(forms))
    }
    verb_words.update(_word_roots("ask check examine identify inspect investigate question search"))
    anchors = objective_words - context_words - generic_words - verb_words - _FINAL_OBJECTIVE_STOP_WORDS - {"s"}
    required_matches = min(2, len(anchors))
    return (
        not anchors
        or len(anchors & evidence_words) >= required_matches
        or _is_grounded_referential_contact(state, quest, tool, owner_events)
    )


def _advance_grounded_objective(
    state: WorldState,
    actor_id: str,
    tool: CharacterTool,
    events: list[HistoryEvent],
    steps_before: dict[str, int],
) -> list[str]:
    """Advance an unchanged search objective when this turn produced concrete discovery evidence."""
    if not isinstance(tool, (Action, Check)):
        return []
    actor = state.characters.get(actor_id)
    if not actor:
        return []
    evidence_markers = (" learns:", " finds ", " found ", " discovers ", " reveals ", " picks up ")
    has_evidence = any(
        actor_id in event.characters
        and not _FAILED_ATTEMPT_PATTERN.search(event.text.casefold())
        and any(marker in f" {event.text.lower()} " for marker in evidence_markers)
        for event in events
    )
    if not has_evidence:
        return []

    location_name = actor.location.replace("-", " ").lower()
    results: list[str] = []
    for quest in state.quests.values():
        if quest.owner != actor_id or quest.status.lower() in {"completed", "failed"}:
            continue
        if quest.current_step != steps_before.get(quest.id) or quest.current_step >= len(quest.plan):
            continue
        objective = quest.plan[quest.current_step].replace("-", " ").lower()
        search_objective = any(verb in objective for verb in ("search", "investigate", "examine", "find clues"))
        if search_objective and location_name in objective:
            if quest.current_step == len(quest.plan) - 1 and not _can_advance_final_objective(quest, tool, events):
                continue
            results.append(WorldOperations(state).advance_quest(quest.id, advance=True))
    return results


def _complete_resolved_final_objective(
    state: WorldState,
    actor_id: str,
    tool: CharacterTool,
    events: list[HistoryEvent],
) -> list[str]:
    """Complete planned quests when their owner directly resolves the final objective early."""
    results: list[str] = []
    for quest in state.quests.values():
        if (
            quest.owner != actor_id
            or quest.status.casefold() in {"completed", "failed"}
            or not quest.plan
            or quest.current_step >= len(quest.plan)
        ):
            continue

        final_step = len(quest.plan) - 1
        final_objective = quest.model_copy(update={"current_step": final_step})
        if not _can_advance_final_objective(final_objective, tool, events):
            continue

        skipped = quest.plan[quest.current_step:final_step]
        quest.steps.extend(f"{objective} — bypassed by direct resolution" for objective in skipped)
        quest.current_step = final_step
        results.append(WorldOperations(state).advance_quest(
            quest.id,
            advance=True,
            step="resolved directly",
        ))
    return results


def _record_resolution_if_needed(
    state: WorldState,
    actor_id: str,
    history_size: int,
    resolution: str,
) -> list[HistoryEvent]:
    """Keep a resolved turn visible when it produced no state-operation event."""
    if len(state.history) == history_size and resolution.strip():
        actor = state.characters.get(actor_id)
        state.history.append(HistoryEvent(
            text=resolution.strip(),
            location=actor.location if actor else "",
            characters=[actor_id],
        ))
    return state.history[history_size:]


def _campaign_outcome(state: WorldState, pc_id: str) -> str | None:
    """Return a terminal outcome once every quest owned by the PC is resolved."""
    quests = [quest for quest in state.quests.values() if quest.owner == pc_id]
    if not quests or any(quest.status.lower() not in {"completed", "failed"} for quest in quests):
        return None
    return "failed" if any(quest.status.lower() == "failed" for quest in quests) else "completed"


def _announce_campaign_outcome(state: WorldState, pc_id: str, logger: Logger, outcome: str) -> None:
    quests = [quest.id for quest in state.quests.values() if quest.owner == pc_id]
    if outcome == "completed":
        print(f"\n=== {pc_id}'s quests are complete. The campaign ends in victory. ===")
        logger.log_event("campaign_completed", pc=pc_id, quests=quests)
    else:
        print(f"\n=== {pc_id}'s quests have failed. The campaign ends in defeat. ===")
        logger.log_event("campaign_failed", pc=pc_id, quests=quests)


def _announce_pc_death(pc_id: str, logger: Logger) -> None:
    print(f"\n=== {pc_id} has died. The campaign ends. ===")
    logger.log_event("pc_death", pc=pc_id)


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
        logger.log_event("player_action", actor=actor_id, tool=intent.kind, action=intent.model_dump(mode="json"))
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
    state: WorldState, new_events: list[HistoryEvent], logger: Logger, replay: ReplayTape | None = None
) -> DMResult:
    """One LLM call: enrich the world, review quests, and estimate time for this tick's new events."""
    if not new_events:
        return DMResult(creates=[], modifies=[], minutes=[])
    if replay and replay.is_playback:
        return replay.dm()
    new_texts = [e.text for e in new_events]
    tail = state.history[: -len(new_events)]  # new_events is non-empty here (checked above)
    context_events = [e.text for e in tail[-10:]]
    label = "dm"
    try:
        with logger.run(label):
            result = await dm_agent.run(
                "Register any new entities, report quest progress, and estimate minutes elapsed for each new event.",
                deps=DMDeps(state=state, context_events=context_events, new_events=new_texts),
                usage_limits=_DM_USAGE,
            )
            logger.log_messages(label, result.all_messages())
    except UsageLimitExceeded as exc:
        print(f"  [limit] dm review ended: {exc}", flush=True)
        fallback = DMResult(creates=[], modifies=[], minutes=[1] * len(new_events))
        return replay.dm(fallback) if replay else fallback
    dm_result = result.output
    if not dm_result.minutes or len(dm_result.minutes) != len(new_events):
        dm_result.minutes = [1] * len(new_events)
    dm_result.creates = sorted(dm_result.creates, key=lambda i: _CREATE_ORDER.get(i.type, 3))
    return replay.dm(dm_result) if replay else dm_result


async def flow_director(
    state: WorldState, location_id: str, logger: Logger, replay: ReplayTape | None = None
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
        except UsageLimitExceeded as exc:
            print(f"  [limit] director beat ended: {exc}", flush=True)
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
    if beat.create:
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
    quest_steps_before = {quest.id: quest.current_step for quest in state.quests.values()}

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
    if replay and replay.is_playback and isinstance(intent, (Action, Check)):
        resolution = replay.action_resolution(actor_id, state)
    else:
        resolution = await resolve(intent, state, logger=logger)
        if replay and isinstance(intent, (Action, Check)):
            replay.action_resolution(actor_id, state, resolution)
    new_events = _record_resolution_if_needed(state, actor_id, pre, resolution)

    # 2. One DM call: time estimates, new entities, and quest progress.
    dm = await flow_dm(state, new_events, logger, replay)
    event_minutes = [max(0, minutes) for minutes in dm.minutes[:len(new_events)]]
    event_minutes.extend([0] * (len(new_events) - len(event_minutes)))
    if event_minutes:
        event_minutes[0] = max(event_minutes[0], _minimum_action_minutes(intent))
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
            _log_rejected_world_update(logger, create_intent, "unsupported_location_evidence")
            continue
        if not _can_anchor_new_npc(create_intent, new_events):
            print(
                f"  [world] {create_intent.name}: cannot reveal NPC at "
                f"{create_intent.location!r} without current-event identity and location evidence."
            )
            _log_rejected_world_update(logger, create_intent, "unsupported_npc_evidence")
            continue
        await resolve(create_intent, state, logger=logger)
    for event in state.history[pre_enrich:]:
        print(f"  {event.text}  (revealed)")

    # 2c. Review quests. Step-level progress; may append XP events via advance_quest.
    pre_quest = len(state.history)
    for update in dm.modifies:
        if update.action == "advance_faction_clock" and not _can_accelerate_faction_clock(
            state,
            intent,
            update.target_id,
            update.other_id,
            new_events,
        ):
            print(f"  [faction] {update.target_id}: this turn cannot accelerate that clock.")
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
            print(f"  [world] {update.target_id}: cannot remove the active player character.")
            continue
        quest_key = next(
            (candidate for candidate in state.quests if slugify(candidate) == slugify(update.target_id)),
            None,
        )
        quest = state.quests.get(quest_key) if quest_key else None
        is_final_status = update.status and update.status.casefold() == "completed"
        if (
            update.action == "update_quest"
            and (update.advance or is_final_status)
            and quest
            and not _can_advance_objective(state, quest, intent, new_events, update.step)
        ):
            print(f"  [quest] {update.target_id}: current objective requires its owner's relevant resolution attempt.")
            continue
        msg = await resolve(update, state, logger=logger)
        print(f"  [quest] {update.target_id}: {msg}")
    for msg in _advance_grounded_objective(state, actor_id, intent, new_events, quest_steps_before):
        print(f"  [quest] grounded fallback: {msg}")
    for msg in _complete_resolved_final_objective(state, actor_id, intent, new_events):
        print(f"  [quest] final resolution: {msg}")
    for event in state.history[pre_quest:]:
        print(f"  {event.text}")

    # 3. Director beat: when the story stalls, one proactive complication breaks it.
    if _is_stalled(state):
        await flow_director(state, state.characters[active_pc_id].location, logger, replay)
        state.last_quest_advance_time = state.time  # suppress consecutive fires
    return True


# ============================================================
# Entry point
# ============================================================

def _snapshot(state: WorldState) -> dict:
    """World/quest/xp counters for scorecard.py — logged before and after every tick."""
    quests = state.quests.values()
    return {
        "quests_active": sum(1 for q in quests if q.status.lower() not in {"completed", "failed"}),
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
        pc_id = new_character["character_id"]
    else:
        pc_id = character_id or manager.manifest["pc"]
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
        metrics_reader=read_metrics if os.getenv("LLM_PROVIDER", "local") == "local" else None,
    )
    run_failed = False
    try:
        manager.save_state(state)
        if progress_callback:
            progress_callback(state)
        logger.log_event("world_snapshot", **_snapshot(state))
        for t in range(max_turns):
            if outcome := _campaign_outcome(state, pc_id):
                _announce_campaign_outcome(state, pc_id, logger, outcome)
                break
            if stop_check and stop_check():
                logger.log_event("run_stopped")
                break
            if state.characters[pc_id].stats.hp <= 0:
                _announce_pc_death(pc_id, logger)
                break
            print(f"\n--- Tick {state.time + 1} ---")
            logger.log_turn(state.time + 1)
            pre_minutes = state.minutes_elapsed
            try:
                turn_completed = await tick(pc_id, state, state.time, logger, pc_controller, replay)
            except AgentRunError as exc:
                logger.log_event("run_error", error=type(exc).__name__, message=str(exc))
                print(f"  [error] campaign stopped: {exc}", flush=True)
                run_failed = True
                break
            if turn_completed is False:
                logger.log_event("run_stopped", reason="no_character_action")
                break
            # Off-screen agendas march with in-world time, not turn count.
            WorldOperations(state).advance_faction_clocks_hourly(pre_minutes, state.minutes_elapsed)
            await compact_history(state, logger, replay)  # between ticks only — never mid-tick
            logger.log_event("world_snapshot", **_snapshot(state))
            state.time += 1
            manager.save_state(state)
            if progress_callback:
                progress_callback(state)
            if outcome := _campaign_outcome(state, pc_id):
                _announce_campaign_outcome(state, pc_id, logger, outcome)
                break
            if state.characters[pc_id].stats.hp <= 0:
                _announce_pc_death(pc_id, logger)
                break
        if replay:
            replay.assert_consumed()
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
    replay: ReplayTape | None = None,
    pc_controller: Callable[[str, WorldState], Awaitable[CharacterTool]] | None = None,
) -> bool:
    # Narrative text (arrows, em-dashes) doesn't fit Windows' default console codepage.
    if isinstance(sys.stdout, TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return asyncio.run(
        _run_game(
            scenario,
            character_id,
            max_turns,
            new_character,
            replay=replay,
            pc_controller=pc_controller,
        )
    )
