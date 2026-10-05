"""Game loop. Each tick: one actor takes a turn → resolve → stamp elapsed time → enrich world → review quests."""

import asyncio
import os
import re
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime
from io import TextIOWrapper

from pydantic_ai.exceptions import ModelAPIError, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from src.engine.state import StateManager, WorldOperations, WorldState, is_dialogue
from src.engine.state.models import HistoryEvent
from src.agents.character.agent import CharacterDeps, agent as character_agent
from src.agents.character.tools import Action, Attack, CharacterTool, Check, Speak, Travel, Wait
from src.agents.dm.agent import DMDeps, DMResult, agent as dm_agent
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


# ─── Scheduler ────────────────────────────────────────────────

def _scene_actors(state: WorldState, active_pc_id: str) -> list[str]:
    """Living characters present in the active PC's current location. PC first, then others by id."""
    scene = state.characters[active_pc_id].location
    present = [cid for cid, c in state.characters.items() if c.location == scene and c.stats.hp > 0]
    present.sort(key=lambda cid: (cid != active_pc_id, cid))
    return present


def _pick_next_actor(state: WorldState, active_pc_id: str, tick: int) -> str:
    actors = _scene_actors(state, active_pc_id)
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


def _can_advance_final_objective(quest, tool: CharacterTool, events: list[HistoryEvent] | None = None) -> bool:
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
    return False


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
        and not any(negative in event.text.lower() for negative in ("nothing", "no new", "cannot", "fails"))
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
) -> None:
    actor_id = _pick_next_actor(state, active_pc_id, tick_index)
    quest_steps_before = {quest.id: quest.current_step for quest in state.quests.values()}

    controller = pc_controller if actor_id == active_pc_id else None
    intent = await flow_agent_turn(actor_id, state, logger, controller, replay)
    if intent is None:
        return
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
    if new_events and dm.minutes:
        dm.minutes[0] = max(dm.minutes[0], _minimum_action_minutes(intent))

    # 2a. Stamp elapsed time onto the resulting events and advance the clock.
    for event, mins in zip(new_events, dm.minutes):
        event.minutes_elapsed = mins
    state.minutes_elapsed += sum(dm.minutes)
    for event in new_events:
        print(f"  {event.text}  (+{event.minutes_elapsed}m)")
    if new_events:
        print(f"  [clock: {format_clock(state.minutes_elapsed)}]")

    # 2b. Enrich the world. Revealed entities are retcons — logged but not time-stamped.
    pre_enrich = len(state.history)
    for create_intent in dm.creates:
        await resolve(create_intent, state, logger=logger)
    for event in state.history[pre_enrich:]:
        print(f"  {event.text}  (revealed)")

    # 2c. Review quests. Step-level progress; may append XP events via advance_quest.
    pre_quest = len(state.history)
    for update in dm.modifies:
        quest = state.quests.get(update.target_id)
        is_final_status = update.status and update.status.casefold() == "completed"
        if (
            update.action == "update_quest"
            and (update.advance or is_final_status)
            and quest
            and not _can_advance_final_objective(quest, intent, new_events)
        ):
            print(f"  [quest] {update.target_id}: final objective requires its owner's active resolution attempt.")
            continue
        msg = await resolve(update, state, logger=logger)
        print(f"  [quest] {update.target_id}: {msg}")
    for msg in _advance_grounded_objective(state, actor_id, intent, new_events, quest_steps_before):
        print(f"  [quest] grounded fallback: {msg}")
    for event in state.history[pre_quest:]:
        print(f"  {event.text}")

    # 3. Director beat: when the story stalls, one proactive complication breaks it.
    if _is_stalled(state):
        await flow_director(state, state.characters[active_pc_id].location, logger, replay)
        state.last_quest_advance_time = state.time  # suppress consecutive fires


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
) -> None:
    if replay:
        scenario, character_id = replay.resolve_context(scenario, character_id)
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    manager = StateManager(scenario=scenario, run_id=None if resume else session_id, resume=resume)
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
                print(f"\n=== {pc_id} has died. The campaign ends. ===")
                logger.log_event("pc_death", pc=pc_id)
                break
            print(f"\n--- Tick {state.time + 1} ---")
            logger.log_turn(t + 1)
            pre_minutes = state.minutes_elapsed
            try:
                await tick(pc_id, state, state.time, logger, pc_controller, replay)
            except ModelAPIError as exc:
                logger.log_event("run_error", error=type(exc).__name__, message=str(exc))
                print(f"  [error] campaign stopped: {exc}", flush=True)
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
    finally:
        logger.close(turns_completed=state.time)
    if replay:
        replay.assert_consumed()


def run_game(
    character_id: str | None = None,
    max_turns: int = 50,
    scenario: str | None = None,
    new_character: dict | None = None,
    *,
    replay: ReplayTape | None = None,
    pc_controller: Callable[[str, WorldState], Awaitable[CharacterTool]] | None = None,
) -> None:
    # Narrative text (arrows, em-dashes) doesn't fit Windows' default console codepage.
    if isinstance(sys.stdout, TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(_run_game(scenario, character_id, max_turns, new_character, replay=replay, pc_controller=pc_controller))
