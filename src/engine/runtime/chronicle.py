"""History compaction: between ticks, archive aged-out events as a prose chapter.

Each chapter is written to <run_dir>/chapters/chapter_NNNN.json with the raw events it
was narrated from; only its short summary stays in state.chronicle, which is kept
bounded by folding older summaries into one "saga so far" entry.

Never call this mid-tick — tick() slices state.history by position (e.g. `state.history[pre:]`)
to find events created during the current turn, and compaction shortens history in place.
"""

import json
import os
from pathlib import Path

from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from src.agents.dm.chronicler import (
    CHAPTER_INSTRUCTIONS,
    SAGA_INSTRUCTIONS,
    agent as chronicler_agent,
    chapter_prompt,
    parse_chapter,
    readable,
)
from src.engine.state import WorldState
from src.engine.state.models import HistoryEvent
from src.interface.session_log import Logger
from .replay import ReplayTape

# Compact once history grows past this many events, keeping only the most recent tail.
HISTORY_COMPACT_THRESHOLD = 60
HISTORY_KEEP_RECENT = 30
# A chapter may take up to this many extra events so it ends on a change of scene.
CHAPTER_BREAK_WINDOW = 10
# state.chronicle holds the saga so far plus this many verbatim chapter summaries.
CHRONICLE_RECENT = 3
SUMMARY_MAX_CHARS = 600
SAGA_MAX_CHARS = 1000
MIN_PROSE_CHARS = 300

_USAGE = UsageLimits(request_limit=2)
# max_tokens bounds a rambling reply (~50s at 40 tokens/s). A presence penalty, which
# keeps the game agents from repeating themselves, mangles names that recur in prose.
_SETTINGS: ModelSettings = {
    "max_tokens": 2000,
    "temperature": 0.7,
    "presence_penalty": 0.0,
}


def clip(text: str, limit: int) -> str:
    """Shorten text to `limit` characters at a word boundary."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def digest(events: list[HistoryEvent]) -> str:
    """Deterministic fallback summary: a plain, truncated join of the event texts."""
    return clip(" ".join(e.text for e in events), SUMMARY_MAX_CHARS)


def chapter_end(history: list[HistoryEvent]) -> int:
    """Index to archive up to: the default cut, moved later to the next change of scene."""
    cut = len(history) - HISTORY_KEEP_RECENT
    for i in range(cut, min(cut + CHAPTER_BREAK_WINDOW, len(history))):
        if history[i].location != history[i - 1].location:
            return i
    return cut


def _fallback_prose(events: list[HistoryEvent]) -> str:
    """One paragraph per scene, made of the event texts."""
    paragraphs: list[list[str]] = []
    for i, event in enumerate(events):
        if i == 0 or event.location != events[i - 1].location:
            paragraphs.append([])
        paragraphs[-1].append(event.text)
    return "\n\n".join(" ".join(p) for p in paragraphs)


async def _ask(
    instructions: str,
    prompt: str,
    key: str,
    logger: Logger,
    replay: ReplayTape | None,
) -> str:
    """One chronicler call; '' on failure. Recorded raw so playback parses identically."""
    if replay and replay.is_playback:
        return replay.chronicle(key=key)
    try:
        with logger.run("chronicle"):
            result = await chronicler_agent.run(
                prompt,
                instructions=instructions,
                usage_limits=_USAGE,
                model_settings=_SETTINGS,
            )
            logger.log_messages("chronicle", result.all_messages())
        reply = result.output
    except AgentRunError as exc:
        print(f"  [fallback] chronicler {key} call failed: {exc}", flush=True)
        reply = ""
    return replay.chronicle(reply, key) if replay else reply


async def compact_history(
    state: WorldState,
    logger: Logger,
    replay: ReplayTape | None,
    run_dir: Path,
    hero: str,
) -> None:
    """Archive the oldest events as the next chapter once history exceeds the threshold."""
    if len(state.history) <= HISTORY_COMPACT_THRESHOLD:
        return
    cut = chapter_end(state.history)
    events = state.history[:cut]
    chapters_dir = Path(run_dir) / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    written = sorted(chapters_dir.glob("chapter_*.json"))
    number = len(written) + 1
    previous_ending = ""
    if written:
        previous = json.loads(written[-1].read_text("utf-8"))
        previous_ending = previous["prose"].rsplit("\n\n", 1)[-1]

    prompt = chapter_prompt(
        state,
        events,
        number=number,
        hero=hero,
        start_minutes=state.minutes_elapsed
        - sum(e.minutes_elapsed for e in state.history),
        previous_ending=previous_ending,
    )
    reply = await _ask(CHAPTER_INSTRUCTIONS, prompt, "chapter", logger, replay)
    title, summary, prose = parse_chapter(reply)
    chapter = {
        "number": number,
        "title": title if 0 < len(title) <= 80 else f"Chapter {number}",
        "prose": prose
        if len(prose) >= MIN_PROSE_CHARS
        else readable(_fallback_prose(events), state),
        "summary": clip(summary, SUMMARY_MAX_CHARS) if summary else digest(events),
        "events": [e.model_dump(mode="json") for e in events],
    }
    path = chapters_dir / f"chapter_{number:04d}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(chapter, indent=2, ensure_ascii=False), "utf-8")
    os.replace(temp, path)

    state.chronicle.append(chapter["summary"])
    del state.history[:cut]
    if len(state.chronicle) > CHRONICLE_RECENT + 1:
        older = state.chronicle[:-CHRONICLE_RECENT]
        saga = await _ask(SAGA_INSTRUCTIONS, "\n\n".join(older), "saga", logger, replay)
        if not saga.strip():  # keep the origin and the latest fold, lose the middle
            joined, half = " ".join(" ".join(older).split()), SAGA_MAX_CHARS // 2 - 2
            saga = joined[:half] + " … " + joined[-half:]
        state.chronicle[:-CHRONICLE_RECENT] = [clip(saga, SAGA_MAX_CHARS)]
