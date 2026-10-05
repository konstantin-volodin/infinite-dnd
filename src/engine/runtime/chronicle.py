"""History compaction: between ticks, roll aged-out events into one chronicle entry.

Never call this mid-tick — tick() slices state.history by position (e.g. `state.history[pre:]`)
to find events created during the current turn, and compaction shortens history in place.
"""

from pydantic_ai.exceptions import ModelAPIError, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from src.agents.dm.chronicler import ChroniclerDeps, agent as chronicler_agent
from src.engine.state import WorldState
from src.engine.state.models import HistoryEvent
from src.interface.session_log import Logger
from .replay import ReplayTape

# Compact once history grows past this many events, keeping only the most recent tail.
HISTORY_COMPACT_THRESHOLD = 60
HISTORY_KEEP_RECENT = 30

_CHRONICLE_USAGE = UsageLimits(request_limit=8)


def digest(events: list[HistoryEvent]) -> str:
    """Deterministic fallback chronicle entry: a plain, truncated join of the event texts."""
    joined = " ".join(e.text for e in events)
    return joined if len(joined) <= 600 else joined[:597] + "..."


async def compact_history(state: WorldState, logger: Logger, replay: ReplayTape | None = None) -> None:
    """Summarize the oldest events into one chronicle entry once history exceeds the threshold."""
    if len(state.history) <= HISTORY_COMPACT_THRESHOLD:
        return
    archived = state.history[:-HISTORY_KEEP_RECENT]

    if replay and replay.is_playback:
        summary = replay.chronicle()
    else:
        try:
            with logger.run("chronicle"):
                result = await chronicler_agent.run(
                    "Compress these events into one short chronicle entry.",
                    deps=ChroniclerDeps(events=[e.text for e in archived]),
                    usage_limits=_CHRONICLE_USAGE,
                )
                logger.log_messages("chronicle", result.all_messages())
            summary = result.output
        except (ModelAPIError, UsageLimitExceeded) as exc:
            print(f"  [fallback] chronicle compaction ended: {exc}", flush=True)
            summary = digest(archived)
        if replay:
            summary = replay.chronicle(summary)

    if not summary.strip():
        summary = digest(archived)
    state.chronicle.append(summary)
    del state.history[:-HISTORY_KEEP_RECENT]
