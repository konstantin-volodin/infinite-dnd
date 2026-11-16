"""Book export: a run's prose chapters plus the not-yet-archived tail as one markdown book.

Deterministic — no LLM calls. Chapters live in <run_dir>/chapters/ (written by
compaction); each is followed by a collapsible log of the raw events it was written
from. Events not yet archived, quests, and the cast close the book.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.rules import get_health_status
from src.engine.state.models import Character, HistoryEvent, Quest, WorldState
from src.interface.world_state import (
    DEFAULT_STATE_DIR,
    format_clock,
    list_state_runs,
    load_view,
)


def load_chapters(run_dir: Path) -> list[dict]:
    """Every archived chapter of a run, in order."""
    paths = sorted(Path(run_dir).glob("chapters/chapter_*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def build_recap(
    state: WorldState,
    *,
    title: str,
    scenario: str,
    run_id: str,
    chapters: list[dict] | None = None,
) -> str:
    """Render a run's chapters and latest WorldState as a markdown book."""
    chapters = chapters or []
    sections = [f"# {title}\n\n*{scenario} · {run_id}*"]
    if not chapters and state.chronicle:  # runs saved before chapters existed
        sections.append("## Previously…\n\n" + "\n\n".join(state.chronicle))
    clock = 0
    for chapter in chapters:
        events = [HistoryEvent.model_validate(e) for e in chapter["events"]]
        sections.append(
            f"## Chapter {chapter['number']}: {chapter['title']}\n\n{chapter['prose']}\n\n"
            f"<details><summary>What happened ({len(events)} events)</summary>\n\n"
            f"{_event_log(events, clock)}\n\n</details>"
        )
        clock += sum(e.minutes_elapsed for e in events)
    if state.history:
        tail_start = state.minutes_elapsed - sum(
            e.minutes_elapsed for e in state.history
        )
        sections.append(
            "## Since the last chapter\n\n" + _event_log(state.history, tail_start)
        )
    sections += [_quest_outcomes(state.quests), _epilogue(state.characters)]
    return "\n\n".join(section for section in sections if section) + "\n"


def _event_log(events: list[HistoryEvent], start_minutes: int) -> str:
    """Events as a list, with a clock-and-place line whenever the place or day changes."""
    lines: list[str] = []
    elapsed, group_key = start_minutes, None
    for event in events:
        clock = format_clock(elapsed)
        key = (clock.split(" · ")[0], event.location)
        if key != group_key:
            group_key = key
            header = f"{clock} · {event.location}" if event.location else clock
            lines.append(f"\n**{header}**\n" if lines else f"**{header}**\n")
        lines.append(f"- {event.text}")
        elapsed += event.minutes_elapsed
    return "\n".join(lines)


def _quest_outcomes(quests: dict[str, Quest]) -> str:
    if not quests:
        return ""
    lines = ["## Quest Outcomes"]
    for quest in quests.values():
        progress = (
            f"{quest.current_step}/{len(quest.plan)} steps"
            if quest.plan
            else f"{len(quest.steps)} steps logged"
        )
        line = f"- **{quest.title}** — {quest.status} ({progress})"
        if quest.steps:
            line += f"; last: {quest.steps[-1]}"
        lines.append(line)
    return "\n".join(lines)


def _epilogue(characters: dict[str, Character]) -> str:
    if not characters:
        return ""
    lines = ["## Epilogue"]
    for character in characters.values():
        role = f" ({character.role})" if character.role else ""
        lines.append(
            f"- **{character.id}**{role} — {get_health_status(character)}, "
            f"level {character.stats.level}, {character.stats.gold} gold, at {character.location or 'unknown'}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a run as a markdown book.")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help="Root directory of saved world states.",
    )
    parser.add_argument(
        "--scenario", help="Scenario id (default: newest run's scenario)."
    )
    parser.add_argument(
        "--run-id", help="Run id (default: newest run of the chosen scenario)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output path (default: <run-id>_recap.md next to the run's snapshots).",
    )
    args = parser.parse_args(argv)

    runs = list_state_runs(args.state_dir)
    if args.scenario:
        runs = [run for run in runs if run["scenario"] == args.scenario]
    if args.run_id:
        runs = [run for run in runs if run["run_id"] == args.run_id]
    if not runs:
        parser.error(f"No matching runs found under {args.state_dir}")

    scenario, run_id = runs[0]["scenario"], runs[0]["run_id"]
    view = load_view(args.state_dir, scenario, run_id)
    state = WorldState.model_validate(view["state"])
    markdown = build_recap(
        state,
        title=view["title"],
        scenario=scenario,
        run_id=run_id,
        chapters=load_chapters(Path(args.state_dir) / scenario / run_id),
    )

    out_path = args.out or (
        Path(args.state_dir) / scenario / run_id / f"{run_id}_recap.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
