"""Story recap: turn a finished run's saved WorldState into readable markdown.

Deterministic — no LLM calls. The narrative already lives in the saved state:
chronicle (compacted eras), history (recent events), quests, and characters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.rules import get_health_status
from src.engine.state.models import Character, HistoryEvent, Quest, WorldState
from src.interface.world_state import DEFAULT_STATE_DIR, format_clock, list_state_runs, load_view


def build_recap(state: WorldState, *, title: str, scenario: str, run_id: str) -> str:
    """Render a finished run's WorldState as a markdown recap."""
    sections = [
        f"# {title} — Recap\n\n*{scenario} · {run_id}*",
        _previously(state.chronicle),
        _story_so_far(state.history),
        _quest_outcomes(state.quests),
        _epilogue(state.characters),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def _previously(chronicle: list[str]) -> str:
    if not chronicle:
        return ""
    return "## Previously…\n\n" + "\n\n".join(chronicle)


def _story_so_far(history: list[HistoryEvent]) -> str:
    """Group events into sections, starting a new one whenever the clock or location shifts."""
    if not history:
        return ""
    blocks = ["## The Story So Far"]
    elapsed = 0
    group_key: tuple[str, str] | None = None
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(" ".join(paragraph))
            paragraph.clear()

    for event in history:
        elapsed += event.minutes_elapsed
        clock = format_clock(elapsed)
        key = (clock.split(" · ")[0], event.location)
        if key != group_key:
            flush()
            group_key = key
            header = f"{clock} · {event.location}" if event.location else clock
            blocks.append(f"### {header}")
        paragraph.append(event.text)
    flush()
    return "\n\n".join(blocks)


def _quest_outcomes(quests: dict[str, Quest]) -> str:
    if not quests:
        return ""
    lines = ["## Quest Outcomes"]
    for quest in quests.values():
        progress = f"{quest.current_step}/{len(quest.plan)} steps" if quest.plan else f"{len(quest.steps)} steps logged"
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
    parser = argparse.ArgumentParser(description="Export a markdown recap of a finished run.")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR, help="Root directory of saved world states.")
    parser.add_argument("--scenario", help="Scenario id (default: newest run's scenario).")
    parser.add_argument("--run-id", help="Run id (default: newest run of the chosen scenario).")
    parser.add_argument("--out", type=Path, help="Output path (default: <run-id>_recap.md next to the run's snapshots).")
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
    markdown = build_recap(state, title=view["title"], scenario=scenario, run_id=run_id)

    out_path = args.out or (Path(args.state_dir) / scenario / run_id / f"{run_id}_recap.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
