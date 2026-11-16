"""Story reading model: prose chapters plus the not-yet-chaptered tail of a run."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.interface.world_state import list_ticks, read_snapshot

_CHAPTER_RE = re.compile(r"^chapter_(\d+)\.json$")


def _chapter_numbers(run_path: Path) -> list[int]:
    return sorted(
        int(match.group(1))
        for path in (run_path / "chapters").glob("chapter_*.json")
        if (match := _CHAPTER_RE.match(path.name))
    )


def _read_chapter(run_path: Path, number: int) -> dict[str, Any]:
    path = run_path / "chapters" / f"chapter_{number:04d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def list_chapters(state_dir: Path, scenario: str, run_id: str) -> list[dict[str, Any]]:
    """Chapter number, title and event count only; prose and events load per chapter."""
    run_path = Path(state_dir) / scenario / run_id
    chapters: list[dict[str, Any]] = []
    for number in _chapter_numbers(run_path):
        try:
            chapter = _read_chapter(run_path, number)
        except (OSError, json.JSONDecodeError):
            continue  # being written right now; the next poll sees it
        chapters.append(
            {
                "number": number,
                "title": chapter.get("title", ""),
                "event_count": len(chapter.get("events", [])),
            }
        )
    return chapters


def load_chapter(
    state_dir: Path, scenario: str, run_id: str, number: int
) -> dict[str, Any]:
    run_path = Path(state_dir) / scenario / run_id
    try:
        return _read_chapter(run_path, number)
    except OSError as exc:
        raise FileNotFoundError(f"Chapter not found: {number}") from exc


def load_tail(state_dir: Path, scenario: str, run_id: str) -> dict[str, Any]:
    """Events after the last chapter (the latest snapshot's history) and the chapter count."""
    run_path = Path(state_dir) / scenario / run_id
    ticks = list_ticks(run_path)
    if not ticks:
        raise FileNotFoundError(f"No snapshots found for '{scenario}' run '{run_id}'")
    state = read_snapshot(run_path, ticks[-1])
    return {
        "tick": ticks[-1],
        "chapter_count": len(_chapter_numbers(run_path)),
        "events": state.get("history", []),
    }
