"""World-state snapshots: discovery, reading, and tick-to-tick diffs for the state dashboard."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.world import read_manifest

DEFAULT_STATE_DIR = Path("world-state")

_TICK_RE = re.compile(r"^world_state_(\d+)\.json$")


# ============================================================
# Discovery
# ============================================================

def list_ticks(scenario_dir: Path) -> list[int]:
    """Sorted tick numbers that have a snapshot file in a scenario's state directory."""
    ticks: list[int] = []
    for candidate in scenario_dir.glob("world_state_*.json"):
        match = _TICK_RE.match(candidate.name)
        if match:
            ticks.append(int(match.group(1)))
    return sorted(ticks)


def list_state_runs(state_dir: Path) -> list[dict[str, Any]]:
    """Run directories (scenario/run_id) under the state dir that hold at least one snapshot, newest first."""
    target = Path(state_dir)
    if not target.exists():
        return []
    result: list[dict[str, Any]] = []
    for scenario_candidate in sorted(target.iterdir()):
        if not scenario_candidate.is_dir():
            continue
        manifest = _safe_manifest(scenario_candidate.name)
        for run_candidate in sorted(scenario_candidate.iterdir()):
            if not run_candidate.is_dir():
                continue
            ticks = list_ticks(run_candidate)
            if not ticks:
                continue
            latest = run_candidate / f"world_state_{ticks[-1]}.json"
            result.append(
                {
                    "scenario": scenario_candidate.name,
                    "run_id": run_candidate.name,
                    "title": manifest.get("title", scenario_candidate.name),
                    "pc": manifest.get("pc"),
                    "tick_count": len(ticks),
                    "latest_tick": ticks[-1],
                    "modified_time": datetime.fromtimestamp(latest.stat().st_mtime).isoformat(),
                }
            )
    result.sort(key=lambda item: item["modified_time"], reverse=True)
    return result


def _safe_manifest(scenario: str) -> dict[str, Any]:
    try:
        return read_manifest(scenario)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ============================================================
# Loading
# ============================================================

def load_view(state_dir: Path, scenario: str, run_id: str, tick: int | None = None) -> dict[str, Any]:
    """One snapshot plus everything the dashboard needs: tick list, PC id, diff vs previous tick, event clocks."""
    run_path = Path(state_dir) / scenario / run_id
    ticks = list_ticks(run_path)
    if not ticks:
        raise FileNotFoundError(f"No snapshots found for scenario '{scenario}' run '{run_id}'")

    if tick is None:
        tick = ticks[-1]
    if tick not in ticks:
        raise FileNotFoundError(f"No snapshot for tick {tick} in scenario '{scenario}' run '{run_id}'")

    state = _read_snapshot(run_path, tick)
    tick_index = ticks.index(tick)
    prev_state = _read_snapshot(run_path, ticks[tick_index - 1]) if tick_index > 0 else None
    manifest = _safe_manifest(scenario)

    return {
        "scenario": scenario,
        "run_id": run_id,
        "title": manifest.get("title", scenario),
        "pc": manifest.get("pc"),
        "tick": tick,
        "ticks": ticks,
        "latest_tick": ticks[-1],
        "state": state,
        "history_clocks": _history_clocks(state),
        "changes": _diff(prev_state, state) if prev_state else None,
    }


def load_series(state_dir: Path, scenario: str, run_id: str) -> dict[str, list[dict[str, int]]]:
    """Character stat history across every available snapshot."""
    run_path = Path(state_dir) / scenario / run_id
    result: dict[str, list[dict[str, int]]] = {}
    for tick in list_ticks(run_path):
        snapshot = _read_snapshot(run_path, tick)
        for character_id, character in snapshot.get("characters", {}).items():
            stats = character.get("stats", {})
            result.setdefault(character_id, []).append({
                "tick": tick,
                "hp": stats.get("hp", 0),
                "xp": stats.get("xp", 0),
                "gold": stats.get("gold", 0),
            })
    return result


def _read_snapshot(run_path: Path, tick: int) -> dict[str, Any]:
    with (run_path / f"world_state_{tick}.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _history_clocks(state: dict[str, Any]) -> list[str]:
    """Running world clock at each history event (events carry per-event minutes, not absolute time)."""
    clocks: list[str] = []
    elapsed = 0
    for event in state.get("history", []):
        elapsed += event.get("minutes_elapsed", 0)
        clocks.append(format_clock(elapsed))
    return clocks


def format_clock(total_minutes: int) -> str:
    day = total_minutes // 1440 + 1
    remainder = total_minutes % 1440
    return f"day {day} · {remainder // 60:02d}:{remainder % 60:02d}"


# ============================================================
# Diffing
# ============================================================

def _diff(prev: dict[str, Any], curr: dict[str, Any]) -> dict[str, Any]:
    """What changed since the previous snapshot, keyed by entity id. Empty sections are omitted per entity."""
    return {
        "characters": _diff_characters(prev.get("characters", {}), curr.get("characters", {})),
        "quests": _diff_quests(prev.get("quests", {}), curr.get("quests", {})),
        "new_locations": [lid for lid in curr.get("locations", {}) if lid not in prev.get("locations", {})],
        "new_history": len(curr.get("history", [])) - len(prev.get("history", [])),
    }


def _diff_characters(prev: dict[str, Any], curr: dict[str, Any]) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}
    for cid, char in curr.items():
        before = prev.get(cid)
        if before is None:
            changes[cid] = {"new": True}
            continue
        delta: dict[str, Any] = {}
        for stat in ("hp", "xp", "gold", "level"):
            diff = char.get("stats", {}).get(stat, 0) - before.get("stats", {}).get(stat, 0)
            if diff:
                delta[stat] = diff
        if char.get("location") != before.get("location"):
            delta["moved_from"] = before.get("location")
        gained = [item for item in char.get("inventory", []) if item not in before.get("inventory", [])]
        lost = [item for item in before.get("inventory", []) if item not in char.get("inventory", [])]
        if gained:
            delta["gained"] = gained
        if lost:
            delta["lost"] = lost
        if delta:
            changes[cid] = delta
    return changes


def _diff_quests(prev: dict[str, Any], curr: dict[str, Any]) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}
    for qid, quest in curr.items():
        before = prev.get(qid)
        if before is None:
            changes[qid] = {"new": True}
            continue
        delta: dict[str, Any] = {}
        if quest.get("status") != before.get("status"):
            delta["status_from"] = before.get("status")
        if quest.get("steps") != before.get("steps"):
            delta["steps_updated"] = True
        if delta:
            changes[qid] = delta
    return changes
