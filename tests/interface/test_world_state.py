import json

import pytest

from src.interface.world_state import format_clock, list_state_runs, list_ticks, load_view


def _snapshot(tick: int, **overrides) -> dict:
    base = {
        "time": tick,
        "minutes_elapsed": 0,
        "locations": {"tavern": {"id": "tavern", "description": "", "connections": [], "features": [], "items": []}},
        "characters": {
            "hero": {
                "id": "hero",
                "location": "tavern",
                "inventory": ["sword"],
                "stats": {"hp": 5, "max_hp": 5, "level": 1, "xp": 0, "gold": 10},
            }
        },
        "quests": {"find-it": {"id": "find-it", "title": "Find It", "description": "", "status": "active", "steps": []}},
        "history": [],
    }
    base.update(overrides)
    return base


def _write(tmp_path, scenario: str, tick: int, snapshot: dict, run_id: str = "run-1") -> None:
    run_dir = tmp_path / scenario / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"world_state_{tick}.json").write_text(json.dumps(snapshot), encoding="utf-8")


def test_list_ticks_sorted_numerically(tmp_path):
    for tick in (0, 2, 10, 1):
        _write(tmp_path, "demo", tick, _snapshot(tick))
    assert list_ticks(tmp_path / "demo" / "run-1") == [0, 1, 2, 10]


def test_list_state_runs(tmp_path):
    _write(tmp_path, "demo", 0, _snapshot(0))
    _write(tmp_path, "demo", 1, _snapshot(1))
    (tmp_path / "empty-dir").mkdir()

    runs = list_state_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["scenario"] == "demo"
    assert runs[0]["run_id"] == "run-1"
    assert runs[0]["tick_count"] == 2
    assert runs[0]["latest_tick"] == 1


def test_list_state_runs_missing_dir(tmp_path):
    assert list_state_runs(tmp_path / "nope") == []


def test_load_view_defaults_to_latest(tmp_path):
    _write(tmp_path, "demo", 0, _snapshot(0))
    _write(tmp_path, "demo", 1, _snapshot(1, minutes_elapsed=95))

    view = load_view(tmp_path, "demo", "run-1", None)
    assert view["tick"] == 1
    assert view["latest_tick"] == 1
    assert view["ticks"] == [0, 1]
    assert view["state"]["minutes_elapsed"] == 95


def test_load_view_missing_tick_raises(tmp_path):
    _write(tmp_path, "demo", 0, _snapshot(0))
    with pytest.raises(FileNotFoundError):
        load_view(tmp_path, "demo", "run-1", 7)
    with pytest.raises(FileNotFoundError):
        load_view(tmp_path, "missing", "run-1", None)


def test_diff_between_ticks(tmp_path):
    first = _snapshot(0)
    second = _snapshot(1)
    second["characters"]["hero"] = {
        "id": "hero",
        "location": "cellar",
        "inventory": ["sword", "lantern"],
        "stats": {"hp": 3, "max_hp": 5, "level": 1, "xp": 30, "gold": 5},
    }
    second["characters"]["rat"] = {"id": "rat", "location": "cellar", "inventory": [], "stats": {"hp": 2, "max_hp": 2, "level": 1, "xp": 0, "gold": 0}}
    second["quests"]["find-it"]["status"] = "completed"
    second["locations"]["cellar"] = {"id": "cellar", "description": "", "connections": [], "features": [], "items": []}
    second["history"] = [{"text": "hero descends", "location": "cellar", "characters": ["hero"], "minutes_elapsed": 5}]
    _write(tmp_path, "demo", 0, first)
    _write(tmp_path, "demo", 1, second)

    changes = load_view(tmp_path, "demo", "run-1", 1)["changes"]
    hero = changes["characters"]["hero"]
    assert hero["hp"] == -2
    assert hero["xp"] == 30
    assert hero["gold"] == -5
    assert hero["moved_from"] == "tavern"
    assert hero["gained"] == ["lantern"]
    assert changes["characters"]["rat"] == {"new": True}
    assert changes["quests"]["find-it"]["status_from"] == "active"
    assert changes["new_locations"] == ["cellar"]
    assert changes["new_history"] == 1


def test_first_tick_has_no_changes(tmp_path):
    _write(tmp_path, "demo", 0, _snapshot(0))
    assert load_view(tmp_path, "demo", "run-1", 0)["changes"] is None


def test_history_clocks_accumulate(tmp_path):
    snapshot = _snapshot(0, history=[
        {"text": "a", "location": "tavern", "characters": [], "minutes_elapsed": 30},
        {"text": "b", "location": "tavern", "characters": [], "minutes_elapsed": 1450},
    ])
    _write(tmp_path, "demo", 0, snapshot)
    assert load_view(tmp_path, "demo", "run-1", 0)["history_clocks"] == ["day 1 · 00:30", "day 2 · 00:40"]


def test_format_clock():
    assert format_clock(0) == "day 1 · 00:00"
    assert format_clock(1439) == "day 1 · 23:59"
    assert format_clock(1440) == "day 2 · 00:00"
