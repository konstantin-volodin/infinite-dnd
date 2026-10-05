"""Scenario discovery + random selection for world starts.

Each scenario lives under `src/world/<scenario-id>/` and contains four files:
    manifest.json, characters.json, locations.json, quests.json
"""

import json
import random
from pathlib import Path

_WORLD_DIR = Path(__file__).resolve().parent
_REQUIRED = ("manifest.json", "characters.json", "locations.json", "quests.json")


def list_scenarios() -> list[str]:
    """Subdirectories of src/world/ that hold a complete scenario, sorted by id."""
    return sorted(
        p.name
        for p in _WORLD_DIR.iterdir()
        if p.is_dir()
        and not p.name.startswith(("_", "."))
        and all((p / f).exists() for f in _REQUIRED)
    )


def pick_scenario(rng: random.Random | None = None) -> str:
    """Pick one scenario id at random. Pass an `rng` for deterministic tests."""
    scenarios = list_scenarios()
    if not scenarios:
        raise RuntimeError(f"No scenarios found under {_WORLD_DIR}")
    return (rng or random).choice(scenarios)


def read_manifest(scenario: str) -> dict:
    """Load the scenario manifest dict (title, pc, hook)."""
    path = _WORLD_DIR / scenario / "manifest.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scenario_dir(scenario: str) -> Path:
    """Absolute path to a scenario's directory. Raises if it doesn't exist."""
    path = _WORLD_DIR / scenario
    if not path.is_dir():
        raise FileNotFoundError(f"Scenario '{scenario}' not found at {path}")
    return path
