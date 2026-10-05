"""State Manager - load/save world state from JSON files."""

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from src.engine.state.models import (
    WorldState,
    Character,
    Location,
    Faction,
    Quest,
)
from src.world import pick_scenario, read_manifest, scenario_dir

_SNAPSHOT_PATTERN = re.compile(r"world_state_(\d+)\.json")


def _snapshots(directory: Path):
    """Yield (numeric id, mtime, filename) for valid snapshot files in a directory."""
    for path in directory.glob("world_state_*.json"):
        match = _SNAPSHOT_PATTERN.fullmatch(path.name)
        if match:
            yield int(match.group(1)), path.stat().st_mtime, path.name


class StateManager:
    """Load, persist, and reset world state for a chosen scenario."""

    def __init__(
        self,
        scenario: str | None = None,
        state_dir: str = "world-state",
        run_id: str | None = None,
        *,
        resume: bool = False,
    ):
        self.ROOT_DIR = Path(__file__).resolve().parents[3]
        self.scenario = scenario or pick_scenario()
        self.manifest = read_manifest(self.scenario)
        self.setup_dir = scenario_dir(self.scenario)
        self.scenario_dir = self.ROOT_DIR / Path(state_dir) / self.scenario
        self.scenario_dir.mkdir(parents=True, exist_ok=True)
        latest_run = self.latest_run_id() if resume and run_id is None else None
        self.run_id = run_id or latest_run or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.state_dir = self.scenario_dir / self.run_id
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def latest_run_id(self) -> str | None:
        """Return the run whose newest valid snapshot was modified most recently."""
        latest: tuple[float, str] | None = None
        for run_dir in self.scenario_dir.iterdir():
            if not run_dir.is_dir():
                continue
            mtimes = [mtime for _, mtime, _ in _snapshots(run_dir)]
            if mtimes:
                candidate = (max(mtimes), run_dir.name)
                if latest is None or candidate > latest:
                    latest = candidate
        return latest[1] if latest else None

    def read_json(self, path: str | Path):
        with open(path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)

    def init_state(self) -> WorldState:
        """Load fresh world from setup files."""

        locations = {
            loc["id"]: Location.model_validate(loc)
            for loc in self.read_json(self.setup_dir / "locations.json")
        }
        characters = {
            char["id"]: Character.model_validate(char)
            for char in self.read_json(self.setup_dir / "characters.json")
        }
        quests = {
            quest["id"]: Quest.model_validate(quest)
            for quest in self.read_json(self.setup_dir / "quests.json")
        }

        factions_path = self.setup_dir / "factions.json"
        factions = (
            {faction["id"]: Faction.model_validate(faction) for faction in self.read_json(factions_path)}
            if factions_path.exists()
            else {}
        )

        return WorldState(locations=locations, characters=characters, quests=quests, factions=factions)

    def load_state(self, world_state_file: str | None = None) -> WorldState:
        """Load saved state, or create fresh from setup if none exists."""

        if world_state_file:
            state_path = self.state_dir / world_state_file
            if not state_path.exists():
                raise FileNotFoundError(f"Specified world state file {state_path} does not exist.")
            return WorldState(**self.read_json(state_path))

        else:
            state = self.init_state()
            self.save_state(state)
            return state

    def latest_snapshot_name(self) -> str | None:
        """Return the highest numbered snapshot in this scenario, if any."""
        latest: tuple[int, str] | None = None
        for num, _, name in _snapshots(self.state_dir):
            if latest is None or num > latest[0]:
                latest = (num, name)
        return latest[1] if latest else None

    def save_state(self, state: WorldState):
        """Atomically save current world state to JSON."""
        state_file = self.state_dir / f"world_state_{state.time}.json"
        serialized_state = state.model_dump_json(indent=2)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.state_dir,
                prefix=f".{state_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(serialized_state)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            os.replace(temp_path, state_file)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
