import json
import os

import pytest

import src.engine.state.loader as loader_module
from src.engine.state.loader import StateManager
from src.engine.state.models import Character, Location, Quest, WorldState
from src.engine.state.operations import WorldOperations
from src.world import list_scenarios


@pytest.mark.parametrize("scenario", list_scenarios())
def test_init_state_for_scenario(scenario, tmp_path):
    manager = StateManager(scenario=scenario, state_dir=str(tmp_path))
    state = manager.init_state()

    assert manager.scenario == scenario
    assert manager.manifest["pc"] in state.characters, f"manifest pc '{manager.manifest['pc']}' not in characters for {scenario}"
    assert isinstance(state, WorldState)

    for char_id, char in state.characters.items():
        assert isinstance(char, Character)
        assert char.id == char_id
        assert char.location in state.locations, f"{char_id} placed at unknown location {char.location}"

    for loc_id, loc in state.locations.items():
        assert isinstance(loc, Location)
        assert loc.id == loc_id
        # Dangling connections are allowed: the DM agent materializes them
        # when events reference the named place.

    for quest_id, quest in state.quests.items():
        assert isinstance(quest, Quest)
        assert quest.id == quest_id
        if quest.owner:
            assert quest.owner in state.characters, f"quest {quest_id} owned by unknown character {quest.owner}"


def test_save_load_round_trip(tmp_path):
    # round-trip save/load on the default (random) scenario
    manager = StateManager(state_dir=str(tmp_path))
    state = manager.load_state()
    state.time += 1
    manager.save_state(state)
    loaded = manager.load_state(world_state_file=f"world_state_{state.time}.json")
    assert loaded == state


def test_save_state_atomically_replaces_existing_snapshot(tmp_path):
    manager = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))
    state = manager.init_state()
    state_file = manager.state_dir / f"world_state_{state.time}.json"
    state_file.write_text("incomplete", encoding="utf-8")

    manager.save_state(state)

    assert WorldState(**json.loads(state_file.read_text(encoding="utf-8"))) == state
    assert list(manager.state_dir.glob("*.tmp")) == []


def test_save_state_leaves_existing_snapshot_intact_if_replace_fails(tmp_path, monkeypatch):
    manager = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))
    state = manager.init_state()
    state_file = manager.state_dir / f"world_state_{state.time}.json"
    state_file.write_text("previous complete snapshot", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(loader_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        manager.save_state(state)

    assert state_file.read_text(encoding="utf-8") == "previous complete snapshot"
    assert list(manager.state_dir.glob("*.tmp")) == []


def test_latest_snapshot_name_uses_numeric_order(tmp_path):
    manager = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))
    assert manager.latest_snapshot_name() is None
    for tick in (2, 10, 3):
        (manager.state_dir / f"world_state_{tick}.json").write_text("{}", encoding="utf-8")
    (manager.state_dir / "world_state_bad.json").write_text("{}", encoding="utf-8")
    assert manager.latest_snapshot_name() == "world_state_10.json"


def test_latest_snapshot_name_skips_corrupt_newer_snapshot(tmp_path):
    manager = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))
    state = manager.init_state()
    state.time = 2
    manager.save_state(state)
    (manager.state_dir / "world_state_3.json").write_text("incomplete", encoding="utf-8")

    assert manager.latest_snapshot_name() == "world_state_2.json"


def test_distinct_run_ids_do_not_collide(tmp_path):
    manager_a = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path), run_id="run-a")
    manager_b = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path), run_id="run-b")

    assert manager_a.state_dir != manager_b.state_dir

    state = manager_a.init_state()
    manager_a.save_state(state)

    assert manager_a.latest_snapshot_name() is not None
    assert manager_b.latest_snapshot_name() is None


def test_resume_selects_latest_run_with_snapshots(tmp_path):
    older = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path), run_id="older")
    newer = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path), run_id="newer")
    empty = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path), run_id="empty")

    older.save_state(older.init_state())
    newer_state = newer.init_state()
    newer_state.time = 2
    newer.save_state(newer_state)
    older_snapshot = older.state_dir / "world_state_0.json"
    newer_snapshot = newer.state_dir / "world_state_2.json"
    older_snapshot.touch()
    newer_snapshot.touch()
    older_snapshot_time = older_snapshot.stat().st_mtime - 10
    os.utime(older_snapshot, (older_snapshot_time, older_snapshot_time))

    resumed = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path), resume=True)

    assert resumed.run_id == "newer"
    assert resumed.latest_snapshot_name() == "world_state_2.json"
    assert empty.latest_snapshot_name() is None


def test_resume_ignores_newer_run_with_only_corrupt_snapshots(tmp_path):
    intact = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path), run_id="intact")
    corrupt = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path), run_id="corrupt")
    intact.save_state(intact.init_state())
    (corrupt.state_dir / "world_state_1.json").write_text("incomplete", encoding="utf-8")

    resumed = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path), resume=True)

    assert resumed.run_id == "intact"
    assert resumed.latest_snapshot_name() == "world_state_0.json"


def test_run_id_auto_generated_when_omitted(tmp_path):
    manager_a = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))
    manager_b = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))

    assert isinstance(manager_a.run_id, str)
    assert manager_a.run_id
    assert manager_a.run_id != manager_b.run_id
    assert manager_a.state_dir != manager_b.state_dir


def test_init_state_backcompat_quest_without_plan(tmp_path, monkeypatch):
    """Old scenario quests.json files (pre-plan) must still load — pydantic defaults fill plan/current_step."""
    manager = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))
    legacy_quests = [{"id": "legacy-quest", "title": "Old Quest", "description": "no plan here", "owner": "", "status": "active"}]

    original_read_json = manager.read_json
    def fake_read_json(path):
        return legacy_quests if str(path).endswith("quests.json") else original_read_json(path)
    monkeypatch.setattr(manager, "read_json", fake_read_json)

    state = manager.init_state()
    quest = state.quests["legacy-quest"]
    assert quest.plan == []
    assert quest.current_step == 0
    assert quest.steps == []


def test_missing_brother_requires_finding_kaelen_after_following_clues(tmp_path):
    state = StateManager(scenario="missing-brother", state_dir=str(tmp_path)).init_state()
    quest = state.quests["find-brother"]
    ops = WorldOperations(state)

    assert quest.plan[-1] == "find Kaelen Swift"
    for _ in quest.plan[:-1]:
        ops.advance_quest(quest.id, advance=True)

    assert quest.status == "active"
    assert quest.current_step == len(quest.plan) - 1
    assert state.characters[quest.owner].stats.xp == 40

    ops.advance_quest(quest.id, advance=True)

    assert quest.status == "completed"
    assert state.characters[quest.owner].stats.xp == 100


def test_init_state_loads_optional_factions(tmp_path):
    manager = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))

    state = manager.init_state()

    faction = state.factions["black-hull-crew"]
    assert faction.goal
    clocks = {clock.id: clock for clock in faction.clocks}
    assert "retaliation" in clocks
    assert clocks["retaliation"].progress == 0


def test_init_state_loads_quest_deadline_clock(tmp_path):
    manager = StateManager(scenario="cursed-heirloom", state_dir=str(tmp_path))

    state = manager.init_state()

    clock = state.factions["encroaching-dawn"].clocks[0]
    assert clock.segments == 6
    assert clock.fail_quest_id == "find-the-circlet"


def test_load_state_backcompat_without_director_fields(tmp_path):
    """Old snapshots (pre-director) must still load — pydantic defaults fill the tracking fields."""
    manager = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))
    data = json.loads(manager.init_state().model_dump_json())
    del data["director_interventions"]
    del data["last_quest_advance_time"]
    (manager.state_dir / "world_state_5.json").write_text(json.dumps(data), encoding="utf-8")

    loaded = manager.load_state(world_state_file="world_state_5.json")
    assert loaded.director_interventions == {}
    assert loaded.last_quest_advance_time == 0


def test_load_state_backcompat_without_chronicle(tmp_path):
    """Old snapshots (pre-chronicle) must still load — pydantic defaults fill an empty list."""
    manager = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))
    data = json.loads(manager.init_state().model_dump_json())
    del data["chronicle"]
    (manager.state_dir / "world_state_5.json").write_text(json.dumps(data), encoding="utf-8")

    loaded = manager.load_state(world_state_file="world_state_5.json")
    assert loaded.chronicle == []


def test_chronicle_survives_save_load_round_trip(tmp_path):
    manager = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))
    state = manager.init_state()
    state.time = 3
    state.chronicle = ["The hero once cleared the smuggler's den."]
    manager.save_state(state)

    loaded = manager.load_state(world_state_file="world_state_3.json")
    assert loaded.chronicle == ["The hero once cleared the smuggler's den."]


def test_director_fields_survive_save_load_round_trip(tmp_path):
    manager = StateManager(scenario="smuggler-cove", state_dir=str(tmp_path))
    state = manager.init_state()
    state.time = 3
    state.last_quest_advance_time = 2
    state.director_interventions = {"expose-the-smugglers": 1, "world": 2}
    manager.save_state(state)

    loaded = manager.load_state(world_state_file="world_state_3.json")
    assert loaded.last_quest_advance_time == 2
    assert loaded.director_interventions == {"expose-the-smugglers": 1, "world": 2}
