import json
import random

import pytest

from src.world import list_scenarios, pick_scenario, read_manifest, scenario_dir


@pytest.fixture
def scenarios() -> list[str]:
    return list_scenarios()


def test_list_scenarios(scenarios):
    assert scenarios, "expected at least one scenario directory"
    assert all(isinstance(s, str) for s in scenarios)


def test_read_manifest(scenarios):
    for s in scenarios:
        manifest = read_manifest(s)
        assert "title" in manifest and "pc" in manifest and "hook" in manifest, f"manifest missing keys for {s}"


def test_pick_scenario_deterministic(scenarios):
    rng = random.Random(0)
    picks = {pick_scenario(rng) for _ in range(20)}
    assert picks.issubset(set(scenarios))

    chosen = pick_scenario(random.Random(42))
    assert chosen in scenarios


def test_scenario_dir(scenarios):
    d = scenario_dir(scenarios[0])
    assert d.is_dir()


def test_scenario_location_connections_resolve_bidirectionally(scenarios):
    for scenario in scenarios:
        locations_path = scenario_dir(scenario) / "locations.json"
        with locations_path.open(encoding="utf-8") as locations_file:
            locations = {
                location["id"]: location
                for location in json.load(locations_file)
            }

        for location_id, location in locations.items():
            for connection_id in location.get("connections", []):
                assert connection_id in locations, (
                    f"{scenario}: '{location_id}' connects to unknown location "
                    f"'{connection_id}'"
                )
                assert location_id in locations[connection_id].get("connections", []), (
                    f"{scenario}: connection '{location_id}' -> '{connection_id}' "
                    "is not bidirectional"
                )


def test_cursed_heirloom_dawn_clock_uses_only_elapsed_time():
    factions_path = scenario_dir("cursed-heirloom") / "factions.json"
    with factions_path.open(encoding="utf-8") as factions_file:
        factions = json.load(factions_file)

    dawn_clock = factions[0]["clocks"][0]
    assert dawn_clock["id"] == "dawn-breaks"
    assert dawn_clock["event_acceleration"] is False
