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
