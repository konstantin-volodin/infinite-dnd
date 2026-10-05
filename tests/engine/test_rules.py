import random

from src.engine.rules import attack_damage, get_health_status, kill_xp, resolve_check
from src.engine.state.models import Character, CharacterStats


def _char(hp: int, max_hp: int, level: int = 1) -> Character:
    return Character(id="test", stats=CharacterStats(hp=hp, max_hp=max_hp, level=level))


def test_get_health_status_thresholds():
    assert get_health_status(_char(0, 20)) == "dead"
    assert get_health_status(_char(5, 20)) == "critical"   # 25%
    assert get_health_status(_char(10, 20)) == "injured"   # 50%
    assert get_health_status(_char(15, 20)) == "healthy"   # 75%
    assert get_health_status(_char(20, 20)) == "healthy"   # 100%
    assert get_health_status(_char(0, 0)) == "dead"        # hp<=0 wins over 0/0 division


def test_get_health_status_fallback():
    assert get_health_status(object()) == "unknown"  # malformed input falls back safely


def test_attack_damage_scales_with_level():
    rng = random.Random(0)
    for level, low, high in [(1, 1, 4), (3, 3, 6)]:
        rolls = {attack_damage(_char(5, 5, level), rng) for _ in range(50)}
        assert min(rolls) == low and max(rolls) == high


def test_kill_xp_scales_with_level():
    assert kill_xp(_char(0, 5, 1)) == 25
    assert kill_xp(_char(0, 5, 3)) == 75


class _Rolls:
    def __init__(self, *values: int):
        self.values = iter(values)

    def randint(self, low: int, high: int) -> int:
        value = next(self.values)
        assert low <= value <= high
        return value


def test_check_uses_injected_d20_and_dc():
    result = resolve_check(15, modifier=3, rng=_Rolls(12))
    assert (result.roll, result.total, result.success) == (12, 15, True)


def test_contested_check_rolls_both_sides_and_ties_favor_defender():
    result = resolve_check(10, modifier=2, opposing_modifier=4, rng=_Rolls(12, 10))
    assert result.difficulty == 10
    assert result.total == result.opposing_total == 14
    assert result.success is False
