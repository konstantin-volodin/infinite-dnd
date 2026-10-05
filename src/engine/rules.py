"""
Game Rules - Mechanics derived from character stats.
"""

import random
from dataclasses import dataclass
from typing import Protocol

from src.engine.state.models import Character, CharacterStats


class DieRoller(Protocol):
    """Small injectable random surface used by deterministic rules tests."""

    def randint(self, a: int, b: int, /) -> int: ...


@dataclass(frozen=True)
class CheckResult:
    roll: int
    modifier: int
    total: int
    success: bool
    difficulty: int
    opposing_roll: int | None = None
    opposing_modifier: int = 0
    opposing_total: int | None = None


def d20(rng: DieRoller | None = None) -> int:
    """Roll a d20 through an injectable RNG."""
    return (rng or random).randint(1, 20)


def resolve_check(
    difficulty: int,
    modifier: int = 0,
    *,
    opposing_modifier: int | None = None,
    rng: DieRoller | None = None,
) -> CheckResult:
    """Resolve a DC or contested d20 check; contested ties favor the defender."""
    roll = d20(rng)
    total = roll + modifier
    if opposing_modifier is None:
        return CheckResult(roll, modifier, total, total >= difficulty, difficulty)

    opposing_roll = d20(rng)
    opposing_total = opposing_roll + opposing_modifier
    return CheckResult(
        roll,
        modifier,
        total,
        total > opposing_total,
        difficulty,
        opposing_roll,
        opposing_modifier,
        opposing_total,
    )


def get_health_status(character: object) -> str:
    """Return a health status string based on current hp percentage."""
    if not isinstance(character, Character) or not isinstance(character.stats, CharacterStats):
        return "unknown"

    hp = character.stats.hp
    max_hp = character.stats.max_hp
    pct = (hp / max_hp) * 100 if max_hp else 0

    if hp <= 0: return "dead"
    if pct <= 25: return "critical"
    if pct <= 50: return "injured"
    return "healthy"


def attack_damage(attacker: Character, rng: random.Random | None = None) -> int:
    """Roll attack damage: d4 plus a flat bonus per level above 1."""
    roll = (rng or random).randint(1, 4)
    return roll + attacker.stats.level - 1


def kill_xp(victim: Character) -> int:
    """XP awarded for defeating a character, scaled by their level."""
    return 25 * victim.stats.level
