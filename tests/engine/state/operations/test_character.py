import pytest

from src.engine.state.models import WorldState, Character, CharacterStats
from src.engine.state.operations.character import CharacterOps


@pytest.fixture
def state(locations) -> WorldState:
    return WorldState(
        locations=locations,
        characters={
            "hero": Character(id="hero", role="warrior", location="tavern", inventory=["sword"], stats=CharacterStats(hp=20, max_hp=20, level=1)),
            "merchant": Character(id="merchant", role="shopkeeper", location="tavern", inventory=["potion", "shield"]),
        },
    )


@pytest.fixture
def ops(state) -> CharacterOps:
    return CharacterOps(state)


# ============ MOVEMENT ============

def test_move_character(state, ops):
    ops.move_character("hero", "forest")
    assert state.characters["hero"].location == "forest"
    ops.move_character("hero", "cave")
    assert state.characters["hero"].location == "cave"
    assert "Cannot reach" in ops.move_character("hero", "tavern")  # cave only connects to forest


# ============ ITEMS ============

def test_take_drop_item(state, ops):
    ops.take_item("hero", "ale")
    assert "ale" in state.characters["hero"].inventory
    assert "ale" not in state.locations["tavern"].items
    ops.drop_item("hero", "ale")
    assert "ale" not in state.characters["hero"].inventory
    assert "ale" in state.locations["tavern"].items


# ============ DIALOGUE ============

def test_speak_rejects_remote_target_without_logging(state, ops):
    ops.move_character("hero", "forest")
    history_before = len(state.history)

    result = ops.speak("hero", "Can you hear me?", target_id="merchant")

    assert "not in the same location" in result
    assert len(state.history) == history_before


# ============ COMBAT ============

class _FixedRng:
    """Stub rng — always rolls the maximum, so damage is deterministic."""
    def randint(self, a: int, b: int) -> int:
        return b


def test_attack_damages_target(state, ops):
    result = ops.attack("hero", "merchant", rng=_FixedRng())
    assert "hero attacks merchant for 4 damage" in result  # d4 max roll, level 1
    assert state.characters["merchant"].stats.hp == 1


def test_attack_kill_awards_xp(state, ops):
    ops.damage("merchant", 4)  # 1 HP left — next max roll kills
    result = ops.attack("hero", "merchant", rng=_FixedRng())
    assert state.characters["merchant"].stats.hp == 0
    assert "falls dead" in state.history[-2].text
    assert "defeating merchant" in result
    assert state.characters["hero"].stats.xp == 25


def test_attack_validation(state, ops):
    assert "not found" in ops.attack("hero", "ghost")
    assert "can't attack themselves" in ops.attack("hero", "hero")
    ops.move_character("hero", "forest")
    assert "not in the same location" in ops.attack("hero", "merchant")
    ops.move_character("hero", "tavern")
    ops.damage("merchant", 100)
    assert "already dead" in ops.attack("hero", "merchant")
    ops.damage("hero", 100)
    state.characters["merchant"].stats.hp = 1
    assert "is dead" in ops.attack("hero", "merchant")


# ============ STATS ============

def test_damage_heal(state, ops):
    ops.damage("hero", 8)
    assert state.characters["hero"].stats.hp == 12
    ops.heal("hero", 3)
    assert state.characters["hero"].stats.hp == 15


def test_damage_and_heal_report_only_applied_hp_change(state, ops):
    damage_result = ops.damage("hero", 8)
    heal_result = ops.heal("hero", 100)

    assert "takes 8 damage" in damage_result
    assert "heals 8 HP" in heal_result
    assert state.history[-2].text == damage_result
    assert state.history[-1].text == heal_result


def test_heal_cannot_revive_a_dead_character(state, ops):
    ops.damage("merchant", 4)
    ops.attack("hero", "merchant", rng=_FixedRng())
    history_before = list(state.history)

    result = ops.heal("merchant", 3)

    assert result == "Cannot heal — 'merchant' is dead."
    assert state.characters["merchant"].stats.hp == 0
    assert state.history == history_before


def test_zero_effect_damage_and_heal_do_not_mutate_history(state, ops):
    state.characters["merchant"].stats.hp = 0

    damage_result = ops.damage("merchant", 5)
    heal_result = ops.heal("hero", 5)

    assert "takes 0 damage" in damage_result
    assert "heals 0 HP" in heal_result
    assert state.history == []


def test_damage_and_heal_reject_negative_amounts_without_mutating_state(state, ops):
    hp_before = state.characters["hero"].stats.hp
    history_before = list(state.history)

    damage_result = ops.damage("hero", -5)
    heal_result = ops.heal("hero", -5)

    assert "amount cannot be negative" in damage_result
    assert "amount cannot be negative" in heal_result
    assert state.characters["hero"].stats.hp == hp_before
    assert state.history == history_before


def test_level_up(state, ops):
    ops.level_up("hero")
    assert state.characters["hero"].stats.level == 2
    assert state.characters["hero"].stats.max_hp == 25
    assert state.characters["hero"].stats.hp == 25


def test_level_up_does_not_revive_a_dead_character(state, ops):
    char = state.characters["hero"]
    char.stats.hp = 0

    ops.level_up("hero")

    assert char.stats.level == 2
    assert char.stats.max_hp == 25
    assert char.stats.hp == 0


def test_level_up_rejects_negative_hp_increase_without_mutating_state(state, ops):
    char = state.characters["hero"]
    stats_before = char.stats.model_copy(deep=True)
    history_before = list(state.history)

    result = ops.level_up("hero", hp_increase=-10)

    assert "HP increase cannot be negative" in result
    assert char.stats == stats_before
    assert state.history == history_before


def test_award_xp(state, ops):
    history_before = len(state.history)
    ops.award_xp("hero", 25, reason="slaying goblins")
    assert state.characters["hero"].stats.xp == 25
    assert state.history[-1].text == "hero earns 25 XP (slaying goblins)."
    assert len(state.history) == history_before + 1
    assert "Cannot award XP" in ops.award_xp("ghost", 10)


def test_award_xp_rejects_negative_amount_without_mutating_state(state, ops):
    char = state.characters["hero"]
    stats_before = char.stats.model_copy(deep=True)
    history_before = list(state.history)

    result = ops.award_xp("hero", -10, reason="invalid adjustment")

    assert "amount cannot be negative" in result
    assert char.stats == stats_before
    assert state.history == history_before


def test_award_xp_auto_levels(state, ops):
    ops.level_up("hero")  # level 2, threshold becomes 200
    msg = ops.award_xp("hero", 200, reason="defeating the ogre")
    assert state.characters["hero"].stats.level == 3
    assert state.characters["hero"].stats.max_hp == 30
    assert "leveled up to level 3" in msg


def test_award_xp_auto_level_does_not_revive_a_dead_character(state, ops):
    char = state.characters["hero"]
    char.stats.hp = 0
    char.stats.xp = 90

    msg = ops.award_xp("hero", 10, reason="quest progress")

    assert char.stats.level == 2
    assert char.stats.max_hp == 25
    assert char.stats.hp == 0
    assert "leveled up to level 2" in msg


def test_give_gold(state, ops):
    state.characters["hero"].stats.gold = 10
    ops.give_gold("hero", "merchant", 4)
    assert state.characters["hero"].stats.gold == 6
    assert state.characters["merchant"].stats.gold == 4
    assert "Cannot give gold" in ops.give_gold("hero", "merchant", 100)  # not enough gold
    assert state.characters["hero"].stats.gold == 6  # unchanged
    assert "Cannot give gold" in ops.give_gold("ghost", "merchant", 1)  # unknown giver
    assert "Cannot give gold" in ops.give_gold("hero", "ghost", 1)  # unknown receiver


def test_give_gold_rejects_remote_receiver_without_mutating_state(state, ops):
    state.characters["hero"].stats.gold = 10
    ops.move_character("hero", "forest")
    history_before = len(state.history)

    result = ops.give_gold("hero", "merchant", 4)

    assert "not in the same location" in result
    assert state.characters["hero"].stats.gold == 10
    assert state.characters["merchant"].stats.gold == 0
    assert len(state.history) == history_before


def test_transactions_reject_same_character_without_mutating_state(state, ops):
    hero = state.characters["hero"]
    hero.stats.gold = 10
    inventory_before = list(hero.inventory)
    history_before = len(state.history)

    gold_result = ops.give_gold("hero", "hero", 4)
    trade_result = ops.trade_item("hero", "hero", "sword", 4)

    assert "different characters" in gold_result
    assert "different characters" in trade_result
    assert hero.stats.gold == 10
    assert hero.inventory == inventory_before
    assert len(state.history) == history_before


def test_transactions_reject_negative_values_without_mutating_state(state, ops):
    hero = state.characters["hero"]
    merchant = state.characters["merchant"]
    hero.stats.gold = 10
    inventories_before = (list(hero.inventory), list(merchant.inventory))
    history_before = len(state.history)

    gold_result = ops.give_gold("hero", "merchant", -4)
    trade_result = ops.trade_item("hero", "merchant", "potion", -4)

    assert "cannot be negative" in gold_result
    assert "cannot be negative" in trade_result
    assert (hero.stats.gold, merchant.stats.gold) == (10, 0)
    assert (hero.inventory, merchant.inventory) == inventories_before
    assert len(state.history) == history_before


# ============ TRADE ============

def test_trade_item(state, ops):
    state.characters["hero"].stats.gold = 10
    ops.trade_item("hero", "merchant", "potion", 4)
    assert "potion" in state.characters["hero"].inventory
    assert "potion" not in state.characters["merchant"].inventory
    assert state.characters["hero"].stats.gold == 6
    assert state.characters["merchant"].stats.gold == 4

    # reverse direction — merchant buys back from hero
    ops.trade_item("merchant", "hero", "potion", 4)
    assert "potion" in state.characters["merchant"].inventory
    assert "potion" not in state.characters["hero"].inventory
    assert state.characters["hero"].stats.gold == 10
    assert state.characters["merchant"].stats.gold == 0

    assert "Cannot trade" in ops.trade_item("hero", "merchant", "wand", 1)  # seller doesn't have item
    assert "Cannot trade" in ops.trade_item("hero", "merchant", "shield", 100)  # buyer can't afford

    ops.move_character("hero", "forest")
    assert "Cannot trade" in ops.trade_item("hero", "merchant", "shield", 1)  # different locations


# ============ RELATIONSHIPS ============

def test_relationships(state, ops):
    ops.update_relationship("hero", "merchant", "friendly")
    assert state.characters["hero"].relationships["merchant"] == "friendly"


def test_relationships_normalize_character_ids(state, ops):
    result = ops.update_relationship("HERO", "Merchant", "friendly")

    assert "hero's relationship with merchant" in result
    assert state.characters["hero"].relationships == {"merchant": "friendly"}


def test_relationships_reject_slug_equivalent_self_target_without_mutating(state, ops):
    hero = state.characters["hero"]
    hero.relationships = {"merchant": "friendly"}

    result = ops.update_relationship("HERO", "hero", "conflicted")

    assert "must be different characters" in result
    assert hero.relationships == {"merchant": "friendly"}


# ============ KNOWLEDGE ============

def test_knowledge(state, ops):
    history_before = len(state.history)
    ops.add_knowledge("hero", "The cave has a hidden passage")
    assert "The cave has a hidden passage" in state.characters["hero"].knowledge
    assert state.history[-1].text == "hero learns: The cave has a hidden passage"
    assert len(state.history) == history_before + 1
    ops.add_knowledge("hero", "The cave has a hidden passage")  # idempotent — no new event
    assert state.characters["hero"].knowledge.count("The cave has a hidden passage") == 1
    assert len(state.history) == history_before + 1


def test_knowledge_rejects_blank_fact_without_mutating(state, ops):
    history_before = list(state.history)

    result = ops.add_knowledge("hero", "  \t\n  ")

    assert result == "Cannot add knowledge — fact cannot be blank."
    assert state.characters["hero"].knowledge == []
    assert state.history == history_before
