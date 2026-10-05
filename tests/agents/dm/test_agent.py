from src.agents.dm.agent import ClockAdvance, NewEntity, QuestUpdate, RelationshipUpdate, DMResult, dm_output
from src.agents.dm.tools import Create, Modify


def test_dm_output_entities_filtered():
    entities = [
        NewEntity(type="npc", name="Dockmaster Alan", description="a shady port official", location="docks", role="dockmaster"),
        NewEntity(type="location", name="The Docks", description="busy port district", location="market-square"),
        NewEntity(type="item", name="", description="nameless item — should be filtered"),
        NewEntity(type="quest", name="Find the Vault", description="locate the cold-hearth vault", owner="alice", plan=["search the cellar", "find the vault key", "open the vault"]),
    ]
    result = dm_output(None, entities, [], [], [], [])  # type: ignore[arg-type]
    assert isinstance(result, DMResult)
    assert len(result.creates) == 3  # empty-name filtered
    assert all(isinstance(t, Create) for t in result.creates)
    assert result.creates[0].type == "npc" and result.creates[0].role == "dockmaster"
    assert result.creates[1].type == "location" and result.creates[1].location == "market-square"
    assert result.creates[2].type == "quest" and result.creates[2].owner == "alice"
    assert result.creates[2].plan == ["search the cellar", "find the vault key", "open the vault"]


def test_dm_output_quest_updates_filtered():
    updates = [
        QuestUpdate(quest_id="find-brother", new_status="completed"),
        QuestUpdate(quest_id="find-alan", step="discovered he works at the docks"),
        QuestUpdate(quest_id="find-kaelen", advance=True),
        QuestUpdate(quest_id="noise"),  # empty — should be dropped
    ]
    result = dm_output(None, [], updates, [], [], [])  # type: ignore[arg-type]
    assert len(result.modifies) == 3  # empty update filtered out
    assert all(isinstance(t, Modify) for t in result.modifies)
    assert result.modifies[0].target_id == "find-brother" and result.modifies[0].status == "completed" and result.modifies[0].step is None
    assert result.modifies[1].target_id == "find-alan" and result.modifies[1].status is None and result.modifies[1].step == "discovered he works at the docks"
    assert result.modifies[2].target_id == "find-kaelen" and result.modifies[2].advance is True


def test_dm_output_relationship_updates_filtered():
    updates = [
        RelationshipUpdate(character_id="hero", target_id="merchant", relation="grateful — she healed me"),
        RelationshipUpdate(character_id="hero", target_id="bandit", relation=""),  # empty — dropped
        RelationshipUpdate(character_id="hero", target_id="hero", relation="self-reflective"),  # self — dropped
    ]
    result = dm_output(None, [], [], updates, [], [])  # type: ignore[arg-type]
    assert len(result.modifies) == 1
    modify = result.modifies[0]
    assert isinstance(modify, Modify)
    assert modify.action == "update_relationship"
    assert modify.target_id == "hero"
    assert modify.other_id == "merchant"
    assert modify.reason == "grateful — she healed me"


def test_dm_output_clock_advances_filtered():
    advances = [
        ClockAdvance(faction_id="black-hull-crew", clock_id="retaliation"),
        ClockAdvance(faction_id="", clock_id="retaliation"),  # empty faction — dropped
        ClockAdvance(faction_id="black-hull-crew", clock_id=" "),  # blank clock — dropped
    ]
    result = dm_output(None, [], [], [], advances, [])  # type: ignore[arg-type]
    assert len(result.modifies) == 1
    modify = result.modifies[0]
    assert isinstance(modify, Modify)
    assert modify.action == "advance_faction_clock"
    assert modify.target_id == "black-hull-crew"
    assert modify.other_id == "retaliation"


def test_dm_output_minutes_clamped():
    result = dm_output(None, [], [], [], [], [0, 1, 30, 9999, -5])  # type: ignore[arg-type]
    assert result.minutes == [0, 1, 30, 1440, 0]


def test_dm_output_all_empty():
    result = dm_output(None, [], [], [], [], [])  # type: ignore[arg-type]
    assert result == DMResult(creates=[], modifies=[], minutes=[])
