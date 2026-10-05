from src.agents.dm.agent import NewEntity
from src.agents.dm.director import DirectorResult, director_output
from src.agents.dm.tools import Create


def test_director_output_event_only():
    result = director_output(None, event=" A bailiff arrives demanding the debt. ")  # type: ignore[arg-type]
    assert result == DirectorResult(event="A bailiff arrives demanding the debt.", create=None, quest_id=None)


def test_director_output_with_entity_and_quest():
    entity = NewEntity(type="npc", name="Bailiff Coss", description="a humourless debt collector", location="tavern", role="bailiff")
    result = director_output(None, event="Bailiff Coss arrives demanding the debt.", entity=entity, quest_id="q1")  # type: ignore[arg-type]
    assert isinstance(result.create, Create)
    assert result.create.type == "npc" and result.create.name == "Bailiff Coss" and result.create.location == "tavern"
    assert result.quest_id == "q1"


def test_director_output_nameless_entity_filtered():
    entity = NewEntity(type="item", name="   ")
    result = director_output(None, event="Something shifts in the dark.", entity=entity)  # type: ignore[arg-type]
    assert result.create is None
