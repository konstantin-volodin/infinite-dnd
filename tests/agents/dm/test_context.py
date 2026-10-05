from src.agents.dm.context import director_context, director_system
from src.agents.dm.director import DirectorDeps
from src.engine.state.models import Character, CharacterStats, Location, WorldState


def test_director_context_includes_scene_affordances_and_motivations():
    state = WorldState(
        locations={
            "tavern": Location(
                id="tavern",
                description="A crowded inn balanced above the canal.",
                features=["a cracked support beam", "a locked cellar hatch"],
                items=["sealed black letter"],
                connections=["canal-walk"],
            ),
            "canal-walk": Location(id="canal-walk"),
        },
        characters={
            "hero": Character(
                id="hero",
                role="smuggler",
                goal="keep the innkeeper safe",
                location="tavern",
                stats=CharacterStats(hp=3, max_hp=5),
            ),
            "remote": Character(id="remote", goal="burn the letter", location="canal-walk"),
        },
    )

    context = director_context(DirectorDeps(state=state, location_id="tavern"))

    assert "A crowded inn balanced above the canal." in context
    assert "a cracked support beam, a locked cellar hatch" in context
    assert "sealed black letter" in context
    assert "canal-walk" in context
    assert "**hero** (smuggler): 3/5 HP; wants keep the innkeeper safe" in context
    assert "remote" not in context


def test_director_prompt_preserves_player_agency():
    prompt = director_system()

    assert "at least two plausible responses" in prompt
    assert "never dictate what a character chooses" in prompt
