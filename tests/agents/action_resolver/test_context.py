from src.agents.action_resolver.context import action_resolver_context, action_resolver_system
from src.engine.state.models import Character, Location, Quest, WorldState


def test_action_context_exposes_scene_and_current_quest_objective():
    state = WorldState(
        locations={
            "archive": Location(
                id="archive",
                description="A flooded records room where chains hold the shelves upright.",
                features=["a bell rope disappearing into the ceiling"],
            )
        },
        characters={"hero": Character(id="hero", location="archive")},
        quests={
            "missing-ledger": Quest(
                id="missing-ledger",
                title="The Missing Ledger",
                description="Recover the harbor accounts.",
                owner="hero",
                plan=["find who moved the ledger", "recover it from the customs vault"],
                current_step=1,
            )
        },
    )

    context = action_resolver_context(
        state.characters["hero"],
        state,
        description="pull the bell rope",
        target="bell rope",
    )

    assert "A flooded records room where chains hold the shelves upright." in context
    assert "a bell rope disappearing into the ceiling" in context
    assert "current objective: recover it from the customs vault" in context


def test_action_resolver_prompt_requires_meaningful_but_non_coercive_outcomes():
    prompt = action_resolver_system()

    assert "produce concrete progress, expose a concrete obstacle, or open a specific new approach" in prompt
    assert "never choose the character's next action" in prompt
    assert "do not invent a consolation prize" in prompt
