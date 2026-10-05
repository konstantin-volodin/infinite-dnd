from src.engine.state.models import Character, Faction, Location, ProgressClock, Quest, WorldState
from src.agents.character.context import character_context, character_system


def _state() -> WorldState:
    return WorldState(
        locations={
            "tavern": Location(
                id="tavern",
                description="A canal-side inn with a sagging floor and too many locked doors.",
                connections=[],
            )
        },
        characters={
            "hero": Character(id="hero", location="tavern", relationships={"merchant": "friendly — she patched me up"}),
            "merchant": Character(id="merchant", role="merchant", location="tavern", relationships={"hero": "wary — caught him stealing"}),
            "bystander": Character(id="bystander", location="tavern"),
        },
    )


def test_context_shows_my_own_relationships():
    state = _state()
    ctx = character_context(state.characters["hero"], state)
    assert "merchant: friendly — she patched me up" in ctx


def test_context_shows_how_present_characters_see_me():
    state = _state()
    ctx = character_context(state.characters["hero"], state)
    assert "thinks of me: wary — caught him stealing" in ctx


def test_context_omits_disposition_when_none_set():
    state = _state()
    ctx = character_context(state.characters["bystander"], state)
    assert "thinks of me" not in ctx


def test_context_includes_location_description_as_an_actionable_scene_cue():
    state = _state()

    ctx = character_context(state.characters["hero"], state)

    assert "**tavern:** A canal-side inn with a sagging floor and too many locked doors." in ctx


def test_context_hides_dangling_location_connections_without_mutating_state():
    state = _state()
    state.locations["tavern"].connections = ["alley", "missing-dock"]
    state.locations["alley"] = Location(id="alley")

    ctx = character_context(state.characters["hero"], state)

    assert "**where i can go from here:** alley" in ctx
    assert "missing-dock" not in ctx
    assert state.locations["tavern"].connections == ["alley", "missing-dock"]


def test_context_shows_canonical_two_hop_routes_for_directions():
    state = _state()
    state.locations["tavern"].connections = ["bridge", "guild-hall"]
    state.locations["bridge"] = Location(id="bridge", connections=["tavern", "forest-trail"])
    state.locations["guild-hall"] = Location(id="guild-hall", connections=["tavern", "trade-dock"])
    state.locations["forest-trail"] = Location(id="forest-trail", connections=["bridge"])
    state.locations["trade-dock"] = Location(id="trade-dock", connections=["guild-hall"])

    ctx = character_context(state.characters["hero"], state)

    assert "## 🧭 nearby routes" in ctx
    assert "via **bridge**: forest-trail" in ctx
    assert "via **guild-hall**: trade-dock" in ctx


def test_context_marks_an_adjacent_location_without_onward_exits_as_a_dead_end():
    state = _state()
    state.locations["tavern"].connections = ["alley"]
    state.locations["alley"] = Location(id="alley", connections=["tavern"])

    ctx = character_context(state.characters["hero"], state)

    assert "via **alley**: dead end" in ctx


def test_context_exposes_owned_quest_deadline_pressure():
    state = _state()
    state.quests["save-inn"] = Quest(
        id="save-inn",
        title="Save the Inn",
        description="Keep the inn standing.",
        owner="hero",
    )
    state.factions["flood"] = Faction(
        id="flood",
        name="The Rising Canal",
        goal="Overtop the banks",
        clocks=[ProgressClock(
            id="high-water",
            name="High water",
            consequence="The canal floods the inn.",
            segments=4,
            progress=2,
            fail_quest_id="save-inn",
        )],
    )

    ctx = character_context(state.characters["hero"], state)

    assert "**High water:** 2/4 — The canal floods the inn." in ctx


def test_character_prompt_prioritizes_current_objective_without_goal_churn():
    prompt = character_system(_state().characters["hero"])

    assert "pursue that objective before later plan steps" in prompt
    assert "never use `new_goal` merely to restate" in prompt
    assert "do not repeat the same action on the same target" in prompt
    assert "never use `new_goal` for the current search method" in prompt
