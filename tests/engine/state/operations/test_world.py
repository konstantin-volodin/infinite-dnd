import pytest

from src.engine.state.models import WorldState, Location, Quest, Character
from src.engine.state.operations import WorldOperations


@pytest.fixture
def state() -> WorldState:
    return WorldState(
        locations={
            "tavern": Location(id="tavern", connections=["forest"]),
            "forest": Location(id="forest", connections=["tavern"]),
        },
        characters={
            "hero": Character(id="hero", role="warrior", location="tavern"),
        },
        quests={
            "q1": Quest(id="q1", title="Clear the Cave", description="Defeat the creatures", owner="hero"),
            "q2": Quest(id="q2", title="Ownerless Task", description="no owner", owner=""),
        },
    )


@pytest.fixture
def ops(state) -> WorldOperations:
    return WorldOperations(state)


# ============ QUESTS ============

def test_advance_quest_awards_xp(state, ops):
    msg = ops.advance_quest("q1", advance=True, step="found the entrance")
    assert state.characters["hero"].stats.xp == 10
    assert "+10 XP to hero" in msg
    msg = ops.advance_quest("q1", new_status="completed")
    assert state.quests["q1"].status == "completed"
    assert state.characters["hero"].stats.xp == 60
    assert "+50 XP to hero" in msg


def test_advance_quest_normalizes_completion_status_and_xp(state, ops):
    msg = ops.advance_quest("q1", new_status="Completed")

    assert state.quests["q1"].status == "completed"
    assert state.characters["hero"].stats.xp == 50
    assert "+50 XP to hero" in msg


def test_advance_ownerless_quest_no_xp(state, ops):
    msg = ops.advance_quest("q2", new_status="completed")
    assert state.quests["q2"].status == "completed"
    assert msg == "Quest 'q2' updated."


@pytest.mark.parametrize("terminal_status", ["completed", "failed", "Completed"])
def test_terminal_quest_rejects_stale_updates_without_awarding_xp(state, ops, terminal_status):
    quest = state.quests["q1"]
    quest.status = terminal_status
    quest.current_step = 1
    quest.steps = ["found the entrance"]
    state.characters["hero"].stats.xp = 50
    state.last_quest_advance_time = 3
    state.time = 8

    msg = ops.advance_quest("q1", new_status="completed", step="duplicate update", advance=True)

    assert "already" in msg
    assert quest.status == terminal_status
    assert quest.current_step == 1
    assert quest.steps == ["found the entrance"]
    assert state.characters["hero"].stats.xp == 50
    assert state.last_quest_advance_time == 3


def test_advance_quest_with_plan_increments_and_logs(state, ops):
    ops.add_quest("q5", title="Clear the Ruins", description="", owner="hero", plan=["scout the entrance", "clear the first hall", "defeat the guardian"])
    msg = ops.advance_quest("q5", advance=True)
    quest = state.quests["q5"]
    assert quest.current_step == 1
    assert quest.steps == ["scout the entrance"]
    assert quest.status == "active"
    assert "+10 XP to hero" in msg

    ops.advance_quest("q5", advance=True, step="tough fight")
    assert quest.current_step == 2
    assert quest.steps[-1] == "clear the first hall — tough fight"
    assert quest.status == "active"

    # last objective — advancing past it auto-completes
    msg = ops.advance_quest("q5", advance=True)
    assert quest.current_step == 3
    assert quest.steps[-1] == "defeat the guardian"
    assert quest.status == "completed"
    assert "+50 XP to hero" in msg


def test_planned_quest_rejects_completion_before_final_objective(state, ops):
    ops.add_quest(
        "q5",
        title="Recover the Circlet",
        description="",
        owner="hero",
        plan=["find a clue", "recover the circlet"],
    )
    quest = state.quests["q5"]

    msg = ops.advance_quest("q5", new_status="completed", step="found the steward's key")

    assert "has not achieved its final objective" in msg
    assert quest.status == "active"
    assert quest.current_step == 0
    assert quest.steps == []
    assert state.characters["hero"].stats.xp == 0

    ops.advance_quest("q5", advance=True)
    msg = ops.advance_quest("q5", advance=True)

    assert quest.status == "completed"
    assert quest.steps == ["find a clue", "recover the circlet"]
    assert state.characters["hero"].stats.xp == 70
    assert "+50 XP to hero" in msg


def test_planned_quest_rejects_completion_after_exhausted_active_plan(state, ops):
    ops.add_quest(
        "q5",
        title="Recover the Circlet",
        description="",
        owner="hero",
        plan=["find a clue", "recover the circlet"],
    )
    quest = state.quests["q5"]
    ops.advance_quest("q5", advance=True)
    ops.advance_quest("q5", advance=True, new_status="active")

    msg = ops.advance_quest("q5", new_status="completed", step="a clue points to the circlet")

    assert "has not achieved its final objective" in msg
    assert quest.status == "active"
    assert quest.current_step == len(quest.plan)
    assert state.characters["hero"].stats.xp == 20

    msg = ops.advance_quest("q5", advance=True)

    assert "has no remaining objectives" in msg
    assert quest.status == "active"
    assert quest.current_step == len(quest.plan)
    assert state.characters["hero"].stats.xp == 20


def test_advance_quest_without_plan_no_autocomplete(state, ops):
    # q1 has no plan — advance just logs a generic note, no plan to fall past.
    ops.advance_quest("q1", advance=True)
    quest = state.quests["q1"]
    assert quest.current_step == 1
    assert quest.steps == ["objective accomplished"]
    assert quest.status == "active"


def test_advance_quest_step_without_advance_no_xp(state, ops):
    # explicit step note without `advance` is a plain log entry — no XP, no pointer move.
    msg = ops.advance_quest("q1", step="a rumor surfaces")
    quest = state.quests["q1"]
    assert quest.steps == ["a rumor surfaces"]
    assert quest.current_step == 0
    assert state.characters["hero"].stats.xp == 0
    assert msg == "Quest 'q1' updated."


def test_add_quest(state, ops):
    history_before = len(state.history)
    msg = ops.add_quest("q3", title="Find the Map", description="locate the buried chart", owner="hero")
    assert msg == "Quest 'q3' added."
    assert "q3" in state.quests and state.quests["q3"].owner == "hero"
    assert state.quests["q3"].plan == []
    assert state.history[-1].location == "tavern"
    assert "owner: hero" in state.history[-1].text
    # idempotent
    assert "already exists" in ops.add_quest("q3", title="Find the Map")
    assert len(state.history) == history_before + 1


def test_add_quest_with_plan(state, ops):
    plan = ["ask around town", "search the ruins", "confront the culprit"]
    ops.add_quest("q6", title="Find the Map", description="", owner="hero", plan=plan)
    assert state.quests["q6"].plan == plan
    assert state.quests["q6"].current_step == 0


def test_add_ownerless_quest(state, ops):
    ops.add_quest("q4", title="Mystery", description="someone, somewhere")
    assert state.quests["q4"].owner == ""


def test_advance_quest_stamps_last_advance_time(state, ops):
    state.time = 7
    ops.advance_quest("q1", advance=True)
    assert state.last_quest_advance_time == 7
    # plain step note is not advancement — no stamp
    state.time = 9
    ops.advance_quest("q1", step="a rumor surfaces")
    assert state.last_quest_advance_time == 7


# ============ CHARACTERS ============

def test_spawn_delete_character(state, ops):
    ops.spawn_character("guard", "warrior", "tavern")
    assert "guard" in state.characters
    assert state.characters["guard"].location == "tavern"
    ops.delete_npc("guard", "Left the town.")
    assert "guard" not in state.characters
    assert "Cannot spawn" in ops.spawn_character("guard", "warrior", "nowhere")  # bad location


# ============ ITEMS ============

def test_create_item(state, ops):
    ops.create_item("torch", "forest")
    assert "torch" in state.locations["forest"].items
    assert "already exists" in ops.create_item("torch", "forest")


# ============ LOCATIONS ============

def test_add_modify_location(state, ops):
    ops.add_location("cave", description="damp cave", connections=["forest"])
    assert "cave" in state.locations
    assert "cave" in state.locations["forest"].connections
    ops.modify_location("cave", description="very dark cave", add_feature="stalactites")
    assert state.locations["cave"].description == "very dark cave"
    assert "stalactites" in state.locations["cave"].features


# ============ EVENTS ============

def test_world_event_witnessed_by_present_characters(state, ops):
    ops.spawn_character("guard", "warrior", "tavern")
    msg = ops.world_event("A horn sounds from the gate.", "tavern")
    assert msg == "A horn sounds from the gate."
    event = state.history[-1]
    assert event.location == "tavern"
    assert set(event.characters) == {"hero", "guard"}
    assert event.minutes_elapsed == 0
