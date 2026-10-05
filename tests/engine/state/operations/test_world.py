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


def test_advance_quest_rejects_unsupported_status_without_mutating(state, ops):
    state.quests["q1"].plan = ["find the entrance"]
    state.time = 4

    msg = ops.advance_quest("q1", new_status="done", step="found it", advance=True)

    quest = state.quests["q1"]
    assert msg == "Cannot advance quest — unsupported status 'done'."
    assert quest.status == "active"
    assert quest.current_step == 0
    assert quest.steps == []
    assert state.characters["hero"].stats.xp == 0
    assert state.last_quest_advance_time == 0


def test_advance_quest_rejects_simultaneous_progress_and_failure(state, ops):
    quest = state.quests["q1"]
    quest.plan = ["defeat the cave guardian"]
    state.time = 4

    msg = ops.advance_quest(
        "q1",
        new_status="failed",
        step="the guardian escaped",
        advance=True,
    )

    assert msg == "Cannot advance quest — 'q1' cannot advance and fail in the same update."
    assert quest.status == "active"
    assert quest.current_step == 0
    assert quest.steps == []
    assert state.characters["hero"].stats.xp == 0
    assert state.last_quest_advance_time == 0


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
    assert state.characters["hero"].stats.xp == 80
    assert "+60 XP to hero" in msg


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
    assert "+60 XP to hero" in msg


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
    ops.advance_quest("q5", advance=True)
    quest.status = "active"  # emulate a stale snapshot whose final pointer was already exhausted

    msg = ops.advance_quest("q5", new_status="completed", step="a clue points to the circlet")

    assert "has not achieved its final objective" in msg
    assert quest.status == "active"
    assert quest.current_step == len(quest.plan)
    assert state.characters["hero"].stats.xp == 70

    msg = ops.advance_quest("q5", advance=True)

    assert "has no remaining objectives" in msg
    assert quest.status == "active"
    assert quest.current_step == len(quest.plan)
    assert state.characters["hero"].stats.xp == 70


def test_final_objective_overrides_redundant_active_status(state, ops):
    ops.add_quest(
        "q5",
        title="Recover the Circlet",
        description="",
        owner="hero",
        plan=["find a clue", "recover the circlet"],
    )
    ops.advance_quest("q5", advance=True)

    msg = ops.advance_quest("q5", advance=True, new_status="active")

    quest = state.quests["q5"]
    assert quest.status == "completed"
    assert quest.current_step == len(quest.plan)
    assert state.characters["hero"].stats.xp == 70
    assert "+60 XP to hero" in msg


def test_advance_quest_normalizes_quest_id(state, ops):
    state.quests["missing-brother"] = Quest(
        id="missing-brother",
        title="The Missing Brother",
        owner="hero",
        plan=["find a lead", "find Kaelen"],
    )

    msg = ops.advance_quest("MISSING BROTHER", advance=True)

    quest = state.quests["missing-brother"]
    assert quest.current_step == 1
    assert quest.steps == ["find a lead"]
    assert state.characters["hero"].stats.xp == 10
    assert msg == "Quest 'missing-brother' updated. (+10 XP to hero)"


def test_advance_quest_resolves_unique_title(state, ops):
    state.quests["missing-brother"] = Quest(
        id="missing-brother",
        title="The Missing Brother",
        owner="hero",
        plan=["find a lead", "find Kaelen"],
    )

    msg = ops.advance_quest("THE MISSING BROTHER", advance=True)

    quest = state.quests["missing-brother"]
    assert quest.current_step == 1
    assert quest.steps == ["find a lead"]
    assert state.characters["hero"].stats.xp == 10
    assert msg == "Quest 'missing-brother' updated. (+10 XP to hero)"


def test_advance_quest_rejects_ambiguous_title_without_mutating(state, ops):
    state.quests["first-cave"] = Quest(id="first-cave", title="Clear the Cave")

    msg = ops.advance_quest("Clear the Cave", advance=True)

    assert msg == "Cannot advance quest — 'Clear the Cave' not found."
    assert state.quests["q1"].current_step == 0
    assert state.quests["first-cave"].current_step == 0
    assert state.characters["hero"].stats.xp == 0


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


def test_add_quest_normalizes_known_owner_id(state, ops):
    ops.add_quest("q3", title="Find the Map", owner="HERO")

    assert state.quests["q3"].owner == "hero"


def test_add_quest_rejects_unknown_owner_without_mutating(state, ops):
    history_before = list(state.history)

    msg = ops.add_quest("q3", title="Find the Map", owner="missing-hero")

    assert msg == "Cannot add quest — owner 'missing-hero' not found."
    assert "q3" not in state.quests
    assert state.history == history_before


def test_add_quest_rejects_overlapping_active_quest_for_same_owner(state, ops):
    state.quests["q1"] = Quest(
        id="q1",
        title="The Missing Brother",
        description=(
            "Find clues about the disappearance of Kaelen Swift, who vanished near the river."
        ),
        owner="hero",
        plan=["track down whoever saw him last", "find Kaelen Swift"],
    )
    history_before = list(state.history)

    result = ops.add_quest(
        "alan-connection",
        title="Alan Connection",
        description=(
            "Investigate Alan the dockmaster's suspicious behavior following Kaelen's disappearance."
        ),
        owner="hero",
        plan=[
            "ask dockmaster alan about kaelen's visit",
            "determine alan's involvement in kaelen's disappearance",
        ],
    )

    assert result == "Cannot add quest — overlaps active quest 'q1' for owner 'hero'."
    assert "alan-connection" not in state.quests
    assert state.history == history_before


def test_add_quest_allows_same_topic_after_existing_quest_resolves(state, ops):
    state.quests["q1"].status = "completed"

    result = ops.add_quest(
        "cave-aftermath",
        title="Cave Aftermath",
        description="Investigate the creatures that escaped from the cave.",
        owner="hero",
    )

    assert result == "Quest 'cave-aftermath' added."
    assert "cave-aftermath" in state.quests


def test_add_quest_with_plan(state, ops):
    plan = ["ask around town", "search the ruins", "confront the culprit"]
    ops.add_quest("q6", title="Find the Map", description="", owner="hero", plan=plan)
    assert state.quests["q6"].plan == plan
    assert state.quests["q6"].current_step == 0


def test_add_quest_rejects_blank_plan_objective_without_mutating(state, ops):
    history_before = list(state.history)

    result = ops.add_quest(
        "q6",
        title="Find the Map",
        owner="hero",
        plan=["ask around town", "   ", "recover the map"],
    )

    assert result == "Cannot add quest — plan objectives cannot be blank."
    assert "q6" not in state.quests
    assert state.history == history_before


def test_add_ownerless_quest(state, ops):
    ops.add_quest("q4", title="Mystery", description="someone, somewhere")
    assert state.quests["q4"].owner == ""


def test_entity_creation_rejects_ids_without_letters_or_numbers(state, ops):
    history_before = list(state.history)

    assert "quest ID must contain" in ops.add_quest("!!!", title="Invalid")
    assert "character ID must contain" in ops.spawn_character("!!!", "stranger", "tavern")
    assert "location ID must contain" in ops.add_location("!!!")

    assert "!!!" not in state.quests
    assert "!!!" not in state.characters
    assert "!!!" not in state.locations
    assert state.history == history_before


def test_entity_creation_rejects_slug_equivalent_ids_without_mutating(state, ops):
    state.quests["side-quest"] = Quest(id="side-quest", title="Existing Quest")
    history_before = list(state.history)
    forest_connections_before = list(state.locations["forest"].connections)

    assert "already exists" in ops.add_quest("SIDE_QUEST", title="Duplicate Quest")
    assert "already exists" in ops.spawn_character("HERO", "impostor", "forest")
    assert "already exists" in ops.add_location("TAVERN", connections=["forest"])

    assert set(state.quests) == {"q1", "q2", "side-quest"}
    assert set(state.characters) == {"hero"}
    assert set(state.locations) == {"tavern", "forest"}
    assert state.locations["forest"].connections == forest_connections_before
    assert state.history == history_before


def test_advance_quest_stamps_last_advance_time(state, ops):
    state.time = 7
    ops.advance_quest("q1", advance=True)
    assert state.last_quest_advance_time == 7
    # plain step note is not advancement — no stamp
    state.time = 9
    ops.advance_quest("q1", step="a rumor surfaces")
    assert state.last_quest_advance_time == 7


def test_redundant_active_status_does_not_reset_quest_stall_time(state, ops):
    state.time = 9
    state.last_quest_advance_time = 3

    msg = ops.advance_quest("q1", new_status="ACTIVE")

    assert msg == "Quest 'q1' updated."
    assert state.quests["q1"].status == "active"
    assert state.last_quest_advance_time == 3


# ============ CHARACTERS ============

def test_spawn_delete_character(state, ops):
    ops.spawn_character("guard", "warrior", "tavern")
    assert "guard" in state.characters
    assert state.characters["guard"].location == "tavern"
    ops.delete_npc("guard", "Left the town.")
    assert "guard" not in state.characters
    assert "Cannot spawn" in ops.spawn_character("guard", "warrior", "nowhere")  # bad location


def test_spawn_character_normalizes_known_location_id(state, ops):
    result = ops.spawn_character("guard", "warrior", "TAVERN")

    assert result == "guard appears at 'tavern'."
    assert state.characters["guard"].location == "tavern"
    assert state.history[-1].location == "tavern"


def test_spawn_character_canonicalizes_deferred_relationship_alias(state, ops):
    state.characters["hero"].relationships = {
        "alan-dockmaster": "contact",
        "alan-smith": "rival",
    }

    result = ops.reveal_character("dockmaster-alan", "dockmaster", "forest")

    assert result == "dockmaster-alan appears at 'forest'."
    assert state.characters["hero"].relationships == {
        "dockmaster-alan": "contact",
        "alan-smith": "rival",
    }


def test_spawn_character_canonicalizes_relationship_alias_with_article(state, ops):
    state.characters["hero"].relationships = {"alan-dockmaster": "contact"}

    result = ops.reveal_character("alan-the-dockmaster", "dockmaster", "forest")

    assert result == "alan-the-dockmaster appears at 'forest'."
    assert state.characters["hero"].relationships == {
        "alan-the-dockmaster": "contact",
    }


def test_spawn_character_canonicalizes_name_title_alias_using_role(state, ops):
    state.characters["hero"].relationships = {"alan-dockmaster": "contact"}

    result = ops.reveal_character("alan-marsh", "dockmaster", "forest")

    assert result == "alan-marsh appears at 'forest'."
    assert state.characters["hero"].relationships == {
        "alan-marsh": "contact",
    }


def test_reveal_character_preserves_unique_deferred_relationship_identity(state, ops):
    state.characters["hero"].relationships = {"alan-dockmaster": "contact"}

    result = ops.reveal_character("alan", "dock-merchant", "forest")

    assert result == "alan-dockmaster appears at 'forest'."
    assert "alan" not in state.characters
    assert state.characters["alan-dockmaster"].role == "dock-merchant"
    assert state.characters["hero"].relationships == {
        "alan-dockmaster": "contact",
    }


def test_reveal_character_keeps_short_name_when_deferred_alias_is_ambiguous(state, ops):
    state.characters["hero"].relationships = {
        "alan-dockmaster": "contact",
        "alan-smith": "rival",
    }

    result = ops.reveal_character("alan", "dock-merchant", "forest")

    assert result == "alan appears at 'forest'."
    assert state.characters["hero"].relationships == {
        "alan-dockmaster": "contact",
        "alan-smith": "rival",
    }


def test_spawn_character_keeps_ambiguous_name_title_alias_deferred(state, ops):
    state.characters["hero"].relationships = {"alan-dockmaster": "contact"}
    state.characters["alan-smith"] = Character(
        id="alan-smith",
        role="dockmaster",
        location="tavern",
    )

    result = ops.reveal_character("alan-marsh", "dockmaster", "forest")

    assert result == "alan-marsh appears at 'forest'."
    assert state.characters["hero"].relationships == {
        "alan-dockmaster": "contact",
    }


def test_delete_npc_rejects_active_quest_owner_without_mutating(state, ops):
    ops.spawn_character("guard", "warrior", "tavern")
    ops.add_quest("guard-duty", title="Guard the Gate", owner="guard")
    history_before = list(state.history)

    result = ops.delete_npc("guard", "Leaves town.")

    assert result == "Cannot delete 'guard' — character owns active quest 'guard-duty'."
    assert "guard" in state.characters
    assert state.quests["guard-duty"].owner == "guard"
    assert state.history == history_before

    state.quests["guard-duty"].status = "completed"
    assert "is gone" in ops.delete_npc("guard", "Leaves town.")
    assert "guard" not in state.characters


def test_delete_npc_removes_dangling_relationships(state, ops):
    ops.spawn_character("guard", "warrior", "tavern")
    ops.spawn_character("merchant", "shopkeeper", "tavern")
    state.characters["hero"].relationships = {
        "guard": "trusted ally",
        "merchant": "friendly",
    }
    state.characters["merchant"].relationships["guard"] = "customer"

    result = ops.delete_npc("guard", "Leaves town.")

    assert "is gone" in result
    assert "guard" not in state.characters
    assert state.characters["hero"].relationships == {"merchant": "friendly"}
    assert state.characters["merchant"].relationships == {}


# ============ ITEMS ============

def test_create_item(state, ops):
    ops.create_item("torch", "forest")
    assert "torch" in state.locations["forest"].items
    assert "already exists" in ops.create_item("torch", "forest")


def test_create_item_rejects_name_without_letters_or_numbers(state, ops):
    history_before = list(state.history)

    result = ops.create_item(" - ", "forest")

    assert result == "Cannot create item — item name must contain a letter or number."
    assert state.locations["forest"].items == []
    assert state.history == history_before


def test_create_item_rejects_existing_item_elsewhere_in_world(state, ops):
    state.locations["tavern"].items = ["sealed letter with black wax"]
    state.characters["hero"].inventory = ["weather-worn logbook"]
    history_before = list(state.history)

    location_result = ops.create_item("Sealed Letter With Black Wax", "forest")
    inventory_result = ops.create_item("weather worn logbook", "forest")

    assert location_result == (
        "Cannot create item — 'Sealed Letter With Black Wax' already exists at 'tavern'."
    )
    assert inventory_result == (
        "Cannot create item — 'weather worn logbook' already belongs to 'hero'."
    )
    assert state.locations["forest"].items == []
    assert state.history == history_before


def test_rename_item_records_held_item_state_and_prevents_repeating_transition(state, ops):
    state.characters["hero"].inventory = ["sealed wax letter"]

    result = ops.rename_item("hero", "sealed wax letter", "opened wax letter")
    repeated = ops.rename_item("hero", "sealed wax letter", "opened wax letter")

    assert result == "hero changes 'sealed wax letter' into 'opened wax letter'."
    assert repeated == "Cannot change 'sealed wax letter' — hero cannot access it."
    assert state.characters["hero"].inventory == ["opened wax letter"]
    assert [event.text for event in state.history] == [result]


# ============ LOCATIONS ============

def test_add_modify_location(state, ops):
    ops.add_location("cave", description="damp cave", connections=["forest"])
    assert "cave" in state.locations
    assert "cave" in state.locations["forest"].connections
    ops.modify_location("cave", description="very dark cave", add_feature="stalactites")
    assert state.locations["cave"].description == "very dark cave"
    assert "stalactites" in state.locations["cave"].features


def test_modify_location_reports_when_requested_changes_are_noops(state, ops):
    location = state.locations["tavern"]
    location.description = "A warm common room."
    location.features = ["stone hearth"]
    before = location.model_dump()

    result = ops.modify_location(
        "tavern",
        description="A warm common room.",
        add_feature="stone hearth",
        remove_feature="missing trapdoor",
    )

    assert result == "Location 'tavern' unchanged."
    assert location.model_dump() == before


@pytest.mark.parametrize("unknown_connection", ["missing-passage", ""])
def test_add_location_rejects_unknown_connection_without_mutating(
    state,
    ops,
    unknown_connection,
):
    locations_before = state.model_dump()["locations"]
    history_before = list(state.history)

    result = ops.add_location(
        "hidden-room",
        description="A room behind the hearth.",
        connections=["tavern", unknown_connection],
    )

    assert result == (
        "Cannot add location 'hidden-room' — connected location "
        f"'{unknown_connection}' not found."
    )
    assert state.model_dump()["locations"] == locations_before
    assert state.history == history_before


# ============ EVENTS ============

def test_world_event_witnessed_by_present_characters(state, ops):
    ops.spawn_character("guard", "warrior", "tavern")
    msg = ops.world_event("A horn sounds from the gate.", "tavern")
    assert msg == "A horn sounds from the gate."
    event = state.history[-1]
    assert event.location == "tavern"
    assert set(event.characters) == {"hero", "guard"}
    assert event.minutes_elapsed == 0
