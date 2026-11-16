import pytest
from pydantic_ai import ModelRetry

from src.agents.dm.storyteller import (
    ARC_DEADLINE_HOURS,
    Arc,
    Person,
    Place,
    StorytellerDeps,
    apply_arc,
    report,
    storyteller_context,
)
from src.engine.state.models import Character, Location, Quest, WorldState


def _state() -> WorldState:
    return WorldState(
        protagonist="hero",
        locations={
            "tavern": Location(id="tavern", connections=["road"]),
            "road": Location(id="road", connections=["tavern"]),
        },
        characters={
            "hero": Character(
                id="hero", location="tavern", relationships={"old-bram": "mentor"}
            ),
            "old-bram": Character(id="old-bram", location="road", goal="retire"),
        },
        quests={
            "q1": Quest(
                id="q1",
                title="Clear the Cave",
                owner="hero",
                status="failed",
                plan=["scout", "clear"],
                steps=["scout", "abandoned, never managed to clear"],
            )
        },
    )


def _arc(**overrides) -> Arc:
    fields = dict(
        hook="A rider brings word of drowned bells at Saltmere.",
        places=[
            Place(
                name="Saltmere Docks", description="Tar and gulls.", connects_to="road"
            ),
            Place(
                name="Bell Tower", description="Leaning.", connects_to="Saltmere Docks"
            ),
        ],
        people=[
            Person(
                name="Harbourmaster Ivo",
                role="harbourmaster",
                goal="hide the debts",
                location="Saltmere Docks",
            )
        ],
        quest_title="The Drowned Bells",
        quest_description="Learn who silenced the bells of Saltmere.",
        plan=[
            "question Harbourmaster Ivo at Saltmere Docks",
            "climb the Bell Tower",
            "confront the bell thief",
        ],
        faction_name="The Tide Court",
        faction_goal="Flood the lower town",
        deadline_consequence="The lower town floods.",
    )
    return Arc(**{**fields, **overrides})


def test_apply_arc_opens_a_connected_region_with_people_quest_and_deadline():
    state = _state()

    assert apply_arc(state, _arc()) == []

    assert state.locations["saltmere-docks"].connections == ["road", "bell-tower"]
    assert "saltmere-docks" in state.locations["road"].connections
    assert state.locations["bell-tower"].connections == ["saltmere-docks"]
    assert state.characters["harbourmaster-ivo"].location == "saltmere-docks"
    quest = state.quests["the-drowned-bells"]
    assert (quest.owner, quest.status, len(quest.plan)) == ("hero", "active", 3)
    clock = state.factions["the-tide-court"].clocks[0]
    assert (clock.segments, clock.fail_quest_id) == (
        ARC_DEADLINE_HOURS,
        "the-drowned-bells",
    )
    assert state.history[0].text == "A rider brings word of drowned bells at Saltmere."
    assert state.characters["hero"].goal == quest.description
    WorldState.model_validate(state.model_dump())


def test_apply_arc_joins_an_unknown_gateway_to_the_heros_location():
    state = _state()
    places = [
        Place(name="Saltmere Docks", connects_to="atlantis"),
        Place(name="Bell Tower", connects_to="nowhere"),
    ]

    assert apply_arc(state, _arc(places=places)) == []

    assert state.locations["saltmere-docks"].connections == ["tavern", "bell-tower"]
    assert state.locations["bell-tower"].connections == ["saltmere-docks"]


def test_apply_arc_moves_a_known_person_into_the_region():
    state = _state()
    people = [Person(name="Old Bram", goal="pay a debt", location="Bell Tower")]

    assert apply_arc(state, _arc(people=people)) == []

    assert state.characters["old-bram"].location == "bell-tower"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"plan": ["a", "b"]}, "plan needs 3 to 5 objectives"),
        ({"places": [Place(name="Saltmere Docks")]}, "places needs 2 to 4"),
        ({"people": []}, "people needs at least one person"),
        (
            {"plan": ["ask around", "look around", "fight"]},
            "name one of the new places",
        ),
        (
            {"places": [Place(name="Road"), Place(name="Bell Tower")]},
            "already exists",
        ),
    ],
)
def test_apply_arc_reports_structural_errors(overrides, error):
    errors = apply_arc(_state(), _arc(**overrides))

    assert len(errors) == 1 and error in errors[0]


def test_report_dry_runs_the_arc_and_asks_for_a_retry_on_errors():
    state = _state()
    ctx = type("Ctx", (), {"deps": StorytellerDeps(state=state)})()
    arc = _arc()

    assert report(ctx, **arc.model_dump()) == arc  # type: ignore[arg-type]
    assert "saltmere-docks" not in state.locations  # dry run only

    with pytest.raises(ModelRetry, match="plan needs"):
        report(ctx, **_arc(plan=["x"]).model_dump())  # type: ignore[arg-type]


def test_storyteller_context_builds_on_the_ended_quest_and_known_people():
    state = _state()
    for index in range(20):
        state.characters[f"stranger-{index}"] = Character(
            id=f"stranger-{index}", location="tavern"
        )

    context = storyteller_context(state)

    assert "**Clear the Cave** — failed" in context
    assert "abandoned, never managed to clear" in context
    assert "**old-bram** at road; wants: retire; hero sees them as: mentor" in context
    assert context.count("**stranger-") == 7  # capped with old-bram first
    assert "- road" in context  # gateway
