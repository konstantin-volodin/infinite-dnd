from src.agents.dm.chronicler import chapter_prompt, parse_chapter, readable
from src.engine.state.models import (
    Character,
    HistoryEvent,
    Location,
    Quest,
    WorldState,
)


def test_parse_chapter_reads_the_plain_text_layout():
    reply = (
        "TITLE: The Crescent in the Net\n"
        "PROSE:\n"
        "First   paragraph.\n\n"
        "Second paragraph.\n"
        "Third, after a single newline.\n\n"
        "SUMMARY: Brenna found powder.\n"
        "Keph gave her a spyglass."
    )
    assert parse_chapter(reply) == (
        "The Crescent in the Net",
        "Brenna found powder. Keph gave her a spyglass.",
        "First paragraph.\n\nSecond paragraph.\n\nThird, after a single newline.",
    )


def test_parse_chapter_tolerates_markdown_and_a_missing_prose_marker():
    reply = (
        "Here you go.\n**Title:** *The Dock*\n\nShe waited.\n\n**Summary**: She waited."
    )
    assert parse_chapter(reply) == ("The Dock", "She waited.", "She waited.")


def test_parse_chapter_of_junk_is_empty():
    assert parse_chapter("I cannot help with that.") == ("", "", "")


def _state() -> WorldState:
    return WorldState(
        locations={"harbor-dock": Location(id="harbor-dock")},
        characters={
            "old-keph": Character(id="old-keph"),
            "keph": Character(id="keph"),
            "brenna-deep": Character(
                id="brenna-deep", role="fisher", backstory="Nets came up heavy."
            ),
        },
        quests={
            "q": Quest(
                id="q", title="What the Nets Brought Up", plan=["find the crate"]
            )
        },
    )


def test_readable_names_entities_and_drops_dice_brackets():
    text = "old-keph succeeds: reads the harbor-dock tide [wisdom; 14+1=15 vs DC 12]. keph nods."
    assert (
        readable(text, _state())
        == "Old Keph succeeds: reads the harbor dock tide. Keph nods."
    )


def test_chapter_prompt_holds_hero_memory_ending_quests_and_scenes():
    state = _state()
    state.chronicle = ["The saga so far."]
    events = [
        HistoryEvent(
            text="brenna-deep waits.", location="harbor-dock", minutes_elapsed=1
        ),
        HistoryEvent(
            text="old-keph arrives.", location="lighthouse", minutes_elapsed=1
        ),
    ]
    prompt = chapter_prompt(
        state,
        events,
        number=3,
        hero="brenna-deep",
        start_minutes=1440,
        previous_ending="She slept.",
    )
    assert "Brenna Deep, fisher. Nets came up heavy." in prompt
    assert "The saga so far." in prompt
    assert "## how chapter 2 ended\nShe slept." in prompt
    assert "- What the Nets Brought Up: find the crate" in prompt
    assert "### day 2, harbor dock\n- Brenna Deep waits." in prompt
    assert "### day 2, lighthouse\n- Old Keph arrives." in prompt
    assert prompt.index("Brenna Deep waits") < prompt.index("Old Keph arrives")
    assert "Write chapter 3." in prompt
