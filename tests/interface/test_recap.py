import json

from src.engine.state.models import (
    Character,
    CharacterStats,
    HistoryEvent,
    Location,
    Quest,
    WorldState,
)
from src.interface.recap import build_recap, main


def _state() -> WorldState:
    return WorldState(
        minutes_elapsed=1440,
        chronicle=["Long ago, the village burned.", "The hero was born in the ashes."],
        locations={
            "tavern": Location(id="tavern"),
            "forest": Location(id="forest"),
        },
        history=[
            HistoryEvent(
                text="Hero arrives at the tavern.",
                location="tavern",
                minutes_elapsed=30,
            ),
            HistoryEvent(
                text="Hero orders a drink.", location="tavern", minutes_elapsed=10
            ),
            HistoryEvent(
                text="Hero travels to the forest.",
                location="forest",
                minutes_elapsed=1400,
            ),
        ],
        quests={
            "q1": Quest(
                id="q1",
                title="Find the Amulet",
                description="",
                status="completed",
                plan=["find the lair", "recover the amulet"],
                current_step=2,
                steps=["found the lair", "recovered the amulet"],
            ),
            "q2": Quest(
                id="q2",
                title="Watch the Border",
                description="",
                status="active",
                steps=["patrol reported quiet"],
            ),
        },
        characters={
            "hero": Character(
                id="hero",
                role="warrior",
                location="forest",
                stats=CharacterStats(hp=0, max_hp=10, level=2, gold=15),
            ),
            "villain": Character(id="villain", stats=CharacterStats(hp=5, max_hp=5)),
        },
    )


def _chapters() -> list[dict]:
    return [
        {
            "number": 1,
            "title": "The Burning",
            "prose": "The village burned.\n\nThe hero rose from the ashes.",
            "summary": "The village burned.",
            "events": [
                {
                    "text": "Fire spreads.",
                    "location": "tavern",
                    "characters": [],
                    "minutes_elapsed": 1500,
                }
            ],
        },
        {
            "number": 2,
            "title": "The Road",
            "prose": "The hero walked on.",
            "summary": "The hero left.",
            "events": [
                {
                    "text": "Hero leaves.",
                    "location": "forest",
                    "characters": ["hero"],
                    "minutes_elapsed": 60,
                }
            ],
        },
    ]


def test_title_and_metadata_header():
    md = build_recap(
        WorldState(), title="The Ashen Vale", scenario="ashen-vale", run_id="run-1"
    )
    assert md.startswith("# The Ashen Vale\n\n*ashen-vale · run-1*")


def test_chapters_render_in_order_with_their_raw_events():
    md = build_recap(
        _state(), title="Demo", scenario="demo", run_id="run-1", chapters=_chapters()
    )
    assert "## Chapter 1: The Burning\n\nThe village burned.\n\nThe hero rose" in md
    assert (
        md.index("## Chapter 1")
        < md.index("## Chapter 2")
        < md.index("## Since the last chapter")
    )
    assert "<summary>What happened (1 events)</summary>" in md
    assert "**day 1 · 00:00 · tavern**\n\n- Fire spreads." in md
    # clocks run on across chapters: chapter 2 starts after chapter 1's 1500 minutes
    assert "**day 2 · 01:00 · forest**\n\n- Hero leaves." in md
    assert "## Previously…" not in md  # chapters supersede the chronicle


def test_chronicle_renders_as_previously_for_runs_without_chapters():
    md = build_recap(_state(), title="Demo", scenario="demo", run_id="run-1")
    assert "## Previously…" in md
    assert "Long ago, the village burned." in md
    assert md.index("## Previously…") < md.index("## Since the last chapter")


def test_unarchived_tail_groups_by_day_and_location():
    md = build_recap(_state(), title="Demo", scenario="demo", run_id="run-1")
    assert (
        "**day 1 · 00:00 · tavern**\n\n- Hero arrives at the tavern.\n- Hero orders a drink."
        in md
    )
    assert "**day 1 · 00:40 · forest**\n\n- Hero travels to the forest." in md


def test_quest_outcomes_show_status_and_progress():
    md = build_recap(_state(), title="Demo", scenario="demo", run_id="run-1")
    assert (
        "**Find the Amulet** — completed (2/2 steps); last: recovered the amulet" in md
    )
    assert (
        "**Watch the Border** — active (1 steps logged); last: patrol reported quiet"
        in md
    )


def test_epilogue_reports_each_characters_fate():
    md = build_recap(_state(), title="Demo", scenario="demo", run_id="run-1")
    assert "**hero** (warrior) — dead, level 2, 15 gold, at forest" in md
    assert "**villain** — healthy, level 1, 0 gold, at unknown" in md


def test_empty_state_does_not_crash():
    md = build_recap(WorldState(), title="Empty", scenario="demo", run_id="run-1")
    assert md.startswith("# Empty\n")
    assert "## Previously…" not in md
    assert "## Since the last chapter" not in md
    assert "## Quest Outcomes" not in md
    assert "## Epilogue" not in md


def test_main_writes_recap_next_to_snapshot(tmp_path):
    run_dir = tmp_path / "demo" / "run-1"
    run_dir.mkdir(parents=True)
    snapshot = json.loads(_state().model_dump_json())
    (run_dir / "world_state_0.json").write_text(json.dumps(snapshot), encoding="utf-8")

    exit_code = main(["--state-dir", str(tmp_path)])

    assert exit_code == 0
    out_path = run_dir / "run-1_recap.md"
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith("# demo\n")


def test_main_includes_the_runs_chapter_files(tmp_path):
    run_dir = tmp_path / "demo" / "run-1"
    (run_dir / "chapters").mkdir(parents=True)
    (run_dir / "world_state_0.json").write_text(_state().model_dump_json())
    for chapter in _chapters():
        path = run_dir / "chapters" / f"chapter_{chapter['number']:04d}.json"
        path.write_text(json.dumps(chapter), encoding="utf-8")

    main(["--state-dir", str(tmp_path)])

    book = (run_dir / "run-1_recap.md").read_text(encoding="utf-8")
    assert "## Chapter 2: The Road" in book


def test_main_errors_when_no_runs_found(tmp_path, capsys):
    try:
        main(["--state-dir", str(tmp_path)])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 2
