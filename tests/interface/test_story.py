import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from src.interface.app import STORY_HTML, _build_handler, _with_nav
from src.interface.story import list_chapters, load_chapter, load_tail

EVENT = {"text": "The door opens.", "location": "tavern", "characters": ["hero"]}


def _run(tmp_path, chapters=(), tail=(EVENT,)):
    run_dir = tmp_path / "demo" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "world_state_3.json").write_text(
        json.dumps({"history": list(tail)}), encoding="utf-8"
    )
    if chapters:
        (run_dir / "chapters").mkdir()
    for number, title in chapters:
        (run_dir / "chapters" / f"chapter_{number:04d}.json").write_text(
            json.dumps(
                {
                    "number": number,
                    "title": title,
                    "prose": "One.\n\nTwo *soft* <b>x</b>.",
                    "summary": "s",
                    "events": [EVENT, EVENT],
                }
            ),
            encoding="utf-8",
        )


@pytest.fixture
def get(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler(tmp_path, tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def fetch(path: str) -> tuple[int, dict]:
        url = f"http://127.0.0.1:{server.server_port}{path}"
        try:
            with urllib.request.urlopen(url) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    yield fetch
    server.shutdown()
    server.server_close()
    thread.join()


def test_chapter_list_is_cheap_and_ordered_numerically(tmp_path):
    _run(tmp_path, [(10, "Ten"), (2, "Two")])
    assert list_chapters(tmp_path, "demo", "run-1") == [
        {"number": 2, "title": "Two", "event_count": 2},
        {"number": 10, "title": "Ten", "event_count": 2},
    ]


def test_missing_chapters_dir_gives_empty_list_and_tail_only(tmp_path):
    _run(tmp_path)
    assert list_chapters(tmp_path, "demo", "run-1") == []
    assert load_tail(tmp_path, "demo", "run-1") == {
        "tick": 3,
        "chapter_count": 0,
        "events": [EVENT],
    }


def test_unreadable_chapter_is_skipped_in_list(tmp_path):
    _run(tmp_path, [(1, "One")])
    (tmp_path / "demo" / "run-1" / "chapters" / "chapter_0002.json").write_text("{")
    assert [c["number"] for c in list_chapters(tmp_path, "demo", "run-1")] == [1]


def test_load_chapter_missing_raises(tmp_path):
    _run(tmp_path, [(1, "One")])
    assert load_chapter(tmp_path, "demo", "run-1", 1)["title"] == "One"
    with pytest.raises(FileNotFoundError):
        load_chapter(tmp_path, "demo", "run-1", 2)


def test_story_endpoints(tmp_path, get):
    _run(tmp_path, [(1, "One"), (2, "Two")])
    assert get("/api/story/demo/run-1") == (
        200,
        {
            "chapters": [
                {"number": 1, "title": "One", "event_count": 2},
                {"number": 2, "title": "Two", "event_count": 2},
            ]
        },
    )
    status, chapter = get("/api/story/demo/run-1/2")
    assert status == 200 and chapter["title"] == "Two" and len(chapter["events"]) == 2
    status, tail = get("/api/story/demo/run-1/tail")
    assert status == 200 and tail["chapter_count"] == 2 and tail["events"] == [EVENT]


def test_story_endpoints_without_chapters_dir(tmp_path, get):
    _run(tmp_path)
    assert get("/api/story/demo/run-1") == (200, {"chapters": []})
    assert get("/api/story/demo/run-1/tail")[1]["chapter_count"] == 0
    assert get("/api/story/demo/run-1/1")[0] == 404


@pytest.mark.parametrize(
    "path",
    [
        "/api/story/demo/run-1/..%2F..%2Fworld_state_3",
        "/api/story/demo/run-1/1.json",
        "/api/story/demo/run-1/-1",
        "/api/story/demo/run-1/%C2%B2",
        "/api/story/..%2Fdemo/run-1",
        "/api/story/demo/..%2Frun-1/tail",
        "/api/story/demo/..",
        "/api/story/..%2F..%2Fetc/passwd/tail",
        "/api/story/nope/run-1",
    ],
)
def test_story_rejects_traversal_and_unknown_runs(tmp_path, get, path):
    _run(tmp_path, [(1, "One")])
    assert get(path)[0] == 404


def test_story_page_has_active_tab():
    html = _with_nav(STORY_HTML, "story")
    assert '<a href="/story" class="active" aria-current="page">Story</a>' in html
    assert 'class="view-story"' in html
