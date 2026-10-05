import json

from src.interface.app import _with_nav
from src.interface.assets import read_asset, static_asset
from src.interface.world_state import load_series


def test_nav_is_added_once():
    html = _with_nav('<style></style><div class="topbar-main"></div>', "world")
    assert html.count('<nav class="tabs" aria-label="Primary">') == 1
    assert 'href="/logs"' in html
    assert 'href="/"' in html


def test_nav_marks_active_tab():
    html = _with_nav('<div class="topbar-main"></div>', "logs")
    assert '<a href="/logs" class="active" aria-current="page">Logs</a>' in html
    assert '<a href="/" class="">World</a>' in html


def test_shell_links_shared_studio_stylesheet():
    html = _with_nav('<head></head><body><div class="topbar-main"></div></body>', "world")
    assert 'href="/static/studio.css"' in html
    assert 'class="view-world"' in html


def test_static_assets_are_whitelisted():
    body, content_type = static_asset("world.css") or (b"", "")
    assert body
    assert content_type == "text/css; charset=utf-8"
    assert static_asset("../world.css") is None


def test_world_controls_have_accessible_names_and_dialog_semantics():
    html = read_asset("templates/world.html")
    assert '<label class="sr-only" for="run">Game run</label>' in html
    assert 'id="tick-prev" type="button"' in html
    assert 'id="map-svg"' in html and 'aria-label="World locations"' in html
    assert 'id="char-drawer" role="dialog" aria-modal="true"' in html
    assert 'id="play-input"' in html and 'aria-describedby="play-help play-error"' in html


def test_dynamic_world_controls_support_keyboard_and_focus_restoration():
    script = read_asset("static/world.js")
    assert 'data-loc="${escapeHtml(n.id)}"' in script and 'role="button" tabindex="0"' in script
    assert 'data-char="${escapeHtml(c.id)}"' in script and 'aria-haspopup="dialog"' in script
    assert 'dom.mapSvg.addEventListener("keydown"' in script
    assert 'dom.charGrid.addEventListener("keydown"' in script
    assert 'state.lastDialogFocus.focus()' in script


def test_play_panel_connects_a_submitted_move_to_the_next_story_event():
    html = read_asset("templates/world.html")
    script = read_asset("static/world.js")
    assert 'class="play-panel"' in html
    assert 'class="play-panel collapsed"' not in html
    assert 'id="play-toggle"' not in html
    assert 'class="sr-only" id="play-title"' in html
    assert 'class="play-alert" id="play-situation" role="status" aria-live="polite" hidden' in html
    assert "Player controls are disabled" in html
    css = read_asset("static/world.css")
    assert "left:50%" in css and "transform:translateX(-50%)" in css and ".play-alert" in css and ".situation-row" in css
    assert "min-height:48px" in css and "font-size:15px" in css
    assert "prefers-reduced-motion" in css and "story-event-in" in css
    assert 'id="play-trail" aria-live="polite" hidden' in html
    assert 'id="play-last-action"' in html
    assert 'id="play-last-outcome"' in html
    assert "function describeAction(action, fallback)" in script
    assert "function renderPlayAlert(text)" in script
    assert "history.length <= recent.historyLength" in script
    assert "recent.outcome = history.at(recent.historyLength)?.text" in script
    assert "label: describeAction(result.action, line)" in script


def test_play_panel_surfaces_contextual_actions_in_the_action_row():
    html = read_asset("templates/world.html")
    script = read_asset("static/world.js")
    assert 'id="play-suggestions" role="group" aria-label="Suggested actions"' in html
    assert '<div class="play-actions">' in html
    assert 'id="play-submit" type="submit" disabled>Act</button>' in html
    assert "function contextualChoices(view, actorId)" in script
    assert "world.locations[destination]" in script
    assert "character.location === actor.location" in script
    assert 'dom.playInput.value = suggestion.dataset.action' in script
    assert 'dom.playSuggestions.addEventListener("click"' in script
    assert 'dom.playForm.dispatchEvent' not in script
    assert "Player controls are disabled" in html


def test_play_panel_and_log_story_surface_terminal_campaign_outcomes():
    world_script = read_asset("static/world.js")
    logs_script = read_asset("static/logs.js")
    assert "function campaignOutcome(view, actorId = view?.pc)" in world_script
    assert '"Campaign complete"' in world_script
    assert '"Campaign failed"' in world_script
    assert 'campaign_completed: "green"' in logs_script
    assert 'campaign_failed: "ember"' in logs_script


def test_log_story_surfaces_runtime_errors():
    script = read_asset("static/logs.js")

    assert 'run_error: "ember"' in script
    assert '"parse_error", "run_error"' in script
    assert 'run_error: "error"' in script
    assert '<div class="insp-label">Errors</div>' in script


def test_log_story_surfaces_rejected_world_updates():
    script = read_asset("static/logs.js")

    assert 'world_update_rejected: "amber"' in script
    assert '"run_error", "world_update_rejected"' in script
    assert 'world_update_rejected: "rejected"' in script


def test_log_filters_and_rows_expose_native_control_semantics():
    html = read_asset("templates/logs.html")
    script = read_asset("static/logs.js")
    assert '<label class="search" for="search">' in html
    assert 'role="group" aria-label="Log detail level"' in html
    assert 'data-detail="story" class="active" aria-pressed="true"' in html
    assert '<button type="button" class="log-row' in script
    assert 'b.setAttribute("aria-pressed", String(active))' in script


def test_series_math(tmp_path):
    run_dir = tmp_path / "demo" / "run-1"
    run_dir.mkdir(parents=True)
    for tick, hp, xp, gold in [(0, 5, 0, 1), (2, 3, 10, 4)]:
        payload = {"characters": {"hero": {"stats": {"hp": hp, "xp": xp, "gold": gold}}}}
        (run_dir / f"world_state_{tick}.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_series(tmp_path, "demo", "run-1")["hero"] == [
        {"tick": 0, "hp": 5, "xp": 0, "gold": 1},
        {"tick": 2, "hp": 3, "xp": 10, "gold": 4},
    ]
