"""Live tests — verify proposer agents emit correct intents and Resolver mutates state.

Requires a reachable LLM provider: local llama.cpp server, ANTHROPIC_API_KEY when
LLM_PROVIDER=anthropic, or OPENAI_API_KEY when LLM_PROVIDER=openai. Skipped
automatically otherwise. Run explicitly with:
    uv run pytest -m integration tests/integration
"""

import asyncio
import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext, capture_run_messages
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits

from src.engine.state import Character, StateManager, WorldState
from src.agents.utils import create_model
from src.agents.character.agent import CharacterDeps, agent as character_agent
from src.agents.action_resolver.agent import (
    ActionResolverDeps,
    agent as resolver_agent,
    discover_exit as discover_exit_tool,
    resolve,
)
from src.agents.character.tools import Action, Speak, Travel

pytestmark = pytest.mark.integration

LOG_DIR = Path(__file__).resolve().parents[2] / "src" / "tests" / "logs"
LOG_DIR.mkdir(exist_ok=True)
_stamp = datetime.now().strftime("%Y-%m-%d")

_MODEL_SETTINGS: ModelSettings = {
    "temperature": 0.0,
    "max_tokens": 512,
    "parallel_tool_calls": False,
    "thinking": False,
}
_USAGE_LIMITS = UsageLimits(request_limit=4, output_tokens_limit=256)


def _provider_reachable() -> bool:
    """Cheap reachability check — no LLM call, just a connectivity/config probe."""
    provider = os.getenv("LLM_PROVIDER", "local")
    if provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1").removesuffix("/v1") + "/health"
    try:
        r = urllib.request.urlopen(url, timeout=5)
        ok = r.status == 200
        r.close()
        return ok
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


@pytest.fixture(autouse=True)
def _skip_without_provider():
    if not _provider_reachable():
        pytest.skip("no live LLM provider reachable (local server down or ANTHROPIC_API_KEY unset)")


@pytest.fixture
def state() -> WorldState:
    return StateManager().init_state()


def _pick_char(state: WorldState) -> Character:
    return next(iter(state.characters.values()))


def _pick_travel_target(state: WorldState, char: Character) -> str:
    loc = state.locations.get(char.location)
    if not loc or not loc.connections:
        raise RuntimeError(f"No travel target for {char.location!r}")
    return loc.connections[0]


def _assert_tool(messages: list[Any], name: str) -> None:
    calls = [
        getattr(p, "tool_name", None)
        for m in messages for p in getattr(m, "parts", [])
        if getattr(p, "part_kind", None) == "tool-call"
    ]
    assert name in calls, f"expected {name!r}, saw {calls!r}"


def _run(
    agent: Any, prefix: str, prompt: str, deps: Any, tool: str, *, expect_done: bool = False,
) -> tuple[Any, list[Any]]:
    with capture_run_messages() as msgs:
        result = agent.run_sync(prompt, deps=deps, model_settings=_MODEL_SETTINGS, usage_limits=_USAGE_LIMITS)
    _assert_tool(msgs, tool)
    if expect_done:
        _assert_tool(msgs, "done")

    trace = [
        {"kind": type(m).__name__, "parts": [
            {"kind": getattr(p, "part_kind", type(p).__name__), "tool": getattr(p, "tool_name", None),
             "content": getattr(p, "content", None), "args": getattr(p, "args", None)}
            for p in getattr(m, "parts", [])
        ]}
        for m in msgs
    ]
    path = LOG_DIR / f"{_stamp}-{prefix}-{tool}.md"
    path.write_text(
        f"# {prefix}: {tool}\n\n## Prompt\n{prompt}\n\n## Output\n{result.output!r}\n\n"
        f"## Trace\n```json\n{json.dumps(trace, indent=2, default=str)}\n```",
        encoding="utf-8",
    )
    return result.output, list(msgs)


# ── server ───────────────────────────────────────────────────────────────

def test_server_health():
    assert _provider_reachable()


def test_basic_completion():
    a = Agent(create_model(), system_prompt="reply with one word only.", output_type=str)
    resp = a.run_sync("Say 'hello'.", model_settings={"temperature": 0.0, "thinking": False}).output
    assert isinstance(resp, str) and len(resp.strip()) > 0


# ── character (intent generator) ─────────────────────────────────────────

def test_character_contextual_tools(state):
    char = _pick_char(state)
    other = next(c for c in state.characters.values() if c.id != char.id)
    other.location = char.location

    deps = CharacterDeps(char=char, state=state)
    ctx = RunContext(deps=deps, model=create_model(), usage=RunUsage(), agent=character_agent)
    output_toolset = character_agent._output_toolset
    prepare_output_tools = character_agent._prepare_output_tools
    assert output_toolset is not None
    assert prepare_output_tools is not None
    prepared = output_toolset.prepared(prepare_output_tools)
    assert prepared is not None
    tools = {n: t.tool_def for n, t in asyncio.run(prepared.get_tools(ctx)).items()}

    expected_targets = [c.id for c in state.characters.values() if c.location == char.location and c.id != char.id]
    for t in expected_targets:
        assert t in tools["speak"].description, f"speak description missing target {t!r}"

    assert "travel" in tools, "travel tool should exist when connections are available"
    expected_travel = [cid for cid in state.locations[char.location].connections if cid in state.locations]
    for loc in expected_travel:
        assert loc in tools["travel"].description, f"travel description missing option {loc!r}"


def test_character_action_intent(state):
    char = _pick_char(state)
    before = len(state.history)
    intent, _ = _run(
        character_agent, "character",
        'Call `action` once with description "carefully inspects the old fountain". No other tools.',
        CharacterDeps(char=char, state=state), "action",
    )
    assert isinstance(intent, Action), f"expected Action, got {type(intent).__name__}"
    assert intent.actor == char.id
    assert len(state.history) == before, "character agent must not mutate state"

    # Resolver executes it
    asyncio.run(resolve(intent, state))
    assert len(state.history) > before, "resolve(action) should append history"


def test_character_speak_intent(state):
    char = _pick_char(state)
    before = len(state.history)
    msg = "I have a feeling something important is happening nearby."
    intent, _ = _run(
        character_agent, "character",
        f'Call `speak` once with message exactly "{msg}". No other tools.',
        CharacterDeps(char=char, state=state), "speak",
    )
    assert isinstance(intent, Speak)
    assert intent.actor == char.id and intent.message == msg
    assert len(state.history) == before, "character agent must not mutate state"

    asyncio.run(resolve(intent, state))
    assert state.history[-1].text == f'{char.id} says: "{msg}"'


def test_character_travel_intent(state):
    char = _pick_char(state)
    target = _pick_travel_target(state, char)
    before = len(state.history)
    intent, _ = _run(
        character_agent, "character",
        f'Call `travel` once to "{target}". No other tools.',
        CharacterDeps(char=char, state=state), "travel",
    )
    assert isinstance(intent, Travel)
    assert intent.actor == char.id and intent.destination == target
    assert len(state.history) == before, "character agent must not mutate state"
    assert char.location != target, "character agent must not move the character"

    asyncio.run(resolve(intent, state))
    assert char.location == target
    assert state.history[-1].text == f"{char.id} moved to '{target}'."


# ── resolver sub-agent (free-form Action path) ──────────────────────────

def test_resolver_remember(state):
    char = _pick_char(state)
    before = len(state.history)
    knowledge = "The old fountain has a loose stone ring around its basin."
    output, _ = _run(
        resolver_agent, "resolver",
        f'Call `remember` once with knowledge exactly "{knowledge}". Then call `done` mentioning the fountain.',
        ActionResolverDeps(char=char, state=state, description="carefully inspects the old fountain"),
        "remember", expect_done=True,
    )
    assert knowledge in char.knowledge
    assert len(state.history) == before + 1
    assert state.history[-1].text == f"{char.id} learns: {knowledge}"
    assert "fountain" in output.lower(), f"done output should mention fountain: {output!r}"


def test_resolver_discover_exit(state):
    char = _pick_char(state)
    ctx = RunContext(
        deps=ActionResolverDeps(char=char, state=state, description="asks about a hidden cellar"),
        model=create_model(), usage=RunUsage(), agent=resolver_agent,
    )
    output = discover_exit_tool(ctx, name="Hidden Cellar", description="A narrow stone stair descends into a damp cellar.")
    loc = state.locations.get("hidden-cellar")
    assert loc, "hidden-cellar should be created"
    assert char.location in loc.connections
    assert output == "Location 'hidden-cellar' added."
