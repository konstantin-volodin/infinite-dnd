import pytest

from src.agents import server
from src.agents.server import _profile_args, _reasoning_args, load_profile

_CONFIG = """\
default:
  ctx-size: 8192
  parallel: 1
  flash-attn: "on"
fast:
  parallel: 3
"""


@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / "llama.yaml"
    path.write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setattr(server, "_CONFIG_PATH", path)
    monkeypatch.delenv("LLM_PROFILE", raising=False)
    return path


def test_load_profile_merges_over_default(config):
    name, profile = load_profile("fast")

    assert name == "fast"
    assert profile == {"ctx-size": 8192, "parallel": 3, "flash-attn": "on"}


def test_load_profile_name_defaults_to_environment(config, monkeypatch):
    monkeypatch.setenv("LLM_PROFILE", "fast")

    assert load_profile()[0] == "fast"


def test_load_profile_falls_back_to_default(config):
    name, profile = load_profile()

    assert name == "default"
    assert profile["parallel"] == 1


def test_load_profile_rejects_unknown_name(config):
    with pytest.raises(RuntimeError, match="Unknown LLM_PROFILE 'nope'"):
        load_profile("nope")


def test_profile_args_render_flags_and_booleans():
    # An unquoted `flash-attn: on` in YAML parses as boolean True.
    args = _profile_args({"ctx-size": 8192, "flash-attn": True}, model="qwen3.5-35b")

    assert args == ["--ctx-size", "8192", "--flash-attn", "on"]


def test_profile_args_route_reasoning_through_gemma_workaround():
    args = _profile_args({"reasoning": "off"}, model="unsloth/gemma-4-26B-A4B-it-GGUF")

    assert args == ["--reasoning", "auto", "--reasoning-budget", "0"]


def test_gemma_4_reasoning_off_uses_tool_safe_zero_budget():
    assert _reasoning_args("unsloth/gemma-4-26B-A4B-it-GGUF", "off") == [
        "--reasoning",
        "auto",
        "--reasoning-budget",
        "0",
    ]


def test_other_models_keep_reasoning_off():
    assert _reasoning_args("qwen3.5-35b", "off") == ["--reasoning", "off"]


def test_explicit_gemma_4_reasoning_mode_is_preserved():
    assert _reasoning_args("gemma-4-26b", "on") == ["--reasoning", "on"]
