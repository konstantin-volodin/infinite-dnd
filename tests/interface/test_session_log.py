from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from src.interface.session_log import Logger, _digest_message, list_logs, load_session


def _write_session(tmp_path) -> None:
    logger = Logger(character_id="hero", max_turns=3, scenario="demo", scenario_title="Demo Quest", log_dir=tmp_path)
    logger.log_turn(1)
    with logger.run("character:hero"):
        logger.log_messages("character:hero", [
            ModelRequest(parts=[SystemPromptPart(content="you are hero"), UserPromptPart(content="act")]),
            ModelResponse(
                parts=[ToolCallPart(tool_name="speak", args='{"message": "hi"}', tool_call_id="t1")],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name="test-model",
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name="speak", content="ok", tool_call_id="t1")]),
        ])
    logger.log_event("resolved", tool="Speak", subject="hero", result="hero says: \"hi\"")
    logger.close()


def test_roundtrip(tmp_path):
    _write_session(tmp_path)

    files = list_logs(tmp_path)
    assert len(files) == 1
    assert files[0]["character_id"] == "hero"
    assert files[0]["scenario_title"] == "Demo Quest"
    assert files[0]["turns_completed"] == 1

    session = load_session(files[0]["path"])
    assert session["summary"]["scenario_title"] == "Demo Quest"
    assert session["summary"]["turns_completed"] == 1
    assert session["stats"]["llm_message_count"] == 3
    assert session["stats"]["error_count"] == 0

    run, = session["runs"]
    assert run["label"] == "character:hero"
    assert run["tool_calls"] == ["speak"]
    assert run["tokens"] == {"input_tokens": 10, "output_tokens": 5}
    assert run["elapsed_s"] is not None
    assert run["output_tokens_per_s"] is not None

    finished = next(e for e in session["entries"] if e["event"] == "agent_run_finished")
    assert finished["raw"]["input_tokens"] == 10
    assert finished["raw"]["output_tokens"] == 5
    assert finished["raw"]["output_tokens_per_s"] > 0
    assert "output tok/s" in finished["summary"]

    request, response, tool_return = [e for e in session["entries"] if e["event"] == "llm_message"]
    assert request["message_kind"] == "model_request"
    assert "you are hero" in request["raw_pretty"]
    assert response["message_kind"] == "model_response"
    assert response["tool_calls"] == ["speak"]
    assert response["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert tool_return["message_kind"] == "tool_return"
    assert tool_return["tool_returns"] == ["speak"]


def test_logger_records_prefill_decode_split(tmp_path):
    # Simulates llama-server /metrics counters advancing across one run:
    # 600 prompt tokens in 6s (prefill) and 40 generated tokens in 5s (decode).
    snapshots = iter([
        {"prompt_tokens": 1000.0, "prompt_seconds": 10.0, "predicted_tokens": 200.0, "predicted_seconds": 20.0},
        {"prompt_tokens": 1600.0, "prompt_seconds": 16.0, "predicted_tokens": 240.0, "predicted_seconds": 25.0},
    ])
    logger = Logger(
        character_id="hero",
        max_turns=1,
        log_dir=tmp_path,
        metrics_reader=lambda: next(snapshots, None),
    )
    logger.log_turn(1)
    with logger.run("character:hero"):
        pass
    logger.close()

    session = load_session(list_logs(tmp_path)[0]["path"])
    run, = session["runs"]
    assert run["timings"] == {
        "prefill_tokens": 600, "prefill_s": 6.0, "prefill_tps": 100.0,
        "decode_tokens": 40, "decode_s": 5.0, "decode_tps": 8.0,
    }
    finished = next(e for e in session["entries"] if e["event"] == "agent_run_finished")
    assert "prefill 100.0 tok/s · decode 8.0 tok/s" in finished["summary"]


def test_logger_omits_split_when_server_unreachable(tmp_path):
    logger = Logger(character_id="hero", max_turns=1, log_dir=tmp_path, metrics_reader=lambda: None)
    logger.log_turn(1)
    with logger.run("character:hero"):
        pass
    logger.close()

    session = load_session(list_logs(tmp_path)[0]["path"])
    run, = session["runs"]
    assert run["timings"] is None
    finished = next(e for e in session["entries"] if e["event"] == "agent_run_finished")
    assert "prefill" not in finished["raw"]


def test_logger_uses_explicit_session_id(tmp_path):
    logger = Logger(
        character_id="hero",
        max_turns=1,
        scenario="demo",
        scenario_title="Demo Quest",
        log_dir=tmp_path,
        session_id="explicit-id",
    )
    logger.close()

    assert logger.session_id == "explicit-id"
    assert (tmp_path / "explicit-id.log").exists()


def test_logger_can_record_only_completed_turns_after_a_failed_turn(tmp_path):
    logger = Logger(character_id="hero", max_turns=3, log_dir=tmp_path)
    logger.log_turn(1)
    logger.close(turns_completed=0)

    session = load_session(list_logs(tmp_path)[0]["path"])

    assert session["summary"]["turns_completed"] == 0


def test_logger_append_preserves_existing_session(tmp_path):
    path = tmp_path / "resumed.log"
    path.write_text('{"event": "earlier"}\n', encoding="utf-8")

    logger = Logger(
        character_id="hero",
        max_turns=1,
        log_dir=tmp_path,
        session_id="resumed",
        append=True,
    )
    logger.close()

    assert path.read_text(encoding="utf-8").startswith('{"event": "earlier"}\n')


def test_digest_legacy_repr_string():
    message = (
        "ModelResponse(parts=[ToolCallPart(tool_name='travel', args='{}', tool_call_id='x')], "
        "usage=RequestUsage(input_tokens=42, cache_read_tokens=7, output_tokens=3), model_name='m')"
    )
    digest = _digest_message(message)
    assert digest["kind"] == "model_response"
    assert digest["tool_calls"] == ["travel"]
    assert digest["usage"] == {"input_tokens": 42, "cache_read_tokens": 7, "output_tokens": 3}
