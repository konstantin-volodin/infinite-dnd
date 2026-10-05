"""Session logs: the JSONL format for game observability — one writer (Logger), one reader (load_session)."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from pydantic_ai.messages import ModelMessagesTypeAdapter

from src.world import list_scenarios, read_manifest

DEFAULT_LOG_DIR = Path("logs")


# ============================================================
# Writer
# ============================================================

def _phase_timings(before: dict[str, float] | None, after: dict[str, float] | None) -> dict[str, Any]:
    """Split a run's server-side work into prefill vs decode from cumulative counter deltas."""
    if not before or not after:
        return {}

    def delta(key: str) -> float:
        return after.get(key, 0.0) - before.get(key, 0.0)

    prefill_tokens, prefill_s = delta("prompt_tokens"), delta("prompt_seconds")
    decode_tokens, decode_s = delta("predicted_tokens"), delta("predicted_seconds")
    if prefill_tokens <= 0 and decode_tokens <= 0:
        return {}
    return {
        "prefill_tokens": int(prefill_tokens),
        "prefill_s": round(prefill_s, 3),
        "prefill_tps": round(prefill_tokens / prefill_s, 1) if prefill_s > 0 else None,
        "decode_tokens": int(decode_tokens),
        "decode_s": round(decode_s, 3),
        "decode_tps": round(decode_tokens / decode_s, 1) if decode_s > 0 else None,
    }


class Logger:
    def __init__(
        self,
        *,
        character_id: str,
        max_turns: int,
        scenario: str | None = None,
        scenario_title: str | None = None,
        log_dir: Path = DEFAULT_LOG_DIR,
        session_id: str | None = None,
        append: bool = False,
        metrics_reader: Callable[[], dict[str, float] | None] | None = None,
    ):
        # Optional snapshot of the LLM server's cumulative prefill/decode counters
        # (see src.agents.server.read_metrics); deltas around a run split its wall
        # time into prompt processing vs generation.
        self._metrics_reader = metrics_reader
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.turn = 0
        self._active_runs: dict[str, list[dict[str, int]]] = {}
        mode = "a" if append else "w"
        self._file = (self.log_dir / f"{self.session_id}.log").open(mode, encoding="utf-8")
        self._log(
            "game_session_started",
            character_id=character_id,
            max_turns=max_turns,
            scenario=scenario,
            scenario_title=scenario_title,
        )

    def close(self, *, turns_completed: int | None = None) -> None:
        self._log(
            "game_session_finished",
            turns_completed=self.turn if turns_completed is None else turns_completed,
        )
        self._file.close()

    def log_turn(self, turn: int) -> None:
        self.turn = turn
        self._log("turn_started", turn=turn)

    @contextmanager
    def run(self, label: str) -> Iterator[None]:
        start = time.perf_counter()
        stats = {"input_tokens": 0, "output_tokens": 0}
        self._active_runs.setdefault(label, []).append(stats)
        metrics_before = self._metrics_reader() if self._metrics_reader else None
        self._log("agent_run_started", label=label, turn=self.turn or None)
        try:
            yield
        finally:
            elapsed = round(time.perf_counter() - start, 3)
            self._active_runs[label].pop()
            if not self._active_runs[label]:
                del self._active_runs[label]
            output_tps = round(stats["output_tokens"] / elapsed, 2) if elapsed and stats["output_tokens"] else None
            metrics_after = self._metrics_reader() if metrics_before is not None else None
            self._log(
                "agent_run_finished",
                label=label,
                turn=self.turn or None,
                elapsed_s=elapsed,
                input_tokens=stats["input_tokens"],
                output_tokens=stats["output_tokens"],
                output_tokens_per_s=output_tps,
                **_phase_timings(metrics_before, metrics_after),
            )

    def log_messages(self, label: str, messages: list[Any]) -> None:
        try:
            dumped = ModelMessagesTypeAdapter.dump_python(messages, mode="json", exclude_none=True)
        except Exception:
            dumped = [str(msg) for msg in messages]
        for msg in dumped:
            if isinstance(msg, dict) and label in self._active_runs and self._active_runs[label]:
                usage = msg.get("usage") or {}
                stats = self._active_runs[label][-1]
                for key in stats:
                    value = usage.get(key)
                    if isinstance(value, int):
                        stats[key] += value
            self._log("llm_message", label=label, turn=self.turn or None, message=msg)

    def log_event(self, event: str, **kwargs: Any) -> None:
        self._log(event, turn=self.turn or None, **kwargs)

    def _log(self, event: str, **kwargs: Any) -> None:
        entry = {
            "time": datetime.now().isoformat(),
            "event": event,
            **{key: value for key, value in kwargs.items() if value is not None},
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()


# ============================================================
# Discovery & sidebar metadata
# ============================================================

_PC_SCENARIO_CACHE: dict[str, tuple[str, str]] | None = None


def _pc_scenario_lookup() -> dict[str, tuple[str, str]]:
    """Map a scenario's default PC id to (scenario id, scenario title), for logs predating scenario logging."""
    global _PC_SCENARIO_CACHE
    if _PC_SCENARIO_CACHE is None:
        lookup: dict[str, tuple[str, str]] = {}
        for scenario in list_scenarios():
            manifest = read_manifest(scenario)
            pc = manifest.get("pc")
            if pc:
                lookup[pc] = (scenario, manifest.get("title", scenario))
        _PC_SCENARIO_CACHE = lookup
    return _PC_SCENARIO_CACHE


def _resolve_scenario(
    character_id: str | None, scenario: str | None, scenario_title: str | None
) -> tuple[str | None, str | None]:
    if scenario:
        return scenario, scenario_title or scenario
    if character_id:
        match = _pc_scenario_lookup().get(character_id)
        if match:
            return match
    return None, None


def _read_header(path: Path) -> dict[str, Any]:
    """Cheap line-by-line scan for sidebar metadata, skipping the expensive per-entry normalization."""
    header: dict[str, Any] = {
        "character_id": None,
        "max_turns": None,
        "scenario": None,
        "scenario_title": None,
        "turns_completed": None,
    }
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = entry.get("event")
            if event == "game_session_started":
                header["character_id"] = entry.get("character_id")
                header["max_turns"] = entry.get("max_turns")
                header["scenario"] = entry.get("scenario")
                header["scenario_title"] = entry.get("scenario_title")
            elif event == "turn_started":
                header["turns_completed"] = entry.get("turn")
            elif event == "game_session_finished":
                header["turns_completed"] = entry.get("turns_completed", header["turns_completed"])
    return header


def discover_log_files(path: Path) -> list[Path]:
    target = Path(path)
    if target.is_file():
        return [target]
    if not target.exists():
        return []
    return sorted(
        (candidate for candidate in target.glob("*.log") if candidate.is_file()),
        key=lambda candidate: candidate.name,
        reverse=True,
    )


def list_logs(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in discover_log_files(path):
        stat = candidate.stat()
        header = _read_header(candidate)
        scenario, scenario_title = _resolve_scenario(
            header["character_id"], header["scenario"], header["scenario_title"]
        )
        result.append(
            {
                "name": candidate.name,
                "path": str(candidate),
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "character_id": header["character_id"],
                "scenario": scenario,
                "scenario_title": scenario_title,
                "turns_completed": header["turns_completed"],
                "max_turns": header["max_turns"],
            }
        )
    return result


# ============================================================
# Session loading
# ============================================================

def load_session(path: Path) -> dict[str, Any]:
    log_path = Path(path)
    entries = _load_entries(log_path)
    runs = _build_runs(entries)
    turns = sorted({entry["turn"] for entry in entries if entry.get("turn") is not None})
    event_counts = Counter(entry["event"] for entry in entries)

    started = next((entry for entry in entries if entry["event"] == "game_session_started"), None)
    finished = next((entry for entry in reversed(entries) if entry["event"] == "game_session_finished"), None)
    first_time = _parse_time(entries[0].get("time")) if entries else None
    last_time = _parse_time(entries[-1].get("time")) if entries else None
    duration_s = round((last_time - first_time).total_seconds(), 3) if first_time and last_time else None

    character_id = started["raw"].get("character_id") if started else None
    scenario, scenario_title = _resolve_scenario(
        character_id,
        started["raw"].get("scenario") if started else None,
        started["raw"].get("scenario_title") if started else None,
    )

    return {
        "file": log_path.name,
        "path": str(log_path),
        "summary": {
            "character_id": character_id,
            "scenario": scenario,
            "scenario_title": scenario_title,
            "max_turns": started["raw"].get("max_turns") if started else None,
            "turns_completed": finished["raw"].get("turns_completed") if finished else (max(turns) if turns else 0),
            "started_at": entries[0].get("time") if entries else None,
            "finished_at": entries[-1].get("time") if entries else None,
            "duration_s": duration_s,
        },
        "turns": turns,
        "event_types": sorted(event_counts.keys()),
        "event_counts": dict(event_counts),
        "stats": {
            "event_count": len(entries),
            "run_count": len(runs),
            "llm_message_count": event_counts.get("llm_message", 0),
            "error_count": event_counts.get("parse_error", 0),
        },
        "runs": runs,
        "entries": entries,
    }


def _load_entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                entries.append(
                    {
                        "line": line_number,
                        "time": None,
                        "display_time": "unparsed",
                        "event": "parse_error",
                        "turn": None,
                        "label": None,
                        "summary": f"Could not parse log line {line_number}: {exc.msg}",
                        "excerpt": _compact(line, 280),
                        "message_kind": "parse_error",
                        "tool_calls": [],
                        "tool_returns": [],
                        "raw": {"line": line, "error": exc.msg},
                    }
                )
                continue
            entries.append(_normalize_entry(entry, line_number))
    return entries


def _normalize_entry(entry: dict[str, Any], line_number: int) -> dict[str, Any]:
    digest = _digest_message(entry.get("message")) if entry.get("event") == "llm_message" else None
    return {
        "line": line_number,
        "time": entry.get("time"),
        "display_time": _display_time(entry.get("time")),
        "event": entry.get("event", "unknown"),
        "turn": entry.get("turn"),
        "label": entry.get("label"),
        "summary": _summarize_entry(entry, digest),
        "excerpt": _compact(digest["text"], 320) if digest and digest["text"] else None,
        "message_kind": digest["kind"] if digest else None,
        "tool_calls": digest["tool_calls"] if digest else [],
        "tool_returns": digest["tool_returns"] if digest else [],
        "usage": digest["usage"] if digest else None,
        "raw": entry,
        "raw_pretty": _format_raw_payload(entry, digest),
    }


_TIMING_KEYS = ("prefill_tokens", "prefill_s", "prefill_tps", "decode_tokens", "decode_s", "decode_tps")


def _build_runs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    open_runs: dict[str, list[dict[str, Any]]] = {}

    for entry in entries:
        event = entry["event"]
        label = entry.get("label")
        if event == "agent_run_started" and label:
            run = {
                "id": f"run-{len(runs) + 1}",
                "label": label,
                "turn": entry.get("turn"),
                "started_at": entry.get("time"),
                "finished_at": None,
                "elapsed_s": None,
                "message_count": 0,
                "tool_calls": [],
                "message_kinds": [],
                "tokens": {},
                "output_tokens_per_s": None,
                "timings": None,
            }
            runs.append(run)
            open_runs.setdefault(label, []).append(run)
            continue

        if event == "llm_message" and label and label in open_runs and open_runs[label]:
            run = open_runs[label][-1]
            run["message_count"] += 1
            run["tool_calls"] = _unique([*run["tool_calls"], *entry.get("tool_calls", [])])
            message_kind = entry.get("message_kind")
            if message_kind:
                run["message_kinds"] = _unique([*run["message_kinds"], message_kind])
            usage = entry.get("usage")
            if usage:
                for key, value in usage.items():
                    run["tokens"][key] = run["tokens"].get(key, 0) + value
            continue

        if event == "agent_run_finished" and label and label in open_runs and open_runs[label]:
            run = open_runs[label].pop()
            run["finished_at"] = entry.get("time")
            run["elapsed_s"] = entry["raw"].get("elapsed_s")
            run["output_tokens_per_s"] = entry["raw"].get("output_tokens_per_s")
            timings = {key: entry["raw"][key] for key in _TIMING_KEYS if entry["raw"].get(key) is not None}
            run["timings"] = timings or None

    return runs


def _summarize_entry(entry: dict[str, Any], digest: dict[str, Any] | None) -> str:
    event = entry.get("event", "unknown")
    if event == "game_session_started":
        return f"Session started for {entry.get('character_id', 'unknown')} with max {entry.get('max_turns', '?')} turns."
    if event == "game_session_finished":
        return f"Session finished after {entry.get('turns_completed', 0)} turns."
    if event == "turn_started":
        return f"Turn {entry.get('turn')} started."
    if event == "agent_run_started":
        return f"{entry.get('label', 'agent')} started."
    if event == "agent_run_finished":
        elapsed = entry.get("elapsed_s")
        if elapsed is None:
            return f"{entry.get('label', 'agent')} finished."
        prefill_tps, decode_tps = entry.get("prefill_tps"), entry.get("decode_tps")
        if prefill_tps is not None or decode_tps is not None:
            suffix = f" · prefill {prefill_tps} tok/s · decode {decode_tps} tok/s"
        else:
            throughput = entry.get("output_tokens_per_s")
            suffix = f" · ~{throughput} output tok/s" if throughput is not None else ""
        return f"{entry.get('label', 'agent')} finished in {elapsed}s{suffix}."
    if event == "llm_message" and digest:
        prefix = (digest["kind"] or "llm").replace("_", " ")
        if digest["tool_calls"]:
            return f"{prefix} issued tool call: {', '.join(digest['tool_calls'])}."
        if digest["tool_returns"]:
            return f"{prefix} received tool return: {', '.join(digest['tool_returns'])}."
        if digest["text"]:
            return f"{prefix}: {_compact(digest['text'], 160)}"
        return prefix
    if event == "resolved":
        return f"{entry.get('tool', 'tool')} resolved: {_compact(str(entry.get('result', '')), 160)}"
    return event.replace("_", " ")


# ============================================================
# Message digestion — structured dicts (new logs) or repr strings (legacy logs)
# ============================================================

_TOOL_CALL_RE = re.compile(r"ToolCallPart\(tool_name='([^']+)'")
_TOOL_RETURN_RE = re.compile(r"ToolReturnPart\(tool_name='([^']+)'")
_USAGE_RE = re.compile(
    r"usage=RequestUsage\(input_tokens=(?P<input_tokens>\d+)(?:, cache_read_tokens=(?P<cache_read_tokens>\d+))?(?:, output_tokens=(?P<output_tokens>\d+))?",
    re.S,
)
_USAGE_KEYS = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")


def _digest_message(message: Any) -> dict[str, Any]:
    """Extract kind, tool names, usage, searchable text, and a pretty rendering from one logged message."""
    if isinstance(message, dict):
        return _digest_structured(message)
    text = _stringify(message)
    return _digest_legacy(text)


def _digest_structured(message: dict[str, Any]) -> dict[str, Any]:
    parts = message.get("parts") or []
    tool_calls = _unique([p["tool_name"] for p in parts if p.get("part_kind") == "tool-call"])
    tool_returns = _unique([p["tool_name"] for p in parts if p.get("part_kind") == "tool-return"])

    if message.get("kind") == "response":
        kind = "model_response"
    elif tool_returns and all(p.get("part_kind") in ("tool-return", "retry-prompt") for p in parts):
        kind = "tool_return"
    else:
        kind = "model_request"

    usage = {
        key: value
        for key, value in (message.get("usage") or {}).items()
        if key in _USAGE_KEYS and isinstance(value, int) and value
    }

    text_chunks = []
    for part in parts:
        content = part.get("content") if isinstance(part.get("content"), str) else part.get("args")
        if isinstance(content, str) and content:
            text_chunks.append(content)

    return {
        "kind": kind,
        "tool_calls": tool_calls,
        "tool_returns": tool_returns,
        "usage": usage or None,
        "text": " ".join(text_chunks),
        "pretty": _render_structured(message, parts, usage),
    }


def _render_structured(message: dict[str, Any], parts: list[dict[str, Any]], usage: dict[str, int]) -> str:
    title = "ModelResponse" if message.get("kind") == "response" else "ModelRequest"
    meta = [str(message[key]) for key in ("model_name", "provider_name", "finish_reason") if message.get(key)]
    lines = [f"{title} ({', '.join(meta)})" if meta else title]

    if usage:
        lines.append("usage: " + " · ".join(f"{key.removesuffix('_tokens')} {value}" for key, value in usage.items()))

    instructions = message.get("instructions")
    if instructions:
        lines.append("instructions:")
        lines.extend(_indent_lines(str(instructions).splitlines() or [""]))

    for part in parts:
        part_kind = part.get("part_kind", "part")
        tool_name = part.get("tool_name")
        lines.append(f"- {part_kind}" + (f": {tool_name}" if tool_name else ""))
        content = part.get("content")
        args = part.get("args")
        if isinstance(args, str) and args:
            lines.extend(_indent_lines(_format_jsonish(args).splitlines() or [""]))
        elif args is not None:
            lines.extend(_indent_lines(json.dumps(args, indent=2, ensure_ascii=False).splitlines()))
        if isinstance(content, str) and content:
            lines.extend(_indent_lines(content.splitlines() or [""]))
        elif content is not None:
            lines.extend(_indent_lines(json.dumps(content, indent=2, ensure_ascii=False).splitlines()))
    return "\n".join(lines)


def _digest_legacy(message: str) -> dict[str, Any]:
    if message.startswith("ModelRequest("):
        kind = "model_request"
    elif message.startswith("ModelResponse("):
        kind = "model_response"
    elif "ToolReturnPart(" in message:
        kind = "tool_return"
    else:
        kind = "llm_message"

    usage_match = _USAGE_RE.search(message)
    usage = (
        {key: int(value) for key, value in usage_match.groupdict().items() if value is not None}
        if usage_match
        else None
    )

    return {
        "kind": kind,
        "tool_calls": _unique(_TOOL_CALL_RE.findall(message)),
        "tool_returns": _unique(_TOOL_RETURN_RE.findall(message)),
        "usage": usage,
        "text": message,
        "pretty": _decode_log_text(message),
    }


# ============================================================
# Formatting helpers
# ============================================================

def _display_time(value: Any) -> str:
    timestamp = _parse_time(value)
    if not timestamp:
        return "unknown"
    return timestamp.strftime("%H:%M:%S")


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


_WHITESPACE_RE = re.compile(r"\s+")


def _compact(text: str, limit: int) -> str:
    compacted = _WHITESPACE_RE.sub(" ", text).strip()
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _format_raw_payload(payload: dict[str, Any], digest: dict[str, Any] | None) -> str:
    lines: list[str] = []
    for key, value in payload.items():
        if key == "message" and digest:
            lines.append(f"{key}:")
            lines.extend(_indent_lines(digest["pretty"].splitlines() or [""]))
        else:
            lines.extend(_format_field(key, value))
    return "\n".join(lines)


def _format_field(name: str, value: Any) -> list[str]:
    if value is None:
        return [f"{name}: null"]

    if isinstance(value, (bool, int, float)):
        return [f"{name}: {value}"]

    if isinstance(value, str):
        if "\n" in value:
            return [f"{name}:", *_indent_lines(value.splitlines())]
        return [f"{name}: {value}"]

    rendered = json.dumps(value, indent=2, ensure_ascii=False)
    return [f"{name}:", *_indent_lines(rendered.splitlines())]


def _format_jsonish(value: str) -> str:
    text = value.strip()
    if text.startswith("{") or text.startswith("["):
        try:
            return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return value
    return value


def _decode_log_text(value: str) -> str:
    return (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\"", '"')
        .replace("\\'", "'")
    )


def _indent_lines(lines: list[str], prefix: str = "  ") -> list[str]:
    return [f"{prefix}{line}" if line else prefix.rstrip() for line in lines]
