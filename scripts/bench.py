"""Benchmark one game run against the local llama-server.

Plays a scenario for N turns with a tuning profile from config/llama.yaml,
samples http://localhost:<port>/metrics and GPU memory while it runs, then folds
in the exact per-run phase timings from the session log. Appends one JSON record
per run to logs/bench.jsonl so profiles can be compared.

Usage:
    uv run python scripts/bench.py --profile default --scenario missing-brother --turns 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agents.server import load_profile, read_metrics  # noqa: E402
from src.interface.scorecard import _score  # noqa: E402
from src.interface.session_log import DEFAULT_LOG_DIR, discover_log_files, load_session  # noqa: E402

BENCH_LOG = REPO / "logs" / "bench.jsonl"
GAME_OUT_DIR = REPO / "logs" / "bench"


def _fetch_props(port: int) -> dict | None:
    """Grab the server's self-reported model configuration once it is reachable."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/props", timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def _gpu_memory_mb() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def _phase_rates(metrics: dict | None) -> dict:
    """Convert cumulative /metrics counters into per-phase throughput."""
    if not metrics:
        return {}
    prefill_s = metrics.get("prompt_seconds", 0.0)
    decode_s = metrics.get("predicted_seconds", 0.0)
    return {
        "prefill_tokens": int(metrics.get("prompt_tokens", 0)),
        "prefill_s": round(prefill_s, 1),
        "prefill_tps": round(metrics.get("prompt_tokens", 0) / prefill_s, 1) if prefill_s else None,
        "decode_tokens": int(metrics.get("predicted_tokens", 0)),
        "decode_s": round(decode_s, 1),
        "decode_tps": round(metrics.get("predicted_tokens", 0) / decode_s, 1) if decode_s else None,
    }


def _session_stats(started: float) -> dict | None:
    """Score the session log this run produced (exact per-run timings, cache reuse)."""
    for path in discover_log_files(DEFAULT_LOG_DIR):
        if path.stat().st_mtime >= started:
            score = _score(load_session(path))
            keys = (
                "file", "turns_completed", "agent_runs", "failed_resolutions",
                "prefill_tps", "decode_tps", "prefill_s", "decode_s", "server_busy_s",
                "cache_reused_runs", "timed_runs", "input_tokens", "output_tokens",
                "output_tokens_per_s",
            )
            return {key: score[key] for key in keys}
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a game run against the local llama-server.")
    parser.add_argument("--scenario", default="missing-brother")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--profile", default=os.getenv("LLM_PROFILE", "default"))
    parser.add_argument("--note", default="", help="Free-form tag stored with the record.")
    args = parser.parse_args()

    profile_name, config = load_profile(args.profile)
    port = int(os.getenv("LLM_PORT", "1234"))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    GAME_OUT_DIR.mkdir(parents=True, exist_ok=True)
    game_out = GAME_OUT_DIR / f"{stamp}_{profile_name}.out"

    print(f"bench: profile={profile_name} scenario={args.scenario} turns={args.turns} (game output -> {game_out})", flush=True)
    started = time.time()
    with game_out.open("w", encoding="utf-8") as sink:
        process = subprocess.Popen(
            ["uv", "run", "infinite-dnd", "--scenario", args.scenario, "--turns", str(args.turns)],
            env={**os.environ, "LLM_PROFILE": profile_name},
            cwd=REPO, stdout=sink, stderr=subprocess.STDOUT,
        )
        metrics = props = None
        max_vram = 0
        while process.poll() is None:
            time.sleep(1)
            if snapshot := read_metrics():
                metrics = snapshot
                props = props or _fetch_props(port)
            if (vram := _gpu_memory_mb()) is not None:
                max_vram = max(max_vram, vram)
    wall_s = round(time.time() - started, 1)

    session = _session_stats(started)
    record = {
        "ts": stamp,
        "profile": profile_name,
        "note": args.note,
        "scenario": args.scenario,
        "turns": args.turns,
        "exit_code": process.returncode,
        "wall_s": wall_s,
        "max_vram_mb": max_vram or None,
        "config": config,
        "server": {
            "model": Path(props["model_path"]).name if props and props.get("model_path") else None,
            "n_ctx": (props or {}).get("default_generation_settings", {}).get("n_ctx"),
            "total_slots": (props or {}).get("total_slots"),
        },
        "metrics": _phase_rates(metrics),
        "session": session,
    }
    BENCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with BENCH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    phase = session or record["metrics"]
    print(f"\n=== bench result ({profile_name}) ===")
    print(f"  exit {process.returncode}   wall {wall_s}s   peak vram {max_vram or '?'} MB")
    if phase:
        print(f"  prefill {phase.get('prefill_tps')} tok/s over {phase.get('prefill_s')}s")
        print(f"  decode  {phase.get('decode_tps')} tok/s over {phase.get('decode_s')}s")
    if session:
        print(f"  kv-cache reuse {session['cache_reused_runs']}/{session['timed_runs']} runs"
              f"   turns {session['turns_completed']}   log {session['file']}")
    print(f"  appended to {BENCH_LOG.relative_to(REPO)}")
    return 0 if process.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
