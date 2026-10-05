"""Aggregate metrics across one or more recorded game sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.interface.session_log import DEFAULT_LOG_DIR, discover_log_files, load_session


def _score(session: dict) -> dict:
    entries = session["entries"]
    resolutions = [e for e in entries if e["event"] == "resolved"]
    failed = [e for e in resolutions if str(e["raw"].get("result", "")).startswith("Cannot")]
    snapshots = [e["raw"] for e in entries if e["event"] == "world_snapshot"]
    baseline = snapshots[0] if snapshots else {}
    final = snapshots[-1] if snapshots else {}

    tokens = {"input_tokens": 0, "output_tokens": 0}
    timings = {"prefill_tokens": 0, "prefill_s": 0.0, "decode_tokens": 0, "decode_s": 0.0}
    timed_runs = cache_reused_runs = 0
    for run in session["runs"]:
        for key in tokens:
            tokens[key] += run.get("tokens", {}).get(key, 0)
        for key in timings:
            timings[key] += (run.get("timings") or {}).get(key, 0)
        if run.get("timings"):
            timed_runs += 1
            # Prefilling fewer tokens than the prompt held means the KV cache served the rest.
            if run["timings"].get("prefill_tokens", 0) < run.get("tokens", {}).get("input_tokens", 0):
                cache_reused_runs += 1
    durations = [run["elapsed_s"] for run in session["runs"] if run.get("elapsed_s") is not None]
    total_run_s = sum(durations)

    return {
        "file": session["file"],
        "turns_completed": session["summary"]["turns_completed"] or 0,
        "max_turns": session["summary"]["max_turns"],
        "duration_s": session["summary"]["duration_s"],
        "resolutions": len(resolutions),
        "failed_resolutions": len(failed),
        "failure_rate": (len(failed) / len(resolutions)) if resolutions else 0.0,
        "quests_completed": final.get("quests_completed", 0) - baseline.get("quests_completed", 0),
        "quests_failed": final.get("quests_failed", 0) - baseline.get("quests_failed", 0),
        "locations_discovered": final.get("locations", 0) - baseline.get("locations", 0),
        "characters_added": final.get("characters", 0) - baseline.get("characters", 0),
        "xp_awarded": final.get("total_xp", 0) - baseline.get("total_xp", 0),
        "minutes_elapsed": final.get("minutes_elapsed", 0),
        "agent_runs": len(session["runs"]),
        "avg_run_s": (sum(durations) / len(durations)) if durations else None,
        "input_tokens": tokens["input_tokens"],
        "output_tokens": tokens["output_tokens"],
        "output_tokens_per_s": (tokens["output_tokens"] / total_run_s) if total_run_s else None,
        "prefill_tps": round(timings["prefill_tokens"] / timings["prefill_s"], 1) if timings["prefill_s"] else None,
        "decode_tps": round(timings["decode_tokens"] / timings["decode_s"], 1) if timings["decode_s"] else None,
        "prefill_s": round(timings["prefill_s"], 1),
        "decode_s": round(timings["decode_s"], 1),
        "server_busy_s": round(timings["prefill_s"] + timings["decode_s"], 1),
        "timed_runs": timed_runs,
        "cache_reused_runs": cache_reused_runs,
    }


def _print_report(score: dict) -> None:
    print(f"\n=== {score['file']} ===")
    print(f"  turns: {score['turns_completed']}/{score['max_turns']}   duration: {score['duration_s']}s")
    print(f"  resolutions: {score['resolutions']} ({score['failed_resolutions']} failed, {score['failure_rate']:.0%} failure rate)")
    print(f"  quests: +{score['quests_completed']} completed, +{score['quests_failed']} failed")
    print(f"  world growth: +{score['locations_discovered']} locations, +{score['characters_added']} characters")
    print(f"  xp awarded: {score['xp_awarded']}   in-game minutes: {score['minutes_elapsed']}")
    print(f"  agent runs: {score['agent_runs']}   avg latency: {score['avg_run_s']}")
    throughput = score["output_tokens_per_s"]
    speed = f"   ~{throughput:.2f} output tok/s end-to-end" if throughput is not None else ""
    print(f"  tokens: {score['input_tokens']} in / {score['output_tokens']} out{speed}")
    if score["prefill_tps"] is not None or score["decode_tps"] is not None:
        print(
            f"  server: prefill {score['prefill_tps']} tok/s over {score['prefill_s']}s"
            f"   decode {score['decode_tps']} tok/s over {score['decode_s']}s"
            f"   busy {score['server_busy_s']}s"
        )
        print(f"  kv-cache reuse: {score['cache_reused_runs']}/{score['timed_runs']} runs")


def _print_comparison(scores: list[dict]) -> None:
    fields = [
        "turns_completed", "failure_rate", "quests_completed", "xp_awarded",
        "locations_discovered", "avg_run_s", "output_tokens_per_s", "prefill_tps", "decode_tps",
        "server_busy_s", "cache_reused_runs", "input_tokens", "output_tokens",
    ]
    print(f"\n=== Comparison ({len(scores)} runs) ===")
    print("  {:<22}".format("metric") + "".join(f"{s['file'][:20]:>22}" for s in scores))
    for field in fields:
        row = "  {:<22}".format(field)
        for s in scores:
            value = s[field]
            row += f"{value:>22.0%}" if field == "failure_rate" else f"{str(value):>22}"
        print(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare metrics across recorded game sessions.")
    parser.add_argument("logs", nargs="*", help="Log files to score (default: 2 most recent in logs/).")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory to search when no logs are given.")
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.logs] or discover_log_files(Path(args.log_dir))[:2]
    if not paths:
        parser.error("No log files found.")
    for path in paths:
        if not path.exists():
            parser.error(f"Log not found: {path}")

    scores = [_score(load_session(path)) for path in paths]
    for score in scores:
        _print_report(score)
    if len(scores) > 1:
        _print_comparison(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
