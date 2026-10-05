"""Manages a local llama.cpp server subprocess."""

import os
import subprocess
import time
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml
from dotenv import load_dotenv
load_dotenv()


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "llama.yaml"

# llama-server /metrics counters that split a request into its two phases:
# prompt processing (prefill) and token generation (decode).
_METRIC_KEYS = {
    "llamacpp:prompt_tokens_total": "prompt_tokens",
    "llamacpp:prompt_seconds_total": "prompt_seconds",
    "llamacpp:tokens_predicted_total": "predicted_tokens",
    "llamacpp:tokens_predicted_seconds_total": "predicted_seconds",
}


def _metrics_url() -> str:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    port = urllib.parse.urlparse(base_url).port or 1234
    return f"http://localhost:{port}/metrics"


def read_metrics() -> dict[str, float] | None:
    """Snapshot llama-server's cumulative prefill/decode counters, or None when unreachable."""
    try:
        with urllib.request.urlopen(_metrics_url(), timeout=2) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return None
    values: dict[str, float] = {}
    for line in text.splitlines():
        name, _, value = line.rpartition(" ")
        if name in _METRIC_KEYS:
            try:
                values[_METRIC_KEYS[name]] = float(value)
            except ValueError:
                continue
    return values or None


def load_profile(name: str | None = None) -> tuple[str, dict]:
    """Resolve a tuning profile from config/llama.yaml, merged over `default`."""
    name = name or os.getenv("LLM_PROFILE", "default")
    profiles = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    if name not in profiles:
        raise RuntimeError(f"Unknown LLM_PROFILE {name!r}; available: {', '.join(profiles)}")
    merged = dict(profiles["default"])
    merged.update(profiles[name])
    return name, merged


def _flag_value(value: object) -> str:
    # YAML may parse unquoted on/off-style values as booleans; llama-server wants on/off.
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def _reasoning_args(model: str, reasoning: str) -> list[str]:
    """Build llama.cpp reasoning flags, including its Gemma 4 tool-call workaround."""
    if reasoning == "off" and "gemma-4" in model.lower():
        # With Gemma 4, --reasoning off makes llama.cpp emit an empty `start`
        # production in its tool-call grammar. Keeping the parser enabled with
        # a zero-token budget disables thinking without breaking that grammar.
        return ["--reasoning", "auto", "--reasoning-budget", "0"]
    return ["--reasoning", reasoning]


def _profile_args(profile: dict, model: str) -> list[str]:
    """Render a profile's keys as llama-server command-line flags."""
    args: list[str] = []
    for key, value in profile.items():
        if key == "reasoning":
            args.extend(_reasoning_args(model, _flag_value(value)))
        elif value is None:  # a bare `key:` renders as a valueless flag, e.g. --no-mmap
            args.append(f"--{key}")
        else:
            args.extend([f"--{key}", _flag_value(value)])
    return args


class LlamaServer:
    """manages a local llama-server subprocess."""

    def __init__(self, profile: str | None = None):
        model = os.getenv("LLM_MODEL")
        model_path = os.getenv("LLM_MODEL_PATH")
        if not model and not model_path:
            raise RuntimeError("LLM_MODEL or LLM_MODEL_PATH must be set")
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
        port = urllib.parse.urlparse(base_url).port or 1234
        self._health_url = f"http://localhost:{port}/health"

        profile_name, config = load_profile(profile)
        server_bin = os.getenv("LLM_SERVER_BIN", "llama-server")
        # -m loads a local file directly (no network); -hf resolves/downloads by repo id.
        model_arg = ["-m", model_path] if model_path else ["-hf", model]
        cmd = [
            server_bin, *model_arg, "--port", str(port),
            *_profile_args(config, model or model_path or ""),
            "--metrics",
            "--jinja",
            "--log-disable",
        ]
        logging.info(f"Starting LlamaServer (profile {profile_name!r}): {' '.join(cmd)}")
        self._process = subprocess.Popen(cmd)

        # Loads of 26B models can take 60s+ from disk; first-time HF downloads take longer.
        deadline = time.time() + 300
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup with code {self._process.returncode}")
            try:
                with urllib.request.urlopen(self._health_url, timeout=1) as response:
                    body = response.read().decode("utf-8", errors="replace")
                # Some builds answer 200 while the model is still loading and requests
                # would still 503; the body carries the real status.
                if '"ok"' in body:
                    return
            except (urllib.error.URLError, urllib.error.HTTPError):
                pass
            time.sleep(0.5)
        self._process.terminate()
        raise RuntimeError(f"llama-server did not become healthy in 300s: {self._health_url}")

    def stop(self):
        self._process.terminate()
        self._process.wait()

    def __enter__(self): return self
    def __exit__(self, *_): self.stop()
