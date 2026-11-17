import os
from typing import cast
from openai.types.shared import ReasoningEffort

from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider


# Each OpenAI-compatible server spells "don't think" differently.
_NO_THINKING = {
    "deepseek": {"thinking": {"type": "disabled"}},
    "ollama": {"reasoning_effort": "none"},
}


# The local model often fumbles one tool call (text instead of a call, a stray
# argument) and fixes it when told; one retry turned too many of those into a
# skipped tick.
OUTPUT_RETRIES = 3


class _DropEmptyReplies(WrapperModel):
    """After an empty model reply pydantic-ai resends the turn with that reply as a
    null-content assistant message, which Ollama rejects with a 400. Drop it."""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        kept = [
            m for m in messages if not (isinstance(m, ModelResponse) and not m.parts)
        ]
        return await super().request(kept, model_settings, model_request_parameters)


def create_model() -> Model:
    """Create a Model from environment variables.

    LLM_PROVIDER=anthropic uses the Claude API (ANTHROPIC_API_KEY); "openai"
    uses the real OpenAI API (OPENAI_API_KEY) with a configurable reasoning
    effort; anything else uses an OpenAI-compatible endpoint (local by
    default). Thinking is disabled for DeepSeek and Ollama so required tool
    calls work and local turns stay fast.
    """
    provider = os.getenv("LLM_PROVIDER", "local")
    if provider == "anthropic":
        return AnthropicModel(
            os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            provider=AnthropicProvider(),
        )
    if provider == "openai":
        effort = os.getenv("OPENAI_REASONING_EFFORT", "medium")
        if effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError(f"Unsupported OPENAI_REASONING_EFFORT: {effort!r}")
        return OpenAIChatModel(
            os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")),
            settings=OpenAIChatModelSettings(
                openai_reasoning_effort=cast(ReasoningEffort, effort),
            ),
        )
    return _DropEmptyReplies(
        OpenAIChatModel(
            os.getenv("LLM_MODEL", ""),
            provider=OpenAIProvider(
                base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
                api_key=os.getenv("LLM_API_KEY", "not-needed"),
            ),
            settings=OpenAIChatModelSettings(extra_body=_NO_THINKING[provider])
            if provider in _NO_THINKING
            else None,
        )
    )


_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(template_path: str, **kwargs) -> str:
    """Render a Jinja2 template relative to src/agents/."""
    return _env.get_template(template_path).render(**kwargs).strip()
