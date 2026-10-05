
import os

from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

def create_model() -> Model:
    """Create a Model from environment variables.

    LLM_PROVIDER=anthropic uses the Claude API (ANTHROPIC_API_KEY); "openai"
    uses the real OpenAI API (OPENAI_API_KEY) with a configurable reasoning
    effort; anything else (default, "local") uses the local
    OpenAI-compatible llama.cpp server.
    """
    provider = os.getenv("LLM_PROVIDER", "local")
    if provider == "anthropic":
        return AnthropicModel(
            os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            provider=AnthropicProvider(),
        )
    if provider == "openai":
        return OpenAIChatModel(
            os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")),
            settings=OpenAIChatModelSettings(
                openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium"),
            ),
        )
    return OpenAIChatModel(
        os.getenv("LLM_MODEL", ""),
        provider=OpenAIProvider(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("LLM_API_KEY", "not-needed"),
        ),
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
