import asyncio
import json
from functools import partial

import httpx
import pytest
from pydantic_ai import Agent, ToolOutput

from src.agents import utils


@pytest.mark.parametrize("provider", ["deepseek", "ollama", "local"])
def test_required_tool_request(monkeypatch, provider):
    """DeepSeek disables thinking; other compatible endpoints keep their defaults."""
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chat-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-test",
                                    "type": "function",
                                    "function": {
                                        "name": "done",
                                        "arguments": '{"response":"ok"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            monkeypatch.setattr(
                utils,
                "OpenAIProvider",
                partial(utils.OpenAIProvider, http_client=client),
            )
            agent = Agent(
                utils.create_model(), output_type=ToolOutput(str, name="done")
            )
            assert (await agent.run("Reply ok")).output == "ok"

    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0]["tool_choice"] == "required"
    if provider == "deepseek":
        assert requests[0]["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in requests[0]
    assert requests[0].get("reasoning_effort") == (
        "none" if provider == "ollama" else None
    )


def test_empty_reply_is_retried_without_a_null_assistant_message(monkeypatch):
    """Ollama 400s on `{"role": "assistant", "content": null}`; never send one."""
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        message = (
            {"role": "assistant", "content": ""}
            if len(requests) == 1
            else {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-test",
                        "type": "function",
                        "function": {"name": "done", "arguments": '{"response":"ok"}'},
                    }
                ],
            }
        )
        return httpx.Response(
            200,
            json={
                "id": "chat-test",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            },
        )

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "m")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            monkeypatch.setattr(
                utils,
                "OpenAIProvider",
                partial(utils.OpenAIProvider, http_client=client),
            )
            agent = Agent(
                utils.create_model(), output_type=ToolOutput(str, name="done")
            )
            assert (await agent.run("Reply ok")).output == "ok"

    asyncio.run(run())
    assert len(requests) == 2
    assert all(m.get("content") is not None for m in requests[1]["messages"])
