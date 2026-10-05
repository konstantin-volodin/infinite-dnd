"""Dump the exact context sent to each agent for inspection."""

import asyncio
import json
import logging

from pydantic_ai import RunContext
from pydantic_ai.usage import RunUsage

from src.engine.state import StateManager
from src.agents.utils import create_model
from src.agents.character.agent import CharacterDeps, agent as character_agent
from src.agents.character.context import character_system, character_context
from src.agents.action_resolver.agent import agent as resolver_agent
from src.agents.action_resolver.context import (
    action_resolver_system,
    action_resolver_context,
)
from src.tests import LOG_DIR, stamp


def _function_tools(agent) -> dict:
    return {
        name: {
            "description": t.tool_def.description,
            "parameters": t.tool_def.parameters_json_schema,
        }
        for name, t in agent._function_toolset.tools.items()
    }


async def _prepared_output_tools(agent, deps) -> dict:
    ctx = RunContext(deps=deps, model=create_model(), usage=RunUsage(), agent=agent)
    prepared = agent._output_toolset.prepared(agent._prepare_output_tools)
    return {
        name: {
            "description": t.tool_def.description,
            "parameters": t.tool_def.parameters_json_schema,
        }
        for name, t in (await prepared.get_tools(ctx)).items()
    }


def _write(name: str, system: str, context: str, tools: dict) -> None:
    path = LOG_DIR / f"{stamp}-context-{name}.md"
    path.write_text(
        f"##### {name}\n\n"
        f"##### System Prompt\n{system}\n\n"
        f"##### Tools\n{json.dumps(tools, indent=2)}\n\n"
        f"##### Context\n{context}",
        encoding="utf-8",
    )
    logging.info(f"saved [{name}] context to {path}")


def dump_all() -> None:
    state = StateManager().init_state()

    for char in state.characters.values():
        _write(
            char.id,
            character_system(char),
            character_context(char, state),
            asyncio.run(
                _prepared_output_tools(
                    character_agent, CharacterDeps(char=char, state=state)
                )
            ),
        )

    char = next(iter(state.characters.values()))
    _write(
        "action-resolver",
        action_resolver_system(),
        action_resolver_context(
            char, state, description="carefully inspects the old fountain"
        ),
        _function_tools(resolver_agent),
    )
