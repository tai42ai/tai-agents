"""The tools-agent factory plus its invoke and raw-stream faces.

``_build_agent_and_input`` compiles a LangGraph ``create_agent`` graph and builds
its input messages and run config. ``ainvoke_tools_agent`` runs it once and
returns the user text plus per-call usage; ``astream_tools_agent`` yields the raw
LangGraph chunks (the normalized event projection lives in ``stream_events``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from tai42_kit.llm.checkpoint.checkpoint_registry import checkpoint_registry
from tai42_kit.llm.middleware.context_overflow import context_overflow_middlewares
from tai42_kit.llm.models import get_llm_async
from tai42_kit.llm.runtime import build_agent_input, build_user_output, extract_structured_output
from tai42_kit.llm.settings import llm_provider_settings, llm_settings
from tai42_kit.logging.settings import logging_settings

from tai42_agents._internal.config_util import init_langgraph_config
from tai42_agents._internal.usage import AgentInvokeResult, aggregate_usage


async def _build_agent_and_input(
    system_message: str,
    user_message: list[str],
    tools: list[StructuredTool],
    llm_provider: str | None = None,
    checkpoint_provider: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    system_content_kwargs: dict[str, Any] | None = None,
    response_format: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Compile the tools agent and build its input messages and run config.

    The model resolves from the kit LLM settings (``llm_kwargs`` override the
    defaults), the checkpointer from the checkpoint registry, and the
    context-overflow middleware is attached. A ``response_format`` (JSON-Schema
    dict or pydantic class) forces the structured output onto
    ``state["structured_response"]``; ``None`` keeps free-form text. Returns
    ``(agent, messages, config)``.
    """
    llm_provider = llm_provider or llm_provider_settings().llm
    llm = await get_llm_async(provider=llm_provider, **llm_settings().with_fallbacks(llm_kwargs or {}))

    checkpoint_provider = checkpoint_provider or llm_provider_settings().checkpoint
    checkpointer = await checkpoint_registry().get_checkpointer(
        provider=checkpoint_provider,
        conn_string=llm_provider_settings().checkpoint_conn_string,
    )

    agent = create_agent(
        llm,
        tools=tools,
        checkpointer=checkpointer,
        middleware=context_overflow_middlewares(),
        debug=logging_settings().is_enabled_for("DEBUG"),
        response_format=response_format,
    )

    config = init_langgraph_config(config)
    messages = build_agent_input(
        *user_message,
        system_message=system_message,
        system_content_kwargs=system_content_kwargs,
    )
    return agent, messages, config


async def ainvoke_tools_agent(
    system_message: str,
    user_message: list[str],
    tools: list[StructuredTool],
    llm_provider: str | None = None,
    checkpoint_provider: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    system_content_kwargs: dict[str, Any] | None = None,
    response_format: Any = None,
) -> AgentInvokeResult:
    """Invoke the tools agent and return the user output, per-call usage
    aggregated from every AIMessage in the run state, and the structured response.

    ``.structured`` holds the forced structured output when a ``response_format``
    was requested — validated against it, raising loudly if missing or
    non-conforming — and is ``None`` otherwise.
    """
    agent, messages, config = await _build_agent_and_input(
        system_message,
        user_message,
        tools,
        llm_provider,
        checkpoint_provider,
        llm_kwargs,
        config,
        system_content_kwargs=system_content_kwargs,
        response_format=response_format,
    )
    state = await agent.ainvoke(messages, config)
    structured = extract_structured_output(state, response_format) if response_format is not None else None
    return AgentInvokeResult(
        output=build_user_output(state),
        usage=aggregate_usage(state),
        structured=structured,
    )


async def astream_tools_agent(
    system_message: str,
    user_message: list[str],
    tools: list[StructuredTool],
    llm_provider: str | None = None,
    checkpoint_provider: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    stream_mode: str = "values",
    system_content_kwargs: dict[str, Any] | None = None,
    response_format: Any = None,
) -> AsyncIterator[Any]:
    """Run the tools agent and yield the raw LangGraph ``astream`` chunks for the
    requested ``stream_mode``; the caller decodes the channel shapes. A
    ``response_format`` forces structured output onto the ``structured_response``
    state channel."""
    agent, messages, config = await _build_agent_and_input(
        system_message,
        user_message,
        tools,
        llm_provider,
        checkpoint_provider,
        llm_kwargs,
        config,
        system_content_kwargs=system_content_kwargs,
        response_format=response_format,
    )
    async for chunk in agent.astream(messages, config, stream_mode=stream_mode):
        yield chunk
