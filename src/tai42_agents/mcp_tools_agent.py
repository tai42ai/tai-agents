"""The ``mcp_tools_agent`` — a tools agent whose tools come from an MCP server.

Given an MCP server configuration (the ``mcpServers`` shape), the agent opens a
``fastmcp`` client, converts the server's MCP tools into LangChain tools, and
delegates to the generic tools-agent event stream with the client held open for
the whole run. :meth:`astream` streams the contract ``StreamEvent`` taxonomy;
:meth:`run` drains it. When ``inject_env`` is set, only the names in
``env_allowlist`` are copied from ``os.environ`` into each server's ``env`` (the
server's own ``env`` wins on conflict) on a fresh copy of the config.

Security / trust model: ``mcp_config`` can define stdio servers (local command
execution), and ``env_allowlist`` is caller-supplied so it limits accidental
exposure, NOT a malicious caller. `mcp_tools_agent` is admin-curated — expose it
ONLY to trusted, access-controlled callers, NEVER to an agent processing untrusted
content. ``base_url``/``api_key`` in ``llm_kwargs`` route to a caller-chosen model
endpoint — same trust caveat.
"""

from __future__ import annotations

import asyncio
import copy
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Any, ClassVar

from fastmcp import Client
from pydantic import BaseModel, ConfigDict, Field
from tai42_contract.agent import Agent
from tai42_contract.agent.events import StreamEvent, StructuredFinal
from tai42_contract.app import tai42_app
from tai42_kit.clients.settings import mcp_client_settings
from tai42_kit.utils.lc.lc_util import mcp_tools_to_lc_tools

from tai42_agents._internal.config_util import build_run_config
from tai42_agents._internal.reject import (
    reject_blank_memory_keys,
    reject_unhonored,
    reject_untitled_response_format,
)
from tai42_agents._internal.render import render_message
from tai42_agents._internal.stream_events import astream_tools_agent_events

# Contract ``Agent.run`` parameters this agent cannot honor, mapped to the reason
# named in the raised error (the keys define this agent's unhonored set). Its tools
# come from ``mcp_config`` and it runs no sub-agent / skills / composition-strategy
# machinery. Honored params are explicit signature parameters, never listed here.
_UNHONORED_REASONS: dict[str, str] = {
    "tools": "its tools come from mcp_config, not caller-supplied live tools",
    "tool_names": "its tools come from mcp_config, not named client tools",
    "presets": "its tools come from mcp_config; it exposes no preset tools",
    "subagents": "it runs no sub-agent machinery",
    "skills": "it loads no skills backend",
    "inline_skills": "it loads no skills backend",
    "strategy": "it applies no composition strategy and will not silently ignore one",
    "interrupt_on": "its delegated graph never pauses for external input",
    "resume": "its delegated graph never interrupts, so there is no paused run to resume",
    "store_provider": "it wires no long-term store",
}
# The unhonored parameters whose unset default is an empty collection — a non-empty
# value is a rejected request. Every other unhonored parameter defaults to ``None``
# and is set when it is not ``None`` (so a falsy-but-present value still raises).
_UNHONORED_COLLECTION_PARAMS: frozenset[str] = frozenset(
    {"tools", "tool_names", "presets", "subagents", "skills", "inline_skills"}
)


class McpToolsAgentInput(BaseModel):
    """JSON tool-face parameters for the ``mcp_tools_agent``.

    ``response_format`` is a JSON-Schema dict with a required top-level ``"title"``
    (used as the structured-output name); when set, the delegated run forces the
    model to emit output matching it and :meth:`run` returns the validated
    structured object.

    ``extra="forbid"`` rejects any unknown key loudly at validation rather than
    letting a typo at the run door vanish silently.
    """

    model_config = ConfigDict(extra="forbid")

    mcp_config: dict[str, Any]
    response_format: dict[str, Any] | None = Field(
        default=None, description="JSON Schema of the forced structured output (needs a top-level 'title')."
    )
    system_message: str = ""
    user_message: str = ""
    system_message_id: str = ""
    user_message_id: str = ""
    system_message_kwargs: dict[str, Any] | None = None
    user_message_kwargs: dict[str, Any] | None = None
    inject_env: bool = False
    env_allowlist: list[str] | None = None
    llm_provider: str | None = None
    checkpoint_provider: str | None = None
    llm_kwargs: dict[str, Any] | None = None
    langgraph_config: dict[str, Any] | None = None


def _inject_env(mcp_config: dict[str, Any], env_allowlist: list[str]) -> dict[str, Any]:
    """Return a copy of ``mcp_config`` with the allow-listed ``os.environ`` values
    merged into each server's ``env``.

    Only the names in ``env_allowlist`` are copied through; the rest of the process
    environment is never injected. The caller's ``mcp_config`` is not mutated
    (modified server entries are deep-copied). A server's own ``env`` wins.
    """
    injected = {name: os.environ[name] for name in env_allowlist if name in os.environ}
    new_servers = {}
    for server_name, server_cfg in mcp_config["mcpServers"].items():
        new_cfg = copy.deepcopy(server_cfg)
        new_cfg["env"] = {**injected, **new_cfg.get("env", {})}
        new_servers[server_name] = new_cfg
    return {**mcp_config, "mcpServers": new_servers}


@tai42_app.agents.agent("mcp_tools_agent", tags={"agents"})
class McpToolsAgent(Agent):
    """Run a tools agent whose tools are loaded from an MCP configuration."""

    tool_name: ClassVar[str] = "mcp_tools_agent"
    tool_description: ClassVar[str] = (
        "Create and run a LangGraph tools agent whose tools are loaded from an MCP configuration. "
        "Security: `mcp_tools_agent` is admin-curated — expose it ONLY to trusted, access-controlled "
        "callers/agents, NEVER to an agent that processes untrusted content."
    )
    ToolInput: ClassVar[type[BaseModel]] = McpToolsAgentInput

    async def run(
        self,
        *,
        mcp_config: dict[str, Any],
        system_message: str = "",
        user_message: str = "",
        system_message_id: str = "",
        user_message_id: str = "",
        system_message_kwargs: dict[str, Any] | None = None,
        user_message_kwargs: dict[str, Any] | None = None,
        inject_env: bool = False,
        env_allowlist: list[str] | None = None,
        thread_id: str | None = None,
        resume_checkpoint_id: str | None = None,
        recursion_limit: int | None = None,
        llm_provider: str | None = None,
        checkpoint_provider: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        langgraph_config: dict[str, Any] | None = None,
        response_format: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Drain the MCP tools-agent stream and return its final value.

        ``thread_id`` / ``resume_checkpoint_id`` / ``recursion_limit`` are forwarded
        to :meth:`astream` (the memory keys overlay ``langgraph_config``'s
        ``configurable``, ``recursion_limit`` its top level).

        ``response_format`` forces the delegated run's structured output; its
        JSON-Schema dict must carry a top-level ``"title"``, and a run producing none
        raises loudly. Contract parameters with no seat on this agent are rejected
        loudly up front, in parity with :meth:`astream`.
        """
        reject_unhonored(
            "mcp_tools_agent.run", kwargs, _UNHONORED_REASONS, collection_params=_UNHONORED_COLLECTION_PARAMS
        )
        reject_blank_memory_keys("mcp_tools_agent.run", thread_id=thread_id, resume_checkpoint_id=resume_checkpoint_id)
        reject_untitled_response_format("mcp_tools_agent", response_format)
        return await self._drain(
            self.astream(
                mcp_config=mcp_config,
                system_message=system_message,
                user_message=user_message,
                system_message_id=system_message_id,
                user_message_id=user_message_id,
                system_message_kwargs=system_message_kwargs,
                user_message_kwargs=user_message_kwargs,
                inject_env=inject_env,
                env_allowlist=env_allowlist,
                thread_id=thread_id,
                resume_checkpoint_id=resume_checkpoint_id,
                recursion_limit=recursion_limit,
                llm_provider=llm_provider,
                checkpoint_provider=checkpoint_provider,
                llm_kwargs=llm_kwargs,
                langgraph_config=langgraph_config,
                response_format=response_format,
            ),
            response_format=response_format,
        )

    async def astream(
        self,
        *,
        mcp_config: dict[str, Any],
        system_message: str = "",
        user_message: str = "",
        system_message_id: str = "",
        user_message_id: str = "",
        system_message_kwargs: dict[str, Any] | None = None,
        user_message_kwargs: dict[str, Any] | None = None,
        inject_env: bool = False,
        env_allowlist: list[str] | None = None,
        thread_id: str | None = None,
        resume_checkpoint_id: str | None = None,
        recursion_limit: int | None = None,
        llm_provider: str | None = None,
        checkpoint_provider: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        langgraph_config: dict[str, Any] | None = None,
        response_format: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a tools-agent run whose tools are loaded from ``mcp_config``.

        The messages render from literal content or a template id, the MCP client
        stays open for the whole run, and the delegated event stream is projected
        event-by-event. The client's connect is bounded by
        ``mcp_client_settings().connect_timeout_seconds`` so a stalled server raises
        ``TimeoutError`` instead of hanging; the exit stays context-managed so the
        client is never leaked.

        ``inject_env=True`` with an empty ``env_allowlist`` is malformed (it would
        inject nothing) and raises ``ValueError``.

        ``thread_id`` / ``resume_checkpoint_id`` overlay ``langgraph_config``'s
        ``configurable`` and ``recursion_limit`` its top level. ``response_format``
        forces the delegated run's structured output, surfaced as a terminal
        :class:`StructuredFinal`; a run producing none raises loudly after the stream
        drains. Contract parameters with no seat here are rejected loudly, in parity
        with :meth:`run`.
        """
        reject_unhonored(
            "mcp_tools_agent.astream", kwargs, _UNHONORED_REASONS, collection_params=_UNHONORED_COLLECTION_PARAMS
        )
        reject_blank_memory_keys(
            "mcp_tools_agent.astream", thread_id=thread_id, resume_checkpoint_id=resume_checkpoint_id
        )
        reject_untitled_response_format("mcp_tools_agent", response_format)
        if inject_env:
            if not env_allowlist:
                raise ValueError(
                    "mcp_tools_agent: inject_env=True requires a non-empty env_allowlist "
                    "(injecting with nothing allow-listed would inject nothing)."
                )
            mcp_config = _inject_env(mcp_config, env_allowlist)

        system = await render_message(system_message, system_message_id, system_message_kwargs)
        user = await render_message(user_message, user_message_id, user_message_kwargs)

        config = build_run_config(langgraph_config, thread_id, resume_checkpoint_id, recursion_limit)
        # Bound only the client ENTER (connect + init): ``asyncio.wait_for`` cancels
        # it on expiry so fastmcp unwinds its own partial transport state, leaving
        # nothing to exit. The EXIT stays context-managed via the exit stack so a
        # client that DID enter is always closed. The per-tool call timeout lives on
        # the kit ``lc_util`` seam.
        timeout = mcp_client_settings().connect_timeout_seconds
        client = Client(mcp_config)
        try:
            await asyncio.wait_for(client.__aenter__(), timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"MCP client connect/init did not complete within {timeout}s (MCP_CLIENT_CONNECT_TIMEOUT_SECONDS)"
            ) from exc
        async with AsyncExitStack() as stack:
            stack.push_async_exit(client.__aexit__)
            tools = await mcp_tools_to_lc_tools(client)
            saw_structured = False
            async for event in astream_tools_agent_events(
                system_message=system,
                user_message=[user],
                tools=tools,
                llm_provider=llm_provider,
                checkpoint_provider=checkpoint_provider,
                llm_kwargs=llm_kwargs,
                config=config,
                response_format=response_format,
            ):
                if isinstance(event, StructuredFinal):
                    saw_structured = True
                yield event
            # A requested response_format that produced no StructuredFinal fails
            # loudly after the stream drains.
            if response_format is not None and not saw_structured:
                raise RuntimeError("agent run requested a response_format but produced no structured output")
