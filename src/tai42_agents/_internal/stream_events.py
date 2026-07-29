"""A normalized event stream over a LangGraph tools-agent run.

``astream_tools_agent_events`` projects the raw LangGraph channel shapes
(``updates`` vs ``messages``) and per-provider quirks into a provider-agnostic
vocabulary:

* :class:`ReasoningStep`   — a chunk of the model's intermediate reasoning.
* :class:`ToolCallStep`    — a tool the agent decided to invoke (name + args).
* :class:`ToolResultStep`  — the value a tool call returned.
* :class:`MessageDelta`    — a token-level chunk of the final answer.
* :class:`MessageFinal`    — the assembled final answer.
* :class:`RunUsage`        — the run's token counts + model label.
* :class:`StructuredFinal` — a requested structured (response_format) output,
  validated against the requested format before emission.

A provider that omits a field (a reasoning block, usage, a tool-call id) simply
produces fewer events. But a chunk whose SHAPE is malformed — not a
``(message, metadata)`` pair, not a node->update mapping, or a node update value
that is neither a channel-write mapping (nor a list of them) nor a known benign
shape (``None``, or the ``__interrupt__`` tuple the resume path reads from the
snapshot) — raises ``ValueError`` rather than being skipped. A structured
terminal that does not conform to the requested ``response_format`` likewise
raises from validation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import StructuredTool

# A node may overwrite a reduced channel by returning ``Overwrite(value=...)``;
# the ``updates`` stream then yields the wrapper, so it must be unwrapped.
from langgraph.types import Overwrite
from tai42_contract.agent.events import (
    MessageDelta,
    MessageFinal,
    ReasoningStep,
    StreamEvent,
    StructuredFinal,
    ToolCallStep,
    ToolResultStep,
)
from tai42_kit.llm.runtime import validate_structured_output

from tai42_agents._internal.base_tool_agent import _build_agent_and_input
from tai42_agents._internal.text import text_of
from tai42_agents._internal.usage import usage_event


def _channel_value(value: Any) -> Any:
    """Unwrap a langgraph ``Overwrite`` channel write to its underlying value."""
    if isinstance(value, Overwrite):
        return value.value
    return value


# --------------------------------------------------------------------------
# Message-shape helpers (per-provider quirks live here)
# --------------------------------------------------------------------------


def _reasoning_text(message: AIMessage) -> str:
    """Extract the model's reasoning/thinking text from an ``AIMessage``, across
    the shapes providers use: Anthropic ``thinking`` content blocks, OpenAI
    ``reasoning`` summaries, and the generic
    ``additional_kwargs['reasoning_content']``. Returns "" when there is none."""
    parts: list[str] = []
    additional = getattr(message, "additional_kwargs", None)
    if isinstance(additional, dict):
        reasoning_content = additional.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            parts.append(reasoning_content)
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in (
                "thinking",
                "reasoning",
                "reasoning_content",
            ):
                text = block.get("thinking") or block.get("reasoning") or block.get("text") or ""
                if text:
                    parts.append(text)
    return "\n".join(part for part in parts if part)


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------


async def astream_tools_agent_events(
    system_message: str,
    user_message: list[str],
    tools: list[StructuredTool],
    llm_provider: str | None = None,
    checkpoint_provider: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    system_content_kwargs: dict[str, Any] | None = None,
    response_format: Any = None,
) -> AsyncIterator[StreamEvent]:
    """Run the tools agent and yield a normalized :class:`StreamEvent` stream.

    Runs ``agent.astream`` with ``stream_mode=["updates", "messages"]``: the
    ``updates`` channel surfaces each node's new messages — the model's
    reasoning blocks and tool calls, and the tools' results — and the
    ``messages`` channel surfaces token-level deltas of the final answer.
    Pass a ``thread_id`` in ``config['configurable']`` to resume a checkpointed
    conversation; omit it for a one-shot run. A ``response_format`` forces the
    run's structured output, which the projection surfaces as a terminal
    :class:`StructuredFinal`.

    Cancellation (``asyncio.CancelledError``) propagates out unchanged so the
    caller can do its own abort bookkeeping.
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
    async for event in aproject_agent_events(agent, messages, config, response_format=response_format):
        yield event


async def aproject_agent_events(
    agent: Any,
    agent_input: Any,
    config: dict[str, Any],
    response_format: Any = None,
) -> AsyncIterator[StreamEvent]:
    """Project a compiled LangGraph agent's ``astream`` into :class:`StreamEvent`s.

    Provider- and harness-agnostic: works for any ``create_agent`` /
    ``create_deep_agent`` compiled graph, already built (model bound,
    ``thread_id`` set). Pass the ``response_format`` the agent was built with so
    the terminal :class:`StructuredFinal` payload is validated against it (a
    non-conforming output raises rather than being emitted).

    For a deep agent, a subagent invocation surfaces as a ``task`` tool
    ToolCall/ToolResult pair; its internal steps stay inside it.

    ``MessageFinal`` is the concatenation of every ``messages``-channel text delta
    across the whole run, falling back to the last ``updates``-channel AIMessage
    text when nothing streamed. ``asyncio.CancelledError`` propagates out unchanged.
    """
    seen_tool_calls: set[str] = set()
    synthetic_call_count = 0
    answer_parts: list[str] = []
    last_update_text = ""
    structured_response: Any = None

    async for item in agent.astream(agent_input, config, stream_mode=["updates", "messages"]):
        # With a list ``stream_mode`` LangGraph yields ``(mode, chunk)``; anything
        # that is not that pair is a bare single-mode chunk, treated as an update.
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            mode, chunk = item
        else:
            mode, chunk = "updates", item

        if mode == "messages":
            # chunk == (AIMessageChunk, metadata): token deltas of the reply.
            if not (isinstance(chunk, tuple) and len(chunk) == 2):
                raise ValueError(f"messages-mode stream chunk is not a (message, metadata) pair: {chunk!r}")
            message_chunk, _metadata = chunk
            if isinstance(message_chunk, AIMessageChunk):
                delta = text_of(message_chunk)
                if delta:
                    answer_parts.append(delta)
                    yield MessageDelta(text=delta)
            continue

        # mode == "updates": chunk == {node_name: {"messages": [...], ...}, ...}
        if not isinstance(chunk, dict):
            raise ValueError(f"updates-mode stream chunk is not a node->update mapping: {chunk!r}")
        # A node update value is normally a channel-write mapping (or a list of them
        # when a node writes the same channel twice). ``None`` (node wrote nothing)
        # and the ``__interrupt__`` tuple (read from the snapshot by the resume path)
        # are benign and skipped; any other shape raises.
        normalized_updates: list[dict[str, Any]] = []
        for node, update in chunk.items():
            if update is None or node == "__interrupt__":
                continue
            for one in update if isinstance(update, list) else [update]:
                if not isinstance(one, dict):
                    raise ValueError(f"updates-mode node update for {node!r} is not a mapping: {one!r}")
                normalized_updates.append(one)
        for update in normalized_updates:
            # Structured output surfaces on the updates channel; keep the latest.
            if update.get("structured_response") is not None:
                structured_response = update["structured_response"]
            for message in _channel_value(update.get("messages")) or []:
                if isinstance(message, AIMessage):
                    reasoning = _reasoning_text(message)
                    if reasoning:
                        yield ReasoningStep(text=reasoning)
                    text = text_of(message)
                    if text:
                        last_update_text = text
                    for tool_call in getattr(message, "tool_calls", None) or []:
                        call_id = tool_call.get("id")
                        if not call_id:
                            # Synthesize an id when the provider omits one; the
                            # ``__synthetic_tool_call_`` prefix is outside every
                            # provider's id namespace and the counter keeps it unique.
                            call_id = f"__synthetic_tool_call_{synthetic_call_count}"
                            synthetic_call_count += 1
                        if call_id in seen_tool_calls:
                            continue
                        seen_tool_calls.add(call_id)
                        yield ToolCallStep(
                            tool=tool_call.get("name", ""),
                            args=tool_call.get("args", {}) or {},
                            call_id=call_id,
                        )
                    usage = usage_event(message)
                    if usage is not None:
                        yield usage
                elif isinstance(message, ToolMessage):
                    yield ToolResultStep(
                        tool=getattr(message, "name", "") or "",
                        call_id=getattr(message, "tool_call_id", "") or "",
                        result=getattr(message, "content", ""),
                        is_error=getattr(message, "status", None) == "error",
                    )

    final_text = "".join(answer_parts).strip()
    if not final_text:
        # No token-streamed answer: fall back to the last AIMessage text so the
        # stream still ends with a MessageFinal.
        final_text = last_update_text.strip()
    if final_text:
        yield MessageFinal(text=final_text)

    if structured_response is not None:
        if response_format is not None:
            structured_response = validate_structured_output(structured_response, response_format)
        yield StructuredFinal(data=structured_response)
