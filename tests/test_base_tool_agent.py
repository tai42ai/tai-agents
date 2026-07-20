"""Tests for the tools-agent factory + its invoke / raw-stream faces.

Every provider seam ``_build_agent_and_input`` reaches (LLM, checkpointer,
middleware, ``create_agent``) is monkeypatched to a scripted double, so the
factory's wiring — default-provider fallback, config init, message build, and
the ``ainvoke`` / ``astream`` faces — is exercised with no LLM, checkpointer, or
network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import StructuredTool
from tai42_kit.utils.data.json_schema_util import JsonSchemaValidationError

from tai42_agents._internal import base_tool_agent as bta
from tai42_agents._internal.usage import CallUsage


def _tool(name: str) -> StructuredTool:
    async def _run(**_: Any) -> str:
        return "ok"

    return StructuredTool.from_function(func=None, coroutine=_run, name=name, description="d")


def _patch_seams(
    monkeypatch: pytest.MonkeyPatch, *, state: Any = None, chunks: list[Any] | None = None
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        bta,
        "llm_provider_settings",
        lambda: SimpleNamespace(llm="def_llm", checkpoint="def_checkpoint", checkpoint_conn_string="cp-conn"),
    )
    monkeypatch.setattr(bta, "llm_settings", lambda: SimpleNamespace(with_fallbacks=lambda kwargs: dict(kwargs)))

    async def fake_get_llm(*, provider: str, **kwargs: Any) -> Any:
        captured["llm_provider"] = provider
        captured["llm_kwargs"] = kwargs
        return "llm-obj"

    monkeypatch.setattr(bta, "get_llm_async", fake_get_llm)

    async def fake_get_checkpointer(*, provider: str, conn_string: str) -> Any:
        captured["checkpoint"] = (provider, conn_string)
        return "checkpointer-obj"

    monkeypatch.setattr(bta, "checkpoint_registry", lambda: SimpleNamespace(get_checkpointer=fake_get_checkpointer))
    monkeypatch.setattr(bta, "context_overflow_middlewares", lambda: ["mw"])
    monkeypatch.setattr(bta, "logging_settings", lambda: SimpleNamespace(is_enabled_for=lambda level: level == "DEBUG"))
    monkeypatch.setattr(bta, "init_langgraph_config", lambda config: {"configurable": {"thread_id": "t"}})

    fake_agent = MagicMock()
    fake_agent.ainvoke = AsyncMock(return_value=state if state is not None else {"messages": []})

    async def fake_astream(messages: Any, config: Any, stream_mode: str) -> AsyncIterator[Any]:
        captured["stream"] = (messages, config, stream_mode)
        for chunk in chunks or []:
            yield chunk

    fake_agent.astream = fake_astream

    def fake_create_agent(
        llm: Any, *, tools: Any, checkpointer: Any, middleware: Any, debug: bool, response_format: Any = None
    ) -> Any:
        captured["create"] = {
            "llm": llm,
            "tools": tools,
            "checkpointer": checkpointer,
            "middleware": middleware,
            "debug": debug,
            "response_format": response_format,
        }
        return fake_agent

    monkeypatch.setattr(bta, "create_agent", fake_create_agent)
    return captured


async def _collect(agen: AsyncIterator[Any]) -> list[Any]:
    return [chunk async for chunk in agen]


class TestBuildAgentAndInput:
    def test_resolves_default_providers_and_wires_create_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_seams(monkeypatch)
        tool = _tool("search")

        agent, messages, config = asyncio.run(
            bta._build_agent_and_input("sys-prompt", ["hi"], [tool], llm_kwargs={"temperature": 0})
        )

        assert agent is not None
        assert config == {"configurable": {"thread_id": "t"}}
        # Default providers fell back to the settings values.
        assert captured["llm_provider"] == "def_llm"
        assert captured["checkpoint"] == ("def_checkpoint", "cp-conn")
        assert captured["llm_kwargs"] == {"temperature": 0}
        # create_agent wired with the resolved model, tools, checkpointer, middleware.
        assert captured["create"]["llm"] == "llm-obj"
        assert captured["create"]["tools"] == [tool]
        assert captured["create"]["checkpointer"] == "checkpointer-obj"
        assert captured["create"]["middleware"] == ["mw"]
        assert captured["create"]["debug"] is True
        # No response_format requested -> the text-behavior default is threaded through.
        assert captured["create"]["response_format"] is None
        # System prompt + user message threaded into the built input.
        roles = {message["role"]: message["content"] for message in messages["messages"]}
        assert roles["system"] == "sys-prompt"
        assert roles["user"] == "hi"

    def test_response_format_is_threaded_into_create_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_seams(monkeypatch)
        schema = {"title": "Answer", "type": "object", "properties": {"value": {"type": "integer"}}}
        asyncio.run(bta._build_agent_and_input("sys", ["hi"], [], response_format=schema))
        assert captured["create"]["response_format"] == schema

    def test_explicit_providers_override_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_seams(monkeypatch)
        asyncio.run(
            bta._build_agent_and_input("sys", ["hi"], [], llm_provider="my_llm", checkpoint_provider="my_checkpoint")
        )
        assert captured["llm_provider"] == "my_llm"
        assert captured["checkpoint"][0] == "my_checkpoint"


class TestAinvoke:
    def test_returns_user_output_and_aggregated_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = {"messages": ["state"]}
        _patch_seams(monkeypatch, state=state)
        usage = CallUsage(3, 2, "scripted")
        monkeypatch.setattr(bta, "build_user_output", lambda s: f"out:{s['messages'][0]}")
        monkeypatch.setattr(bta, "aggregate_usage", lambda s: usage)

        result = asyncio.run(bta.ainvoke_tools_agent("sys", ["hi"], [_tool("t")]))

        assert result.output == "out:state"
        assert result.usage is usage
        # No response_format requested -> no structured payload on the result.
        assert result.structured is None

    def test_returns_structured_response_when_requested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = {"messages": ["state"], "structured_response": {"value": 7}}
        _patch_seams(monkeypatch, state=state)
        monkeypatch.setattr(bta, "build_user_output", lambda s: "text")
        monkeypatch.setattr(bta, "aggregate_usage", lambda s: CallUsage(0, 0, None))

        schema = {"title": "Answer", "type": "object", "properties": {"value": {"type": "integer"}}}
        result = asyncio.run(bta.ainvoke_tools_agent("sys", ["hi"], [_tool("t")], response_format=schema))

        # The structured output the run wrote to state["structured_response"] is
        # surfaced on the invoke result for the direct-invoke structured path.
        assert result.structured == {"value": 7}

    def test_requested_but_missing_structured_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = {"messages": ["state"]}
        _patch_seams(monkeypatch, state=state)
        monkeypatch.setattr(bta, "build_user_output", lambda s: "text")
        monkeypatch.setattr(bta, "aggregate_usage", lambda s: CallUsage(0, 0, None))

        schema = {"title": "Answer", "type": "object", "properties": {"value": {"type": "integer"}}}
        with pytest.raises(RuntimeError, match="no structured_response"):
            asyncio.run(bta.ainvoke_tools_agent("sys", ["hi"], [_tool("t")], response_format=schema))

    def test_nonconforming_structured_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A structured output violating a schema constraint keyword raises loudly
        instead of being surfaced on the result."""
        state = {"messages": ["state"], "structured_response": {"value": -1}}
        _patch_seams(monkeypatch, state=state)
        monkeypatch.setattr(bta, "build_user_output", lambda s: "text")
        monkeypatch.setattr(bta, "aggregate_usage", lambda s: CallUsage(0, 0, None))

        schema = {
            "title": "Answer",
            "type": "object",
            "properties": {"value": {"type": "integer", "minimum": 0}},
            "required": ["value"],
        }
        with pytest.raises(JsonSchemaValidationError):
            asyncio.run(bta.ainvoke_tools_agent("sys", ["hi"], [_tool("t")], response_format=schema))


class TestAstream:
    def test_yields_raw_chunks_and_threads_stream_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_seams(monkeypatch, chunks=["chunk-a", "chunk-b"])

        chunks = asyncio.run(_collect(bta.astream_tools_agent("sys", ["hi"], [_tool("t")], stream_mode="updates")))

        assert chunks == ["chunk-a", "chunk-b"]
        _, config, stream_mode = captured["stream"]
        assert stream_mode == "updates"
        assert config == {"configurable": {"thread_id": "t"}}
