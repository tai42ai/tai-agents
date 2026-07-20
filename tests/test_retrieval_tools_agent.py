"""Tests for the ``retrieval_tools_agent`` package.

Covers registration through the recording app, the graph's conditional routing
and wiring, the context-reduction node, the per-tool-set store namespace, the
``select_tools`` tool-logic-vs-infrastructure error split, stale-id raising,
the terminal-envelope ``result`` surfaced by the agent's ``astream``/``run``,
and the normalized ``StreamEvent`` projection over a real compiled graph driven
by a scripted fake chat model + stub store.

Async code is driven with ``asyncio.run`` (the repo does not use
``pytest-asyncio``); no live model / embedding / store is touched.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool, ToolException
from langgraph.constants import END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.store.memory import InMemoryStore
from pydantic import PrivateAttr, ValidationError
from tai42_contract.agent import (
    Agent,
    MessageDelta,
    MessageFinal,
    ReasoningStep,
    RunUsage,
    StructuredFinal,
    ToolCallStep,
    ToolResultStep,
)
from tai42_contract.app import tai42_app
from tai42_kit.utils.data.json_schema_util import JsonSchemaValidationError
from tai42_kit.utils.data.string_util import text_to_md5

from tai42_agents._internal.reject import reject_unhonored
from tai42_agents._internal.stream_events import aproject_agent_events
from tai42_agents.retrieval_tools_agent import agent as ragent
from tai42_agents.retrieval_tools_agent import graph as rgraph
from tai42_agents.retrieval_tools_agent.agent import (
    _UNHONORED_REASONS,
    RetrievalToolsAgent,
    RetrievalToolsAgentInput,
    _embedding_dims,
    _embedding_dims_cache,
    _terminal_result,
)
from tai42_agents.retrieval_tools_agent.graph import RetrievalToolsGraph

AGENT_NAME = "retrieval_tools_agent"


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


def _tool(name: str, description: str = "d") -> StructuredTool:
    async def _run(**_: Any) -> str:
        return "ok"

    return StructuredTool.from_function(func=None, coroutine=_run, name=name, description=description)


class StubStore(InMemoryStore):
    """A real ``BaseStore`` (so langgraph injects it) whose semantic ``asearch``
    returns a fixed key list instead of running a vector index."""

    def __init__(self, keys: list[str]) -> None:
        super().__init__()
        self._keys = keys

    async def asearch(self, namespace_prefix: Any, /, **kwargs: Any) -> Any:  # type: ignore[override]
        return [SimpleNamespace(key=key) for key in self._keys]


class RecordingStore(InMemoryStore):
    """A real ``BaseStore`` (so ``InjectedStore`` validation accepts it) that
    records the namespace passed to each call so the per-tool-set namespace can
    be asserted, and returns canned search results."""

    def __init__(self) -> None:
        super().__init__()
        self.puts: list[tuple[Any, str, dict[str, Any]]] = []
        self.deletes: list[tuple[Any, str]] = []
        self.searches: list[tuple[Any, str | None, int]] = []

    async def aget(self, namespace: Any, key: str, **kwargs: Any) -> Any:
        return None

    async def aput(self, namespace: Any, key: str, value: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        self.puts.append((namespace, key, value))

    async def adelete(self, namespace: Any, key: str) -> None:
        self.deletes.append((namespace, key))

    async def asearch(
        self, namespace_prefix: Any, /, *, query: str | None = None, limit: int = 10, **kwargs: Any
    ) -> Any:
        self.searches.append((namespace_prefix, query, limit))
        return [SimpleNamespace(key="id1"), SimpleNamespace(key="id2")]


class ScriptedChatModel(BaseChatModel):
    """Returns a fixed list of messages in order; ``bind_tools`` is a no-op so
    the agent node can bind the retrieve/selected tools and still be scripted."""

    _responses: list[BaseMessage] = PrivateAttr(default_factory=list)
    _index: int = PrivateAttr(default=0)

    def __init__(self, responses: Sequence[BaseMessage], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._responses = list(responses)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        message = self._responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self


_RETRIEVAL_SCHEMA = {"title": "Answer", "type": "object", "properties": {"value": {"type": "integer"}}}


class _StructuredLLM:
    """A stand-in chat model whose ``with_structured_output`` binds a schema and
    returns a runner whose ``ainvoke`` yields a fixed structured payload."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.captured: dict[str, Any] = {}

    def with_structured_output(self, schema: Any, include_raw: bool = True) -> Any:
        self.captured["schema"] = schema
        self.captured["include_raw"] = include_raw
        outer = self

        class _Runner:
            async def ainvoke(self, messages: Any) -> Any:
                outer.captured["messages"] = messages
                return outer._payload

        return _Runner()


class _BoomStructuredLLM:
    """A structured llm whose finalization ``ainvoke`` raises, standing in for an
    unparseable structured-output response that must propagate loudly."""

    def with_structured_output(self, schema: Any, include_raw: bool = True) -> Any:
        class _Runner:
            async def ainvoke(self, messages: Any) -> Any:
                raise ValueError("model returned unparseable structured output")

        return _Runner()


def _graph(**kwargs: Any) -> RetrievalToolsGraph:
    return RetrievalToolsGraph(tools=[], llm=MagicMock(), **kwargs)


async def _collect(agen: AsyncIterator[Any]) -> list[Any]:
    return [event async for event in agen]


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


class TestRegistration:
    def test_decorator_registers_a_live_instance(self) -> None:
        agent = tai42_app.agents.get_agent(AGENT_NAME)
        assert isinstance(agent, RetrievalToolsAgent)
        assert isinstance(agent, Agent)

    def test_tool_metadata(self) -> None:
        assert RetrievalToolsAgent.tool_name == AGENT_NAME
        assert RetrievalToolsAgent.tool_description
        # The tool-face input model follows the ``<AgentClass>Input`` convention.
        assert RetrievalToolsAgent.ToolInput is RetrievalToolsAgentInput
        # Live tools are an API-only astream input, never in the JSON tool schema.
        assert "tools" not in RetrievalToolsAgent.ToolInput.model_fields


# --------------------------------------------------------------------------
# Conditional routing
# --------------------------------------------------------------------------


class TestShouldContinueRouting:
    def _decide(self, agent: RetrievalToolsGraph, last: BaseMessage) -> Any:
        node = agent.should_continue_node()
        return node({"messages": [last]}, store=MagicMock())

    def test_continue_status_routes_through_context_node(self) -> None:
        agent = _graph()
        decision = self._decide(agent, AIMessage(content=json.dumps({"status": "continue"})))
        assert decision == agent.context_node_name
        assert decision != agent.agent_node_name

    def test_success_status_ends(self) -> None:
        agent = _graph()
        assert self._decide(agent, AIMessage(content=json.dumps({"status": "success"}))) == END

    def test_error_status_ends(self) -> None:
        agent = _graph()
        assert self._decide(agent, AIMessage(content=json.dumps({"status": "error"}))) == END

    def test_success_status_in_list_content_ends(self) -> None:
        # A provider that returns content blocks (list) instead of a bare string:
        # the routing node concatenates the text blocks before parsing the JSON.
        agent = _graph()
        last = AIMessage(content=[{"type": "text", "text": json.dumps({"status": "success", "result": "ok"})}])
        assert self._decide(agent, last) == END

    def test_success_status_in_bare_string_list_content_ends(self) -> None:
        # Some providers return a list of bare strings rather than typed blocks;
        # the routing node concatenates those too before parsing the JSON.
        agent = _graph()
        last = AIMessage(content=[json.dumps({"status": "success", "result": "ok"})])
        assert self._decide(agent, last) == END

    def test_pending_tool_calls_dispatch_to_execute(self) -> None:
        agent = _graph()
        last = AIMessage(content="", tool_calls=[{"id": "t1", "name": "foo", "args": {}}])
        sends = self._decide(agent, last)
        assert [s.node for s in sends] == [agent.execute_tools_node_name]

    def test_retrieve_tools_call_dispatches_to_select(self) -> None:
        agent = _graph()
        last = AIMessage(content="", tool_calls=[{"id": "r1", "name": agent.retrieve_tools_tool_name, "args": {}}])
        sends = self._decide(agent, last)
        assert [s.node for s in sends] == [agent.select_tools_node_name]

    def test_mixed_calls_dispatch_to_both_tool_nodes(self) -> None:
        agent = _graph()
        last = AIMessage(
            content="",
            tool_calls=[
                {"id": "r1", "name": agent.retrieve_tools_tool_name, "args": {}},
                {"id": "t1", "name": "foo", "args": {}},
            ],
        )
        sends = self._decide(agent, last)
        assert sorted(s.node for s in sends) == sorted([agent.select_tools_node_name, agent.execute_tools_node_name])


# --------------------------------------------------------------------------
# Raise-on-malformed-terminal routing
# --------------------------------------------------------------------------


class TestMalformedTerminalRaises:
    def _decide(self, last: BaseMessage) -> Any:
        node = _graph().should_continue_node()
        return node({"messages": [last]}, store=MagicMock())

    def test_non_json_terminal_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid status JSON"):
            self._decide(AIMessage(content="all done!"))

    def test_empty_terminal_raises(self) -> None:
        with pytest.raises(ValueError, match="no text content"):
            self._decide(AIMessage(content=""))

    def test_non_object_json_terminal_raises(self) -> None:
        with pytest.raises(ValueError, match="not a JSON object"):
            self._decide(AIMessage(content="5"))

    def test_unknown_status_raises(self) -> None:
        with pytest.raises(ValueError, match="status must be one of"):
            self._decide(AIMessage(content=json.dumps({"status": "weird"})))


# --------------------------------------------------------------------------
# Graph wiring + context node
# --------------------------------------------------------------------------


class TestGraphWiring:
    def test_every_loop_back_routes_through_context(self) -> None:
        builder = _graph().build()
        assert ("__start__", "context") in builder.edges
        assert ("context", "agent") in builder.edges
        assert ("execute_tools", "context") in builder.edges
        assert ("select_tools", "context") in builder.edges
        assert ("execute_tools", "agent") not in builder.edges
        assert ("select_tools", "agent") not in builder.edges


class TestContextNode:
    def _run(self, agent: RetrievalToolsGraph, state: dict[str, Any]) -> Any:
        afunc = agent.context_node().afunc
        assert afunc is not None

        async def _invoke() -> Any:
            return await afunc(state, None, store=MagicMock())

        return asyncio.run(_invoke())

    def test_no_reduction_returns_empty_update(self) -> None:
        agent = _graph()
        with patch.object(rgraph, "areduce_context", new=AsyncMock(return_value=None)):
            out = self._run(agent, {"messages": [HumanMessage("hi", id="1")]})
        assert out == {}

    def test_reduction_wraps_with_remove_all(self) -> None:
        agent = _graph()
        reduced = [HumanMessage("summary", id="s")]
        with patch.object(rgraph, "areduce_context", new=AsyncMock(return_value=reduced)):
            out = self._run(agent, {"messages": [HumanMessage("hi", id="1")]})
        messages = out["messages"]
        assert isinstance(messages[0], RemoveMessage)
        assert messages[0].id == REMOVE_ALL_MESSAGES
        assert messages[1:] == reduced


# --------------------------------------------------------------------------
# abuild
# --------------------------------------------------------------------------


class TestAbuild:
    def test_abuild_raises_without_store(self) -> None:
        agent = RetrievalToolsGraph(tools=[], llm=MagicMock(), store=None)
        with pytest.raises(ValueError, match="requires a store"):
            asyncio.run(agent.abuild())


# --------------------------------------------------------------------------
# Per-tool-set store namespace
# --------------------------------------------------------------------------


class TestNamespace:
    def test_namespace_is_per_tool_set_and_deterministic(self) -> None:
        g1 = RetrievalToolsGraph(tools=[_tool("a"), _tool("b")], llm=MagicMock())
        g2 = RetrievalToolsGraph(tools=[_tool("b"), _tool("a")], llm=MagicMock())
        g3 = RetrievalToolsGraph(tools=[_tool("a"), _tool("c")], llm=MagicMock())

        assert g1.namespace[0] == "tools"
        assert len(g1.namespace) == 2
        # Same tool-set (any input order) -> same namespace, so the embedding
        # cache still reuses across identical runs.
        assert g1.namespace == g2.namespace
        # A different tool-set -> a different namespace, so the indexes don't
        # share one global bucket.
        assert g1.namespace != g3.namespace

    def test_add_to_store_uses_the_tool_set_namespace(self) -> None:
        agent = _graph()
        store = RecordingStore()
        asyncio.run(agent.add_to_store(store, "k1", _tool("foo", "does foo")))  # type: ignore[arg-type]
        assert store.puts == [(agent.namespace, "k1", {"description": "foo: does foo"})]

    def test_overwrite_store_deletes_before_put(self) -> None:
        agent = _graph(overwrite_store=True)
        store = RecordingStore()
        asyncio.run(agent.add_to_store(store, "k1", _tool("foo", "does foo")))  # type: ignore[arg-type]
        assert store.deletes == [(agent.namespace, "k1")]
        assert store.puts == [(agent.namespace, "k1", {"description": "foo: does foo"})]

    def test_retrieve_tool_searches_with_the_tool_set_namespace(self) -> None:
        agent = _graph(tools_limit=5)
        store = RecordingStore()
        tool = agent.retrieve_tools_tool()
        out = asyncio.run(tool.ainvoke({"query": "find", "store": store}))
        assert out == ["id1", "id2"]
        assert store.searches == [(agent.namespace, "find", 5)]


# --------------------------------------------------------------------------
# selected-tool-id merge reducer
# --------------------------------------------------------------------------


class TestMergedSelectedTools:
    def test_dedups_within_right_and_against_left(self) -> None:
        # ``b`` repeats within ``right`` and ``a`` already sits in ``left``: each
        # survives once, so a tool is never bound twice.
        assert rgraph._merged_selected_tools(["a"], ["b", "a", "b", "c", "b"]) == ["a", "b", "c"]

    def test_preserves_first_seen_order(self) -> None:
        assert rgraph._merged_selected_tools([], ["c", "a", "b", "a", "c"]) == ["c", "a", "b"]


# --------------------------------------------------------------------------
# select_tools: tool-logic vs infrastructure errors, event fidelity
# --------------------------------------------------------------------------


class TestSelectToolsNode:
    def _run(self, agent: RetrievalToolsGraph, tool_calls: list[dict], store: Any) -> Any:
        afunc = agent.select_tools_node().afunc
        assert afunc is not None

        async def _invoke() -> Any:
            return await afunc(tool_calls, None, store=store)

        return asyncio.run(_invoke())

    def test_success_reports_available_tools_ids_and_labels_message(self) -> None:
        agent = _graph()
        agent.tool_registry = {"id1": _tool("alpha"), "id2": _tool("beta")}
        store = RecordingStore()
        out = self._run(agent, [{"id": "c1", "name": "retrieve_tools", "args": {"query": "x"}}], store)
        assert out["selected_tool_ids"] == ["id1", "id2"]
        message = out["messages"][0]
        assert isinstance(message, ToolMessage)
        assert message.tool_call_id == "c1"
        # The result message is labeled with the retrieve tool's name and is not
        # flagged as an error.
        assert message.name == "retrieve_tools"
        assert message.status != "error"
        assert "alpha" in message.content
        assert "beta" in message.content

    def test_tool_logic_error_becomes_labeled_error_tool_message(self) -> None:
        agent = _graph()

        class BoomStore(RecordingStore):
            async def asearch(self, *args: Any, **kwargs: Any) -> Any:
                raise ToolException("boom")

        out = self._run(agent, [{"id": "c1", "name": "retrieve_tools", "args": {"query": "x"}}], BoomStore())
        assert out["selected_tool_ids"] == []
        message = out["messages"][0]
        assert message.name == "retrieve_tools"
        assert message.status == "error"
        assert message.content.startswith("Error:")
        assert "boom" in message.content

    def test_infrastructure_error_propagates(self) -> None:
        # A store/embedding/connection failure is NOT a tool-logic signal — it
        # propagates loudly instead of being masked as a ToolMessage.
        agent = _graph()

        class OutageStore(RecordingStore):
            async def asearch(self, *args: Any, **kwargs: Any) -> Any:
                raise ConnectionError("vector store unreachable")

        with pytest.raises(ConnectionError):
            self._run(agent, [{"id": "c1", "name": "retrieve_tools", "args": {"query": "x"}}], OutageStore())

    def test_programming_error_after_retrieval_propagates(self) -> None:
        # The except is narrowed to the retrieve invocation, so a fault in the
        # id->name mapping (here: a non-iterable retrieval result) is NOT masked
        # as a ToolMessage — it propagates.
        agent = _graph()

        async def _bad(query: str, *, store: Any) -> Any:
            """Return a non-iterable so the id->name mapping faults."""
            return 123

        agent._retrieve_tools_tool = StructuredTool.from_function(func=None, coroutine=_bad, name="retrieve_tools")
        with pytest.raises(TypeError):
            self._run(agent, [{"id": "c1", "name": "retrieve_tools", "args": {"query": "x"}}], RecordingStore())

    def test_unregistered_retrieved_id_raises(self) -> None:
        # A retrieved id addresses this tool-set's own namespace, so a miss is a
        # real inconsistency and raises rather than being silently dropped.
        agent = _graph()
        with pytest.raises(KeyError, match="no entry in this tool-set"):
            self._run(agent, [{"id": "c1", "name": "retrieve_tools", "args": {"query": "x"}}], StubStore(["ghost"]))


# --------------------------------------------------------------------------
# agent_node stale-id handling
# --------------------------------------------------------------------------


class TestAgentNodeStaleId:
    def test_unregistered_selected_id_raises(self) -> None:
        agent = _graph()
        afunc = agent.agent_node().afunc
        assert afunc is not None

        async def _invoke() -> Any:
            return await afunc(
                {"messages": [HumanMessage("hi")], "selected_tool_ids": ["ghost"]}, None, store=MagicMock()
            )

        with pytest.raises(KeyError, match="no entry in this tool-set"):
            asyncio.run(_invoke())


# --------------------------------------------------------------------------
# Terminal-envelope result helper
# --------------------------------------------------------------------------


class TestTerminalResultHelper:
    def test_extracts_result_from_success_envelope(self) -> None:
        assert _terminal_result(json.dumps({"status": "success", "message": "m", "result": "ok"})) == "ok"

    def test_takes_last_of_concatenated_envelopes(self) -> None:
        text = json.dumps({"status": "continue", "message": "m", "result": None}) + json.dumps(
            {"status": "success", "message": "done", "result": "final"}
        )
        assert _terminal_result(text) == "final"

    def test_error_envelope_with_null_result_is_empty_string(self) -> None:
        assert _terminal_result(json.dumps({"status": "error", "message": "x", "result": None})) == ""

    def test_non_string_result_is_stringified(self) -> None:
        assert _terminal_result(json.dumps({"status": "success", "message": "m", "result": 7})) == "7"

    def test_non_json_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid status JSON"):
            _terminal_result("all done")

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="no status JSON content"):
            _terminal_result("   ")

    def test_non_object_raises(self) -> None:
        with pytest.raises(ValueError, match="not a JSON object"):
            _terminal_result("5")

    def test_unknown_status_raises(self) -> None:
        with pytest.raises(ValueError, match="status must be success/error"):
            _terminal_result(json.dumps({"status": "continue", "result": "x"}))

    def test_missing_result_field_raises(self) -> None:
        with pytest.raises(ValueError, match="no 'result' field"):
            _terminal_result(json.dumps({"status": "success", "message": "m"}))


# --------------------------------------------------------------------------
# StreamEvent projection over a real compiled graph
# --------------------------------------------------------------------------


class TestStreamEvents:
    def test_full_run_emits_the_event_taxonomy(self) -> None:
        alpha = _tool("alpha", "the alpha tool")
        alpha_id = text_to_md5("alpha")

        llm = ScriptedChatModel(
            [
                AIMessage(
                    content="",
                    additional_kwargs={"reasoning_content": "thinking about alpha"},
                    tool_calls=[{"id": "r1", "name": "retrieve_tools", "args": {"query": "need alpha"}}],
                ),
                AIMessage(
                    content=json.dumps({"status": "success", "message": "done", "result": "ok"}),
                    usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                ),
            ]
        )
        graph = asyncio.run(RetrievalToolsGraph(tools=[alpha], llm=llm, store=StubStore([alpha_id])).abuild())

        messages = {"messages": [{"role": "user", "content": "do it"}]}
        config = {"configurable": {"thread_id": "t1"}}
        events = asyncio.run(_collect(aproject_agent_events(graph, messages, config)))

        kinds = [type(event) for event in events]
        assert ReasoningStep in kinds
        assert ToolCallStep in kinds
        assert ToolResultStep in kinds
        assert RunUsage in kinds
        assert MessageFinal in kinds

        tool_call = next(e for e in events if isinstance(e, ToolCallStep))
        assert tool_call.tool == "retrieve_tools"
        tool_result = next(e for e in events if isinstance(e, ToolResultStep))
        # The retrieve result step is labeled with the retrieve tool's name and
        # is not an error.
        assert tool_result.tool == "retrieve_tools"
        assert tool_result.is_error is False
        assert "alpha" in str(tool_result.result)
        usage = next(e for e in events if isinstance(e, RunUsage))
        assert usage.total_tokens == 8

    def test_retrieve_error_projects_a_labeled_error_tool_result(self) -> None:
        class BoomStore(StubStore):
            async def asearch(self, namespace_prefix: Any, /, **kwargs: Any) -> Any:
                raise ToolException("no match")

        llm = ScriptedChatModel(
            [
                AIMessage(content="", tool_calls=[{"id": "r1", "name": "retrieve_tools", "args": {"query": "q"}}]),
                AIMessage(content=json.dumps({"status": "error", "message": "gave up", "result": None})),
            ]
        )
        graph = asyncio.run(RetrievalToolsGraph(tools=[_tool("alpha", "a")], llm=llm, store=BoomStore([])).abuild())

        events = asyncio.run(
            _collect(
                aproject_agent_events(
                    graph, {"messages": [{"role": "user", "content": "x"}]}, {"configurable": {"thread_id": "t2"}}
                )
            )
        )
        tool_result = next(e for e in events if isinstance(e, ToolResultStep))
        assert tool_result.tool == "retrieve_tools"
        assert tool_result.is_error is True
        assert "Error:" in str(tool_result.result)


# --------------------------------------------------------------------------
# Agent astream/run surface the terminal envelope's ``result``
# --------------------------------------------------------------------------


class TestTerminalResultSurfacing:
    def _agent_over_graph(self, monkeypatch: pytest.MonkeyPatch, graph: Any) -> RetrievalToolsAgent:
        async def fake_build(self: RetrievalToolsAgent, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
            return (
                graph,
                {"messages": [{"role": "user", "content": "do it"}]},
                {"configurable": {"thread_id": "t1"}},
                None,
            )

        monkeypatch.setattr(RetrievalToolsAgent, "_build", fake_build)
        return RetrievalToolsAgent()

    def _two_step_graph(self, result: str) -> Any:
        alpha = _tool("alpha", "the alpha tool")
        alpha_id = text_to_md5("alpha")
        llm = ScriptedChatModel(
            [
                AIMessage(content="", tool_calls=[{"id": "r1", "name": "retrieve_tools", "args": {"query": "a"}}]),
                AIMessage(content=json.dumps({"status": "success", "message": "done", "result": result})),
            ]
        )
        return asyncio.run(RetrievalToolsGraph(tools=[alpha], llm=llm, store=StubStore([alpha_id])).abuild())

    def test_run_returns_terminal_result_not_the_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = self._agent_over_graph(monkeypatch, self._two_step_graph("THE RESULT"))
        assert asyncio.run(agent.run(user_message="x")) == "THE RESULT"

    def test_astream_final_is_result_and_no_envelope_leaks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = self._agent_over_graph(monkeypatch, self._two_step_graph("THE RESULT"))
        events = asyncio.run(_collect(agent.astream(user_message="x")))

        finals = [e for e in events if isinstance(e, MessageFinal)]
        assert len(finals) == 1
        assert finals[0].text == "THE RESULT"
        # Per-step / interim status JSON must not leak into the streamed answer.
        assert not any(isinstance(e, MessageDelta) for e in events)
        assert "status" not in finals[0].text

    def test_multi_step_run_surfaces_only_the_terminal_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        alpha = _tool("alpha", "the alpha tool")
        alpha_id = text_to_md5("alpha")
        llm = ScriptedChatModel(
            [
                AIMessage(content="", tool_calls=[{"id": "r1", "name": "retrieve_tools", "args": {"query": "a"}}]),
                # An interim "continue" step loops back through context; its status
                # JSON must not bleed into the final answer.
                AIMessage(content=json.dumps({"status": "continue", "message": "more", "result": None})),
                AIMessage(content=json.dumps({"status": "success", "message": "done", "result": "FINAL ANSWER"})),
            ]
        )
        graph = asyncio.run(RetrievalToolsGraph(tools=[alpha], llm=llm, store=StubStore([alpha_id])).abuild())
        agent = self._agent_over_graph(monkeypatch, graph)
        assert asyncio.run(agent.run(user_message="x")) == "FINAL ANSWER"

    def test_astream_raises_when_no_terminal_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_build(self: RetrievalToolsAgent, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
            return "graph", "messages", "config", None

        async def fake_project(agent: Any, messages: Any, config: Any) -> AsyncIterator[Any]:
            yield ReasoningStep(text="thinking")

        monkeypatch.setattr(RetrievalToolsAgent, "_build", fake_build)
        monkeypatch.setattr(ragent, "aproject_agent_events", fake_project)
        agent = RetrievalToolsAgent()
        with pytest.raises(ValueError, match="produced no terminal message"):
            asyncio.run(_collect(agent.astream(user_message="x")))

    def test_astream_suppresses_interim_status_deltas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A live provider streams the per-step status envelope as ``MessageDelta``
        tokens; those are dropped so only the terminal ``result`` surfaces."""

        async def fake_build(self: RetrievalToolsAgent, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
            return "graph", "messages", "config", None

        async def fake_project(agent: Any, messages: Any, config: Any) -> AsyncIterator[Any]:
            yield MessageDelta(text='{"status":"continue"')
            yield MessageDelta(text=',"result":null}')
            yield MessageFinal(text=json.dumps({"status": "success", "message": "done", "result": "THE RESULT"}))

        monkeypatch.setattr(RetrievalToolsAgent, "_build", fake_build)
        monkeypatch.setattr(ragent, "aproject_agent_events", fake_project)
        agent = RetrievalToolsAgent()
        events = asyncio.run(_collect(agent.astream(user_message="x")))

        assert not any(isinstance(e, MessageDelta) for e in events)
        finals = [e for e in events if isinstance(e, MessageFinal)]
        assert len(finals) == 1
        assert finals[0].text == "THE RESULT"


# --------------------------------------------------------------------------
# Agent wrapper: _embedding_dims cache, _build wiring, astream, run
# --------------------------------------------------------------------------


class TestEmbeddingDimsCache:
    def test_probes_once_then_serves_from_cache(self) -> None:
        _embedding_dims_cache.clear()
        embedding = MagicMock()
        embedding.aembed_query = AsyncMock(return_value=[0.0, 0.1, 0.2])

        first = asyncio.run(_embedding_dims("openai", embedding, None))
        second = asyncio.run(_embedding_dims("openai", embedding, None))

        assert first == second == 3
        # Second call is served from the cache — no second probe embed.
        embedding.aembed_query.assert_awaited_once()

    def test_distinct_kwargs_are_distinct_cache_keys(self) -> None:
        _embedding_dims_cache.clear()
        embedding = MagicMock()
        embedding.aembed_query = AsyncMock(side_effect=[[0.0, 0.1], [0.0, 0.1, 0.2, 0.3]])

        a = asyncio.run(_embedding_dims("openai", embedding, {"model": "small"}))
        b = asyncio.run(_embedding_dims("openai", embedding, {"model": "large"}))

        assert (a, b) == (2, 4)
        assert embedding.aembed_query.await_count == 2

    def test_lru_evicts_oldest_past_the_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the bound forced to 2, a third distinct key evicts the oldest,
        the cache never exceeds the bound, and re-probing the evicted key runs a
        fresh embed."""
        _embedding_dims_cache.clear()
        monkeypatch.setattr(ragent, "agents_limits_settings", lambda: SimpleNamespace(embedding_dims_cache_size=2))
        embedding = MagicMock()
        # k1 -> 1 dim, k2 -> 2 dims, k3 -> 3 dims, then k1 re-probes -> 1 dim.
        embedding.aembed_query = AsyncMock(side_effect=[[0.0], [0.0, 0.0], [0.0, 0.0, 0.0], [0.0]])

        asyncio.run(_embedding_dims("p", embedding, {"model": "k1"}))
        asyncio.run(_embedding_dims("p", embedding, {"model": "k2"}))
        asyncio.run(_embedding_dims("p", embedding, {"model": "k3"}))

        assert len(_embedding_dims_cache) == 2
        # k1 was the oldest, so it was evicted; a fourth call re-probes it.
        again = asyncio.run(_embedding_dims("p", embedding, {"model": "k1"}))
        assert again == 1
        assert embedding.aembed_query.await_count == 4
        assert len(_embedding_dims_cache) == 2

    def test_hit_refreshes_recency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cache hit on the oldest key moves it to most-recent, so the NEXT
        insert evicts a different key — never exceeding the bound."""
        _embedding_dims_cache.clear()
        monkeypatch.setattr(ragent, "agents_limits_settings", lambda: SimpleNamespace(embedding_dims_cache_size=2))
        embedding = MagicMock()
        embedding.aembed_query = AsyncMock(side_effect=[[0.0], [0.0, 0.0], [0.0, 0.0, 0.0]])

        asyncio.run(_embedding_dims("p", embedding, {"model": "k1"}))
        asyncio.run(_embedding_dims("p", embedding, {"model": "k2"}))
        # Hit on k1 refreshes its recency, so k2 is now the oldest.
        asyncio.run(_embedding_dims("p", embedding, {"model": "k1"}))
        # Inserting k3 evicts k2, leaving k1 (still cached — no re-probe).
        asyncio.run(_embedding_dims("p", embedding, {"model": "k3"}))

        assert len(_embedding_dims_cache) == 2
        # k1 survived: serving it takes no further probe.
        served = asyncio.run(_embedding_dims("p", embedding, {"model": "k1"}))
        assert served == 1
        assert embedding.aembed_query.await_count == 3


def _patch_build_seams(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Monkeypatch every provider seam ``_build`` reaches, and return the
    captured wiring so a test can assert defaults were resolved and threaded."""
    captured: dict[str, Any] = {}

    provider_settings = SimpleNamespace(
        llm="def_llm",
        embedding="def_embedding",
        checkpoint="def_checkpoint",
        store="def_store",
        store_conn_string="store-conn",
        checkpoint_conn_string="checkpoint-conn",
    )
    monkeypatch.setattr(ragent, "llm_provider_settings", lambda: provider_settings)
    monkeypatch.setattr(ragent, "llm_settings", lambda: SimpleNamespace(with_fallbacks=lambda kwargs: dict(kwargs)))
    monkeypatch.setattr(
        ragent, "embedding_settings", lambda: SimpleNamespace(with_fallbacks=lambda kwargs: dict(kwargs))
    )

    async def fake_resolve_tools(app_tools: Any, names: list[str], tools: list[Any], presets: list[Any]) -> list[Any]:
        captured["resolve"] = (names, tools, presets)
        return ["resolved-tool"]

    monkeypatch.setattr(ragent, "resolve_tools", fake_resolve_tools)

    async def fake_get_llm(*, provider: str, **kwargs: Any) -> Any:
        captured["llm_provider"] = provider
        captured["llm_kwargs"] = kwargs
        return "llm-obj"

    monkeypatch.setattr(ragent, "get_llm_async", fake_get_llm)

    embedding = MagicMock()
    embedding.aembed_query = AsyncMock(return_value=[0.0] * 6)

    async def fake_get_embedding(*, provider: str, **kwargs: Any) -> Any:
        captured["embedding_provider"] = provider
        captured["embedding_kwargs"] = kwargs
        return embedding

    monkeypatch.setattr(ragent, "get_embedding_async", fake_get_embedding)

    async def fake_get_store(*, provider: str, conn_string: str, **kwargs: Any) -> Any:
        captured["store"] = (provider, conn_string, kwargs)
        return "store-obj"

    monkeypatch.setattr(ragent, "store_registry", lambda: SimpleNamespace(get_store=fake_get_store))

    async def fake_get_checkpointer(*, provider: str, conn_string: str) -> Any:
        captured["checkpoint"] = (provider, conn_string)
        return "checkpointer-obj"

    monkeypatch.setattr(ragent, "checkpoint_registry", lambda: SimpleNamespace(get_checkpointer=fake_get_checkpointer))

    class FakeGraph:
        def __init__(self, **kwargs: Any) -> None:
            captured["graph_kwargs"] = kwargs

        async def abuild(self) -> str:
            return "compiled-graph"

    monkeypatch.setattr(ragent, "RetrievalToolsGraph", FakeGraph)
    monkeypatch.setattr(ragent, "init_langgraph_config", lambda config: {"configurable": {"thread_id": "t"}})
    _embedding_dims_cache.clear()
    return captured


class TestBuild:
    def test_resolves_defaults_and_returns_agent_messages_config(
        self, monkeypatch: pytest.MonkeyPatch, resource_manager: Any
    ) -> None:
        captured = _patch_build_seams(monkeypatch)
        agent = RetrievalToolsAgent()

        compiled, messages, config, llm = asyncio.run(
            agent._build(system_message="be brief", user_message="do it", tools_limit=7)
        )

        assert compiled == "compiled-graph"
        assert config == {"configurable": {"thread_id": "t"}}
        # ``_build`` hands back the resolved llm for the structured finalization pass.
        assert llm == "llm-obj"
        # Rendered user message + system prompt thread into the built agent input.
        roles = {message["role"]: message["content"] for message in messages["messages"]}
        assert roles["user"] == "do it"
        assert "be brief" in roles["system"]

        # Providers fell back to the settings defaults and were threaded through.
        assert captured["llm_provider"] == "def_llm"
        assert captured["embedding_provider"] == "def_embedding"
        assert captured["store"][0] == "def_store"
        assert captured["store"][1] == "store-conn"
        assert captured["checkpoint"] == ("def_checkpoint", "checkpoint-conn")
        # Embedding dims probed (6) and passed into the store index spec.
        assert captured["store"][2]["index"]["dims"] == 6
        # Resolved tools + limit reach the graph.
        assert captured["graph_kwargs"]["tools"] == ["resolved-tool"]
        assert captured["graph_kwargs"]["tools_limit"] == 7

    def test_explicit_providers_override_defaults(self, monkeypatch: pytest.MonkeyPatch, resource_manager: Any) -> None:
        captured = _patch_build_seams(monkeypatch)
        agent = RetrievalToolsAgent()

        asyncio.run(
            agent._build(
                user_message="hi",
                llm_provider="my_llm",
                embedding_provider="my_embedding",
                checkpoint_provider="my_checkpoint",
                store_provider="my_store",
            )
        )

        assert captured["llm_provider"] == "my_llm"
        assert captured["embedding_provider"] == "my_embedding"
        assert captured["store"][0] == "my_store"
        assert captured["checkpoint"][0] == "my_checkpoint"

    def test_run_raises_when_required_user_message_slot_is_unset(self, resource_manager: Any) -> None:
        """``_build`` renders the user message with ``allow_empty=False``: an unset
        slot (empty content AND empty id) surfaces the resource manager's fail-loud
        message rather than silently building on an empty prompt. Reached through the
        real ``run`` face — before any provider seam — so a dropped render.py
        ``allow_empty`` passthrough turns this red."""
        agent = RetrievalToolsAgent()
        with pytest.raises(ValueError, match="must provide either"):
            asyncio.run(agent.run())


# Every key in ``_UNHONORED_REASONS`` paired with a representative SET value; the
# falsy-but-present scalars (strategy="", interrupt_on={}, resume=False) pin that a
# scalar is set whenever it is not None. Distinct params here must equal the full map.
_UNHONORED_CASES = [
    ("subagents", [object()]),
    ("strategy", "parallel"),
    ("strategy", ""),
    ("skills", ["s"]),
    ("inline_skills", [{"name": "n", "content": "c"}]),
    ("interrupt_on", {"tool": True}),
    ("interrupt_on", {}),
    ("resume", {"answer": "y"}),
    ("resume", False),
]

# The unhonored params whose ABC ``Agent.run`` default is an empty collection (``()`` /
# ``""``), read from the contract signature — the independent source of truth for which
# unhonored params are collection-typed. Intersected with this agent's reasons map it is
# exactly the set ``_UNHONORED_COLLECTION_PARAMS`` must classify as collections. The
# empty-collection test below parametrizes from HERE, not from the frozenset, so a member
# dropped from the frozenset (reclassifying it as a scalar) turns a case red rather than
# silently vanishing.
_EMPTY_COLLECTION_ABC_DEFAULTS = frozenset(
    name
    for name, parameter in inspect.signature(Agent.run).parameters.items()
    if isinstance(parameter.default, (tuple, list, str)) and not parameter.default
)
_COLLECTION_REJECT_PARAMS = sorted(_UNHONORED_REASONS.keys() & _EMPTY_COLLECTION_ABC_DEFAULTS)


class TestAstreamAndRun:
    def _script(self, monkeypatch: pytest.MonkeyPatch, events: list[Any], llm: Any = None) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        async def fake_build(self: RetrievalToolsAgent, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
            captured["build_kwargs"] = kwargs
            return "graph", "messages", "config", llm

        async def fake_project(agent: Any, messages: Any, config: Any) -> AsyncIterator[Any]:
            captured["project"] = (agent, messages, config)
            for event in events:
                yield event

        monkeypatch.setattr(RetrievalToolsAgent, "_build", fake_build)
        monkeypatch.setattr(ragent, "aproject_agent_events", fake_project)
        return captured

    def _final(self, result: str) -> MessageFinal:
        return MessageFinal(text=json.dumps({"status": "success", "message": "m", "result": result}))

    def test_astream_honors_thread_id_and_forwards_build_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # thread_id is an honored ABC parameter: it is mapped into the run config's
        # ``configurable`` (never dropped), while the ``_build`` parameters pass
        # through and the projection is yielded untouched.
        captured = self._script(monkeypatch, [self._final("done")])
        agent = RetrievalToolsAgent()

        events = asyncio.run(_collect(agent.astream(user_message="hi", tools_limit=3, thread_id="keep-me")))

        assert [type(event) for event in events] == [MessageFinal]
        assert events[0].text == "done"
        assert captured["build_kwargs"] == {
            "user_message": "hi",
            "tools_limit": 3,
            "config": {"configurable": {"thread_id": "keep-me"}},
        }
        assert captured["project"] == ("graph", "messages", "config")

    def test_run_honors_thread_id_and_resume_checkpoint_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The run face drains astream, so both honored memory parameters reach the
        # same config seam: thread_id -> configurable.thread_id, resume_checkpoint_id
        # -> configurable.checkpoint_id.
        captured = self._script(monkeypatch, [self._final("done")])
        agent = RetrievalToolsAgent()

        asyncio.run(agent.run(user_message="hi", thread_id="th", resume_checkpoint_id="cp"))

        assert captured["build_kwargs"]["config"]["configurable"] == {"thread_id": "th", "checkpoint_id": "cp"}

    def test_honored_config_overlays_a_caller_supplied_config_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # thread_id overlays onto a caller-supplied ``langgraph_config`` base — the
        # ecosystem-standard base-config name every agent honors — rather than
        # discarding it, preserving other configurable entries.
        captured = self._script(monkeypatch, [self._final("done")])
        agent = RetrievalToolsAgent()

        asyncio.run(
            _collect(
                agent.astream(
                    user_message="hi",
                    thread_id="th",
                    langgraph_config={"configurable": {"monitoring_trace_id": "x"}},
                )
            )
        )

        assert captured["build_kwargs"]["config"]["configurable"] == {"monitoring_trace_id": "x", "thread_id": "th"}

    @pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
    def test_unsupported_abc_param_raises_naming_the_exact_face(
        self, monkeypatch: pytest.MonkeyPatch, param: str, value: Any
    ) -> None:
        # Every ABC parameter this runtime cannot honor raises a RuntimeError naming
        # the offending parameter AND the exact face it was called on, never a silent
        # drop. Falsy-but-present scalars (strategy="", interrupt_on={}, resume=False)
        # still raise — a scalar parameter is set whenever it is not None, so they
        # never slip through a truthiness gate. Asserting the exact face token makes
        # each face's guard load-bearing: dropping the run-face guard (which delegates
        # to astream) would surface the ``astream`` token and fail the run assertion.
        self._script(monkeypatch, [self._final("done")])
        agent = RetrievalToolsAgent()

        with pytest.raises(RuntimeError, match=rf"retrieval_tools_agent\.astream does not support .*\b{param}\b"):
            asyncio.run(_collect(agent.astream(user_message="hi", **{param: value})))
        with pytest.raises(RuntimeError, match=rf"retrieval_tools_agent\.run does not support .*\b{param}\b"):
            asyncio.run(agent.run(user_message="hi", **{param: value}))

    def test_unhonored_cases_cover_the_full_reasons_map(self) -> None:
        # Every key in the guard's reasons map has a parametrized reject case above, so
        # a key added to the map without a matching test fails here immediately.
        assert {param for param, _ in _UNHONORED_CASES} == set(_UNHONORED_REASONS)

    def test_collection_params_match_the_abc_collection_defaults(self) -> None:
        # _UNHONORED_COLLECTION_PARAMS is exactly this agent's unhonored params whose ABC
        # default is an empty collection: no scalar wrongly listed (which would let a
        # meaningful falsy value slip through), none dropped (which would over-reject the
        # not-requested empty default).
        assert set(ragent._UNHONORED_COLLECTION_PARAMS) == set(_COLLECTION_REJECT_PARAMS)

    @pytest.mark.parametrize("empty", [[], ""])
    @pytest.mark.parametrize("param", _COLLECTION_REJECT_PARAMS)
    def test_reject_unhonored_permits_empty_collection_param(self, param: str, empty: object) -> None:
        # An empty collection is the ABC's "not requested" default for a collection
        # parameter, so the guard does not raise for it — in either falsy empty form ([] /
        # ""). Were the parameter dropped from _UNHONORED_COLLECTION_PARAMS it would be
        # classified as a scalar (set whenever it is not None) and this empty value would
        # raise.
        reject_unhonored(
            "retrieval_tools_agent.run",
            {param: empty},
            _UNHONORED_REASONS,
            collection_params=ragent._UNHONORED_COLLECTION_PARAMS,
        )

    @pytest.mark.parametrize("blank", ["", "   "])
    @pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
    def test_astream_rejects_blank_memory_key(self, monkeypatch: pytest.MonkeyPatch, key: str, blank: str) -> None:
        # A present-but-blank thread_id / resume_checkpoint_id is malformed — it would
        # silently share a checkpoint namespace across independent runs — so astream
        # raises before building rather than writing it into configurable verbatim.
        self._script(monkeypatch, [self._final("done")])
        agent = RetrievalToolsAgent()
        with pytest.raises(ValueError, match=rf"retrieval_tools_agent\.astream: {key} must be a non-empty string"):
            asyncio.run(_collect(agent.astream(user_message="hi", **{key: blank})))

    @pytest.mark.parametrize("blank", ["", "   "])
    @pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
    def test_run_rejects_blank_memory_key(self, monkeypatch: pytest.MonkeyPatch, key: str, blank: str) -> None:
        # Parity with astream: the run face rejects a present-but-blank memory key loudly.
        self._script(monkeypatch, [self._final("done")])
        agent = RetrievalToolsAgent()
        with pytest.raises(ValueError, match=rf"retrieval_tools_agent\.run: {key} must be a non-empty string"):
            asyncio.run(agent.run(user_message="hi", **{key: blank}))

    @pytest.mark.parametrize("value", [123, ["x"]])
    @pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
    def test_astream_rejects_non_string_memory_key(self, monkeypatch: pytest.MonkeyPatch, key: str, value: Any) -> None:
        # A non-string thread_id / resume_checkpoint_id is a type violation — it cannot
        # name a checkpoint namespace — so astream raises TypeError naming the offending
        # param AND the received type, rather than probing a non-string for whitespace.
        self._script(monkeypatch, [self._final("done")])
        agent = RetrievalToolsAgent()
        with pytest.raises(
            TypeError,
            match=rf"retrieval_tools_agent\.astream: {key} must be a string or None; got {type(value).__name__}",
        ):
            asyncio.run(_collect(agent.astream(user_message="hi", **{key: value})))

    @pytest.mark.parametrize("value", [123, ["x"]])
    @pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
    def test_run_rejects_non_string_memory_key(self, monkeypatch: pytest.MonkeyPatch, key: str, value: Any) -> None:
        # Parity with astream: the run face rejects a non-string memory key with a
        # TypeError naming the offending param and the received type.
        self._script(monkeypatch, [self._final("done")])
        agent = RetrievalToolsAgent()
        with pytest.raises(
            TypeError, match=rf"retrieval_tools_agent\.run: {key} must be a string or None; got {type(value).__name__}"
        ):
            asyncio.run(agent.run(user_message="hi", **{key: value}))

    def test_astream_honors_recursion_limit_into_build_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # recursion_limit is a standard RunnableConfig key the compiled graph reads,
        # so it is overlaid onto the config astream hands to _build (which threads it
        # through init_langgraph_config to the graph invocation — see
        # test_config_util) rather than rejected. A falsy 0 is a real forwarded value.
        captured = self._script(monkeypatch, [self._final("done")])
        agent = RetrievalToolsAgent()
        asyncio.run(_collect(agent.astream(user_message="hi", recursion_limit=0)))
        assert captured["build_kwargs"]["config"]["recursion_limit"] == 0

    def test_run_honors_recursion_limit_into_build_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The run face drains astream, so the honored recursion_limit reaches the same
        # config seam.
        captured = self._script(monkeypatch, [self._final("done")])
        agent = RetrievalToolsAgent()
        asyncio.run(agent.run(user_message="hi", recursion_limit=9))
        assert captured["build_kwargs"]["config"]["recursion_limit"] == 9

    def test_run_drains_astream_to_final_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._script(monkeypatch, [self._final("the answer")])
        agent = RetrievalToolsAgent()
        assert asyncio.run(agent.run(user_message="hi")) == "the answer"

    def test_astream_with_response_format_emits_one_structured_final(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # With a response_format set, the terminal result is forced into the schema
        # by a structured finalization pass over the resolved llm: the run ends with
        # exactly one StructuredFinal (carrying the validated object) and NO text
        # MessageFinal, and the schema is passed straight to the provider.
        llm = _StructuredLLM({"value": 7})
        self._script(monkeypatch, [self._final("the answer")], llm=llm)
        agent = RetrievalToolsAgent()

        events = asyncio.run(_collect(agent.astream(user_message="hi", response_format=_RETRIEVAL_SCHEMA)))

        finals = [e for e in events if isinstance(e, StructuredFinal)]
        assert len(finals) == 1
        assert finals[0].data == {"value": 7}
        assert not any(isinstance(e, MessageFinal) for e in events)
        assert llm.captured["schema"] == _RETRIEVAL_SCHEMA
        assert llm.captured["include_raw"] is False
        # The finalization message carries the terminal envelope's plain-text result.
        assert llm.captured["messages"][0].content == "the answer"

    def test_run_with_response_format_returns_the_structured_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = _StructuredLLM({"value": 7})
        self._script(monkeypatch, [self._final("the answer")], llm=llm)
        agent = RetrievalToolsAgent()
        assert asyncio.run(agent.run(user_message="hi", response_format=_RETRIEVAL_SCHEMA)) == {"value": 7}

    def test_run_response_format_without_title_raises_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._script(monkeypatch, [self._final("the answer")])
        agent = RetrievalToolsAgent()
        with pytest.raises(ValueError, match="top-level 'title'"):
            asyncio.run(agent.run(user_message="hi", response_format={"type": "object"}))

    def test_astream_response_format_without_title_raises_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The streaming face — the one the public run door drives — rejects an
        untitled ``response_format`` up front, not after the whole run has executed
        and the finalization pass reaches langchain."""
        self._script(monkeypatch, [self._final("the answer")])
        agent = RetrievalToolsAgent()
        with pytest.raises(ValueError, match="top-level 'title'"):
            asyncio.run(_collect(agent.astream(user_message="hi", response_format={"type": "object"})))

    def test_run_response_format_unparseable_finalization_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An unparseable structured-finalization output propagates the raise loudly —
        # never silence, never a text fallback.
        llm = _BoomStructuredLLM()
        self._script(monkeypatch, [self._final("the answer")], llm=llm)
        agent = RetrievalToolsAgent()
        with pytest.raises(ValueError, match="unparseable structured output"):
            asyncio.run(agent.run(user_message="hi", response_format=_RETRIEVAL_SCHEMA))

    def test_run_response_format_nonconforming_structured_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A finalization payload violating a schema constraint keyword (minimum)
        # raises loudly from the validation step instead of being returned.
        schema = {
            "title": "Answer",
            "type": "object",
            "properties": {"value": {"type": "integer", "minimum": 0}},
            "required": ["value"],
        }
        llm = _StructuredLLM({"value": -1})
        self._script(monkeypatch, [self._final("the answer")], llm=llm)
        agent = RetrievalToolsAgent()
        with pytest.raises(JsonSchemaValidationError):
            asyncio.run(agent.run(user_message="hi", response_format=schema))


class TestInputModel:
    def test_input_rejects_unknown_key(self) -> None:
        # ``extra="forbid"`` turns a typo at the run door into a loud validation
        # error rather than a silently ignored field.
        with pytest.raises(ValidationError):
            RetrievalToolsAgentInput.model_validate({"user_message": "hi", "unknown_key": 1})

    def test_input_advertises_response_format(self) -> None:
        # ``response_format`` is an advertised, honored field (round-trips through the
        # tool schema for the preset/authoring path).
        assert "response_format" in RetrievalToolsAgentInput.model_json_schema()["properties"]
        parsed = RetrievalToolsAgentInput.model_validate({"user_message": "hi", "response_format": _RETRIEVAL_SCHEMA})
        assert parsed.response_format == _RETRIEVAL_SCHEMA
