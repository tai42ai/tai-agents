"""Tests for the ``refine_agent`` — registration, tool resolution, the
Evaluator↔Critic loop, the approval-token detection, and the streamed
final-evaluator-pass event taxonomy.

No live model or network: ``create_agent`` and the kit LLM/checkpoint/logging
seams are monkeypatched so the loop runs against scripted fake evaluator/critic
agents. Fake evaluators/critics return canned ``ainvoke`` states; the fake
evaluator's ``astream`` replays a fixed ``(mode, chunk)`` script that
``aproject_agent_events`` decodes into the contract event vocabulary.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError
from tai42_contract.agent import (
    Agent,
    MessageDelta,
    MessageFinal,
    ReasoningStep,
    RunUsage,
    StreamEvent,
    StructuredFinal,
    ToolCallStep,
    ToolResultStep,
)
from tai42_contract.app import tai42_app
from tai42_kit.utils.data.json_schema_util import JsonSchemaValidationError

import tai42_agents.refine_agent.agent as agent_mod
from tai42_agents._internal.reject import reject_unhonored
from tai42_agents.refine_agent.agent import RefineAgent, RefineAgentInput
from tai42_agents.refine_agent.prompt import CRITIC_APPROVAL_MESSAGE

AGENT_NAME = "refine_agent"


class FakeAgent:
    """A stand-in compiled LangGraph agent.

    ``ainvoke`` pops the next canned content off ``invoke_contents`` and wraps it
    in a one-message state (what ``build_user_output`` reads). ``astream`` replays
    a fixed list of ``(mode, chunk)`` items for the final-pass projection.
    """

    def __init__(
        self, invoke_contents: list[str], stream_items: list[Any] | None = None, history: list[Any] | None = None
    ) -> None:
        self._invoke_contents = list(invoke_contents)
        self._stream_items = list(stream_items or [])
        self._history = list(history or [])
        self.ainvoke_inputs: list[Any] = []
        self.astream_inputs: list[Any] = []
        self.get_state_calls = 0

    async def ainvoke(self, agent_input: Any, config: Any) -> dict[str, Any]:
        self.ainvoke_inputs.append(agent_input)
        content = self._invoke_contents.pop(0)
        return {"messages": [AIMessage(content=content)]}

    async def astream(self, agent_input: Any, config: Any, stream_mode: Any = None):
        self.astream_inputs.append(agent_input)
        for item in self._stream_items:
            yield item

    async def aget_state(self, config: Any) -> Any:
        # The structured final-pass reads the loop thread's negotiation history from
        # the checkpoint here; a fake snapshot exposes ``.values["messages"]``.
        self.get_state_calls += 1
        return SimpleNamespace(values={"messages": list(self._history)})


class CreateAgentRecorder:
    """A ``create_agent`` replacement: hands out queued fakes and records the tools
    and ``response_format`` each was compiled with (the latter is ``None`` on the
    text-loop evaluator/critic and the schema only on the structured final pass)."""

    def __init__(self, agents: list[FakeAgent]) -> None:
        self._queue = list(agents)
        self.tools_per_call: list[list[Any]] = []
        self.response_formats: list[Any] = []

    def __call__(
        self,
        llm: Any,
        *,
        tools: Any,
        checkpointer: Any,
        middleware: Any,
        debug: Any,
        response_format: Any = None,
    ) -> FakeAgent:
        self.tools_per_call.append(list(tools))
        self.response_formats.append(response_format)
        return self._queue.pop(0)


class _ProviderSettings:
    llm = "fake-llm"
    checkpoint = "fake-ckpt"
    checkpoint_conn_string = None


class _LlmSettings:
    def with_fallbacks(self, kwargs: dict[str, Any] | None) -> dict[str, Any]:
        return {}


class _CheckpointRegistry:
    async def get_checkpointer(self, provider: str, conn_string: Any) -> None:
        return None


class _LoggingSettings:
    def is_enabled_for(self, level: str) -> bool:
        return False


def _patch_loop(monkeypatch: pytest.MonkeyPatch, agents: list[FakeAgent]) -> CreateAgentRecorder:
    """Monkeypatch every non-loop seam so ``_run_refine_loop`` runs offline
    against the supplied fake evaluator/critic. Returns the create-agent recorder
    (evaluator is compiled first, critic second)."""
    recorder = CreateAgentRecorder(agents)
    monkeypatch.setattr(agent_mod, "create_agent", recorder)

    async def _get_llm_async(provider: str, **kwargs: Any) -> str:
        return "llm"

    monkeypatch.setattr(agent_mod, "get_llm_async", _get_llm_async)
    monkeypatch.setattr(agent_mod, "checkpoint_registry", lambda: _CheckpointRegistry())
    monkeypatch.setattr(agent_mod, "context_overflow_middlewares", list)
    monkeypatch.setattr(agent_mod, "logging_settings", lambda: _LoggingSettings())
    monkeypatch.setattr(agent_mod, "llm_provider_settings", lambda: _ProviderSettings())
    monkeypatch.setattr(agent_mod, "llm_settings", lambda: _LlmSettings())

    def _init_config(config: dict[str, Any] | None) -> dict[str, Any]:
        return config or {"configurable": {"thread_id": "t"}}

    monkeypatch.setattr(agent_mod, "init_langgraph_config", _init_config)
    return recorder


def _final_pass_script() -> list[Any]:
    """A final-evaluator ``astream`` script exercising the whole event taxonomy:
    reasoning, a tool call + result, token deltas, and a structured response."""
    reasoning_and_call = AIMessage(
        content=[{"type": "thinking", "thinking": "weighing the options"}],
        tool_calls=[{"id": "c1", "name": "lookup", "args": {"q": 1}}],
    )
    usage_message = AIMessage(
        content="",
        usage_metadata={"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
        response_metadata={"model_name": "fake-model"},
    )
    tool_result = ToolMessage(content="tool said hi", name="lookup", tool_call_id="c1")
    return [
        ("updates", {"model": {"messages": [reasoning_and_call]}}),
        ("updates", {"tools": {"messages": [tool_result]}}),
        ("updates", {"model": {"messages": [usage_message]}}),
        ("messages", (AIMessageChunk(content="Final "), {})),
        ("messages", (AIMessageChunk(content="answer"), {})),
        ("updates", {"model": {"messages": [], "structured_response": {"ok": True}}}),
    ]


def _collect(agent: Agent, **kwargs: Any) -> list[StreamEvent]:
    async def go() -> list[StreamEvent]:
        return [event async for event in agent.astream(**kwargs)]

    return asyncio.run(go())


def _a_tool() -> StructuredTool:
    def _f(x: int) -> int:
        return x

    return StructuredTool.from_function(func=_f, name="t1", description="a tool")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_decorator_registers_a_live_instance() -> None:
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    assert isinstance(agent, RefineAgent)
    assert isinstance(agent, Agent)
    assert agent.tool_name == AGENT_NAME


# ---------------------------------------------------------------------------
# Tool resolution
# ---------------------------------------------------------------------------


def test_tools_are_resolved_by_name_and_passed_to_both_agents(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    tool = _a_tool()
    app_tools.client_tools["t1"] = tool
    evaluator = FakeAgent(invoke_contents=["draft"], stream_items=[("messages", (AIMessageChunk(content="ok"), {}))])
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    recorder = _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    _collect(agent, evaluator_message="write it", critic_message="review it", tool_names=["t1"])

    assert recorder.tools_per_call == [[tool], [tool]]  # evaluator, then critic


def test_unknown_tool_name_raises(monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any) -> None:
    evaluator = FakeAgent(invoke_contents=["draft"])
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match="unknown client tools"):
        _collect(agent, evaluator_message="write it", critic_message="review it", tool_names=["missing"])


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_approval_on_first_iteration_streams_the_final_pass(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    evaluator = FakeAgent(invoke_contents=["draft"], stream_items=_final_pass_script())
    critic = FakeAgent(invoke_contents=[f"looks great {CRITIC_APPROVAL_MESSAGE}"])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = _collect(agent, evaluator_message="write it", critic_message="review it")

    # loop ran exactly one evaluator+critic round, then the final pass streamed.
    assert len(evaluator.ainvoke_inputs) == 1
    assert len(critic.ainvoke_inputs) == 1
    assert len(evaluator.astream_inputs) == 1
    assert any(isinstance(e, MessageFinal) for e in events)


def test_approval_on_a_later_iteration(monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any) -> None:
    evaluator = FakeAgent(
        invoke_contents=["draft-1", "draft-2", "draft-3"],
        stream_items=[("messages", (AIMessageChunk(content="done"), {}))],
    )
    critic = FakeAgent(invoke_contents=["needs work", "still off", f"ok {CRITIC_APPROVAL_MESSAGE}"])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = _collect(agent, evaluator_message="write it", critic_message="review it", max_iterations=5)

    assert len(critic.ainvoke_inputs) == 3  # approved on the third round
    assert len(evaluator.ainvoke_inputs) == 3
    assert any(isinstance(e, MessageFinal) and e.text == "done" for e in events)


def test_max_iterations_without_approval_raises(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    evaluator = FakeAgent(invoke_contents=["d1", "d2"])
    critic = FakeAgent(invoke_contents=["nope", "still nope"])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match=r"Max iterations \(2\) reached without critic approval"):
        _collect(agent, evaluator_message="write it", critic_message="review it", max_iterations=2)

    assert len(evaluator.ainvoke_inputs) == 2  # both budgeted rounds attempted
    assert len(critic.ainvoke_inputs) == 2


def test_empty_critic_feedback_raises(monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any) -> None:
    evaluator = FakeAgent(invoke_contents=["draft"])
    critic = FakeAgent(invoke_contents=[""])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match="No critic feedback found"):
        _collect(agent, evaluator_message="write it", critic_message="review it")


# ---------------------------------------------------------------------------
# Approval-token detection (single case-sensitive policy)
# ---------------------------------------------------------------------------


def test_exact_token_substring_is_recognized_as_approval(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """The exact token embedded in surrounding text counts as approval, even at
    the last budgeted iteration — the check must actually detect it."""
    evaluator = FakeAgent(invoke_contents=["draft"], stream_items=[("messages", (AIMessageChunk(content="done"), {}))])
    critic = FakeAgent(invoke_contents=[f"All good here: {CRITIC_APPROVAL_MESSAGE} — ship it"])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = _collect(agent, evaluator_message="write it", critic_message="review it", max_iterations=1)

    assert any(isinstance(e, MessageFinal) and e.text == "done" for e in events)


def test_wrong_case_token_is_not_approval(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """Detection is case-sensitive against the exact sentinel: an upper-cased
    look-alike is NOT approval, so the run fails loudly at max_iterations."""
    evaluator = FakeAgent(invoke_contents=["draft"])
    critic = FakeAgent(invoke_contents=[CRITIC_APPROVAL_MESSAGE.upper()])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match="Max iterations"):
        _collect(agent, evaluator_message="write it", critic_message="review it", max_iterations=1)


# ---------------------------------------------------------------------------
# Final-pass event taxonomy
# ---------------------------------------------------------------------------


def test_final_pass_emits_the_full_event_taxonomy(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    evaluator = FakeAgent(invoke_contents=["draft"], stream_items=_final_pass_script())
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = _collect(agent, evaluator_message="write it", critic_message="review it")

    assert [type(e) for e in events] == [
        ReasoningStep,
        ToolCallStep,
        ToolResultStep,
        RunUsage,
        MessageDelta,
        MessageDelta,
        MessageFinal,
        StructuredFinal,
    ]
    reasoning = next(e for e in events if isinstance(e, ReasoningStep))
    assert reasoning.text == "weighing the options"
    call = next(e for e in events if isinstance(e, ToolCallStep))
    assert call.tool == "lookup"
    assert call.call_id == "c1"
    result = next(e for e in events if isinstance(e, ToolResultStep))
    assert result.call_id == "c1"
    assert result.result == "tool said hi"
    usage = next(e for e in events if isinstance(e, RunUsage))
    assert usage.total_tokens == 12
    assert usage.model == "fake-model"
    final = next(e for e in events if isinstance(e, MessageFinal))
    assert final.text == "Final answer"
    structured = next(e for e in events if isinstance(e, StructuredFinal))
    assert structured.data == {"ok": True}


# ---------------------------------------------------------------------------
# run() drains the stream to the final value
# ---------------------------------------------------------------------------


def test_run_drains_stream_to_final_text(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    evaluator = FakeAgent(
        invoke_contents=["draft"],
        stream_items=[
            ("messages", (AIMessageChunk(content="Final "), {})),
            ("messages", (AIMessageChunk(content="answer"), {})),
        ],
    )
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    result = asyncio.run(agent.run(evaluator_message="write it", critic_message="review it"))
    assert result == "Final answer"


_REFINE_SCHEMA = {"title": "Answer", "type": "object", "properties": {"answer": {"type": "string"}}}


def _structured_stream_items(payload: dict[str, Any]) -> list[Any]:
    """A structured final-pass ``astream`` script: one updates chunk carrying the
    forced structured response the projection turns into a ``StructuredFinal``."""
    return [("updates", {"model": {"messages": [], "structured_response": payload}})]


def test_run_with_response_format_forces_final_answer_on_a_fresh_thread(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """With a ``response_format`` set the loop iterations stay text and only the final
    approved answer is forced: a SECOND, structured evaluator (a distinct graph built
    with the schema) runs the final pass, fed the loop thread's negotiation history
    EXPLICITLY as input (read from the checkpoint) rather than via a cross-topology
    checkpoint resume, and ``run`` returns the validated structured object."""
    history = [AIMessage(content="draft under negotiation")]
    evaluator = FakeAgent(invoke_contents=["draft"], history=history)
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    structured = FakeAgent(invoke_contents=[], stream_items=_structured_stream_items({"answer": "final"}))
    recorder = _patch_loop(monkeypatch, [evaluator, critic, structured])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    result = asyncio.run(
        agent.run(evaluator_message="write it", critic_message="review it", response_format=_REFINE_SCHEMA)
    )

    assert result == {"answer": "final"}
    # Only the THIRD create_agent call (the structured final pass) carried the schema;
    # the text-loop evaluator and critic were built without one.
    assert recorder.response_formats == [None, None, _REFINE_SCHEMA]
    # The loop's evaluator (not the structured graph) was resumed via aget_state to
    # read history, and the structured pass is a DISTINCT graph (no cross-topology
    # resume of the text-loop thread).
    assert evaluator.get_state_calls == 1
    assert structured is not evaluator
    # The structured pass was fed the negotiation history + the approval prompt as
    # EXPLICIT input, not resumed from the loop thread's checkpoint.
    fed = structured.astream_inputs[0]["messages"]
    assert fed[0] is history[0]
    assert fed[-1] == {"role": "user", "content": "Critic Approved."}


def test_run_with_response_format_but_no_structured_raises_loudly(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """A requested ``response_format`` the final pass produced none for raises loudly
    rather than returning the approved text."""
    evaluator = FakeAgent(invoke_contents=["draft"], history=[AIMessage(content="ctx")])
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    # The structured pass streams only text — no structured_response is written.
    structured = FakeAgent(invoke_contents=[], stream_items=[("messages", (AIMessageChunk(content="text"), {}))])
    _patch_loop(monkeypatch, [evaluator, critic, structured])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match="no structured output"):
        asyncio.run(agent.run(evaluator_message="write it", critic_message="review it", response_format=_REFINE_SCHEMA))


def test_astream_with_response_format_emits_one_structured_final(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """The stream face surfaces exactly one terminal ``StructuredFinal`` from the
    structured final pass."""
    evaluator = FakeAgent(invoke_contents=["draft"], history=[AIMessage(content="ctx")])
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    structured = FakeAgent(invoke_contents=[], stream_items=_structured_stream_items({"answer": "final"}))
    _patch_loop(monkeypatch, [evaluator, critic, structured])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = _collect(agent, evaluator_message="write it", critic_message="review it", response_format=_REFINE_SCHEMA)
    finals = [e for e in events if isinstance(e, StructuredFinal)]
    assert len(finals) == 1
    assert finals[0].data == {"answer": "final"}


def test_astream_with_response_format_but_no_structured_raises_loudly(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """A requested ``response_format`` the final pass produced no ``StructuredFinal``
    for raises loudly after the stream drains — in parity with ``run`` — rather than
    silently omitting the frame."""
    evaluator = FakeAgent(invoke_contents=["draft"], history=[AIMessage(content="ctx")])
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    # The structured pass streams only text — no structured_response is written.
    structured = FakeAgent(invoke_contents=[], stream_items=[("messages", (AIMessageChunk(content="text"), {}))])
    _patch_loop(monkeypatch, [evaluator, critic, structured])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match="no structured output"):
        _collect(agent, evaluator_message="write it", critic_message="review it", response_format=_REFINE_SCHEMA)


def test_astream_nonconforming_structured_raises(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """A structured final pass violating a schema constraint keyword raises loudly
    from the projection's validation step — pinning that ``response_format`` is
    threaded into the final pass's projection."""
    schema = {
        "title": "Answer",
        "type": "object",
        "properties": {"answer": {"type": "string", "minLength": 1}},
        "required": ["answer"],
    }
    evaluator = FakeAgent(invoke_contents=["draft"], history=[AIMessage(content="ctx")])
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    structured = FakeAgent(invoke_contents=[], stream_items=_structured_stream_items({"answer": ""}))
    _patch_loop(monkeypatch, [evaluator, critic, structured])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(JsonSchemaValidationError):
        _collect(agent, evaluator_message="write it", critic_message="review it", response_format=schema)


def test_run_response_format_without_title_raises_loudly(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """A ``response_format`` dict lacking a top-level ``"title"`` is rejected loudly at
    the run seam before the loop runs."""
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(ValueError, match="top-level 'title'"):
        asyncio.run(
            agent.run(evaluator_message="write it", critic_message="review it", response_format={"type": "object"})
        )


def test_astream_response_format_without_title_raises_loudly(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """The streaming face — the one the public run door drives — rejects an untitled
    ``response_format`` up front, exactly as the invoke face does."""
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(ValueError, match="top-level 'title'"):
        _collect(agent, evaluator_message="write it", critic_message="review it", response_format={"type": "object"})


# ---------------------------------------------------------------------------
# Contract-parameter parity: honor tool_names on both faces; raise on the rest
# ---------------------------------------------------------------------------


def test_tool_names_honored_on_run_face(monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any) -> None:
    """The ``run`` face resolves ``tool_names`` and compiles both agents with the
    live tool — parity with the ``astream`` face, not a silent drop."""
    tool = _a_tool()
    app_tools.client_tools["t1"] = tool
    evaluator = FakeAgent(
        invoke_contents=["draft"],
        stream_items=[("messages", (AIMessageChunk(content="ok"), {}))],
    )
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    recorder = _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    asyncio.run(agent.run(evaluator_message="write it", critic_message="review it", tool_names=["t1"]))

    assert recorder.tools_per_call == [[tool], [tool]]  # evaluator, then critic


# Each unhonored contract parameter with a MEANINGFUL value — a truthy collection or
# a not-``None`` scalar — plus two falsy-but-meaningful scalars (``strategy=""``,
# ``resume=False``) that must still raise (they are set whenever not ``None``, so they
# never slip through a truthiness gate). ``X=None`` / ``X=()`` are the ABC's own
# "not requested" sentinels and are pinned to PASS by the tests below.
_UNHONORED_CASES = [
    ("tools", [object()]),
    ("presets", [object()]),
    ("subagents", [object()]),
    ("strategy", "react"),
    ("strategy", ""),
    ("system_message", "sys"),
    ("user_message", "usr"),
    ("interrupt_on", {"tool": True}),
    ("skills", ["s"]),
    ("inline_skills", [{"name": "s", "content": "c"}]),
    ("recursion_limit", 0),
    ("thread_id", "t"),
    ("resume", False),
    ("resume_checkpoint_id", "cp"),
    ("llm_provider", "openai"),
    ("store_provider", "redis"),
    ("llm_kwargs", {"model": "x"}),
]


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_astream_rejects_unhonored_contract_param(
    param: str, value: Any, app_tools: Any, resource_manager: Any
) -> None:
    """Every contract ``Agent.run`` parameter with no seat in the role-named loop
    raises loudly on the stream face, naming the offending parameter and the exact
    ``astream`` face it was called on. Falsy-but-meaningful scalars (``strategy=""``,
    ``resume=False``) still raise."""
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match=rf"refine_agent\.astream does not support .*\b{param}\b"):
        _collect(agent, evaluator_message="write it", critic_message="review it", **{param: value})


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_run_rejects_unhonored_contract_param(param: str, value: Any, app_tools: Any, resource_manager: Any) -> None:
    """The ``run`` face rejects the same unsupported parameters, naming the exact
    ``run`` face — the run-face guard is load-bearing (were it dropped, the delegated
    ``astream`` guard would surface the ``astream`` token and fail this assertion)."""
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match=rf"refine_agent\.run does not support .*\b{param}\b"):
        # The base ``Agent.run`` signature types some of these non-optional, so the mismatch is expected.
        asyncio.run(agent.run(evaluator_message="write it", critic_message="review it", **{param: value}))  # type: ignore[arg-type]


def test_unhonored_cases_cover_the_full_reasons_map() -> None:
    """Every key in the guard's reasons map has a parametrized reject case, so a key
    added to the map without a matching test fails here immediately."""
    assert {param for param, _ in _UNHONORED_CASES} == set(agent_mod._UNHONORED_REASONS)


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
_COLLECTION_REJECT_PARAMS = sorted(agent_mod._UNHONORED_REASONS.keys() & _EMPTY_COLLECTION_ABC_DEFAULTS)


def test_collection_params_match_the_abc_collection_defaults() -> None:
    """``_UNHONORED_COLLECTION_PARAMS`` is exactly this agent's unhonored params whose ABC
    default is an empty collection: no scalar wrongly listed (which would let a meaningful
    falsy value slip through), none dropped (which would over-reject the not-requested
    empty default)."""
    assert set(agent_mod._UNHONORED_COLLECTION_PARAMS) == set(_COLLECTION_REJECT_PARAMS)


@pytest.mark.parametrize("empty", [[], ""])
@pytest.mark.parametrize("param", _COLLECTION_REJECT_PARAMS)
def test_reject_unhonored_permits_empty_collection_param(param: str, empty: object) -> None:
    """An empty collection is the ABC's "not requested" default for a collection parameter,
    so the guard does not raise for it — in either falsy empty form (``[]`` / ``""``). Were
    the parameter dropped from ``_UNHONORED_COLLECTION_PARAMS`` it would be classified as a
    scalar (set whenever it is not ``None``) and this empty value would raise."""
    reject_unhonored(
        "refine_agent.run",
        {param: empty},
        agent_mod._UNHONORED_REASONS,
        collection_params=agent_mod._UNHONORED_COLLECTION_PARAMS,
    )


@pytest.mark.parametrize(
    "param", ["response_format", "strategy", "thread_id", "resume", "llm_provider", "recursion_limit"]
)
def test_unhonored_scalar_none_passes_the_guard(param: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ABC's ``None`` "not requested" sentinel passes the guard rather than being
    over-rejected: an explicit ``response_format=None`` / ``resume=None`` / … reaches
    the loop and completes, in parity with the LangGraph-backed agents."""
    evaluator = FakeAgent(invoke_contents=["draft"], stream_items=[("messages", (AIMessageChunk(content="ok"), {}))])
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = _collect(agent, evaluator_message="write it", critic_message="review it", **{param: None})
    assert any(isinstance(e, MessageFinal) for e in events)


def test_unhonored_collection_empty_passes_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty collection is the ABC's "not requested" sentinel and passes the guard:
    ``tools=()`` / ``presets=[]`` reach the loop rather than raising."""
    evaluator = FakeAgent(invoke_contents=["draft"], stream_items=[("messages", (AIMessageChunk(content="ok"), {}))])
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = _collect(agent, evaluator_message="write it", critic_message="review it", tools=(), presets=[])
    assert any(isinstance(e, MessageFinal) for e in events)


def test_extension_kwarg_is_not_rejected(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """A plain extension kwarg beyond the contract names is the documented
    extension point and passes through untouched rather than raising."""
    evaluator = FakeAgent(invoke_contents=["draft"], stream_items=[("messages", (AIMessageChunk(content="ok"), {}))])
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = _collect(agent, evaluator_message="write it", critic_message="review it", role_specific_extra="ok")

    assert any(isinstance(e, MessageFinal) for e in events)


# ---------------------------------------------------------------------------
# Input model rejects unknown keys (extra="forbid")
# ---------------------------------------------------------------------------


def test_run_raises_when_required_evaluator_message_slot_is_unset(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """The evaluator message renders with ``allow_empty=False``: an unset slot
    (empty content AND empty id) surfaces the resource manager's fail-loud message
    rather than silently running the loop on an empty prompt. The evaluator/critic
    seams are faked so that WITHOUT render.py's ``allow_empty`` passthrough the loop
    would instead run to approval — a dropped passthrough turns this red rather than
    letting the empty prompt slip through."""
    evaluator = FakeAgent(invoke_contents=["draft"], stream_items=[("messages", (AIMessageChunk(content="ok"), {}))])
    critic = FakeAgent(invoke_contents=[f"approved {CRITIC_APPROVAL_MESSAGE}"])
    _patch_loop(monkeypatch, [evaluator, critic])

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(ValueError, match="must provide either"):
        asyncio.run(agent.run(critic_message="review it"))


def test_tool_input_rejects_unknown_key() -> None:
    """A typo'd key is rejected loudly at validation rather than silently ignored."""
    with pytest.raises(ValidationError, match="max_iteration"):
        RefineAgentInput.model_validate({"evaluator_message": "hi", "max_iteration": 5})
