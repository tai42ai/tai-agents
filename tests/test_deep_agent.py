"""Runtime + wiring tests for the ``deep_agent`` :class:`Agent`.

Covers three layers, all with fakes (no live LLM, no real store):

* ``tool_spec`` — JSON ``DeepSubAgentSpec`` resolves to ``ResolvedSubAgentSpec``
  with tool *names* turned into live tools (one level of nesting), and
  ``response_format`` schema validation.
* ``DeepAgent`` build path — subagents/inline-skills reach the factory, the
  recursion cap and resume-checkpoint pinning land on the config, pending
  interrupts are read from the paused snapshot, and ``astream`` enforces exactly
  one of ``user_message`` / ``resume``.
* ``DeepAgent.astream`` projection — a scripted compiled graph drives every
  contract event kind, including one ``InterruptFinal`` per pending interrupt.

Async code is driven with ``asyncio.run`` (the suite does not use
``pytest-asyncio``).
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
from tai42_contract.agent import Agent
from tai42_contract.agent.base import AgentInterruptedError, PresetSpec
from tai42_contract.agent.base import SubAgentSpec as NeutralSubAgentSpec
from tai42_contract.agent.events import (
    InterruptFinal,
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

from tai42_agents._internal.reject import reject_unhonored
from tai42_agents.deep_agent import agent as agent_mod
from tai42_agents.deep_agent.agent import DeepAgent, DeepAgentInput, _neutral_to_internal
from tai42_agents.deep_agent.spec import InlineSkill, ResolvedSubAgentSpec
from tai42_agents.deep_agent.tool_spec import DeepSubAgentSpec, resolve_subagent_specs


def _client_tool(name: str) -> StructuredTool:
    async def call_tool(**kwargs: object) -> object:
        return kwargs

    return StructuredTool.from_function(
        func=None,
        coroutine=call_tool,
        name=name,
        description="client tool",
        args_schema={"type": "object", "properties": {}, "required": []},
    )


# ===========================================================================
# tool_spec: DeepSubAgentSpec -> ResolvedSubAgentSpec
# ===========================================================================


def test_resolve_subagent_specs_resolves_tool_names_and_nesting(app_tools: Any) -> None:
    for name in ("search", "fetch", "summarize"):
        app_tools.client_tools[name] = _client_tool(name)
    specs = [
        DeepSubAgentSpec(
            name="researcher",
            description="does research",
            system_prompt="research things",
            tools=["search", "fetch"],
            response_format={
                "title": "Summary",
                "type": "object",
                "properties": {"summary": {"type": "string"}},
            },
            subagents=[
                DeepSubAgentSpec(
                    name="summarizer",
                    description="summarizes",
                    system_prompt="summarize",
                    tools=["summarize"],
                )
            ],
        )
    ]

    resolved = asyncio.run(resolve_subagent_specs(specs))

    assert len(resolved) == 1
    parent = resolved[0]
    assert isinstance(parent, ResolvedSubAgentSpec)
    assert [tool.name for tool in parent.tools] == ["search", "fetch"]
    assert all(isinstance(tool, StructuredTool) for tool in parent.tools)

    assert len(parent.subagents) == 1
    child = parent.subagents[0]
    assert isinstance(child, ResolvedSubAgentSpec)
    assert [tool.name for tool in child.tools] == ["summarize"]

    assert parent.response_format == {
        "title": "Summary",
        "type": "object",
        "properties": {"summary": {"type": "string"}},
    }
    assert child.response_format is None


def test_resolve_subagent_specs_passes_inline_skills_through(app_tools: Any) -> None:
    specs = [
        DeepSubAgentSpec(
            name="researcher",
            description="does research",
            system_prompt="research",
            inline_skills=[InlineSkill(name="demo", content="# demo body")],
        )
    ]
    (resolved,) = asyncio.run(resolve_subagent_specs(specs))
    assert resolved.inline_skills is not None
    assert resolved.inline_skills[0].name == "demo"
    assert resolved.inline_skills[0].content == "# demo body"


def test_deep_subagent_spec_rejects_unknown_key() -> None:
    """The JSON sub-agent shape sets ``extra="forbid"``, so a per-sub key deep cannot
    honor — e.g. a per-sub ``strategy`` — is a loud validation error rather than a
    silently dropped field, matching ``DeepAgentInput``'s own strictness at the run door."""
    with pytest.raises(ValidationError):
        DeepSubAgentSpec.model_validate({"name": "s", "description": "d", "system_prompt": "p", "strategy": "vote"})
    with pytest.raises(ValidationError):
        DeepSubAgentSpec.model_validate(
            {"name": "s", "description": "d", "system_prompt": "p", "totally_unknown_key": 1}
        )


def test_resolve_subagent_specs_empty_returns_empty() -> None:
    assert asyncio.run(resolve_subagent_specs(None)) == []
    assert asyncio.run(resolve_subagent_specs([])) == []


def test_resolve_subagent_specs_rejects_response_format_without_title() -> None:
    specs = [
        DeepSubAgentSpec(
            name="researcher",
            description="does research",
            system_prompt="research",
            response_format={"type": "object", "properties": {"x": {"type": "string"}}},
        )
    ]
    with pytest.raises(ValueError, match="top-level 'title'"):
        asyncio.run(resolve_subagent_specs(specs))


# ===========================================================================
# _neutral_to_internal
# ===========================================================================


def test_neutral_to_internal_rejects_strategy() -> None:
    spec = NeutralSubAgentSpec(name="s", description="d", system_prompt="p", strategy="vote")
    with pytest.raises(ValueError, match="strategy"):
        asyncio.run(_neutral_to_internal(spec))


def test_neutral_to_internal_resolves_tools_and_coerces_inline_skills(app_tools: Any) -> None:
    app_tools.client_tools["search"] = _client_tool("search")
    spec = NeutralSubAgentSpec(
        name="s",
        description="d",
        system_prompt="p",
        tool_names=["search"],
        inline_skills=[{"name": "demo", "content": "# demo"}],
        skills=["/skills/ref/"],
    )
    internal = asyncio.run(_neutral_to_internal(spec))
    assert isinstance(internal, ResolvedSubAgentSpec)
    assert [t.name for t in internal.tools] == ["search"]
    assert internal.skills == ["/skills/ref/"]
    assert internal.inline_skills is not None
    assert isinstance(internal.inline_skills[0], InlineSkill)
    assert internal.inline_skills[0].name == "demo"


# ===========================================================================
# DeepAgent build path
# ===========================================================================


class _FakeRegistry:
    async def get_checkpointer(self, **kwargs: object) -> None:
        return None

    async def get_store(self, **kwargs: object) -> None:
        return None


class _FakeProviderSettings:
    llm = "fake"
    checkpoint = "memory"
    checkpoint_conn_string = None
    store = "memory"
    store_conn_string = None


class _FakeLlmSettings:
    def with_fallbacks(self, kwargs: object) -> dict[str, Any]:
        return {}


def _patch_build(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    async def fake_build_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    async def fake_get_llm_async(provider: str, **kwargs: object) -> object:
        return object()

    monkeypatch.setattr(agent_mod, "build_deep_agent", fake_build_deep_agent)
    monkeypatch.setattr(agent_mod, "get_llm_async", fake_get_llm_async)
    monkeypatch.setattr(agent_mod, "checkpoint_registry", _FakeRegistry)
    monkeypatch.setattr(agent_mod, "store_registry", _FakeRegistry)
    monkeypatch.setattr(agent_mod, "llm_provider_settings", _FakeProviderSettings)
    monkeypatch.setattr(agent_mod, "llm_settings", _FakeLlmSettings)
    monkeypatch.setattr(agent_mod, "init_langgraph_config", lambda config=None: dict(config or {}))


def _build_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "tools": [],
        "subagents": [],
        "skills": None,
        "inline_skills": None,
        "system_message": "",
        "response_format": None,
        "interrupt_on": None,
        "thread_id": "thread",
        "resume_checkpoint_id": None,
        "llm_provider": None,
        "checkpoint_provider": None,
        "store_provider": None,
        "llm_kwargs": None,
        "recursion_limit": None,
    }
    base.update(overrides)
    return base


def test_build_agent_passes_subagents_to_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _patch_build(monkeypatch, captured)

    spec = ResolvedSubAgentSpec(name="helper", description="d", system_prompt="p")
    agent: Any = DeepAgent()
    asyncio.run(agent._build_agent(**_build_kwargs(subagents=[spec])))
    assert captured["subagents"] == [spec]

    captured.clear()
    asyncio.run(agent._build_agent(**_build_kwargs(subagents=[])))
    assert captured["subagents"] is None


def test_build_agent_forwards_inline_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    """inline_skills reach build_deep_agent on the stream path; empty → None."""
    captured: dict[str, Any] = {}
    _patch_build(monkeypatch, captured)

    agent: Any = DeepAgent()
    skill = InlineSkill(name="demo", content="# demo")
    asyncio.run(agent._build_agent(**_build_kwargs(inline_skills=[skill])))
    assert captured["inline_skills"] == [skill]

    captured.clear()
    asyncio.run(agent._build_agent(**_build_kwargs(inline_skills=[])))
    assert captured["inline_skills"] is None


def test_build_agent_caps_recursion_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_build(monkeypatch, {})
    agent: Any = DeepAgent()
    _, config = asyncio.run(agent._build_agent(**_build_kwargs(recursion_limit=4242)))
    assert config["recursion_limit"] == 4242


def test_build_agent_pins_resume_checkpoint_in_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_build(monkeypatch, {})
    agent: Any = DeepAgent()
    _, config = asyncio.run(agent._build_agent(**_build_kwargs(resume_checkpoint_id="cp-7")))
    assert config["configurable"]["checkpoint_id"] == "cp-7"


def test_build_agent_omits_thread_id_when_keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    """A keyless one-shot stream must not pin ``thread_id=None`` (which collides all
    keyless runs on the shared checkpoint thread); the config carries no thread_id
    key, so ``init_langgraph_config`` mints a fresh isolated one instead."""
    _patch_build(monkeypatch, {})
    agent: Any = DeepAgent()
    _, config = asyncio.run(agent._build_agent(**_build_kwargs(thread_id=None)))
    assert "thread_id" not in config["configurable"]


def test_astream_resolves_json_subagent_specs_from_the_run_door(app_tools: Any) -> None:
    """The public SSE run door validates ``subagents`` into JSON ``DeepSubAgentSpec``
    (tool NAMES, no live tools) and hands them to ``astream``. astream resolves that
    shape — not only the programmatic ``NeutralSubAgentSpec`` — via ``_to_internal``,
    so a deep run WITH subagents does not die with an AttributeError over the streaming
    door (which is what happened when astream only understood the neutral shape)."""
    app_tools.client_tools["search"] = _client_tool("search")
    json_spec = DeepSubAgentSpec(name="helper", description="d", system_prompt="p", tools=["search"])
    resolved = asyncio.run(agent_mod._to_internal(json_spec))
    assert isinstance(resolved, ResolvedSubAgentSpec)
    assert resolved.name == "helper"
    # The neutral shape still resolves (both faces accept both).
    neutral = NeutralSubAgentSpec(name="native", description="d", system_prompt="p")
    same = asyncio.run(agent_mod._to_internal(neutral))
    assert same.name == "native"


def test_pending_interrupts_reads_paused_snapshot() -> None:
    class FakeAgent:
        async def aget_state(self, config: object) -> SimpleNamespace:
            return SimpleNamespace(interrupts=[SimpleNamespace(id="i-9", value={"q": "?"})])

    agent: Any = DeepAgent()
    (interrupt,) = asyncio.run(agent._pending_interrupts(FakeAgent(), {}))
    assert (interrupt.interrupt_id, interrupt.payload) == ("i-9", {"q": "?"})


def test_astream_requires_exactly_one_of_message_or_resume() -> None:
    agent = DeepAgent()

    async def drain(gen: Any) -> None:
        async for _ in gen:
            pass

    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(drain(agent.astream(thread_id="t")))
    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(drain(agent.astream(thread_id="t", user_message="hi", resume={"x": 1})))


# ===========================================================================
# DeepAgent.run wiring (JSON tool-face -> drained streaming core)
# ===========================================================================


def test_deep_agent_registers() -> None:
    agent = tai42_app.agents.get_agent("deep_agent")
    assert isinstance(agent, DeepAgent)
    assert isinstance(agent, Agent)


def test_deep_agent_input_rejects_unknown_key() -> None:
    """``DeepAgentInput`` sets ``extra="forbid"`` so an unknown key at the run door is a
    loud validation error rather than a silently dropped typo."""
    with pytest.raises(ValidationError):
        DeepAgentInput.model_validate({"totally_unknown_key": 1})
    # A field the runtime does not honor (deep applies no composition strategy) is
    # simply not part of the schema, so it is rejected like any other unknown key.
    with pytest.raises(ValidationError):
        DeepAgentInput.model_validate({"strategy": "vote"})


def test_deep_agent_astream_rejects_response_format_without_title() -> None:
    """The streaming face — the one the public run door drives — rejects an untitled
    ``response_format`` up front, exactly as the invoke face does."""
    with pytest.raises(ValueError, match="top-level 'title'"):
        _drain_astream(
            DeepAgent().astream(
                tool_names=[],
                user_message="go",
                response_format={"type": "object", "properties": {"x": {"type": "string"}}},
            )
        )


def test_deep_agent_rejects_response_format_without_title() -> None:
    with pytest.raises(ValueError, match="top-level 'title'"):
        asyncio.run(
            DeepAgent().run(
                tool_names=[],
                user_message="go",
                response_format={"type": "object", "properties": {"x": {"type": "string"}}},
            )
        )


def _install_fake_resolve(monkeypatch: pytest.MonkeyPatch, agent: DeepAgent, graph: _FakeCompiledGraph) -> None:
    """Point the agent's ``_resolve_and_build`` at a scripted compiled graph, so
    ``run`` drains the fake through the shared streaming core with no live LLM."""

    async def fake_resolve_and_build(**_: Any) -> _FakeCompiledGraph:
        return graph

    monkeypatch.setattr(agent, "_resolve_and_build", fake_resolve_and_build)


def test_run_drains_streaming_core_with_resolved_inputs(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """run resolves tool names + JSON subagents + rendered messages, then drains
    the shared streaming core (the same one ``astream`` yields) to a final value."""
    for name in ("calc", "search"):
        app_tools.client_tools[name] = _client_tool(name)

    captured: dict[str, Any] = {}
    graph = _FakeCompiledGraph(_scripted_chunks(), interrupts=[])

    async def fake_resolve_and_build(**kwargs: Any) -> _FakeCompiledGraph:
        captured.update(kwargs)
        return graph

    agent: Any = DeepAgent()
    monkeypatch.setattr(agent, "_resolve_and_build", fake_resolve_and_build)

    subagents = [
        DeepSubAgentSpec(name="researcher", description="does research", system_prompt="research", tools=["search"])
    ]

    result = asyncio.run(
        agent.run(
            tool_names=["calc"],
            subagents=subagents,
            skills=["/skills/foo"],
            inline_skills=[InlineSkill(name="demo", content="# demo")],
            system_message="SYS",
            user_message="go",
            interrupt_on={"calc": True},
            response_format={"title": "N", "type": "object", "properties": {"n": {"type": "integer"}}},
            langgraph_config={"configurable": {"thread_id": "t"}},
        )
    )

    # The scripted run emits a StructuredFinal, so the drained value is the
    # structured object (response_format was requested).
    assert result == {"answer": "ok"}

    # tool names -> live tools; JSON subagents -> resolved core specs.
    assert [tool.name for tool in captured["tools"]] == ["calc"]
    assert len(captured["subagents"]) == 1
    sub = captured["subagents"][0]
    assert isinstance(sub, ResolvedSubAgentSpec)
    assert [tool.name for tool in sub.tools] == ["search"]

    # Rendered system prompt + skills/inline-skills/interrupt_on reach the builder.
    assert captured["system_message"] == "SYS"
    assert captured["skills"] == ["/skills/foo"]
    assert [s.name for s in captured["inline_skills"]] == ["demo"]
    assert captured["interrupt_on"] == {"calc": True}

    # A fresh turn feeds the rendered user message as the graph input.
    assert graph.received_input == {"messages": [{"role": "user", "content": "go"}]}


def test_run_raises_agent_interrupted_on_paused_run(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """A run that pauses on an interrupt raises AgentInterruptedError (the terminal
    rule), even with response_format set — never a pre-interrupt value or a
    misleading RuntimeError about missing structured output."""
    graph = _FakeCompiledGraph(_scripted_chunks(), interrupts=[SimpleNamespace(id="i-run", value={"q": "pick"})])
    agent: Any = DeepAgent()
    _install_fake_resolve(monkeypatch, agent, graph)

    with pytest.raises(AgentInterruptedError) as excinfo:
        asyncio.run(
            agent.run(
                user_message="go",
                interrupt_on={"task": True},
                response_format={"title": "N", "type": "object", "properties": {"n": {"type": "integer"}}},
            )
        )
    assert [intr.interrupt_id for intr in excinfo.value.interrupts] == ["i-run"]


def test_run_resume_feeds_a_command(monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any) -> None:
    """run with a resume payload feeds a langgraph Command(resume=...) — no user
    message rendered — and drains the resumed run to its value."""
    from langgraph.types import Command

    graph = _FakeCompiledGraph([], interrupts=[])
    agent: Any = DeepAgent()
    _install_fake_resolve(monkeypatch, agent, graph)

    asyncio.run(agent.run(resume={"answer": 1}, resume_checkpoint_id="cp-7", thread_id="t"))

    assert isinstance(graph.received_input, Command)
    assert graph.received_input.resume == {"answer": 1}


def test_run_rejects_resume_with_user_message(app_tools: Any, resource_manager: Any) -> None:
    """run cannot both answer an interrupt and start a fresh turn — a resume with a
    user message raises loudly rather than silently dropping one."""
    with pytest.raises(ValueError, match="exactly one of user_message or resume"):
        asyncio.run(DeepAgent().run(resume={"x": 1}, user_message="go"))


def test_run_rejects_neither_message_nor_resume() -> None:
    """run with neither a user_message/user_message_id nor resume raises in the
    caller's own vocabulary — the same exactly-one-of guard astream uses — rather
    than falling through to the resource manager's template-vocabulary message."""
    with pytest.raises(ValueError, match="exactly one of user_message or resume") as excinfo:
        asyncio.run(DeepAgent().run())
    assert "template" not in str(excinfo.value)


def test_run_honors_live_tools(monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any) -> None:
    """run honors live ``tools`` (as astream does): they reach the builder combined
    with the client tools resolved from ``tool_names`` — never silently dropped."""
    app_tools.client_tools["calc"] = _client_tool("calc")
    live = _client_tool("live")

    captured: dict[str, Any] = {}

    async def fake_resolve_and_build(**kwargs: Any) -> _FakeCompiledGraph:
        captured.update(kwargs)
        return _FakeCompiledGraph(_scripted_chunks(), interrupts=[])

    agent: Any = DeepAgent()
    monkeypatch.setattr(agent, "_resolve_and_build", fake_resolve_and_build)

    asyncio.run(agent.run(tools=[live], tool_names=["calc"], user_message="go"))
    assert [tool.name for tool in captured["tools"]] == ["live", "calc"]


def test_run_rejects_presets() -> None:
    """Main-agent presets are not a composable deep_agent input; run raises loudly
    rather than silently dropping them."""
    with pytest.raises(RuntimeError, match="does not support presets"):
        asyncio.run(DeepAgent().run(user_message="go", presets=[PresetSpec(name="p", base_tool="calc")]))


def test_run_rejects_strategy() -> None:
    """deep_agent applies no composition strategy; run raises rather than ignoring one."""
    with pytest.raises(RuntimeError, match="does not support strategy"):
        asyncio.run(DeepAgent().run(user_message="go", strategy="vote"))


def test_run_names_both_offenders_at_once() -> None:
    """A caller passing BOTH unhonored params is named both at once — one message
    lists ``presets`` and ``strategy`` together, so the caller fixes both in one pass
    rather than one raise per run."""
    with pytest.raises(RuntimeError, match=r"does not support presets, strategy"):
        asyncio.run(
            DeepAgent().run(user_message="go", presets=[PresetSpec(name="p", base_tool="calc")], strategy="vote")
        )


# Every key in ``_UNHONORED_REASONS`` paired with a representative SET value. Both
# faces reject each, so dropping a key from the map fails a case here.
_UNHONORED_CASES = [
    ("presets", [PresetSpec(name="p", base_tool="calc")]),
    ("strategy", "vote"),
]


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_run_rejects_every_unhonored_param(param: str, value: Any) -> None:
    """run rejects every key in the guard's reasons map, naming it and the run face."""
    with pytest.raises(RuntimeError, match=rf"deep_agent\.run does not support .*\b{param}\b"):
        asyncio.run(DeepAgent().run(user_message="go", **{param: value}))


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_astream_rejects_every_unhonored_param(param: str, value: Any) -> None:
    """astream rejects the same full set as run — parity — naming the astream face."""
    with pytest.raises(RuntimeError, match=rf"deep_agent\.astream does not support .*\b{param}\b"):
        _drain_astream(DeepAgent().astream(user_message="go", **{param: value}))


def test_unhonored_cases_cover_the_full_reasons_map() -> None:
    """Every key in the reasons map has a parametrized reject case; a key added without
    a test fails here immediately."""
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
        "deep_agent.run",
        {param: empty},
        agent_mod._UNHONORED_REASONS,
        collection_params=agent_mod._UNHONORED_COLLECTION_PARAMS,
    )


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_run_rejects_blank_memory_key(key: str, blank: str) -> None:
    """A present-but-blank memory key is malformed — it would silently share a
    checkpoint namespace across runs — so run raises rather than minting/overlaying it."""
    with pytest.raises(ValueError, match=rf"deep_agent\.run: {key} must be a non-empty string"):
        # The dynamic key spreads into a typed run parameter, so the arg-type mismatch is expected.
        asyncio.run(DeepAgent().run(user_message="go", **{key: blank}))  # type: ignore[arg-type]


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_astream_rejects_blank_memory_key(key: str, blank: str) -> None:
    """Parity with run: astream rejects a present-but-blank memory key loudly."""
    with pytest.raises(ValueError, match=rf"deep_agent\.astream: {key} must be a non-empty string"):
        # The dynamic key spreads into a typed astream parameter, so the arg-type mismatch is expected.
        _drain_astream(DeepAgent().astream(user_message="go", **{key: blank}))  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [123, ["x"]])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_run_rejects_non_string_memory_key(key: str, value: Any) -> None:
    """A non-string memory key is a type violation — it cannot name a checkpoint
    namespace — so run raises TypeError naming the offending param and the received
    type rather than probing a non-string for whitespace."""
    with pytest.raises(
        TypeError, match=rf"deep_agent\.run: {key} must be a string or None; got {type(value).__name__}"
    ):
        # The dynamic key spreads into a typed run parameter, so the arg-type mismatch is expected.
        asyncio.run(DeepAgent().run(user_message="go", **{key: value}))  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [123, ["x"]])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_astream_rejects_non_string_memory_key(key: str, value: Any) -> None:
    """Parity with run: astream rejects a non-string memory key with a TypeError naming
    the offending param and the received type."""
    with pytest.raises(
        TypeError, match=rf"deep_agent\.astream: {key} must be a string or None; got {type(value).__name__}"
    ):
        # The dynamic key spreads into a typed astream parameter, so the arg-type mismatch is expected.
        _drain_astream(DeepAgent().astream(user_message="go", **{key: value}))  # type: ignore[arg-type]


def test_run_config_pins_thread_and_checkpoint() -> None:
    """_run_config overlays an explicit thread_id / resume_checkpoint_id /
    recursion_limit onto a caller-supplied langgraph_config base."""
    config = DeepAgent._run_config({"configurable": {"extra": 1}}, "t", "cp-7", 99)
    assert config["configurable"]["thread_id"] == "t"
    assert config["configurable"]["checkpoint_id"] == "cp-7"
    assert config["configurable"]["extra"] == 1
    assert config["recursion_limit"] == 99


def test_astream_honors_langgraph_config_thread_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """astream honors a ``configurable.thread_id`` carried in the caller's
    ``langgraph_config`` — the same overlay run applies — so a pinned conversation
    thread reaches the compiled graph's checkpointer over the streaming face too.
    The config's other ``configurable`` keys survive the overlay."""
    graph = _FakeCompiledGraph(_scripted_chunks(), interrupts=[])
    agent: Any = DeepAgent()
    _install_fake_resolve(monkeypatch, agent, graph)

    async def collect() -> list[Any]:
        return [
            event
            async for event in agent.astream(
                user_message="go",
                langgraph_config={"configurable": {"thread_id": "T-42", "tenant": "acme"}},
            )
        ]

    asyncio.run(collect())
    assert graph.received_config["configurable"]["thread_id"] == "T-42"
    assert graph.received_config["configurable"]["tenant"] == "acme"


def test_astream_explicit_thread_id_wins_over_langgraph_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The explicit ``thread_id`` parameter overlays the caller's ``langgraph_config``
    on the streaming face, exactly as it does on run."""
    graph = _FakeCompiledGraph(_scripted_chunks(), interrupts=[])
    agent: Any = DeepAgent()
    _install_fake_resolve(monkeypatch, agent, graph)

    async def collect() -> list[Any]:
        return [
            event
            async for event in agent.astream(
                user_message="go",
                thread_id="explicit",
                langgraph_config={"configurable": {"thread_id": "from-config"}},
            )
        ]

    asyncio.run(collect())
    assert graph.received_config["configurable"]["thread_id"] == "explicit"


def test_run_config_mints_fresh_thread_when_keyless() -> None:
    """With no thread pinned, each run gets a fresh isolated thread_id (never a
    shared/None key that would collide runs on one checkpoint thread)."""
    first = DeepAgent._run_config(None, None, None, None)
    second = DeepAgent._run_config(None, None, None, None)
    t1 = first["configurable"]["thread_id"]
    t2 = second["configurable"]["thread_id"]
    assert t1
    assert t2
    assert t1 != t2


# ===========================================================================
# DeepAgent.astream projection — every event kind + the interrupt path
# ===========================================================================


class _FakeCompiledGraph:
    """A scripted compiled graph: ``astream`` replays fixed ``(mode, chunk)``
    pairs and ``aget_state`` reports the interrupts a paused run waits on."""

    def __init__(self, chunks: list[tuple[str, Any]], interrupts: list[Any]) -> None:
        self._chunks = chunks
        self._interrupts = interrupts
        self.received_input: Any = None
        self.received_config: Any = None

    async def astream(self, agent_input: Any, config: Any, stream_mode: Any = None) -> Any:
        self.received_input = agent_input
        self.received_config = config
        for chunk in self._chunks:
            yield chunk

    async def aget_state(self, config: Any) -> SimpleNamespace:
        return SimpleNamespace(interrupts=self._interrupts)


def _scripted_chunks() -> list[tuple[str, Any]]:
    reasoning_and_call = AIMessage(
        content=[{"type": "thinking", "thinking": "let me plan"}],
        tool_calls=[{"name": "task", "args": {"description": "sub"}, "id": "call_1", "type": "tool_call"}],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "scripted"},
    )
    tool_result = ToolMessage(content="sub-result", name="task", tool_call_id="call_1", status="success")
    return [
        ("updates", {"agent": {"messages": [reasoning_and_call]}}),
        ("updates", {"tools": {"messages": [tool_result]}}),
        ("messages", (AIMessageChunk(content="Hello "), {})),
        ("messages", (AIMessageChunk(content="world"), {})),
        ("updates", {"agent": {"structured_response": {"answer": "ok"}}}),
    ]


def _install_fake_graph(monkeypatch: pytest.MonkeyPatch, agent: DeepAgent, graph: _FakeCompiledGraph) -> None:
    async def fake_build_agent(**kwargs: Any) -> tuple[_FakeCompiledGraph, dict[str, Any]]:
        return graph, {"configurable": {"thread_id": "t"}}

    monkeypatch.setattr(agent, "_build_agent", fake_build_agent)


def test_astream_emits_every_event_kind_and_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = DeepAgent()
    graph = _FakeCompiledGraph(
        _scripted_chunks(),
        interrupts=[SimpleNamespace(id="i-1", value={"q": "pick"})],
    )
    _install_fake_graph(monkeypatch, agent, graph)

    async def collect() -> list[Any]:
        return [event async for event in agent.astream(user_message="go", interrupt_on={"task": True}, thread_id="t")]

    events = asyncio.run(collect())
    by_type = {type(e) for e in events}
    assert ReasoningStep in by_type
    assert ToolCallStep in by_type
    assert ToolResultStep in by_type
    assert MessageDelta in by_type
    assert MessageFinal in by_type
    assert RunUsage in by_type
    assert StructuredFinal in by_type
    assert InterruptFinal in by_type

    reasoning = next(e for e in events if isinstance(e, ReasoningStep))
    assert reasoning.text == "let me plan"

    # A subagent invocation surfaces as one `task` call/result pair.
    call = next(e for e in events if isinstance(e, ToolCallStep))
    assert call.tool == "task"
    assert call.call_id == "call_1"
    result = next(e for e in events if isinstance(e, ToolResultStep))
    assert result.tool == "task"
    assert result.call_id == "call_1"
    assert result.is_error is False

    deltas = [e.text for e in events if isinstance(e, MessageDelta)]
    assert deltas == ["Hello ", "world"]
    final = next(e for e in events if isinstance(e, MessageFinal))
    assert final.text == "Hello world"

    usage = next(e for e in events if isinstance(e, RunUsage))
    assert (usage.input_tokens, usage.output_tokens, usage.model) == (10, 5, "scripted")

    structured = next(e for e in events if isinstance(e, StructuredFinal))
    assert structured.data == {"answer": "ok"}

    interrupt = next(e for e in events if isinstance(e, InterruptFinal))
    assert (interrupt.interrupt_id, interrupt.payload) == ("i-1", {"q": "pick"})

    # A fresh turn feeds the built input messages (not a resume Command).
    assert graph.received_input == {"messages": [{"role": "user", "content": "go"}]}


def test_astream_omits_interrupt_read_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no interrupt_on the paused-state read is skipped, so no InterruptFinal."""
    agent = DeepAgent()
    graph = _FakeCompiledGraph(
        _scripted_chunks(),
        interrupts=[SimpleNamespace(id="i-1", value={"q": "pick"})],
    )
    _install_fake_graph(monkeypatch, agent, graph)

    async def collect() -> list[Any]:
        return [event async for event in agent.astream(user_message="go", thread_id="t")]

    events = asyncio.run(collect())
    assert not any(isinstance(e, InterruptFinal) for e in events)


def test_astream_resume_feeds_a_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resume payload is delivered as a langgraph Command(resume=...)."""
    from langgraph.types import Command

    agent = DeepAgent()
    graph = _FakeCompiledGraph([], interrupts=[])
    _install_fake_graph(monkeypatch, agent, graph)

    async def collect() -> list[Any]:
        return [event async for event in agent.astream(resume={"answer": 1}, thread_id="t")]

    asyncio.run(collect())
    assert isinstance(graph.received_input, Command)
    assert graph.received_input.resume == {"answer": 1}


def test_astream_honors_tool_names(monkeypatch: pytest.MonkeyPatch, app_tools: Any) -> None:
    """astream honors ``tool_names`` (as run does): they resolve to client tools and
    reach the builder combined with live ``tools`` — never silently dropped."""
    app_tools.client_tools["calc"] = _client_tool("calc")
    live = _client_tool("live")

    captured: dict[str, Any] = {}

    async def fake_build_agent(**kwargs: Any) -> tuple[_FakeCompiledGraph, dict[str, Any]]:
        captured.update(kwargs)
        return _FakeCompiledGraph([], interrupts=[]), {"configurable": {"thread_id": "t"}}

    agent: Any = DeepAgent()
    monkeypatch.setattr(agent, "_build_agent", fake_build_agent)

    async def collect() -> list[Any]:
        return [event async for event in agent.astream(user_message="go", tools=[live], tool_names=["calc"])]

    asyncio.run(collect())
    assert [tool.name for tool in captured["tools"]] == ["live", "calc"]


def _drain_astream(gen: Any) -> None:
    async def drain() -> None:
        async for _ in gen:
            pass

    asyncio.run(drain())


def test_astream_rejects_presets() -> None:
    """Main-agent presets are not a composable deep_agent input; astream raises loudly
    rather than silently dropping them (parity with run)."""
    with pytest.raises(RuntimeError, match="does not support presets"):
        _drain_astream(DeepAgent().astream(user_message="go", presets=[PresetSpec(name="p", base_tool="calc")]))


def test_astream_rejects_strategy() -> None:
    """deep_agent applies no composition strategy; astream raises rather than ignoring
    one (parity with run)."""
    with pytest.raises(RuntimeError, match="does not support strategy"):
        _drain_astream(DeepAgent().astream(user_message="go", strategy="vote"))


# ===========================================================================
# Structured-final parity: run and astream agree on the same response_format
# ===========================================================================

_RESPONSE_FORMAT: dict[str, Any] = {
    "title": "N",
    "type": "object",
    "properties": {"n": {"type": "integer"}},
}


def _chunks_without_structured() -> list[tuple[str, Any]]:
    """A scripted run that answers in text but writes no ``structured_response`` —
    so a requested ``response_format`` yields no ``StructuredFinal``."""
    return [("messages", (AIMessageChunk(content="Hello"), {}))]


def test_structured_final_parity_when_structured_is_produced(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the SAME response_format on both faces and a structured result produced:
    ``run`` returns the structured value AND ``astream`` emits one ``StructuredFinal``."""
    run_agent: Any = DeepAgent()
    _install_fake_resolve(monkeypatch, run_agent, _FakeCompiledGraph(_scripted_chunks(), interrupts=[]))
    result = asyncio.run(run_agent.run(user_message="go", response_format=_RESPONSE_FORMAT))
    assert result == {"answer": "ok"}

    stream_agent = DeepAgent()
    _install_fake_graph(monkeypatch, stream_agent, _FakeCompiledGraph(_scripted_chunks(), interrupts=[]))

    async def collect() -> list[Any]:
        return [event async for event in stream_agent.astream(user_message="go", response_format=_RESPONSE_FORMAT)]

    events = asyncio.run(collect())
    structured = [event for event in events if isinstance(event, StructuredFinal)]
    assert len(structured) == 1
    assert structured[0].data == {"answer": "ok"}


def test_structured_final_parity_when_structured_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the SAME response_format on both faces and NO structured result: ``run``
    raises the missing-structured RuntimeError AND ``astream`` raises the SAME error
    out of the generator (no silent omission of the ``StructuredFinal`` frame)."""
    run_agent: Any = DeepAgent()
    _install_fake_resolve(monkeypatch, run_agent, _FakeCompiledGraph(_chunks_without_structured(), interrupts=[]))
    with pytest.raises(RuntimeError) as run_exc:
        asyncio.run(run_agent.run(user_message="go", response_format=_RESPONSE_FORMAT))

    stream_agent = DeepAgent()
    _install_fake_graph(monkeypatch, stream_agent, _FakeCompiledGraph(_chunks_without_structured(), interrupts=[]))

    async def drain() -> None:
        async for _ in stream_agent.astream(user_message="go", response_format=_RESPONSE_FORMAT):
            pass

    with pytest.raises(RuntimeError) as stream_exc:
        asyncio.run(drain())

    # The stream face raises the SAME error the invoke face raises via _drain.
    assert str(stream_exc.value) == str(run_exc.value)
    assert "structured output" in str(stream_exc.value)


def test_astream_nonconforming_structured_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A structured result violating a schema constraint keyword raises loudly from
    the projection's validation step — pinning that ``response_format`` is threaded
    through ``_astream_built`` into the projection."""
    schema = {
        "title": "N",
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 0}},
        "required": ["n"],
    }
    agent = DeepAgent()
    chunks: list[tuple[str, Any]] = [("updates", {"agent": {"structured_response": {"n": -1}}})]
    _install_fake_graph(monkeypatch, agent, _FakeCompiledGraph(chunks, interrupts=[]))
    with pytest.raises(JsonSchemaValidationError):
        _drain_astream(agent.astream(user_message="go", response_format=schema))


def test_astream_does_not_raise_missing_structured_when_interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A paused streaming run surfaces its ``InterruptFinal`` and does NOT raise the
    missing-structured error even with response_format set — the run paused rather
    than finished (mirrors ``_drain``, where an interrupt takes precedence)."""
    agent = DeepAgent()
    graph = _FakeCompiledGraph(
        _chunks_without_structured(), interrupts=[SimpleNamespace(id="i-1", value={"q": "pick"})]
    )
    _install_fake_graph(monkeypatch, agent, graph)

    async def collect() -> list[Any]:
        return [
            event
            async for event in agent.astream(
                user_message="go", interrupt_on={"task": True}, response_format=_RESPONSE_FORMAT
            )
        ]

    events = asyncio.run(collect())
    assert any(isinstance(event, InterruptFinal) for event in events)
    assert not any(isinstance(event, StructuredFinal) for event in events)
