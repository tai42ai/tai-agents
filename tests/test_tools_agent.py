"""Tests for the ``tools_agent`` :class:`Agent`.

Covers registration through the real decorator, tool resolution through the app
tool facet (including a preset routed through ``tool_runners``), the ``astream``
projection of the full contract event taxonomy from a scripted stream, ``run``
returning the invoked agent's final text, and the ``StructuredFinal`` path. No
LLM or network: the invoke / stream seams are monkeypatched to scripted doubles.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError
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
from tai42_contract.agent.base import PresetSpec, SubAgentSpec
from tai42_contract.app import tai42_app

from tai42_agents import tools_agent as tools_agent_module
from tai42_agents._internal.reject import reject_unhonored
from tai42_agents._internal.resolve_tools import resolve_tools
from tai42_agents._internal.usage import AgentInvokeResult, CallUsage
from tai42_agents.tools_agent import ToolsAgent, ToolsAgentInput

AGENT_NAME = "tools_agent"


def make_tool(name: str, props: dict[str, Any] | None = None) -> StructuredTool:
    async def call_tool(**kwargs: Any) -> Any:
        return kwargs

    return StructuredTool.from_function(
        func=None,
        coroutine=call_tool,
        name=name,
        description="t",
        args_schema={"type": "object", "properties": props or {}, "required": []},
    )


def _get_agent() -> ToolsAgent:
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    assert isinstance(agent, ToolsAgent)
    return agent


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_decorator_registers_a_live_instance() -> None:
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    assert isinstance(agent, ToolsAgent)
    assert isinstance(agent, Agent)
    assert agent.tool_name == AGENT_NAME
    assert agent.ToolInput is ToolsAgentInput


# --------------------------------------------------------------------------
# ToolInput validation
# --------------------------------------------------------------------------


def test_tool_input_schema_is_json_able_and_has_no_live_tools_field() -> None:
    """ToolInput carries only JSON-able fields — no live ``tools`` field — so its
    JSON schema builds without a vendor type."""
    schema = ToolsAgentInput.model_json_schema()
    assert "tools" not in schema["properties"]
    assert set(schema["properties"]) >= {"tool_names", "presets", "system_message", "user_message"}


def test_tool_input_parses_presets_as_models() -> None:
    parsed = ToolsAgentInput.model_validate(
        {
            "tool_names": ["search"],
            "presets": [{"name": "my_flow", "base_tool": "flow", "fixed_kwargs": {"flow_graph": {}}}],
        }
    )
    assert parsed.tool_names == ["search"]
    assert parsed.presets is not None
    assert isinstance(parsed.presets[0], PresetSpec)
    assert parsed.presets[0].base_tool == "flow"


def test_tool_input_rejects_wrong_typed_field() -> None:
    with pytest.raises(ValidationError):
        ToolsAgentInput.model_validate({"tool_names": "not-a-list"})


# --------------------------------------------------------------------------
# Authorability: the composable spec field set + the from_tool_input mapping
# --------------------------------------------------------------------------


def test_spec_runnable_is_true() -> None:
    """``tools_agent`` declares itself authorable, so a deployment that registers it
    has a composable base agent for the authoring UI."""
    assert ToolsAgent.spec_runnable is True


def test_tool_input_advertises_only_the_honored_composable_fields() -> None:
    """``ToolsAgentInput`` advertises exactly the composable spec fields this agent's
    runtime honors — ``system_prompt`` / ``tool_names`` / ``presets`` /
    ``response_format``. The composable fields the runtime cannot honor —
    ``subagents`` / ``strategy`` — are absent from the model and its JSON schema
    entirely (schema-as-capability), not advertised and then rejected at run time."""
    fields = set(ToolsAgentInput.model_fields)
    assert {"subagents", "strategy"}.isdisjoint(fields)
    assert {"system_prompt", "tool_names", "presets", "response_format"} <= fields

    parsed = ToolsAgentInput.model_validate(
        {
            "tool_names": ["search"],
            "presets": [{"name": "p", "base_tool": "flow", "fixed_kwargs": {}}],
            "system_prompt": "you are helpful",
        }
    )
    assert parsed.presets is not None
    assert isinstance(parsed.presets[0], PresetSpec)
    assert parsed.system_prompt == "you are helpful"

    schema = ToolsAgentInput.model_json_schema()
    assert {"system_prompt", "tool_names", "presets", "response_format"} <= set(schema["properties"])
    assert {"subagents", "strategy"}.isdisjoint(schema["properties"])


def test_tool_input_response_format_round_trips() -> None:
    """``response_format`` (a JSON-Schema dict) round-trips through the ToolInput and
    the ``from_tool_input`` mapping — the field feeds the preset/authoring path."""
    schema = {"title": "Answer", "type": "object", "properties": {"value": {"type": "integer"}}}
    assert "response_format" in ToolsAgentInput.model_json_schema()["properties"]
    validated = ToolsAgentInput.model_validate({"user_message": "hi", "response_format": schema})
    run_kwargs = ToolsAgent.from_tool_input(validated)
    assert run_kwargs["response_format"] == schema


def test_tool_input_rejects_unknown_key() -> None:
    """``extra="forbid"`` turns an unknown key into a loud validation error rather
    than a silently ignored field — including the composable fields this agent drops,
    so a spec that bakes ``strategy`` / ``subagents`` / ``response_format`` fails loud
    instead of having them silently vanish."""
    with pytest.raises(ValidationError):
        ToolsAgentInput.model_validate({"strategy": "vote"})
    with pytest.raises(ValidationError):
        ToolsAgentInput.model_validate({"totally_unknown_key": 1})


def test_from_tool_input_maps_system_prompt_to_system_message() -> None:
    """The ``from_tool_input`` override renames the composable ``system_prompt`` field
    to the ``system_message`` run/astream kwarg (the mapping lives ONLY here), and
    passes every other set field through unchanged — the raw ``system_prompt`` key
    never reaches the run layer."""
    validated = ToolsAgentInput.model_validate({"system_prompt": "SYS", "tool_names": ["search"]})
    run_kwargs = ToolsAgent.from_tool_input(validated)
    assert run_kwargs["system_message"] == "SYS"
    assert "system_prompt" not in run_kwargs
    assert run_kwargs["tool_names"] == ["search"]


def test_from_tool_input_without_system_prompt_leaves_kwargs_untouched() -> None:
    """With no ``system_prompt`` set, the override is a pure pass-through of set
    fields (no spurious ``system_message`` key introduced)."""
    validated = ToolsAgentInput.model_validate({"tool_names": ["search"]})
    run_kwargs = ToolsAgent.from_tool_input(validated)
    assert run_kwargs == {"tool_names": ["search"]}


def test_from_tool_input_rejects_both_system_prompt_and_system_message() -> None:
    """``system_prompt`` and ``system_message`` both map to the ``system_message`` run
    kwarg, so setting BOTH is a conflict — rejected loudly rather than silently
    dropping one. This is what stops a run request's ``system_message`` from silently
    overriding an authored agent's baked ``system_prompt``."""
    validated = ToolsAgentInput.model_validate({"system_prompt": "baked", "system_message": "from-request"})
    with pytest.raises(ValueError, match="only one of system_prompt or system_message"):
        ToolsAgent.from_tool_input(validated)


# --------------------------------------------------------------------------
# run: tool resolution + delegation to the invoke seam
# --------------------------------------------------------------------------


def test_run_resolves_tools_and_returns_final_text(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """``run`` resolves ``tool_names`` + ``presets`` through the app facet, renders
    the messages, and returns the invoked agent's user-facing output. The invoke
    seam is scripted; its captured arguments prove the wiring."""
    app_tools.client_tools["search"] = make_tool("search")
    app_tools.client_tools["flow"] = make_tool("flow", {"flow_graph": {"type": "object"}, "q": {"type": "string"}})
    routed: list[tuple[str, dict[str, Any]]] = []
    app_tools.tool_runners["flow"] = lambda **kwargs: routed.append(("flow", kwargs)) or {"ok": True}

    captured: dict[str, Any] = {}

    async def fake_invoke(**kwargs: Any) -> AgentInvokeResult:
        captured.update(kwargs)
        return AgentInvokeResult(output="the answer", usage=CallUsage(0, 0, None))

    monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)

    preset = PresetSpec(name="my_flow", base_tool="flow", fixed_kwargs={"flow_graph": {"nodes": []}})
    agent = _get_agent()
    result = asyncio.run(
        agent.run(
            tool_names=["search"],
            presets=[preset],
            system_message="sys",
            user_message="hi",
        )
    )

    assert result == "the answer"
    # live tools = resolved client tool + the preset, deduped and in order.
    resolved = captured["tools"]
    assert [tool.name for tool in resolved] == ["search", "my_flow"]
    # rendered messages threaded through as literal content.
    assert captured["system_message"] == "sys"
    assert captured["user_message"] == ["hi"]

    # the preset tool routes its call through the app's run_tool with fixed kwargs
    # merged under the runtime kwargs.
    preset_tool = next(tool for tool in resolved if tool.name == "my_flow")
    asyncio.run(preset_tool.arun({"q": "search-term"}))
    key, arguments = routed[-1]
    assert key == "flow"
    assert arguments == {"flow_graph": {"nodes": []}, "q": "search-term"}


def test_preset_tool_invocation_raises_on_unknown_base_tool(app_tools: Any) -> None:
    """A preset whose base tool is a registered client tool (so binding succeeds)
    but is absent from ``tool_runners`` surfaces the recording facet's
    unknown-base-tool ``RuntimeError`` when the bound tool is invoked: binding does
    not pre-check the runner map, so the failure lands at call time and propagates
    out of the bound tool rather than being swallowed into a result."""
    app_tools.client_tools["flow"] = make_tool("flow", {"q": {"type": "string"}})
    preset = PresetSpec(name="my_flow", base_tool="flow")
    (preset_tool,) = asyncio.run(resolve_tools(tai42_app.tools, [], [], [preset]))
    with pytest.raises(RuntimeError, match="unknown base tool"):
        asyncio.run(preset_tool.arun({"q": "x"}))


def test_run_treats_unset_message_slots_as_not_provided(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """A message slot's ``content`` and ``id`` both default to the empty string
    ``""``; the render boundary maps that empty-string sentinel to "not provided"
    so a slot never trips the manager's exactly-one-of guard against its own unset
    counterpart.

    Two runs pin both halves of the translation. First, a fully unset system slot:
    ``run(user_message="hi")`` renders and returns rather than raising the
    exactly-one-of ``ValueError`` on ``("", "")``. Second, a system rendered from a
    ``system_message_id`` while ``system_message`` stays unset: the unset content
    must read as "not provided" beside the real id (an un-translated ``""`` content
    would trip the guard), so the template renders through."""
    resource_manager.templates["sys_tpl"] = "SYSTEM"
    captured: dict[str, Any] = {}

    async def fake_invoke(**kwargs: Any) -> AgentInvokeResult:
        captured.update(kwargs)
        return AgentInvokeResult(output="the answer", usage=CallUsage(0, 0, None))

    monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)

    agent = _get_agent()
    result = asyncio.run(agent.run(user_message="hi"))

    assert result == "the answer"
    assert captured["system_message"] == ""
    assert captured["user_message"] == ["hi"]

    result = asyncio.run(agent.run(system_message_id="sys_tpl", user_message="hi"))

    assert result == "the answer"
    assert captured["system_message"] == "SYSTEM"


def test_run_rejects_both_system_message_and_system_message_id(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """The sentinel translation only maps the empty-string default to "not provided":
    a genuine ``system_message`` AND ``system_message_id`` for the same slot still
    trips the manager's exactly-one-of guard, so ``run`` raises ``ValueError``."""

    async def fake_invoke(**kwargs: Any) -> AgentInvokeResult:  # pragma: no cover - never reached
        return AgentInvokeResult(output="unreached", usage=CallUsage(0, 0, None))

    monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)

    agent = _get_agent()
    with pytest.raises(ValueError, match="not both"):
        asyncio.run(agent.run(system_message="sys", system_message_id="sys_id", user_message="hi"))


_STRUCTURED_SCHEMA = {"title": "Answer", "type": "object", "properties": {"value": {"type": "integer"}}}


def test_run_with_response_format_returns_the_structured_object(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """``run`` with a ``response_format`` forces the run's structured output and
    returns the structured object the invoke seam produced — not the text — with the
    schema threaded down to the invoke seam."""
    payload = {"value": 7}
    captured: dict[str, Any] = {}

    async def fake_invoke(**kwargs: Any) -> AgentInvokeResult:
        captured.update(kwargs)
        return AgentInvokeResult(output="ignored text", usage=CallUsage(0, 0, None), structured=payload)

    monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)

    agent = _get_agent()
    result = asyncio.run(agent.run(user_message="hi", response_format=_STRUCTURED_SCHEMA))
    assert result == payload
    assert captured["response_format"] == _STRUCTURED_SCHEMA


def test_run_with_response_format_but_no_structured_raises_loudly(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """A requested ``response_format`` that the run produced none for raises loudly
    rather than silently falling back to text. The raise happens inside
    ``ainvoke_tools_agent`` (its structured extraction), and ``run`` propagates it
    unchanged."""

    async def fake_invoke(**kwargs: Any) -> AgentInvokeResult:
        raise RuntimeError(
            "response_format was requested but the agent produced no structured_response (missing/None in final state)."
        )

    monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)

    agent = _get_agent()
    with pytest.raises(RuntimeError, match="no structured_response"):
        asyncio.run(agent.run(user_message="hi", response_format=_STRUCTURED_SCHEMA))


def test_run_response_format_without_title_raises_loudly(app_tools: Any, resource_manager: Any) -> None:
    """A ``response_format`` JSON-Schema dict lacking a top-level ``"title"`` is
    rejected loudly at the run seam (the title names the structured output)."""
    agent = _get_agent()
    with pytest.raises(ValueError, match="top-level 'title'"):
        asyncio.run(agent.run(user_message="hi", response_format={"type": "object"}))


def test_run_with_subagents_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run`` never silently drops baked ``subagents``: sub-agent delegation is
    deep_agent's domain, so a non-empty value fails loud rather than being ignored."""
    invoked = False

    async def fake_invoke(**kwargs: Any) -> AgentInvokeResult:
        nonlocal invoked
        invoked = True
        return AgentInvokeResult(output="x", usage=CallUsage(0, 0, None))

    monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)
    agent = _get_agent()
    with pytest.raises(RuntimeError, match="subagents"):
        asyncio.run(agent.run(user_message="hi", subagents=[SubAgentSpec(name="helper")]))
    assert invoked is False


def test_run_with_strategy_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run`` never silently drops a baked ``strategy``: it applies none, so a
    non-null value fails loud rather than being ignored."""
    invoked = False

    async def fake_invoke(**kwargs: Any) -> AgentInvokeResult:
        nonlocal invoked
        invoked = True
        return AgentInvokeResult(output="x", usage=CallUsage(0, 0, None))

    monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)
    agent = _get_agent()
    with pytest.raises(RuntimeError, match="strategy"):
        asyncio.run(agent.run(user_message="hi", strategy="react"))
    assert invoked is False


def test_run_renders_message_by_template_id(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """A ``system_message_id`` renders through the template manager's stored map."""
    resource_manager.templates["greeting"] = "rendered-system"
    captured: dict[str, Any] = {}

    async def fake_invoke(**kwargs: Any) -> AgentInvokeResult:
        captured.update(kwargs)
        return AgentInvokeResult(output="x", usage=CallUsage(0, 0, None))

    monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)

    agent = _get_agent()
    asyncio.run(agent.run(system_message_id="greeting", user_message="hi"))
    assert captured["system_message"] == "rendered-system"


def test_run_honors_live_tools_and_thread_config(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """``run`` honors the same seams ``astream`` does: live ``tools`` resolve into
    the invoked tool set, and ``thread_id`` / ``resume_checkpoint_id`` map into the
    run config's ``configurable`` — none of them is silently dropped."""
    app_tools.client_tools["search"] = make_tool("search")
    captured: dict[str, Any] = {}

    async def fake_invoke(**kwargs: Any) -> AgentInvokeResult:
        captured.update(kwargs)
        return AgentInvokeResult(output="ok", usage=CallUsage(0, 0, None))

    monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)

    live_tool = make_tool("live")
    agent = _get_agent()
    asyncio.run(
        agent.run(
            tools=[live_tool],
            tool_names=["search"],
            user_message="hi",
            thread_id="th-1",
            resume_checkpoint_id="ck-9",
        )
    )
    # live tool first, then the resolved client tool — the same order ``astream`` uses.
    assert [tool.name for tool in captured["tools"]] == ["live", "search"]
    assert captured["config"] == {"configurable": {"thread_id": "th-1", "checkpoint_id": "ck-9"}}


def test_run_keyless_does_not_pin_thread_id(
    monkeypatch: pytest.MonkeyPatch, app_tools: Any, resource_manager: Any
) -> None:
    """A keyless ``run`` (no ``thread_id``) leaves the config's ``configurable``
    empty rather than pinning ``thread_id`` to ``None`` — so the run config helper
    mints a fresh isolated thread instead of colliding every keyless run."""
    captured: dict[str, Any] = {}

    async def fake_invoke(**kwargs: Any) -> AgentInvokeResult:
        captured.update(kwargs)
        return AgentInvokeResult(output="ok", usage=CallUsage(0, 0, None))

    monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)
    agent = _get_agent()
    asyncio.run(agent.run(user_message="hi"))
    assert captured["config"] == {"configurable": {}}


# The ABC ``run`` parameters this agent's runtime cannot honor — ALL seven the guard
# is handed. Each is rejected loudly on BOTH faces rather than silently dropped, so
# dropping any one from the guard call site fails a test here. ``resume=False`` pins
# the falsy-but-present case (a value parameter is set whenever it is not ``None``,
# so a falsy value still raises rather than slipping through a truthiness gate).
# ``response_format`` is NOT here — it is honored (forces structured output).
_UNHONORED_CASES = [
    ("subagents", [SubAgentSpec(name="helper")]),
    ("strategy", "react"),
    ("interrupt_on", {"some_tool": {}}),
    ("skills", ["research"]),
    ("inline_skills", [{"name": "s", "content": "c"}]),
    ("resume", "answer"),
    ("resume", False),
    ("store_provider", "redis"),
]


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_run_rejects_unhonored_params(param: str, value: Any) -> None:
    """``run`` never silently drops an ABC parameter its runtime cannot honor: each
    fails loud with a message naming the offending parameter, on the ``run`` face."""
    agent = _get_agent()
    with pytest.raises(RuntimeError, match=rf"tools_agent\.run does not support .*\b{param}\b"):
        asyncio.run(agent.run(user_message="hi", **{param: value}))


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_astream_rejects_unhonored_params(param: str, value: Any) -> None:
    """``astream`` rejects the same unhonored ABC parameters as ``run`` — parity —
    each with a message naming the offending parameter, on the ``astream`` face."""
    agent = _get_agent()
    with pytest.raises(RuntimeError, match=rf"tools_agent\.astream does not support .*\b{param}\b"):
        _collect(agent, user_message="hi", **{param: value})


def test_run_honors_recursion_limit_into_config(monkeypatch: pytest.MonkeyPatch, resource_manager: Any) -> None:
    """``recursion_limit`` is a standard ``RunnableConfig`` key the compiled graph
    reads, so ``run`` overlays it onto the run config rather than rejecting it — the
    captured config the invoke seam receives carries the value (a falsy ``0`` too)."""
    captured: dict[str, Any] = {}

    async def fake_invoke(**kwargs: Any) -> AgentInvokeResult:
        captured.update(kwargs)
        return AgentInvokeResult(output="ok", usage=CallUsage(0, 0, None))

    monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)
    agent = _get_agent()
    asyncio.run(agent.run(user_message="hi", recursion_limit=0))
    assert captured["config"]["recursion_limit"] == 0


def test_astream_honors_recursion_limit_into_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """``astream`` overlays ``recursion_limit`` onto the run config the delegated
    event stream receives — in parity with ``run`` — rather than rejecting it. A
    falsy ``0`` is a real forwarded value: a truthiness overlay would drop it, so
    pinning ``0`` keeps the overlay load-bearing."""
    captured: dict[str, Any] = {}
    _script_astream(monkeypatch, [MessageFinal(text="ok")], captured)
    agent = _get_agent()
    _collect(agent, user_message="hi", recursion_limit=0)
    assert captured["config"]["recursion_limit"] == 0


# --------------------------------------------------------------------------
# langgraph_config: the two faces build the run config identically
# --------------------------------------------------------------------------


def _capture_config(monkeypatch: pytest.MonkeyPatch, face: str, **kwargs: Any) -> dict[str, Any]:
    """Drive one face with ``kwargs`` against a scripted seam and return the run
    config that seam received, so a single assertion body can be run against both
    faces and prove they agree."""
    captured: dict[str, Any] = {}
    agent = _get_agent()

    if face == "run":

        async def fake_invoke(**seen: Any) -> AgentInvokeResult:
            captured.update(seen)
            return AgentInvokeResult(output="ok", usage=CallUsage(0, 0, None))

        monkeypatch.setattr(tools_agent_module, "ainvoke_tools_agent", fake_invoke)
        asyncio.run(agent.run(**kwargs))
    else:
        _script_astream(monkeypatch, [MessageFinal(text="ok")], captured)
        _collect(agent, **kwargs)
    return captured["config"]


@pytest.mark.parametrize("face", ["run", "astream"])
def test_face_honors_langgraph_config_thread_id(monkeypatch: pytest.MonkeyPatch, face: str) -> None:
    """Both faces honor a ``configurable.thread_id`` carried directly in the
    caller's ``langgraph_config``, so a caller pinning a conversation's checkpointed
    memory reaches the graph's checkpointer on either face."""
    config = _capture_config(
        monkeypatch, face, user_message="hi", langgraph_config={"configurable": {"thread_id": "T-42"}}
    )
    assert config["configurable"]["thread_id"] == "T-42"


@pytest.mark.parametrize("face", ["run", "astream"])
def test_face_honors_langgraph_config_checkpoint_id(monkeypatch: pytest.MonkeyPatch, face: str) -> None:
    """Both faces honor a ``configurable.checkpoint_id`` carried directly in the
    caller's ``langgraph_config`` — the checkpoint a run forks from."""
    config = _capture_config(
        monkeypatch,
        face,
        user_message="hi",
        langgraph_config={"configurable": {"thread_id": "T-42", "checkpoint_id": "CK-7"}},
    )
    assert config["configurable"]["checkpoint_id"] == "CK-7"


@pytest.mark.parametrize("face", ["run", "astream"])
def test_face_explicit_memory_keys_win_over_langgraph_config(monkeypatch: pytest.MonkeyPatch, face: str) -> None:
    """The explicit ``thread_id`` / ``resume_checkpoint_id`` parameters overlay the
    caller's ``langgraph_config``, so they win over the same keys carried in its
    ``configurable`` — and the same one wins on both faces."""
    config = _capture_config(
        monkeypatch,
        face,
        user_message="hi",
        thread_id="explicit",
        resume_checkpoint_id="explicit-cp",
        langgraph_config={"configurable": {"thread_id": "from-config", "checkpoint_id": "from-config-cp"}},
    )
    assert config["configurable"]["thread_id"] == "explicit"
    assert config["configurable"]["checkpoint_id"] == "explicit-cp"


@pytest.mark.parametrize("face", ["run", "astream"])
def test_face_preserves_the_rest_of_langgraph_config(monkeypatch: pytest.MonkeyPatch, face: str) -> None:
    """The overlay keeps every other key of the caller's config — the ``configurable``
    entries it does not own and the top-level ``RunnableConfig`` keys alike — and
    leaves the caller's own dict unmutated (a config handed to several runs is not
    scribbled on by one of them)."""
    base = {
        "configurable": {"tenant": "acme"},
        "recursion_limit": 12,
        "tags": ["t1"],
        "metadata": {"origin": "api"},
    }
    config = _capture_config(monkeypatch, face, user_message="hi", thread_id="T-1", langgraph_config=base)

    assert config["configurable"] == {"tenant": "acme", "thread_id": "T-1"}
    assert config["recursion_limit"] == 12
    assert config["tags"] == ["t1"]
    assert config["metadata"] == {"origin": "api"}
    assert base == {
        "configurable": {"tenant": "acme"},
        "recursion_limit": 12,
        "tags": ["t1"],
        "metadata": {"origin": "api"},
    }
    assert config["configurable"] is not base["configurable"]


@pytest.mark.parametrize("face", ["run", "astream"])
def test_face_recursion_limit_param_overlays_langgraph_config(monkeypatch: pytest.MonkeyPatch, face: str) -> None:
    """An explicit ``recursion_limit`` wins over one carried in the caller's config,
    identically on both faces."""
    config = _capture_config(
        monkeypatch, face, user_message="hi", recursion_limit=3, langgraph_config={"recursion_limit": 12}
    )
    assert config["recursion_limit"] == 3


# --------------------------------------------------------------------------
# Message rendering and the response_format guard: parity across the faces
# --------------------------------------------------------------------------


def test_astream_renders_messages_by_template_id(monkeypatch: pytest.MonkeyPatch, resource_manager: Any) -> None:
    """``astream`` renders the system/user messages through the template manager —
    the same rendering ``run`` applies — so a caller supplying a stored template id
    (and its kwargs) reaches the streamed run with the rendered content."""
    resource_manager.templates["sys-tpl"] = "rendered-system"
    resource_manager.templates["user-tpl"] = "rendered-user"
    captured: dict[str, Any] = {}
    _script_astream(monkeypatch, [MessageFinal(text="ok")], captured)

    agent = _get_agent()
    _collect(
        agent,
        system_message_id="sys-tpl",
        user_message_id="user-tpl",
        system_message_kwargs={"a": 1},
        user_message_kwargs={"b": 2},
    )

    assert captured["system_message"] == "rendered-system"
    assert captured["user_message"] == ["rendered-user"]


def test_astream_unknown_template_id_raises_loudly(monkeypatch: pytest.MonkeyPatch, resource_manager: Any) -> None:
    """An unresolvable template id on the streaming face propagates rather than
    reaching the run as an empty message."""
    _script_astream(monkeypatch, [MessageFinal(text="ok")], {})
    agent = _get_agent()
    with pytest.raises(RuntimeError, match="unknown template id: nope"):
        _collect(agent, user_message_id="nope")


@pytest.mark.parametrize("face", ["run", "astream"])
def test_face_rejects_response_format_without_title(monkeypatch: pytest.MonkeyPatch, face: str) -> None:
    """A ``response_format`` dict with no top-level ``"title"`` (the structured-output
    name) is rejected loudly on both faces rather than handed to the graph."""
    with pytest.raises(ValueError, match="top-level 'title'"):
        _capture_config(monkeypatch, face, user_message="hi", response_format={"type": "object"})


# --------------------------------------------------------------------------
# astream: full contract event taxonomy from a scripted stream
# --------------------------------------------------------------------------


def _script_astream(monkeypatch: pytest.MonkeyPatch, events: list[StreamEvent], captured: dict[str, Any]) -> None:
    async def fake_events(**kwargs: Any) -> AsyncIterator[StreamEvent]:
        captured.update(kwargs)
        for event in events:
            yield event

    monkeypatch.setattr(tools_agent_module, "astream_tools_agent_events", fake_events)


def _collect(agent: Agent, **kwargs: Any) -> list[StreamEvent]:
    async def go() -> list[StreamEvent]:
        return [event async for event in agent.astream(**kwargs)]

    return asyncio.run(go())


def test_astream_emits_the_full_event_taxonomy(monkeypatch: pytest.MonkeyPatch) -> None:
    """``astream`` surfaces the projected taxonomy — reasoning, a tool call/result
    pair, message deltas, the final message, and usage — and threads live tools
    and the resume config through to the projection."""
    script: list[StreamEvent] = [
        ReasoningStep(text="planning"),
        ToolCallStep(tool="search", args={"q": "x"}, call_id="c1"),
        ToolResultStep(tool="search", call_id="c1", result="hit"),
        MessageDelta(text="Hel"),
        MessageDelta(text="lo"),
        MessageFinal(text="Hello"),
        RunUsage(input_tokens=3, output_tokens=2, total_tokens=5, model="scripted"),
    ]
    captured: dict[str, Any] = {}
    _script_astream(monkeypatch, script, captured)

    live_tool = make_tool("search")
    agent = _get_agent()
    events = _collect(
        agent,
        tools=[live_tool],
        system_message="sys",
        user_message="hi",
        thread_id="th-1",
        resume_checkpoint_id="ck-9",
    )

    assert [type(event) for event in events] == [
        ReasoningStep,
        ToolCallStep,
        ToolResultStep,
        MessageDelta,
        MessageDelta,
        MessageFinal,
        RunUsage,
    ]
    call, result = events[1], events[2]
    assert isinstance(call, ToolCallStep)
    assert isinstance(result, ToolResultStep)
    assert call.call_id == result.call_id == "c1"
    assert events[5] == MessageFinal(text="Hello")

    # live tools passed through, and the resume checkpoint threaded into config.
    assert captured["tools"] == [live_tool]
    assert captured["user_message"] == ["hi"]
    assert captured["config"] == {"configurable": {"thread_id": "th-1", "checkpoint_id": "ck-9"}}


def test_astream_resolves_baked_tool_names_and_presets(monkeypatch: pytest.MonkeyPatch, app_tools: Any) -> None:
    """``astream`` resolves ``tool_names`` + ``presets`` through the app facet — the
    SAME resolution ``run`` applies — so a baked spec's tools drive the streamed run
    instead of being silently dropped."""
    app_tools.client_tools["search"] = make_tool("search")
    app_tools.client_tools["flow"] = make_tool("flow", {"flow_graph": {"type": "object"}, "q": {"type": "string"}})
    captured: dict[str, Any] = {}
    _script_astream(monkeypatch, [MessageFinal(text="ok")], captured)
    agent = _get_agent()
    preset = PresetSpec(name="my_flow", base_tool="flow", fixed_kwargs={"flow_graph": {"nodes": []}})
    _collect(agent, tool_names=["search"], presets=[preset], user_message="hi")
    assert [tool.name for tool in captured["tools"]] == ["search", "my_flow"]


def test_astream_with_subagents_raises_loudly() -> None:
    """``astream`` never silently drops baked ``subagents`` — parity with ``run``."""
    agent = _get_agent()
    with pytest.raises(RuntimeError, match="subagents"):
        _collect(agent, user_message="hi", subagents=[SubAgentSpec(name="helper")])


def test_astream_with_strategy_raises_loudly() -> None:
    """``astream`` never silently drops a baked ``strategy`` — parity with ``run``."""
    agent = _get_agent()
    with pytest.raises(RuntimeError, match="strategy"):
        _collect(agent, user_message="hi", strategy="react")


def test_astream_with_response_format_emits_structured_final(monkeypatch: pytest.MonkeyPatch) -> None:
    """``astream`` with a ``response_format`` threads the schema to the delegated
    event stream (which forces the structured output) and surfaces the terminal
    :class:`StructuredFinal` the projection emits."""
    captured: dict[str, Any] = {}
    _script_astream(monkeypatch, [MessageFinal(text="text"), StructuredFinal(data={"value": 7})], captured)
    agent = _get_agent()
    events = _collect(agent, user_message="hi", response_format=_STRUCTURED_SCHEMA)

    finals = [event for event in events if isinstance(event, StructuredFinal)]
    assert len(finals) == 1
    assert finals[0].data == {"value": 7}
    assert captured["response_format"] == _STRUCTURED_SCHEMA


def test_astream_with_response_format_but_no_structured_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A requested ``response_format`` the stream produced no ``StructuredFinal``
    for raises loudly after the stream drains — in parity with ``run`` — rather
    than silently omitting the frame."""
    captured: dict[str, Any] = {}
    _script_astream(monkeypatch, [MessageFinal(text="text only")], captured)
    agent = _get_agent()
    with pytest.raises(RuntimeError, match="no structured output"):
        _collect(agent, user_message="hi", response_format=_STRUCTURED_SCHEMA)


def test_astream_without_resume_omits_checkpoint_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _script_astream(monkeypatch, [MessageFinal(text="ok")], captured)
    agent = _get_agent()
    _collect(agent, user_message="hi")
    assert captured["config"] == {"configurable": {}}


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_astream_rejects_blank_memory_key(key: str, blank: str) -> None:
    """A present-but-blank ``thread_id`` / ``resume_checkpoint_id`` is malformed — it
    names a checkpoint namespace two independent runs would silently share — so
    ``astream`` raises rather than writing it verbatim. Only the keyless default
    ``None`` is the unset path."""
    agent = _get_agent()
    with pytest.raises(ValueError, match=rf"tools_agent\.astream: {key} must be a non-empty string"):
        _collect(agent, user_message="hi", **{key: blank})


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_run_rejects_blank_memory_key(key: str, blank: str, app_tools: Any, resource_manager: Any) -> None:
    """Parity with ``astream``: ``run`` rejects a present-but-blank memory key loudly
    rather than forwarding it into the run config's ``configurable``."""
    agent = _get_agent()
    with pytest.raises(ValueError, match=rf"tools_agent\.run: {key} must be a non-empty string"):
        # The dynamic key spreads into a typed run parameter, so the arg-type mismatch is expected.
        asyncio.run(agent.run(user_message="hi", **{key: blank}))  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [123, ["x"]])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_astream_rejects_non_string_memory_key(key: str, value: Any) -> None:
    """A non-string ``thread_id`` / ``resume_checkpoint_id`` is a type violation — it
    cannot name a checkpoint namespace — so ``astream`` raises ``TypeError`` naming the
    offending param and the received type rather than probing a non-string for
    whitespace."""
    agent = _get_agent()
    with pytest.raises(
        TypeError,
        match=rf"tools_agent\.astream: {key} must be a string or None; got {type(value).__name__}",
    ):
        _collect(agent, user_message="hi", **{key: value})


@pytest.mark.parametrize("value", [123, ["x"]])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_run_rejects_non_string_memory_key(key: str, value: Any, app_tools: Any, resource_manager: Any) -> None:
    """Parity with ``astream``: ``run`` rejects a non-string memory key with a
    ``TypeError`` naming the offending param and the received type."""
    agent = _get_agent()
    with pytest.raises(
        TypeError,
        match=rf"tools_agent\.run: {key} must be a string or None; got {type(value).__name__}",
    ):
        # The dynamic key spreads into a typed run parameter, so the arg-type mismatch is expected.
        asyncio.run(agent.run(user_message="hi", **{key: value}))  # type: ignore[arg-type]


def test_unhonored_cases_cover_the_full_reasons_map() -> None:
    """Every key in the guard's reasons map has a parametrized reject case, so a key
    added to the map without a matching test fails here immediately."""
    assert {param for param, _ in _UNHONORED_CASES} == set(tools_agent_module._UNHONORED_REASONS)


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
_COLLECTION_REJECT_PARAMS = sorted(tools_agent_module._UNHONORED_REASONS.keys() & _EMPTY_COLLECTION_ABC_DEFAULTS)


def test_collection_params_match_the_abc_collection_defaults() -> None:
    """``_UNHONORED_COLLECTION_PARAMS`` is exactly this agent's unhonored params whose ABC
    default is an empty collection: no scalar wrongly listed (which would let a meaningful
    falsy value slip through), none dropped (which would over-reject the not-requested
    empty default)."""
    assert set(tools_agent_module._UNHONORED_COLLECTION_PARAMS) == set(_COLLECTION_REJECT_PARAMS)


@pytest.mark.parametrize("empty", [[], ""])
@pytest.mark.parametrize("param", _COLLECTION_REJECT_PARAMS)
def test_reject_unhonored_permits_empty_collection_param(param: str, empty: object) -> None:
    """An empty collection is the ABC's "not requested" default for a collection parameter,
    so the guard does not raise for it — in either falsy empty form (``[]`` / ``""``). Were
    the parameter dropped from ``_UNHONORED_COLLECTION_PARAMS`` it would be classified as a
    scalar (set whenever it is not ``None``) and this empty value would raise."""
    reject_unhonored(
        "tools_agent.run",
        {param: empty},
        tools_agent_module._UNHONORED_REASONS,
        collection_params=tools_agent_module._UNHONORED_COLLECTION_PARAMS,
    )


def test_astream_keyless_run_does_not_pin_thread_id_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A keyless ``astream`` (no ``thread_id``) leaves ``thread_id`` out of the
    configurable entirely, rather than pinning it to ``None`` — so the run config
    helper mints a fresh isolated thread instead of colliding every keyless run
    on one shared checkpoint thread."""
    captured: dict[str, Any] = {}
    _script_astream(monkeypatch, [MessageFinal(text="ok")], captured)
    agent = _get_agent()
    _collect(agent, user_message="hi")
    assert "thread_id" not in captured["config"]["configurable"]


def test_astream_emits_structured_final(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the projected run carries a structured response, ``astream`` surfaces a
    terminal :class:`StructuredFinal`."""

    class Answer(BaseModel):
        value: int

    payload = Answer(value=7)
    script: list[StreamEvent] = [
        MessageDelta(text="done"),
        MessageFinal(text="done"),
        StructuredFinal(data=payload),
    ]
    _script_astream(monkeypatch, script, {})
    agent = _get_agent()
    events = _collect(agent, user_message="hi")

    finals = [event for event in events if isinstance(event, StructuredFinal)]
    assert len(finals) == 1
    assert finals[0].data == payload
    assert finals[0].final is True


def test_drain_structured_stream_returns_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Draining a stream that ends in a :class:`StructuredFinal` with a requested
    ``response_format`` returns the structured payload (the contract terminal rule)."""

    class Answer(BaseModel):
        value: int

    payload = Answer(value=7)
    _script_astream(monkeypatch, [MessageFinal(text="text"), StructuredFinal(data=payload)], {})
    agent = _get_agent()

    async def go() -> Any:
        return await agent._drain(agent.astream(user_message="hi"), response_format=Answer)

    assert asyncio.run(go()) == payload


def test_astream_never_emits_an_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """This agent has no interrupt source: a normal run's stream carries no
    ``InterruptFinal`` (its absence is by design, not a partial run)."""
    from tai42_contract.agent import InterruptFinal

    _script_astream(monkeypatch, [MessageDelta(text="hi"), MessageFinal(text="hi")], {})
    agent = _get_agent()
    events = _collect(agent, user_message="hi")
    assert not any(isinstance(event, InterruptFinal) for event in events)
