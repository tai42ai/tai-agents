"""Tests for the voting agent.

Fakes the two internal LLM entrypoints the agent reaches — ``ainvoke_tools_agent``
(each voter) and ``astream_tools_agent_events`` (the judge stream) — so the whole
workflow runs with no live model or network. They exercise: registration through
``tai42_app``; judge/voter tool-name resolution to live client tools; the judge's
per-step event kinds followed by the terminal ``StructuredFinal``; ``run``
draining that stream to a ``VotingOutput``; a list of voter specs that can name
the same provider more than once and pin per-voter models; the model label each
``VoteInfo`` records; and that a list of judge tools flows through as a
``list[StructuredTool] | None`` to the judge stream.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError
from tai42_contract.agent import Agent
from tai42_contract.agent.events import (
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

from tai42_agents._internal.reject import reject_unhonored
from tai42_agents._internal.usage import AgentInvokeResult, CallUsage
from tai42_agents.voting_agent import agent as agent_module
from tai42_agents.voting_agent.agent import VotingAgent, VotingAgentInput
from tai42_agents.voting_agent.model import VoteInfo, VoterSpec, VotingOutput

AGENT_NAME = "voting_agent"


def _make_tool(name: str) -> StructuredTool:
    async def call(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    return StructuredTool.from_function(
        func=None,
        coroutine=call,
        name=name,
        description="t",
        args_schema={"type": "object", "properties": {}, "required": []},
    )


async def _collect(agen: Any) -> list[StreamEvent]:
    return [event async for event in agen]


def _fake_voter_invoke(calls: list[dict[str, Any]], usage_model: str | None = None):
    """A fake ``ainvoke_tools_agent`` that records each voter call and returns an
    output tagged with the voter's provider. ``usage_model`` sets the model the
    run reports back (as a real provider response would); ``None`` mirrors a
    provider that omits the model."""

    async def fake(**kwargs: Any) -> AgentInvokeResult:
        calls.append(kwargs)
        return AgentInvokeResult(
            output=f"voter-{kwargs['llm_provider']}",
            usage=CallUsage(input_tokens=0, output_tokens=0, model=usage_model),
        )

    return fake


def _fake_full_judge_stream(captured: dict[str, Any]):
    """A fake judge stream that records the tools it was handed and yields every
    per-step event kind the projection can produce, ending with the assembled
    ``MessageFinal``."""

    async def fake(**kwargs: Any):
        captured["judge_tools"] = kwargs["tools"]
        captured["judge_llm_provider"] = kwargs["llm_provider"]
        yield ReasoningStep(text="weighing the voters")
        yield ToolCallStep(tool="jt", args={}, call_id="c1")
        yield ToolResultStep(tool="jt", call_id="c1", result="ok")
        yield MessageDelta(text="VER")
        yield MessageDelta(text="DICT")
        yield MessageFinal(text="VERDICT")
        yield RunUsage(input_tokens=1, output_tokens=2, total_tokens=3, model="judge-model")

    return fake


def test_decorator_registers_a_live_voting_agent() -> None:
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    assert isinstance(agent, VotingAgent)
    assert isinstance(agent, Agent)
    assert agent.tool_name == AGENT_NAME
    assert VotingAgent.ToolInput is not None
    assert "voter" in agent.tool_description.lower()


def test_astream_emits_every_kind_then_structured_final(monkeypatch, app_tools, resource_manager) -> None:
    """The judge's per-step events stream in order, then a terminal
    ``StructuredFinal`` carrying the full ``VotingOutput``."""
    app_tools.client_tools["jt"] = _make_tool("jt")
    app_tools.client_tools["vt"] = _make_tool("vt")

    voter_calls: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke(voter_calls))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream(captured))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = asyncio.run(
        _collect(
            agent.astream(
                judge_tools=["jt"],
                voter_tools=["vt"],
                judge_message="decide",
                voter_message="answer",
                judge_llm_provider="jp",
                voters=[VoterSpec(provider="p1", model="m1")],
            )
        )
    )

    assert [e.type for e in events] == [
        "reasoning_step",
        "tool_call_step",
        "tool_result_step",
        "message_delta",
        "message_delta",
        "message_final",
        "run_usage",
        "structured_final",
    ]

    # Judge got the resolved judge tool; the single voter got the resolved voter tool.
    assert [t.name for t in captured["judge_tools"]] == ["jt"]
    assert captured["judge_llm_provider"] == "jp"
    assert len(voter_calls) == 1
    assert [t.name for t in voter_calls[0]["tools"]] == ["vt"]
    assert voter_calls[0]["llm_provider"] == "p1"
    assert voter_calls[0]["llm_kwargs"] == {"model": "m1"}

    final = events[-1]
    assert isinstance(final, StructuredFinal)
    assert isinstance(final.data, VotingOutput)
    # The judge label is the model its run actually reported, not a placeholder.
    assert final.data.judge == VoteInfo(provider="jp", model="judge-model", verdict="VERDICT")
    assert final.data.voters == [VoteInfo(provider="p1", model="m1", verdict="voter-p1")]


def test_astream_event_order_tool_call_final_structured(monkeypatch, app_tools, resource_manager) -> None:
    """The decisive-stage ordering: a judge tool call, the judge's final message,
    then the structured terminal."""

    async def fake_judge(**kwargs: Any):
        yield ToolCallStep(tool="jt", args={}, call_id="c1")
        yield MessageFinal(text="VERDICT")

    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke([]))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", fake_judge)

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = asyncio.run(
        _collect(
            agent.astream(
                judge_message="decide",
                voter_message="answer",
                judge_llm_provider="jp",
                voters=[VoterSpec(provider="p1", model="m1")],
            )
        )
    )
    assert [e.type for e in events] == ["tool_call_step", "message_final", "structured_final"]
    # No RunUsage in this stream and no judge kwargs, so the judge model is unknown.
    final = events[-1]
    assert isinstance(final, StructuredFinal)
    assert final.data.judge.model is None


def test_run_drains_to_voting_output(monkeypatch, app_tools, resource_manager) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke([]))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream(captured))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    result = asyncio.run(
        agent.run(
            judge_message="decide",
            voter_message="answer",
            judge_llm_provider="jp",
            judge_llm_kwargs={"model": "jm"},
            voters=[VoterSpec(provider="p1", model="m1")],
        )
    )
    assert isinstance(result, VotingOutput)
    # The run reports "judge-model", which is the model that actually ran and so
    # wins over the requested "jm".
    assert result.judge == VoteInfo(provider="jp", model="judge-model", verdict="VERDICT")
    assert [v.verdict for v in result.voters] == ["voter-p1"]


def test_default_voter_uses_judge_provider(monkeypatch, app_tools, resource_manager) -> None:
    """With no ``voters``, a single voter runs on the judge's provider and
    kwargs, and its model label is ``None`` when nothing names a model."""
    voter_calls: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke(voter_calls))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream(captured))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = asyncio.run(
        _collect(
            agent.astream(
                judge_message="decide",
                voter_message="answer",
                judge_llm_provider="jp",
            )
        )
    )
    assert len(voter_calls) == 1
    assert voter_calls[0]["llm_provider"] == "jp"
    final = events[-1]
    assert isinstance(final, StructuredFinal)
    assert final.data.voters == [VoteInfo(provider="jp", model=None, verdict="voter-jp")]


def test_multiple_judge_tools_flow_through_as_a_list(monkeypatch, app_tools, resource_manager) -> None:
    """A list of judge tools resolves and reaches the judge stream as a
    ``list[StructuredTool] | None`` — carrying every tool, not just one."""
    app_tools.client_tools["jt1"] = _make_tool("jt1")
    app_tools.client_tools["jt2"] = _make_tool("jt2")

    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke([]))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream(captured))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    asyncio.run(
        _collect(
            agent.astream(
                judge_tools=["jt1", "jt2"],
                judge_message="decide",
                voter_message="answer",
                judge_llm_provider="jp",
                voters=[VoterSpec(provider="p1", model="m1")],
            )
        )
    )
    assert isinstance(captured["judge_tools"], list)
    assert [t.name for t in captured["judge_tools"]] == ["jt1", "jt2"]


def test_duplicate_provider_voters_each_run(monkeypatch, app_tools, resource_manager) -> None:
    """Two voter specs naming the SAME provider each spawn their own voter — the
    list shape expresses duplicates a provider-keyed dict would have collapsed —
    and each records its own pinned model in order."""
    voter_calls: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke(voter_calls))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream(captured))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = asyncio.run(
        _collect(
            agent.astream(
                judge_message="decide",
                voter_message="answer",
                judge_llm_provider="jp",
                voters=[
                    VoterSpec(provider="p1", model="a"),
                    VoterSpec(provider="p1", model="b"),
                ],
            )
        )
    )
    assert len(voter_calls) == 2
    assert [c["llm_provider"] for c in voter_calls] == ["p1", "p1"]
    assert [c["llm_kwargs"] for c in voter_calls] == [{"model": "a"}, {"model": "b"}]
    final = events[-1]
    assert isinstance(final, StructuredFinal)
    voters = final.data.voters
    assert voters == [
        VoteInfo(provider="p1", model="a", verdict="voter-p1"),
        VoteInfo(provider="p1", model="b", verdict="voter-p1"),
    ]


def test_voter_model_prefers_run_reported_model(monkeypatch, app_tools, resource_manager) -> None:
    """When the voter run reports the model it actually ran, that real model is
    the label — winning over the model the spec pinned."""
    voter_calls: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        agent_module, "ainvoke_tools_agent", _fake_voter_invoke(voter_calls, usage_model="resolved-model")
    )
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream(captured))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = asyncio.run(
        _collect(
            agent.astream(
                judge_message="decide",
                voter_message="answer",
                judge_llm_provider="jp",
                voters=[VoterSpec(provider="p1", model="requested")],
            )
        )
    )
    final = events[-1]
    assert isinstance(final, StructuredFinal)
    assert final.data.voters == [VoteInfo(provider="p1", model="resolved-model", verdict="voter-p1")]


def test_voter_spec_model_kwarg_folds_into_llm_kwargs(monkeypatch, app_tools, resource_manager) -> None:
    """A spec's ``llm_kwargs`` reach the voter run, and the explicit ``model``
    field is folded into them and takes precedence over any ``model`` inside
    ``llm_kwargs``."""
    voter_calls: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke(voter_calls))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream(captured))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = asyncio.run(
        _collect(
            agent.astream(
                judge_message="decide",
                voter_message="answer",
                judge_llm_provider="jp",
                voters=[VoterSpec(provider="p1", model="explicit", llm_kwargs={"model": "shadowed", "temperature": 0})],
            )
        )
    )
    assert voter_calls[0]["llm_kwargs"] == {"model": "explicit", "temperature": 0}
    final = events[-1]
    assert isinstance(final, StructuredFinal)
    assert final.data.voters == [VoteInfo(provider="p1", model="explicit", verdict="voter-p1")]


class _NotVotingOutput(BaseModel):
    """A response_format the voting agent does not produce."""

    answer: str


def _fake_checkpoint_judge_stream(captured: dict[str, Any]):
    """A judge stream that records the ``checkpoint_provider`` it was handed."""

    async def fake(**kwargs: Any):
        captured["judge_checkpoint_provider"] = kwargs["checkpoint_provider"]
        yield MessageFinal(text="VERDICT")

    return fake


def test_checkpoint_provider_reaches_the_seam_on_both_faces(monkeypatch, app_tools, resource_manager) -> None:
    """``checkpoint_provider`` is the one ABC param the voting runtime honors: it
    reaches both the voter invocations and the judge stream on ``astream`` and,
    identically, when ``run`` drains that same stream."""
    voter_calls: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke(voter_calls))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_checkpoint_judge_stream(captured))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    asyncio.run(
        _collect(
            agent.astream(
                judge_message="decide",
                voter_message="answer",
                judge_llm_provider="jp",
                checkpoint_provider="cp",
            )
        )
    )
    assert captured["judge_checkpoint_provider"] == "cp"
    assert voter_calls[0]["checkpoint_provider"] == "cp"

    voter_calls.clear()
    captured.clear()
    asyncio.run(
        agent.run(
            judge_message="decide",
            voter_message="answer",
            judge_llm_provider="jp",
            checkpoint_provider="cp",
        )
    )
    assert captured["judge_checkpoint_provider"] == "cp"
    assert voter_calls[0]["checkpoint_provider"] == "cp"


def test_voting_output_response_format_is_accepted_on_both_faces(monkeypatch, app_tools, resource_manager) -> None:
    """Passing the ``VotingOutput`` the agent already produces is redundant but
    accepted on both faces — it is not a foreign structured type."""
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke([]))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream({}))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = asyncio.run(
        _collect(
            agent.astream(
                judge_message="decide",
                voter_message="answer",
                judge_llm_provider="jp",
                response_format=VotingOutput,
            )
        )
    )
    assert isinstance(events[-1], StructuredFinal)

    result = asyncio.run(
        agent.run(
            judge_message="decide",
            voter_message="answer",
            judge_llm_provider="jp",
            response_format=VotingOutput,
        )
    )
    assert isinstance(result, VotingOutput)


def test_astream_rejects_foreign_response_format(monkeypatch, app_tools, resource_manager) -> None:
    """A ``response_format`` other than ``VotingOutput`` is rejected loudly on the
    stream face rather than silently ignored."""
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke([]))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream({}))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match=r"voting_agent\.astream does not support response_format"):
        asyncio.run(_collect(agent.astream(judge_message="decide", response_format=_NotVotingOutput)))


def test_run_rejects_foreign_response_format(monkeypatch, app_tools, resource_manager) -> None:
    """A ``response_format`` other than ``VotingOutput`` is rejected loudly on the
    run face rather than silently replaced by the constant drain format. The run-face
    guard is load-bearing: dropping it would surface the ``astream`` token here."""
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke([]))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream({}))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match=r"voting_agent\.run does not support response_format"):
        asyncio.run(agent.run(judge_message="decide", response_format=_NotVotingOutput))


def test_astream_rejects_unhonored_abc_param(monkeypatch, app_tools, resource_manager) -> None:
    """An ABC ``Agent.run`` param the voting runtime has no seat for (``thread_id``)
    is rejected loudly on the stream face, naming it and the exact ``astream`` face."""
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke([]))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream({}))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match=r"voting_agent\.astream does not support .*\bthread_id\b"):
        asyncio.run(_collect(agent.astream(judge_message="decide", thread_id="t1")))


def test_run_rejects_unhonored_abc_param(monkeypatch, app_tools, resource_manager) -> None:
    """An ABC ``Agent.run`` param the voting runtime has no seat for (``tools``) is
    rejected loudly on the run face, naming it and the exact ``run`` face. The
    run-face guard is load-bearing: dropping it would surface the ``astream`` token."""
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke([]))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream({}))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match=r"voting_agent\.run does not support .*\btools\b"):
        asyncio.run(agent.run(judge_message="decide", tools=[_make_tool("x")]))


@pytest.mark.parametrize(("param", "value"), [("strategy", ""), ("resume", False), ("recursion_limit", 0)])
def test_run_rejects_falsy_but_meaningful_abc_param(monkeypatch, app_tools, resource_manager, param, value) -> None:
    """A falsy-but-meaningful scalar (``strategy=""`` / ``resume=False`` /
    ``recursion_limit=0``) the voting runtime has no seat for still raises — a scalar
    is set whenever it is not ``None``, so it never slips through a truthiness gate."""
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke([]))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream({}))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match=rf"voting_agent\.run does not support .*\b{param}\b"):
        # The base Agent.run signature types some of these non-optional, so the mismatch is expected.
        asyncio.run(agent.run(judge_message="decide", voter_message="answer", **{param: value}))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "param", ["thread_id", "resume", "recursion_limit", "strategy", "response_format", "llm_provider"]
)
def test_unhonored_none_passes_the_guard(monkeypatch, app_tools, resource_manager, param) -> None:
    """The ABC's ``None`` "not requested" sentinel passes the guard rather than being
    over-rejected: an explicit ``thread_id=None`` / ``response_format=None`` / … reaches
    the voting flow and produces the terminal ``VotingOutput``, in parity with the other
    agents. (``response_format=None`` is not the foreign-type case, which still raises.)"""
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke([]))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream({}))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    # The base Agent.run signature types some of these non-optional, so the None mismatch is expected.
    result = asyncio.run(agent.run(judge_message="decide", voter_message="answer", **{param: None}))  # type: ignore[arg-type]
    assert isinstance(result, VotingOutput)


def _install_reject_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake the voter/judge seams so that if the guard were bypassed the run would
    SUCCEED (returning a VotingOutput) — a dropped reasons key then fails the
    pytest.raises below cleanly rather than erroring on a real LLM call."""
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke([]))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream({}))


# Every key in ``_UNHONORED_REASONS`` paired with a representative SET value.
# ``response_format`` folds into the same guard: a foreign type is an offender.
# ``resume=False`` pins the falsy-but-present scalar case.
_UNHONORED_CASES = [
    ("tools", [_make_tool("x")]),
    ("tool_names", ["x"]),
    ("presets", [object()]),
    ("subagents", [object()]),
    ("system_message", "sys"),
    ("user_message", "usr"),
    ("strategy", "vote"),
    ("interrupt_on", {"tool": True}),
    ("skills", ["s"]),
    ("inline_skills", [{"name": "s", "content": "c"}]),
    ("recursion_limit", 5),
    ("thread_id", "t1"),
    ("resume", False),
    ("resume_checkpoint_id", "cp"),
    ("llm_provider", "openai"),
    ("store_provider", "redis"),
    ("llm_kwargs", {"model": "x"}),
    ("response_format", _NotVotingOutput),
]


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_run_rejects_every_unhonored_param(monkeypatch, app_tools, resource_manager, param, value) -> None:
    """run rejects every key in the guard's reasons map, naming it and the run face."""
    _install_reject_fakes(monkeypatch)
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match=rf"voting_agent\.run does not support .*\b{param}\b"):
        asyncio.run(agent.run(judge_message="decide", **{param: value}))


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_astream_rejects_every_unhonored_param(monkeypatch, app_tools, resource_manager, param, value) -> None:
    """astream rejects the same full set as run — parity — naming the astream face."""
    _install_reject_fakes(monkeypatch)
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match=rf"voting_agent\.astream does not support .*\b{param}\b"):
        asyncio.run(_collect(agent.astream(judge_message="decide", **{param: value})))


def test_unhonored_cases_cover_the_full_reasons_map() -> None:
    """Every key in the reasons map has a parametrized reject case; a key added without
    a test fails here immediately."""
    assert {param for param, _ in _UNHONORED_CASES} == set(agent_module._UNHONORED_REASONS)


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
_COLLECTION_REJECT_PARAMS = sorted(agent_module._UNHONORED_REASONS.keys() & _EMPTY_COLLECTION_ABC_DEFAULTS)


def test_collection_params_match_the_abc_collection_defaults() -> None:
    """``_UNHONORED_COLLECTION_PARAMS`` is exactly this agent's unhonored params whose ABC
    default is an empty collection: no scalar wrongly listed (which would let a meaningful
    falsy value slip through), none dropped (which would over-reject the not-requested
    empty default)."""
    assert set(agent_module._UNHONORED_COLLECTION_PARAMS) == set(_COLLECTION_REJECT_PARAMS)


@pytest.mark.parametrize("empty", [[], ""])
@pytest.mark.parametrize("param", _COLLECTION_REJECT_PARAMS)
def test_reject_unhonored_permits_empty_collection_param(param: str, empty: object) -> None:
    """An empty collection is the ABC's "not requested" default for a collection parameter,
    so the guard does not raise for it — in either falsy empty form (``[]`` / ``""``). Were
    the parameter dropped from ``_UNHONORED_COLLECTION_PARAMS`` it would be classified as a
    scalar (set whenever it is not ``None``) and this empty value would raise."""
    reject_unhonored(
        "voting_agent.run",
        {param: empty},
        agent_module._UNHONORED_REASONS,
        collection_params=agent_module._UNHONORED_COLLECTION_PARAMS,
    )


def test_run_names_response_format_and_another_offender_at_once(monkeypatch, app_tools, resource_manager) -> None:
    """A foreign ``response_format`` AND another unhonored param are named in ONE raise:
    the response_format check folds into the shared guard, so the caller fixes both in a
    single pass rather than one raise per run."""
    _install_reject_fakes(monkeypatch)
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError) as excinfo:
        # subagents is typed Sequence[SubAgentSpec]; the object() is deliberately wrong to exercise the guard.
        asyncio.run(agent.run(judge_message="decide", response_format=_NotVotingOutput, subagents=[object()]))  # type: ignore[list-item]
    message = str(excinfo.value)
    assert "response_format" in message
    assert "subagents" in message


def test_run_raises_when_required_judge_message_slot_is_unset(monkeypatch, app_tools, resource_manager) -> None:
    """The judge message renders with ``allow_empty=False``: an unset slot (empty
    content AND empty id, with no alternative like resume) surfaces the resource
    manager's fail-loud message rather than silently running on an empty prompt.
    The voter/judge seams are faked so that WITHOUT render.py's ``allow_empty``
    passthrough the run would instead complete — a dropped passthrough turns this
    red rather than letting the empty prompt slip through."""
    _install_reject_fakes(monkeypatch)
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(ValueError, match="must provide either"):
        asyncio.run(agent.run(voter_message="answer"))


def test_run_propagates_unknown_template_id_from_the_handle(monkeypatch, app_tools, resource_manager) -> None:
    """A ``judge_message_id`` naming no registered template surfaces the resource
    manager's unknown-id ``RuntimeError`` (it propagates and aborts the run) rather
    than being silently treated as an empty prompt."""
    _install_reject_fakes(monkeypatch)
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(RuntimeError, match="unknown template id"):
        asyncio.run(agent.run(judge_message_id="no-such-template", voter_message="answer"))


def test_voting_agent_input_rejects_unknown_key() -> None:
    """``extra="forbid"`` turns an unknown key on the JSON tool-face into a loud
    validation error rather than a silently ignored field."""
    with pytest.raises(ValidationError):
        VotingAgentInput.model_validate({"judge_message": "decide", "totally_unknown_key": 1})


def test_voter_spec_rejects_unknown_key() -> None:
    """``extra="forbid"`` on the nested ``VoterSpec`` rejects an ``llm_kwargs`` typo
    loudly instead of letting it vanish."""
    with pytest.raises(ValidationError):
        VoterSpec.model_validate({"provider": "p1", "llm_kwarg": {"temperature": 0}})


def test_over_limit_voters_raise_before_any_llm_call(monkeypatch, app_tools, resource_manager) -> None:
    """A voters list longer than ``max_voters`` raises a ``ValueError`` naming the
    env var BEFORE any voter LLM runs — the mocked ``ainvoke_tools_agent`` is never
    invoked."""
    monkeypatch.setattr(
        agent_module, "agents_limits_settings", lambda: SimpleNamespace(max_voters=2, voter_concurrency=8)
    )
    voter_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_voter_invoke(voter_calls))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream({}))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    with pytest.raises(ValueError, match="TAI_AGENTS_MAX_VOTERS"):
        asyncio.run(
            _collect(
                agent.astream(
                    judge_message="decide",
                    voter_message="answer",
                    judge_llm_provider="jp",
                    voters=[VoterSpec(provider="p1"), VoterSpec(provider="p2"), VoterSpec(provider="p3")],
                )
            )
        )
    assert voter_calls == []


def _fake_overlap_recording_invoke(state: dict[str, Any]):
    """A fake voter that tracks how many voters are concurrently in-flight, so a
    test can assert a bounded semaphore never lets two overlap."""

    async def fake(**kwargs: Any) -> AgentInvokeResult:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        # Yield to the loop: without the semaphore both gathered voters would be
        # in-flight here at once and ``max_active`` would climb to 2.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        state["active"] -= 1
        return AgentInvokeResult(
            output=f"voter-{kwargs['llm_provider']}",
            usage=CallUsage(input_tokens=0, output_tokens=0, model=None),
        )

    return fake


def test_voter_concurrency_one_never_overlaps(monkeypatch, app_tools, resource_manager) -> None:
    """With ``voter_concurrency`` forced to 1, two voters run strictly one at a
    time — the recorded peak in-flight count never exceeds 1."""
    monkeypatch.setattr(
        agent_module, "agents_limits_settings", lambda: SimpleNamespace(max_voters=16, voter_concurrency=1)
    )
    state: dict[str, Any] = {"active": 0, "max_active": 0}
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_overlap_recording_invoke(state))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream({}))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    asyncio.run(
        _collect(
            agent.astream(
                judge_message="decide",
                voter_message="answer",
                judge_llm_provider="jp",
                voters=[VoterSpec(provider="p1"), VoterSpec(provider="p2")],
            )
        )
    )
    assert state["max_active"] == 1


def _fake_delayed_invoke(delays: dict[str, float]):
    """A fake voter whose per-provider delay makes voters COMPLETE out of input
    order, so a test can pin that verdict assembly still follows input order."""

    async def fake(**kwargs: Any) -> AgentInvokeResult:
        provider = kwargs["llm_provider"]
        await asyncio.sleep(delays.get(provider, 0.0))
        return AgentInvokeResult(
            output=f"voter-{provider}",
            usage=CallUsage(input_tokens=0, output_tokens=0, model=None),
        )

    return fake


def test_verdict_order_matches_input_order_despite_completion_order(monkeypatch, app_tools, resource_manager) -> None:
    """Voters finish in reverse of input order, yet the assembled verdicts stay in
    input order — the bounded fan-out preserves index alignment."""
    monkeypatch.setattr(
        agent_module, "agents_limits_settings", lambda: SimpleNamespace(max_voters=16, voter_concurrency=8)
    )
    # p1 finishes last, p3 first — completion order is the reverse of input order.
    monkeypatch.setattr(agent_module, "ainvoke_tools_agent", _fake_delayed_invoke({"p1": 0.03, "p2": 0.015, "p3": 0.0}))
    monkeypatch.setattr(agent_module, "astream_tools_agent_events", _fake_full_judge_stream({}))

    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = asyncio.run(
        _collect(
            agent.astream(
                judge_message="decide",
                voter_message="answer",
                judge_llm_provider="jp",
                voters=[VoterSpec(provider="p1"), VoterSpec(provider="p2"), VoterSpec(provider="p3")],
            )
        )
    )
    final = events[-1]
    assert isinstance(final, StructuredFinal)
    assert [v.provider for v in final.data.voters] == ["p1", "p2", "p3"]
    assert [v.verdict for v in final.data.voters] == ["voter-p1", "voter-p2", "voter-p3"]
