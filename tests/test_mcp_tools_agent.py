"""Tests for the ``mcp_tools_agent``.

Importing the agent module registers it on the recording app bound by
``conftest`` (``@tai42_app.agents.agent``); the live instance is fetched back
through ``tai42_app.agents.get_agent``. The ``fastmcp`` client and the delegated
tools-agent event stream are faked, so no live LLM, MCP server, or network is
touched: ``Client`` becomes a scripted async context manager,
``mcp_tools_to_lc_tools`` returns a fixed tool list, and
``astream_tools_agent_events`` yields a scripted :class:`StreamEvent` sequence.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import ValidationError
from tai42_contract.agent import Agent
from tai42_contract.agent.events import (
    MessageDelta,
    MessageFinal,
    ReasoningStep,
    RunUsage,
    StreamEvent,
    ToolCallStep,
    ToolResultStep,
)
from tai42_contract.app import tai42_app

import tai42_agents.mcp_tools_agent as mcp_mod
from tai42_agents._internal.reject import reject_unhonored


async def _collect(agen: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in agen]


class _FakeClient:
    """A scripted stand-in for ``fastmcp.Client`` that records the config it was
    opened with and behaves as an async context manager."""

    opened_with: dict[str, Any] | None = None

    def __init__(self, config: dict[str, Any]) -> None:
        type(self).opened_with = config
        self._config = config

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _install_fakes(monkeypatch: Any, events: list[StreamEvent]) -> dict[str, Any]:
    """Fake the client, the MCP→LC tool conversion, and the delegated event
    stream; return a dict the delegated stream fills with its captured kwargs."""
    captured: dict[str, Any] = {}
    _FakeClient.opened_with = None

    async def fake_tools(client: Any, tools_names: Any = None) -> list[Any]:
        return ["tool-a", "tool-b"]

    async def fake_stream(**kwargs: Any) -> AsyncIterator[StreamEvent]:
        captured.update(kwargs)
        for event in events:
            yield event

    monkeypatch.setattr(mcp_mod, "Client", _FakeClient)
    monkeypatch.setattr(mcp_mod, "mcp_tools_to_lc_tools", fake_tools)
    monkeypatch.setattr(mcp_mod, "astream_tools_agent_events", fake_stream)
    return captured


def test_agent_is_registered() -> None:
    """The module import registered the agent under its name with the declared
    tool metadata."""
    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    assert isinstance(agent, mcp_mod.McpToolsAgent)
    assert agent.tool_name == "mcp_tools_agent"
    assert agent.tool_description
    assert agent.ToolInput is mcp_mod.McpToolsAgentInput


def test_astream_forwards_config_and_streams_taxonomy(monkeypatch, resource_manager) -> None:
    """astream opens the client on the caller's ``mcp_config``, converts its
    tools, and projects the delegated event stream in order."""
    events: list[StreamEvent] = [
        ReasoningStep(text="thinking"),
        ToolCallStep(tool="tool-a", args={"x": 1}, call_id="c1"),
        ToolResultStep(tool="tool-a", call_id="c1", result="done"),
        MessageDelta(text="Hel"),
        MessageFinal(text="Hello"),
        RunUsage(input_tokens=3, output_tokens=2, total_tokens=5),
    ]
    captured = _install_fakes(monkeypatch, events)

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    cfg = {"mcpServers": {"s1": {"command": "run"}}}
    langgraph_config = {"configurable": {"thread_id": "t"}}
    out = asyncio.run(
        _collect(
            agent.astream(
                mcp_config=cfg,
                user_message="hi",
                system_message="sys",
                llm_provider="prov",
                langgraph_config=langgraph_config,
            )
        )
    )

    assert [e.type for e in out] == [
        "reasoning_step",
        "tool_call_step",
        "tool_result_step",
        "message_delta",
        "message_final",
        "run_usage",
    ]
    # the client was opened on the caller's exact mcp_config
    assert _FakeClient.opened_with is cfg
    # the converted MCP tools and the rendered messages reach the delegated stream
    assert captured["tools"] == ["tool-a", "tool-b"]
    assert captured["system_message"] == "sys"
    assert captured["user_message"] == ["hi"]
    assert captured["llm_provider"] == "prov"
    # the langgraph config forwards intact as the delegated stream's config
    assert captured["config"] == langgraph_config


def test_inject_env_injects_only_allowlisted_vars(monkeypatch, resource_manager) -> None:
    """inject_env copies only the allow-listed ``os.environ`` names into each
    server's ``env`` (never the whole environment), the server's own value wins on
    conflict, and the langgraph config forwarded to the delegated stream is left
    untouched."""
    monkeypatch.setenv("MCP_TEST_ENV", "from-os")
    monkeypatch.setenv("MCP_SECRET", "top-secret")
    captured = _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    cfg = {
        "mcpServers": {
            "s1": {},
            "s2": {"env": {"OWN": "keep", "MCP_TEST_ENV": "server-wins"}},
        }
    }
    langgraph_config = {"configurable": {"thread_id": "t"}}
    out = asyncio.run(
        _collect(
            agent.astream(
                mcp_config=cfg,
                user_message="hi",
                inject_env=True,
                env_allowlist=["MCP_TEST_ENV"],
                langgraph_config=langgraph_config,
            )
        )
    )

    assert [e.type for e in out] == ["message_final"]
    # the config the client is opened on is a fresh copy, not the caller's object
    opened = _FakeClient.opened_with
    assert opened is not None
    assert opened is not cfg
    # only the allow-listed var is injected — the secret is never copied through
    assert opened["mcpServers"]["s1"]["env"] == {"MCP_TEST_ENV": "from-os"}
    assert "MCP_SECRET" not in opened["mcpServers"]["s1"]["env"]
    # the server's own env is preserved and wins on conflict
    assert opened["mcpServers"]["s2"]["env"]["OWN"] == "keep"
    assert opened["mcpServers"]["s2"]["env"]["MCP_TEST_ENV"] == "server-wins"
    # the langgraph config reaches the agent intact, not a server's env dict
    assert captured["config"] == langgraph_config


def test_inject_env_does_not_mutate_caller_config(monkeypatch, resource_manager) -> None:
    """inject_env never mutates the caller's ``mcp_config``: the injected env lands
    only on the fresh copy the client is opened with."""
    monkeypatch.setenv("MCP_TEST_ENV", "from-os")
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    cfg = {"mcpServers": {"s1": {}, "s2": {"env": {"OWN": "keep"}}}}
    asyncio.run(
        _collect(
            agent.astream(
                mcp_config=cfg,
                user_message="hi",
                inject_env=True,
                env_allowlist=["MCP_TEST_ENV"],
            )
        )
    )

    # the caller's dict is unchanged: no env injected into it
    assert cfg == {"mcpServers": {"s1": {}, "s2": {"env": {"OWN": "keep"}}}}


def test_inject_env_without_allowlist_raises(monkeypatch, resource_manager) -> None:
    """inject_env=True with no allow-list is a malformed request (it would inject
    nothing) and raises loudly rather than silently doing nothing."""
    monkeypatch.setenv("MCP_TEST_ENV", "from-os")
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    cfg = {"mcpServers": {"s1": {}}}
    with pytest.raises(ValueError, match="inject_env=True requires a non-empty env_allowlist"):
        asyncio.run(_collect(agent.astream(mcp_config=cfg, user_message="hi", inject_env=True)))


def test_run_drains_astream_to_final_text(monkeypatch, resource_manager) -> None:
    """run drains the scripted stream and returns the terminal message text."""
    _install_fakes(
        monkeypatch,
        [MessageDelta(text="Hel"), MessageFinal(text="Hello")],
    )

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    out = asyncio.run(agent.run(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi"))
    assert out == "Hello"


_MCP_SCHEMA = {"title": "Answer", "type": "object", "properties": {"value": {"type": "integer"}}}


def test_run_with_response_format_returns_the_structured_object(monkeypatch, resource_manager) -> None:
    """``response_format`` is honored on the run face: the delegated run forces the
    structured output and ``run`` returns the structured object drained from the
    stream, with the schema threaded down to the delegated stream."""
    from tai42_contract.agent.events import StructuredFinal

    captured = _install_fakes(monkeypatch, [MessageFinal(text="text"), StructuredFinal(data={"value": 7})])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    out = asyncio.run(agent.run(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", response_format=_MCP_SCHEMA))
    assert out == {"value": 7}
    assert captured["response_format"] == _MCP_SCHEMA


def test_run_response_format_without_structured_raises_loudly(monkeypatch, resource_manager) -> None:
    """A requested ``response_format`` the delegated run produces none for raises
    loudly (``_drain``'s rule) rather than silently returning the message text."""
    _install_fakes(monkeypatch, [MessageFinal(text="text")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(RuntimeError, match="no structured output"):
        asyncio.run(agent.run(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", response_format=_MCP_SCHEMA))


def test_run_response_format_without_title_raises_loudly(monkeypatch, resource_manager) -> None:
    """A ``response_format`` dict lacking a top-level ``"title"`` is rejected loudly
    at the run seam."""
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(ValueError, match="top-level 'title'"):
        asyncio.run(
            agent.run(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", response_format={"type": "object"})
        )


def test_astream_response_format_without_title_raises_loudly(monkeypatch, resource_manager) -> None:
    """The streaming face — the one the public run door drives — rejects an untitled
    ``response_format`` up front, exactly as the invoke face does. Without the guard
    the same request fails late, from inside langchain, after the MCP client has
    already connected."""
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(ValueError, match="top-level 'title'"):
        asyncio.run(
            _collect(
                agent.astream(
                    mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", response_format={"type": "object"}
                )
            )
        )


def test_astream_overlays_thread_id_and_checkpoint_onto_configurable(monkeypatch, resource_manager) -> None:
    """astream honors ``thread_id`` / ``resume_checkpoint_id`` by overlaying them
    onto the ``configurable`` section of the caller's ``langgraph_config`` (merged
    with its existing keys, and mapping ``resume_checkpoint_id`` to
    ``checkpoint_id``) rather than dropping them."""
    captured = _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    base = {"recursion_limit": 7, "configurable": {"keep": "me"}}
    asyncio.run(
        _collect(
            agent.astream(
                mcp_config={"mcpServers": {"s1": {}}},
                user_message="hi",
                thread_id="t1",
                resume_checkpoint_id="ckpt-9",
                langgraph_config=base,
            )
        )
    )

    assert captured["config"]["recursion_limit"] == 7
    assert captured["config"]["configurable"] == {
        "keep": "me",
        "thread_id": "t1",
        "checkpoint_id": "ckpt-9",
    }
    # the caller's config object is not mutated by the overlay
    assert base == {"recursion_limit": 7, "configurable": {"keep": "me"}}


def test_run_overlays_thread_id_onto_configurable(monkeypatch, resource_manager) -> None:
    """run forwards ``thread_id`` to the stream face, where it lands in the
    delegated config's ``configurable`` — the honor reaches the seam on the run
    face too, not only on astream."""
    captured = _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    asyncio.run(
        agent.run(
            mcp_config={"mcpServers": {"s1": {}}},
            user_message="hi",
            thread_id="t1",
        )
    )

    assert captured["config"]["configurable"]["thread_id"] == "t1"


def test_astream_with_only_checkpoint_omits_thread_id(monkeypatch, resource_manager) -> None:
    """With only ``resume_checkpoint_id`` set, the delegated config carries
    ``checkpoint_id`` on its ``configurable`` and no ``thread_id`` — so the keyless
    run still gets a freshly minted thread — and the caller's base is untouched."""
    captured = _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    base = {"configurable": {"keep": "me"}}
    asyncio.run(
        _collect(
            agent.astream(
                mcp_config={"mcpServers": {"s1": {}}},
                user_message="hi",
                resume_checkpoint_id="cp",
                langgraph_config=base,
            )
        )
    )

    assert captured["config"]["configurable"] == {"keep": "me", "checkpoint_id": "cp"}
    assert "thread_id" not in captured["config"]["configurable"]
    # the caller's config object is not mutated by the overlay
    assert base == {"configurable": {"keep": "me"}}


def test_astream_with_neither_memory_key_forwards_the_base_by_value(monkeypatch, resource_manager) -> None:
    """With neither ``thread_id`` nor ``resume_checkpoint_id`` set, the caller's
    ``langgraph_config`` reaches the delegated stream intact — but as a fresh dict,
    never the caller's own object, so a config shared across runs cannot be
    scribbled on by one of them."""
    captured = _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    base = {"recursion_limit": 5, "configurable": {"keep": "me"}}
    asyncio.run(
        _collect(agent.astream(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", langgraph_config=base))
    )

    assert captured["config"] == base
    assert captured["config"] is not base
    assert captured["config"]["configurable"] is not base["configurable"]


def test_astream_honors_recursion_limit_into_config(monkeypatch, resource_manager) -> None:
    """astream honors the ABC ``recursion_limit`` param by overlaying it onto the
    delegated run config's top level (a standard ``RunnableConfig`` key the graph
    reads), leaving the caller's ``langgraph_config`` base untouched. A falsy ``0`` is
    a real forwarded value."""
    captured = _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    base = {"configurable": {"keep": "me"}}
    asyncio.run(
        _collect(
            agent.astream(
                mcp_config={"mcpServers": {"s1": {}}},
                user_message="hi",
                recursion_limit=0,
                langgraph_config=base,
            )
        )
    )

    assert captured["config"]["recursion_limit"] == 0
    assert captured["config"]["configurable"] == {"keep": "me"}
    # the caller's config object is not mutated by the overlay
    assert base == {"configurable": {"keep": "me"}}


def test_run_honors_recursion_limit_into_config(monkeypatch, resource_manager) -> None:
    """run forwards the ABC ``recursion_limit`` to the stream face, where it overlays
    the delegated config's top level — the honor reaches the seam on the run face too."""
    captured = _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    asyncio.run(
        agent.run(
            mcp_config={"mcpServers": {"s1": {}}},
            user_message="hi",
            recursion_limit=11,
        )
    )

    assert captured["config"]["recursion_limit"] == 11


def test_run_rejects_unhonored_contract_param(monkeypatch, resource_manager) -> None:
    """A contract parameter with no seat on this agent (``subagents``) raises a
    ``RuntimeError`` naming it and the exact ``run`` face rather than being silently
    dropped."""
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(RuntimeError, match=r"mcp_tools_agent\.run does not support .*\bsubagents\b"):
        asyncio.run(
            agent.run(
                mcp_config={"mcpServers": {"s1": {}}},
                user_message="hi",
                subagents=[{"name": "child"}],  # type: ignore[arg-type]  # deliberately wrong type to exercise the guard
            )
        )


def test_astream_rejects_unhonored_contract_param(monkeypatch, resource_manager) -> None:
    """The stream face rejects the same unhonored contract parameter loudly, naming
    the exact ``astream`` face, in parity with the run face."""
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(RuntimeError, match=r"mcp_tools_agent\.astream does not support .*\bstrategy\b"):
        asyncio.run(
            _collect(
                agent.astream(
                    mcp_config={"mcpServers": {"s1": {}}},
                    user_message="hi",
                    strategy="parallel",
                )
            )
        )


def test_astream_with_response_format_emits_structured_final(monkeypatch, resource_manager) -> None:
    """``response_format`` is honored on the stream face — in parity with run: it
    threads to the delegated stream (which forces the structured output) and the
    terminal :class:`StructuredFinal` surfaces."""
    from tai42_contract.agent.events import StructuredFinal

    captured = _install_fakes(monkeypatch, [MessageFinal(text="text"), StructuredFinal(data={"value": 7})])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    out = asyncio.run(
        _collect(agent.astream(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", response_format=_MCP_SCHEMA))
    )
    finals = [e for e in out if isinstance(e, StructuredFinal)]
    assert len(finals) == 1
    assert finals[0].data == {"value": 7}
    assert captured["response_format"] == _MCP_SCHEMA


def test_astream_with_response_format_but_no_structured_raises_loudly(monkeypatch, resource_manager) -> None:
    """A requested ``response_format`` the delegated stream produced no
    ``StructuredFinal`` for raises loudly after the stream drains — in parity with
    ``run`` — rather than silently omitting the frame."""
    _install_fakes(monkeypatch, [MessageFinal(text="text only")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(RuntimeError, match="no structured output"):
        asyncio.run(
            _collect(
                agent.astream(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", response_format=_MCP_SCHEMA)
            )
        )


# Every key in ``_UNHONORED_REASONS`` paired with a representative SET value: a
# non-empty collection for a collection param, a meaningful value for a scalar
# (``resume=False`` pins the falsy-but-present case). Each is rejected on BOTH faces,
# so dropping any one key from the guard's map fails a case here.
_UNHONORED_CASES = [
    ("tools", [object()]),
    ("tool_names", ["x"]),
    ("presets", [object()]),
    ("subagents", [{"name": "child"}]),
    ("skills", ["research"]),
    ("inline_skills", [{"name": "s", "content": "c"}]),
    ("strategy", "parallel"),
    ("interrupt_on", {"tool": True}),
    ("resume", False),
    ("store_provider", "redis"),
]


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_run_rejects_every_unhonored_param(monkeypatch, resource_manager, param, value) -> None:
    """``run`` rejects every parameter in the guard's reasons map, naming it and the
    exact ``run`` face — never a silent drop."""
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])
    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(RuntimeError, match=rf"mcp_tools_agent\.run does not support .*\b{param}\b"):
        asyncio.run(agent.run(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", **{param: value}))


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_astream_rejects_every_unhonored_param(monkeypatch, resource_manager, param, value) -> None:
    """``astream`` rejects the same full set as ``run`` — parity — each naming the
    exact ``astream`` face."""
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])
    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(RuntimeError, match=rf"mcp_tools_agent\.astream does not support .*\b{param}\b"):
        asyncio.run(_collect(agent.astream(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", **{param: value})))


def test_unhonored_cases_cover_the_full_reasons_map() -> None:
    """Every key in the guard's reasons map has a parametrized reject case, so a key
    added to the map without a matching test fails here immediately."""
    assert {param for param, _ in _UNHONORED_CASES} == set(mcp_mod._UNHONORED_REASONS)


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
_COLLECTION_REJECT_PARAMS = sorted(mcp_mod._UNHONORED_REASONS.keys() & _EMPTY_COLLECTION_ABC_DEFAULTS)


def test_collection_params_match_the_abc_collection_defaults() -> None:
    """``_UNHONORED_COLLECTION_PARAMS`` is exactly this agent's unhonored params whose ABC
    default is an empty collection: no scalar wrongly listed (which would let a meaningful
    falsy value slip through), none dropped (which would over-reject the not-requested
    empty default)."""
    assert set(mcp_mod._UNHONORED_COLLECTION_PARAMS) == set(_COLLECTION_REJECT_PARAMS)


@pytest.mark.parametrize("empty", [[], ""])
@pytest.mark.parametrize("param", _COLLECTION_REJECT_PARAMS)
def test_reject_unhonored_permits_empty_collection_param(param: str, empty: object) -> None:
    """An empty collection is the ABC's "not requested" default for a collection parameter,
    so the guard does not raise for it — in either falsy empty form (``[]`` / ``""``). Were
    the parameter dropped from ``_UNHONORED_COLLECTION_PARAMS`` it would be classified as a
    scalar (set whenever it is not ``None``) and this empty value would raise."""
    reject_unhonored(
        "mcp_tools_agent.run",
        {param: empty},
        mcp_mod._UNHONORED_REASONS,
        collection_params=mcp_mod._UNHONORED_COLLECTION_PARAMS,
    )


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_astream_rejects_blank_memory_key(monkeypatch, resource_manager, key, blank) -> None:
    """A present-but-blank ``thread_id`` / ``resume_checkpoint_id`` is malformed — it
    would silently share a checkpoint namespace across independent runs — so ``astream``
    raises before the overlay rather than writing it verbatim."""
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])
    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(ValueError, match=rf"mcp_tools_agent\.astream: {key} must be a non-empty string"):
        asyncio.run(_collect(agent.astream(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", **{key: blank})))


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_run_rejects_blank_memory_key(monkeypatch, resource_manager, key, blank) -> None:
    """Parity with ``astream``: ``run`` rejects a present-but-blank memory key loudly."""
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])
    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(ValueError, match=rf"mcp_tools_agent\.run: {key} must be a non-empty string"):
        asyncio.run(agent.run(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", **{key: blank}))


@pytest.mark.parametrize("value", [123, ["x"]])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_astream_rejects_non_string_memory_key(monkeypatch, resource_manager, key, value) -> None:
    """A non-string ``thread_id`` / ``resume_checkpoint_id`` is a type violation — it
    cannot name a checkpoint namespace — so ``astream`` raises ``TypeError`` naming the
    offending param and the received type before the overlay, rather than probing a
    non-string for whitespace."""
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])
    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(
        TypeError, match=rf"mcp_tools_agent\.astream: {key} must be a string or None; got {type(value).__name__}"
    ):
        asyncio.run(_collect(agent.astream(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", **{key: value})))


@pytest.mark.parametrize("value", [123, ["x"]])
@pytest.mark.parametrize("key", ["thread_id", "resume_checkpoint_id"])
def test_run_rejects_non_string_memory_key(monkeypatch, resource_manager, key, value) -> None:
    """Parity with ``astream``: ``run`` rejects a non-string memory key with a
    ``TypeError`` naming the offending param and the received type."""
    _install_fakes(monkeypatch, [MessageFinal(text="ok")])
    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    with pytest.raises(
        TypeError, match=rf"mcp_tools_agent\.run: {key} must be a string or None; got {type(value).__name__}"
    ):
        asyncio.run(agent.run(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi", **{key: value}))


def test_astream_allows_role_specific_extension_kwarg(monkeypatch, resource_manager) -> None:
    """A non-contract extension kwarg (the contract's ``**kwargs`` extension point)
    is not an unhonored contract parameter and passes through without raising."""
    captured = _install_fakes(monkeypatch, [MessageFinal(text="ok")])

    agent = tai42_app.agents.get_agent("mcp_tools_agent")
    out = asyncio.run(
        _collect(
            agent.astream(
                mcp_config={"mcpServers": {"s1": {}}},
                user_message="hi",
                some_role_specific_extra="value",
            )
        )
    )

    assert [e.type for e in out] == ["message_final"]
    assert captured["tools"] == ["tool-a", "tool-b"]


def test_astream_bounds_client_connect_and_leaks_no_client(monkeypatch, resource_manager) -> None:
    """A client whose ENTER (connect/init) never completes is bounded by
    ``MCP_CLIENT_CONNECT_TIMEOUT_SECONDS``: ``astream`` raises a ``TimeoutError``
    naming the env var, and no client is left un-exited — the stalled enter is
    cancelled (nothing successfully entered, so nothing to close and nothing
    leaked)."""
    from tai42_kit.settings.cache_registry import reset_all_settings

    class _BlockingClient:
        """A client whose context enter blocks forever; records whether it was
        entered, exited, and whether its blocked enter was cancelled."""

        entered = False
        exited = False
        cancelled = False

        def __init__(self, config: dict[str, Any]) -> None:
            pass

        async def __aenter__(self) -> _BlockingClient:
            try:
                await asyncio.Event().wait()  # never set -> blocks forever
            except asyncio.CancelledError:
                type(self).cancelled = True
                raise
            type(self).entered = True
            return self

        async def __aexit__(self, *exc: object) -> bool:
            type(self).exited = True
            return False

    # Fake the tool conversion + delegated stream so the only thing under test is
    # the bounded client enter; the blocked enter means neither is ever reached.
    _install_fakes(monkeypatch, [MessageFinal(text="unreached")])
    monkeypatch.setattr(mcp_mod, "Client", _BlockingClient)
    monkeypatch.setenv("MCP_CLIENT_CONNECT_TIMEOUT_SECONDS", "0.01")
    reset_all_settings()
    try:
        agent = tai42_app.agents.get_agent("mcp_tools_agent")
        with pytest.raises(TimeoutError, match="MCP_CLIENT_CONNECT_TIMEOUT_SECONDS"):
            asyncio.run(_collect(agent.astream(mcp_config={"mcpServers": {"s1": {}}}, user_message="hi")))
    finally:
        reset_all_settings()

    # The enter never completed, so nothing entered and nothing is left to exit —
    # no client leaked open. The blocked enter was cancelled by the wait_for bound.
    assert _BlockingClient.entered is False
    assert _BlockingClient.exited is False
    assert _BlockingClient.cancelled is True


def test_tool_input_rejects_unknown_key() -> None:
    """``McpToolsAgentInput`` forbids unknown keys: a typo'd flag (e.g. a security
    flag misspelled) raises at validation rather than silently vanishing."""
    with pytest.raises(ValidationError):
        mcp_mod.McpToolsAgentInput.model_validate({"mcp_config": {}, "inject_envv": True})
