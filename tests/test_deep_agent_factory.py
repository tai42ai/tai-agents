"""Tests for the deep-agent factory: subagent resolution + config validation guards."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from tai42_agents.deep_agent import factory
from tai42_agents.deep_agent.backend import SKILLS_ROOT
from tai42_agents.deep_agent.factory import (
    _collect_inline_skills,
    _resolve_subagent,
    _skills_with_inline,
    _validate,
    build_deep_agent,
)
from tai42_agents.deep_agent.spec import InlineSkill, ResolvedSubAgentSpec

_FAKE_LLM = cast(BaseChatModel, "LLM")


class _FakeRunnable:
    """A minimal runnable stand-in for a compiled nested subagent.

    ``deepagents``' ``SubAgentMiddleware`` eagerly compiles each nested runnable
    at construction (calling ``with_config``), so the fake ``create_deep_agent``
    must return an object exposing it rather than a bare string.
    """

    def with_config(self, *args: object, **kwargs: object) -> _FakeRunnable:
        return self


def _tool(name: str) -> StructuredTool:
    async def call_tool(**kwargs: object) -> object:
        return kwargs

    return StructuredTool.from_function(
        func=None,
        coroutine=call_tool,
        name=name,
        description="d",
        args_schema={"type": "object", "properties": {}, "required": []},
    )


# --- _resolve_subagent -----------------------------------------------------


def test_resolve_subagent_emits_only_set_keys() -> None:
    """Inheritance relies on optional keys being ABSENT, not None."""
    spec = ResolvedSubAgentSpec(name="b", description="d", system_prompt="p")
    sub = cast(dict[str, Any], asyncio.run(_resolve_subagent(spec)))
    assert sub == {"name": "b", "description": "d", "system_prompt": "p"}


def test_resolve_subagent_resolves_model_when_provider_set(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_llm_async(provider: str, **kwargs: object) -> str:
        return f"LLM:{provider}"

    monkeypatch.setattr(factory, "get_llm_async", fake_get_llm_async)
    spec = ResolvedSubAgentSpec(name="b", description="d", system_prompt="p", llm_provider="openai")
    sub = cast(dict[str, Any], asyncio.run(_resolve_subagent(spec)))
    assert sub["model"] == "LLM:openai"


def test_resolve_subagent_passes_through_tools_skills_interrupt() -> None:
    spec = ResolvedSubAgentSpec(
        name="b",
        description="d",
        system_prompt="p",
        tools=[_tool("x")],
        skills=[f"{SKILLS_ROOT}jq/"],
        interrupt_on={"edit_file": True},
    )
    sub = cast(dict[str, Any], asyncio.run(_resolve_subagent(spec)))
    assert [t.name for t in sub["tools"]] == ["x"]
    assert sub["skills"] == [f"{SKILLS_ROOT}jq/"]
    assert sub["interrupt_on"] == {"edit_file": True}


# --- _validate guards ------------------------------------------------------


def test_validate_rejects_duplicate_subagents() -> None:
    specs = [
        ResolvedSubAgentSpec(name="x", description="d", system_prompt="p"),
        ResolvedSubAgentSpec(name="x", description="d", system_prompt="p"),
    ]
    with pytest.raises(ValueError, match="duplicate subagent"):
        _validate([], specs, None)


def test_validate_rejects_duplicate_tools() -> None:
    with pytest.raises(ValueError, match="duplicate tool"):
        _validate([_tool("a"), _tool("a")], [], None)


def test_validate_rejects_builtin_tool_collision() -> None:
    with pytest.raises(ValueError, match="built-in"):
        _validate([_tool("task")], [], None)


def test_validate_rejects_subagent_named_like_builtin() -> None:
    spec = ResolvedSubAgentSpec(name="task", description="d", system_prompt="p")
    with pytest.raises(ValueError, match="built-in tool names"):
        _validate([], [spec], None)


def test_validate_rejects_offroot_skill() -> None:
    with pytest.raises(ValueError, match="must start with"):
        _validate([], [], ["/wrong/x/"])


def test_validate_rejects_offroot_subagent_skill() -> None:
    spec = ResolvedSubAgentSpec(name="b", description="d", system_prompt="p", skills=["/nope/"])
    with pytest.raises(ValueError, match="must start with"):
        _validate([], [spec], None)


def test_validate_accepts_clean_config() -> None:
    spec = ResolvedSubAgentSpec(name="b", description="d", system_prompt="p", skills=[f"{SKILLS_ROOT}jq/"])
    _validate([_tool("search")], [spec], [f"{SKILLS_ROOT}flow/"])  # no raise


# --- build_deep_agent wiring ----------------------------------------------


def test_build_deep_agent_runs_validation_before_create(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation must reject bad config before reaching create_deep_agent."""
    called = {"create": False}

    def fake_create(*args: object, **kwargs: object) -> str:
        called["create"] = True
        return "AGENT"

    monkeypatch.setattr(factory, "create_deep_agent", fake_create)
    with pytest.raises(ValueError, match="must start with"):
        asyncio.run(
            build_deep_agent(
                llm=_FAKE_LLM,
                store=InMemoryStore(),
                checkpointer=InMemorySaver(),
                skills=["/bad/"],
            )
        )
    assert called["create"] is False


def test_resolve_subagent_emits_response_format() -> None:
    class M(BaseModel):
        x: int

    spec = ResolvedSubAgentSpec(name="b", description="d", system_prompt="p", response_format=M)
    sub = cast(dict[str, Any], asyncio.run(_resolve_subagent(spec)))
    assert sub["response_format"] is M


def test_build_deep_agent_passes_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    class M(BaseModel):
        x: int

    captured: dict[str, Any] = {}

    def fake_create(*args: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "AGENT"

    monkeypatch.setattr(factory, "create_deep_agent", fake_create)
    asyncio.run(
        build_deep_agent(
            llm=_FAKE_LLM,
            store=InMemoryStore(),
            checkpointer=InMemorySaver(),
            response_format=M,
        )
    )
    assert captured["response_format"] is M


def test_build_deep_agent_collapses_empty_subagents(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_create(*args: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "AGENT"

    monkeypatch.setattr(factory, "create_deep_agent", fake_create)
    agent = asyncio.run(
        build_deep_agent(
            llm=_FAKE_LLM,
            store=InMemoryStore(),
            checkpointer=InMemorySaver(),
            tools=[],
        )
    )
    assert agent == "AGENT"
    assert captured["subagents"] is None  # empty -> None, so deepagents adds its default


# --- nested subagents --------------------------------------------------------


def _nested_pair(
    child_tools: list[StructuredTool] | None = None,
    parent_tools: list[StructuredTool] | None = None,
) -> tuple[ResolvedSubAgentSpec, ResolvedSubAgentSpec]:
    child = ResolvedSubAgentSpec(name="finder", description="cd", system_prompt="cp", tools=child_tools or [])
    parent = ResolvedSubAgentSpec(
        name="advisor",
        description="d",
        system_prompt="p",
        tools=parent_tools or [],
        subagents=[child],
    )
    return parent, child


def test_validate_rejects_two_level_nesting() -> None:
    grandchild = ResolvedSubAgentSpec(name="g", description="gd", system_prompt="gp")
    child = ResolvedSubAgentSpec(name="c", description="cd", system_prompt="cp", subagents=[grandchild])
    parent = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p", subagents=[child])
    with pytest.raises(ValueError, match="one level deep"):
        _validate([], [parent], None)


def test_validate_rejects_duplicate_nested_names() -> None:
    children = [
        ResolvedSubAgentSpec(name="c", description="cd", system_prompt="cp"),
        ResolvedSubAgentSpec(name="c", description="cd", system_prompt="cp"),
    ]
    parent = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p", subagents=children)
    with pytest.raises(ValueError, match="duplicate nested subagent names"):
        _validate([], [parent], None)


def test_validate_rejects_nested_builtin_name() -> None:
    child = ResolvedSubAgentSpec(name="task", description="cd", system_prompt="cp")
    parent = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p", subagents=[child])
    with pytest.raises(ValueError, match="built-in tool names"):
        _validate([], [parent], None)


def test_validate_rejects_nested_offroot_skill() -> None:
    child = ResolvedSubAgentSpec(name="c", description="cd", system_prompt="cp", skills=["/nope/"])
    parent = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p", subagents=[child])
    with pytest.raises(ValueError, match="must start with"):
        _validate([], [parent], None)


def test_validate_accepts_clean_nested_config() -> None:
    child = ResolvedSubAgentSpec(name="c", description="cd", system_prompt="cp", skills=[f"{SKILLS_ROOT}finder/"])
    parent = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p", subagents=[child])
    _validate([], [parent], None)  # no raise


def test_resolve_subagent_with_nested_requires_main_pieces() -> None:
    parent, _ = _nested_pair()
    with pytest.raises(ValueError, match="requires"):
        asyncio.run(_resolve_subagent(parent))


def test_resolve_subagent_compiles_nested_into_middleware(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return _FakeRunnable()

    monkeypatch.setattr(factory, "create_deep_agent", fake_create)
    parent, _ = _nested_pair(child_tools=[_tool("search")], parent_tools=[_tool("x")])
    backend = object()
    sub = cast(
        dict[str, Any],
        asyncio.run(_resolve_subagent(parent, llm=_FAKE_LLM, tools=[], store=InMemoryStore(), backend=backend)),
    )
    # The nested child compiled as its own deep agent, inheriting the parent model.
    assert captured[0]["model"] == "LLM"
    assert [t.name for t in captured[0]["tools"]] == ["search"]
    assert captured[0]["system_prompt"] == "cp"
    assert captured[0]["backend"] is backend
    # ...and attached to the parent through a SubAgentMiddleware.
    (middleware,) = sub["middleware"]
    assert isinstance(middleware, factory.SubAgentMiddleware)


def test_nested_child_inherits_parent_tools_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return _FakeRunnable()

    monkeypatch.setattr(factory, "create_deep_agent", fake_create)
    parent, _ = _nested_pair(parent_tools=[_tool("x")])
    asyncio.run(_resolve_subagent(parent, llm=_FAKE_LLM, tools=[], store=InMemoryStore(), backend=object()))
    assert [t.name for t in captured[0]["tools"]] == ["x"]


def test_nested_child_resolves_own_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return _FakeRunnable()

    async def fake_get_llm_async(provider: str, **kwargs: object) -> str:
        return f"LLM:{provider}"

    monkeypatch.setattr(factory, "create_deep_agent", fake_create)
    monkeypatch.setattr(factory, "get_llm_async", fake_get_llm_async)
    child = ResolvedSubAgentSpec(name="finder", description="cd", system_prompt="cp", llm_provider="openai")
    parent = ResolvedSubAgentSpec(name="advisor", description="d", system_prompt="p", subagents=[child])
    asyncio.run(_resolve_subagent(parent, llm=_FAKE_LLM, tools=[], store=InMemoryStore(), backend=object()))
    assert captured[0]["model"] == "LLM:openai"


def test_build_deep_agent_passes_nested_subagents_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _FakeRunnable()

    monkeypatch.setattr(factory, "create_deep_agent", fake_create)
    parent, _ = _nested_pair(child_tools=[_tool("search")])
    asyncio.run(
        build_deep_agent(
            llm=_FAKE_LLM,
            store=InMemoryStore(),
            checkpointer=InMemorySaver(),
            subagents=[parent],
        )
    )
    main_call = calls[-1]
    (advisor,) = main_call["subagents"]
    assert advisor["name"] == "advisor"
    assert isinstance(advisor["middleware"][0], factory.SubAgentMiddleware)


# --- inline skills -----------------------------------------------------------


def _inline(name: str, content: str) -> InlineSkill:
    return InlineSkill(name=name, content=content)


def test_collect_inline_skills_spans_all_agents() -> None:
    """Inline skills from the main agent + subagent + nested merge into one map."""
    child = ResolvedSubAgentSpec(name="c", description="cd", system_prompt="cp", inline_skills=[_inline("child", "C")])
    parent = ResolvedSubAgentSpec(
        name="a",
        description="d",
        system_prompt="p",
        inline_skills=[_inline("parent", "P")],
        subagents=[child],
    )
    collected = _collect_inline_skills([_inline("main", "M")], [parent])
    assert collected == {"main": "M", "parent": "P", "child": "C"}


def test_collect_inline_skills_shared_name_identical_content_ok() -> None:
    """A name reused across agents with identical content collapses to one mount."""
    parent = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p", inline_skills=[_inline("shared", "X")])
    collected = _collect_inline_skills([_inline("shared", "X")], [parent])
    assert collected == {"shared": "X"}


def test_collect_inline_skills_name_collision_differing_content_raises() -> None:
    parent = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p", inline_skills=[_inline("dup", "B")])
    with pytest.raises(ValueError, match="different content"):
        _collect_inline_skills([_inline("dup", "A")], [parent])


def test_skills_with_inline_auto_loads_inline_paths() -> None:
    """Inline skill names become /skills/<name>/ sources, alongside reference skills."""
    sources = _skills_with_inline([f"{SKILLS_ROOT}ref/"], [_inline("demo", "x")])
    assert sources == [f"{SKILLS_ROOT}ref/", f"{SKILLS_ROOT}demo/"]


def test_skills_with_inline_none_when_no_skills() -> None:
    assert _skills_with_inline(None, None) is None


def test_skills_with_inline_no_duplicate_when_already_listed() -> None:
    """Naming an inline skill in `skills` too does not duplicate its source."""
    sources = _skills_with_inline([f"{SKILLS_ROOT}demo/"], [_inline("demo", "x")])
    assert sources == [f"{SKILLS_ROOT}demo/"]


def test_resolve_subagent_auto_loads_its_inline_skills() -> None:
    """A subagent's inline skills are mounted via its loaded skill sources."""
    spec = ResolvedSubAgentSpec(
        name="b",
        description="d",
        system_prompt="p",
        skills=[f"{SKILLS_ROOT}ref/"],
        inline_skills=[_inline("demo", "x")],
    )
    sub = cast(dict[str, Any], asyncio.run(_resolve_subagent(spec)))
    assert sub["skills"] == [f"{SKILLS_ROOT}ref/", f"{SKILLS_ROOT}demo/"]


def test_build_deep_agent_mounts_and_auto_loads_inline_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_deep_agent feeds inline content to the backend and auto-loads the path."""
    captured: dict[str, Any] = {}
    built_backends: list[Any] = []

    def fake_create(*args: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "AGENT"

    def fake_build_backend(inline_skills: object = None) -> object:
        built_backends.append(inline_skills)
        return object()

    monkeypatch.setattr(factory, "create_deep_agent", fake_create)
    monkeypatch.setattr(factory, "build_backend", fake_build_backend)
    asyncio.run(
        build_deep_agent(
            llm=_FAKE_LLM,
            store=InMemoryStore(),
            checkpointer=InMemorySaver(),
            inline_skills=[_inline("inline-demo", "# inline-demo\nbody")],
        )
    )
    # Content reached the backend as a name -> content map...
    assert built_backends[-1] == {"inline-demo": "# inline-demo\nbody"}
    # ...and the inline skill's source path is auto-loaded for the main agent.
    assert captured["skills"] == [f"{SKILLS_ROOT}inline-demo/"]


def test_build_deep_agent_inline_collision_raises_before_create(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"create": False}

    def fake_create(*args: object, **kwargs: object) -> str:
        called["create"] = True
        return "AGENT"

    monkeypatch.setattr(factory, "create_deep_agent", fake_create)
    sub = ResolvedSubAgentSpec(name="b", description="d", system_prompt="p", inline_skills=[_inline("dup", "B")])
    with pytest.raises(ValueError, match="different content"):
        asyncio.run(
            build_deep_agent(
                llm=_FAKE_LLM,
                store=InMemoryStore(),
                checkpointer=InMemorySaver(),
                inline_skills=[_inline("dup", "A")],
                subagents=[sub],
            )
        )
    assert called["create"] is False


def test_build_deep_agent_no_inline_skills_keeps_skills_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_create(*args: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "AGENT"

    monkeypatch.setattr(factory, "create_deep_agent", fake_create)
    asyncio.run(
        build_deep_agent(
            llm=_FAKE_LLM,
            store=InMemoryStore(),
            checkpointer=InMemorySaver(),
        )
    )
    assert captured["skills"] is None
