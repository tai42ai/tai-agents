"""Tests for ResolvedSubAgentSpec / InlineSkill — required fields, defaults, validation.

Async code is driven with ``asyncio.run`` (the suite does not use
``pytest-asyncio``).
"""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

from tai42_agents.deep_agent.spec import InlineSkill, ResolvedSubAgentSpec


def _tool(name: str = "t") -> StructuredTool:
    async def call_tool(**kwargs: object) -> object:
        return kwargs

    return StructuredTool.from_function(
        func=None,
        coroutine=call_tool,
        name=name,
        description="d",
        args_schema={"type": "object", "properties": {}, "required": []},
    )


def test_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        ResolvedSubAgentSpec(name="a", description="d")  # missing system_prompt  # type: ignore[call-arg]


def test_defaults() -> None:
    spec = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p")
    assert spec.tools == []
    assert spec.skills is None
    assert spec.inline_skills is None
    assert spec.llm_provider is None
    assert spec.llm_kwargs is None
    assert spec.interrupt_on is None


def test_accepts_inline_skills() -> None:
    spec = ResolvedSubAgentSpec(
        name="a",
        description="d",
        system_prompt="p",
        inline_skills=[InlineSkill(name="demo", content="# demo")],
    )
    assert spec.inline_skills is not None
    assert spec.inline_skills[0].name == "demo"
    assert spec.inline_skills[0].content == "# demo"


def test_inline_skill_rejects_unknown_key() -> None:
    """``InlineSkill`` sets ``extra="forbid"``, so a typo'd key (e.g. ``contnt``) is a
    loud validation error rather than a silently dropped field, matching the
    strictness of the enclosing ``DeepAgentInput``."""
    with pytest.raises(ValidationError):
        InlineSkill.model_validate({"name": "demo", "content": "x", "contnt": "typo"})
    with pytest.raises(ValidationError):
        InlineSkill.model_validate({"name": "demo", "content": "x", "totally_unknown_key": 1})


@pytest.mark.parametrize("bad_name", ["", "a/b", "/leading"])
def test_inline_skill_rejects_invalid_name(bad_name: str) -> None:
    """The name mounts at SKILLS_ROOT<name>/ — empty or slash-bearing names would
    break that path, so they are rejected at construction."""
    with pytest.raises(ValidationError):
        InlineSkill(name=bad_name, content="x")


def test_accepts_structured_tool() -> None:
    spec = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p", tools=[_tool("search")])
    assert [t.name for t in spec.tools] == ["search"]


def test_response_format_defaults_none() -> None:
    spec = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p")
    assert spec.response_format is None


def test_accepts_response_format() -> None:
    class M(BaseModel):
        x: int

    spec = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p", response_format=M)
    assert spec.response_format is M


def test_subagents_default_empty() -> None:
    spec = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p")
    assert spec.subagents == []


def test_accepts_nested_subagents() -> None:
    child = ResolvedSubAgentSpec(name="c", description="cd", system_prompt="cp")
    spec = ResolvedSubAgentSpec(name="a", description="d", system_prompt="p", subagents=[child])
    assert [s.name for s in spec.subagents] == ["c"]
