"""Declarative spec for a deep-agent subagent.

:class:`ResolvedSubAgentSpec` is the authoring shape: it names a provider (resolved
through ``get_llm_async``) rather than a resolved model object, and carries live
``StructuredTool`` objects rather than tool names. The async resolution to a
deepagents ``SubAgent`` dict happens in :mod:`tai42_agents.deep_agent.factory`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator


class InlineSkill(BaseModel):
    """A skill supplied directly in the call rather than read from the template store.

    ``name`` is mounted at ``<SKILLS_ROOT><name>/SKILL.md`` and auto-loaded into the
    owning agent's skill set; ``content`` is the raw ``SKILL.md`` body, served
    verbatim (never jinja-rendered). ``extra="forbid"`` rejects any unknown key
    loudly at validation.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Skill identifier; mounted at SKILLS_ROOT<name>/ and auto-loaded.")
    content: str = Field(description="Raw SKILL.md body (frontmatter + markdown); served verbatim.")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value or "/" in value:
            raise ValueError(
                f"InlineSkill name must be non-empty and contain no '/' (it is mounted at "
                f"SKILLS_ROOT<name>/): {value!r}"
            )
        return value


class ResolvedSubAgentSpec(BaseModel):
    """A subagent the main deep agent can invoke via its ``task`` tool.

    ``llm_provider`` / ``llm_kwargs`` are optional (the subagent inherits the main
    agent's model when omitted); ``tools`` inherits the parent's tools when empty.

    ``base_url``/``api_key`` in ``llm_kwargs`` route to a caller-chosen model
    endpoint, so expose any agent carrying these kwargs only to trusted callers — an
    injected parent could redirect the model call and leak the key/context.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Identifier the main agent uses to call this subagent.")
    description: str = Field(description="When to use this subagent; read by the main agent to route to it.")
    system_prompt: str = Field(description="The subagent's own system instructions.")

    llm_provider: str | None = Field(
        default=None,
        description="LLM provider for this subagent; inherits the main agent's model when None.",
    )
    llm_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Provider kwargs merged over llm_settings() defaults when llm_provider is set.",
    )
    tools: list[StructuredTool] = Field(
        default_factory=list,
        description="Tools available to this subagent; inherits the parent's tools when empty.",
    )
    skills: list[str] | None = Field(
        default=None,
        description="Skill source paths (under SKILLS_ROOT) loaded only for this subagent.",
    )
    inline_skills: list[InlineSkill] | None = Field(
        default=None,
        description="Skills supplied inline for this subagent; mounted under SKILLS_ROOT and "
        "auto-loaded (no template provider needed).",
    )
    interrupt_on: dict[str, Any] | None = Field(
        default=None,
        description="Per-tool human-in-the-loop interrupt config; inherits the main agent's when None.",
    )
    response_format: Any = Field(
        default=None,
        description="Structured-output schema for this subagent (a pydantic model or a "
        "langchain response strategy). When set, the subagent returns a "
        "structured object instead of free-form text. None = free-form.",
    )
    subagents: list[ResolvedSubAgentSpec] = Field(
        default_factory=list,
        description="Nested subagents this subagent can invoke via its own task tool. "
        "One level only: a nested subagent declaring subagents of its own "
        "is rejected at build time.",
    )
