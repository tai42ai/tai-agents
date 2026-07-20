"""The deep-agent package — a deepagents-harness :class:`Agent`.

Importing this package registers ``DeepAgent`` through ``tai_app`` (the
``@tai_app.agents.agent("deep_agent")`` decorator runs on import of
:mod:`tai_agents.deep_agent.agent`). The host loads the package via the
manifest's ``agents[].module`` entry (``tai_agents.deep_agent``).

A thin layer over ``deepagents.create_deep_agent`` that wires the harness to the
kit infrastructure (LLM/checkpoint/store registries) and the ecosystem
conventions, so every deep agent is built the same way.

Public surface:

* :class:`ResolvedSubAgentSpec` — declarative spec for a subagent, carrying
  resolved live tools (resolved to a deepagents ``SubAgent`` dict by the factory).
* :class:`InlineSkill` — a skill authored inline (name + ``SKILL.md`` content),
  the element type of the ``inline_skills`` argument of :func:`build_deep_agent`
  and the ``inline_skills`` field of :class:`ResolvedSubAgentSpec`.
* :func:`build_deep_agent` — build a compiled deep agent from resolved pieces.
* :func:`build_backend` — the composite skills(template-provider) + scratch(state) backend.
* :data:`SKILLS_ROOT` — mount point for skills read live from the template provider.
"""

from __future__ import annotations

from tai_agents.deep_agent.agent import DeepAgent
from tai_agents.deep_agent.backend import SKILLS_ROOT, build_backend
from tai_agents.deep_agent.factory import build_deep_agent
from tai_agents.deep_agent.spec import InlineSkill, ResolvedSubAgentSpec

__all__ = [
    "SKILLS_ROOT",
    "DeepAgent",
    "InlineSkill",
    "ResolvedSubAgentSpec",
    "build_backend",
    "build_deep_agent",
]
