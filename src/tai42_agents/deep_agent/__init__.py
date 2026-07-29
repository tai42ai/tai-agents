"""The deep-agent package — a deepagents-harness :class:`Agent`.

Importing this package registers ``DeepAgent`` through ``tai42_app``. A thin layer
over ``deepagents.create_deep_agent`` that wires the harness to the kit
infrastructure (LLM/checkpoint/store registries) and ecosystem conventions.

Public surface:

* :class:`ResolvedSubAgentSpec` — declarative subagent spec carrying resolved live tools.
* :class:`InlineSkill` — a skill authored inline (name + ``SKILL.md`` content).
* :func:`build_deep_agent` — build a compiled deep agent from resolved pieces.
* :func:`build_backend` — the composite skills + scratch backend.
* :data:`SKILLS_ROOT` — mount point for skills read live from the template provider.
"""

from __future__ import annotations

from tai42_agents.deep_agent.agent import DeepAgent
from tai42_agents.deep_agent.backend import SKILLS_ROOT, build_backend
from tai42_agents.deep_agent.factory import build_deep_agent
from tai42_agents.deep_agent.spec import InlineSkill, ResolvedSubAgentSpec

__all__ = [
    "SKILLS_ROOT",
    "DeepAgent",
    "InlineSkill",
    "ResolvedSubAgentSpec",
    "build_backend",
    "build_deep_agent",
]
