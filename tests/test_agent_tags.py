"""Every agent-as-tool declares the native ``agents`` tag at registration.

Importing the agent modules runs their ``@tai42_app.agents.agent(...)`` registration
through the recording app bound in ``conftest``, which records the declared tags.
"""

from __future__ import annotations

import pytest

from tai42_agents import mcp_tools_agent, tools_agent, vqa_agent  # noqa: F401
from tai42_agents.deep_agent import agent as _deep_agent  # noqa: F401
from tai42_agents.refine_agent import agent as _refine_agent  # noqa: F401
from tai42_agents.retrieval_tools_agent import agent as _retrieval_agent  # noqa: F401
from tai42_agents.voting_agent import agent as _voting_agent  # noqa: F401
from tests.conftest import APP

_AGENT_NAMES = [
    "deep_agent",
    "refine_agent",
    "voting_agent",
    "retrieval_tools_agent",
    "tools_agent",
    "mcp_tools_agent",
    "vqa_agent",
]


@pytest.mark.parametrize("name", _AGENT_NAMES)
def test_agent_tagged_agents(name: str) -> None:
    assert APP.agents.tags[name] == {"agents"}
