"""The voting agent package.

Importing this package registers ``VotingAgent`` through ``tai_app`` (the
``@tai_app.agents.agent("voting_agent")`` decorator runs on import of
:mod:`tai_agents.voting_agent.agent`). The host loads the package via the
manifest's ``agents[].module`` entry (``tai_agents.voting_agent``).
"""

from __future__ import annotations

from tai_agents.voting_agent.agent import VotingAgent

__all__ = ["VotingAgent"]
