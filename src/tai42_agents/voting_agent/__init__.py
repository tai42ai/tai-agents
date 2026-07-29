"""The voting agent package.

Importing this package registers ``VotingAgent`` through ``tai42_app`` (the
``@tai42_app.agents.agent("voting_agent")`` decorator runs on import of
:mod:`tai42_agents.voting_agent.agent`). The host loads the package via the
manifest's ``agents[].module`` entry (``tai42_agents.voting_agent``).
"""

from __future__ import annotations

from tai42_agents.voting_agent.agent import VotingAgent

__all__ = ["VotingAgent"]
