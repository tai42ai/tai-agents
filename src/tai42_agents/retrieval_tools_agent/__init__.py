"""The ``retrieval_tools_agent`` package.

Importing this package registers :class:`RetrievalToolsAgent` through the global
``tai42_app`` handle (``@tai42_app.agents.agent("retrieval_tools_agent")``). The host
loads it via the manifest's ``agents[].module`` field; nothing here runs until
the module is imported.
"""

from __future__ import annotations

from tai42_agents.retrieval_tools_agent.agent import RetrievalToolsAgent

__all__ = ["RetrievalToolsAgent"]
