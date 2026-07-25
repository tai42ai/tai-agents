"""The ``refine_agent`` package.

Importing this package registers :class:`~tai42_agents.refine_agent.agent.RefineAgent`
on the bound app (the ``@tai42_app.agents.agent("refine_agent")`` decorator runs at
import time), mirroring how the host imports the module named by a manifest's
``agents:`` entry.
"""

from tai42_agents.refine_agent.agent import RefineAgent

__all__ = ["RefineAgent"]
