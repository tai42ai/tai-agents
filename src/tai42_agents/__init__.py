"""tai42-agents — the reference agents package for the TAI ecosystem.

An opt-in, manifest-loaded collection of generic **agents** built on the
deepagents/LangGraph runtime. Nothing here is imported at package import time:
each module registers its agent through the global ``tai42_app`` handle from
``tai42_contract.app`` (``@tai42_app.agents.agent(name)`` on a subclass of the
contract ``Agent`` ABC) and is loaded by the host via the manifest's
``agents[].module`` field. Agents stream the contract's ``StreamEvent``
taxonomy from ``astream`` and expose an auto-generated JSON ``run`` tool.
"""
