"""Bind a recording app to the ``tai42_app`` handle before any test module imports.

Agent modules register through ``tai42_app.agents.agent(name)`` at import time,
mirroring how the host binds the app and then imports the module named by the
manifest's ``agents:`` entry. Binding a recording app here lets a test module
register an agent with the real decorator syntax at module level and then fetch
the live instance back through ``tai42_app.agents.get_agent``.

The recording facets mirror the contract app protocols the agents reach
through the handle:

* ``agents`` (``AppAgents``) — the decorator instantiates the class and stores
  the instance by name; ``get_agent`` raises on an unknown name (never a silent
  ``None``).
* ``monitoring`` (``AppMonitoring``) — ``active`` exposes a stub backend whose
  writer records each ``TraceContext`` and returns a fixed callback list, so the
  run-config helper can be exercised with no live backend.
* ``tools`` (``AppTools``) — ``get_client_tools`` / ``run_tool`` are backed by the
  mutable ``client_tools`` / ``tool_runners`` maps. A test populates the map it
  needs; an unknown name raises a ``RuntimeError`` carrying the name — the
  ``AppTools`` contract prescribes no exception type, so that is the fake's own
  unknown-tool signal. The ``app_tools`` fixture clears both maps per test so
  nothing leaks between tests.
* ``storage`` (``AppStorage``) — ``resource_manager.render_by_id_or_content`` is
  backed by the mutable ``templates`` map with the real manager's semantics: a
  non-``None`` ``content`` renders to itself, a non-``None`` ``template_id`` looks
  up the map, providing both raises ``ValueError``, and an empty call returns
  ``""`` (or raises when ``allow_empty`` is false). An unknown id raises.
  ``resource_manager.normalize_media`` records each call and returns an image
  ``ContentPart`` (a URL passes through, an id becomes a data-URI), raising on a
  resolvable non-image source like the real image-only guard. The
  ``resource_manager`` fixture clears both per test.

Facets a given agent does not use are simply left unconfigured; a test that
needs different behavior monkeypatches the exact seam it calls.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from tai42_contract.agent import Agent
from tai42_contract.app import tai42_app
from tai42_contract.monitoring import TraceContext


class RecordingAgents:
    """An ``AppAgents`` impl that stores live agent instances by name."""

    def __init__(self) -> None:
        self.registry: dict[str, Agent] = {}
        self.tags: dict[str, set[str]] = {}

    def agent(self, name: str, tags: set[str] | None = None) -> Callable[[type[Agent]], type[Agent]]:
        def decorator(agent_cls: type[Agent]) -> type[Agent]:
            self.registry[name] = agent_cls()
            self.tags[name] = tags or set()
            return agent_cls

        return decorator

    def get_agent(self, name: str) -> Agent:
        agent = self.registry.get(name)
        if agent is None:
            raise RuntimeError(f"No such agent: {name}.")
        return agent


CALLBACKS: list[str] = ["callback-a", "callback-b"]


class RecordingMonitoringWriter:
    """A ``MonitoringWriter`` stub: records each ``TraceContext`` it is asked for
    and hands back a fixed copy of the callback sentinels."""

    def __init__(self) -> None:
        self.contexts: list[TraceContext] = []

    def get_monitoring_callbacks(self, ctx: TraceContext) -> list[str]:
        self.contexts.append(ctx)
        return list(CALLBACKS)


class RecordingMonitoring:
    """A ``Monitoring`` stub exposing the recording writer."""

    def __init__(self) -> None:
        self.writer = RecordingMonitoringWriter()


class RecordingMonitoringFacet:
    """An ``AppMonitoring`` impl whose ``active`` returns the stub backend."""

    def __init__(self) -> None:
        self._backend = RecordingMonitoring()

    @property
    def active(self) -> RecordingMonitoring:
        return self._backend


class RecordingTools:
    """An ``AppTools`` impl backed by mutable maps a test populates.

    ``client_tools`` maps a name to a live ``StructuredTool`` returned by
    ``get_client_tools``; ``tool_runners`` maps a base-tool name to a callable
    invoked by ``run_tool`` (used by preset binding). An unknown name raises a
    ``RuntimeError`` carrying the tool name: the ``AppTools`` contract declares no
    exception type for these calls, so this is the fake's own unknown-tool signal
    and tests assert against it, not against any app-side exception hierarchy.
    """

    def __init__(self) -> None:
        self.client_tools: dict[str, StructuredTool] = {}
        self.tool_runners: dict[str, Callable[..., Any]] = {}

    async def get_client_tools(self, names: list[str] | None = None) -> list[StructuredTool]:
        if names is None:
            return list(self.client_tools.values())
        missing = [name for name in names if name not in self.client_tools]
        if missing:
            raise RuntimeError(f"unknown client tools: {missing}")
        return [self.client_tools[name] for name in names]

    async def run_tool(self, key: str, arguments: dict[str, Any], *, offload_sync: bool = False) -> Any:
        # ``offload_sync`` mirrors the real facet's keyword-only argument; the fake
        # runs its recorded runners synchronously and has nothing to offload.
        if key not in self.tool_runners:
            raise RuntimeError(f"unknown base tool: {key}")
        result = self.tool_runners[key](**arguments)
        if isinstance(result, Awaitable):
            return await result
        return result


def _looks_like_image(source: str) -> bool:
    """Whether ``source`` names an image, mirroring the real manager's image-only
    guard as closely as a fake can. A resolvable non-``image/*`` suffix is a
    non-image (the real raises); an unresolvable suffix is unknown — the real
    passes such URLs through (the model dereferences) and dereferences storage
    ids to their stored mime — so the fake treats unknown as an image."""
    mime, _ = mimetypes.guess_type(source)
    return mime is None or mime.startswith("image/")


class RecordingResourceManager:
    """A ``resource_manager`` stub for ``render_by_id_or_content`` and
    ``normalize_media``."""

    def __init__(self) -> None:
        self.templates: dict[str, str] = {}
        self.media_calls: list[str | bytes] = []

    async def normalize_media(self, source: str | bytes) -> dict[str, Any]:
        """Record the call and return a model-ready image ``ContentPart``.

        Mirrors the real ``ResourceManager.normalize_media`` branch without any
        network or storage: a public ``http(s)`` URL passes through unchanged; a
        storage id (or raw bytes) resolves to a stand-in base64 ``data:`` URI. So
        an agent using this handle gets a deterministic, image-only content part.
        """
        self.media_calls.append(source)
        if isinstance(source, str) and not _looks_like_image(source):
            raise ValueError(f"normalize_media requires an image resource; resolved non-image mime for {source!r}")
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            return {"type": "image_url", "image_url": {"url": source}}
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}

    async def render_by_id_or_content(
        self,
        content: str | None = None,
        template_id: str | None = None,
        kwargs: dict[str, Any] | None = None,
        allow_empty: bool = True,
    ) -> str:
        if content is not None and template_id is not None:
            raise ValueError("Provide either 'content' OR 'template_id', not both.")
        if content is not None:
            return content
        if template_id is not None:
            if template_id not in self.templates:
                # The real manager raises ``TemplateNotFoundError`` (a bare
                # ``Exception`` subclass tai42-agents cannot import); the agents
                # never catch it, so what a caller observes is "propagates and
                # aborts". ``RuntimeError`` reproduces that without a type — like
                # ``KeyError`` — that a narrow ``except LookupError`` would catch.
                raise RuntimeError(f"unknown template id: {template_id}")
            return self.templates[template_id]
        if allow_empty:
            return ""
        raise ValueError("You must provide either a template or a template_id.")


class RecordingStorage:
    """An ``AppStorage`` impl exposing the recording resource manager."""

    def __init__(self) -> None:
        self.resource_manager = RecordingResourceManager()


class RecordingApp:
    agents = RecordingAgents()
    monitoring = RecordingMonitoringFacet()
    tools = RecordingTools()
    storage = RecordingStorage()


APP = RecordingApp()
tai42_app.bind(APP)


@pytest.fixture
def app_tools() -> Iterator[RecordingTools]:
    """The bound app's ``tools`` facet, cleared before and after each test."""
    APP.tools.client_tools.clear()
    APP.tools.tool_runners.clear()
    yield APP.tools
    APP.tools.client_tools.clear()
    APP.tools.tool_runners.clear()


@pytest.fixture
def resource_manager() -> Iterator[RecordingResourceManager]:
    """The bound app's ``resource_manager``, cleared before and after each test."""
    APP.storage.resource_manager.templates.clear()
    APP.storage.resource_manager.media_calls.clear()
    yield APP.storage.resource_manager
    APP.storage.resource_manager.templates.clear()
    APP.storage.resource_manager.media_calls.clear()
