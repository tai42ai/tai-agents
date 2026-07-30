"""End-to-end integration test: the real ``tools_agent`` authored and streamed
through the real tai42-skeleton app.

Every other tai42-agents test drives an agent against a recording app double; this
one proves the cross-repo seam. It boots the concrete tai42-skeleton app in-process
with a manifest that registers the real ``tools_agent`` (this package), then walks
the full authoring path the skeleton exposes:

* ``GET /api/agents`` lists ``tools_agent`` and reports ``spec_runnable: true`` —
  the capability marker ``ToolsAgent`` declares, read by the skeleton, never
  inferred;
* ``GET /api/agents/spec-runnable`` includes it (the compose UI's base-agent
  picker source);
* the agent is authored into a named preset by baking a ``system_prompt`` as fixed
  kwargs, registered through ``PresetManager.register`` — the same in-memory binding
  call the ``POST /api/presets`` route funnels into (that route, its versioned-store
  gate, and store persistence are covered by tai42-skeleton's own suite; this seam
  test registers directly, so it needs no store);
* ``POST /api/agents/authored/{name}/runs`` streams a run whose baked
  ``system_prompt`` reaches the agent's ``astream`` MAPPED to the ``system_message``
  run kwarg — the ``from_tool_input`` override this package delivers — and whose
  scripted events arrive as ordered SSE frames.

tai42-skeleton is a DEV/TEST-only dependency (see ``pyproject.toml``); the shipped
wheel never imports it. Determinism: the LLM seam (``astream_tools_agent_events``)
is scripted so no model or network is touched, the preset is registered in memory
so boot configures no store and never opens a Postgres connection, and the
access-control startup probes (access control stays ENABLED, the default) run
against in-fixture fakes so boot needs no Redis.
"""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from tai42_contract.access_control.identity import IdentityProvider
from tai42_contract.access_control.registry import (
    get_identity_provider_factory,
    register_identity_provider,
    reset_registry,
)
from tai42_contract.agent.events import MessageFinal, StructuredFinal
from tai42_contract.app import tai42_app

# Safe under any binding — these modules touch no tai42_app HTTP decorators at import
# (unlike the routers, which are imported only after the skeleton app is bound).
from tai42_kit.settings import reset_all_settings
from tai42_skeleton.app import instance, lifecycle
from tai42_skeleton.manifest import Manifest

from tests.conftest import APP as RECORDING_APP

# One agents entry: the real generic tools-agent, gated in by its registration name.
_MANIFEST = {"agents": [{"title": "tai42-agents", "module": "tai42_agents.tools_agent", "include": ["tools_agent"]}]}

_BAKED_SYSTEM_PROMPT = "You are a helpdesk agent."


# -- request / response helpers (mirror the skeleton router test harness) ------


def _json_request(method: str, path: str, *, body: Any = None, **path_params: str) -> Request:
    payload = b"" if body is None else json.dumps(body).encode()
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "path_params": path_params,
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(scope, receive)


def _run_request(name: str, body: Any) -> Request:
    """A POST for the authored-run door whose client never disconnects, so the
    disconnect monitor stays False and the run drains to completion."""
    payload = json.dumps(body).encode()
    scripted = [{"type": "http.request", "body": payload, "more_body": False}]
    idx = {"i": 0}

    async def receive() -> dict[str, Any]:
        i = idx["i"]
        if i < len(scripted):
            idx["i"] += 1
            return scripted[i]
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": f"/api/agents/authored/{name}/runs",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "client": ("1.2.3.4", 1),
        "path_params": {"name": name},
    }
    return Request(scope, receive)


def _data(resp: Response) -> Any:
    return json.loads(bytes(resp.body))["data"]


async def _collect_frames(response: StreamingResponse) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    async for chunk in response.body_iterator:
        text = chunk if isinstance(chunk, str) else bytes(chunk).decode()
        if text.startswith(":"):  # keep-alive comment
            continue
        assert text.startswith("data: "), text
        frames.append(json.loads(text[len("data: ") :].strip()))
    return frames


# -- fixture: bind the real skeleton app, restore the recording app afterwards --


class _NoopHealthcheckProvider(IdentityProvider):
    """An identity provider registered under the default ``auth_provider`` name so
    the access-control identity startup probe resolves a factory. It inherits the
    contract's default no-op ``healthcheck``, so that probe runs for real and
    passes with no provider backend. ``validate_token`` is never reached: this e2e
    invokes the skeleton routers as plain functions, so no request crosses the
    authentication middleware."""

    def __init__(self, settings: Any) -> None: ...

    async def validate_token(self, token: str) -> None:
        return None


def _register_offline_identity_provider() -> None:
    """Register the no-op stand-in under ``"redis"`` (the skeleton's
    ``auth_provider`` settings default) unless that name is already taken."""
    try:
        get_identity_provider_factory("redis")
    except KeyError:
        register_identity_provider("redis", _NoopHealthcheckProvider)


@pytest.fixture
def skeleton(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Bind the concrete tai42-skeleton app and expose its routers.

    The suite's ``conftest`` binds a recording app double to ``tai42_app`` for every
    other test; this e2e alone needs the real skeleton app bound so its HTTP
    routers register and its ``app_context`` boots. The router modules fire
    ``@tai42_app.http.custom_route`` at import, so the skeleton must be bound BEFORE
    they import — hence the deferred import here. The prior binding is restored and
    any preset registered during the test is torn down on teardown, so no later
    test observes the skeleton app or a leaked registration.

    Access control stays ENABLED (the settings default): ``build_app`` constructs
    the ``AuthAdapter`` and registers the access-control startup probe, and
    entering ``app_context`` runs it for real. This suite has no Redis, so the
    probe's backend seam is faked while the probe code itself executes:

    * ``probe_identity_provider`` resolves the configured ``auth_provider``
      (default ``"redis"``) through the module-level identity registry and awaits
      that provider's ``healthcheck()``. The skeleton ships no concrete provider —
      a deployment's manifest names an identity plugin whose import registers one
      — and this test's manifest carries none, while boot clears the registry
      before importing manifest modules. So the fixture emulates a manifest that
      lists such a plugin: it registers a no-op-``healthcheck`` stand-in up front
      AND wraps the lifecycle's registry reset to re-register it after each clear
      (the reset itself still runs, so its duplicate-guard purpose is untouched).
      Teardown empties the registry so the stand-in never reaches another test.

    ``reset_all_settings()`` runs before the build: the ``@settings_cache``d
    settings the boot reads are process-wide, so the caches earlier tests
    populated are dropped and the boot reads the real environment. ``build_app``
    memoises ``instance._app``, so both the reset and the ``_app`` nulling must
    land BEFORE the first ``build_app`` call.

    Nulling ``instance._app`` is also what isolates the presets a test authors: the
    ``PresetManager`` (its spec map and quarantine set) and the tool registry the
    presets bind into both hang off the app object, so each test that takes this
    fixture builds its own and no registration can reach the next one."""

    def reset_then_reregister() -> None:
        # The lifecycle module's ``reset_identity_registry`` is the contract's
        # ``reset_registry`` under an import alias; call the contract function so
        # the real clear still runs before the stand-in is re-registered.
        reset_registry()
        _register_offline_identity_provider()

    monkeypatch.setattr(lifecycle, "reset_identity_registry", reset_then_reregister)
    _register_offline_identity_provider()

    monkeypatch.setattr(instance, "_app", None)
    reset_all_settings()
    tai42_app.bind(instance.build_app())
    from tai42_skeleton.routers import agents as agents_router
    from tai42_skeleton.routers import presets as presets_router

    try:
        yield SimpleNamespace(agents=agents_router, presets=presets_router)
    finally:
        # Restore the recording app the suite's conftest binds for every other test.
        tai42_app.bind(RECORDING_APP)
        # Leave the module-level identity registry as this fixture found it: empty.
        reset_registry()


# -- the end-to-end scenario ---------------------------------------------------


def test_tools_agent_authored_and_streamed_through_the_skeleton(skeleton: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def run() -> None:
        async with instance.app.app_context(Manifest.model_validate(_MANIFEST)):
            # The manifest importer re-execs ``tools_agent`` on boot, so patch the
            # LLM seam on the LIVE module object; ``astream`` resolves the name from
            # that module's globals at call time, so the scripted generator runs.
            live_module = sys.modules["tai42_agents.tools_agent"]

            async def fake_events(**kwargs: Any) -> Any:
                captured.update(kwargs)
                yield MessageFinal(text=f"system={kwargs.get('system_message', '')}")

            monkeypatch.setattr(live_module, "astream_tools_agent_events", fake_events)

            # 1. The real tools_agent is registered and reports its capability marker.
            items = _data(await skeleton.agents.list_agents(_json_request("GET", "/api/agents")))["items"]
            by_name = {it["name"]: it for it in items}
            assert "tools_agent" in by_name
            assert by_name["tools_agent"]["spec_runnable"] is True
            assert by_name["tools_agent"]["tool_name"] == "tools_agent"
            # The list schema is the agent's own ToolInput schema (the run-tool source).
            assert "system_prompt" in by_name["tools_agent"]["input_schema"]["properties"]

            # 2. The spec-runnable picker route includes it.
            sr = _data(
                await skeleton.agents.list_spec_runnable_agents(_json_request("GET", "/api/agents/spec-runnable"))
            )
            assert {it["name"] for it in sr["items"]} == {"tools_agent"}

            # 3. Author it: bake a system_prompt as fixed kwargs over the agent tool.
            #    Register through PresetManager.register — the same in-memory binding
            #    the POST /api/presets route funnels into. That route, its versioned-
            #    store 503 gate, and store persistence are covered in tai42-skeleton's own
            #    suite; this cross-repo test targets the streaming seam, so it registers
            #    directly on the process manager the run path reads and needs no store.
            await instance.app.preset_manager.register(
                "support_bot",
                "tools_agent",
                {"system_prompt": _BAKED_SYSTEM_PROMPT},
                [],
                "A helpdesk agent.",
            )

            # 4. Stream a run: the request supplies only the non-baked user_message.
            run_resp = await skeleton.agents.run_authored_agent(
                _run_request("support_bot", {"user_message": "my order is late"})
            )
            assert isinstance(run_resp, StreamingResponse)
            frames = await _collect_frames(run_resp)

            assert [f["type"] for f in frames] == ["message_final", "stream.end"]
            # The baked system_prompt reached astream MAPPED to system_message (a raw
            # splat would have passed system_prompt through unmapped), echoed back by
            # the scripted stream.
            assert frames[0]["text"] == f"system={_BAKED_SYSTEM_PROMPT}"
            assert captured["system_message"] == _BAKED_SYSTEM_PROMPT
            assert "system_prompt" not in captured
            assert captured["user_message"] == ["my order is late"]

    asyncio.run(run())


def test_tools_agent_response_format_streams_structured_through_the_skeleton(
    skeleton: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``response_format`` supplied on the run request reaches the real
    ``tools_agent`` through the skeleton's authored-run door — advertised on
    ``ToolsAgentInput`` (so the door accepts it) and threaded to ``astream`` — and
    the forced structured output surfaces as a ``structured_final`` SSE frame."""
    captured: dict[str, Any] = {}
    schema = {"title": "Answer", "type": "object", "properties": {"value": {"type": "integer"}}}

    async def run() -> None:
        async with instance.app.app_context(Manifest.model_validate(_MANIFEST)):
            live_module = sys.modules["tai42_agents.tools_agent"]

            async def fake_events(**kwargs: Any) -> Any:
                captured.update(kwargs)
                yield StructuredFinal(data={"value": 7})

            monkeypatch.setattr(live_module, "astream_tools_agent_events", fake_events)

            await instance.app.preset_manager.register(
                "struct_bot",
                "tools_agent",
                {"system_prompt": _BAKED_SYSTEM_PROMPT},
                [],
                "A structured helpdesk agent.",
            )

            run_resp = await skeleton.agents.run_authored_agent(
                _run_request("struct_bot", {"user_message": "give me a number", "response_format": schema})
            )
            assert isinstance(run_resp, StreamingResponse)
            frames = await _collect_frames(run_resp)

            assert [f["type"] for f in frames] == ["structured_final", "stream.end"]
            assert frames[0]["data"] == {"value": 7}
            # The request response_format reached astream and threaded to the seam.
            assert captured["response_format"] == schema

    asyncio.run(run())


def test_an_authored_preset_does_not_reach_a_later_test(skeleton: Any) -> None:
    """The ``skeleton`` fixture registers no teardown for the presets a test authors,
    because it does not need one: nulling ``instance._app`` makes every test that takes
    the fixture build its own app, and the ``PresetManager`` and the tool registry both
    hang off that app. This test is what holds that invariant up.

    The preceding tests author ``support_bot`` and ``struct_bot`` and never remove them.
    Entering here on a fresh app, neither the spec map nor the live tool registry may
    carry either leaked preset; the fresh app carries the manifest's own ``tools_agent``.
    Drop ``monkeypatch.setattr(instance, "_app", None)``
    from the fixture and this test goes red: the memoised app still holds the previous
    test's registrations, so re-entering ``app_context`` raises ``Component already
    exists: tool:tools_agent`` before the assertions below are even reached. Selected on
    its own the check still holds, just without the preceding authoring to isolate it
    from."""

    async def run() -> None:
        async with instance.app.app_context(Manifest.model_validate(_MANIFEST)):
            manager = instance.build_app().preset_manager
            assert list(manager.registered_names()) == []
            assert list(manager.quarantined_names()) == []
            tools = await instance.app.tools.get_tools()
            assert "support_bot" not in tools
            assert "struct_bot" not in tools
            assert "tools_agent" in tools

    asyncio.run(run())


def test_boot_fails_loudly_when_the_identity_probe_healthcheck_raises(
    skeleton: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin that the identity-provider startup probe actually RUNS during this boot.

    The ``skeleton`` fixture's positive-path tests cannot tell a probe that passed
    through the no-op stand-in from a probe that never ran. This test can: the
    registered provider's ``healthcheck()`` is repointed at one that RAISES, and
    entering ``app_context`` must abort loudly with that failure. Access control
    being enabled is what registers the probe, so a boot where it is silently off
    (a stray ``ACCESS_CONTROL_ENABLE`` in the environment, a skeleton change
    dropping the handler) runs no probe, does not raise, and FAILS this test — the
    suite's claim that the real skeleton boots with access control ON is enforced,
    not assumed."""

    class _RaisingHealthcheckProvider(IdentityProvider):
        """An identity provider whose startup ``healthcheck`` refuses, standing in
        for a deployment against a backend the provider cannot reach."""

        def __init__(self, settings: Any) -> None: ...

        async def validate_token(self, token: str) -> None:
            return None

        async def healthcheck(self) -> None:
            raise RuntimeError("identity healthcheck refused")

    # Boot clears and re-populates the identity registry (the fixture wraps that
    # reset to re-register the offline stand-in); repoint the wrapper at the raising
    # provider so the probe that runs during boot resolves the failing one.
    def reset_then_register_raising() -> None:
        reset_registry()
        register_identity_provider("redis", _RaisingHealthcheckProvider)

    monkeypatch.setattr(lifecycle, "reset_identity_registry", reset_then_register_raising)
    reset_then_register_raising()

    async def run() -> None:
        with pytest.raises(RuntimeError, match="identity healthcheck refused"):
            async with instance.app.app_context(Manifest.model_validate(_MANIFEST)):
                raise AssertionError("boot must not succeed when the identity probe fails")

    asyncio.run(run())


def test_boot_fails_loudly_when_no_identity_provider_registered(skeleton: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin that the identity startup probe actually RUNS during this suite's boot.

    The companion of the RedisJSON pin above, for ``probe_identity_provider``: the
    ``skeleton`` fixture's re-registering wrapper is unwound back to the contract's
    plain ``reset_registry``, so boot's registry clear leaves no provider under the
    configured ``auth_provider`` name and the probe's factory lookup must abort
    the boot loudly with the unknown-provider error. The RedisJSON stand-in stays
    in place, so the identity probe is the only failure the boot can report. This
    is also what proves the fixture's wrapper is load-bearing: were boot not
    clearing the registry, or the probe not resolving through it, the fixture's
    up-front registration alone would carry the boot and this test would fail."""
    monkeypatch.setattr(lifecycle, "reset_identity_registry", reset_registry)

    async def run() -> None:
        with pytest.raises(RuntimeError, match=r"probe_identity_provider: KeyError.*Unknown identity provider"):
            async with instance.app.app_context(Manifest.model_validate(_MANIFEST)):
                raise AssertionError("boot must not succeed with no identity provider registered")

    asyncio.run(run())
