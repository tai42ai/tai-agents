"""Build the LangGraph run config for an agent invocation.

Two helpers, applied in order by the agents:

* :func:`build_run_config` overlays a run's memory keys (``thread_id``,
  ``resume_checkpoint_id``) and its ``recursion_limit`` onto the caller's
  ``langgraph_config`` base. Every agent that honors those parameters builds its
  run config through this one helper, so all of them — and both faces of each —
  apply the identical overlay semantics.
* :func:`init_langgraph_config` turns such a config into the config the compiled
  graph is actually run with: it ensures a ``thread_id`` and wires the monitoring
  callbacks.

:func:`init_langgraph_config` returns a NEW config dict, treating the caller's ``config`` as read-only: the
incoming mapping, its ``configurable`` section, and its ``callbacks`` list are
copied by value, never mutated. This lets one config object be handed to many
parallel invocations (e.g. the voting agent fanning the same config to N voter
tasks) without them colliding on a shared ``thread_id`` or accumulating each
other's callbacks.

Ensures a ``thread_id`` on the returned config's ``configurable`` section, wires
the active monitoring backend's trace callbacks onto ``config["callbacks"]``, and
resets the OpenTelemetry context when starting an independent (parent-less) trace
so the run does not inherit spans or tags from a caller. A caller-supplied
``configurable`` and ``callbacks`` are preserved by value and extended.

The active monitoring backend is reached through the ``tai42_app`` handle
(``tai42_app.monitoring.active``); the trace-context type comes from the
vendor-neutral monitoring contract.
"""

from __future__ import annotations

import uuid
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from tai42_contract.app import tai42_app
from tai42_contract.monitoring import TraceContext


def build_run_config(
    langgraph_config: dict[str, Any] | None,
    thread_id: str | None = None,
    resume_checkpoint_id: str | None = None,
    recursion_limit: int | None = None,
) -> dict[str, Any]:
    """Overlay a run's memory keys and step bound onto the caller's config base.

    The caller's ``langgraph_config`` is the base and is treated as read-only: the
    top level and the ``configurable`` section — the only parts this overlay writes —
    are rebuilt, so no caller dict is mutated (two runs sharing a base can never
    scribble a ``thread_id`` into each other's ``configurable``). Other sections a
    base may carry (``metadata``, ``tags``, ``callbacks``) are referenced, not copied,
    since this overlay never writes them; :func:`init_langgraph_config` rebuilds
    ``callbacks`` before it wires the monitoring hooks, so they are not mutated either.

    An explicit ``thread_id`` (the LangGraph memory key) and
    ``resume_checkpoint_id`` (the checkpoint a run forks from, mapped to
    ``checkpoint_id``) overlay the ``configurable`` section, so they win over the
    same keys carried in the base; ``recursion_limit`` — the standard
    ``RunnableConfig`` key the compiled graph reads — overlays the top level. Every
    other key of the base, in ``configurable`` and at the top level alike, survives
    the overlay untouched.

    With no memory key set, ``configurable`` is emitted as it stands in the base
    (an empty dict when the base carries none). :func:`init_langgraph_config` mints
    a fresh ``thread_id`` whenever the section holds none, so a keyless run gets an
    isolated thread rather than colliding with every other keyless run.
    """
    config = dict(langgraph_config or {})
    configurable = dict(config.get("configurable", {}))
    if thread_id is not None:
        configurable["thread_id"] = thread_id
    if resume_checkpoint_id is not None:
        configurable["checkpoint_id"] = resume_checkpoint_id
    config["configurable"] = configurable
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


def init_langgraph_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    source = config or {}
    new_config = dict(source)
    configurable = dict(source.get("configurable", {}))
    new_config["configurable"] = configurable

    if "thread_id" not in configurable:
        configurable["thread_id"] = session_id

    trace_id = configurable.get("monitoring_trace_id", str(uuid.uuid4()).replace("-", ""))
    parent_span_id = configurable.get("monitoring_parent_span_id")
    ctx = TraceContext(trace_id=trace_id, parent_span_id=parent_span_id)
    callbacks = tai42_app.monitoring.active.writer.get_monitoring_callbacks(ctx)

    # When starting an independent trace (no parent_span_id), reset OTel
    # context so this flow doesn't inherit spans or tags from a caller
    # (e.g. a builder agent invoking evaluation as a tool).
    if not parent_span_id:
        otel_context.attach(Context())

    new_config["callbacks"] = [*source.get("callbacks", []), *callbacks]
    return new_config
