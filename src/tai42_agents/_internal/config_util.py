"""Build the LangGraph run config for an agent invocation.

* :func:`build_run_config` overlays a run's memory keys (``thread_id``,
  ``resume_checkpoint_id``) and ``recursion_limit`` onto the caller's
  ``langgraph_config`` base.
* :func:`init_langgraph_config` ensures a ``thread_id``, wires the active
  monitoring backend's trace callbacks (``tai42_app.monitoring.active``), and
  resets the OpenTelemetry context when starting a parent-less trace.

Both treat the caller's config as read-only, copying the mapping, its
``configurable``, and its ``callbacks`` by value so one base config can feed many
parallel invocations without colliding on ``thread_id`` or accumulating each
other's callbacks.
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

    ``thread_id`` and ``resume_checkpoint_id`` (mapped to ``checkpoint_id``) overlay
    the ``configurable`` section, winning over the same keys in the base;
    ``recursion_limit`` overlays the top level. The base is read-only — the written
    sections are rebuilt, never mutated — so two runs sharing a base cannot scribble
    into each other's config. With no memory key set, ``configurable`` passes
    through as the base carries it; :func:`init_langgraph_config` mints a fresh
    ``thread_id`` for a keyless run.
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

    # Independent trace (no parent): reset OTel context so this flow doesn't
    # inherit spans or tags from a caller.
    if not parent_span_id:
        otel_context.attach(Context())

    new_config["callbacks"] = [*source.get("callbacks", []), *callbacks]
    return new_config
