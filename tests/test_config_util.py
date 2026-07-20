"""Unit tests for the LangGraph run-config builders.

Exercises ``init_langgraph_config`` against the recording ``tai42_app`` bound in
``conftest.py``: thread-id defaulting, the monitoring callbacks appended from the
recording backend, the ``TraceContext`` it is built with, and preservation of an
existing config's ``configurable`` and ``callbacks``. No live monitoring backend
is involved — the recording facet's writer stands in.

``build_run_config`` — the overlay every honoring agent builds its run config
through — is exercised here too: the memory keys and the step bound it overlays,
the caller keys it preserves, and the copy-by-value discipline that keeps a
caller's config dict out of the returned config.
"""

from __future__ import annotations

from typing import cast

from tai42_contract.app import tai42_app

from tai42_agents._internal.config_util import build_run_config, init_langgraph_config
from tests.conftest import RecordingMonitoringWriter

CALLBACKS = ["callback-a", "callback-b"]


def _writer() -> RecordingMonitoringWriter:
    return cast(RecordingMonitoringWriter, tai42_app.monitoring.active.writer)


def test_thread_id_defaulted_when_absent() -> None:
    config = init_langgraph_config()
    thread_id = config["configurable"]["thread_id"]
    assert isinstance(thread_id, str)
    assert thread_id


def test_callbacks_appended_from_active_backend() -> None:
    writer = _writer()
    before = len(writer.contexts)
    config = init_langgraph_config()

    assert config["callbacks"] == CALLBACKS
    # A fresh TraceContext was built from the (auto-generated) trace id and
    # handed to the active backend to produce the callbacks.
    assert len(writer.contexts) == before + 1
    ctx = writer.contexts[-1]
    assert ctx.trace_id
    assert ctx.parent_span_id is None


def test_existing_config_is_preserved_and_extended() -> None:
    writer = _writer()
    before = len(writer.contexts)
    existing = {
        "configurable": {
            "thread_id": "keep-me",
            "monitoring_trace_id": "trace-123",
            "monitoring_parent_span_id": "parent-9",
        },
        "callbacks": ["existing-cb"],
    }

    result = init_langgraph_config(existing)

    # A new config is returned, not the caller's object.
    assert result is not existing
    # The caller's thread id is carried over.
    assert result["configurable"]["thread_id"] == "keep-me"
    # Existing callbacks kept; monitoring callbacks appended after them.
    assert result["callbacks"] == ["existing-cb", *CALLBACKS]
    # The explicit trace/parent ids flow into the TraceContext.
    ctx = writer.contexts[-1]
    assert len(writer.contexts) == before + 1
    assert ctx.trace_id == "trace-123"
    assert ctx.parent_span_id == "parent-9"


def test_top_level_recursion_limit_is_preserved() -> None:
    # ``recursion_limit`` is a standard ``RunnableConfig`` top-level key; the
    # builder copies the incoming mapping by value, so an overlaid limit (a falsy
    # ``0`` too) survives onto the returned config the graph is invoked with. This
    # is the seam the tools/retrieval/mcp agents honor the parameter through.
    result = init_langgraph_config({"recursion_limit": 0})
    assert result["recursion_limit"] == 0

    result = init_langgraph_config({"recursion_limit": 12})
    assert result["recursion_limit"] == 12


def test_input_config_is_not_mutated() -> None:
    existing = {
        "configurable": {"monitoring_trace_id": "trace-abc"},
        "callbacks": ["existing-cb"],
    }

    init_langgraph_config(existing)

    # The caller's dict, its ``configurable`` section, and its ``callbacks``
    # list are all left exactly as they were passed in.
    assert existing == {
        "configurable": {"monitoring_trace_id": "trace-abc"},
        "callbacks": ["existing-cb"],
    }


def test_same_input_yields_independent_thread_ids_and_callbacks() -> None:
    # A single config object fanned out to parallel voters must not let them
    # collide on a shared thread id or accumulate each other's callbacks.
    shared = {"configurable": {}, "callbacks": ["existing-cb"]}

    first = init_langgraph_config(shared)
    second = init_langgraph_config(shared)

    # Each call gets its own fresh, distinct thread id.
    assert first["configurable"]["thread_id"] != second["configurable"]["thread_id"]
    # Callbacks are not accumulated across calls: each result carries exactly
    # the caller's callbacks plus one set of monitoring callbacks.
    assert first["callbacks"] == ["existing-cb", *CALLBACKS]
    assert second["callbacks"] == ["existing-cb", *CALLBACKS]
    # The shared input is untouched by either call.
    assert shared == {"configurable": {}, "callbacks": ["existing-cb"]}


def test_build_run_config_overlays_memory_keys_and_recursion_limit() -> None:
    # The memory keys land in ``configurable`` (``resume_checkpoint_id`` under the
    # LangGraph name ``checkpoint_id``); ``recursion_limit`` overlays the top level.
    config = build_run_config(None, "t-1", "cp-9", 0)
    assert config["configurable"] == {"thread_id": "t-1", "checkpoint_id": "cp-9"}
    assert config["recursion_limit"] == 0


def test_build_run_config_explicit_keys_win_over_the_base() -> None:
    base = {"configurable": {"thread_id": "from-base", "checkpoint_id": "cp-base"}, "recursion_limit": 12}
    config = build_run_config(base, "t-1", "cp-9", 3)
    assert config["configurable"]["thread_id"] == "t-1"
    assert config["configurable"]["checkpoint_id"] == "cp-9"
    assert config["recursion_limit"] == 3


def test_build_run_config_preserves_the_base_and_never_aliases_it() -> None:
    base = {"configurable": {"tenant": "acme", "thread_id": "t-base"}, "tags": ["t1"], "metadata": {"origin": "api"}}

    config = build_run_config(base, None, None, None)

    # Every caller key survives — the base's own pinned thread included.
    assert config == base
    # ...on a fresh dict, so no run can scribble on a config shared with another.
    assert config is not base
    assert config["configurable"] is not base["configurable"]

    config["configurable"]["thread_id"] = "overwritten"
    assert base["configurable"]["thread_id"] == "t-base"


def test_build_run_config_keyless_run_pins_no_thread() -> None:
    # With no memory key the ``configurable`` section carries no ``thread_id``, so
    # ``init_langgraph_config`` mints a fresh isolated one per run rather than the
    # runs colliding on one shared checkpoint thread.
    assert build_run_config(None) == {"configurable": {}}
    first = init_langgraph_config(build_run_config(None))["configurable"]["thread_id"]
    second = init_langgraph_config(build_run_config(None))["configurable"]["thread_id"]
    assert first != second
