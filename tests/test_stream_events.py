"""Tests for ``aproject_agent_events`` — the shared updates/messages projection.

Feeds a fabricated ``astream`` (no LLM) and asserts the normalized event
vocabulary, the monotonic synthetic-id behavior, the final-message fallback, and
the raise-on-malformed-chunk boundary.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Overwrite
from tai42_kit.utils.data.json_schema_util import JsonSchemaValidationError

import tai42_agents._internal.stream_events as stream_events
from tai42_agents._internal.stream_events import (
    MessageDelta,
    MessageFinal,
    ReasoningStep,
    StructuredFinal,
    ToolCallStep,
    ToolResultStep,
    aproject_agent_events,
    astream_tools_agent_events,
)
from tai42_agents._internal.text import text_of


class _StubAgent:
    """An agent whose astream yields a fixed list of (mode, chunk) items."""

    def __init__(self, items):
        self._items = items

    async def astream(self, agent_input, config, stream_mode=None):
        for item in self._items:
            yield item


def _collect(items, response_format=None):
    async def go():
        return [
            event
            async for event in aproject_agent_events(
                _StubAgent(items), {"messages": []}, {}, response_format=response_format
            )
        ]

    return asyncio.run(go())


def test_projection_basic():
    ai = AIMessage(content="", tool_calls=[{"id": "t1", "name": "task", "args": {"x": 1}}])
    items = [
        ("updates", {"model": {"messages": [ai]}}),
        ("updates", {"tools": {"messages": [ToolMessage(content="ok", name="task", tool_call_id="t1")]}}),
        ("messages", (AIMessageChunk(content="Hel"), {})),
        ("messages", (AIMessageChunk(content="lo"), {})),
    ]
    events = _collect(items)
    assert any(isinstance(e, ToolCallStep) and e.call_id == "t1" and e.tool == "task" for e in events)
    assert any(isinstance(e, ToolResultStep) and e.result == "ok" and e.call_id == "t1" for e in events)
    assert sum(isinstance(e, MessageDelta) for e in events) == 2
    assert any(isinstance(e, MessageFinal) and e.text == "Hello" for e in events)


def test_synthetic_ids_are_monotonic_and_unique():
    """No-id tool calls get distinct ids from a monotonic counter (not set size)."""
    # AIMessage's constructor requires an id on each tool_call, so assign the
    # id-less calls after construction to exercise the synthesis path.
    ai1 = AIMessage(content="")
    ai1.tool_calls = [{"name": "a", "args": {}, "id": None}]
    ai2 = AIMessage(content="")
    ai2.tool_calls = [{"name": "b", "args": {}, "id": None}]
    items = [
        ("updates", {"model": {"messages": [ai1]}}),
        ("updates", {"model": {"messages": [ai2]}}),
    ]
    ids = [e.call_id for e in _collect(items) if isinstance(e, ToolCallStep)]
    assert ids == ["__synthetic_tool_call_0", "__synthetic_tool_call_1"]


def test_synthetic_id_does_not_collide_with_real_id_either_direction():
    """Synthesized ids live in their own namespace, so they can never collide
    with a provider's real id — whether the real id appears before OR after the
    id-less call."""
    no_id = AIMessage(content="")
    no_id.tool_calls = [{"name": "a", "args": {}, "id": None}]
    real_call0 = AIMessage(content="", tool_calls=[{"name": "b", "args": {}, "id": "call_0"}])
    items = [
        ("updates", {"model": {"messages": [no_id]}}),
        ("updates", {"model": {"messages": [real_call0]}}),
    ]
    calls = [e for e in _collect(items) if isinstance(e, ToolCallStep)]
    ids = [e.call_id for e in calls]
    assert len(calls) == 2  # neither dropped, in either direction
    assert len(set(ids)) == 2
    assert "call_0" in ids  # the real id preserved


def test_duplicate_tool_call_id_deduped():
    ai = AIMessage(content="", tool_calls=[{"id": "t1", "name": "task", "args": {}}])
    # same message observed twice across chunks
    items = [
        ("updates", {"model": {"messages": [ai]}}),
        ("updates", {"model": {"messages": [ai]}}),
    ]
    calls = [e for e in _collect(items) if isinstance(e, ToolCallStep)]
    assert len(calls) == 1


def test_final_fallback_from_updates_when_no_deltas():
    """A final answer surfaced only on the updates channel still yields MessageFinal."""
    final = AIMessage(content="done")  # no tool calls, no token deltas
    events = _collect([("updates", {"model": {"messages": [final]}})])
    assert any(isinstance(e, MessageFinal) and e.text == "done" for e in events)


def test_deltas_win_over_updates_fallback():
    """When tokens stream, the assembled deltas are the final — not the updates text."""
    final = AIMessage(content="IGNORED")
    items = [
        ("updates", {"model": {"messages": [final]}}),
        ("messages", (AIMessageChunk(content="streamed"), {})),
    ]
    finals = [e for e in _collect(items) if isinstance(e, MessageFinal)]
    assert len(finals) == 1
    assert finals[0].text == "streamed"


def test_reasoning_step_emitted():
    ai = AIMessage(content=[{"type": "thinking", "thinking": "hmm"}])
    events = _collect([("updates", {"model": {"messages": [ai]}})])
    assert any(isinstance(e, ReasoningStep) and e.text == "hmm" for e in events)


def test_reasoning_omission_is_not_an_error():
    """A message with no reasoning simply produces no ReasoningStep (honest omission)."""
    ai = AIMessage(content="plain answer")
    events = _collect([("updates", {"model": {"messages": [ai]}})])
    assert not any(isinstance(e, ReasoningStep) for e in events)


def test_structured_final_emitted_from_updates():
    """When a node update carries structured_response, emit StructuredFinal once."""
    items = [
        ("updates", {"model": {"messages": [AIMessage(content="done")], "structured_response": {"a": 1}}}),
    ]
    finals = [e for e in _collect(items) if isinstance(e, StructuredFinal)]
    assert len(finals) == 1
    assert finals[0].data == {"a": 1}


def test_no_structured_final_without_response():
    items = [("updates", {"model": {"messages": [AIMessage(content="x")]}})]
    assert not any(isinstance(e, StructuredFinal) for e in _collect(items))


_PERSON_SCHEMA = {
    "title": "Person",
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
    "required": ["name", "age"],
}


def test_structured_final_validated_against_response_format():
    """With a response_format, the terminal payload is validated before emission."""
    items = [
        ("updates", {"model": {"messages": [], "structured_response": {"name": "ada", "age": 36}}}),
    ]
    finals = [e for e in _collect(items, response_format=_PERSON_SCHEMA) if isinstance(e, StructuredFinal)]
    assert len(finals) == 1
    assert finals[0].data == {"name": "ada", "age": 36}


def test_structured_final_nonconforming_raises():
    """A structured payload violating a schema constraint keyword raises loudly
    instead of being emitted."""
    items = [
        ("updates", {"model": {"messages": [], "structured_response": {"name": "ada", "age": -1}}}),
    ]
    with pytest.raises(JsonSchemaValidationError):
        _collect(items, response_format=_PERSON_SCHEMA)


def test_structured_final_keeps_latest_and_follows_message_final():
    """Multiple structured updates → keep the latest; emit StructuredFinal after MessageFinal."""
    items = [
        ("updates", {"model": {"messages": [], "structured_response": {"v": 1}}}),
        ("updates", {"model": {"messages": [], "structured_response": {"v": 2}}}),
        ("updates", {"model": {"messages": [AIMessage(content="final text")]}}),
    ]
    evs = _collect(items)
    sf = [e for e in evs if isinstance(e, StructuredFinal)]
    mf = [e for e in evs if isinstance(e, MessageFinal)]
    assert len(sf) == 1
    assert sf[0].data == {"v": 2}  # latest non-null wins
    assert len(mf) == 1
    assert evs.index(sf[0]) > evs.index(mf[0])  # structured after final text


def test_structured_final_after_streamed_delta_final():
    """Structured output coexists with token-streamed text: MessageFinal (from
    deltas) then StructuredFinal."""
    items = [
        ("messages", (AIMessageChunk(content="hello"), {})),
        ("updates", {"model": {"messages": [], "structured_response": {"v": 1}}}),
    ]
    evs = _collect(items)
    mf = [e for e in evs if isinstance(e, MessageFinal)]
    sf = [e for e in evs if isinstance(e, StructuredFinal)]
    assert mf
    assert mf[0].text == "hello"
    assert sf
    assert sf[0].data == {"v": 1}
    assert evs.index(sf[0]) > evs.index(mf[0])


def test_bare_dict_item_is_treated_as_an_update():
    """A bare single-mode chunk (not a (mode, chunk) pair) is decoded as updates."""
    events = _collect([{"model": {"messages": [AIMessage(content="done")]}}])
    assert any(isinstance(e, MessageFinal) and e.text == "done" for e in events)


def test_malformed_messages_chunk_raises():
    """A messages-mode chunk that is not a (message, metadata) pair raises."""
    with pytest.raises(ValueError, match="messages-mode stream chunk"):
        _collect([("messages", "not-a-tuple")])


def test_malformed_messages_chunk_wrong_arity_raises():
    """A messages-mode chunk tuple of the wrong length cannot be unpacked → raises."""
    with pytest.raises(ValueError, match="messages-mode stream chunk"):
        _collect([("messages", (AIMessageChunk(content="x"), {}, "extra"))])


def test_malformed_updates_chunk_raises():
    """An updates-mode chunk that is not a node->update mapping raises."""
    with pytest.raises(ValueError, match="updates-mode stream chunk"):
        _collect([("updates", "not-a-dict")])


def test_per_node_non_dict_update_raises():
    """A per-node update value that is neither a mapping nor a known benign shape
    cannot be decoded and raises (not silently skipped)."""
    with pytest.raises(ValueError, match="updates-mode node update for 'model'"):
        _collect([("updates", {"model": "not-a-mapping"})])


def test_per_node_none_update_is_skipped():
    """A node that wrote no state (``None``) is benign and does not raise; a
    sibling node with real messages still projects."""
    items = [
        ("updates", {"noop": None, "model": {"messages": [AIMessage(content="done")]}}),
    ]
    events = _collect(items)
    assert any(isinstance(e, MessageFinal) and e.text == "done" for e in events)


def test_interrupt_channel_update_is_skipped():
    """The ``__interrupt__`` channel's tuple value is benign here (the resume path
    reads interrupts from the graph snapshot), so it does not raise."""
    items = [
        ("updates", {"__interrupt__": ("some-interrupt-payload",)}),
        ("updates", {"model": {"messages": [AIMessage(content="done")]}}),
    ]
    events = _collect(items)
    assert any(isinstance(e, MessageFinal) and e.text == "done" for e in events)


def test_list_valued_node_update_expands_each_mapping_in_order():
    """A node that writes the same channel more than once yields a list of update
    mappings; each is decoded in order (tool call from the first, final text from
    the second)."""
    items = [
        (
            "updates",
            {
                "model": [
                    {"messages": [AIMessage(content="", tool_calls=[{"id": "c1", "name": "search", "args": {}}])]},
                    {"messages": [AIMessage(content="second")]},
                ]
            },
        ),
    ]
    events = _collect(items)
    call = next(e for e in events if isinstance(e, ToolCallStep))
    assert call.tool == "search"
    assert call.call_id == "c1"
    assert any(isinstance(e, MessageFinal) and e.text == "second" for e in events)


def test_non_dict_element_in_list_update_raises():
    """A non-mapping element inside a list-valued update cannot be decoded and
    raises (not silently skipped)."""
    with pytest.raises(ValueError, match="updates-mode node update for 'model'"):
        _collect([("updates", {"model": [{"messages": []}, "bad"]})])


def test_overwrite_wrapped_messages_are_unwrapped():
    """A ``messages`` update wrapped in langgraph's ``Overwrite`` (as deepagents'
    patch-tool-calls middleware writes it) is unwrapped before iterating, so the
    ToolResultStep still surfaces."""
    items = [
        (
            "updates",
            {"tools": {"messages": Overwrite(value=[ToolMessage(content="ok", name="task", tool_call_id="t1")])}},
        ),
    ]
    events = _collect(items)
    assert any(isinstance(e, ToolResultStep) and e.result == "ok" and e.call_id == "t1" for e in events)


def test_media_tool_result_is_media_safe_and_json_native():
    """A media-carrying tool result is media-safe: (a) its content is projected
    verbatim, NEVER routed through ``text_of`` (which strips non-text blocks); and
    (b) the projected ``ToolResultStep.result`` is JSON-native, so
    ``model_dump_json`` round-trips it without raising."""
    media: list[str | dict[str, Any]] = [{"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}]
    items = [("updates", {"tools": {"messages": [ToolMessage(content=media, name="snap", tool_call_id="c1")]}})]
    result_steps = [e for e in _collect(items) if isinstance(e, ToolResultStep)]
    assert len(result_steps) == 1
    step = result_steps[0]

    # Guard (a): the content passes through untouched. ``text_of`` would strip the
    # image block to "" — so an equal, non-empty result proves it was NOT applied.
    assert step.result == media
    assert text_of(SimpleNamespace(content=media)) == ""

    # Guard (b): the result is JSON-native, so serialization can't drop or fail on it.
    dumped = step.model_dump_json()
    assert json.loads(dumped)["result"] == media


def test_astream_tools_agent_events_threads_build_into_projection(monkeypatch):
    """The build->project glue runs its real body: monkeypatch only the underlying
    build to return a stub agent, and assert the projected events come through."""
    items = [
        ("updates", {"model": {"messages": [AIMessage(content="answer")]}}),
        ("messages", (AIMessageChunk(content="answer"), {})),
    ]

    async def fake_build(*_args, **_kwargs):
        return _StubAgent(items), {"messages": []}, {}

    monkeypatch.setattr(stream_events, "_build_agent_and_input", fake_build)

    async def go():
        return [
            event
            async for event in astream_tools_agent_events(
                system_message="sys",
                user_message=["hi"],
                tools=[],
            )
        ]

    events = asyncio.run(go())
    assert any(isinstance(e, MessageFinal) and e.text == "answer" for e in events)
    assert sum(isinstance(e, MessageDelta) for e in events) == 1
