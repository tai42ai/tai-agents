"""Tests for the usage helpers — ``aggregate_usage`` (per-call token/model
aggregation over a state) and ``usage_event`` (a single message → ``RunUsage``
stream event).

Fabricated ``AIMessage`` state (no LLM) covers the honest-omission zero path,
token summing across calls, the most-recent-model pick, and the
raise-on-malformed-usage_metadata boundary.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tai42_agents._internal.usage import CallUsage, aggregate_usage, usage_event


def _ai(input_tokens=None, output_tokens=None, model=None):
    message = AIMessage(content="x")
    if input_tokens is not None or output_tokens is not None:
        message.usage_metadata = {
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
        }
    if model is not None:
        message.response_metadata = {"model_name": model}
    return message


def test_no_usage_returns_zero_and_no_model():
    """Honest omission: no message carries usage → zeros, model None."""
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content="reply")]}
    assert aggregate_usage(state) == CallUsage(input_tokens=0, output_tokens=0, model=None)


def test_empty_state_returns_zero():
    assert aggregate_usage({}) == CallUsage(input_tokens=0, output_tokens=0, model=None)


def test_sums_tokens_across_messages_and_takes_latest_model():
    state = {
        "messages": [
            _ai(input_tokens=10, output_tokens=3, model="model-a"),
            _ai(input_tokens=5, output_tokens=2, model="model-b"),
        ]
    }
    assert aggregate_usage(state) == CallUsage(input_tokens=15, output_tokens=5, model="model-b")


def test_model_kept_when_a_later_message_omits_it():
    state = {
        "messages": [
            _ai(input_tokens=4, output_tokens=1, model="model-a"),
            _ai(input_tokens=1, output_tokens=1),  # no model_name
        ]
    }
    assert aggregate_usage(state).model == "model-a"


def test_non_mapping_usage_metadata_raises():
    message = AIMessage(content="x")
    message.usage_metadata = "not-a-mapping"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="usage_metadata is not a mapping"):
        aggregate_usage({"messages": [message]})


def test_non_integer_token_count_raises():
    message = AIMessage(content="x")
    message.usage_metadata = {"input_tokens": "lots", "output_tokens": 1}  # type: ignore[typeddict-item]
    with pytest.raises(ValueError, match=r"input_tokens.*not an integer"):
        aggregate_usage({"messages": [message]})


class TestUsageEvent:
    """``usage_event`` builds a single-message ``RunUsage`` stream event, or
    ``None`` when the provider surfaced no usage."""

    def test_none_when_provider_reports_no_usage(self):
        assert usage_event(SimpleNamespace(usage_metadata=None)) is None

    def test_reads_counts_and_model_label(self):
        message = SimpleNamespace(
            usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            response_metadata={"model_name": "m-1"},
        )
        usage = usage_event(message)
        assert usage is not None
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens, usage.model) == (1, 2, 3, "m-1")

    def test_model_falls_back_to_model_key(self):
        message = SimpleNamespace(
            usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            response_metadata={"model": "m-2"},
        )
        usage = usage_event(message)
        assert usage is not None
        assert usage.model == "m-2"
