"""Per-call LLM usage extraction from a LangChain agent run state.

Each ``AIMessage`` in ``state["messages"]`` may expose ``usage_metadata``
(``input_tokens`` / ``output_tokens``) and ``response_metadata.model_name``.
``aggregate_usage`` sums the token counts across every ``AIMessage`` and takes the
model label from the most recent one that carries it.

When no message carries ``usage_metadata`` a zero-token ``CallUsage(model=None)``
is returned; a present but malformed ``usage_metadata`` (not a mapping, or a
non-integer token count) raises ``ValueError`` rather than being coerced away.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tai42_contract.agent.events import RunUsage


@dataclass(frozen=True)
class CallUsage:
    """Token counters and model label for one agent invocation.

    ``model`` is None when no ``AIMessage`` in the state exposes a
    ``response_metadata.model_name``.
    """

    input_tokens: int
    output_tokens: int
    model: str | None


@dataclass(frozen=True)
class AgentInvokeResult:
    """An agent invocation's result: user output, per-call usage, and structured
    response.

    ``structured`` is ``None`` when no ``response_format`` was requested; otherwise
    it is ``state["structured_response"]`` validated against that format (a missing
    or non-conforming result raises in the invoke path, never surfaces as ``None``).
    """

    output: str
    usage: CallUsage
    structured: Any = None


def _int_token_count(usage_metadata: Mapping[str, Any], key: str) -> int:
    """Read ``key`` from ``usage_metadata`` as an int token count.

    A missing / null value counts as zero (honest omission of one field). A
    present but non-integer value is malformed and RAISES ``ValueError``.
    """
    value = usage_metadata.get(key, 0) or 0
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AIMessage.usage_metadata[{key!r}] is not an integer token count: {value!r}") from exc


def aggregate_usage(state: dict[str, Any]) -> CallUsage:
    """Sum ``usage_metadata`` across every ``AIMessage`` in ``state["messages"]``
    and pick the most recent ``response_metadata.model_name``.

    Returns a zero-token ``CallUsage`` with ``model=None`` when no message
    carries usage (honest provider omission / fabricated test state). A present
    but malformed ``usage_metadata`` raises ``ValueError``.
    """
    input_tokens = 0
    output_tokens = 0
    model: str | None = None
    for message in state.get("messages", []) or []:
        usage_metadata = getattr(message, "usage_metadata", None)
        if usage_metadata:
            if not isinstance(usage_metadata, Mapping):
                raise ValueError(f"AIMessage.usage_metadata is not a mapping: {usage_metadata!r}")
            input_tokens += _int_token_count(usage_metadata, "input_tokens")
            output_tokens += _int_token_count(usage_metadata, "output_tokens")
        response_metadata = getattr(message, "response_metadata", None) or {}
        candidate_model = response_metadata.get("model_name")
        if candidate_model:
            model = candidate_model
    return CallUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )


def usage_event(message: Any) -> RunUsage | None:
    """Build a :class:`RunUsage` stream event from a single message's
    ``usage_metadata`` and model label, or ``None`` when the provider surfaced no
    usage (an honest omission — fewer events, never a fabricated zero record)."""
    usage_metadata = getattr(message, "usage_metadata", None)
    if not usage_metadata:
        return None
    response_metadata = getattr(message, "response_metadata", None) or {}
    return RunUsage(
        input_tokens=usage_metadata.get("input_tokens"),
        output_tokens=usage_metadata.get("output_tokens"),
        total_tokens=usage_metadata.get("total_tokens"),
        model=response_metadata.get("model_name") or response_metadata.get("model"),
    )
