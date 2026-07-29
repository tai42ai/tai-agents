"""Tests for the ``vqa_agent`` module.

Cover registration through the real decorator, the hand-rolled ``astream``
projection (token deltas → assembled final → summed usage), ``run`` draining
that stream to the final text with no app facets touched, and ``ToolInput``
validation. A scripted fake LLM whose ``astream`` yields ``AIMessageChunk``s
stands in for a real provider — no network, no live model.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk
from pydantic import ValidationError
from tai42_contract.agent import Agent
from tai42_contract.agent.events import MessageDelta, MessageFinal, RunUsage, StreamEvent
from tai42_contract.app import tai42_app
from tai42_kit.utils.data.json_schema_util import JsonSchemaValidationError

import tai42_agents.vqa_agent as vqa
from tai42_agents._internal.reject import reject_unhonored
from tai42_agents._internal.text import text_of
from tai42_agents.vqa_agent import VqaAgent, VqaAgentInput


class _FakeLLM:
    """A stand-in chat model whose ``astream`` yields two scripted chunks. The
    usage is split across both chunks so an assertion on ``total_tokens`` passes
    only when the accumulator sums them (a keep-last accumulator would give 2)."""

    async def astream(self, _messages: object) -> AsyncIterator[AIMessageChunk]:
        yield AIMessageChunk(
            content="Hel",
            usage_metadata={"input_tokens": 3, "output_tokens": 0, "total_tokens": 3},
        )
        yield AIMessageChunk(
            content="lo",
            usage_metadata={"input_tokens": 0, "output_tokens": 2, "total_tokens": 2},
        )


class _Settings:
    def with_fallbacks(self, _user_config: object) -> dict[str, object]:
        return {}


class _ProviderSettings:
    llm = "fake"


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the module's llm factory + settings at the scripted fake."""

    async def fake_get_llm(provider: str, **_kwargs: object) -> _FakeLLM:
        assert provider == "fake"
        return _FakeLLM()

    monkeypatch.setattr(vqa, "get_llm_async", fake_get_llm)
    monkeypatch.setattr(vqa, "llm_settings", lambda: _Settings())
    monkeypatch.setattr(vqa, "llm_provider_settings", lambda: _ProviderSettings())


async def _collect(agen: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in agen]


class _ScriptedLLM:
    """A stand-in chat model whose ``astream`` replays a fixed list of chunks — an
    empty list is an empty stream."""

    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self._chunks = chunks

    async def astream(self, _messages: object) -> AsyncIterator[AIMessageChunk]:
        for chunk in self._chunks:
            yield chunk


def _install_scripted_llm(monkeypatch: pytest.MonkeyPatch, chunks: list[AIMessageChunk]) -> None:
    """Point the module's llm factory + settings at a fake replaying ``chunks``."""

    async def fake_get_llm(provider: str, **_kwargs: object) -> _ScriptedLLM:
        assert provider == "fake"
        return _ScriptedLLM(chunks)

    monkeypatch.setattr(vqa, "get_llm_async", fake_get_llm)
    monkeypatch.setattr(vqa, "llm_settings", lambda: _Settings())
    monkeypatch.setattr(vqa, "llm_provider_settings", lambda: _ProviderSettings())


def test_decorator_registers_a_live_instance() -> None:
    agent = tai42_app.agents.get_agent("vqa_agent")
    assert isinstance(agent, VqaAgent)
    assert isinstance(agent, Agent)
    assert agent.tool_name == "vqa_agent"
    assert agent.ToolInput is VqaAgentInput


def test_astream_projects_the_llm_stream(fake_llm: None) -> None:
    events = asyncio.run(_collect(VqaAgent().astream(image_url="http://img", query="describe")))

    assert [event.type for event in events] == [
        "message_delta",
        "message_delta",
        "run_usage",
        "message_final",
    ]
    assert [event.text for event in events if isinstance(event, MessageDelta)] == ["Hel", "lo"]
    final = next(event for event in events if isinstance(event, MessageFinal))
    assert final.text == "Hello"
    assert final.final is True
    usage = next(event for event in events if isinstance(event, RunUsage))
    assert usage.input_tokens == 3
    assert usage.output_tokens == 2
    assert usage.total_tokens == 5


def test_run_drains_to_the_final_text(fake_llm: None) -> None:
    result = asyncio.run(VqaAgent().run(image_url="http://img", query="describe"))
    assert result == "Hello"


def test_astream_empty_stream_yields_nothing_and_run_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty model stream is an empty run: nothing accumulates, so no usage and no
    final are emitted, and draining it returns ``''`` — the documented empty-run result,
    never a partial."""
    _install_scripted_llm(monkeypatch, [])
    assert asyncio.run(_collect(VqaAgent().astream(image_url="http://img", query="describe"))) == []
    assert asyncio.run(VqaAgent().run(image_url="http://img", query="describe")) == ""


def test_astream_empty_text_chunk_yields_no_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk whose decoded text is empty contributes no ``MessageDelta``: only the
    accumulated usage surfaces, and the empty answer yields no final."""
    _install_scripted_llm(
        monkeypatch,
        [AIMessageChunk(content="", usage_metadata={"input_tokens": 4, "output_tokens": 0, "total_tokens": 4})],
    )
    events = asyncio.run(_collect(VqaAgent().astream(image_url="http://img", query="describe")))
    assert [event.type for event in events] == ["run_usage"]
    usage = next(event for event in events if isinstance(event, RunUsage))
    assert usage.total_tokens == 4


def test_astream_chunk_without_usage_emits_no_run_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the provider reports no token counts, ``usage_event`` returns ``None`` and the
    stream omits ``RunUsage`` — an honest omission — while the delta and the final for the
    answer text still flow."""
    _install_scripted_llm(monkeypatch, [AIMessageChunk(content="Hi")])
    events = asyncio.run(_collect(VqaAgent().astream(image_url="http://img", query="describe")))
    assert [event.type for event in events] == ["message_delta", "message_final"]
    final = next(event for event in events if isinstance(event, MessageFinal))
    assert final.text == "Hi"


def test_astream_whitespace_only_answer_yields_no_final(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only answer strips to empty, so no ``MessageFinal`` is emitted: the
    delta carrying the whitespace still flows, but the run has no terminal."""
    _install_scripted_llm(monkeypatch, [AIMessageChunk(content="   ")])
    events = asyncio.run(_collect(VqaAgent().astream(image_url="http://img", query="describe")))
    assert [event.type for event in events] == ["message_delta"]
    assert all(not isinstance(event, MessageFinal) for event in events)


def test_vqa_message_normalizes_a_url_through_the_handle(resource_manager: Any) -> None:
    """A public URL is normalized via the ``resource_manager`` handle (awaited) and
    passes through as the image content part, alongside the text query."""
    message = asyncio.run(vqa._vqa_message("http://img/cat.png", "describe"))
    assert message.content == [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "http://img/cat.png"}},
    ]
    assert resource_manager.media_calls == ["http://img/cat.png"]


def test_vqa_message_normalizes_a_storage_id_through_the_handle(resource_manager: Any) -> None:
    """A storage id is resolved via the same handle to a base64 data-URI image
    content part — proving VqaAgent works with stored images, not just URLs."""
    message = asyncio.run(vqa._vqa_message("skills/cat.png", "describe"))
    content = message.content
    assert isinstance(content, list)
    text_part, image_part = content
    assert text_part == {"type": "text", "text": "describe"}
    assert isinstance(image_part, dict)
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/")
    assert resource_manager.media_calls == ["skills/cat.png"]


def test_run_rejects_non_image_media_through_the_handle(fake_llm: None, resource_manager: Any) -> None:
    """A resolvable non-image source (a ``.pdf``) is rejected by the resource
    manager handle's image-only guard, and the vqa media path propagates that
    ``ValueError`` rather than sending a non-image to the model. Draining ``run``
    exercises the same ``astream`` media path."""
    with pytest.raises(ValueError, match="image"):
        asyncio.run(VqaAgent().run(image_url="notes.pdf", query="describe"))
    assert resource_manager.media_calls == ["notes.pdf"]


_VQA_SCHEMA = {"title": "Answer", "type": "object", "properties": {"answer": {"type": "string"}}}


class _StructuredLLM:
    """A stand-in chat model whose ``with_structured_output`` binds a schema and
    returns a runner whose ``ainvoke`` yields a fixed structured payload; its
    ``astream`` raises so a structured run never falls back to the token path."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.captured: dict[str, Any] = {}

    def with_structured_output(self, schema: Any, include_raw: bool = True) -> Any:
        self.captured["schema"] = schema
        self.captured["include_raw"] = include_raw
        outer = self

        class _Runner:
            async def ainvoke(self, messages: Any) -> Any:
                outer.captured["messages"] = messages
                return outer._payload

        return _Runner()

    def astream(self, _messages: object) -> Any:  # pragma: no cover - never reached on the structured path
        raise AssertionError("structured path must not stream tokens")


def _install_structured_llm(monkeypatch: pytest.MonkeyPatch, llm: _StructuredLLM) -> None:
    async def fake_get_llm(provider: str, **_kwargs: object) -> _StructuredLLM:
        assert provider == "fake"
        return llm

    monkeypatch.setattr(vqa, "get_llm_async", fake_get_llm)
    monkeypatch.setattr(vqa, "llm_settings", lambda: _Settings())
    monkeypatch.setattr(vqa, "llm_provider_settings", lambda: _ProviderSettings())


def test_astream_with_response_format_emits_one_structured_final(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a ``response_format`` set, the completion forces the structured output via
    ``with_structured_output`` and emits exactly one :class:`StructuredFinal` (no
    token deltas), passing the schema straight to the provider (converter off the
    force path)."""
    from tai42_contract.agent.events import StructuredFinal

    llm = _StructuredLLM({"answer": "a cat"})
    _install_structured_llm(monkeypatch, llm)

    events = asyncio.run(
        _collect(VqaAgent().astream(image_url="http://img", query="describe", response_format=_VQA_SCHEMA))
    )
    assert [type(event) for event in events] == [StructuredFinal]
    final = events[0]
    assert isinstance(final, StructuredFinal)
    assert final.data == {"answer": "a cat"}
    assert llm.captured["schema"] == _VQA_SCHEMA
    assert llm.captured["include_raw"] is False


def test_run_with_response_format_returns_the_structured_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run`` with a ``response_format`` drains its structured astream and returns the
    structured object rather than text."""
    llm = _StructuredLLM({"answer": "a cat"})
    _install_structured_llm(monkeypatch, llm)
    result = asyncio.run(VqaAgent().run(image_url="http://img", query="describe", response_format=_VQA_SCHEMA))
    assert result == {"answer": "a cat"}


def test_run_response_format_without_title_raises_loudly(fake_llm: None) -> None:
    """A ``response_format`` JSON-Schema dict lacking a top-level ``"title"`` is
    rejected loudly at the run seam."""
    with pytest.raises(ValueError, match="top-level 'title'"):
        asyncio.run(VqaAgent().run(image_url="http://img", query="describe", response_format={"type": "object"}))


def test_astream_response_format_without_title_raises_loudly(fake_llm: None) -> None:
    """The streaming face — the one the public run door drives — rejects an untitled
    ``response_format`` up front, exactly as the invoke face does."""
    with pytest.raises(ValueError, match="top-level 'title'"):
        asyncio.run(
            _collect(VqaAgent().astream(image_url="http://img", query="describe", response_format={"type": "object"}))
        )


def test_run_response_format_unparseable_finalization_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unparseable structured-finalization output propagates the raise loudly —
    never a silent text fallback."""

    class _BoomLLM(_StructuredLLM):
        def with_structured_output(self, schema: Any, include_raw: bool = True) -> Any:
            class _Runner:
                async def ainvoke(self, messages: Any) -> Any:
                    raise ValueError("model returned unparseable structured output")

            return _Runner()

    _install_structured_llm(monkeypatch, _BoomLLM(None))
    with pytest.raises(ValueError, match="unparseable structured output"):
        asyncio.run(VqaAgent().run(image_url="http://img", query="describe", response_format=_VQA_SCHEMA))


def test_run_response_format_nonconforming_structured_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A structured payload violating a schema constraint keyword raises loudly
    from the validation step instead of being returned."""
    schema = {
        "title": "Answer",
        "type": "object",
        "properties": {"answer": {"type": "string", "minLength": 1}},
        "required": ["answer"],
    }
    _install_structured_llm(monkeypatch, _StructuredLLM({"answer": ""}))
    with pytest.raises(JsonSchemaValidationError):
        asyncio.run(VqaAgent().run(image_url="http://img", query="describe", response_format=schema))


# Every key in ``_UNHONORED_REASONS`` paired with a representative SET value: a
# non-empty collection for a collection param, a meaningful value for a scalar
# (``resume=False`` pins the falsy-but-present case). Each is rejected on BOTH faces,
# so dropping any one key from the guard's map fails a case here.
_UNHONORED_CASES = [
    ("tools", [object()]),
    ("tool_names", ["x"]),
    ("presets", [object()]),
    ("subagents", [object()]),
    ("system_message", "sys"),
    ("user_message", "usr"),
    ("strategy", "react"),
    ("interrupt_on", {"tool": True}),
    ("skills", ["s"]),
    ("inline_skills", [{"name": "s", "content": "c"}]),
    ("recursion_limit", 5),
    ("thread_id", "t"),
    ("resume", False),
    ("resume_checkpoint_id", "cp"),
    ("checkpoint_provider", "redis"),
    ("store_provider", "redis"),
]


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_run_rejects_unhonored_contract_param(fake_llm: None, param: str, value: Any) -> None:
    """Every contract ``Agent.run`` parameter with no seam in a single completion
    is rejected loudly on the run face, naming the offending parameter and the exact
    ``run`` face it was called on."""
    with pytest.raises(RuntimeError, match=rf"vqa_agent\.run does not support .*\b{param}\b"):
        # The base ``Agent.run`` signature types some of these non-optional, so the mismatch is expected.
        asyncio.run(VqaAgent().run(image_url="http://img", query="describe", **{param: value}))  # type: ignore[arg-type]


@pytest.mark.parametrize(("param", "value"), _UNHONORED_CASES)
def test_astream_rejects_unhonored_contract_param(fake_llm: None, param: str, value: Any) -> None:
    """Parity with :meth:`run`: the stream face rejects the same unhonored contract
    parameters loudly, naming the offending parameter and the exact ``astream`` face.
    The run-face guard is load-bearing: dropping it would surface the ``astream``
    token on a ``run`` call and fail :func:`test_run_rejects_unhonored_contract_param`."""
    with pytest.raises(RuntimeError, match=rf"vqa_agent\.astream does not support .*\b{param}\b"):
        asyncio.run(_collect(VqaAgent().astream(image_url="http://img", query="describe", **{param: value})))  # type: ignore[arg-type]


def test_unhonored_cases_cover_the_full_reasons_map() -> None:
    """Every key in the guard's reasons map has a parametrized reject case, so a key
    added to the map without a matching test fails here immediately."""
    assert {param for param, _ in _UNHONORED_CASES} == set(vqa._UNHONORED_REASONS)


# The unhonored params whose ABC ``Agent.run`` default is an empty collection (``()`` /
# ``""``), read from the contract signature — the independent source of truth for which
# unhonored params are collection-typed. Intersected with this agent's reasons map it is
# exactly the set ``_UNHONORED_COLLECTION_PARAMS`` must classify as collections. The
# empty-collection test below parametrizes from HERE, not from the frozenset, so a member
# dropped from the frozenset (reclassifying it as a scalar) turns a case red rather than
# silently vanishing.
_EMPTY_COLLECTION_ABC_DEFAULTS = frozenset(
    name
    for name, parameter in inspect.signature(Agent.run).parameters.items()
    if isinstance(parameter.default, (tuple, list, str)) and not parameter.default
)
_COLLECTION_REJECT_PARAMS = sorted(vqa._UNHONORED_REASONS.keys() & _EMPTY_COLLECTION_ABC_DEFAULTS)


def test_collection_params_match_the_abc_collection_defaults() -> None:
    """``_UNHONORED_COLLECTION_PARAMS`` is exactly this agent's unhonored params whose ABC
    default is an empty collection: no scalar wrongly listed (which would let a meaningful
    falsy value slip through), none dropped (which would over-reject the not-requested
    empty default)."""
    assert set(vqa._UNHONORED_COLLECTION_PARAMS) == set(_COLLECTION_REJECT_PARAMS)


@pytest.mark.parametrize("empty", [[], ""])
@pytest.mark.parametrize("param", _COLLECTION_REJECT_PARAMS)
def test_reject_unhonored_permits_empty_collection_param(param: str, empty: object) -> None:
    """An empty collection is the ABC's "not requested" default for a collection parameter,
    so the guard does not raise for it — in either falsy empty form (``[]`` / ``""``). Were
    the parameter dropped from ``_UNHONORED_COLLECTION_PARAMS`` it would be classified as a
    scalar (set whenever it is not ``None``) and this empty value would raise."""
    reject_unhonored(
        "vqa_agent.run",
        {param: empty},
        vqa._UNHONORED_REASONS,
        collection_params=vqa._UNHONORED_COLLECTION_PARAMS,
    )


@pytest.mark.parametrize(("param", "value"), [("resume", False), ("recursion_limit", 0), ("strategy", "")])
def test_run_rejects_falsy_but_meaningful_contract_param(fake_llm: None, param: str, value: Any) -> None:
    """A falsy-but-meaningful scalar (``resume=False`` / ``recursion_limit=0`` /
    ``strategy=""``) is still a request the agent cannot honor and raises — a scalar
    is set whenever it is not ``None``, so it never slips through a truthiness gate."""
    with pytest.raises(RuntimeError, match=rf"vqa_agent\.run does not support .*\b{param}\b"):
        asyncio.run(VqaAgent().run(image_url="http://img", query="describe", **{param: value}))  # type: ignore[arg-type]


@pytest.mark.parametrize("param", ["response_format", "resume", "thread_id", "recursion_limit", "strategy"])
def test_unhonored_none_passes_the_guard(fake_llm: None, param: str) -> None:
    """The ABC's ``None`` "not requested" sentinel passes the guard rather than being
    over-rejected: an explicit ``response_format=None`` / ``resume=None`` / … reaches
    the single completion and returns its text, in parity with the other agents."""
    result = asyncio.run(VqaAgent().run(image_url="http://img", query="describe", **{param: None}))
    assert result == "Hello"


def test_unhonored_collection_empty_passes_the_guard(fake_llm: None) -> None:
    """An empty collection is the ABC's "not requested" sentinel and passes the guard:
    ``tools=()`` / ``system_message=""`` reach the completion rather than raising."""
    result = asyncio.run(
        VqaAgent().run(image_url="http://img", query="describe", tools=(), subagents=[], system_message="")
    )
    assert result == "Hello"


def test_plain_extension_kwargs_pass_through_both_faces(fake_llm: None) -> None:
    """A kwarg that is NOT a contract ``Agent.run`` parameter is the ABC's role
    extension point and stays allowed — it is not rejected on either face."""
    result = asyncio.run(VqaAgent().run(image_url="http://img", query="describe", vqa_specific_extra="ok"))
    assert result == "Hello"
    stream = VqaAgent().astream(image_url="http://img", query="describe", vqa_specific_extra="ok")
    events = asyncio.run(_collect(stream))
    assert [event.type for event in events][-1] == "message_final"


def test_honored_params_reach_the_llm_seam_on_both_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    """``llm_provider`` and ``llm_kwargs`` are honored: they reach the llm factory
    seam identically on the run and astream faces."""
    captured: dict[str, object] = {}

    async def capture_get_llm(provider: str, **kwargs: object) -> _FakeLLM:
        captured["provider"] = provider
        captured["kwargs"] = kwargs
        return _FakeLLM()

    class _CaptureSettings:
        def with_fallbacks(self, user_config: object) -> dict[str, object]:
            captured["fallbacks"] = user_config
            return {"merged": True}

    monkeypatch.setattr(vqa, "get_llm_async", capture_get_llm)
    monkeypatch.setattr(vqa, "llm_settings", lambda: _CaptureSettings())
    monkeypatch.setattr(vqa, "llm_provider_settings", lambda: _ProviderSettings())

    stream = VqaAgent().astream(image_url="http://img", query="q", llm_provider="fake", llm_kwargs={"temperature": 0.1})
    asyncio.run(_collect(stream))
    assert captured["provider"] == "fake"
    assert captured["fallbacks"] == {"temperature": 0.1}
    assert captured["kwargs"] == {"merged": True}

    captured.clear()
    asyncio.run(VqaAgent().run(image_url="http://img", query="q", llm_provider="fake", llm_kwargs={"temperature": 0.1}))
    assert captured["provider"] == "fake"
    assert captured["fallbacks"] == {"temperature": 0.1}
    assert captured["kwargs"] == {"merged": True}


def test_tool_input_rejects_unknown_key() -> None:
    """``extra="forbid"`` rejects an unknown key loudly rather than dropping a typo."""
    with pytest.raises(ValidationError):
        VqaAgentInput(image_url="http://img", query="describe", llm_providerr="oops")  # type: ignore[call-arg]


def test_tool_input_requires_image_url_and_query() -> None:
    valid = VqaAgentInput(image_url="http://img", query="describe")
    assert valid.image_url == "http://img"
    assert valid.query == "describe"
    assert valid.llm_provider is None
    assert valid.llm_kwargs is None

    with pytest.raises(ValidationError):
        VqaAgentInput(image_url="http://img")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        VqaAgentInput(query="describe")  # type: ignore[call-arg]


class TestTextOf:
    """The shared ``text_of`` helper (``tai42_agents._internal.text``) that both the
    vqa stream and the tools-agent projection use to decode message text."""

    def test_string_content_returned_verbatim(self) -> None:
        assert text_of(SimpleNamespace(content="hi")) == "hi"

    def test_list_content_joins_text_blocks_and_skips_non_text(self) -> None:
        message = SimpleNamespace(
            content=[
                "a",  # a bare string block
                {"type": "text", "text": "b"},  # an explicit text block
                {"text": "c"},  # type omitted -> treated as text
                {"type": "image_url", "image_url": {"url": "http://img"}},  # skipped
            ]
        )
        assert text_of(message) == "abc"

    def test_missing_content_is_empty(self) -> None:
        assert text_of(object()) == ""

    def test_non_str_non_list_content_is_empty(self) -> None:
        assert text_of(SimpleNamespace(content=123)) == ""
