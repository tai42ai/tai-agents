"""``vqa_agent`` as an :class:`Agent` — visual question answering.

The simplest agent in the package: a single multimodal LLM completion. It takes
an image reference (a public URL *or* a storage id) and a text query, builds one
``HumanMessage`` carrying both, and streams the model's answer directly — no
tools, no checkpointer, no LangGraph graph. The image is normalized into a
model-ready content part through the ``tai_app.storage.resource_manager`` handle
(``normalize_media``), so a stored image resolves to a base64 data-URI while a
public URL passes through untouched; model access goes through tai-kit's llm
factory. This is PRE-run normalization — the agent has no tool loop.

Two faces, both derived from the one class:

* :meth:`astream` — the in-process streaming face. It drives the model's own
  ``astream`` by hand and projects it into the contract ``StreamEvent``
  vocabulary: :class:`MessageDelta` per token chunk, a :class:`RunUsage` when
  the provider surfaced token counts, then one assembled :class:`MessageFinal`.
  Usage precedes the terminal final so a consumer that stops at the first
  ``final=True`` event has still seen the run's token counts — matching the
  generic tools-agent projection's ordering.
* :meth:`run` — the JSON tool-face. It drains its own :meth:`astream` via
  :meth:`Agent._drain` and returns the final answer text.

PARTIAL EVENT TAXONOMY
----------------------
A single multimodal completion produces, in the text case, exactly three kinds
of event: :class:`MessageDelta`, :class:`MessageFinal`, and :class:`RunUsage`;
in the forced-structured case (``response_format`` set), exactly one
:class:`StructuredFinal`. There are no :class:`ReasoningStep`, tool-call, or
interrupt events because the agent runs no reasoning loop, invokes no tools, and
never pauses a graph. This narrow set is the complete taxonomy for what the
agent does, not a truncated one — the missing kinds have no source here.
:class:`RunUsage` itself is emitted only when the provider reports usage; an
honest omission simply yields fewer events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field
from tai_contract.agent import Agent
from tai_contract.agent.events import MessageDelta, MessageFinal, StreamEvent, StructuredFinal
from tai_contract.app import tai_app
from tai_kit.llm.models import get_llm_async
from tai_kit.llm.runtime import validate_structured_output
from tai_kit.llm.settings import llm_provider_settings, llm_settings

from tai_agents._internal.reject import reject_unhonored, reject_untitled_response_format
from tai_agents._internal.text import text_of
from tai_agents._internal.usage import usage_event


async def _vqa_message(image_url: str, query: str) -> HumanMessage:
    """One multimodal ``HumanMessage`` carrying the text query and the image.

    ``image_url`` is normalized into a model-ready image content part via the
    ``resource_manager`` handle: a public URL passes through unchanged, a storage
    id resolves to a base64 data-URI. ``normalize_media`` is async, so this is
    awaited before the model call.
    """
    image_part = await tai_app.storage.resource_manager.normalize_media(image_url)
    return HumanMessage(content=[{"type": "text", "text": query}, image_part])


class VqaAgentInput(BaseModel):
    """JSON tool-face parameters for ``vqa_agent``.

    ``image_url`` accepts either a public image URL or a storage id — both are a
    plain ``str``, resolved by ``ResourceManager.normalize_media``.

    ``response_format`` is a JSON-Schema dict with a required top-level ``"title"``
    (used as the structured-output name); when set, the completion forces the model
    to emit output matching it and :meth:`run` returns the validated structured
    object.

    ``base_url``/``api_key`` in ``llm_kwargs`` legitimately route to a caller-chosen
    model endpoint; expose any agent or tool carrying these kwargs only to trusted
    callers — an injected parent agent could redirect the model call to a hostile
    endpoint and leak the key/context.

    ``extra="forbid"`` rejects any unknown key loudly at validation rather than
    letting a typo at the run door vanish silently.
    """

    model_config = ConfigDict(extra="forbid")

    image_url: str
    query: str
    response_format: dict[str, Any] | None = Field(
        default=None, description="JSON Schema of the forced structured output (needs a top-level 'title')."
    )
    llm_provider: str | None = None
    llm_kwargs: dict[str, Any] | None = None


# Contract ``Agent.run`` parameters ``vqa_agent`` cannot honor, mapped to the
# reason named in the raised error (the keys also define this agent's unhonored
# set). A single multimodal completion has no tool loop, no LangGraph graph, and no
# checkpointer, so the contract parameters with a seam here are ``llm_provider`` and
# ``llm_kwargs`` (they reach the llm factory) plus ``response_format`` (forced via
# ``with_structured_output``). Every other ABC-named parameter is rejected loudly on
# both faces rather than silently dropped. Plain extension ``**kwargs`` beyond the
# ABC names stay allowed.
_UNHONORED_REASONS: dict[str, str] = {
    "tools": "it runs a single multimodal completion and loads no tools",
    "tool_names": "it runs a single multimodal completion and loads no tools",
    "presets": "it runs a single multimodal completion and exposes no preset tools",
    "subagents": "it runs a single multimodal completion with no sub-agent machinery",
    "system_message": "it builds its own prompt from image_url and query and takes no system_message",
    "user_message": "it builds its own prompt from image_url and query and takes no user_message",
    "strategy": "it applies no composition strategy",
    "interrupt_on": "it runs no graph and never pauses for external input",
    "skills": "it loads no skills backend",
    "inline_skills": "it loads no skills backend",
    "recursion_limit": "it runs no graph, so there is no step count to bound",
    "thread_id": "it runs no checkpointer, so there is no thread to pin",
    "resume": "it runs no graph and never interrupts, so there is no paused run to resume",
    "resume_checkpoint_id": "it runs no checkpointer, so there is no checkpoint to fork from",
    "checkpoint_provider": "it runs no checkpointer",
    "store_provider": "it wires no long-term store",
}
# The unhonored parameters whose unset default is an empty sequence or empty string
# — a truthy value is a rejected request. Every other unhonored parameter defaults
# to ``None`` and is set when it is not ``None`` (so ``response_format=None`` passes
# but a falsy-but-meaningful ``strategy=""`` or ``resume=False`` still raises).
_UNHONORED_COLLECTION_PARAMS: frozenset[str] = frozenset(
    {"tools", "tool_names", "presets", "subagents", "skills", "inline_skills", "system_message", "user_message"}
)


@tai_app.agents.agent("vqa_agent")
class VqaAgent(Agent):
    tool_name: ClassVar[str] = "vqa_agent"
    tool_description: ClassVar[str] = "Analyze an image and answer a query about it using a multimodal LLM."
    ToolInput: ClassVar[type[BaseModel]] = VqaAgentInput

    async def run(
        self,
        *,
        image_url: str,
        query: str,
        response_format: Any = None,
        llm_provider: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Drain :meth:`astream` and return the final value.

        The agent honors ``image_url``/``query`` (the multimodal prompt),
        ``response_format`` (the forced structured output), and ``llm_provider``/
        ``llm_kwargs`` (the model selection). A ``response_format`` JSON-Schema dict
        must carry a top-level ``"title"`` (the structured-output name); when set,
        the validated structured object is returned, and a run that produces none
        raises loudly. Every other contract :meth:`Agent.run` parameter —
        ``tools``/``thread_id``/… (no tool loop, graph, or checkpointer) — is
        rejected loudly rather than silently dropped, in parity with
        :meth:`astream`.
        """
        reject_unhonored("vqa_agent.run", kwargs, _UNHONORED_REASONS, collection_params=_UNHONORED_COLLECTION_PARAMS)
        reject_untitled_response_format("vqa_agent", response_format)
        return await self._drain(
            self.astream(
                image_url=image_url,
                query=query,
                response_format=response_format,
                llm_provider=llm_provider,
                llm_kwargs=llm_kwargs,
            ),
            response_format=response_format,
        )

    async def astream(
        self,
        *,
        image_url: str,
        query: str,
        response_format: Any = None,
        llm_provider: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one multimodal completion.

        Without a ``response_format`` the model's answer streams as token
        :class:`MessageDelta`s, then — when the provider reports it — the run's
        usage, then the assembled final :class:`MessageFinal`. Usage is yielded
        before the terminal final so a consumer that stops at the first
        ``final=True`` event has still observed the token counts, matching the
        generic tools-agent projection. The model chunks are accumulated so the
        usage totals are summed across every chunk (a provider may split token
        counts over the stream); the model label and counts are read from that
        accumulated message.

        With a ``response_format`` the completion instead forces the structured
        output via ``with_structured_output`` and emits exactly one
        :class:`StructuredFinal` (no token deltas) carrying the validated object.

        Contract parameters this agent cannot honor (``tools``, ``thread_id``, …)
        are rejected loudly here too — in parity with :meth:`run` — rather than
        silently dropped.
        """
        reject_unhonored(
            "vqa_agent.astream", kwargs, _UNHONORED_REASONS, collection_params=_UNHONORED_COLLECTION_PARAMS
        )
        reject_untitled_response_format("vqa_agent", response_format)
        provider = llm_provider or llm_provider_settings().llm
        llm = await get_llm_async(provider=provider, **llm_settings().with_fallbacks(llm_kwargs or {}))

        message = await _vqa_message(image_url, query)

        if response_format is not None:
            structured = await llm.with_structured_output(response_format, include_raw=False).ainvoke([message])
            yield StructuredFinal(data=validate_structured_output(structured, response_format))
            return

        parts: list[str] = []
        accumulated: Any = None
        async for chunk in llm.astream([message]):
            accumulated = chunk if accumulated is None else accumulated + chunk
            text = text_of(chunk)
            if text:
                parts.append(text)
                yield MessageDelta(text=text)

        if accumulated is not None:
            usage = usage_event(accumulated)
            if usage is not None:
                yield usage
        final = "".join(parts).strip()
        if final:
            yield MessageFinal(text=final)
