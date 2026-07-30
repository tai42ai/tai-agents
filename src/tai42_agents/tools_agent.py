"""``tools_agent`` as an :class:`Agent`.

The plain/advanced LangGraph tools agent. Its three tool inputs are uniform —
``tool_names`` (client tools resolved through the app registry), live ``tools``,
and ``presets`` (a base tool bound to fixed kwargs, exposed as a callable tool) —
and are resolved together by :func:`resolve_tools`. A sub-flow is just a preset:
``base_tool="flow"`` with ``fixed_kwargs={"flow_graph": ...}``.

Two faces:

* :meth:`run` — the JSON tool-face. Resolves ``tool_names`` + ``presets`` through
  the app's tool facet, renders the system/user messages through the template
  manager, invokes the agent once, and returns the final answer as a string.
* :meth:`astream` — the in-process streaming face the API drives with live
  ``tools``. It projects the run into the contract ``StreamEvent`` taxonomy:
  :class:`ReasoningStep`, :class:`ToolCallStep` / :class:`ToolResultStep` pairs,
  :class:`MessageDelta`, :class:`MessageFinal`, :class:`RunUsage`, and — when the
  underlying run carries a structured response — :class:`StructuredFinal`. A
  requested ``response_format`` the run produces no structured response for
  raises after the stream drains rather than silently omitting the frame.

This agent has no interrupt source: neither face pauses the graph for external
input, so the stream never contains an ``InterruptFinal``. Its absence is by
design, not a partial run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field
from tai42_contract.agent import Agent
from tai42_contract.agent.base import PresetSpec, SubAgentSpec
from tai42_contract.agent.events import StreamEvent, StructuredFinal
from tai42_contract.app import tai42_app

from tai42_agents._internal.base_tool_agent import ainvoke_tools_agent
from tai42_agents._internal.config_util import build_run_config
from tai42_agents._internal.reject import (
    reject_blank_memory_keys,
    reject_unhonored,
    reject_untitled_response_format,
)
from tai42_agents._internal.render import render_message
from tai42_agents._internal.resolve_tools import resolve_tools
from tai42_agents._internal.stream_events import astream_tools_agent_events

# ABC ``run`` parameters this agent's runtime cannot honor, mapped to the reason
# named in the raised error (the keys also define this agent's unhonored set).
# ``tools_agent`` runs a single LangGraph tools agent: no sub-agents, no
# composition strategy, no skill backend, no interrupt source, and no long-term
# store. ``response_format`` is NOT here — it forces the run's structured output
# through ``create_agent`` and is honored on both faces. ``recursion_limit`` is
# NOT here either — it is a standard ``RunnableConfig`` key the compiled graph
# reads, so it is honored (overlaid onto the run config) rather than rejected.
_UNHONORED_REASONS: dict[str, str] = {
    "subagents": "sub-agent delegation is deep_agent's domain; this agent never exposes sub-agents as callable tools",
    "strategy": "it applies no composition strategy and will not silently ignore one",
    "skills": "skill backends are deep_agent's domain; this agent loads none",
    "inline_skills": "skill backends are deep_agent's domain; this agent loads none",
    "interrupt_on": "this agent has no interrupt source and never pauses the graph for external input",
    "resume": "this agent has no interrupt source, so there is no paused turn to resume",
    "store_provider": "this agent wires no long-term store",
}

# The unhonored parameters whose unset default is an empty sequence — a non-empty
# value is set. Every other unhonored parameter defaults to ``None`` and is set
# when not ``None`` (so a falsy-but-present value like ``resume=False`` still raises).
_UNHONORED_COLLECTION_PARAMS = frozenset({"subagents", "skills", "inline_skills"})


class ToolsAgentInput(BaseModel):
    """JSON tool-face parameters for ``tools_agent``.

    Live ``tools`` are absent from this JSON schema — they carry live objects a
    JSON caller cannot express. Both in-process faces (:meth:`ToolsAgent.run` and
    :meth:`ToolsAgent.astream`) accept them directly. ``presets`` bind base tools
    to fixed kwargs and are exposed to the model as callable tools (e.g. a flow
    graph).

    The schema advertises exactly the composable spec fields this agent's runtime
    honors — ``system_prompt``, ``tool_names``, ``presets``, ``response_format`` —
    so an authored agent (``ToolsAgent.spec_runnable = True``) bakes only fields
    this agent actually applies. The composable fields the runtime cannot honor
    (``subagents``, ``strategy``) are absent from the schema, and
    ``extra="forbid"`` rejects them — and any other unknown key — loudly at
    validation rather than silently ignoring them. :meth:`ToolsAgent.run` /
    :meth:`ToolsAgent.astream` carry a matching defense-in-depth raise for
    in-process callers that bypass this validation.

    ``response_format`` is a JSON-Schema dict with a required top-level ``"title"``
    (used as the structured-output name); when set, the run forces the model to
    emit output matching it and returns the validated structured object.

    ``base_url``/``api_key`` in ``llm_kwargs`` legitimately route to a caller-chosen
    model endpoint; expose any agent or tool carrying these kwargs only to trusted
    callers — an injected parent agent could redirect the model call to a hostile
    endpoint and leak the key/context.
    """

    model_config = ConfigDict(extra="forbid")

    tool_names: list[str] = Field(default_factory=list, description="Client tool names to load.")
    presets: list[PresetSpec] | None = Field(
        default=None,
        description="Base tools bound to fixed kwargs, exposed as callable tools (e.g. a flow graph).",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt baked into an authored agent (mapped to the system_message run kwarg).",
    )
    response_format: dict[str, Any] | None = Field(
        default=None, description="JSON Schema of the forced structured output (needs a top-level 'title')."
    )
    system_message: str | None = ""
    user_message: str | None = ""
    system_message_id: str | None = ""
    user_message_id: str | None = ""
    system_message_kwargs: dict[str, Any] | None = None
    user_message_kwargs: dict[str, Any] | None = None
    llm_provider: str | None = None
    checkpoint_provider: str | None = None
    llm_kwargs: dict[str, Any] | None = None
    langgraph_config: dict[str, Any] | None = None


@tai42_app.agents.agent("tools_agent", tags={"agents"})
class ToolsAgent(Agent):
    tool_name: ClassVar[str] = "tools_agent"
    tool_description: ClassVar[str] = (
        "Create and run a LangGraph tools agent. Loads tools by name (and optional "
        "presets) and runs with the given system/user messages."
    )
    ToolInput: ClassVar[type[BaseModel]] = ToolsAgentInput
    spec_runnable: ClassVar[bool] = True

    @classmethod
    def from_tool_input(cls, validated: BaseModel) -> dict[str, Any]:
        """Map a validated :class:`ToolsAgentInput` to ``run``/``astream`` kwargs.

        Extends the base pass-through with this agent's one field rename: the
        composable ``system_prompt`` spec field becomes the ``system_message``
        run kwarg. Every other set field passes through unchanged.

        ``system_prompt`` and ``system_message`` both define the system message, so
        supplying a NON-EMPTY ``system_message`` alongside ``system_prompt`` is a
        conflict — it is rejected loudly rather than silently dropping one. (This is
        what stops a run request's ``system_message`` from silently overriding an
        authored agent's baked ``system_prompt``, which maps to the same run kwarg.) An
        empty ``system_message`` — the field default the tool-face wrapper materializes
        — is not a real value and is simply superseded by the mapped ``system_prompt``.
        """
        run_kwargs = super().from_tool_input(validated)
        if "system_prompt" in run_kwargs:
            if run_kwargs.get("system_message"):
                raise ValueError(
                    "set only one of system_prompt or system_message: both define the system "
                    "message (system_prompt is the composable spec field mapped to system_message)"
                )
            run_kwargs["system_message"] = run_kwargs.pop("system_prompt")
        return run_kwargs

    async def run(
        self,
        *,
        tools: Sequence[StructuredTool] = (),
        tool_names: Sequence[str] = (),
        presets: Sequence[PresetSpec] | None = None,
        subagents: Sequence[SubAgentSpec] | None = None,
        strategy: str | None = None,
        response_format: Any = None,
        interrupt_on: dict[str, Any] | None = None,
        skills: Sequence[str] = (),
        inline_skills: Sequence[dict[str, Any]] = (),
        recursion_limit: int | None = None,
        resume: Any = None,
        store_provider: str | None = None,
        system_message: str = "",
        user_message: str = "",
        system_message_id: str = "",
        user_message_id: str = "",
        system_message_kwargs: dict[str, Any] | None = None,
        user_message_kwargs: dict[str, Any] | None = None,
        llm_provider: str | None = None,
        checkpoint_provider: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        thread_id: str | None = None,
        resume_checkpoint_id: str | None = None,
        langgraph_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        """Resolve tools, render the messages, run once, return the final value.

        ``tools`` (live objects), ``tool_names``, and ``presets`` resolve together
        through the app's tool facet — the SAME resolution :meth:`astream` applies;
        the system/user messages render through the template manager (literal
        content or a stored template id); ``thread_id`` (the LangGraph memory key)
        and ``resume_checkpoint_id`` (the checkpoint a run forks from) map into the
        run config's ``configurable``. The resolved tools, rendered messages, and
        config drive a single agent invocation whose result is returned.

        ``response_format`` forces the run's structured output: when set, the model
        is constrained to emit output matching the schema and the validated
        structured object is returned instead of the text — a JSON-Schema dict must
        carry a top-level ``"title"`` (the structured-output name). A requested
        ``response_format`` that the run produces none for raises loudly rather than
        silently falling back to text.

        ``recursion_limit`` (a standard ``RunnableConfig`` key the compiled graph
        reads) overlays the run config's top level, so a caller's step bound reaches
        the graph. The ABC ``run`` parameters this agent's runtime cannot honor are
        rejected loudly rather than silently dropped: ``subagents`` (sub-agent
        delegation is deep_agent's domain — tools_agent never exposes them as
        callable tools), ``strategy`` (no composition strategy is applied),
        ``skills`` / ``inline_skills`` (skill backends are deep_agent's domain),
        ``interrupt_on`` / ``resume`` (this agent has no interrupt source), and
        ``store_provider`` (no long-term store is wired).
        """
        reject_unhonored(
            "tools_agent.run",
            {
                "subagents": subagents,
                "strategy": strategy,
                "interrupt_on": interrupt_on,
                "skills": skills,
                "inline_skills": inline_skills,
                "resume": resume,
                "store_provider": store_provider,
            },
            _UNHONORED_REASONS,
            collection_params=_UNHONORED_COLLECTION_PARAMS,
        )
        reject_blank_memory_keys("tools_agent.run", thread_id=thread_id, resume_checkpoint_id=resume_checkpoint_id)
        reject_untitled_response_format("tools_agent", response_format)
        resolved_tools = await resolve_tools(tai42_app.tools, list(tool_names), list(tools), list(presets or []))
        system_message = await render_message(
            system_message,
            system_message_id,
            system_message_kwargs,
        )
        user_message = await render_message(
            user_message,
            user_message_id,
            user_message_kwargs,
        )
        config = build_run_config(langgraph_config, thread_id, resume_checkpoint_id, recursion_limit)
        result = await ainvoke_tools_agent(
            system_message=system_message,
            user_message=[user_message],
            tools=resolved_tools,
            llm_provider=llm_provider,
            checkpoint_provider=checkpoint_provider,
            llm_kwargs=llm_kwargs,
            config=config,
            response_format=response_format,
        )
        if response_format is not None:
            # ainvoke_tools_agent already validated the structured output and
            # raised on a requested-but-missing one.
            return result.structured
        return result.output

    async def astream(
        self,
        *,
        tools: Sequence[StructuredTool] = (),
        tool_names: Sequence[str] = (),
        presets: Sequence[PresetSpec] | None = None,
        subagents: Sequence[SubAgentSpec] | None = None,
        strategy: str | None = None,
        response_format: Any = None,
        interrupt_on: dict[str, Any] | None = None,
        skills: Sequence[str] = (),
        inline_skills: Sequence[dict[str, Any]] = (),
        recursion_limit: int | None = None,
        resume: Any = None,
        store_provider: str | None = None,
        system_message: str = "",
        user_message: str = "",
        system_message_id: str = "",
        user_message_id: str = "",
        system_message_kwargs: dict[str, Any] | None = None,
        user_message_kwargs: dict[str, Any] | None = None,
        llm_provider: str | None = None,
        checkpoint_provider: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        thread_id: str | None = None,
        resume_checkpoint_id: str | None = None,
        langgraph_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn and yield the contract event stream.

        ``tools`` (live objects), ``tool_names`` (names resolved through the app
        facet), and ``presets`` resolve together into the callable tool set — the
        SAME resolution :meth:`run` applies — so a baked ``tool_names`` / ``presets``
        spec drives the streamed run instead of being dropped. The system/user
        messages render through the template manager — the SAME rendering
        :meth:`run` applies — so a stored template id reaches the streamed run as
        its rendered content. ``thread_id`` is the LangGraph memory key;
        ``resume_checkpoint_id`` forks past an aborted turn (the conversation's last
        completed checkpoint). ``langgraph_config`` is the base run config: a
        ``configurable.thread_id`` (or ``checkpoint_id``) it carries directly is
        honored here — in parity with :meth:`run`, through the one shared
        :func:`~tai42_agents._internal.config_util.build_run_config` helper — so a
        caller pinning a thread over this face resumes its checkpointed memory.
        A keyless run pins no ``thread_id`` at all, so the run config helper mints a
        fresh isolated one per run rather than collapsing every keyless run onto one
        shared checkpoint thread.

        ``response_format`` forces the run's structured output — in parity with
        :meth:`run` — surfaced as a terminal :class:`StructuredFinal`; a JSON-Schema
        dict must carry a top-level ``"title"`` (the structured-output name), and a
        requested ``response_format`` the run produces none for raises loudly after
        the stream drains rather than silently omitting the frame.
        ``recursion_limit`` (a standard ``RunnableConfig`` key the compiled graph
        reads) overlays the run config's top level, in parity with :meth:`run`, so a
        caller's step bound reaches the graph. The ABC parameters this agent's
        runtime cannot honor are rejected loudly here too — in parity with
        :meth:`run` — rather than silently dropped: ``subagents`` (delegation is
        deep_agent's domain), ``strategy`` (no composition strategy is applied),
        ``skills`` / ``inline_skills`` (skill backends are deep_agent's domain),
        ``interrupt_on`` / ``resume`` (this agent has no interrupt source), and
        ``store_provider`` (no long-term store is wired).
        """
        reject_unhonored(
            "tools_agent.astream",
            {
                "subagents": subagents,
                "strategy": strategy,
                "interrupt_on": interrupt_on,
                "skills": skills,
                "inline_skills": inline_skills,
                "resume": resume,
                "store_provider": store_provider,
            },
            _UNHONORED_REASONS,
            collection_params=_UNHONORED_COLLECTION_PARAMS,
        )
        reject_blank_memory_keys("tools_agent.astream", thread_id=thread_id, resume_checkpoint_id=resume_checkpoint_id)
        reject_untitled_response_format("tools_agent", response_format)
        resolved_tools = await resolve_tools(tai42_app.tools, list(tool_names), list(tools), list(presets or []))
        rendered_system = await render_message(system_message, system_message_id, system_message_kwargs)
        rendered_user = await render_message(user_message, user_message_id, user_message_kwargs)
        config = build_run_config(langgraph_config, thread_id, resume_checkpoint_id, recursion_limit)
        saw_structured = False
        async for event in astream_tools_agent_events(
            system_message=rendered_system,
            user_message=[rendered_user],
            tools=resolved_tools,
            llm_provider=llm_provider,
            checkpoint_provider=checkpoint_provider,
            llm_kwargs=llm_kwargs,
            config=config,
            response_format=response_format,
        ):
            if isinstance(event, StructuredFinal):
                saw_structured = True
            yield event
        # Structured-final parity with the invoke face: a requested response_format
        # that produced no StructuredFinal fails loudly after the stream drains
        # rather than silently omitting the frame.
        if response_format is not None and not saw_structured:
            raise RuntimeError("agent run requested a response_format but produced no structured output")
