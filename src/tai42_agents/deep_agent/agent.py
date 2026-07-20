"""``deep_agent`` as an :class:`Agent`.

Two faces, both built from the one shared streaming core (:meth:`DeepAgent._astream_built`):

* :meth:`DeepAgent.run` — the JSON tool-face. Renders the messages, resolves
  tool names + JSON subagents into live tools + core specs, then DRAINS the same
  streaming path via :meth:`Agent._drain`, returning the structured object or
  answer string. Draining honors the interrupt terminal rule: a run that pauses
  on an interrupt raises :class:`~tai42_contract.agent.base.AgentInterruptedError`
  rather than returning pre-interrupt text.
* :meth:`DeepAgent.astream` — the in-process streaming face the API drives with
  live tool closures: it builds the compiled graph from the shared registries,
  projects its events, and surfaces each pending platform interrupt as an
  :class:`InterruptFinal` after the stream for the leg drains.

Both faces treat a requested ``response_format`` identically: the invoke face
returns the structured value (or raises via ``_drain`` when none was produced),
and the streaming face emits the :class:`StructuredFinal` frame (or raises the
same missing-structured-output error after the stream drains). Neither silently
omits it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar

from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from tai42_contract.agent import Agent
from tai42_contract.agent.base import PresetSpec
from tai42_contract.agent.base import SubAgentSpec as NeutralSubAgentSpec
from tai42_contract.agent.events import InterruptFinal, StreamEvent, StructuredFinal
from tai42_contract.app import tai42_app
from tai42_kit.llm.checkpoint.checkpoint_registry import checkpoint_registry
from tai42_kit.llm.models import get_llm_async
from tai42_kit.llm.runtime import build_agent_input
from tai42_kit.llm.settings import llm_provider_settings, llm_settings
from tai42_kit.llm.store.store_registry import store_registry

from tai42_agents._internal.config_util import build_run_config, init_langgraph_config
from tai42_agents._internal.reject import (
    reject_blank_memory_keys,
    reject_unhonored,
    reject_untitled_response_format,
)
from tai42_agents._internal.render import render_message
from tai42_agents._internal.resolve_tools import resolve_tools
from tai42_agents._internal.stream_events import aproject_agent_events
from tai42_agents.deep_agent.factory import build_deep_agent
from tai42_agents.deep_agent.spec import InlineSkill, ResolvedSubAgentSpec
from tai42_agents.deep_agent.tool_spec import DeepSubAgentSpec, resolve_subagent_specs

# The two ABC ``run``/``astream`` parameters ``deep_agent``'s runtime cannot honor
# on the main agent, mapped to the reason named in the raised error (the keys also
# define this agent's unhonored set). Every other composable field it honors.
# ``presets`` is truthiness-checked (an empty value is a no-op), ``strategy`` is
# set whenever it is not ``None``. Both faces reject through the shared
# :func:`reject_unhonored` guard, so a caller passing both is named both at once.
_UNHONORED_REASONS: dict[str, str] = {
    "presets": (
        "its tool set is composed from tool_names and live tools on the main agent, not presets, "
        "and it will not silently ignore one"
    ),
    "strategy": "the deepagents runtime applies no composition strategy and will not silently ignore one",
}
_UNHONORED_COLLECTION_PARAMS: frozenset[str] = frozenset({"presets"})


class DeepAgentInput(BaseModel):
    """JSON tool-face parameters for ``deep_agent``. Live ``tools=`` are absent
    from this JSON schema (a live ``StructuredTool`` is not JSON-serializable), but
    both in-process faces — :meth:`DeepAgent.run` and :meth:`DeepAgent.astream` —
    accept them directly.

    The schema advertises exactly the composable fields ``deep_agent``'s runtime
    honors — ``subagents``, ``skills``, ``inline_skills``, ``interrupt_on``,
    ``response_format`` alongside the ``tool_names`` / message / provider plumbing.
    It carries no ``strategy`` field: the deepagents runtime has no composition
    strategy to apply (its sub-agent path rejects a per-sub ``strategy`` outright),
    so advertising one would be a schema lie. ``extra="forbid"`` rejects any
    unknown key loudly at validation rather than letting a typo at the run door
    vanish silently.

    ``base_url``/``api_key`` in ``llm_kwargs`` legitimately route to a caller-chosen
    model endpoint; expose any agent or tool carrying these kwargs only to trusted
    callers — an injected parent agent could redirect the model call to a hostile
    endpoint and leak the key/context.
    """

    model_config = ConfigDict(extra="forbid")

    tool_names: list[str] = Field(default_factory=list, description="Client tool names to load.")
    subagents: list[DeepSubAgentSpec] | None = Field(
        default=None, description="Subagents the main agent can invoke via its task tool."
    )
    skills: list[str] | None = Field(default=None, description="Skill source paths under SKILLS_ROOT.")
    inline_skills: list[InlineSkill] | None = Field(
        default=None, description="Skills supplied inline (name + SKILL.md content)."
    )
    system_message: str | None = ""
    user_message: str | None = ""
    system_message_id: str | None = ""
    user_message_id: str | None = ""
    system_message_kwargs: dict[str, Any] | None = None
    user_message_kwargs: dict[str, Any] | None = None
    interrupt_on: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = Field(
        default=None, description="JSON Schema of the forced structured output (needs a top-level 'title')."
    )
    llm_provider: str | None = None
    checkpoint_provider: str | None = None
    store_provider: str | None = None
    llm_kwargs: dict[str, Any] | None = None
    langgraph_config: dict[str, Any] | None = None


async def _to_internal(spec: NeutralSubAgentSpec | DeepSubAgentSpec) -> ResolvedSubAgentSpec:
    """Resolve either subagent shape into the core spec.

    The two faces are handed different shapes for the same thing: a programmatic
    caller passes the contract's :class:`NeutralSubAgentSpec` (live tools, a
    strategy), while the JSON tool/run door validates into :class:`DeepSubAgentSpec`
    (tool NAMES, no strategy). Both faces accept both, so a spec that works over one
    door works over the other.
    """
    if isinstance(spec, DeepSubAgentSpec):
        return (await resolve_subagent_specs([spec]))[0]
    return await _neutral_to_internal(spec)


async def _neutral_to_internal(spec: NeutralSubAgentSpec) -> ResolvedSubAgentSpec:
    """Map a neutral (live-tools) sub-agent spec to the internal deepagents spec.

    Resolves the neutral spec's ``tools`` / ``tool_names`` / ``presets`` into a
    flat ``StructuredTool`` list; defaults the internal-only fields the neutral
    shape intentionally drops. The internal spec has no ``strategy`` field, so a
    neutral ``strategy`` cannot be honored — it is rejected rather than dropped
    silently (the only live producer, the mcp-finder, never sets it).
    """
    if spec.strategy is not None:
        raise ValueError(
            f"sub-agent {spec.name!r} sets strategy={spec.strategy!r}, which the "
            "deepagents sub-agent spec cannot carry; pass response_format as a "
            "ToolStrategy on the parent instead."
        )
    tools = await resolve_tools(tai42_app.tools, list(spec.tool_names), list(spec.tools), list(spec.presets))
    subagents = [await _neutral_to_internal(child) for child in spec.subagents]
    # Neutral inline_skills are plain dicts (the API can't reference InlineSkill);
    # coerce to InlineSkill instances the factory needs.
    reject_untitled_response_format(f"subagent {spec.name!r}", spec.response_format)
    inline_skills = [s if isinstance(s, InlineSkill) else InlineSkill(**s) for s in (spec.inline_skills or [])]
    return ResolvedSubAgentSpec(
        name=spec.name,
        description=spec.description,
        system_prompt=spec.system_prompt,
        tools=tools,
        skills=list(spec.skills) or None,
        inline_skills=inline_skills or None,
        response_format=spec.response_format,
        subagents=subagents,
    )


@tai42_app.agents.agent("deep_agent")
class DeepAgent(Agent):
    tool_name: ClassVar[str] = "deep_agent"
    tool_description: ClassVar[str] = (
        "Create and run a deep agent (planning + subagents + skills + filesystem). "
        "Loads tools by name and runs with the given system/user messages. With "
        "response_format set, returns a validated structured object and fails loudly "
        "if the agent produces none."
    )
    ToolInput: ClassVar[type[BaseModel]] = DeepAgentInput

    async def run(
        self,
        *,
        tools: Sequence[StructuredTool] = (),
        tool_names: Sequence[str] = (),
        presets: Sequence[PresetSpec] | None = None,
        subagents: list[DeepSubAgentSpec] | None = None,
        skills: list[str] | None = None,
        inline_skills: list[InlineSkill] | None = None,
        system_message: str = "",
        user_message: str = "",
        system_message_id: str = "",
        user_message_id: str = "",
        system_message_kwargs: dict[str, Any] | None = None,
        user_message_kwargs: dict[str, Any] | None = None,
        interrupt_on: dict[str, Any] | None = None,
        response_format: Any = None,
        strategy: str | None = None,
        thread_id: str | None = None,
        resume: Any = None,
        resume_checkpoint_id: str | None = None,
        recursion_limit: int | None = None,
        llm_provider: str | None = None,
        checkpoint_provider: str | None = None,
        store_provider: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        langgraph_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        """Resolve the JSON tool inputs and drain the streaming core to a value.

        Live ``tools`` are combined with the client tools resolved from
        ``tool_names``, and the JSON ``DeepSubAgentSpec`` subagents resolve to core
        ``ResolvedSubAgentSpec`` objects, then the messages are rendered and the
        same streaming core the API drives is drained via :meth:`Agent._drain` (the
        interrupt terminal rule): the structured object (``response_format`` set) or
        the answer string is returned, and a run that pauses on an interrupt raises
        :class:`~tai42_contract.agent.base.AgentInterruptedError` rather than
        returning pre-interrupt text.

        Provide exactly one of ``user_message`` (a fresh turn) or ``resume`` (the
        payload answering a prior interrupt); ``resume_checkpoint_id`` forks past
        an aborted turn. A ``response_format`` must be a JSON Schema with a
        top-level ``title``; when set, a missing structured result raises loudly.

        ``presets`` and ``strategy`` are not part of ``deep_agent``'s composable
        inputs; either one raises rather than being silently ignored, and a caller
        passing both is named both at once.
        """
        reject_unhonored(
            "deep_agent.run",
            {"presets": presets, "strategy": strategy},
            _UNHONORED_REASONS,
            collection_params=_UNHONORED_COLLECTION_PARAMS,
        )
        reject_blank_memory_keys("deep_agent.run", thread_id=thread_id, resume_checkpoint_id=resume_checkpoint_id)
        reject_untitled_response_format("deep_agent", response_format)

        if resume is None and not (user_message or user_message_id):
            raise ValueError("deep_agent.run requires exactly one of user_message or resume")
        if resume is not None:
            if user_message or user_message_id:
                raise ValueError("deep_agent.run requires exactly one of user_message or resume, not both.")
            rendered_user: str | None = None
        else:
            rendered_user = await render_message(user_message, user_message_id, user_message_kwargs, allow_empty=False)
        rendered_system = await render_message(system_message, system_message_id, system_message_kwargs)

        client_tools = await tai42_app.tools.get_client_tools(list(tool_names)) if tool_names else []
        resolved_tools = [*tools, *client_tools]
        internal_subagents = await resolve_subagent_specs(subagents)
        coerced_inline_skills = [s if isinstance(s, InlineSkill) else InlineSkill(**s) for s in (inline_skills or [])]

        agent = await self._resolve_and_build(
            tools=resolved_tools,
            subagents=internal_subagents,
            skills=skills,
            inline_skills=coerced_inline_skills or None,
            system_message=rendered_system,
            response_format=response_format,
            interrupt_on=interrupt_on,
            llm_provider=llm_provider,
            checkpoint_provider=checkpoint_provider,
            store_provider=store_provider,
            llm_kwargs=llm_kwargs,
        )
        config = self._run_config(langgraph_config, thread_id, resume_checkpoint_id, recursion_limit)
        if resume is not None:
            agent_input: Any = Command(resume=resume)
        else:
            # The resume/user guard above makes rendered_user a str on this branch.
            assert rendered_user is not None
            agent_input = build_agent_input(rendered_user)

        return await self._drain(
            self._astream_built(agent, agent_input, config, interrupt_on, response_format=response_format),
            response_format=response_format,
        )

    @staticmethod
    def _run_config(
        langgraph_config: dict[str, Any] | None,
        thread_id: str | None,
        resume_checkpoint_id: str | None,
        recursion_limit: int | None,
    ) -> dict[str, Any]:
        """Build the run config both faces run the graph with.

        The caller's ``langgraph_config`` is the base — carried through by value,
        never mutated; an explicit ``thread_id`` and ``resume_checkpoint_id``
        overlay its ``configurable`` section and ``recursion_limit`` overlays the
        top level, through the shared
        :func:`~tai42_agents._internal.config_util.build_run_config` helper. With no
        thread pinned (neither argument nor a ``configurable.thread_id`` in
        ``langgraph_config``), :func:`init_langgraph_config` mints a fresh isolated
        one, so keyless runs never collide on a shared checkpoint thread.

        ``recursion_limit`` bounds the TOP-LEVEL graph ONLY. Each subagent spawned
        via the task tool runs its own graph with its own recursion limit (the
        LangGraph default of 25 unless the subagent sets one), so the effective
        step budget is MULTIPLICATIVE across nesting depth — the top-level cap is
        not a total-spend ceiling. Threading a shared total step/token budget down
        to subagents (a budget contract across the deepagents subagent boundary) is
        deliberately NOT implemented here; it is a future user decision.
        """
        return init_langgraph_config(
            config=build_run_config(langgraph_config, thread_id, resume_checkpoint_id, recursion_limit)
        )

    async def astream(
        self,
        *,
        tools: Sequence[StructuredTool] = (),
        tool_names: Sequence[str] = (),
        presets: Sequence[PresetSpec] | None = None,
        subagents: Sequence[NeutralSubAgentSpec | DeepSubAgentSpec] | None = None,
        skills: list[str] | None = None,
        inline_skills: Sequence[dict[str, Any]] | None = None,
        system_message: str = "",
        user_message: str | None = None,
        response_format: Any = None,
        strategy: str | None = None,
        interrupt_on: dict[str, Any] | None = None,
        thread_id: str | None = None,
        resume: Any = None,
        resume_checkpoint_id: str | None = None,
        llm_provider: str | None = None,
        checkpoint_provider: str | None = None,
        store_provider: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        recursion_limit: int | None = None,
        langgraph_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn and yield the platform event stream, then one
        :class:`InterruptFinal` per pending interrupt.

        Provide exactly one of ``user_message`` (a fresh turn) or ``resume`` (a
        payload answering a prior interrupt). The system prompt is passed
        verbatim (no template rendering); the API resolved it already. Live
        ``tools`` are combined with the client tools resolved from ``tool_names``
        to form the agent's tool set.

        ``langgraph_config`` is the base run config, honored here in parity with
        :meth:`run` (both faces build through :meth:`_run_config`): a
        ``configurable.thread_id`` or ``checkpoint_id`` it carries directly pins the
        run's checkpointed memory, and an explicit ``thread_id`` /
        ``resume_checkpoint_id`` / ``recursion_limit`` overlays it.

        A requested ``response_format`` that produces no structured result raises
        the same missing-structured-output error the invoke face raises via
        :meth:`Agent._drain`, after the stream drains — never a silent omission.

        ``presets`` and ``strategy`` are not part of ``deep_agent``'s composable
        inputs; either one raises rather than being silently ignored, and a caller
        passing both is named both at once.
        """
        reject_unhonored(
            "deep_agent.astream",
            {"presets": presets, "strategy": strategy},
            _UNHONORED_REASONS,
            collection_params=_UNHONORED_COLLECTION_PARAMS,
        )
        reject_blank_memory_keys("deep_agent.astream", thread_id=thread_id, resume_checkpoint_id=resume_checkpoint_id)
        reject_untitled_response_format("deep_agent", response_format)
        if (user_message is None) == (resume is None):
            raise ValueError("deep_agent.astream requires exactly one of user_message or resume")

        client_tools = await tai42_app.tools.get_client_tools(list(tool_names)) if tool_names else []
        internal_subagents = [await _to_internal(spec) for spec in (subagents or [])]
        coerced_inline_skills = [s if isinstance(s, InlineSkill) else InlineSkill(**s) for s in (inline_skills or [])]
        agent, config = await self._build_agent(
            tools=[*tools, *client_tools],
            subagents=internal_subagents,
            skills=skills,
            inline_skills=coerced_inline_skills or None,
            system_message=system_message,
            response_format=response_format,
            interrupt_on=interrupt_on,
            thread_id=thread_id,
            resume_checkpoint_id=resume_checkpoint_id,
            llm_provider=llm_provider,
            checkpoint_provider=checkpoint_provider,
            store_provider=store_provider,
            llm_kwargs=llm_kwargs,
            recursion_limit=recursion_limit,
            langgraph_config=langgraph_config,
        )
        if resume is not None:
            agent_input: Any = Command(resume=resume)
        else:
            # The exactly-one-of guard above makes user_message non-None here.
            assert user_message is not None
            agent_input = build_agent_input(user_message)

        saw_structured = False
        saw_interrupt = False
        async for event in self._astream_built(
            agent, agent_input, config, interrupt_on, response_format=response_format
        ):
            if isinstance(event, StructuredFinal):
                saw_structured = True
            elif isinstance(event, InterruptFinal):
                saw_interrupt = True
            yield event
        # Structured-final parity with the invoke face (Agent._drain): a requested
        # response_format that produced no StructuredFinal fails loudly rather than
        # silently omitting the frame. A pending interrupt means the run paused
        # rather than finishing — as in _drain, where an interrupt takes precedence
        # over the missing-structured raise — so it is not raised in that case.
        if response_format is not None and not saw_structured and not saw_interrupt:
            raise RuntimeError("agent run requested a response_format but produced no structured output")

    async def _astream_built(
        self,
        agent: Any,
        agent_input: Any,
        config: dict[str, Any],
        interrupt_on: dict[str, Any] | None,
        response_format: Any = None,
    ) -> AsyncIterator[StreamEvent]:
        """Project a built agent's run into contract events, then one
        :class:`InterruptFinal` per pending interrupt.

        The shared streaming core behind both faces: :meth:`astream` (live-tools)
        yields from it, and :meth:`run` (JSON tool-face) drains it. A graph can
        only pause when ``interrupt_on`` is configured, so the extra paused-state
        read is skipped otherwise. ``response_format`` is handed to the projection
        so the structured terminal is validated against it.
        """
        async for event in aproject_agent_events(agent, agent_input, config, response_format=response_format):
            yield event
        if interrupt_on:
            for interrupt in await self._pending_interrupts(agent, config):
                yield interrupt

    async def _resolve_and_build(
        self,
        *,
        tools: list[StructuredTool],
        subagents: list[ResolvedSubAgentSpec],
        skills: list[str] | None,
        inline_skills: list[InlineSkill] | None,
        system_message: str,
        response_format: Any,
        interrupt_on: dict[str, Any] | None,
        llm_provider: str | None,
        checkpoint_provider: str | None,
        store_provider: str | None,
        llm_kwargs: dict[str, Any] | None,
    ) -> Any:
        """Resolve the LLM / checkpointer / store from the registries and assemble
        the compiled deep agent. The caller builds the run config separately."""
        provider = llm_provider or llm_provider_settings().llm
        llm = await get_llm_async(provider=provider, **llm_settings().with_fallbacks(llm_kwargs or {}))

        cp_provider = checkpoint_provider or llm_provider_settings().checkpoint
        checkpointer = await checkpoint_registry().get_checkpointer(
            provider=cp_provider, conn_string=llm_provider_settings().checkpoint_conn_string
        )
        st_provider = store_provider or llm_provider_settings().store
        store = await store_registry().get_store(
            provider=st_provider, conn_string=llm_provider_settings().store_conn_string
        )

        return await build_deep_agent(
            llm=llm,
            store=store,
            checkpointer=checkpointer,
            tools=tools,
            skills=skills or None,
            inline_skills=inline_skills or None,
            system_prompt=system_message or None,
            interrupt_on=interrupt_on,
            response_format=response_format,
            subagents=subagents or None,
        )

    async def _build_agent(
        self,
        *,
        tools: list[StructuredTool],
        subagents: list[ResolvedSubAgentSpec],
        skills: list[str] | None,
        inline_skills: list[InlineSkill] | None,
        system_message: str,
        response_format: Any,
        interrupt_on: dict[str, Any] | None,
        thread_id: str | None,
        resume_checkpoint_id: str | None,
        llm_provider: str | None,
        checkpoint_provider: str | None,
        store_provider: str | None,
        llm_kwargs: dict[str, Any] | None,
        recursion_limit: int | None,
        langgraph_config: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Assemble the compiled deep agent and its run config for the streaming face.

        Wraps :meth:`_resolve_and_build` with the run config :meth:`_run_config`
        builds — the SAME one the invoke face runs with, so the caller's
        ``langgraph_config`` base, a pinned ``thread_id`` / resume checkpoint, and a
        recursion cap are honored identically on both faces. A keyless run gets a
        freshly minted thread, so one-shot streams never collide on a shared
        checkpoint thread.

        The recursion cap bounds the TOP-LEVEL graph ONLY: each subagent spawned via
        the task tool runs its own graph with its own recursion limit (the LangGraph
        default of 25 unless the subagent sets one), so the effective step budget is
        MULTIPLICATIVE across nesting depth, not a total-spend ceiling. Threading a
        shared total budget down to subagents is deliberately NOT implemented here;
        it is a future user decision.
        """
        agent = await self._resolve_and_build(
            tools=tools,
            subagents=subagents,
            skills=skills,
            inline_skills=inline_skills,
            system_message=system_message,
            response_format=response_format,
            interrupt_on=interrupt_on,
            llm_provider=llm_provider,
            checkpoint_provider=checkpoint_provider,
            store_provider=store_provider,
            llm_kwargs=llm_kwargs,
        )
        config = self._run_config(langgraph_config, thread_id, resume_checkpoint_id, recursion_limit)
        return agent, config

    @staticmethod
    async def _pending_interrupts(agent: Any, config: dict[str, Any]) -> list[InterruptFinal]:
        """Read the interrupts a paused graph is waiting on, if any.

        An empty list means the run completed normally. A failure to read the
        snapshot propagates — a paused run whose interrupt we cannot read would
        otherwise hang invisibly.
        """
        snapshot = await agent.aget_state(config)
        interrupts = list(getattr(snapshot, "interrupts", None) or [])
        return [InterruptFinal(interrupt_id=intr.id, payload=intr.value) for intr in interrupts]
