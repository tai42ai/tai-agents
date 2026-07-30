"""The ``refine_agent`` — an Evaluator↔Critic refinement loop.

The Evaluator drafts, the Critic reviews, the Evaluator revises; the two alternate
until the Critic emits the approval token
(:data:`~tai42_agents.refine_agent.prompt.CRITIC_APPROVAL_MESSAGE`) or the
iteration budget is exhausted — which fails loudly with a ``RuntimeError`` rather
than returning an unapproved draft. An empty critic response also raises.

Only the final, approved evaluator pass streams; the negotiation runs silently in
the prelude (plain ``ainvoke`` per turn). ``run`` drains that same stream.

With a ``response_format`` set, the loop iterations stay text and only the final
approved answer is forced into the schema — run on a second, structured evaluator
fed the negotiation history explicitly on a fresh thread, surfaced as a terminal
:class:`~tai42_contract.agent.events.StructuredFinal`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from langchain.agents import create_agent
from pydantic import BaseModel, ConfigDict, Field
from tai42_contract.agent import Agent
from tai42_contract.agent.events import StreamEvent, StructuredFinal
from tai42_contract.app import tai42_app
from tai42_kit.llm.checkpoint.checkpoint_registry import checkpoint_registry
from tai42_kit.llm.middleware.context_overflow import context_overflow_middlewares
from tai42_kit.llm.models import get_llm_async
from tai42_kit.llm.runtime import build_agent_input, build_user_output
from tai42_kit.llm.settings import llm_provider_settings, llm_settings
from tai42_kit.logging.settings import logging_settings

from tai42_agents._internal.config_util import init_langgraph_config
from tai42_agents._internal.reject import reject_unhonored, reject_untitled_response_format
from tai42_agents._internal.render import render_message
from tai42_agents._internal.stream_events import aproject_agent_events
from tai42_agents.refine_agent.prompt import (
    CRITIC_APPROVAL_MESSAGE,
    CRITIC_SYSTEM_MESSAGE,
    EVALUATOR_SYSTEM_MESSAGE,
)

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool


# Contract ``Agent.run`` parameters the refine loop has no seat for, mapped to the
# reason named in the raised error (the keys define this agent's unhonored set). The
# loop drives its Evaluator↔Critic negotiation through role-named parameters
# (evaluator_/critic_) and resolves tools from ``tool_names``. Honored params
# (``tool_names``, ``checkpoint_provider``, ``response_format``) are absent here.
_UNHONORED_REASONS: dict[str, str] = {
    "tools": "the loop resolves tools from tool_names, not caller-supplied live tools",
    "presets": "the loop resolves tools from tool_names and exposes no preset tools",
    "subagents": "the loop runs no sub-agent machinery",
    "strategy": "it applies no composition strategy and will not silently ignore one",
    "system_message": "the loop uses role-named evaluator_/critic_ messages and takes no system_message",
    "user_message": "the loop uses role-named evaluator_/critic_ messages and takes no user_message",
    "interrupt_on": "its evaluator/critic graphs never pause for external input",
    "skills": "it loads no skills backend",
    "inline_skills": "it loads no skills backend",
    "recursion_limit": (
        "recursion_limit is per-role RunnableConfig content set via evaluator_/critic_ "
        "langgraph_config; a single global value would mask the per-role setting"
    ),
    "thread_id": (
        "thread_id is per-role RunnableConfig content set via evaluator_/critic_ langgraph_config's configurable; "
        "a single global thread_id would collide the two role graphs' checkpoint namespaces"
    ),
    "resume": "its graphs never interrupt, so there is no paused run to resume",
    "resume_checkpoint_id": (
        "resume_checkpoint_id is per-role RunnableConfig content set via evaluator_/critic_ langgraph_config's "
        "configurable; a single global checkpoint would collide the two role graphs' checkpoint namespaces"
    ),
    "llm_provider": "the loop uses role-named evaluator_llm_provider/critic_llm_provider",
    "store_provider": "it wires no long-term store",
    "llm_kwargs": "the loop uses role-named evaluator_llm_kwargs/critic_llm_kwargs",
}
# The unhonored parameters whose unset default is an empty sequence or empty string
# — a truthy value is a rejected request. Every other unhonored parameter defaults
# to ``None`` and is set when it is not ``None`` (so ``response_format=None`` passes
# but a falsy-but-meaningful ``strategy=""`` or ``resume=False`` still raises).
_UNHONORED_COLLECTION_PARAMS: frozenset[str] = frozenset(
    {"tools", "presets", "subagents", "skills", "inline_skills", "system_message", "user_message"}
)


def _final_evaluator_input() -> dict[str, Any]:
    """The prompt that asks the approved Evaluator for its final user-facing
    answer, run against the same checkpointed thread the loop built up."""
    return {"messages": [{"role": "user", "content": "Critic Approved."}]}


async def _run_refine_loop(
    tools: Sequence[StructuredTool],
    evaluator_message: str,
    evaluator_message_id: str,
    critic_message: str,
    critic_message_id: str,
    evaluator_message_kwargs: dict[str, Any] | None,
    critic_message_kwargs: dict[str, Any] | None,
    max_iterations: int,
    evaluator_llm_provider: str | None,
    critic_llm_provider: str | None,
    checkpoint_provider: str | None,
    evaluator_llm_kwargs: dict[str, Any] | None,
    critic_llm_kwargs: dict[str, Any] | None,
    evaluator_config: dict[str, Any] | None,
    critic_config: dict[str, Any] | None,
    response_format: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run the Evaluator↔Critic loop to approval.

    Returns ``(final_agent, final_input, final_config)`` for the final approved
    evaluator pass the caller streams.

    Without a ``response_format`` this is the loop's own evaluator resumed on its
    checkpointed thread with a "Critic Approved." prompt. With a ``response_format``
    set, the final pass runs on a SECOND evaluator built with that schema on a FRESH
    thread, fed the negotiation history explicitly as input — never a checkpoint
    resume of the text-shaped loop thread into the structured graph.

    Raises ``RuntimeError`` when the Critic returns no feedback or ``max_iterations``
    is reached without approval — an unapproved draft is never returned.
    """
    evaluator_message = await render_message(
        evaluator_message, evaluator_message_id, evaluator_message_kwargs, allow_empty=False
    )
    critic_message = await render_message(critic_message, critic_message_id, critic_message_kwargs, allow_empty=False)

    evaluator_llm_provider = evaluator_llm_provider or llm_provider_settings().llm
    critic_llm_provider = critic_llm_provider or llm_provider_settings().llm

    evaluator_llm = await get_llm_async(
        provider=evaluator_llm_provider, **llm_settings().with_fallbacks(evaluator_llm_kwargs or {})
    )
    critic_llm = await get_llm_async(
        provider=critic_llm_provider, **llm_settings().with_fallbacks(critic_llm_kwargs or {})
    )

    checkpoint_provider = checkpoint_provider or llm_provider_settings().checkpoint
    checkpointer = await checkpoint_registry().get_checkpointer(
        provider=checkpoint_provider,
        conn_string=llm_provider_settings().checkpoint_conn_string,
    )
    is_enabled_for_debug = logging_settings().is_enabled_for("DEBUG")
    evaluator: Any = create_agent(
        evaluator_llm,
        tools=list(tools),
        checkpointer=checkpointer,
        middleware=context_overflow_middlewares(),
        debug=is_enabled_for_debug,
    )
    critic: Any = create_agent(
        critic_llm,
        tools=list(tools),
        checkpointer=checkpointer,
        middleware=context_overflow_middlewares(),
        debug=is_enabled_for_debug,
    )

    evaluator_config = init_langgraph_config(evaluator_config)
    critic_config = init_langgraph_config(critic_config)

    iteration = 0
    evaluator_input: dict[str, Any] = build_agent_input(evaluator_message, system_message=EVALUATOR_SYSTEM_MESSAGE)

    while iteration < max_iterations:
        evaluator_state = await evaluator.ainvoke(evaluator_input, evaluator_config)
        evaluator_output = build_user_output(evaluator_state)

        if iteration > 0:
            critic_input: dict[str, Any] = {"messages": [{"role": "user", "content": evaluator_output}]}
        else:
            critic_input = build_agent_input(critic_message, evaluator_output, system_message=CRITIC_SYSTEM_MESSAGE)

        critic_state = await critic.ainvoke(critic_input, critic_config)
        critic_output = build_user_output(critic_state)

        if not critic_output:
            raise RuntimeError("No critic feedback found.")

        if CRITIC_APPROVAL_MESSAGE in critic_output:
            break

        evaluator_input = {"messages": [{"role": "user", "content": critic_output}]}
        iteration += 1
    else:
        raise RuntimeError(f"Max iterations ({max_iterations}) reached without critic approval")

    if response_format is None:
        return evaluator, _final_evaluator_input(), evaluator_config

    # Force the final answer into the schema without re-shaping the loop thread:
    # feed its negotiation history as input to a second, structured evaluator on a
    # fresh thread (never a cross-topology resume).
    snapshot = await evaluator.aget_state(evaluator_config)
    history = list(snapshot.values.get("messages", []))
    structured_evaluator: Any = create_agent(
        evaluator_llm,
        tools=list(tools),
        checkpointer=checkpointer,
        middleware=context_overflow_middlewares(),
        debug=is_enabled_for_debug,
        response_format=response_format,
    )
    final_input = {"messages": [*history, {"role": "user", "content": "Critic Approved."}]}
    return structured_evaluator, final_input, init_langgraph_config(None)


class RefineAgentInput(BaseModel):
    """JSON tool-face parameters for ``refine_agent``. ``tool_names`` is a list of
    registered client-tool names (resolved to live tools at run time); the two
    role prompts are supplied inline or by template id.

    ``base_url``/``api_key`` in ``evaluator_llm_kwargs``/``critic_llm_kwargs``
    legitimately route to a caller-chosen model endpoint; expose any agent or tool
    carrying these kwargs only to trusted callers — an injected parent agent could
    redirect the model call to a hostile endpoint and leak the key/context.

    ``extra="forbid"`` rejects any unknown key loudly at validation rather than
    letting a typo at the run door vanish silently."""

    model_config = ConfigDict(extra="forbid")

    evaluator_message: str = ""
    evaluator_message_id: str = ""
    critic_message: str = ""
    critic_message_id: str = ""
    evaluator_message_kwargs: dict[str, Any] | None = None
    critic_message_kwargs: dict[str, Any] | None = None
    tool_names: list[str] | None = None
    max_iterations: int = 3
    response_format: dict[str, Any] | None = Field(
        default=None, description="JSON Schema of the forced structured output (needs a top-level 'title')."
    )
    evaluator_llm_provider: str | None = None
    critic_llm_provider: str | None = None
    checkpoint_provider: str | None = None
    evaluator_llm_kwargs: dict[str, Any] | None = None
    critic_llm_kwargs: dict[str, Any] | None = None
    evaluator_langgraph_config: dict[str, Any] | None = None
    critic_langgraph_config: dict[str, Any] | None = None


@tai42_app.agents.agent("refine_agent", tags={"agents"})
class RefineAgent(Agent):
    """Evaluator↔Critic refinement agent. ``astream`` runs the loop silently, then
    streams the final approved evaluator pass; ``run`` drains that stream."""

    tool_name: ClassVar[str] = "refine_agent"
    tool_description: ClassVar[str] = (
        "Run a refine workflow: an Evaluator LLM and a Critic LLM iterate until the "
        "Critic approves or max_iterations is reached."
    )
    ToolInput: ClassVar[type[BaseModel]] = RefineAgentInput

    async def run(self, **kwargs: Any) -> Any:
        """Reject the contract parameters the loop has no seat for, then run the loop
        and return the final approved answer by draining :meth:`astream`.

        A ``response_format`` is honored: its JSON-Schema dict must carry a top-level
        ``"title"``, and the drain returns the validated structured object the final
        approved pass forces, raising loudly if the run produced none.
        """
        reject_unhonored("refine_agent.run", kwargs, _UNHONORED_REASONS, collection_params=_UNHONORED_COLLECTION_PARAMS)
        response_format = kwargs.get("response_format")
        reject_untitled_response_format("refine_agent", response_format)
        return await self._drain(self.astream(**kwargs), response_format=response_format)

    async def astream(
        self,
        *,
        evaluator_message: str = "",
        evaluator_message_id: str = "",
        critic_message: str = "",
        critic_message_id: str = "",
        evaluator_message_kwargs: dict[str, Any] | None = None,
        critic_message_kwargs: dict[str, Any] | None = None,
        tool_names: list[str] | None = None,
        max_iterations: int = 3,
        evaluator_llm_provider: str | None = None,
        critic_llm_provider: str | None = None,
        checkpoint_provider: str | None = None,
        evaluator_llm_kwargs: dict[str, Any] | None = None,
        critic_llm_kwargs: dict[str, Any] | None = None,
        evaluator_langgraph_config: dict[str, Any] | None = None,
        critic_langgraph_config: dict[str, Any] | None = None,
        response_format: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the final evaluator pass; the Evaluator↔Critic loop runs silently
        until approval (or raises at ``max_iterations``).

        ``tool_names``, ``checkpoint_provider``, and ``response_format`` are the
        honored contract parameters; every other :class:`~tai42_contract.agent.Agent`
        parameter has no seat in the role-named loop and is rejected loudly (see
        :data:`_UNHONORED_REASONS`).

        With a ``response_format`` set, only the final approved answer is forced into
        the schema, surfaced as a terminal
        :class:`~tai42_contract.agent.events.StructuredFinal`; a final pass producing
        none raises loudly after the stream drains.
        """
        reject_unhonored(
            "refine_agent.astream", kwargs, _UNHONORED_REASONS, collection_params=_UNHONORED_COLLECTION_PARAMS
        )
        reject_untitled_response_format("refine_agent", response_format)
        resolved_tools = await tai42_app.tools.get_client_tools(tool_names) if tool_names else []
        final_agent, final_input, final_config = await _run_refine_loop(
            tools=resolved_tools,
            evaluator_message=evaluator_message,
            evaluator_message_id=evaluator_message_id,
            critic_message=critic_message,
            critic_message_id=critic_message_id,
            evaluator_message_kwargs=evaluator_message_kwargs,
            critic_message_kwargs=critic_message_kwargs,
            max_iterations=max_iterations,
            evaluator_llm_provider=evaluator_llm_provider,
            critic_llm_provider=critic_llm_provider,
            checkpoint_provider=checkpoint_provider,
            evaluator_llm_kwargs=evaluator_llm_kwargs,
            critic_llm_kwargs=critic_llm_kwargs,
            evaluator_config=evaluator_langgraph_config,
            critic_config=critic_langgraph_config,
            response_format=response_format,
        )
        saw_structured = False
        async for event in aproject_agent_events(
            final_agent, final_input, final_config, response_format=response_format
        ):
            if isinstance(event, StructuredFinal):
                saw_structured = True
            yield event
        # A requested response_format that produced no StructuredFinal fails loudly
        # after the stream drains.
        if response_format is not None and not saw_structured:
            raise RuntimeError("agent run requested a response_format but produced no structured output")
