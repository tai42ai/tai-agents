"""The voting agent — N voter LLMs answer in parallel, one judge LLM decides.

Each ``VoterSpec`` in ``voters`` spawns one voter; a list lets the same provider
appear more than once and each voter pin its own model. Every voter answers the
same rendered ``voter_message`` in parallel, then the judge receives the rendered
``judge_message`` followed by one line per voter (provider, model, verdict) and
decides by majority vote, breaking ties with its own reasoning.

Only the judge streams; the voters run silently in the prelude and their verdicts
are collected before the judge starts. ``astream`` emits the judge's per-step
events and ends with a terminal ``StructuredFinal`` carrying the full
``VotingOutput``; ``run`` drains that stream.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict
from tai42_contract.agent import Agent
from tai42_contract.agent.events import MessageFinal, RunUsage, StreamEvent, StructuredFinal
from tai42_contract.app import tai42_app
from tai42_kit.llm.settings import llm_provider_settings

from tai42_agents._internal.base_tool_agent import ainvoke_tools_agent
from tai42_agents._internal.reject import reject_unhonored
from tai42_agents._internal.render import render_message
from tai42_agents._internal.stream_events import astream_tools_agent_events
from tai42_agents.settings import agents_limits_settings
from tai42_agents.voting_agent.model import VoteInfo, VoterSpec, VotingOutput
from tai42_agents.voting_agent.prompt import JUDGE_SYSTEM_MESSAGE, VOTER_SYSTEM_MESSAGE


async def _run_voters(
    judge_message: str | None,
    voter_message: str | None,
    judge_message_id: str | None,
    voter_message_id: str | None,
    judge_message_kwargs: dict[str, Any] | None,
    voter_message_kwargs: dict[str, Any] | None,
    judge_llm_provider: str | None,
    judge_llm_kwargs: dict[str, Any] | None,
    voters: list[VoterSpec] | None,
    voter_tools: list[StructuredTool] | None,
    checkpoint_provider: str | None,
    voter_config: dict[str, Any] | None,
) -> tuple[list[str], list[VoteInfo], str, dict[str, Any]]:
    """Render the messages, run every voter in parallel, and assemble the judge's
    input.

    Returns ``(user_messages, voters_info, judge_llm_provider, judge_llm_kwargs)``.
    ``judge_llm_provider`` / ``judge_llm_kwargs`` come back already defaulted (never
    ``None``); ``user_messages`` is the rendered judge message followed by one line
    per voter. When ``voters`` is empty, a single voter runs on the judge's provider.

    Each ``VoteInfo.model`` records the model the voter actually ran on when known,
    else the model its spec pinned, else ``None`` — never a placeholder.
    """
    rendered_judge_message: str = await render_message(
        judge_message, judge_message_id, judge_message_kwargs, allow_empty=False
    )
    rendered_voter_message: str = await render_message(
        voter_message, voter_message_id, voter_message_kwargs, allow_empty=False
    )
    resolved_judge_llm_provider = judge_llm_provider or llm_provider_settings().llm
    resolved_judge_llm_kwargs = judge_llm_kwargs or {}

    voters = voters or [VoterSpec(provider=resolved_judge_llm_provider, llm_kwargs=resolved_judge_llm_kwargs or None)]
    voter_tools = voter_tools or []

    limits = agents_limits_settings()
    if len(voters) > limits.max_voters:
        raise ValueError(
            f"voting_agent got {len(voters)} voters; the limit is {limits.max_voters} (TAI_AGENTS_MAX_VOTERS)"
        )

    sem = asyncio.Semaphore(limits.voter_concurrency)

    async def _run_one(spec: VoterSpec) -> Any:
        async with sem:
            return await ainvoke_tools_agent(
                system_message=VOTER_SYSTEM_MESSAGE,
                user_message=[rendered_voter_message],
                tools=voter_tools,
                llm_provider=spec.provider,
                checkpoint_provider=checkpoint_provider,
                llm_kwargs=spec.effective_llm_kwargs(),
                config=voter_config,
            )

    results = await asyncio.gather(*(_run_one(spec) for spec in voters))

    voters_info = [
        VoteInfo(
            provider=spec.provider,
            model=results[i].usage.model or spec.declared_model,
            verdict=results[i].output,
        )
        for i, spec in enumerate(voters)
    ]

    voter_messages = [
        f"Provider: {info.provider}, Model: {info.model}, Verdict: {info.verdict}" for info in voters_info
    ]
    user_messages = [rendered_judge_message, *voter_messages]
    return user_messages, voters_info, resolved_judge_llm_provider, resolved_judge_llm_kwargs


# ABC ``Agent.run`` parameters the voting runtime has no seat for, mapped to the
# reason named in the raised error (the keys define this agent's unhonored set).
# Voting drives its judge and voters through role-named params (``judge_*``/
# ``voter_*``) plus ``checkpoint_provider``. ``response_format`` is conditional:
# ``VotingOutput`` (the type this agent always produces) is accepted and removed
# before the guard runs, any other type is an offender.
_UNHONORED_REASONS: dict[str, str] = {
    "tools": "it drives its judge and voters through role-named judge_tools/voter_tools",
    "tool_names": "it drives its judge and voters through role-named judge_tools/voter_tools",
    "presets": "it exposes no preset tools; judge and voter tools are role-named",
    "subagents": "it runs no sub-agent machinery",
    "system_message": "it renders role-named judge_message/voter_message and takes no system_message",
    "user_message": "it renders role-named judge_message/voter_message and takes no user_message",
    "strategy": "it applies no composition strategy and will not silently ignore one",
    "interrupt_on": "its judge and voter graphs never pause for external input",
    "skills": "it loads no skills backend",
    "inline_skills": "it loads no skills backend",
    "recursion_limit": (
        "recursion_limit is per-role RunnableConfig content set via judge_/voter_ langgraph_config; "
        "a single global value would mask the per-role setting"
    ),
    "thread_id": (
        "thread_id is per-role RunnableConfig content set via judge_/voter_ langgraph_config's configurable; "
        "a single global thread_id would collide the two role graphs' checkpoint namespaces"
    ),
    "resume": "its judge and voter graphs never interrupt, so there is no paused run to resume",
    "resume_checkpoint_id": (
        "resume_checkpoint_id is per-role RunnableConfig content set via judge_/voter_ langgraph_config's "
        "configurable; a single global checkpoint would collide the two role graphs' checkpoint namespaces"
    ),
    "llm_provider": "it uses role-named judge_llm_provider and per-voter providers",
    "store_provider": "it wires no long-term store",
    "llm_kwargs": "it uses role-named judge_llm_kwargs and per-voter kwargs",
    "response_format": (
        "it always produces a VotingOutput and cannot honor a different structured output type; "
        "pass VotingOutput or omit it"
    ),
}
# The unhonored parameters whose unset default is an empty sequence/string; every
# other defaults to ``None`` and is set when not ``None``.
_UNHONORED_COLLECTION_PARAMS: frozenset[str] = frozenset(
    {"tools", "tool_names", "presets", "subagents", "skills", "inline_skills", "system_message", "user_message"}
)


def _reject_unhonored(face: str, response_format: Any, extra_kwargs: dict[str, Any]) -> None:
    """Reject, in ONE raise, every ABC ``Agent.run`` parameter the voting runtime
    cannot honor.

    ``response_format`` folds into the same guard: ``VotingOutput`` (or an unset
    ``None``) is dropped before the guard runs, any other type is an offender named
    alongside the rest. ``face`` is the ``voting_agent.run`` / ``.astream`` label."""
    checked = dict(extra_kwargs)
    checked["response_format"] = response_format
    if response_format is VotingOutput:
        del checked["response_format"]
    reject_unhonored(face, checked, _UNHONORED_REASONS, collection_params=_UNHONORED_COLLECTION_PARAMS)


class VotingAgentInput(BaseModel):
    """JSON tool-face params. ``judge_tools`` / ``voter_tools`` are client-tool
    names resolved to live tools at run time (live tools are API-only via
    ``astream`` and are not part of this JSON shape).

    ``base_url``/``api_key`` in ``judge_llm_kwargs`` legitimately route to a
    caller-chosen model endpoint; expose any agent or tool carrying these kwargs
    only to trusted callers — an injected parent agent could redirect the model call
    to a hostile endpoint and leak the key/context.

    ``extra="forbid"`` rejects any unknown key loudly at validation rather than
    letting a typo at the run door vanish silently."""

    model_config = ConfigDict(extra="forbid")

    judge_tools: list[str] | None = None
    voter_tools: list[str] | None = None
    judge_message: str | None = ""
    voter_message: str | None = ""
    judge_message_id: str | None = ""
    voter_message_id: str | None = ""
    judge_message_kwargs: dict[str, Any] | None = None
    voter_message_kwargs: dict[str, Any] | None = None
    judge_llm_provider: str | None = None
    checkpoint_provider: str | None = None
    judge_llm_kwargs: dict[str, Any] | None = None
    voters: list[VoterSpec] | None = None
    judge_langgraph_config: dict[str, Any] | None = None
    voter_langgraph_config: dict[str, Any] | None = None


@tai42_app.agents.agent("voting_agent", tags={"agents"})
class VotingAgent(Agent):
    """Run a voting workflow: voter LLMs answer in parallel, then a judge LLM
    decides. ``run`` returns the ``VotingOutput``; ``astream`` streams the judge's
    steps and ends with a ``StructuredFinal`` of the same ``VotingOutput``."""

    tool_name: ClassVar[str] = "voting_agent"
    tool_description: ClassVar[str] = (
        "Run a voting workflow: multiple voter LLM agents in parallel, then a judge "
        "LLM agent evaluates their responses."
    )
    ToolInput: ClassVar[type[BaseModel]] = VotingAgentInput

    async def run(self, **kwargs: Any) -> VotingOutput:
        """Drain ``astream`` to its terminal ``VotingOutput``. A ``response_format``
        other than ``VotingOutput``, or any unhonored ABC parameter, is rejected
        before draining. The drain forces the ``VotingOutput`` structured terminal —
        a missing one raises rather than falling back to message text."""
        _reject_unhonored("voting_agent.run", kwargs.get("response_format"), kwargs)
        return await self._drain(self.astream(**kwargs), response_format=VotingOutput)

    async def astream(
        self,
        *,
        judge_tools: list[str] | None = None,
        voter_tools: list[str] | None = None,
        judge_message: str = "",
        voter_message: str = "",
        judge_message_id: str = "",
        voter_message_id: str = "",
        judge_message_kwargs: dict[str, Any] | None = None,
        voter_message_kwargs: dict[str, Any] | None = None,
        judge_llm_provider: str | None = None,
        checkpoint_provider: str | None = None,
        judge_llm_kwargs: dict[str, Any] | None = None,
        voters: list[VoterSpec] | None = None,
        judge_langgraph_config: dict[str, Any] | None = None,
        voter_langgraph_config: dict[str, Any] | None = None,
        response_format: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Resolve the judge/voter tool names, run the voters silently, stream the
        judge's events, then yield a terminal ``StructuredFinal`` carrying the full
        ``VotingOutput``.

        A ``response_format`` other than ``VotingOutput``, or any unhonored ABC
        parameter, is rejected loudly here in parity with :meth:`run`."""
        _reject_unhonored("voting_agent.astream", response_format, kwargs)
        resolved_judge_tools: list[StructuredTool] = (
            await tai42_app.tools.get_client_tools(judge_tools) if judge_tools else []
        )
        resolved_voter_tools: list[StructuredTool] = (
            await tai42_app.tools.get_client_tools(voter_tools) if voter_tools else []
        )

        user_messages, voters_info, judge_llm_provider, judge_llm_kwargs = await _run_voters(
            judge_message,
            voter_message,
            judge_message_id,
            voter_message_id,
            judge_message_kwargs,
            voter_message_kwargs,
            judge_llm_provider,
            judge_llm_kwargs,
            voters,
            resolved_voter_tools,
            checkpoint_provider,
            voter_langgraph_config,
        )

        judge_verdict = ""
        judge_model: str | None = None
        async for event in astream_tools_agent_events(
            system_message=JUDGE_SYSTEM_MESSAGE,
            user_message=user_messages,
            tools=resolved_judge_tools,
            llm_provider=judge_llm_provider,
            checkpoint_provider=checkpoint_provider,
            llm_kwargs=judge_llm_kwargs,
            config=judge_langgraph_config,
        ):
            if isinstance(event, MessageFinal):
                judge_verdict = event.text
            elif isinstance(event, RunUsage):
                judge_model = event.model
            yield event

        yield StructuredFinal(
            data=VotingOutput(
                judge=VoteInfo(
                    provider=judge_llm_provider,
                    model=judge_model or judge_llm_kwargs.get("model"),
                    verdict=judge_verdict,
                ),
                voters=voters_info,
            )
        )
