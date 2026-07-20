"""CI proof for the agent harness: a test-only fake agent registered through
the real decorator, streaming a scripted contract-event sequence.

The fake lives here in ``tests/`` — it is NOT part of the shipped package. It
subclasses the contract ``Agent`` ABC, registers via
``@tai42_app.agents.agent(name)`` at module import (the recording app is bound in
``conftest.py`` before this module imports, mirroring the host's
bind-then-import order), and its ``astream`` yields a fixed
``ReasoningStep`` → ``MessageDelta``s → ``MessageFinal`` → ``RunUsage``
sequence. The tests assert registration end-to-end and the exact event stream.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
from tai42_contract.agent import (
    Agent,
    MessageDelta,
    MessageFinal,
    ReasoningStep,
    RunUsage,
    StreamEvent,
)
from tai42_contract.app import tai42_app

AGENT_NAME = "fake_streamer"


class FakeStreamInput(BaseModel):
    user_message: str = ""


@tai42_app.agents.agent(AGENT_NAME)
class FakeStreamingAgent(Agent):
    """Streams a fixed reasoning → deltas → final → usage script."""

    tool_name = AGENT_NAME
    tool_description = "Test-only agent that streams a scripted event sequence."
    ToolInput = FakeStreamInput

    async def run(self, **kwargs: Any) -> Any:
        return await self._drain(self.astream(**kwargs))

    async def astream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        yield ReasoningStep(text="planning the answer")
        yield MessageDelta(text="hello ")
        yield MessageDelta(text="world")
        yield MessageFinal(text="hello world")
        yield RunUsage(input_tokens=3, output_tokens=2, total_tokens=5, model="scripted")


async def _collect_events(agent: Agent) -> list[StreamEvent]:
    return [event async for event in agent.astream(user_message="hi")]


def test_decorator_registers_a_live_instance() -> None:
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    assert isinstance(agent, FakeStreamingAgent)
    assert isinstance(agent, Agent)


def test_astream_emits_the_exact_scripted_sequence() -> None:
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    events = asyncio.run(_collect_events(agent))

    assert [type(event) for event in events] == [
        ReasoningStep,
        MessageDelta,
        MessageDelta,
        MessageFinal,
        RunUsage,
    ]
    reasoning, delta_one, delta_two, final, usage = events
    assert reasoning == ReasoningStep(text="planning the answer")
    assert delta_one == MessageDelta(text="hello ")
    assert delta_two == MessageDelta(text="world")
    assert final == MessageFinal(text="hello world")
    assert usage == RunUsage(input_tokens=3, output_tokens=2, total_tokens=5, model="scripted")
    assert [event.type for event in events] == [
        "reasoning_step",
        "message_delta",
        "message_delta",
        "message_final",
        "run_usage",
    ]
    assert final.final is True


def test_run_drains_the_stream_to_the_final_text() -> None:
    agent = tai42_app.agents.get_agent(AGENT_NAME)
    assert asyncio.run(agent.run(user_message="hi")) == "hello world"
