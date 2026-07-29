"""The input and output shapes for the voting agent.

``VoterSpec`` configures one voter (provider, optional explicit model, extra LLM
kwargs); voters are a LIST, so the same provider may appear more than once.
``VotingOutput`` is the terminal value: the judge's verdict plus every voter's, each
a ``VoteInfo`` recording the LLM (provider + model) and the verdict text.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class VoterSpec(BaseModel):
    """One voter's configuration.

    ``provider`` selects the LLM provider; ``model`` optionally pins a model name
    (winning over a ``model`` inside ``llm_kwargs``); ``llm_kwargs`` carries further
    provider kwargs.

    ``base_url``/``api_key`` in ``llm_kwargs`` route to a caller-chosen model
    endpoint, so expose any agent carrying these kwargs only to trusted callers — an
    injected parent could redirect the model call and leak the key/context.

    ``extra="forbid"`` rejects any unknown key loudly at validation.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str | None = None
    llm_kwargs: dict[str, Any] | None = None

    def effective_llm_kwargs(self) -> dict[str, Any]:
        """The kwargs handed to the LLM: ``llm_kwargs`` with the explicit
        ``model`` field folded in when it is set."""
        kwargs = dict(self.llm_kwargs or {})
        if self.model is not None:
            kwargs["model"] = self.model
        return kwargs

    @property
    def declared_model(self) -> str | None:
        """The model this spec pins, if any — the explicit ``model`` field or a
        ``model`` inside ``llm_kwargs``. ``None`` when the spec leaves the model
        to settings resolution."""
        if self.model is not None:
            return self.model
        return (self.llm_kwargs or {}).get("model")


class VoteInfo(BaseModel):
    """One LLM's verdict — the provider and model that produced it, and its
    answer text.

    ``model`` is the actual model that ran (as reported by the run's usage) when
    known, otherwise the model the spec pinned; it is ``None`` when neither the
    run nor the spec names a model, so no misleading placeholder is recorded.
    """

    provider: str
    model: str | None
    verdict: str


class VotingOutput(BaseModel):
    """The voting agent's result: the judge's deciding verdict and the list of
    every voter's verdict."""

    judge: VoteInfo
    voters: list[VoteInfo]
