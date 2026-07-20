"""The shared honor-or-raise guard for contract ``Agent.run`` parameters.

Every registered agent honors only the subset of the contract ``Agent.run`` /
``astream`` parameters its runtime has a seat for; the rest it rejects loudly
rather than silently dropping. :func:`reject_unhonored` is the one place that
rejection lives, so all agents raise the same shape of error and apply the same
"was this parameter actually set?" rule.

Each agent supplies its own ``reasons`` map — the keys DEFINE that agent's
unhonored set, the values name why each cannot be honored — and the set of
``collection_params`` whose unset default is falsy-but-not-``None`` (an empty
sequence or an empty string).

The agents that HONOR ``thread_id`` / ``resume_checkpoint_id`` share a second
guard here — :func:`reject_blank_memory_keys` — so a memory key that is not a
string, or is an empty/whitespace-only one, is rejected as malformed in one place
rather than written verbatim into the checkpoint namespace.

The agents that honor ``response_format`` share a third —
:func:`reject_untitled_response_format` — so a JSON-Schema dict carrying no
structured-output name is rejected the same way, and up front, on every agent and
on both of each agent's faces (the invoke face and the streaming face the public
run door drives).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def reject_unhonored(
    qualified_face: str,
    kwargs: Mapping[str, Any],
    reasons: Mapping[str, str],
    *,
    collection_params: frozenset[str] = frozenset(),
) -> None:
    """Raise ``RuntimeError`` naming every unhonored parameter the caller set.

    ``qualified_face`` is the ``agent.method`` label that opens the message (e.g.
    ``"tools_agent.run"``). ``kwargs`` is the call's keyword arguments; ``reasons``
    maps each unhonorable parameter name to the reason it cannot be honored, and
    its keys are exactly the set this guard checks.

    A parameter counts as SET — and is therefore rejected — when it is in
    ``collection_params`` and truthy (a non-empty sequence or string), or when it
    is not in ``collection_params`` and is not ``None``. So the contract's own
    "not requested" sentinels pass untouched: ``None`` for a scalar, ``()`` / ``[]``
    / ``""`` for a collection parameter — while a falsy-but-meaningful value on a
    scalar (``resume=False``, ``strategy=""``, ``recursion_limit=0``) still raises.
    Extension kwargs not named in ``reasons`` are the contract's ``**kwargs``
    extension point and pass through untouched.

    All offenders are collected and named together, sorted, so a caller passing
    several unhonored parameters fixes them in one pass rather than one per run.
    """
    offenders = sorted(
        name
        for name in reasons
        if (bool(kwargs.get(name)) if name in collection_params else kwargs.get(name) is not None)
    )
    if offenders:
        reason = "; ".join(f"{name}: {reasons[name]}" for name in offenders)
        raise RuntimeError(f"{qualified_face} does not support {', '.join(offenders)}: {reason}")


def reject_blank_memory_keys(
    qualified_face: str,
    *,
    thread_id: str | None,
    resume_checkpoint_id: str | None,
) -> None:
    """Raise when an honored memory key is present but malformed.

    An honoring agent maps ``thread_id`` / ``resume_checkpoint_id`` into the run
    config's ``configurable`` to pin or fork checkpointed memory. ``None`` is the
    "not set" sentinel — no overlay applies and a fresh ``thread_id`` is minted as
    usual. A value that is present but empty or whitespace-only is NOT a sentinel:
    it names a checkpoint namespace two independent runs would silently share,
    contaminating each other's memory. Such a value is malformed and raises here
    rather than being written verbatim, ``qualified_face`` opening the message as
    the ``agent.method`` face it was called on.

    The ``isinstance`` check is not redundant with the annotation. The agents whose
    faces take ``**kwargs`` reach this guard with an ``Any``, which satisfies
    ``str | None`` statically and can be any object at run time; the annotation only
    catches a caller that passes a *typed* non-string. A non-string is a type
    violation (``TypeError``); a string that carries no name is a value one
    (``ValueError``). Neither is diagnosed by an ``AttributeError`` from probing a
    non-string for whitespace.
    """
    for name, value in (("thread_id", thread_id), ("resume_checkpoint_id", resume_checkpoint_id)):
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"{qualified_face}: {name} must be a string or None; got {type(value).__name__}")
        if not value.strip():
            raise ValueError(f"{qualified_face}: {name} must be a non-empty string; got {value!r}")


def reject_untitled_response_format(agent_name: str, response_format: Any) -> None:
    """Raise ``ValueError`` when a JSON-Schema ``response_format`` carries no ``"title"``.

    The top-level ``"title"`` is the structured-output name the run forces the
    model into, so a dict schema without one is a malformed request and is named as
    such — ``agent_name`` opening the message — rather than handed to the graph.
    A ``None`` (no structured output requested) and a pydantic class (which carries
    its own name) both pass untouched.
    """
    if isinstance(response_format, dict) and "title" not in response_format:
        raise ValueError(
            f"{agent_name} response_format must be a JSON Schema with a top-level 'title' "
            "(used as the structured-output name)."
        )
