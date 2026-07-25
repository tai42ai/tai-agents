"""Shared honor-or-raise guards for contract ``Agent.run`` parameters.

Each agent honors only the subset of ``Agent.run`` / ``astream`` parameters its
runtime supports and rejects the rest loudly. :func:`reject_unhonored` is the one
place that rejection lives; each agent supplies its own ``reasons`` map (its keys
define the unhonored set) and the ``collection_params`` whose unset default is an
empty sequence/string. :func:`reject_blank_memory_keys` guards ``thread_id`` /
``resume_checkpoint_id``, and :func:`reject_untitled_response_format` guards the
structured-output name on a JSON-Schema ``response_format``.
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

    ``qualified_face`` is the ``agent.method`` label opening the message (e.g.
    ``"tools_agent.run"``); ``reasons`` maps each unhonorable parameter to why, and
    its keys are exactly the set checked.

    A parameter counts as SET when it is in ``collection_params`` and truthy, or not
    in ``collection_params`` and not ``None``. So "not requested" sentinels pass
    (``None`` for a scalar, ``()`` / ``[]`` / ``""`` for a collection), while a
    falsy-but-meaningful scalar (``resume=False``, ``recursion_limit=0``) still
    raises. Kwargs not named in ``reasons`` are the contract's ``**kwargs``
    extension point and pass through. Offenders are collected, sorted, and named
    together.
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

    ``thread_id`` / ``resume_checkpoint_id`` map into the run config's
    ``configurable`` to pin or fork checkpointed memory. ``None`` means "not set". A
    present but empty or whitespace-only value names a checkpoint namespace two
    independent runs would silently share, so it is rejected rather than written
    verbatim, ``qualified_face`` opening the message.

    The ``isinstance`` check is not redundant with the annotation: agents whose
    faces take ``**kwargs`` reach here with an ``Any``. A non-string is a
    ``TypeError``; a string carrying no name is a ``ValueError``.
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

    The top-level ``"title"`` is the structured-output name the run forces, so a
    dict schema without one is malformed and is named (``agent_name`` opening the
    message). ``None`` and a pydantic class (which carries its own name) pass
    untouched.
    """
    if isinstance(response_format, dict) and "title" not in response_format:
        raise ValueError(
            f"{agent_name} response_format must be a JSON Schema with a top-level 'title' "
            "(used as the structured-output name)."
        )
