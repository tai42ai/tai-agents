"""Render a message through the resource manager.

The agent faces use the empty string ``""`` as the "unset message" sentinel while
the resource manager uses ``None``, and the manager rejects being given both a
``content`` and a ``template_id``. :func:`render_message` maps ``""`` to ``None``
so an unset slot reads as "not provided" instead of tripping that guard, while a
genuine value is preserved.
"""

from __future__ import annotations

from typing import Any

from tai42_contract.app import tai42_app


async def render_message(
    content: str | None,
    template_id: str | None,
    kwargs: dict[str, Any] | None = None,
    *,
    allow_empty: bool = True,
) -> str:
    """Render a message from ``content`` or ``template_id`` via the resource manager.

    Maps the agents' empty-string "unset" sentinel to the manager's ``None``
    sentinel so an unset message or id is treated as "not provided" rather than
    tripping the manager's exactly-one-of guard.
    """
    rendered = await tai42_app.storage.resource_manager.render_by_id_or_content(
        content=content or None,
        template_id=template_id or None,
        kwargs=kwargs,
        allow_empty=allow_empty,
    )
    return rendered
