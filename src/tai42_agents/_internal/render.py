"""The one place that renders a message through the resource manager.

The agent tool-faces and their ``run`` / ``astream`` signatures materialize an
*absent* message (or template id) as the empty string ``""`` — e.g.
``system_message: str = ""`` — while the resource manager's "not provided"
sentinel is ``None``. The manager also enforces an exactly-one-of guard: passing
both a ``content`` and a ``template_id`` raises ``ValueError``. If the agents
handed their empty-string defaults straight through, an unset message and an
unset id would both read as "provided" and trip that guard on every render.

:func:`render_message` is the single seam that maps the empty-string sentinel to
``None`` (``content or None`` turns ``""`` into ``None`` and keeps any non-empty
string). An unset slot becomes "not provided"; a genuine value is preserved, so
supplying both a non-empty content and a non-empty id for the same slot still
raises, which is correct.
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
