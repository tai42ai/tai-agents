"""Shared message-text extraction.

:func:`text_of` pulls the plain text out of a message whose ``content`` is either
a string or a list of content blocks (the shape newer LangChain providers use).
"""

from __future__ import annotations

from typing import Any


def text_of(message: Any) -> str:
    """The plain text of a message whose ``content`` is either a string or a list
    of content blocks (newer LangChain providers). Non-text blocks (image_url,
    tool_use, thinking, …) are skipped."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""
