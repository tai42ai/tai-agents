"""Resolve an agent's declared tool inputs into a flat ``StructuredTool`` list.

The three inputs — ``tools`` (live objects), ``tool_names`` (resolved through the
app's tool registry), and ``presets`` (a base tool bound to fixed kwargs) — all
resolve here.

A preset becomes a ``StructuredTool`` whose LLM-visible arguments are the base
tool's arguments minus the fixed ones (so the agent cannot override a fixed
value); its bound callable invokes the base tool via ``app_tools.run_tool`` with
the fixed kwargs merged under the caller's runtime kwargs.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from tai42_contract.agent.base import PresetSpec
from tai42_contract.tools import AppTools


def _assert_unique_names(tools: list[StructuredTool]) -> None:
    """Reject duplicate tool names — the agent dispatches by name, so a collision
    would make selection ambiguous."""
    names = [tool.name for tool in tools]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"agent has duplicate tool names across tools/tool_names/presets: {duplicates}. Tool names must be unique."
        )


async def _as_structured_tool(
    app_tools: AppTools,
    preset: PresetSpec,
) -> StructuredTool:
    """Bind ``preset`` into a ``StructuredTool``.

    The base tool's argument schema, minus the preset's fixed keys, becomes the
    new tool's argument schema; the bound callable merges the runtime args over
    the fixed kwargs and invokes the base tool via ``app_tools.run_tool``.
    """

    async def run_impl(**runtime: Any) -> Any:
        # A plain-dict args_schema does not strip out-of-schema keys, so drop the
        # fixed keys here to keep the bound values immutable on the merge below.
        runtime = {key: value for key, value in runtime.items() if key not in preset.fixed_kwargs}
        return await app_tools.run_tool(preset.base_tool, {**preset.fixed_kwargs, **runtime})

    base_tool = (await app_tools.get_client_tools([preset.base_tool]))[0]
    # ``args_schema`` is a JSON-schema dict, a pydantic model class, or None. Read its
    # ``required`` in the same namespace ``base_tool.args`` keys, so filtering against
    # ``props`` below never downgrades a mandatory field.
    base_schema = base_tool.args_schema
    if base_schema is None:
        base_required: list[str] = []
    elif isinstance(base_schema, dict):
        base_required = base_schema.get("required", [])
    elif isinstance(base_schema, type) and issubclass(base_schema, BaseModel):
        # ``args`` keys a model field by field name, so read ``required`` by field name.
        base_required = base_schema.model_json_schema(by_alias=False).get("required", [])
    else:
        # A class reaching here (e.g. a pydantic-v1 model) is named by its own
        # ``__name__``; ``type(...).__name__`` would report its metaclass instead.
        # Any other value is named by its type — a function or module also carries
        # a ``__name__``, which names the value, not the type that was rejected.
        offender = base_schema.__name__ if isinstance(base_schema, type) else type(base_schema).__name__
        raise TypeError(
            f"preset base tool {preset.base_tool!r} exposes an unsupported args_schema of type "
            f"{offender}; expected a JSON-schema dict, a pydantic model class, or None"
        )
    # Accept a list or tuple of names; reject any other shape rather than iterating
    # it (a bare string would be walked char by char, matching no property and
    # silently downgrading a mandatory argument to optional).
    if not isinstance(base_required, (list, tuple)) or not all(isinstance(name, str) for name in base_required):
        raise TypeError(
            f"preset base tool {preset.base_tool!r} declares a malformed args_schema 'required': "
            f"expected a list or tuple of property names, got {base_required!r}"
        )
    # Filter ``required`` against the surviving properties so a fixed key (or a
    # required field ``args`` omits) cannot slip into the exposed ``required`` list.
    props = {key: value for key, value in base_tool.args.items() if key not in preset.fixed_kwargs}
    required = [name for name in base_required if name in props]
    args_schema = {"type": "object", "properties": props, "required": required}

    return StructuredTool.from_function(
        func=None,
        coroutine=run_impl,
        name=preset.name,
        description=preset.description,
        args_schema=args_schema,
    )


async def resolve_tools(
    app_tools: AppTools,
    tool_names: list[str],
    tools: list[StructuredTool],
    presets: list[PresetSpec],
) -> list[StructuredTool]:
    """Resolve ``tools`` + ``tool_names`` + ``presets`` into one deduplicated
    ``StructuredTool`` list (live tools first, then resolved names, then
    presets). ``app_tools`` is the app's tool facet (``tai42_app.tools``)."""
    out: list[StructuredTool] = list(tools or [])
    if tool_names:
        out += await app_tools.get_client_tools(list(tool_names))
    for preset in presets or []:
        out.append(await _as_structured_tool(app_tools, preset))
    _assert_unique_names(out)
    return out
