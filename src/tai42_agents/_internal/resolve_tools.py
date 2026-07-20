"""Resolve an agent's declared tool inputs into a flat ``StructuredTool`` list.

The tools-agent-shaped agents accept the same three inputs — ``tools`` (live
objects), ``tool_names`` (names resolved through the app's tool registry), and
``presets`` (a base tool bound to fixed kwargs) — and resolve them the same way.
This is the one place that resolution lives.

A preset becomes a ``StructuredTool`` whose LLM-visible arguments are the base
tool's arguments MINUS the fixed ones (so the agent cannot override a fixed
value). The bound callable follows the ``PresetRegistry.preset_tool`` contract:
it invokes the base tool through ``app_tools.run_tool`` with the fixed kwargs
merged UNDER the caller-supplied runtime kwargs.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from tai42_contract.agent.base import PresetSpec
from tai42_contract.tools import AppTools


def _assert_unique_names(tools: list[StructuredTool]) -> None:
    """Reject duplicate tool names — the agent dispatches tools by name, so a
    collision (two presets, or a preset shadowing a client tool) would make
    selection ambiguous."""
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
        # A StructuredTool with a plain-dict args_schema does not strip keys
        # outside the schema, so a model that passed a fixed key would override
        # the bound value on the merge below. Drop fixed keys here to keep the
        # bound values immutable.
        runtime = {key: value for key, value in runtime.items() if key not in preset.fixed_kwargs}
        return await app_tools.run_tool(preset.base_tool, {**preset.fixed_kwargs, **runtime})

    base_tool = (await app_tools.get_client_tools([preset.base_tool]))[0]
    # Read the base tool's ``required`` list before deriving ``args``, so an
    # unusable schema is named here rather than surfacing from inside the tool's
    # own argument derivation. ``args_schema`` carries the list in whichever
    # shape the base tool declared it: a JSON-schema dict, a pydantic model class
    # of either major version, or nothing when the tool declares no schema.
    base_schema = base_tool.args_schema
    if base_schema is None:
        base_required: list[str] = []
    elif isinstance(base_schema, dict):
        base_required = base_schema.get("required", [])
    elif isinstance(base_schema, type) and issubclass(base_schema, BaseModel):
        # ``base_tool.args`` keys a pydantic v2 model by field name, so read
        # ``required`` in the field-name namespace to match: the default
        # alias-keyed list would be filtered out against ``props`` below,
        # silently downgrading an aliased mandatory field to optional.
        base_required = base_schema.model_json_schema(by_alias=False).get("required", [])
    elif isinstance(base_schema, type) and issubclass(base_schema, BaseModelV1):
        # ``base_tool.args`` keys a pydantic v1 model by alias, so ``schema()``'s
        # own alias-keyed ``required`` already matches ``props`` below.
        base_required = base_schema.schema().get("required", [])
    else:
        raise TypeError(
            f"preset base tool {preset.base_tool!r} exposes an unsupported args_schema of type "
            f"{type(base_schema).__name__}; expected a JSON-schema dict, a pydantic model class, or None"
        )
    # A hand-authored JSON-schema dict can carry a ``required`` of any shape, and
    # ``StructuredTool`` preserves either a list or a tuple of names. Accept both,
    # but reject any other shape rather than iterating it: the string
    # ``"session_id"`` would be walked character by character, matching no property,
    # and would silently downgrade a mandatory argument to an optional one.
    if not isinstance(base_required, (list, tuple)) or not all(isinstance(name, str) for name in base_required):
        raise TypeError(
            f"preset base tool {preset.base_tool!r} declares a malformed args_schema 'required': "
            f"expected a list or tuple of property names, got {base_required!r}"
        )
    # A mandatory runtime arg stays mandatory. Filter against the surviving
    # properties so a fixed key, or a field the base schema requires but ``args``
    # omits, cannot slip into the exposed ``required`` list.
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
