"""Tests for ``resolve_tools`` — the shared tool-input resolver.

Covers name resolution, preset binding (base tool + fixed kwargs, with fixed
keys hidden from the exposed schema), and the duplicate-name guard. A fake tool
facet (mirroring ``tai42_app.tools``) stands in for a live app — no LLM.
"""

import asyncio
from typing import cast

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as V1BaseModel
from tai42_contract.agent.base import PresetSpec
from tai42_contract.tools import AppTools

from tai42_agents._internal.resolve_tools import resolve_tools


def make_tool(name, props=None, required=None):
    async def call_tool(**kwargs):
        return kwargs

    args_schema = {"type": "object", "properties": props or {}}
    # ``required=None`` omits the key entirely (the no-``required``-key case);
    # a list pins it explicitly.
    if required is not None:
        args_schema["required"] = required
    return StructuredTool.from_function(
        func=None,
        coroutine=call_tool,
        name=name,
        description="t",
        args_schema=args_schema,
    )


class FakeTools:
    """A stand-in for the app's ``tools`` facet: only the two members
    ``resolve_tools`` reaches (``get_client_tools`` / ``run_tool``)."""

    def __init__(self):
        self.run_tool_calls = []

    async def get_client_tools(self, names=None):
        # Base tools a preset can bind to, keyed by name -> (props, required).
        # A ``required`` of ``None`` omits the schema's ``required`` key.
        props = {"flow_graph": {"type": "object"}, "flow_graph_kwargs": {"type": "object"}}
        schemas = {
            # "flow" carries no ``required`` key.
            "flow": (props, None),
            # "flow_required" marks both runtime args as required.
            "flow_required": (props, ["flow_graph", "flow_graph_kwargs"]),
        }
        return [make_tool(n, *schemas.get(n, (None, None))) for n in (names or [])]

    async def run_tool(self, key, arguments):
        self.run_tool_calls.append((key, arguments))
        return {"ok": True, "args": arguments}


def _app_tools(fake: FakeTools) -> AppTools:
    return cast(AppTools, fake)


def test_resolves_names_and_live_tools():
    app = FakeTools()
    live = make_tool("live")
    out = asyncio.run(resolve_tools(_app_tools(app), ["a", "b"], [live], []))
    assert [t.name for t in out] == ["live", "a", "b"]


def test_preset_hides_fixed_keys_and_binds():
    app = FakeTools()
    preset = PresetSpec(
        name="my_flow",
        description="run a flow",
        base_tool="flow",
        fixed_kwargs={"flow_graph": {"nodes": []}},
    )
    out = asyncio.run(resolve_tools(_app_tools(app), [], [], [preset]))
    (tool,) = out
    assert tool.name == "my_flow"
    # fixed key removed from the exposed schema; runtime key kept.
    assert "flow_graph" not in tool.args
    assert "flow_graph_kwargs" in tool.args
    # invoking merges fixed kwargs under runtime kwargs and calls the base tool.
    asyncio.run(tool.arun({"flow_graph_kwargs": {"x": 1}}))
    key, arguments = app.run_tool_calls[-1]
    assert key == "flow"
    assert arguments == {"flow_graph": {"nodes": []}, "flow_graph_kwargs": {"x": 1}}


def test_preset_cannot_override_a_fixed_key():
    """A model that supplies a fixed key does not override the bound value."""
    app = FakeTools()
    preset = PresetSpec(
        name="my_flow",
        description="run a flow",
        base_tool="flow",
        fixed_kwargs={"flow_graph": {"nodes": []}},
    )
    (tool,) = asyncio.run(resolve_tools(_app_tools(app), [], [], [preset]))
    asyncio.run(tool.arun({"flow_graph": {"nodes": ["evil"]}, "flow_graph_kwargs": {"x": 1}}))
    _key, arguments = app.run_tool_calls[-1]
    assert arguments["flow_graph"] == {"nodes": []}  # bound value immutable


def test_preset_keeps_a_required_runtime_key_required():
    """A required base-tool arg that is not fixed stays required on the bound
    preset tool, so the model cannot silently omit it."""
    app = FakeTools()
    preset = PresetSpec(
        name="my_flow",
        description="run a flow",
        base_tool="flow_required",
        fixed_kwargs={"flow_graph": {"nodes": []}},
    )
    (tool,) = asyncio.run(resolve_tools(_app_tools(app), [], [], [preset]))
    assert isinstance(tool.args_schema, dict)
    assert tool.args_schema["required"] == ["flow_graph_kwargs"]


def test_preset_drops_a_fixed_required_key():
    """A required base-tool arg that the preset fixes is removed from the
    exposed schema's ``required`` list."""
    app = FakeTools()
    preset = PresetSpec(
        name="my_flow",
        description="run a flow",
        base_tool="flow_required",
        fixed_kwargs={"flow_graph": {"nodes": []}},
    )
    (tool,) = asyncio.run(resolve_tools(_app_tools(app), [], [], [preset]))
    assert isinstance(tool.args_schema, dict)
    assert "flow_graph" not in tool.args_schema["required"]


def test_preset_base_schema_without_required_key():
    """A base tool whose schema carries no ``required`` key binds to an empty
    ``required`` list rather than raising."""
    app = FakeTools()
    preset = PresetSpec(
        name="my_flow",
        description="run a flow",
        base_tool="flow",
        fixed_kwargs={"flow_graph": {"nodes": []}},
    )
    (tool,) = asyncio.run(resolve_tools(_app_tools(app), [], [], [preset]))
    assert isinstance(tool.args_schema, dict)
    assert tool.args_schema["required"] == []


def test_rejects_duplicate_names():
    app = FakeTools()
    a = make_tool("dup")
    b = make_tool("dup")
    with pytest.raises(ValueError, match="duplicate tool names"):
        asyncio.run(resolve_tools(_app_tools(app), [], [a, b], []))

    preset = PresetSpec(name="dup", base_tool="flow", fixed_kwargs={})
    with pytest.raises(ValueError, match="duplicate tool names"):
        asyncio.run(resolve_tools(_app_tools(app), [], [make_tool("dup")], [preset]))


class _SchemaTools:
    """A ``tools`` facet serving one base tool with a caller-chosen ``args_schema``."""

    def __init__(self, args_schema):
        self._args_schema = args_schema

    async def get_client_tools(self, names=None):
        async def call_tool(**kwargs):
            return kwargs

        tool = StructuredTool.from_function(
            func=None,
            coroutine=call_tool,
            name="base",
            description="t",
            args_schema={"type": "object", "properties": {}},
        )
        tool.args_schema = self._args_schema
        return [tool]

    async def run_tool(self, name, arguments):
        return arguments


def _preset() -> PresetSpec:
    return PresetSpec(name="p", description="d", base_tool="base", fixed_kwargs={})


def test_preset_reads_required_from_a_pydantic_args_schema():
    """A base tool declaring its schema as a pydantic model keeps its required list."""

    class Schema(BaseModel):
        keep: str
        drop: str = "d"

    app = cast(AppTools, _SchemaTools(Schema))
    (tool,) = asyncio.run(resolve_tools(app, [], [], [_preset()]))
    assert isinstance(tool.args_schema, dict)
    assert tool.args_schema["required"] == ["keep"]


def test_preset_keeps_an_aliased_required_field_required():
    """An aliased pydantic field declared required stays required on the bound
    preset tool. ``base_tool.args`` keys a field by its name, so the exposed
    ``required`` carries the field name and the tool is callable with it."""

    class Schema(BaseModel):
        session_id: str = Field(alias="sessionId")

    app = cast(AppTools, _SchemaTools(Schema))
    (tool,) = asyncio.run(resolve_tools(app, [], [], [_preset()]))
    assert isinstance(tool.args_schema, dict)
    assert tool.args_schema["required"] == ["session_id"]
    assert "session_id" in tool.args
    assert asyncio.run(tool.arun({"session_id": "s"})) == {"session_id": "s"}


def test_preset_base_tool_without_an_args_schema_requires_nothing():
    """A base tool that declares no schema requires nothing, rather than erroring."""
    app = cast(AppTools, _SchemaTools(None))
    (tool,) = asyncio.run(resolve_tools(app, [], [], [_preset()]))
    assert isinstance(tool.args_schema, dict)
    assert tool.args_schema["required"] == []


def test_preset_base_tool_with_an_unsupported_args_schema_raises():
    """An ``args_schema`` of an unexpected type is refused loudly, never defaulted away,
    and the error names the value's own type."""
    app = cast(AppTools, _SchemaTools(42))
    with pytest.raises(TypeError, match="unsupported args_schema") as excinfo:
        asyncio.run(resolve_tools(app, [], [], [_preset()]))
    assert "of type int" in str(excinfo.value)


def test_preset_base_tool_with_an_unsupported_args_schema_class_names_the_class():
    """An unsupported ``args_schema`` supplied as a class — a pydantic-v1 model, say —
    is named by the class itself, not by its metaclass (``ModelMetaclass``), which would
    tell the reader nothing about which schema was rejected."""

    class LegacySchema(V1BaseModel):
        keep: str

    app = cast(AppTools, _SchemaTools(LegacySchema))
    with pytest.raises(TypeError, match="unsupported args_schema") as excinfo:
        asyncio.run(resolve_tools(app, [], [], [_preset()]))
    assert "of type LegacySchema" in str(excinfo.value)
    assert "ModelMetaclass" not in str(excinfo.value)


def test_preset_base_tool_with_an_unsupported_args_schema_function_names_the_type():
    """A non-class value that carries its own ``__name__`` — a function, say — is
    named by its type, not by that ``__name__``: the message reports what kind of
    value was rejected, never the value's own label."""

    def my_helper() -> None: ...

    app = cast(AppTools, _SchemaTools(my_helper))
    with pytest.raises(TypeError, match="unsupported args_schema") as excinfo:
        asyncio.run(resolve_tools(app, [], [], [_preset()]))
    assert "of type function" in str(excinfo.value)
    assert "my_helper" not in str(excinfo.value)


def _named_preset() -> PresetSpec:
    return PresetSpec(name="p", description="d", base_tool="wobbly_tool", fixed_kwargs={})


@pytest.mark.parametrize(
    "malformed_required",
    ["session_id", None, ["ok", 123]],
    ids=["bare-string", "none", "list-with-non-string"],
)
def test_preset_base_tool_with_malformed_required_raises(malformed_required):
    """A hand-authored JSON-schema ``required`` that is not a list of names is refused
    loudly — a bare string would be walked character by character and silently downgrade a
    mandatory argument to optional — and the error names the offending base tool."""
    args_schema = {"type": "object", "properties": {"keep": {"type": "string"}}, "required": malformed_required}
    app = cast(AppTools, _SchemaTools(args_schema))
    with pytest.raises(TypeError, match="malformed args_schema 'required'") as excinfo:
        asyncio.run(resolve_tools(app, [], [], [_named_preset()]))
    assert "wobbly_tool" in str(excinfo.value)


def test_preset_reads_required_from_a_well_formed_dict_args_schema():
    """A well-formed JSON-schema ``required`` list of names passes the guard, and its
    names survive onto the exposed schema."""
    args_schema = {"type": "object", "properties": {"keep": {"type": "string"}}, "required": ["keep"]}
    app = cast(AppTools, _SchemaTools(args_schema))
    (tool,) = asyncio.run(resolve_tools(app, [], [], [_named_preset()]))
    assert isinstance(tool.args_schema, dict)
    assert tool.args_schema["required"] == ["keep"]


def test_preset_accepts_a_tuple_required():
    """A ``required`` supplied as a tuple is accepted — ``StructuredTool`` preserves
    either shape — and its surviving names land on the exposed schema as a list."""
    args_schema = {"type": "object", "properties": {"keep": {"type": "string"}}, "required": ("keep",)}
    app = cast(AppTools, _SchemaTools(args_schema))
    (tool,) = asyncio.run(resolve_tools(app, [], [], [_named_preset()]))
    assert isinstance(tool.args_schema, dict)
    assert tool.args_schema["required"] == ["keep"]
