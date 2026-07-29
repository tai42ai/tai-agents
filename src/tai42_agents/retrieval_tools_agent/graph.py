"""The retrieval-tools ``StateGraph``.

Instead of binding every tool to the model up front, this graph embeds each
tool's ``name: description`` into a vector store and exposes a single
``retrieve_tools`` search tool. The model asks for tools by describing what it
needs; the matching tool ids come back, get bound for the next step, and the
loop continues until the model emits a terminal ``{"status": ...}`` JSON object.

Node loop (every turn passes through ``context`` first so the history is reduced
before the next model call):

    context -> agent -> [select_tools | execute_tools] -> context -> agent -> ...
                     agent -> END (on a terminal status JSON)

``select_tools`` runs the ``retrieve_tools`` search and feeds the found tool
names back to the model as a ``ToolMessage``; ``execute_tools`` runs the actual
tools the model selected. Routing (``should_continue``) inspects the model's
last message: pending tool calls are dispatched, otherwise the message must be
the terminal status JSON.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import InjectedStore, ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Send
from langgraph.utils.runnable import RunnableCallable
from pydantic import ValidationError
from tai42_kit.llm.middleware.context_overflow import areduce_context, context_overflow_middlewares
from tai42_kit.utils.data.string_util import text_to_md5


def _merged_selected_tools(left: list[str], right: list[str]) -> list[str]:
    # De-dup against ``left`` AND within ``right`` (order preserved): one turn's
    # multiple retrieve calls can each return the same tool id, and binding the
    # same tool twice makes some providers reject the duplicate function name.
    seen = set(left)
    merged = list(left)
    for item in right:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


class State(MessagesState):
    selected_tool_ids: Annotated[list[str], _merged_selected_tools]


class RetrievalToolsGraph:
    """Builds the retrieval-tools ``StateGraph``.

    ``abuild`` embeds every tool's description into ``store`` under
    :attr:`namespace`, compiles the graph, and returns the compiled graph.
    """

    retrieve_tools_tool_name = "retrieve_tools"
    agent_node_name = "agent"
    context_node_name = "context"
    select_tools_node_name = "select_tools"
    execute_tools_node_name = "execute_tools"

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: BaseChatModel,
        store: BaseStore | None = None,
        checkpoint: BaseCheckpointSaver | None = None,
        overwrite_store: bool = False,
        tools_limit: int = 10,
    ):
        self.tools = list(tools)
        self.llm = llm
        self.store = store
        self.checkpoint = checkpoint
        self.tool_registry: dict[str, BaseTool] = {}

        # Namespace the store index per resolved tool-set (a stable hash of the
        # sorted tool names) so concurrent sessions with different tool-sets never
        # share one index and crowd each other's ``tools_limit``. Sorting keeps the
        # hash deterministic, so identical runs reuse the embedding cache.
        tool_set_hash = text_to_md5("\n".join(sorted(tool.name for tool in self.tools)))
        self.namespace: tuple[str, ...] = ("tools", tool_set_hash)

        self._agent_node: RunnableCallable | None = None
        self._context_node: RunnableCallable | None = None
        self._retrieve_tools_tool: StructuredTool | None = None
        self._select_tools_node: RunnableCallable | None = None
        self._execute_tools_node: ToolNode | None = None
        self._should_continue_node: Callable[..., Any] | None = None
        self._overwrite_store = overwrite_store
        self._tools_limit = tools_limit
        self._context_middlewares = context_overflow_middlewares()

    async def abuild(self) -> Any:
        store = self.store
        if store is None:
            raise ValueError("RetrievalToolsGraph requires a store to embed tool descriptions into")

        for tool in self.tools:
            self.tool_registry[text_to_md5(tool.name)] = tool

        for tool_id, tool in self.tool_registry.items():
            await self.add_to_store(store, tool_id, tool)

        builder = self.build()
        return builder.compile(store=store, checkpointer=self.checkpoint)

    async def add_to_store(self, store: BaseStore, key: str, tool: BaseTool) -> None:
        item = None
        if self._overwrite_store:
            await store.adelete(namespace=self.namespace, key=key)
        else:
            item = await store.aget(namespace=self.namespace, key=key)

        if item is None:
            await store.aput(
                namespace=self.namespace,
                key=key,
                value={"description": f"{tool.name}: {tool.description}"},
            )

    def _registered_tool(self, tool_id: str) -> BaseTool:
        # A retrieved id always addresses this tool-set's own namespace, so it
        # must resolve to a registered tool. A miss signals a real inconsistency
        # (stale/foreign id) and raises rather than being silently dropped.
        tool = self.tool_registry.get(tool_id)
        if tool is None:
            raise KeyError(
                f"tool id {tool_id!r} has no entry in this tool-set's registry; "
                "the retrieved id does not belong to the embedded tool-set"
            )
        return tool

    def retrieve_tools_tool(self) -> StructuredTool:
        if self._retrieve_tools_tool:
            return self._retrieve_tools_tool

        async def tool(query: str, *, store: Annotated[BaseStore, InjectedStore]) -> list[str]:
            """Retrieve tool IDs from store via semantic search."""
            results = await store.asearch(self.namespace, query=query, limit=self._tools_limit)
            return [result.key for result in results]

        self._retrieve_tools_tool = StructuredTool.from_function(
            func=None, coroutine=tool, name=self.retrieve_tools_tool_name
        )
        return self._retrieve_tools_tool

    def agent_node(self) -> RunnableCallable:
        if self._agent_node:
            return self._agent_node

        async def node(state: State, config: RunnableConfig, *, store: BaseStore) -> dict[str, Any]:
            ids = state.get("selected_tool_ids", [])
            selected_tools = [self._registered_tool(tool_id) for tool_id in ids]
            llm_with_tools = self.llm.bind_tools([self.retrieve_tools_tool(), *selected_tools])

            response = await llm_with_tools.ainvoke(state["messages"])
            return {"messages": [response]}

        self._agent_node = RunnableCallable(func=None, afunc=node)
        return self._agent_node

    def context_node(self) -> RunnableCallable:
        if self._context_node:
            return self._context_node

        async def node(state: State, config: RunnableConfig, *, store: BaseStore) -> dict[str, Any]:
            # Apply the configured context-overflow strategies as an explicit
            # graph node: create_agent runs middleware automatically, but a raw
            # StateGraph must reduce the history itself.
            reduced = await areduce_context(state["messages"], middlewares=self._context_middlewares, state=dict(state))
            if reduced is None:
                return {}

            return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *reduced]}

        self._context_node = RunnableCallable(func=None, afunc=node)
        return self._context_node

    def execute_tools_node(self) -> ToolNode:
        if self._execute_tools_node:
            return self._execute_tools_node

        self._execute_tools_node = ToolNode(list(self.tool_registry.values()))
        return self._execute_tools_node

    def select_tools_node(self) -> RunnableCallable:
        if self._select_tools_node:
            return self._select_tools_node

        async def node(tool_calls: list[dict], config: RunnableConfig, *, store: BaseStore) -> dict[str, Any]:
            messages: list[ToolMessage] = []
            tool_ids: list[str] = []
            retrieve_tool = self.retrieve_tools_tool()
            for tool_call in tool_calls:
                kwargs = {**tool_call["args"], "store": store}
                try:
                    batch = await retrieve_tool.ainvoke(kwargs, config)
                except (ToolException, ValidationError) as exc:
                    # Only genuine tool-logic failures (a ``ToolException`` or
                    # malformed call args) are surfaced back to the model as a
                    # ToolMessage. Infrastructure failures (store outage, embedding
                    # error) are not caught and propagate loudly. The except scope
                    # covers only the invocation, so a bug in the mapping below is
                    # not masked.
                    messages.append(
                        ToolMessage(
                            f"Error: {exc}",
                            tool_call_id=tool_call["id"],
                            name=self.retrieve_tools_tool_name,
                            status="error",
                        )
                    )
                    continue
                tool_names = [self._registered_tool(id_).name for id_ in batch]
                messages.append(
                    ToolMessage(
                        f"Available tools: {tool_names}",
                        tool_call_id=tool_call["id"],
                        name=self.retrieve_tools_tool_name,
                    )
                )
                tool_ids.extend(batch)

            return {"messages": messages, "selected_tool_ids": tool_ids}

        self._select_tools_node = RunnableCallable(func=None, afunc=node)
        return self._select_tools_node

    def should_continue_node(self) -> Callable[..., Any]:
        if self._should_continue_node:
            return self._should_continue_node

        def node(state: State, *, store: BaseStore) -> Any:
            last = state["messages"][-1]

            # 1) any pending tool_calls are dispatched to the matching node first.
            tool_calls = getattr(last, "tool_calls", None) or []
            if tool_calls:
                retrieve_calls = [c for c in tool_calls if c["name"] == self.retrieve_tools_tool_name]
                other_calls = [c for c in tool_calls if c["name"] != self.retrieve_tools_tool_name]
                destinations: list[Send] = []
                if retrieve_calls:
                    destinations.append(Send(self.select_tools_node_name, retrieve_calls))
                if other_calls:
                    destinations.append(Send(self.execute_tools_node_name, other_calls))
                return destinations

            # 2) no tool calls left: the message MUST be the terminal status JSON.
            # A missing or malformed terminal payload means the agent never
            # produced its required contract output — a real failure, so it
            # raises rather than silently ending the run.
            content = getattr(last, "content", "")
            if isinstance(content, list):
                texts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            texts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        texts.append(part)
                content = "".join(texts)
            content = content.strip() if isinstance(content, str) else ""

            if not content:
                raise ValueError(
                    "retrieval agent produced a terminal message with no text content; "
                    "the required status JSON was never emitted"
                )
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"retrieval agent terminal message is not valid status JSON: {content!r}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"retrieval agent terminal payload is not a JSON object: {content!r}")

            status = parsed.get("status")
            if status in ("success", "error"):
                return END
            if status == "continue":
                # No tool_calls left; loop back through the context node so the
                # history is reduced before the next agent turn.
                return self.context_node_name
            raise ValueError(
                f"retrieval agent terminal status must be one of continue/success/error, got {status!r} in {content!r}"
            )

        self._should_continue_node = node
        return self._should_continue_node

    def build(self) -> StateGraph:
        builder = StateGraph(State)
        builder.add_node(self.context_node_name, self.context_node())
        builder.add_node(self.agent_node_name, self.agent_node())
        builder.add_node(self.select_tools_node_name, self.select_tools_node())
        builder.add_node(self.execute_tools_node_name, self.execute_tools_node())

        # Every turn passes through the context node before the agent, so the
        # history is reduced (trim / summarize / context-edit) on entry and on
        # each loop-back from tool execution.
        builder.set_entry_point(self.context_node_name)
        builder.add_edge(self.context_node_name, self.agent_node_name)
        builder.add_conditional_edges(self.agent_node_name, self.should_continue_node())

        builder.add_edge(self.execute_tools_node_name, self.context_node_name)
        builder.add_edge(self.select_tools_node_name, self.context_node_name)

        return builder
