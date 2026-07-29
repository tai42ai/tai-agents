# tai42-agents

[![CI](https://github.com/tai42ai/tai-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/tai42ai/tai-agents/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

The reference agents package for the TAI ecosystem — an opt-in, manifest-loaded
collection of generic **agents** built on the deepagents/LangGraph runtime.

Every agent here registers through the `tai42_app` handle from `tai42_contract.app`
and is loaded by the host from the manifest (`agents[].module`). Its only tai-*
dependencies are `tai42-contract` (the `Agent` ABC, the `StreamEvent` taxonomy,
and the `tai42_app` handle it registers through) and `tai42-kit` (settings
machinery and the llm factories model access goes through). It **never**
imports the skeleton — tai42-agents is contract-facing.

## The TAI ecosystem

TAI is an open-source runtime for MCP tools, agents, and workflows. An agent is a
capability the runtime hosts and exposes as a tool; the seven here are the
platform's ready-made, batteries-included set. This repo is their per-agent
reference doc home; the documentation site covers using them and the
platform-level story:

- Use the ready-made agents (guide): https://tai42.ai/guides/use-the-ready-made-agents
- Agents concept: https://tai42.ai/concepts/agents
- Ecosystem catalog: https://tai42.ai/reference/catalog

## Install

Requires **Python 3.13+**. Install from PyPI into the environment that runs the
server:

```bash
uv add tai42-agents
```

Or from source — clone this repo and add it as an editable dependency. Clone
`tai-contract`, `tai-kit`, and `tai-skeleton` beside this repo first —
`[tool.uv.sources]` resolves them from sibling paths.

```bash
git clone https://github.com/tai42ai/tai-agents   # next to your app checkout
cd /path/to/your/app
uv add --editable ../tai-agents
```

The agent runtime (`deepagents`, `langgraph`, `langchain-core`, `langchain`,
`pydantic`, `pydantic-settings`, `fastmcp`, `opentelemetry-api`, `wcmatch`) is a
base dependency — agents are this package's purpose, so there is no runtime extra
to opt into. Model-provider SDKs are **never** direct dependencies here: model
access goes through tai42-kit's llm factories, configured per deployment.

## Registering an agent

An agent is a class subclassing the contract `Agent` ABC, registered under a
name with the `@tai42_app.agents.agent(name)` decorator. Registration fires when
the module imports (import-to-register); the host imports the module because
the manifest names it:

```yaml
agents:
  - title: my-agents
    module: tai42_agents.<module>
    # include: [<agent name>]   # optional — omit to expose all agents in the module
```

```python
from pydantic import BaseModel
from tai42_contract.agent import Agent
from tai42_contract.app import tai42_app


class EchoInput(BaseModel):
    user_message: str = ""


@tai42_app.agents.agent("echo")
class EchoAgent(Agent):
    tool_name = "echo"
    tool_description = "Echoes the user message back."
    ToolInput = EchoInput  # a JSON-able pydantic model of the tool params

    async def run(self, *, user_message: str = "", **kwargs):
        return user_message
```

Registration gives each agent two faces, both derived from the one class:

- an in-process `astream` method (API / SSE facing) that yields the contract's
  `StreamEvent` taxonomy (`ReasoningStep`, `ToolCallStep`/`ToolResultStep`,
  `MessageDelta`, `MessageFinal`, `RunUsage`, `StructuredFinal`,
  `InterruptFinal`);
- an auto-generated JSON `run` tool (LLM / MCP / flow-engine facing) whose
  signature is the agent's `ToolInput` model.

## Agents

The package ships seven agents, each in its own module so a manifest can load
exactly the ones a deployment wants:

- **`tools_agent`** (`tai42_agents.tools_agent`) — the plain/advanced LangGraph
  tools agent. Uniform tool inputs: `tool_names` (client tools resolved through
  the app registry), live `tools`, and `presets` (a base tool bound to fixed
  kwargs; a sub-flow is `base_tool="flow"` with `fixed_kwargs={"flow_graph": ...}`).
- **`deep_agent`** (`tai42_agents.deep_agent`) — a deepagents-harness agent:
  planning, a per-thread scratch filesystem, skills (served live from the
  template provider or supplied inline), one level of nested subagents, and
  human-in-the-loop interrupts with resume via a LangGraph `Command`. Both faces
  fail loudly when a requested `response_format` produces no structured output:
  the invoke face raises on drain, and the stream face raises after the stream
  drains (a pending interrupt takes precedence over the raise).
- **`retrieval_tools_agent`** (`tai42_agents.retrieval_tools_agent`) — a tools
  agent that does not bind every tool to the model up front: it embeds each
  tool's description into a vector store and exposes a `retrieve_tools`
  semantic-search tool, binding matches on demand until the model emits a
  terminal `{"status": ...}` object. Useful when the tool set is large.
- **`mcp_tools_agent`** (`tai42_agents.mcp_tools_agent`) — a tools agent whose
  tools come from an MCP server: it opens a `fastmcp` client from a caller's
  `mcpServers` config, converts those tools to LangChain tools, and runs with the
  client held open. With `inject_env=True`, only the environment variable names
  listed in `env_allowlist` are copied from `os.environ` into each server's `env`
  (the server's own `env` wins on conflict); `inject_env=True` with an empty or
  missing `env_allowlist` is a malformed request and raises `ValueError` rather
  than silently injecting nothing. `mcp_tools_agent` is admin-curated — expose it
  ONLY to trusted, access-controlled callers/agents, NEVER to an agent that
  processes untrusted content.
- **`voting_agent`** (`tai42_agents.voting_agent`) — runs N voter LLMs in parallel
  over one prompt, then a judge LLM decides by majority vote (breaking ties with
  its own reasoning). Returns a `VotingOutput`; only the judge streams.
- **`refine_agent`** (`tai42_agents.refine_agent`) — an Evaluator↔Critic loop: the
  evaluator drafts, the critic reviews, and they alternate until the critic emits
  the approval token or the iteration budget is exhausted (a loud `RuntimeError`,
  never an unapproved draft). Only the final approved pass streams.
- **`vqa_agent`** (`tai42_agents.vqa_agent`) — visual question answering: a single
  multimodal completion over an `image_url` and a `query`. No tools, no graph.

**Expose kwargs-carrying agents to trusted callers only.** `base_url`/`api_key`
in `llm_kwargs`/`embedding_kwargs` legitimately route to a caller-chosen
model/embedding endpoint; expose any agent or tool carrying these kwargs only to
trusted callers — an injected parent agent could redirect the model/embedding
call to a hostile endpoint and leak the key/context.

Wire the ones you want in the manifest — one `agents:` entry per module, each
with a `title`; add `include:` to expose a subset of a module's agents:

```yaml
agents:
  - title: tools-agent
    module: tai42_agents.tools_agent
  - title: deep-agent
    module: tai42_agents.deep_agent
  - title: retrieval-tools-agent
    module: tai42_agents.retrieval_tools_agent
  - title: mcp-tools-agent
    module: tai42_agents.mcp_tools_agent
  - title: voting-agent
    module: tai42_agents.voting_agent
  - title: refine-agent
    module: tai42_agents.refine_agent
  - title: vqa-agent
    module: tai42_agents.vqa_agent
```

## Import rule

The shipped `tai42_agents` package imports `tai42-contract`, `tai42-kit`, and the
agent runtime (`deepagents` / `langgraph` / `langchain-core` / `langchain` /
`pydantic` / `fastmcp` / `opentelemetry` / `wcmatch`) — the declared
dependencies **and their resolved dependency closure** — plus the standard
library. It never imports `tai42-skeleton`, which sits a layer above, and never
reaches for a package that is not a dependency of the shipped wheel. The rule is
enforced twice: ruff (`flake8-tidy-imports` bans) fails lint on a skeleton
import, and an import-graph test asserts every root in the module graph is on
the allowlist — once by importing every shipped module in a fresh subprocess and
inspecting `sys.modules`, and once by parsing every shipped source file, so an
import nested in a function body, a class body, or a `TYPE_CHECKING` block is
caught too.

## Development

```bash
uv venv --python 3.13
uv pip install --no-sources --editable ".[dev]"
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest --cov --cov-report=term-missing
```

**Dependency install is plain.** The deepagents/LangGraph stack (`deepagents`,
`langgraph`, `langchain-core`, `langchain`, `pydantic`, `opentelemetry-api`,
`fastmcp`) resolves as ordinary wheels from the index — no vendor directory and
no `PYTHONPATH` bridge.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
