# Contributing to tai42-agents

`tai42-agents` is the reference package of generic **agents** for the TAI
ecosystem, built on the deepagents/LangGraph runtime. The hard rule (the plugin
rule): **it depends on `tai42-contract` + `tai42-kit` + the agent runtime
(`deepagents` / `langgraph` / `langchain-core` / `langchain` / `pydantic` /
`fastmcp` / `opentelemetry-api`) only and never imports anything outside that
allowlist — in particular, never the skeleton.** Every agent registers through the
`tai42_app` handle from `tai42_contract.app` and is loaded by the host from the
manifest (`agents[].module`) by dynamic import — there is no import edge to the
skeleton in either direction.

## Ground rules

- **Imports stay on the allowlist.** The package is contract-facing: it
  imports `tai42-contract` + `tai42-kit` + the agent runtime only and never imports
  `tai42-skeleton`. The rule is enforced by ruff (`flake8-tidy-imports`) and by the
  import-graph test, which walks every shipped module and fails lint and CI on
  any root outside the allowlist:
  ```bash
  grep -rn "tai42_skeleton" src/   # must be empty
  ```
- **No model-provider SDK dependencies.** Model access goes through tai42-kit's
  llm factories, configured per deployment — never a direct provider SDK dep.
- **Contract fidelity.** An agent's `astream` yields the contract's
  `StreamEvent` taxonomy correctly for what the agent actually does; `run`
  drains to the final value per the contract's terminal rule.
- **Loud errors.** No swallowed exceptions, silent fallbacks, or silent
  truncation. A bound exceeded, a missing input, or a failed sub-step raises.
- **Deterministic tests.** Agent tests use fake/scripted LLMs and assert event
  sequences; no live-LLM tests in CI.
- **Typed package** (`py.typed`). Pyright runs clean.

## Layout

- `src/tai42_agents/` — the agent modules, each registering its agent through
  `@tai42_app.agents.agent(name)`.
- `src/tai42_agents/_internal/` — private helpers shared by the agent modules;
  nothing here registers through `tai42_app`.
- `tests/` mirrors `src/`.

## Naming

PyPI is a flat namespace with no owner in the path, so distributions carry the
`tai42-` prefix. GitHub repositories keep their `tai-` names, because the
`tai42ai` organisation already namespaces them. Import packages follow the
distribution.

| Surface | Form |
| --- | --- |
| Distribution — PyPI, `pip install`, dependency pins | `tai42-<name>` |
| Import package | `tai42_<name>` |
| GitHub repository | `tai-<name>` |

So a dependency is declared as `tai42-<name>` while its repository is named
`tai-<name>`, and both spellings are correct in their own context.

Some surfaces are deliberately neither, and must not be renamed: the `tai` CLI
command (`tai42` is an alias), the Prometheus metric namespace (`tai_tool_*`),
`TAI_*` environment variables, and the `tai-plugin.yml` descriptor filename.

## Dev

```bash
uv venv --python 3.13
uv pip install --no-sources --editable ".[dev]"
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest --cov --cov-report=term-missing
```

`make dev` installs the sibling `tai-contract`, `tai-kit`, and `tai-skeleton` repos as editable installs for local cross-repo development.

Before any commit, run a secret scan over `src/` and `tests/` (e.g.
`detect-secrets scan`).

## License

By contributing you agree your contributions are licensed under Apache-2.0.
