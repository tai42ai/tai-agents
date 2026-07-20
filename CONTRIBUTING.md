# Contributing to tai-agents

`tai-agents` is the reference package of generic **agents** for the TAI
ecosystem, built on the deepagents/LangGraph runtime. The hard rule (the plugin
rule): **it depends on `tai-contract` + `tai-kit` + the agent runtime
(`deepagents` / `langgraph` / `langchain-core` / `langchain` / `pydantic` /
`fastmcp` / `opentelemetry-api`) only and never imports anything outside that
allowlist — in particular, never the skeleton.** Every agent registers through the
`tai_app` handle from `tai_contract.app` and is loaded by the host from the
manifest (`agents[].module`) by dynamic import — there is no import edge to the
skeleton in either direction.

## Ground rules

- **Imports stay on the allowlist.** The package is contract-facing: it
  imports `tai-contract` + `tai-kit` + the agent runtime only and never imports
  `tai-skeleton`. The rule is enforced by ruff (`flake8-tidy-imports`) and by the
  import-graph test, which walks every shipped module and fails lint and CI on
  any root outside the allowlist:
  ```bash
  grep -rn "tai_skeleton" src/   # must be empty
  ```
- **No model-provider SDK dependencies.** Model access goes through tai-kit's
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

- `src/tai_agents/` — the agent modules, each registering its agent through
  `@tai_app.agents.agent(name)`.
- `src/tai_agents/_internal/` — private helpers shared by the agent modules;
  nothing here registers through `tai_app`.
- `tests/` mirrors `src/`.

## Dev

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

For local cross-repo work, `make dev` editable-installs the sibling `tai-*`
checkouts this package builds on into the venv. While `[tool.uv.sources]` pins
those siblings to local paths, `uv sync` already installs them editable and
`make dev` changes nothing; once the lock resolves them from the registry,
`uv sync` / `uv run` installs the published builds instead, so re-run
`make dev` afterward to restore the editable links.

Before any commit, run a secret scan over `src/` and `tests/` (e.g.
`detect-secrets scan`).

## License

By contributing you agree your contributions are licensed under Apache-2.0.
