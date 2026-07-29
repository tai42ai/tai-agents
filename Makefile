# Link sibling checkouts into the venv as editable installs for cross-repo development.
# While [tool.uv.sources] pins the siblings to local paths this is a no-op —
# `uv sync` already installs them editable. Once the lock resolves them from the
# registry, re-run this after ANY `uv sync` / `uv run` to restore the links.
.PHONY: dev
dev:
	uv pip install -e ../tai-contract -e ../tai-kit -e ../tai-skeleton
