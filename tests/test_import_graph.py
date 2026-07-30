"""Import-graph guard for the shipped package.

Two complementary walks assert the same rule: every import root reachable from
``tai42_agents`` is on the allowlist. The rule (see the README): the shipped
package imports ``tai42-contract`` + ``tai42-kit`` + the agent runtime (deepagents /
langgraph / langchain-core / langchain / langchain-anthropic / pydantic /
fastmcp / mcp / opentelemetry) and their dependency closure ONLY, plus the
Python standard library. Anything else -- ``tai42-skeleton`` (which sits a layer
above and must never be pulled in) or any package that is not a declared
dependency of the shipped wheel -- is absent from the allowlist and fails the
test loudly.

The runtime walk imports ``tai42_agents`` and every submodule in a fresh
subprocess, then inspects ``sys.modules``. Running it in a subprocess that
imports ONLY ``tai42_agents`` means the assertion covers the SHIPPED package's
true import closure and never observes roots that a sibling test module or a
conftest pulled into this process's global ``sys.modules`` (e.g. a future
integration test that takes tai42-skeleton as a dev dependency). A submodule that
fails to import raises loudly and fails the test too.

The static walk parses every shipped source file and collects import roots at
ANY nesting depth. This is what catches an import that the runtime walk cannot
see: one placed inside a function body, a class body, or a ``TYPE_CHECKING``
block never executes on a plain package import, so it would leave no trace in
``sys.modules``. Both walks share one allowlist, so neither is a weaker gate.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

# The shipped package and the public first-party packages it may import.
PACKAGE = "tai42_agents"
ALLOWED_FIRST_PARTY = frozenset({PACKAGE, "tai42_contract", "tai42_kit"})

# The agent runtime and its full dependency closure -- every third-party root
# the shipped ``tai42_agents`` graph pulls in. This mirrors the resolved runtime
# environment: adding a runtime dependency that brings a new root means adding
# that root here, but only when it is a genuine dependency of the shipped
# package -- the walks below are never widened just to make the test pass.
ALLOWED_THIRD_PARTY = frozenset(
    {
        "annotated_types",
        "anthropic",
        "anyio",
        "attr",
        "attrs",
        "beartype",
        "bracex",
        "cachetools",
        "certifi",
        "charset_normalizer",
        "click",
        "deepagents",
        "distro",
        "docstring_parser",
        "dotenv",
        "exceptiongroup",
        "fastmcp",
        "filetype",
        "httpx",
        "httpx_sse",
        "idna",
        "jiter",
        "jsonpatch",
        "jsonpointer",
        "jsonschema",
        "jsonschema_specifications",
        "key_value",
        "langchain",
        "langchain_anthropic",
        "langchain_core",
        "langchain_protocol",
        "langgraph",
        "langgraph_sdk",
        "langsmith",
        "mcp",
        "opentelemetry",
        "orjson",
        "ormsgpack",
        "packaging",
        "platformdirs",
        "pydantic",
        "pydantic_core",
        "pydantic_settings",
        "pygments",
        "python_multipart",
        "referencing",
        "requests",
        "requests_toolbelt",
        "rfc3339_validator",
        "rich",
        "rpds",
        "ruamel",
        "six",
        "sniffio",
        "sse_starlette",
        "starlette",
        "tenacity",
        "typing_extensions",
        "typing_inspection",
        "urllib3",
        "uuid_utils",
        "uvicorn",
        "watchfiles",
        "wcmatch",
        "websockets",
        "xxhash",
        "yaml",
        "zstandard",
    }
)

# Interpreter, compiler, and virtual-env roots that land in ``sys.modules`` as
# ambient side effects of importing compiled extensions or running under a
# virtual environment. They are not dependency packages, and their exact names
# are build/platform/version specific (a mypyc module group is hash-named, the
# cython runtime carries its version, sysconfigdata carries the platform), so
# they are matched by shape, never by literal.
_ARTIFACT_ROOTS = frozenset({"__main__", "__mp_main__", "cython_runtime", "_virtualenv"})


def _is_runtime_artifact(root: str) -> bool:
    return root in _ARTIFACT_ROOTS or root.endswith("__mypyc") or root.startswith(("_cython_", "_sysconfigdata"))


def _allowed(root: str) -> bool:
    return (
        root in sys.stdlib_module_names
        or root in ALLOWED_FIRST_PARTY
        or root in ALLOWED_THIRD_PARTY
        or _is_runtime_artifact(root)
    )


# Program run in the subprocess: bind a stub app to the ``tai42_app`` handle (the
# agent modules register through ``tai42_app.agents.agent(name)`` at import time,
# so the handle must be bound first, exactly as the host binds it before importing
# a manifest module), import tai42_agents and every submodule, then print each
# imported root that is NOT on the allowlist. A submodule that fails to import
# propagates as an uncaught exception, giving a non-zero exit the parent turns
# into a loud failure.
_CHILD_PROGRAM = f"""
import importlib
import pkgutil
import sys

from tai42_contract.app import tai42_app

PACKAGE = {PACKAGE!r}
ALLOWED_FIRST_PARTY = {set(ALLOWED_FIRST_PARTY)!r}
ALLOWED_THIRD_PARTY = {set(ALLOWED_THIRD_PARTY)!r}
_ARTIFACT_ROOTS = {set(_ARTIFACT_ROOTS)!r}


def _is_runtime_artifact(root):
    return (
        root in _ARTIFACT_ROOTS
        or root.endswith("__mypyc")
        or root.startswith(("_cython_", "_sysconfigdata"))
    )


def _allowed(root):
    return (
        root in sys.stdlib_module_names
        or root in ALLOWED_FIRST_PARTY
        or root in ALLOWED_THIRD_PARTY
        or _is_runtime_artifact(root)
    )


class _StubAgents:
    def agent(self, name, tags=None):
        def decorator(agent_cls):
            return agent_cls

        return decorator


class _StubApp:
    agents = _StubAgents()


tai42_app.bind(_StubApp())

package = importlib.import_module(PACKAGE)
for module_info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
    importlib.import_module(module_info.name)

offenders = sorted(
    root for root in {{name.partition(".")[0] for name in sys.modules}} if not _allowed(root)
)
for name in offenders:
    print(name)
"""


def _source_root() -> Path:
    root = Path(__file__).resolve().parents[1] / "src" / PACKAGE
    assert root.is_dir(), f"shipped package source not found at {root}"
    return root


def _static_import_roots() -> dict[str, set[str]]:
    """Map each import root in the shipped sources to the files that import it.

    Walks the full AST of every source file, so an import nested inside a
    function body, a class body, or a conditional block is collected exactly
    like a module-level one. Relative imports address the shipped package
    itself and carry no root to check.
    """
    roots: dict[str, set[str]] = {}
    source_root = _source_root()
    paths = sorted(source_root.rglob("*.py"))
    assert paths, f"no source files found under {source_root}"
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.level == 0 and node.module else []
            else:
                continue
            for name in names:
                roots.setdefault(name.partition(".")[0], set()).add(str(path.relative_to(source_root)))
    return roots


def test_shipped_package_imports_only_allowlisted_roots() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"importing the shipped tai42_agents graph failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    offenders = [line for line in result.stdout.splitlines() if line]
    assert offenders == [], f"non-allowlisted roots in the tai42_agents module graph: {offenders}"


def test_shipped_sources_name_only_allowlisted_roots() -> None:
    offenders = {root: sorted(files) for root, files in _static_import_roots().items() if not _allowed(root)}
    assert offenders == {}, f"non-allowlisted import roots in the tai42_agents sources: {offenders}"
