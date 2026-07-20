"""tai-plugin.yml: the shipped plugin spec validates and stays in sync."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import yaml
from tai42_contract.plugins import PluginSpec

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ROOT_SPEC = _REPO_ROOT / "tai-plugin.yml"
_PACKAGED_SPEC = _REPO_ROOT / "src" / "tai42_agents" / "tai-plugin.yml"


def _spec() -> PluginSpec:
    return PluginSpec.model_validate(yaml.safe_load(_ROOT_SPEC.read_text(encoding="utf-8")))


def test_plugin_spec_validates_and_names_this_listing():
    spec = _spec()
    assert spec.ref == "tai42/agents"
    for item in spec.provides:
        assert importlib.util.find_spec(item.module) is not None, (
            f"provides item {item.kind.value}/{item.name} names module {item.module!r}, which does not resolve"
        )


def test_plugin_spec_matches_the_project_metadata():
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    spec = _spec()
    assert spec.package == project["name"]
    assert spec.version == project["version"]
    assert spec.description == project["description"]


def test_packaged_spec_is_declared_in_package_data():
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    owners = [package for package, patterns in package_data.items() if "tai-plugin.yml" in patterns]
    assert owners == ["tai42_agents"], (
        "exactly one package-data entry must ship tai-plugin.yml so the wheel includes the spec; "
        f"found owners {owners!r}"
    )


def test_packaged_copy_is_identical_to_the_root_spec():
    assert _PACKAGED_SPEC.read_bytes() == _ROOT_SPEC.read_bytes()
