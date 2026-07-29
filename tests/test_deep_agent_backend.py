"""Tests for build_backend — composite shape + live template-backed skills.

Skills are served read-only from the app's template provider. These tests drive
an in-memory fake template manager (same surface as
``tai42_app.storage.resource_manager``: ``list_resources`` / ``fetch_template`` /
``upload_template``) bound onto the recording app's storage facet, and prove a
skill uploaded into the provider is read back through the composite backend, and
that every write path raises.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from deepagents.backends import CompositeBackend, StateBackend
from tai42_contract.app import tai42_app

from tai42_agents.deep_agent.backend import (
    SKILLS_ROOT,
    InlineSkillsBackend,
    TemplateSkillsBackend,
    build_backend,
)


class _TemplateMissing(Exception):
    """Raised by the fake provider for an absent template id.

    Mirrors the real provider, whose ``fetch_template`` returns ``str`` (``""``
    for an empty body) or raises for a missing key — it never returns ``None``.
    """


class _FakeResourceManager:
    """In-memory stand-in for ``tai42_app.storage.resource_manager``.

    Stores raw key -> content. ``fetch_template`` is the RAW (non-jinja) read the
    backend uses; like the real provider it returns the stored text (``""`` for
    an empty body) or raises :class:`_TemplateMissing` for a missing key.
    ``list_resources`` returns flat keys like the real providers (no leading
    slash). ``phantom_keys`` are listed but absent from the store, modelling a key
    that vanishes between the listing and the fetch.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.phantom_keys: set[str] = set()

    async def upload_template(self, path: str, content: str) -> None:
        self.store[path] = content

    async def list_resources(self) -> list[str]:
        return list(self.store) + [key for key in self.phantom_keys if key not in self.store]

    async def fetch_template(self, template_id: str) -> str:
        try:
            return self.store[template_id]
        except KeyError as exc:
            raise _TemplateMissing(f"Template '{template_id}' not found.") from exc


@pytest.fixture
def fake_tm(monkeypatch: pytest.MonkeyPatch) -> _FakeResourceManager:
    tm = _FakeResourceManager()
    monkeypatch.setattr(tai42_app.storage, "resource_manager", tm)
    return tm


def test_build_backend_shape() -> None:
    backend = build_backend()
    assert isinstance(backend, CompositeBackend)
    assert isinstance(backend.default, StateBackend)
    assert SKILLS_ROOT in backend.routes
    assert isinstance(backend.routes[SKILLS_ROOT], TemplateSkillsBackend)


def test_skill_uploaded_to_provider_reads_back_through_backend(fake_tm: _FakeResourceManager) -> None:
    """A skill uploaded under the ``skills/`` key convention is read live back
    through the composite backend at its ``/skills/<name>/SKILL.md`` path."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "hello")
        backend = build_backend()
        return await backend.aread(f"{SKILLS_ROOT}jq/SKILL.md")

    res = asyncio.run(go())
    assert res.error is None
    assert res.file_data["content"] == "hello"


def test_read_missing_skill_reports_not_found(fake_tm: _FakeResourceManager) -> None:
    async def go() -> Any:
        return await build_backend().aread(f"{SKILLS_ROOT}missing/SKILL.md")

    res = asyncio.run(go())
    assert res.error is not None
    assert "not found" in res.error


def test_als_lists_skill_dirs_for_middleware(fake_tm: _FakeResourceManager) -> None:
    """``als`` returns one is_dir entry per skill dir whose SKILL.md exists, so
    the skills middleware (source ``/skills/<name>/``) finds and downloads it."""

    async def go() -> tuple[Any, Any]:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        await fake_tm.upload_template("skills/flow/SKILL.md", "y")
        backend = build_backend()
        whole = await backend.als(SKILLS_ROOT)
        single = await backend.als(f"{SKILLS_ROOT}jq/")
        return whole, single

    whole, single = asyncio.run(go())
    assert {e["path"] for e in whole.entries} == {f"{SKILLS_ROOT}jq/", f"{SKILLS_ROOT}flow/"}
    assert all(e["is_dir"] for e in whole.entries)
    assert [e["path"] for e in single.entries] == [f"{SKILLS_ROOT}jq/"]


def test_listing_filters_non_skill_keys(fake_tm: _FakeResourceManager) -> None:
    """Only ``skills/<name>/SKILL.md`` keys are skills; other provider keys
    (wrong prefix, or a non-SKILL.md file under skills/) are ignored."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        await fake_tm.upload_template("skills/jq/README.md", "ignore")
        await fake_tm.upload_template("other/stuff", "ignore")
        return await build_backend().als(SKILLS_ROOT)

    result = asyncio.run(go())
    assert [e["path"] for e in result.entries] == [f"{SKILLS_ROOT}jq/"]


def test_download_files_returns_raw_and_marks_missing(fake_tm: _FakeResourceManager) -> None:
    async def go() -> list[Any]:
        await fake_tm.upload_template("skills/jq/SKILL.md", "body")
        skills = build_backend().routes[SKILLS_ROOT]
        return await skills.adownload_files(["/jq/SKILL.md", "/nope/SKILL.md"])

    found, missing = asyncio.run(go())
    assert found.content == b"body"
    assert found.error is None
    assert missing.content is None
    assert missing.error == "file_not_found"


def test_writes_raise(fake_tm: _FakeResourceManager) -> None:
    skills = TemplateSkillsBackend()

    async def go() -> None:
        with pytest.raises(PermissionError, match="read-only"):
            await skills.awrite("/jq/SKILL.md", "x")
        with pytest.raises(PermissionError, match="read-only"):
            await skills.aedit("/jq/SKILL.md", "a", "b")
        with pytest.raises(PermissionError, match="read-only"):
            await skills.aupload_files([("/jq/SKILL.md", b"x")])

    asyncio.run(go())


def test_sync_methods_raise_not_implemented(fake_tm: _FakeResourceManager) -> None:
    """The backend is async-only; every sync read entrypoint raises loudly so a
    sync caller fails fast instead of silently re-entering the event loop."""
    skills = TemplateSkillsBackend()
    with pytest.raises(NotImplementedError, match="async-only"):
        skills.read("/jq/SKILL.md")
    with pytest.raises(NotImplementedError, match="async-only"):
        skills.ls("/")
    with pytest.raises(NotImplementedError, match="async-only"):
        skills.glob("*")
    with pytest.raises(NotImplementedError, match="async-only"):
        skills.grep("x")
    with pytest.raises(NotImplementedError, match="async-only"):
        skills.download_files(["/jq/SKILL.md"])


def test_skill_body_is_not_jinja_rendered(fake_tm: _FakeResourceManager) -> None:
    """Skill markdown is read raw — jinja-looking braces survive verbatim."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "use {{ not_rendered }} here")
        return await build_backend().aread(f"{SKILLS_ROOT}jq/SKILL.md")

    res = asyncio.run(go())
    assert res.file_data["content"] == "use {{ not_rendered }} here"


def test_skill_reflects_provider_overwrite(fake_tm: _FakeResourceManager) -> None:
    """Re-uploading a skill under the same key is read back as the new content —
    the backend serves the provider live, so an overwrite is reflected with no
    re-seed or cache to clear."""

    async def go() -> tuple[Any, Any]:
        await fake_tm.upload_template("skills/jq/SKILL.md", "VERSION ONE")
        first = await build_backend().aread(f"{SKILLS_ROOT}jq/SKILL.md")
        await fake_tm.upload_template("skills/jq/SKILL.md", "VERSION TWO")
        second = await build_backend().aread(f"{SKILLS_ROOT}jq/SKILL.md")
        return first, second

    first, second = asyncio.run(go())
    assert first.file_data["content"] == "VERSION ONE"
    assert second.file_data["content"] == "VERSION TWO"


# --- inline skills overlay -------------------------------------------------


def test_build_backend_routes_inline_skills_backend(fake_tm: _FakeResourceManager) -> None:
    backend = build_backend(inline_skills={"demo": "x"})
    assert isinstance(backend.routes[SKILLS_ROOT], InlineSkillsBackend)


def test_build_backend_empty_inline_skills_uses_template_backend(fake_tm: _FakeResourceManager) -> None:
    """An empty/None inline map keeps the plain template backend (no overlay)."""
    assert isinstance(build_backend(None).routes[SKILLS_ROOT], TemplateSkillsBackend)
    assert isinstance(build_backend({}).routes[SKILLS_ROOT], TemplateSkillsBackend)


def test_inline_skill_reads_back_through_backend(fake_tm: _FakeResourceManager) -> None:
    """An inline skill's content is served at its /skills/<name>/SKILL.md path."""

    async def go() -> Any:
        backend = build_backend(inline_skills={"demo": "INLINE BODY"})
        return await backend.aread(f"{SKILLS_ROOT}demo/SKILL.md")

    res = asyncio.run(go())
    assert res.error is None
    assert res.file_data["content"] == "INLINE BODY"


def test_inline_skill_body_is_not_jinja_rendered(fake_tm: _FakeResourceManager) -> None:
    async def go() -> Any:
        backend = build_backend(inline_skills={"demo": "use {{ raw }} here"})
        return await backend.aread(f"{SKILLS_ROOT}demo/SKILL.md")

    res = asyncio.run(go())
    assert res.file_data["content"] == "use {{ raw }} here"


def test_inline_and_template_skills_coexist(fake_tm: _FakeResourceManager) -> None:
    """Inline names overlay the template store; listings and reads cover both."""

    async def go() -> tuple[Any, Any, Any]:
        await fake_tm.upload_template("skills/jq/SKILL.md", "TEMPLATE JQ")
        backend = build_backend(inline_skills={"demo": "INLINE DEMO"})
        listed = await backend.als(SKILLS_ROOT)
        inline_read = await backend.aread(f"{SKILLS_ROOT}demo/SKILL.md")
        template_read = await backend.aread(f"{SKILLS_ROOT}jq/SKILL.md")
        return listed, inline_read, template_read

    listed, inline_read, template_read = asyncio.run(go())
    assert {e["path"] for e in listed.entries} == {f"{SKILLS_ROOT}jq/", f"{SKILLS_ROOT}demo/"}
    assert inline_read.file_data["content"] == "INLINE DEMO"
    assert template_read.file_data["content"] == "TEMPLATE JQ"


def test_inline_skill_shadows_template_skill_of_same_name(fake_tm: _FakeResourceManager) -> None:
    """When a name exists in both, the inline content wins (overlay)."""

    async def go() -> tuple[Any, Any]:
        await fake_tm.upload_template("skills/jq/SKILL.md", "FROM TEMPLATE")
        backend = build_backend(inline_skills={"jq": "FROM INLINE"})
        listed = await backend.als(SKILLS_ROOT)
        read = await backend.aread(f"{SKILLS_ROOT}jq/SKILL.md")
        return listed, read

    listed, read = asyncio.run(go())
    assert [e["path"] for e in listed.entries] == [f"{SKILLS_ROOT}jq/"]
    assert read.file_data["content"] == "FROM INLINE"


def test_inline_shadow_skips_vanished_template_fetch(fake_tm: _FakeResourceManager) -> None:
    """An inline skill shadowing a template skill whose key has vanished from the
    store must not fetch that template during ``agrep``: the inline body overrides
    it, so the vanished template is irrelevant and its absence must not raise. The
    shadow-skip runs before the fetch, so ``agrep`` returns the inline match."""

    async def go() -> Any:
        fake_tm.phantom_keys.add("skills/jq/SKILL.md")
        skills = build_backend(inline_skills={"jq": "INLINE"}).routes[SKILLS_ROOT]
        return await skills.agrep("INLINE")

    result = asyncio.run(go())
    assert {match["path"] for match in result.matches} == {"/jq/SKILL.md"}


def test_inline_download_files_serves_inline_and_delegates(fake_tm: _FakeResourceManager) -> None:
    """download_files returns inline content for inline names and delegates the rest."""

    async def go() -> list[Any]:
        await fake_tm.upload_template("skills/jq/SKILL.md", "TEMPLATE JQ")
        skills = build_backend(inline_skills={"demo": "INLINE DEMO"}).routes[SKILLS_ROOT]
        return await skills.adownload_files(["/demo/SKILL.md", "/jq/SKILL.md", "/missing/SKILL.md"])

    responses = asyncio.run(go())
    inline, template, missing = responses
    # Order is preserved one-to-one with the requested paths.
    assert [r.path for r in responses] == ["/demo/SKILL.md", "/jq/SKILL.md", "/missing/SKILL.md"]
    assert inline.content == b"INLINE DEMO"
    assert inline.error is None
    assert template.content == b"TEMPLATE JQ"
    assert template.error is None
    assert missing.content is None
    assert missing.error == "file_not_found"


def test_inline_aglob_merges_inline_and_template(fake_tm: _FakeResourceManager) -> None:
    """aglob returns the union of inline and template skill SKILL.md paths."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        skills = build_backend(inline_skills={"demo": "y"}).routes[SKILLS_ROOT]
        return await skills.aglob("*/SKILL.md")

    result = asyncio.run(go())
    assert {m["path"] for m in result.matches} == {"/jq/SKILL.md", "/demo/SKILL.md"}


def test_inline_aglob_exact_file_base(fake_tm: _FakeResourceManager) -> None:
    """The inline overlay honors a ``path`` naming an exact inline skill file (the
    exact-file base branch over the inline path set), scoping to that one file."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        skills = build_backend(inline_skills={"demo": "y"}).routes[SKILLS_ROOT]
        return await skills.aglob("SKILL.md", path="/demo/SKILL.md")

    result = asyncio.run(go())
    assert {m["path"] for m in result.matches} == {"/demo/SKILL.md"}


def test_inline_aglob_directory_base_with_and_without_trailing_slash(fake_tm: _FakeResourceManager) -> None:
    """The inline overlay scopes to a directory ``path`` and normalizes a trailing
    slash, so ``/demo`` and ``/demo/`` match the inline skill identically while an
    out-of-scope template skill is filtered out."""

    async def go() -> tuple[Any, Any]:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        skills = build_backend(inline_skills={"demo": "y"}).routes[SKILLS_ROOT]
        no_slash = await skills.aglob("*.md", path="/demo")
        with_slash = await skills.aglob("*.md", path="/demo/")
        return no_slash, with_slash

    no_slash, with_slash = asyncio.run(go())
    assert {m["path"] for m in no_slash.matches} == {"/demo/SKILL.md"}
    assert {m["path"] for m in with_slash.matches} == {"/demo/SKILL.md"}


def test_inline_aglob_whitespace_only_path_is_empty(fake_tm: _FakeResourceManager) -> None:
    """A whitespace-only ``path`` yields no matches on the inline overlay either —
    both the template delegate and the inline path set reject it."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        skills = build_backend(inline_skills={"demo": "y"}).routes[SKILLS_ROOT]
        return await skills.aglob("*", path="   ")

    result = asyncio.run(go())
    assert result.matches == []


def test_inline_aglob_directory_base_no_match(fake_tm: _FakeResourceManager) -> None:
    """A directory ``path`` whose pattern matches nothing returns empty on the inline
    overlay."""

    async def go() -> Any:
        skills = build_backend(inline_skills={"demo": "y"}).routes[SKILLS_ROOT]
        return await skills.aglob("*.txt", path="/demo")

    result = asyncio.run(go())
    assert result.matches == []


def test_inline_agrep_shadows_template_and_includes_inline(fake_tm: _FakeResourceManager) -> None:
    """agrep searches inline content and the template skills it does not shadow;
    a template skill shadowed by an inline name of the same name is not searched."""

    async def go() -> tuple[Any, Any, Any]:
        await fake_tm.upload_template("skills/jq/SKILL.md", "TEMPLATE_MARKER")
        await fake_tm.upload_template("skills/keep/SKILL.md", "KEEP_MARKER")
        # 'jq' inline shadows the template 'jq'; its body carries no marker.
        skills = build_backend(inline_skills={"jq": "INLINE_MARKER"}).routes[SKILLS_ROOT]
        shadowed = await skills.agrep("TEMPLATE_MARKER")
        inline_hit = await skills.agrep("INLINE_MARKER")
        unshadowed = await skills.agrep("KEEP_MARKER")
        return shadowed, inline_hit, unshadowed

    shadowed, inline_hit, unshadowed = asyncio.run(go())
    assert shadowed.matches == []
    assert {m["path"] for m in inline_hit.matches} == {"/jq/SKILL.md"}
    assert {m["path"] for m in unshadowed.matches} == {"/keep/SKILL.md"}


def test_inline_agrep_propagates_vanished_listed_key(fake_tm: _FakeResourceManager) -> None:
    """A template key the provider lists but which is absent from the store
    propagates the provider's own error out of the inline overlay's agrep, rather
    than being silently dropped from the search corpus."""

    async def go() -> None:
        fake_tm.phantom_keys.add("skills/gone/SKILL.md")
        skills = build_backend(inline_skills={"demo": "INLINE"}).routes[SKILLS_ROOT]
        with pytest.raises(_TemplateMissing):
            await skills.agrep("anything")

    asyncio.run(go())


def test_inline_agrep_indexes_empty_body_template_skill(fake_tm: _FakeResourceManager) -> None:
    """An un-shadowed empty-string template body stays in the inline overlay's
    search corpus: grepping the empty pattern matches its one empty line, proving
    it was indexed (guards against an ``if content:`` truthiness check dropping
    it)."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/empty/SKILL.md", "")
        skills = build_backend(inline_skills={"demo": "INLINE"}).routes[SKILLS_ROOT]
        return await skills.agrep("")

    result = asyncio.run(go())
    assert "/empty/SKILL.md" in {m["path"] for m in result.matches}


def test_inline_als_skips_inline_dir_not_under_path(fake_tm: _FakeResourceManager) -> None:
    """An inline skill dir that is not under the requested listing base is excluded
    (the ``skill_dir.startswith(normalized)`` false leg): listing under one skill's
    dir returns only that dir, never an out-of-scope inline skill."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/other/SKILL.md", "x")
        backend = build_backend(inline_skills={"demo": "y"})
        return await backend.als(f"{SKILLS_ROOT}other/")

    result = asyncio.run(go())
    assert [e["path"] for e in result.entries] == [f"{SKILLS_ROOT}other/"]


def test_inline_skills_writes_raise(fake_tm: _FakeResourceManager) -> None:
    skills = InlineSkillsBackend({"demo": "x"})

    async def go() -> None:
        with pytest.raises(PermissionError, match="read-only"):
            await skills.awrite("/demo/SKILL.md", "x")
        with pytest.raises(PermissionError, match="read-only"):
            await skills.aedit("/demo/SKILL.md", "a", "b")
        with pytest.raises(PermissionError, match="read-only"):
            await skills.aupload_files([("/demo/SKILL.md", b"x")])

    asyncio.run(go())


def test_inline_sync_methods_raise_not_implemented(fake_tm: _FakeResourceManager) -> None:
    """The inline overlay is async-only too; every sync read entrypoint raises."""
    skills = InlineSkillsBackend({"demo": "x"})
    with pytest.raises(NotImplementedError, match="async-only"):
        skills.read("/demo/SKILL.md")
    with pytest.raises(NotImplementedError, match="async-only"):
        skills.ls("/")
    with pytest.raises(NotImplementedError, match="async-only"):
        skills.glob("*")
    with pytest.raises(NotImplementedError, match="async-only"):
        skills.grep("x")
    with pytest.raises(NotImplementedError, match="async-only"):
        skills.download_files(["/demo/SKILL.md"])


# --- template backend read/glob/grep edge paths ----------------------------


def test_template_agrep_searches_skill_bodies(fake_tm: _FakeResourceManager) -> None:
    """agrep on the plain template backend finds a literal match in a skill body
    and reports the matching skill's path; a non-matching skill is excluded."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "look for FINDME here")
        await fake_tm.upload_template("skills/other/SKILL.md", "nothing to see")
        skills = build_backend().routes[SKILLS_ROOT]
        return await skills.agrep("FINDME")

    result = asyncio.run(go())
    assert {m["path"] for m in result.matches} == {"/jq/SKILL.md"}


def test_template_agrep_propagates_vanished_listed_key(fake_tm: _FakeResourceManager) -> None:
    """A key the provider lists but which is absent from the store (it vanished
    between the listing and the fetch) propagates the provider's own error out of
    agrep, rather than being silently dropped from the search corpus."""

    async def go() -> None:
        fake_tm.phantom_keys.add("skills/gone/SKILL.md")
        skills = build_backend().routes[SKILLS_ROOT]
        with pytest.raises(_TemplateMissing):
            await skills.agrep("anything")

    asyncio.run(go())


def test_vanished_listed_key_propagates_across_read_grep_download(fake_tm: _FakeResourceManager) -> None:
    """A key present in the provider's listing but absent from the store propagates
    the provider's own error out of aread, agrep, and adownload_files — a genuine
    store inconsistency surfaced loudly, never masked as a not-found value."""

    async def go() -> None:
        fake_tm.phantom_keys.add("skills/gone/SKILL.md")
        skills = build_backend().routes[SKILLS_ROOT]
        with pytest.raises(_TemplateMissing):
            await skills.aread("/gone/SKILL.md")
        with pytest.raises(_TemplateMissing):
            await skills.agrep("anything")
        with pytest.raises(_TemplateMissing):
            await skills.adownload_files(["/gone/SKILL.md"])

    asyncio.run(go())


def test_template_agrep_indexes_empty_body_skill(fake_tm: _FakeResourceManager) -> None:
    """An empty-string skill body is a valid body and stays in the search corpus:
    grepping the empty pattern matches its one empty line, proving it was indexed
    (guards against an ``if content:`` truthiness check dropping it)."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/empty/SKILL.md", "")
        skills = build_backend().routes[SKILLS_ROOT]
        return await skills.agrep("")

    result = asyncio.run(go())
    assert "/empty/SKILL.md" in {m["path"] for m in result.matches}


def test_template_aread_offset_beyond_file_reports_error(fake_tm: _FakeResourceManager) -> None:
    """An offset past the file length surfaces the slice error verbatim, never a
    silently empty read."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "one\ntwo\n")
        return await build_backend().aread(f"{SKILLS_ROOT}jq/SKILL.md", offset=100)

    res = asyncio.run(go())
    assert res.file_data is None
    assert res.error is not None
    assert "exceeds file length" in res.error


def test_template_aglob_no_match_returns_empty(fake_tm: _FakeResourceManager) -> None:
    """A glob that matches nothing returns an empty match list, not the backend's
    'No files found' sentinel string."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        skills = build_backend().routes[SKILLS_ROOT]
        return await skills.aglob("*/NOPE.md")

    result = asyncio.run(go())
    assert result.matches == []


def test_template_aglob_star_is_non_recursive(fake_tm: _FakeResourceManager) -> None:
    """``*`` does NOT cross ``/``: ``*.md`` must not match a nested
    ``jq/SKILL.md`` (the wcmatch semantic that plain fnmatch would break)."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        skills = build_backend().routes[SKILLS_ROOT]
        return await skills.aglob("*.md")

    result = asyncio.run(go())
    assert result.matches == []


def test_template_aglob_globstar_is_recursive(fake_tm: _FakeResourceManager) -> None:
    """``**`` crosses ``/`` (explicit recursion): ``**/*.md`` matches nested
    skill files."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        await fake_tm.upload_template("skills/flow/SKILL.md", "y")
        skills = build_backend().routes[SKILLS_ROOT]
        return await skills.aglob("**/*.md")

    result = asyncio.run(go())
    assert {m["path"] for m in result.matches} == {"/jq/SKILL.md", "/flow/SKILL.md"}


def test_template_aglob_brace_expansion(fake_tm: _FakeResourceManager) -> None:
    """Brace alternation expands: ``{a,b}/SKILL.md`` matches both branches (the
    wcmatch semantic that plain fnmatch would not honor)."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/a/SKILL.md", "x")
        await fake_tm.upload_template("skills/b/SKILL.md", "y")
        await fake_tm.upload_template("skills/c/SKILL.md", "z")
        skills = build_backend().routes[SKILLS_ROOT]
        return await skills.aglob("{a,b}/SKILL.md")

    result = asyncio.run(go())
    assert {m["path"] for m in result.matches} == {"/a/SKILL.md", "/b/SKILL.md"}


def test_template_aglob_leading_slash_pattern_matches(fake_tm: _FakeResourceManager) -> None:
    """A leading ``/`` on the pattern is stripped before matching, so
    ``/jq/SKILL.md`` matches the relative ``jq/SKILL.md``."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        skills = build_backend().routes[SKILLS_ROOT]
        return await skills.aglob("/jq/SKILL.md")

    result = asyncio.run(go())
    assert {m["path"] for m in result.matches} == {"/jq/SKILL.md"}


def test_template_aglob_exact_file_base(fake_tm: _FakeResourceManager) -> None:
    """A ``path`` naming an exact skill file is the search base and matches on the
    bare filename (the exact-file base branch + its filename-relativisation arm),
    scoping the result to that one file."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        await fake_tm.upload_template("skills/flow/SKILL.md", "y")
        skills = build_backend().routes[SKILLS_ROOT]
        return await skills.aglob("SKILL.md", path="/jq/SKILL.md")

    result = asyncio.run(go())
    assert {m["path"] for m in result.matches} == {"/jq/SKILL.md"}


def test_template_aglob_directory_base_with_and_without_trailing_slash(fake_tm: _FakeResourceManager) -> None:
    """A directory ``path`` scopes the search to that skill dir and matches relative
    to it (the directory-prefix filter + its ``<dir>/`` relativisation arm); a
    trailing slash is normalized away, so ``/jq`` and ``/jq/`` behave identically."""

    async def go() -> tuple[Any, Any]:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        await fake_tm.upload_template("skills/flow/SKILL.md", "y")
        skills = build_backend().routes[SKILLS_ROOT]
        no_slash = await skills.aglob("*.md", path="/jq")
        with_slash = await skills.aglob("*.md", path="/jq/")
        return no_slash, with_slash

    no_slash, with_slash = asyncio.run(go())
    # Only the /jq dir is in scope — /flow is filtered out by the directory prefix.
    assert {m["path"] for m in no_slash.matches} == {"/jq/SKILL.md"}
    assert {m["path"] for m in with_slash.matches} == {"/jq/SKILL.md"}


def test_template_aglob_whitespace_only_path_is_empty(fake_tm: _FakeResourceManager) -> None:
    """A whitespace-only ``path`` is invalid and yields no matches (the guard)."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        skills = build_backend().routes[SKILLS_ROOT]
        return await skills.aglob("*", path="   ")

    result = asyncio.run(go())
    assert result.matches == []


def test_template_aglob_directory_base_no_match(fake_tm: _FakeResourceManager) -> None:
    """A directory ``path`` whose pattern matches nothing under it returns empty."""

    async def go() -> Any:
        await fake_tm.upload_template("skills/jq/SKILL.md", "x")
        skills = build_backend().routes[SKILLS_ROOT]
        return await skills.aglob("*.txt", path="/jq")

    result = asyncio.run(go())
    assert result.matches == []


def test_inline_aread_offset_beyond_inline_reports_error(fake_tm: _FakeResourceManager) -> None:
    """An offset past an inline skill's length surfaces the slice error too (the
    inline overlay's own read path, not the template delegate's)."""

    async def go() -> Any:
        backend = build_backend(inline_skills={"demo": "a\nb\n"})
        return await backend.aread(f"{SKILLS_ROOT}demo/SKILL.md", offset=100)

    res = asyncio.run(go())
    assert res.file_data is None
    assert res.error is not None
    assert "exceeds file length" in res.error
