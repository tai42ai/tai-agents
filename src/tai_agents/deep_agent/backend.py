"""Storage backend for deep agents.

A deep agent's backend serves two very different needs through one interface:

* **Skills** — read-only ``SKILL.md`` artifacts, identical for every session.
  They live in the app's pluggable template provider (reached through
  ``tai_app.storage.resource_manager``), which is already the artifact store the
  ecosystem uses. The deep agent reads them live from there at run time, and they
  are managed (created/updated) through the template tools (``upload_template`` /
  ``list_resources``).
* **Scratch filesystem** — the files the agent writes while it works
  (``write_file``, todos, messages offloaded by the summarizer). These must be
  isolated per conversation: two concurrent sessions sharing one filesystem
  namespace would read and clobber each other's working files.

:func:`build_backend` composes both: a :class:`StateBackend` default (scratch
lives in per-thread agent state, so it is isolated by ``thread_id`` and travels
in the checkpoint) with skills routed to a read-only
:class:`TemplateSkillsBackend` mounted at :data:`SKILLS_ROOT`.
"""

from __future__ import annotations

from collections.abc import Iterable

import wcmatch.glob as wcglob
from deepagents.backends import BackendProtocol, CompositeBackend, StateBackend
from deepagents.backends.protocol import (
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    create_file_data,
    slice_read_response,
)
from tai_contract.app import tai_app

#: Mount point for skills inside the agent's virtual filesystem. Skill source
#: paths passed to ``create_deep_agent`` live under this prefix (e.g.
#: ``"/skills/jq/"``); :func:`build_backend` routes the prefix to the template
#: backend.
SKILLS_ROOT = "/skills/"

#: Prefix under which skill artifacts live in the template provider. A skill
#: named ``jq`` is the template key ``skills/jq/SKILL.md``. This is the key
#: convention the ``upload_template`` tool must use, and the one this backend
#: reads — the two provably agree, with no LangGraph store in the loop.
_SKILLS_KEY_PREFIX = "skills/"

#: File every skill directory must contain, per the Agent Skills specification.
_SKILL_FILENAME = "SKILL.md"


def _backend_path_to_key(backend_path: str) -> str:
    """Map a deepagents backend path to a template-provider key.

    ``CompositeBackend`` strips the :data:`SKILLS_ROOT` route prefix before
    delegating, so this backend receives a path rooted at the skills mount with
    a leading slash (e.g. ``"/jq/SKILL.md"``). The template key is that path
    without the leading slash, under :data:`_SKILLS_KEY_PREFIX`
    (``"skills/jq/SKILL.md"``).
    """
    return f"{_SKILLS_KEY_PREFIX}{backend_path.lstrip('/')}"


def _key_to_backend_path(key: str) -> str:
    """Map a template-provider key back to a deepagents backend path.

    Inverse of :func:`_backend_path_to_key`: ``"skills/jq/SKILL.md"`` ->
    ``"/jq/SKILL.md"``.
    """
    return "/" + key[len(_SKILLS_KEY_PREFIX) :]


def _glob_skill_paths(paths: Iterable[str], pattern: str, path: str) -> list[str]:
    """Return the backend paths matching a glob, over a fixed set of skill paths.

    Reproduces the deepagents filesystem glob so a skills backend matches exactly
    as the rest of the agent's virtual filesystem does:

    * ``path`` selects the search base. It is normalized to an absolute,
      trailing-slash-free form (root stays ``"/"``); only paths at or below it are
      considered and are matched relative to it — under root the leading slash is
      dropped, an exact-file base matches on the bare filename, and a directory
      base strips the ``"<dir>/"`` prefix.
    * A whitespace-only ``path`` is invalid and yields no matches.
    * A leading ``"/"`` on ``pattern`` is stripped, since matching runs against the
      relative paths — so ``"/jq/SKILL.md"`` and ``"jq/SKILL.md"`` match the same.
    * :func:`wcmatch.glob.globmatch` with ``BRACE | GLOBSTAR`` performs the match:
      ``*`` does NOT cross ``/`` (``"*.md"`` is non-recursive), ``**`` is the
      explicit recursive form, and ``{a,b}`` brace alternation expands.

    Matches are returned sorted ascending. The deepagents helper sorts by
    ``modified_at`` descending; that is immaterial here because these backends
    stamp every skill file identically, so a deterministic ascending sort is used.
    """
    base = path or "/"
    if not base.strip():
        return []
    normalized = base if base.startswith("/") else "/" + base
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized.rstrip("/")

    all_paths = list(paths)
    if normalized in all_paths:
        candidates = [normalized]
    elif normalized == "/":
        candidates = [p for p in all_paths if p.startswith("/")]
    else:
        dir_prefix = normalized + "/"
        candidates = [p for p in all_paths if p.startswith(dir_prefix)]

    effective_pattern = pattern.lstrip("/")
    matches: list[str] = []
    for file_path in candidates:
        if normalized == "/":
            relative = file_path[1:]
        elif file_path == normalized:
            relative = file_path.split("/")[-1]
        else:
            relative = file_path[len(normalized) + 1 :]
        if wcglob.globmatch(relative, effective_pattern, flags=wcglob.BRACE | wcglob.GLOBSTAR):
            matches.append(file_path)
    return sorted(matches)


class TemplateSkillsBackend(BackendProtocol):
    """Read-only deepagents backend backed by the app's template provider.

    Serves ``SKILL.md`` artifacts live from ``tai_app.storage.resource_manager``
    so the deep agent always sees the current skill content with no separate
    store or seed step. Skill markdown is read with the provider's RAW loader
    (``resource_manager.fetch_template``), never ``render_by_id`` — a skill body
    must never be passed through jinja.

    Reads only: every write/edit/upload path raises loudly, because skills are
    managed through the ``upload_template`` / ``list_resources`` tools, not the
    agent's filesystem.
    """

    async def _list_skill_keys(self) -> list[str]:
        """Return all ``skills/<name>/SKILL.md`` keys in the template provider."""
        keys = await tai_app.storage.resource_manager.list_resources()
        return [key for key in keys if key.startswith(_SKILLS_KEY_PREFIX) and key.endswith("/" + _SKILL_FILENAME)]

    async def als(self, path: str) -> LsResult:
        """List skill directories visible under ``path``.

        ``path`` is rooted at the skills mount (e.g. ``"/"`` for the whole
        skills tree, or ``"/jq/"`` for one skill). Returns one ``is_dir`` entry
        per skill directory whose ``SKILL.md`` exists at or below ``path``; the
        skills middleware then downloads ``<dir>/SKILL.md`` for each.
        """
        normalized = path if path.endswith("/") else path + "/"
        entries: list[FileInfo] = []
        for key in await self._list_skill_keys():
            skill_dir = _key_to_backend_path(key)[: -len(_SKILL_FILENAME)]
            if skill_dir.startswith(normalized):
                entries.append(FileInfo(path=skill_dir, is_dir=True, size=0, modified_at=""))
        entries.sort(key=lambda info: info["path"])
        return LsResult(entries=entries)

    def ls(self, path: str) -> LsResult:
        """Async-only; use :meth:`als`.

        Skills are read through the provider's async API. The sync default's
        ``asyncio.to_thread`` bridge would re-enter the running loop, so the
        skills loader and filesystem tools call :meth:`als`.
        """
        raise NotImplementedError("TemplateSkillsBackend is async-only; use als/aread/adownload_files.")

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a skill file's raw content for the requested line window.

        The provider's listing is authoritative for existence: a path whose key
        is not listed is reported as not found without a fetch. A listed key is
        read through ``resource_manager.fetch_template`` (the provider's raw
        loader) so skill markdown is returned verbatim, never jinja-evaluated;
        should the key vanish between the listing and the fetch, the provider's
        own error propagates. Line-number formatting is applied by the middleware.
        """
        key = _backend_path_to_key(file_path)
        if key not in await self._list_skill_keys():
            return ReadResult(error=f"File '{file_path}' not found")
        content = await tai_app.storage.resource_manager.fetch_template(key)
        file_data = create_file_data(content, encoding="utf-8")
        sliced = slice_read_response(file_data, offset, limit)
        if isinstance(sliced, ReadResult):
            return sliced
        return ReadResult(file_data={"content": sliced, "encoding": "utf-8"})

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Async-only; use :meth:`aread`."""
        raise NotImplementedError("TemplateSkillsBackend is async-only; use als/aread/adownload_files.")

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download raw skill files, one response per path (order preserved).

        Used by the skills middleware to fetch each ``SKILL.md``. The provider's
        listing is authoritative: a path whose key is not listed is reported as
        ``file_not_found`` for that path (matched against the requested path)
        without a fetch, so a partial set still yields the skills that do exist.
        A listed key is read through the provider's raw loader; should it vanish
        between the listing and the fetch, the provider's own error propagates.
        """
        listed = set(await self._list_skill_keys())
        responses: list[FileDownloadResponse] = []
        for path in paths:
            key = _backend_path_to_key(path)
            if key not in listed:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue
            content = await tai_app.storage.resource_manager.fetch_template(key)
            responses.append(FileDownloadResponse(path=path, content=content.encode("utf-8"), error=None))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async-only; use :meth:`adownload_files`."""
        raise NotImplementedError("TemplateSkillsBackend is async-only; use als/aread/adownload_files.")

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find skill ``SKILL.md`` files matching a glob pattern."""
        skill_paths = [_key_to_backend_path(key) for key in await self._list_skill_keys()]
        matches = [
            FileInfo(path=p, is_dir=False, size=0, modified_at="")
            for p in _glob_skill_paths(skill_paths, pattern, path)
        ]
        return GlobResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Async-only; use :meth:`aglob`."""
        raise NotImplementedError("TemplateSkillsBackend is async-only; use als/aread/adownload_files.")

    async def agrep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        """Search skill files for a literal text pattern."""
        from deepagents.backends.utils import grep_matches_from_files

        files = {}
        for key in await self._list_skill_keys():
            content = await tai_app.storage.resource_manager.fetch_template(key)
            files[_key_to_backend_path(key)] = create_file_data(content, encoding="utf-8")
        return grep_matches_from_files(files, pattern, path, glob)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        """Async-only; use :meth:`agrep`."""
        raise NotImplementedError("TemplateSkillsBackend is async-only; use als/aread/adownload_files.")

    # -- read-only: every mutation raises loudly --------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        """Reject writes — skills are managed via the ``upload_template`` tool."""
        raise PermissionError(
            f"Skills are read-only: cannot write {SKILLS_ROOT}{file_path.lstrip('/')}. "
            f"Manage skills with the upload_template tool."
        )

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Reject writes — skills are managed via the ``upload_template`` tool."""
        return self.write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Reject edits — skills are managed via the ``upload_template`` tool."""
        raise PermissionError(
            f"Skills are read-only: cannot edit {SKILLS_ROOT}{file_path.lstrip('/')}. "
            f"Manage skills with the upload_template tool."
        )

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Reject edits — skills are managed via the ``upload_template`` tool."""
        return self.edit(file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Reject uploads — skills are managed via the ``upload_template`` tool."""
        raise PermissionError(
            "Skills are read-only: cannot upload into the skills backend. Manage skills with the upload_template tool."
        )

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Reject uploads — skills are managed via the ``upload_template`` tool."""
        return self.upload_files(files)


def _inline_name_from_path(backend_path: str) -> str:
    """Return the skill name from a backend path rooted at the skills mount.

    ``CompositeBackend`` strips the :data:`SKILLS_ROOT` route prefix, so this
    backend receives paths like ``"/jq/SKILL.md"`` or ``"/jq/"``. The skill name
    is the first path segment (``"jq"``).
    """
    return backend_path.lstrip("/").split("/", 1)[0]


class InlineSkillsBackend(BackendProtocol):
    """Read-only skills backend overlaying call-supplied skills on the template store.

    A deep-agent run can carry inline skills — ``SKILL.md`` bodies passed
    directly in the call rather than read from the template provider. This
    backend maps each inline skill name to its content and, for any requested
    skills path, serves the inline body when the name matches and otherwise
    delegates to a :class:`TemplateSkillsBackend`. Listing operations
    (``als`` / ``aglob`` / ``adownload_files`` / ``agrep``) return the UNION of
    inline names and template skills, so the skills middleware discovers both.

    Inline content is served raw (never jinja-rendered), matching the template
    backend. Every write/edit/upload path raises loudly: skills are managed via
    the ``upload_template`` tool (template skills) or supplied inline at call
    time, never written through the agent's filesystem.
    """

    def __init__(self, inline_skills: dict[str, str]) -> None:
        """Store the inline ``name -> SKILL.md content`` map and the delegate.

        ``inline_skills`` maps each skill name to its ``SKILL.md`` body; the
        delegate :class:`TemplateSkillsBackend` serves every name not supplied
        inline.
        """
        self._inline = dict(inline_skills)
        self._template = TemplateSkillsBackend()

    def _inline_skill_md_path(self, name: str) -> str:
        """Backend path of an inline skill's ``SKILL.md`` (e.g. ``"/jq/SKILL.md"``)."""
        return f"/{name}/{_SKILL_FILENAME}"

    def _inline_skill_dir(self, name: str) -> str:
        """Backend path of an inline skill's directory (e.g. ``"/jq/"``)."""
        return f"/{name}/"

    async def als(self, path: str) -> LsResult:
        """List inline and template skill directories visible under ``path``.

        Returns one ``is_dir`` entry per skill directory at or below ``path``,
        merging inline skill names with the template provider's skills. An inline
        name shadows a template skill of the same name.
        """
        normalized = path if path.endswith("/") else path + "/"
        dirs: dict[str, FileInfo] = {}
        template_result = await self._template.als(path)
        for entry in template_result.entries or []:
            dirs[entry["path"]] = entry
        for name in self._inline:
            skill_dir = self._inline_skill_dir(name)
            if skill_dir.startswith(normalized):
                dirs[skill_dir] = FileInfo(path=skill_dir, is_dir=True, size=0, modified_at="")
        entries = sorted(dirs.values(), key=lambda info: info["path"])
        return LsResult(entries=entries)

    def ls(self, path: str) -> LsResult:
        """Async-only; use :meth:`als`."""
        raise NotImplementedError("InlineSkillsBackend is async-only; use als/aread/adownload_files.")

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read an inline or template skill file for the requested line window.

        Inline content is returned raw for a matching name; every other path
        delegates to the template backend (also raw).
        """
        name = _inline_name_from_path(file_path)
        if name in self._inline and file_path == self._inline_skill_md_path(name):
            file_data = create_file_data(self._inline[name], encoding="utf-8")
            sliced = slice_read_response(file_data, offset, limit)
            if isinstance(sliced, ReadResult):
                return sliced
            return ReadResult(file_data={"content": sliced, "encoding": "utf-8"})
        return await self._template.aread(file_path, offset, limit)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Async-only; use :meth:`aread`."""
        raise NotImplementedError("InlineSkillsBackend is async-only; use als/aread/adownload_files.")

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download raw inline or template skill files, one response per path.

        An inline name's ``SKILL.md`` is served from the inline map; every other
        path delegates to the template backend, which reports a key absent from
        the provider's listing as ``file_not_found`` for that path.
        """
        inline_responses: dict[int, FileDownloadResponse] = {}
        delegated_indices: list[int] = []
        delegated_paths: list[str] = []
        for index, path in enumerate(paths):
            name = _inline_name_from_path(path)
            if name in self._inline and path == self._inline_skill_md_path(name):
                inline_responses[index] = FileDownloadResponse(
                    path=path, content=self._inline[name].encode("utf-8"), error=None
                )
            else:
                delegated_indices.append(index)
                delegated_paths.append(path)
        delegated_responses = (
            dict(zip(delegated_indices, await self._template.adownload_files(delegated_paths), strict=True))
            if delegated_paths
            else {}
        )
        return [
            inline_responses[index] if index in inline_responses else delegated_responses[index]
            for index in range(len(paths))
        ]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async-only; use :meth:`adownload_files`."""
        raise NotImplementedError("InlineSkillsBackend is async-only; use als/aread/adownload_files.")

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find inline and template skill ``SKILL.md`` files matching a glob."""
        template_matches = await self._template.aglob(pattern, path)
        matched_paths = {match["path"] for match in template_matches.matches or []}
        inline_paths = [self._inline_skill_md_path(name) for name in self._inline]
        matched_paths.update(_glob_skill_paths(inline_paths, pattern, path))
        matches = [FileInfo(path=p, is_dir=False, size=0, modified_at="") for p in sorted(matched_paths)]
        return GlobResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Async-only; use :meth:`aglob`."""
        raise NotImplementedError("InlineSkillsBackend is async-only; use als/aread/adownload_files.")

    async def agrep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        """Search inline and template skill files for a literal text pattern."""
        from deepagents.backends.utils import grep_matches_from_files

        files = {}
        for key in await self._template._list_skill_keys():
            name = key[len(_SKILLS_KEY_PREFIX) :].split("/", 1)[0]
            if name in self._inline:
                continue
            content = await tai_app.storage.resource_manager.fetch_template(key)
            files[_key_to_backend_path(key)] = create_file_data(content, encoding="utf-8")
        for name, content in self._inline.items():
            files[self._inline_skill_md_path(name)] = create_file_data(content, encoding="utf-8")
        return grep_matches_from_files(files, pattern, path, glob)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        """Async-only; use :meth:`agrep`."""
        raise NotImplementedError("InlineSkillsBackend is async-only; use als/aread/adownload_files.")

    # -- read-only: every mutation raises loudly --------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        """Reject writes — skills are read-only (managed via upload_template or inline)."""
        raise PermissionError(
            f"Skills are read-only: cannot write {SKILLS_ROOT}{file_path.lstrip('/')}. "
            f"Manage skills with the upload_template tool or supply them inline."
        )

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Reject writes — skills are read-only (managed via upload_template or inline)."""
        return self.write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Reject edits — skills are read-only (managed via upload_template or inline)."""
        raise PermissionError(
            f"Skills are read-only: cannot edit {SKILLS_ROOT}{file_path.lstrip('/')}. "
            f"Manage skills with the upload_template tool or supply them inline."
        )

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Reject edits — skills are read-only (managed via upload_template or inline)."""
        return self.edit(file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Reject uploads — skills are read-only (managed via upload_template or inline)."""
        raise PermissionError(
            "Skills are read-only: cannot upload into the skills backend. "
            "Manage skills with the upload_template tool or supply them inline."
        )

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Reject uploads — skills are read-only (managed via upload_template or inline)."""
        return self.upload_files(files)


def build_backend(inline_skills: dict[str, str] | None = None) -> CompositeBackend:
    """Build the composite backend for a deep agent.

    ``default`` (everything except skills) uses a :class:`StateBackend`: the
    agent's scratch files live in graph state, so concurrent sessions never
    share or collide on the filesystem and the files are checkpointed with the
    rest of the conversation.

    The :data:`SKILLS_ROOT` prefix is routed to a read-only skills backend that
    serves skills live from ``tai_app.storage.resource_manager``, so every
    session reads the current skill content with no store and no seed step. When
    ``inline_skills`` (a ``name -> SKILL.md content`` map) is supplied, the route
    is an :class:`InlineSkillsBackend` overlaying that content on the template
    store: a requested name served inline when present, otherwise delegated to
    the template provider, with listings returning the union of both.
    """
    skills_backend: BackendProtocol = InlineSkillsBackend(inline_skills) if inline_skills else TemplateSkillsBackend()
    return CompositeBackend(
        default=StateBackend(),
        routes={SKILLS_ROOT: skills_backend},
    )
