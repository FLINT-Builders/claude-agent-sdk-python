"""Directory-scoped session APIs must not cross project folders.

Long working directories share one 200-character sanitized prefix and are
stored as ``<prefix>-<suffix>``. When the exact sanitized folder for the
requested directory is absent, resolution must not read or modify a
different project whose folder only shares that prefix.

Store-backed helpers (``list_sessions_from_store``,
``get_session_messages_from_store``, ``get_session_info_from_store``,
``list_subagents_from_store``, ``get_subagent_messages_from_store``,
``rename_session_via_store``, ``tag_session_via_store``,
``delete_session_via_store``, ``fork_session_via_store``) key on the full
sanitized project key and are not part of this contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from claude_agent_sdk import (
    InMemorySessionStore,
    delete_session,
    fork_session,
    get_session_info,
    get_session_messages,
    get_subagent_messages,
    import_session_to_store,
    list_sessions,
    list_subagents,
    rename_session,
    tag_session,
)
from claude_agent_sdk._internal.sessions import (
    MAX_SANITIZED_LENGTH,
    _canonicalize_path,
    _get_worktree_paths,
    _sanitize_path,
)

_VICTIM_PROMPT = "tenant-b-workspace-prompt"
_VICTIM_TOOL_OUTPUT = "tenant-b-tool-output"
_VICTIM_SUBAGENT_PROMPT = "tenant-b-subagent-prompt"
_CALLER_PROMPT = "tenant-a-workspace-prompt"


@dataclass
class _Layout:
    """Two long cwds whose sanitized names share a 200-character prefix."""

    caller_cwd: str
    victim_dir: Path
    victim_session_id: str
    subagent_id: str
    snapshot: dict[str, bytes]


def _snapshot(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        return {"__missing__": b"1"}
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def _write_transcript(
    path: Path, session_id: str, prompt: str, tool_output: str
) -> None:
    user_id = str(uuid.uuid4())
    assistant_id = str(uuid.uuid4())
    entries = [
        {
            "type": "user",
            "uuid": user_id,
            "parentUuid": None,
            "sessionId": session_id,
            "message": {"role": "user", "content": prompt},
        },
        {
            "type": "assistant",
            "uuid": assistant_id,
            "parentUuid": user_id,
            "sessionId": session_id,
            "message": {"role": "assistant", "content": tool_output},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(entry, separators=(",", ":")) for entry in entries) + "\n",
        encoding="utf-8",
    )


def _colliding_cwds(root: Path) -> tuple[Path, Path]:
    """Build two real directories that differ only after the truncation point."""
    shared = root / ("d" * MAX_SANITIZED_LENGTH)
    caller = shared / "tenant-a" / "workspace"
    victim = shared / "tenant-b" / "workspace"
    caller.mkdir(parents=True)
    victim.mkdir(parents=True)
    canonical_caller = _canonicalize_path(str(caller))
    canonical_victim = _canonicalize_path(str(victim))
    sanitized_caller = _sanitize_path(canonical_caller)
    sanitized_victim = _sanitize_path(canonical_victim)
    assert len(sanitized_caller) > MAX_SANITIZED_LENGTH
    assert (
        sanitized_caller[:MAX_SANITIZED_LENGTH]
        == sanitized_victim[:MAX_SANITIZED_LENGTH]
    )
    assert sanitized_caller != sanitized_victim
    assert _get_worktree_paths(canonical_caller) == []
    return caller, victim


def _plant_victim(project_dir: Path) -> tuple[str, str, dict[str, bytes]]:
    session_id = str(uuid.uuid4())
    subagent_id = "tenant-b-agent"
    _write_transcript(
        project_dir / f"{session_id}.jsonl",
        session_id,
        _VICTIM_PROMPT,
        _VICTIM_TOOL_OUTPUT,
    )
    subagents = project_dir / session_id / "subagents"
    subagents.mkdir(parents=True)
    _write_transcript(
        subagents / f"agent-{subagent_id}.jsonl",
        session_id,
        _VICTIM_SUBAGENT_PROMPT,
        _VICTIM_TOOL_OUTPUT,
    )
    return session_id, subagent_id, _snapshot(project_dir)


def _assert_unchanged(layout: _Layout) -> None:
    assert _snapshot(layout.victim_dir) == layout.snapshot


@pytest.fixture(params=["sole_prefix_match", "first_of_two_prefix_matches"])
def layout(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _Layout:
    """On-disk projects that share a prefix. The requested cwd's exact folder is absent.

    ``sole_prefix_match`` stores only the other tenant, under that tenant's own
    sanitized name. ``first_of_two_prefix_matches`` stores both tenants under
    suffixes that are not the SDK hash, and yields the other tenant's folder
    first from ``Path.iterdir``.
    """
    config = tmp_path / "claude-config"
    projects = config / "projects"
    projects.mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))

    caller, victim = _colliding_cwds(tmp_path / "cwds")
    canonical_caller = _canonicalize_path(str(caller))
    canonical_victim = _canonicalize_path(str(victim))
    sanitized_caller = _sanitize_path(canonical_caller)
    sanitized_victim = _sanitize_path(canonical_victim)
    prefix = sanitized_caller[:MAX_SANITIZED_LENGTH]
    # The exact folder the SDK computes for the caller must not exist.
    # Prefix fallback is the path under test.
    assert not (projects / sanitized_caller).exists()

    if request.param == "sole_prefix_match":
        victim_dir = projects / sanitized_victim
        victim_dir.mkdir()
    else:
        victim_dir = projects / f"{prefix}-suffix-tenant-b"
        caller_dir = projects / f"{prefix}-suffix-tenant-a"
        assert victim_dir.name != sanitized_caller
        assert caller_dir.name != sanitized_caller
        assert victim_dir.name != sanitized_victim
        victim_dir.mkdir()
        caller_dir.mkdir()
        caller_session = str(uuid.uuid4())
        _write_transcript(
            caller_dir / f"{caller_session}.jsonl",
            caller_session,
            _CALLER_PROMPT,
            "tenant-a-tool-output",
        )
        original_iterdir = Path.iterdir

        def _iterdir(self: Path) -> Iterator[Path]:
            entries = list(original_iterdir(self))
            try:
                is_projects = self.resolve() == projects.resolve()
            except OSError:
                is_projects = False
            if is_projects:
                entries.sort(
                    key=lambda entry: (entry.name != victim_dir.name, entry.name)
                )
            return iter(entries)

        monkeypatch.setattr(Path, "iterdir", _iterdir)

    assert victim_dir.name.startswith(prefix + "-")
    session_id, subagent_id, snapshot = _plant_victim(victim_dir)
    return _Layout(
        caller_cwd=str(caller),
        victim_dir=victim_dir,
        victim_session_id=session_id,
        subagent_id=subagent_id,
        snapshot=snapshot,
    )


def test_list_sessions_does_not_return_the_other_project(layout: _Layout) -> None:
    """list_sessions(directory=caller) is scoped to the caller project."""
    listed = list_sessions(directory=layout.caller_cwd)
    listed_without_worktrees = list_sessions(
        directory=layout.caller_cwd, include_worktrees=False
    )
    assert listed == []
    assert listed_without_worktrees == []
    _assert_unchanged(layout)


def test_get_session_messages_does_not_read_the_other_project(layout: _Layout) -> None:
    messages = get_session_messages(
        layout.victim_session_id, directory=layout.caller_cwd
    )
    assert messages == []
    _assert_unchanged(layout)


def test_get_session_info_does_not_read_the_other_project(layout: _Layout) -> None:
    info = get_session_info(layout.victim_session_id, directory=layout.caller_cwd)
    assert info is None
    _assert_unchanged(layout)


def test_list_subagents_does_not_read_the_other_project(layout: _Layout) -> None:
    assert list_subagents(layout.victim_session_id, directory=layout.caller_cwd) == []
    _assert_unchanged(layout)


def test_get_subagent_messages_does_not_read_the_other_project(layout: _Layout) -> None:
    messages = get_subagent_messages(
        layout.victim_session_id,
        layout.subagent_id,
        directory=layout.caller_cwd,
    )
    assert messages == []
    _assert_unchanged(layout)


def test_delete_session_does_not_remove_the_other_project(layout: _Layout) -> None:
    with pytest.raises(FileNotFoundError):
        delete_session(layout.victim_session_id, directory=layout.caller_cwd)
    _assert_unchanged(layout)


def test_rename_session_does_not_append_to_the_other_project(layout: _Layout) -> None:
    with pytest.raises(FileNotFoundError):
        rename_session(
            layout.victim_session_id,
            "renamed-from-other-cwd",
            directory=layout.caller_cwd,
        )
    _assert_unchanged(layout)


def test_tag_session_does_not_append_to_the_other_project(layout: _Layout) -> None:
    with pytest.raises(FileNotFoundError):
        tag_session(
            layout.victim_session_id,
            "from-other-cwd",
            directory=layout.caller_cwd,
        )
    _assert_unchanged(layout)


def test_fork_session_does_not_write_into_the_other_project(layout: _Layout) -> None:
    with pytest.raises(FileNotFoundError):
        fork_session(layout.victim_session_id, directory=layout.caller_cwd)
    _assert_unchanged(layout)


@pytest.mark.anyio
async def test_import_session_to_store_does_not_read_the_other_project(
    layout: _Layout,
) -> None:
    store = InMemorySessionStore()
    with pytest.raises(FileNotFoundError):
        await import_session_to_store(
            layout.victim_session_id,
            store,
            directory=layout.caller_cwd,
        )
    assert store.size == 0
    _assert_unchanged(layout)


def test_list_sessions_ignores_prefix_match_across_git_worktrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo with two long worktree paths still must not use a shared prefix.

    The other worktree's transcripts live under a non-exact suffix, so they
    are not that worktree's exact project folder. Listing the caller worktree
    must not return them.
    """
    config = tmp_path / "claude-config"
    projects = config / "projects"
    projects.mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))

    shared = tmp_path / "cwds" / ("d" * MAX_SANITIZED_LENGTH)
    caller = shared / "tenant-a" / "workspace"
    victim = shared / "tenant-b" / "workspace"
    caller.mkdir(parents=True)
    git_env = os.environ.copy()
    git_env.update(
        {
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=caller, check=True, env=git_env)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=caller,
        check=True,
        env=git_env,
    )
    subprocess.run(
        ["git", "worktree", "add", "-b", "other", str(victim)],
        cwd=caller,
        check=True,
        env=git_env,
    )

    canonical_caller = _canonicalize_path(str(caller))
    canonical_victim = _canonicalize_path(str(victim))
    sanitized_caller = _sanitize_path(canonical_caller)
    sanitized_victim = _sanitize_path(canonical_victim)
    prefix = sanitized_caller[:MAX_SANITIZED_LENGTH]
    assert (
        sanitized_caller[:MAX_SANITIZED_LENGTH]
        == sanitized_victim[:MAX_SANITIZED_LENGTH]
    )
    assert len(_get_worktree_paths(canonical_caller)) >= 2
    assert not (projects / sanitized_caller).exists()

    victim_dir = projects / f"{prefix}-suffix-tenant-b"
    assert victim_dir.name != sanitized_victim
    victim_dir.mkdir()
    session_id, _, snapshot = _plant_victim(victim_dir)

    listed = list_sessions(directory=str(caller))
    assert listed == []
    messages = get_session_messages(session_id, directory=str(caller))
    assert messages == []
    assert _snapshot(victim_dir) == snapshot
