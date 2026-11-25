"""Directory-scoped session APIs must not follow a planted git worktree.

``git worktree list --porcelain`` prints a worktree path taken from
``.git/worktrees/<id>/gitdir`` inside the directory the caller passed. That
file is ordinary repository content. These tests plant a gitdir that names
another project's ``.git`` and then call every directory-scoped session API
with the attacker's directory. Each call must miss, and none of them may
read or modify the other project's transcript.

The assertions describe the safe behavior. They fail while the APIs treat
every path git prints as another root of the caller's project.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
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
from claude_agent_sdk._internal.sessions import _sanitize_path

VICTIM_TITLE = "victim-private-title"
VICTIM_PROMPT = "victim-private-prompt"
VICTIM_AGENT = "victim-agent"
ATTACKER_PROMPT = "attacker-own-prompt"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _plant_foreign_worktree(attacker: Path, victim: Path) -> None:
    """Make ``git worktree list`` in ``attacker`` report ``victim``.

    Mirrors a tenant who can write their own ``.git``: ``gitdir`` contains
    the other project's ``.git`` path, and ``HEAD`` is present so the entry
    is a normal linked-worktree record. The victim repository does not point
    back at the attacker.
    """
    _git(attacker, "init", "-q")
    _git(victim, "init", "-q")
    record = attacker / ".git" / "worktrees" / "x"
    record.mkdir(parents=True)
    (record / "gitdir").write_text(f"{victim}/.git\n", encoding="utf-8")
    (record / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")


def _worktree_paths(cwd: Path) -> list[str]:
    paths: list[str] = []
    for line in _git(cwd, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            paths.append(line[len("worktree ") :])
    return paths


def _project_dir(config_dir: Path, project_path: str) -> Path:
    project_dir = config_dir / "projects" / _sanitize_path(project_path)
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def _write_transcript(
    project_dir: Path,
    session_id: str,
    *,
    prompt: str,
    title: str | None = None,
) -> Path:
    user_uuid = str(uuid.uuid4())
    assistant_uuid = str(uuid.uuid4())
    lines = [
        {
            "type": "user",
            "uuid": user_uuid,
            "parentUuid": None,
            "sessionId": session_id,
            "message": {"role": "user", "content": prompt},
        },
        {
            "type": "assistant",
            "uuid": assistant_uuid,
            "parentUuid": user_uuid,
            "sessionId": session_id,
            "message": {"role": "assistant", "content": "ack"},
        },
    ]
    if title is not None:
        lines.append(
            {
                "type": "custom-title",
                "customTitle": title,
                "sessionId": session_id,
            }
        )
    path = project_dir / f"{session_id}.jsonl"
    path.write_text(
        "\n".join(json.dumps(line, separators=(",", ":")) for line in lines) + "\n",
        encoding="utf-8",
    )
    return path


def _write_subagent(project_dir: Path, session_id: str) -> Path:
    agent_uuid = str(uuid.uuid4())
    subagents = project_dir / session_id / "subagents"
    subagents.mkdir(parents=True)
    agent_file = subagents / f"agent-{VICTIM_AGENT}.jsonl"
    agent_file.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": agent_uuid,
                "parentUuid": None,
                "sessionId": session_id,
                "message": {"role": "user", "content": VICTIM_PROMPT},
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return agent_file


def _snapshot(root: Path) -> dict[str, bytes]:
    snap: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            snap[str(path.relative_to(root))] = path.read_bytes()
    return snap


@pytest.fixture
def shared_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / ".claude"
    (config_dir / "projects").mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    return config_dir


@pytest.fixture
def tenants(tmp_path: Path, shared_config: Path) -> dict[str, object]:
    """Two projects, one config dir, attacker gitdir names the victim."""
    attacker = Path(os.path.realpath(tmp_path / "tenant-a"))
    victim = Path(os.path.realpath(tmp_path / "tenant-b"))
    attacker.mkdir()
    victim.mkdir()
    _plant_foreign_worktree(attacker, victim)

    reported = _worktree_paths(attacker)
    assert str(victim) in reported, (
        "git did not report the planted worktree path; the session APIs "
        "would not be exercised"
    )

    victim_id = str(uuid.uuid4())
    attacker_id = str(uuid.uuid4())
    victim_project = _project_dir(shared_config, str(victim))
    attacker_project = _project_dir(shared_config, str(attacker))
    victim_file = _write_transcript(
        victim_project,
        victim_id,
        prompt=VICTIM_PROMPT,
        title=VICTIM_TITLE,
    )
    _write_subagent(victim_project, victim_id)
    _write_transcript(attacker_project, attacker_id, prompt=ATTACKER_PROMPT)

    return {
        "attacker": attacker,
        "victim": victim,
        "victim_id": victim_id,
        "attacker_id": attacker_id,
        "victim_project": victim_project,
        "victim_file": victim_file,
        "before": _snapshot(victim_project),
    }


def _attacker(tenants: dict[str, object]) -> str:
    return str(tenants["attacker"])


def _victim_unchanged(tenants: dict[str, object]) -> None:
    project = tenants["victim_project"]
    assert isinstance(project, Path)
    assert _snapshot(project) == tenants["before"]


class TestPlantedWorktreeDoesNotCrossProjects:
    """Calls pass the attacker's directory and must not touch the victim."""

    def test_list_sessions_omits_foreign_session(
        self, tenants: dict[str, object]
    ) -> None:
        sessions = list_sessions(directory=_attacker(tenants))
        ids = {session.session_id for session in sessions}
        summaries = {session.summary for session in sessions}
        titles = {session.custom_title for session in sessions}
        prompts = {session.first_prompt for session in sessions}

        assert tenants["attacker_id"] in ids
        assert tenants["victim_id"] not in ids
        assert VICTIM_TITLE not in summaries
        assert VICTIM_TITLE not in titles
        assert VICTIM_PROMPT not in prompts
        assert VICTIM_PROMPT not in summaries

    def test_get_session_info_misses_foreign_session(
        self, tenants: dict[str, object]
    ) -> None:
        info = get_session_info(str(tenants["victim_id"]), directory=_attacker(tenants))
        assert info is None

    def test_get_session_messages_misses_foreign_transcript(
        self, tenants: dict[str, object]
    ) -> None:
        messages = get_session_messages(
            str(tenants["victim_id"]), directory=_attacker(tenants)
        )
        assert messages == []

    def test_list_subagents_misses_foreign_session(
        self, tenants: dict[str, object]
    ) -> None:
        agents = list_subagents(str(tenants["victim_id"]), directory=_attacker(tenants))
        assert agents == []

    def test_get_subagent_messages_misses_foreign_transcript(
        self, tenants: dict[str, object]
    ) -> None:
        messages = get_subagent_messages(
            str(tenants["victim_id"]),
            VICTIM_AGENT,
            directory=_attacker(tenants),
        )
        assert messages == []

    def test_rename_session_does_not_append_to_foreign_transcript(
        self, tenants: dict[str, object]
    ) -> None:
        with pytest.raises(FileNotFoundError):
            rename_session(
                str(tenants["victim_id"]),
                "attacker-chosen-title",
                directory=_attacker(tenants),
            )
        _victim_unchanged(tenants)

    def test_tag_session_does_not_append_to_foreign_transcript(
        self, tenants: dict[str, object]
    ) -> None:
        with pytest.raises(FileNotFoundError):
            tag_session(
                str(tenants["victim_id"]),
                "attacker-tag",
                directory=_attacker(tenants),
            )
        _victim_unchanged(tenants)

    def test_delete_session_does_not_remove_foreign_transcript(
        self, tenants: dict[str, object]
    ) -> None:
        with pytest.raises(FileNotFoundError):
            delete_session(str(tenants["victim_id"]), directory=_attacker(tenants))
        _victim_unchanged(tenants)

    def test_fork_session_does_not_copy_foreign_transcript(
        self, tenants: dict[str, object]
    ) -> None:
        with pytest.raises(FileNotFoundError):
            fork_session(str(tenants["victim_id"]), directory=_attacker(tenants))
        _victim_unchanged(tenants)

    @pytest.mark.anyio
    async def test_import_session_to_store_does_not_copy_foreign_transcript(
        self, tenants: dict[str, object]
    ) -> None:
        store = InMemorySessionStore()
        with pytest.raises(FileNotFoundError):
            await import_session_to_store(
                str(tenants["victim_id"]),
                store,
                directory=_attacker(tenants),
            )
        assert store._store == {}
        _victim_unchanged(tenants)
