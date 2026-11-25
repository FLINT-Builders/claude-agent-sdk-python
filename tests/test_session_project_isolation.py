"""Distinct working directories must not share Claude session transcripts.

Session APIs name a project by replacing every non-letter and non-digit in
the working directory with a hyphen (and, for long paths, by accepting
another directory that shares that sanitized prefix). These tests place a
transcript where the SDK stores the victim directory, then call the public
session APIs with a different directory that the sanitizer collapses onto
the same project. They fail while that collapse is the project identity,
and pass once each canonical path has its own key and on-disk folder.

The victim directory can still read its own transcript. The attacker
directory must not.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from claude_agent_sdk import (
    ClaudeAgentOptions,
    InMemorySessionStore,
    delete_session,
    delete_session_via_store,
    fork_session,
    fork_session_via_store,
    get_session_info,
    get_session_info_from_store,
    get_session_messages,
    get_session_messages_from_store,
    get_subagent_messages,
    get_subagent_messages_from_store,
    import_session_to_store,
    list_sessions,
    list_sessions_from_store,
    list_subagents,
    list_subagents_from_store,
    project_key_for_directory,
    rename_session,
    rename_session_via_store,
    tag_session,
    tag_session_via_store,
)
from claude_agent_sdk._internal.session_resume import materialize_resume_session
from claude_agent_sdk._internal.sessions import _canonicalize_path, _get_project_dir
from claude_agent_sdk.types import SessionKey

# Distinctive transcript text. Not a credential.
_PROMPT = "victim-only prompt for the acme workspace"
_TOOL_OUTPUT = "victim-only tool output for the acme workspace"
_SUBAGENT_PROMPT = "victim-only subagent transcript for the acme workspace"

_SHORT_KINDS = (
    "slash_hyphen",
    "dot_underscore",
    "dot_hyphen",
    "underscore_hyphen",
)
_FILESYSTEM_KINDS = (
    *_SHORT_KINDS,
    "long_prefix",
)


@pytest.fixture
def claude_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point session storage at a temporary config directory."""
    config_dir = tmp_path / ".claude"
    config_dir.mkdir()
    (config_dir / "projects").mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    return config_dir


def _pair(root: Path, kind: str) -> tuple[Path, Path]:
    """Two real directories the lossy sanitizer maps to one project."""
    if kind == "slash_hyphen":
        victim = root / "srv" / "workspaces" / "acme" / "repo"
        attacker = root / "srv" / "workspaces" / "acme-repo"
    elif kind == "dot_underscore":
        victim = root / "customers" / "my.project"
        attacker = root / "customers" / "my_project"
    elif kind == "dot_hyphen":
        victim = root / "customers" / "my.project"
        attacker = root / "customers" / "my-project"
    elif kind == "underscore_hyphen":
        victim = root / "customers" / "my_project"
        attacker = root / "customers" / "my-project"
    elif kind == "long_prefix":
        # Longer than the 200-character sanitized prefix. The tails differ,
        # so a hash of the full path would not match, but prefix lookup
        # still accepts the other directory's folder.
        shared = root / ("d" + ("p" * 220))
        victim = shared / "ends-with-victim"
        attacker = shared / "ends-with-attacker"
    else:
        raise AssertionError(f"unknown collision kind: {kind}")
    victim.mkdir(parents=True)
    attacker.mkdir(parents=True)
    return victim, attacker


def _transcript_lines(
    session_id: str, prompt: str, tool_output: str, *, cwd: str
) -> list[dict[str, Any]]:
    user_id = str(uuid.uuid4())
    assistant_id = str(uuid.uuid4())
    return [
        {
            "type": "user",
            "uuid": user_id,
            "parentUuid": None,
            "sessionId": session_id,
            "cwd": cwd,
            "timestamp": "2024-01-01T00:00:00.000Z",
            "message": {"role": "user", "content": prompt},
        },
        {
            "type": "assistant",
            "uuid": assistant_id,
            "parentUuid": user_id,
            "sessionId": session_id,
            "timestamp": "2024-01-01T00:00:01.000Z",
            "message": {"role": "assistant", "content": tool_output},
        },
    ]


def _write_jsonl(path: Path, lines: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines)
    path.write_text(payload, encoding="utf-8")


def _seed_filesystem(
    victim: Path, *, prompt: str = _PROMPT, tool_output: str = _TOOL_OUTPUT
) -> tuple[str, Path]:
    """Store a transcript where the SDK maps ``victim``.

    Uses the same canonical path and project directory the read and
    mutation APIs resolve, so the file sits on the lookup path.
    """
    session_id = str(uuid.uuid4())
    canonical = _canonicalize_path(str(victim))
    project_dir = _get_project_dir(canonical)
    transcript = project_dir / f"{session_id}.jsonl"
    _write_jsonl(
        transcript,
        _transcript_lines(session_id, prompt, tool_output, cwd=canonical),
    )
    subagent = project_dir / session_id / "subagents" / "agent-worker.jsonl"
    _write_jsonl(
        subagent,
        _transcript_lines(
            session_id,
            _SUBAGENT_PROMPT,
            _SUBAGENT_PROMPT,
            cwd=canonical,
        ),
    )
    return session_id, transcript


def _message_blob(messages: list[Any]) -> str:
    return json.dumps([m.message for m in messages])


def _refused(
    exc_type: type[BaseException], func: Any, *args: Any, **kwargs: Any
) -> None:
    """Call ``func`` and require it not to perform the cross-project action.

    A refusal raised after the write still fails the caller's byte check.
    """
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    pytest.fail(f"{func.__name__} acted on the other directory's session")


@pytest.mark.parametrize("kind", _SHORT_KINDS)
def test_project_keys_and_storage_dirs_differ(tmp_path: Path, kind: str) -> None:
    """Slash, hyphen, dot, and underscore must not share a project identity."""
    victim, attacker = _pair(tmp_path, kind)
    assert project_key_for_directory(victim) != project_key_for_directory(attacker)
    victim_canon = _canonicalize_path(str(victim))
    attacker_canon = _canonicalize_path(str(attacker))
    assert victim_canon != attacker_canon
    assert _get_project_dir(victim_canon) != _get_project_dir(attacker_canon)


@pytest.mark.parametrize("kind", _FILESYSTEM_KINDS)
def test_list_and_read_do_not_cross_directories(
    claude_config_dir: Path, tmp_path: Path, kind: str
) -> None:
    victim, attacker = _pair(tmp_path / kind, kind)
    session_id, _ = _seed_filesystem(victim)

    own = list_sessions(directory=str(victim))
    assert any(s.session_id == session_id for s in own)
    own_messages = get_session_messages(session_id, directory=str(victim))
    blob = _message_blob(own_messages)
    assert _PROMPT in blob
    assert _TOOL_OUTPUT in blob

    foreign = list_sessions(directory=str(attacker))
    assert not any(s.session_id == session_id for s in foreign)
    assert all(_PROMPT not in (s.summary or "") for s in foreign)
    assert all(_PROMPT not in (s.first_prompt or "") for s in foreign)

    foreign_messages = get_session_messages(session_id, directory=str(attacker))
    assert _PROMPT not in _message_blob(foreign_messages)
    assert _TOOL_OUTPUT not in _message_blob(foreign_messages)
    assert get_session_info(session_id, directory=str(attacker)) is None


@pytest.mark.parametrize("kind", _FILESYSTEM_KINDS)
def test_subagent_reads_do_not_cross_directories(
    claude_config_dir: Path, tmp_path: Path, kind: str
) -> None:
    victim, attacker = _pair(tmp_path / kind, kind)
    session_id, _ = _seed_filesystem(victim)

    assert list_subagents(session_id, directory=str(victim)) == ["worker"]
    own = get_subagent_messages(session_id, "worker", directory=str(victim))
    assert _SUBAGENT_PROMPT in _message_blob(own)

    assert list_subagents(session_id, directory=str(attacker)) == []
    foreign = get_subagent_messages(session_id, "worker", directory=str(attacker))
    assert _SUBAGENT_PROMPT not in _message_blob(foreign)


@pytest.mark.parametrize("kind", _FILESYSTEM_KINDS)
def test_delete_does_not_remove_other_directory(
    claude_config_dir: Path, tmp_path: Path, kind: str
) -> None:
    victim, attacker = _pair(tmp_path / kind, kind)
    session_id, transcript = _seed_filesystem(victim)
    before = transcript.read_bytes()
    subagent = transcript.with_suffix("") / "subagents" / "agent-worker.jsonl"
    subagent_before = subagent.read_bytes()

    _refused(
        FileNotFoundError,
        delete_session,
        session_id,
        directory=str(attacker),
    )

    assert transcript.read_bytes() == before
    assert subagent.read_bytes() == subagent_before
    messages = get_session_messages(session_id, directory=str(victim))
    assert _PROMPT in _message_blob(messages)


@pytest.mark.parametrize("kind", _FILESYSTEM_KINDS)
def test_rename_and_tag_do_not_modify_other_directory(
    claude_config_dir: Path, tmp_path: Path, kind: str
) -> None:
    victim, attacker = _pair(tmp_path / kind, kind)
    session_id, transcript = _seed_filesystem(victim)
    before = transcript.read_bytes()

    _refused(
        FileNotFoundError,
        rename_session,
        session_id,
        "taken over",
        directory=str(attacker),
    )
    _refused(
        FileNotFoundError,
        tag_session,
        session_id,
        "taken-over",
        directory=str(attacker),
    )

    assert transcript.read_bytes() == before
    info = get_session_info(session_id, directory=str(victim))
    assert info is not None
    assert info.custom_title is None
    assert info.tag is None


@pytest.mark.parametrize("kind", _FILESYSTEM_KINDS)
def test_fork_does_not_copy_other_directory(
    claude_config_dir: Path, tmp_path: Path, kind: str
) -> None:
    victim, attacker = _pair(tmp_path / kind, kind)
    session_id, transcript = _seed_filesystem(victim)
    before = transcript.read_bytes()
    project_dir = transcript.parent
    names_before = sorted(p.name for p in project_dir.iterdir())

    _refused(
        FileNotFoundError,
        fork_session,
        session_id,
        directory=str(attacker),
    )

    assert transcript.read_bytes() == before
    names_after = sorted(p.name for p in project_dir.iterdir())
    assert names_after == names_before
    assert _PROMPT not in "".join(
        p.read_text(encoding="utf-8")
        for p in project_dir.glob("*.jsonl")
        if p != transcript
    )


@pytest.mark.parametrize("kind", _FILESYSTEM_KINDS)
@pytest.mark.anyio
async def test_import_does_not_copy_other_directory(
    claude_config_dir: Path, tmp_path: Path, kind: str
) -> None:
    victim, attacker = _pair(tmp_path / kind, kind)
    session_id, transcript = _seed_filesystem(victim)
    before = transcript.read_bytes()
    store = InMemorySessionStore()

    with pytest.raises(FileNotFoundError):
        await import_session_to_store(session_id, store, directory=str(attacker))

    assert transcript.read_bytes() == before
    assert store.size == 0


def _store_entries(
    session_id: str, prompt: str, tool_output: str
) -> list[dict[str, Any]]:
    return _transcript_lines(session_id, prompt, tool_output, cwd="victim")


async def _seed_store(
    store: InMemorySessionStore, victim: Path
) -> tuple[str, SessionKey]:
    session_id = str(uuid.uuid4())
    key: SessionKey = {
        "project_key": project_key_for_directory(victim),
        "session_id": session_id,
    }
    entries = _store_entries(session_id, _PROMPT, _TOOL_OUTPUT)
    await store.append(key, entries)  # type: ignore[arg-type]
    sub_key: SessionKey = {**key, "subpath": "subagents/agent-worker"}
    sub_entries = _store_entries(session_id, _SUBAGENT_PROMPT, _SUBAGENT_PROMPT)
    await store.append(sub_key, sub_entries)  # type: ignore[arg-type]
    return session_id, key


@pytest.mark.parametrize("kind", _SHORT_KINDS)
@pytest.mark.anyio
async def test_store_reads_do_not_cross_directories(tmp_path: Path, kind: str) -> None:
    victim, attacker = _pair(tmp_path / kind, kind)
    store = InMemorySessionStore()
    session_id, _ = await _seed_store(store, victim)

    own = await list_sessions_from_store(store, directory=str(victim))
    assert any(s.session_id == session_id for s in own)
    own_messages = await get_session_messages_from_store(
        store, session_id, directory=str(victim)
    )
    assert _PROMPT in _message_blob(own_messages)
    assert _TOOL_OUTPUT in _message_blob(own_messages)
    own_info = await get_session_info_from_store(
        store, session_id, directory=str(victim)
    )
    assert own_info is not None
    assert own_info.session_id == session_id

    foreign = await list_sessions_from_store(store, directory=str(attacker))
    assert not any(s.session_id == session_id for s in foreign)
    assert all(_PROMPT not in (s.summary or "") for s in foreign)
    foreign_messages = await get_session_messages_from_store(
        store, session_id, directory=str(attacker)
    )
    assert _PROMPT not in _message_blob(foreign_messages)
    assert _TOOL_OUTPUT not in _message_blob(foreign_messages)
    assert (
        await get_session_info_from_store(store, session_id, directory=str(attacker))
        is None
    )


@pytest.mark.parametrize("kind", _SHORT_KINDS)
@pytest.mark.anyio
async def test_store_subagent_reads_do_not_cross_directories(
    tmp_path: Path, kind: str
) -> None:
    victim, attacker = _pair(tmp_path / kind, kind)
    store = InMemorySessionStore()
    session_id, _ = await _seed_store(store, victim)

    assert await list_subagents_from_store(
        store, session_id, directory=str(victim)
    ) == ["worker"]
    own = await get_subagent_messages_from_store(
        store, session_id, "worker", directory=str(victim)
    )
    assert _SUBAGENT_PROMPT in _message_blob(own)

    assert (
        await list_subagents_from_store(store, session_id, directory=str(attacker))
        == []
    )
    foreign = await get_subagent_messages_from_store(
        store, session_id, "worker", directory=str(attacker)
    )
    assert _SUBAGENT_PROMPT not in _message_blob(foreign)


@pytest.mark.parametrize("kind", _SHORT_KINDS)
@pytest.mark.anyio
async def test_store_mutations_do_not_touch_other_directory(
    tmp_path: Path, kind: str
) -> None:
    victim, attacker = _pair(tmp_path / kind, kind)
    store = InMemorySessionStore()
    session_id, key = await _seed_store(store, victim)
    before = await store.load(key)
    assert before is not None

    await rename_session_via_store(
        store, session_id, "taken over", directory=str(attacker)
    )
    await tag_session_via_store(
        store, session_id, "taken-over", directory=str(attacker)
    )
    await delete_session_via_store(store, session_id, directory=str(attacker))
    with pytest.raises(FileNotFoundError):
        await fork_session_via_store(store, session_id, directory=str(attacker))

    assert await store.load(key) == before
    victim_ids = [
        entry["session_id"] for entry in await store.list_sessions(key["project_key"])
    ]
    assert victim_ids == [session_id]
    messages = await get_session_messages_from_store(
        store, session_id, directory=str(victim)
    )
    assert _PROMPT in _message_blob(messages)
    attacker_messages = await get_session_messages_from_store(
        store, session_id, directory=str(attacker)
    )
    assert _PROMPT not in _message_blob(attacker_messages)


@pytest.mark.parametrize("kind", _SHORT_KINDS)
@pytest.mark.anyio
async def test_resume_does_not_load_other_directory(
    claude_config_dir: Path, tmp_path: Path, kind: str
) -> None:
    victim, attacker = _pair(tmp_path / kind, kind)
    store = InMemorySessionStore()
    session_id, key = await _seed_store(store, victim)
    before = await store.load(key)

    for options in (
        ClaudeAgentOptions(
            cwd=attacker,
            session_store=store,
            resume=session_id,
            env={"CLAUDE_CONFIG_DIR": str(claude_config_dir)},
        ),
        ClaudeAgentOptions(
            cwd=attacker,
            session_store=store,
            continue_conversation=True,
            env={"CLAUDE_CONFIG_DIR": str(claude_config_dir)},
        ),
    ):
        materialized = await materialize_resume_session(options)
        if materialized is not None:
            try:
                leaked = _PROMPT in _tree_text(materialized.config_dir)
            finally:
                await materialized.cleanup()
            pytest.fail(
                "resume materialized the other directory's session "
                f"(leaked={leaked}, id={materialized.resume_session_id})"
            )

    assert await store.load(key) == before

    # The victim directory still resumes its own transcript.
    own = await materialize_resume_session(
        ClaudeAgentOptions(
            cwd=victim,
            session_store=store,
            resume=session_id,
            env={"CLAUDE_CONFIG_DIR": str(claude_config_dir)},
        )
    )
    assert own is not None
    try:
        assert own.resume_session_id == session_id
        assert _PROMPT in _tree_text(own.config_dir)
        assert _TOOL_OUTPUT in _tree_text(own.config_dir)
    finally:
        await own.cleanup()


def _tree_text(root: Path) -> str:
    parts: list[str] = []
    for path in root.rglob("*"):
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)
