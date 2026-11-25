"""``ClaudeAgentOptions.user`` is a session identifier, not a Unix account.

``query()`` and ``ClaudeSDKClient`` must be able to start the CLI when that
field holds an application login, and the CLI must keep the caller's uid,
euid, gid, and ``HOME``. These tests fail while the field is forwarded to
the subprocess as the process user.
"""

from __future__ import annotations

import json
import os
import pwd
import shutil
import sys
import tempfile
import textwrap
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, query
from claude_agent_sdk._internal.transport import subprocess_cli as cli_mod

pytestmark = pytest.mark.anyio

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process user")

_FAKE_CLI = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json
    import os
    import sys

    if "-v" in sys.argv or "--version" in sys.argv:
        print("2.0.0 (Claude Code)")
        raise SystemExit(0)

    def readable(path: str) -> bool:
        try:
            with open(path, "rb"):
                return True
        except OSError:
            return False

    proof = {
        "uid": os.getuid(),
        "euid": os.geteuid(),
        "gid": os.getgid(),
        "home": os.environ.get("HOME"),
        "can_read_parent_only": readable(os.environ["PARENT_ONLY"]),
        "can_read_other_only": readable(os.environ["OTHER_ONLY"]),
    }
    with open(os.environ["PROOF"], "w", encoding="utf-8") as handle:
        json.dump(proof, handle)
        handle.write("\\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "control_request":
            print(
                json.dumps(
                    {
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": msg["request_id"],
                            "response": {},
                        },
                    }
                ),
                flush=True,
            )
            continue
        if msg.get("type") == "user":
            print(
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "duration_ms": 0,
                        "duration_api_ms": 0,
                        "is_error": False,
                        "num_turns": 1,
                        "session_id": "s",
                        "result": "ok",
                    }
                ),
                flush=True,
            )
    """
)


@pytest.fixture
def session_dir() -> Iterator[Path]:
    """A directory the switched account can traverse.

    pytest's tmp_path is mode 0700, so a child setuid to another account
    cannot exec a CLI stored there. This directory is mode 0755 under /tmp.
    """
    path = Path(tempfile.mkdtemp(prefix="session-user-"))
    path.chmod(0o755)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_cli(directory: Path) -> Path:
    script = directory / "fake_claude.py"
    script.write_text(_FAKE_CLI)
    script.chmod(0o755)
    return script


def _foreign_account() -> str:
    """A local account that is not the account running the test."""
    current = pwd.getpwuid(os.getuid()).pw_name
    for name in ("nobody", "root"):
        if name == current:
            continue
        try:
            pwd.getpwnam(name)
        except KeyError:
            continue
        return name
    pytest.skip("no local account other than the current user")
    return ""


def _prepare(
    tmp_path: Path, session_user: str
) -> tuple[ClaudeAgentOptions, Path, bool]:
    cli = _write_cli(tmp_path)
    parent_only = tmp_path / "parent-only.txt"
    parent_only.write_text("placeholder-parent")
    parent_only.chmod(0o600)

    other_only = tmp_path / "other-only.txt"
    other_only.write_text("placeholder-other")
    other_only.chmod(0o600)
    other_owned = False
    try:
        foreign = pwd.getpwnam(session_user)
    except KeyError:
        foreign = None
    if foreign is not None:
        try:
            os.chown(other_only, foreign.pw_uid, foreign.pw_gid)
        except PermissionError:
            pass
        else:
            other_owned = True

    proof = tmp_path / "proof.json"
    proof.write_text("")
    proof.chmod(0o666)

    options = ClaudeAgentOptions(
        cli_path=str(cli),
        user=session_user,
        env={
            "PROOF": str(proof),
            "PARENT_ONLY": str(parent_only),
            "OTHER_ONLY": str(other_only),
        },
    )
    return options, proof, other_owned


async def _run(entry: str, options: ClaudeAgentOptions) -> None:
    if entry == "query":
        async for _message in query(prompt="hello", options=options):
            pass
        return
    async with ClaudeSDKClient(options=options) as client:
        await client.query("hello")
        async for _message in client.receive_response():
            pass


async def _exercise(
    entry: str,
    options: ClaudeAgentOptions,
    proof: Path,
    *,
    other_owned: bool,
) -> None:
    """Run one entry point and require every spawn to keep the caller's account.

    The wrapper delegates to the real ``open_process``. Recording the ``user``
    argument catches a fix that resolves the name and still passes it through,
    including a retry that drops the argument only after ``setreuid`` fails.
    """
    seen: list[object] = []
    original = cli_mod.anyio.open_process

    async def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("user"))
        return await original(*args, **kwargs)

    error: Exception | None = None
    with patch.object(cli_mod.anyio, "open_process", spy):
        try:
            await _run(entry, options)
        except Exception as exc:
            error = exc

    assert seen, "CLI subprocess was not started"
    assert all(user is None for user in seen), (seen, proof.read_text())
    if error is not None:
        raise error
    _assert_same_account(proof, other_owned=other_owned)


def _assert_same_account(proof_path: Path, *, other_owned: bool) -> None:
    payload: dict[str, Any] = json.loads(proof_path.read_text())
    assert payload["uid"] == os.getuid()
    assert payload["euid"] == os.geteuid()
    assert payload["gid"] == os.getgid()
    assert payload["home"] == os.environ.get("HOME")
    assert payload["can_read_parent_only"] is True
    # Root can read another account's mode-600 file without switching uid.
    # A non-root caller must not gain that read by adopting the other uid.
    if other_owned and os.geteuid() != 0:
        assert payload["can_read_other_only"] is False


@posix_only
@pytest.mark.parametrize("entry", ["query", "client"])
async def test_session_user_matching_os_account_does_not_switch(
    session_dir: Path, entry: str
) -> None:
    """A session user that is also a local account must not change the CLI uid."""
    session_user = _foreign_account()
    options, proof, other_owned = _prepare(session_dir, session_user)
    await _exercise(entry, options, proof, other_owned=other_owned)


@posix_only
@pytest.mark.parametrize("entry", ["query", "client"])
async def test_application_user_id_still_starts_cli(
    session_dir: Path, entry: str
) -> None:
    """An application id that is not a local account must still start the CLI."""
    session_user = f"tenant-{uuid.uuid4().hex}"
    options, proof, other_owned = _prepare(session_dir, session_user)
    await _exercise(entry, options, proof, other_owned=other_owned)
