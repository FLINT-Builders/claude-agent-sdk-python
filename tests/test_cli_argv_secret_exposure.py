"""CLI argv must not expose MCP credentials or inlined settings.

``SubprocessCLITransport.connect()`` builds the Claude Code command and spawns
it. These tests start that process and read the argument vector the kernel
publishes for it. They fail while configured MCP env vars, HTTP/SSE headers,
or settings values are present in that vector.
"""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from claude_agent_sdk.types import ClaudeAgentOptions

# Obvious placeholders. These stand in for values a caller would put in MCP
# server env/headers or a settings file. They are not credentials.
STDIO_ENV = "PLACEHOLDER_STDIO_ENV"
HTTP_HEADER = "PLACEHOLDER_HTTP_HEADER"
SSE_HEADER = "PLACEHOLDER_SSE_HEADER"
SDK_ENV = "PLACEHOLDER_SDK_ENV"
SDK_INSTANCE = "PLACEHOLDER_SDK_INSTANCE"
MCP_JSON_ENV = "PLACEHOLDER_MCP_JSON_ENV"
SETTINGS_FILE = "PLACEHOLDER_SETTINGS_FILE"
SETTINGS_JSON = "PLACEHOLDER_SETTINGS_JSON"
SETTINGS_JSON_ONLY = "PLACEHOLDER_SETTINGS_JSON_ONLY"

# Sleep in-process. A shell script that ``exec``s sleep replaces the process
# image and drops the argv this test has to observe.
_SLEEP_SCRIPT = "#!/usr/bin/python3\nimport time\ntime.sleep(60)\n"


@pytest.fixture(autouse=True)
def _skip_cli_version_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")


@pytest.fixture
def cli_path(tmp_path: Path) -> Path:
    script = tmp_path / "fake-claude"
    script.write_text(_SLEEP_SCRIPT, encoding="utf-8")
    script.chmod(0o755)
    return script


def _options(cli_path: Path, **kwargs: Any) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(cli_path=cli_path, **kwargs)


def _cmdline(pid: int) -> bytes:
    path = Path(f"/proc/{pid}/cmdline")
    mode = path.stat().st_mode
    assert mode & stat.S_IROTH, (
        f"/proc/{pid}/cmdline is mode {oct(mode & 0o777)}; "
        "this test needs the world-readable argv channel"
    )
    return path.read_bytes()


def _cmdline_as_other_user(pid: int) -> bytes | None:
    """Read argv as a different local account, the way the attack does.

    Returns None when this environment cannot drop to another user. The
    world-readable ``/proc/<pid>/cmdline`` bytes are still checked.
    """
    completed = subprocess.run(
        [
            "sudo",
            "-n",
            "-u",
            "nobody",
            "python3",
            "-c",
            "import pathlib,sys; sys.stdout.buffer.write("
            "pathlib.Path(sys.argv[1]).read_bytes())",
            f"/proc/{pid}/cmdline",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout


def _assert_absent_from_argv(pid: int, *placeholders: str) -> None:
    own = _cmdline(pid)
    other = _cmdline_as_other_user(pid)
    visible = other if other is not None else own
    ps_args = subprocess.check_output(
        ["ps", "-ww", "-o", "args=", "-p", str(pid)],
    )
    for placeholder in placeholders:
        needle = placeholder.encode()
        assert needle not in visible
        assert needle not in own
        assert needle not in ps_args


async def _spawn_and_check(options: ClaudeAgentOptions, *placeholders: str) -> None:
    transport = SubprocessCLITransport(prompt="test", options=options)
    await transport.connect()
    try:
        process = transport._process
        assert process is not None
        assert process.returncode is None
        _assert_absent_from_argv(process.pid, *placeholders)
    finally:
        await transport.close()


@pytest.mark.anyio
async def test_stdio_mcp_env_not_in_cli_argv(cli_path: Path) -> None:
    """External stdio servers are copied into ``--mcp-config`` unchanged."""
    await _spawn_and_check(
        _options(
            cli_path,
            mcp_servers={
                "payments": {
                    "type": "stdio",
                    "command": "payments-mcp",
                    "args": ["--serve"],
                    "env": {"SERVICE_API_KEY": STDIO_ENV},
                }
            },
        ),
        STDIO_ENV,
    )


@pytest.mark.anyio
async def test_http_mcp_headers_not_in_cli_argv(cli_path: Path) -> None:
    """HTTP MCP header values are copied into ``--mcp-config`` unchanged."""
    await _spawn_and_check(
        _options(
            cli_path,
            mcp_servers={
                "remote": {
                    "type": "http",
                    "url": "https://mcp.example/rpc",
                    "headers": {"Authorization": HTTP_HEADER},
                }
            },
        ),
        HTTP_HEADER,
    )


@pytest.mark.anyio
async def test_sse_mcp_headers_not_in_cli_argv(cli_path: Path) -> None:
    """SSE MCP header values are copied into ``--mcp-config`` unchanged."""
    await _spawn_and_check(
        _options(
            cli_path,
            mcp_servers={
                "events": {
                    "type": "sse",
                    "url": "https://mcp.example/sse",
                    "headers": {"Authorization": SSE_HEADER},
                }
            },
        ),
        SSE_HEADER,
    )


@pytest.mark.anyio
async def test_mcp_config_json_string_not_in_cli_argv(cli_path: Path) -> None:
    """A JSON string passed as ``mcp_servers`` is forwarded as ``--mcp-config``."""
    payload = (
        '{"mcpServers": {"payments": {"type": "stdio", "command": "payments-mcp", '
        f'"env": {{"SERVICE_API_KEY": "{MCP_JSON_ENV}"}}}}}}'
    )
    await _spawn_and_check(
        _options(cli_path, mcp_servers=payload),
        MCP_JSON_ENV,
    )


@pytest.mark.anyio
async def test_sdk_mcp_non_instance_fields_not_in_cli_argv(cli_path: Path) -> None:
    """SDK server configs drop ``instance`` and still serialize every other field."""
    await _spawn_and_check(
        _options(
            cli_path,
            mcp_servers={
                "inproc": {
                    "type": "sdk",
                    "name": "inproc",
                    "instance": SDK_INSTANCE,
                    "env": {"SERVICE_API_KEY": SDK_ENV},
                }
            },
        ),
        SDK_ENV,
        SDK_INSTANCE,
    )


@pytest.mark.anyio
async def test_settings_file_merged_with_sandbox_not_in_cli_argv(
    cli_path: Path, tmp_path: Path
) -> None:
    """A settings path plus sandbox is read and passed inline as ``--settings``."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        '{"env": {"SERVICE_API_KEY": "' + SETTINGS_FILE + '"}}',
        encoding="utf-8",
    )
    settings_path.chmod(0o600)
    await _spawn_and_check(
        _options(
            cli_path,
            settings=str(settings_path),
            sandbox={"enabled": True},
        ),
        SETTINGS_FILE,
    )


@pytest.mark.anyio
async def test_settings_json_merged_with_sandbox_not_in_cli_argv(
    cli_path: Path,
) -> None:
    """Inline settings JSON is merged with sandbox and passed as ``--settings``."""
    settings = '{"env": {"SERVICE_API_KEY": "' + SETTINGS_JSON + '"}}'
    await _spawn_and_check(
        _options(cli_path, settings=settings, sandbox={"enabled": True}),
        SETTINGS_JSON,
    )


@pytest.mark.anyio
async def test_settings_json_without_sandbox_not_in_cli_argv(
    cli_path: Path,
) -> None:
    """Settings JSON is passed through as ``--settings`` when sandbox is unset."""
    settings = '{"env": {"SERVICE_API_KEY": "' + SETTINGS_JSON_ONLY + '"}}'
    await _spawn_and_check(
        _options(cli_path, settings=settings),
        SETTINGS_JSON_ONLY,
    )
