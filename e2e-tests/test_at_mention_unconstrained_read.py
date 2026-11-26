"""At-mentions in prompts must not attach file bytes the caller did not authorize.

Claude Code expands ``@path`` tokens in a user message by reading the named
file and placing the bytes on the model request. That read does not go
through the tool permission callback, and it is not limited to ``cwd`` or to
the tools allowlist. The SDK only sets ``client_composed`` (which tells the
CLI to skip that expansion) when ``verbatim_prompts`` is true, and that
option defaults to false.

These tests drive the real CLI (``claude`` on ``PATH``) and a local stand-in
for the model API. They fail while an unmarked prompt still causes the file
bytes to be sent, and they pass once those bytes stay off the model request.
``verbatim_prompts=True`` is the control: it already keeps the bytes off the
request.

Run:

    python -m pytest e2e-tests/test_at_mention_unconstrained_read.py -v
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import anyio
import pytest

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultDeny,
    ToolPermissionContext,
    query,
)

# Obvious stand-in. The local server never checks it; it only has to be
# present so the CLI will build a model request.
_PLACEHOLDER_API_KEY = "placeholder-api-key"

_Entry = str


def _require_cli() -> str:
    found = shutil.which("claude")
    if found is None:
        pytest.fail("claude CLI is not on PATH; these tests drive the real CLI")
    return found


def _sse_response() -> bytes:
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_local",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-haiku-4-5",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "UNKNOWN"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return b"".join(
        f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
        for name, payload in events
    )


@dataclass
class _Capture:
    bodies: list[bytes]
    port: int

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@contextmanager
def _capturing_model_api() -> Iterator[_Capture]:
    """Accept the CLI's model calls and keep every POST body."""
    bodies: list[bytes] = []
    sse = _sse_response()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            bodies.append(body)
            streaming = b'"stream":true' in body or b'"stream": true' in body
            if streaming:
                payload = sse
                content_type = "text/event-stream"
            else:
                payload = json.dumps(
                    {
                        "id": "msg_local",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-haiku-4-5",
                        "content": [{"type": "text", "text": "UNKNOWN"}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Capture(bodies=bodies, port=port)
    finally:
        server.shutdown()
        server.server_close()


def _marker() -> str:
    return f"MARKER-{uuid.uuid4().hex}"


def _write_marker(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"The marker is {marker}.\n")


@dataclass
class _Layout:
    cwd: Path
    home: Path
    marker: str
    mention: str

    def prompt(self) -> str:
        return f"Reply with only UNKNOWN if you cannot see a marker. {self.mention}"


def _layout(tmp_path: Path, mention: str, marker_file: Path, marker: str) -> _Layout:
    cwd = tmp_path / "cwd"
    home = tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    (home / ".claude").mkdir()
    _write_marker(marker_file, marker)
    assert cwd.resolve() not in marker_file.resolve().parents
    return _Layout(cwd=cwd, home=home, marker=marker, mention=mention)


def _outside_file(tmp_path: Path) -> _Layout:
    marker = _marker()
    secret = tmp_path / "outside" / "secret.txt"
    return _layout(tmp_path, f"@{secret}", secret, marker)


def _options(
    layout: _Layout,
    capture: _Capture,
    *,
    tools: list[str] | None = None,
    include_tools: bool = True,
    disallowed_tools: list[str] | None = None,
    verbatim_prompts: bool = False,
    tool_calls: list[str] | None = None,
    stderr: list[str] | None = None,
) -> ClaudeAgentOptions:
    calls = tool_calls if tool_calls is not None else []

    async def _deny(
        tool_name: str,
        _tool_input: dict[str, Any],
        _context: ToolPermissionContext,
    ) -> PermissionResultDeny:
        calls.append(tool_name)
        return PermissionResultDeny(message="denied")

    kwargs: dict[str, Any] = {
        "cli_path": _require_cli(),
        "cwd": str(layout.cwd),
        "model": "claude-haiku-4-5",
        "max_turns": 1,
        "can_use_tool": _deny,
        "verbatim_prompts": verbatim_prompts,
        "env": {
            "HOME": str(layout.home),
            "CLAUDE_CONFIG_DIR": str(layout.home / ".claude"),
            "ANTHROPIC_API_KEY": _PLACEHOLDER_API_KEY,
            "ANTHROPIC_BASE_URL": capture.base_url,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
        },
    }
    if include_tools:
        kwargs["tools"] = [] if tools is None else tools
    if disallowed_tools is not None:
        kwargs["disallowed_tools"] = disallowed_tools
    if stderr is not None:
        kwargs["stderr"] = stderr.append
    return ClaudeAgentOptions(**kwargs)


async def _user_messages(
    content: str | list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }


async def _drive(entry: _Entry, text: str, options: ClaudeAgentOptions) -> None:
    if entry == "query-string":
        async for _message in query(prompt=text, options=options):
            pass
        return
    if entry == "query-stream":
        async for _message in query(prompt=_user_messages(text), options=options):
            pass
        return
    if entry == "query-blocks":
        blocks = [{"type": "text", "text": text}]
        async for _message in query(prompt=_user_messages(blocks), options=options):
            pass
        return
    if entry == "client-connect":
        client = ClaudeSDKClient(options=options)
        await client.connect(text)
        try:
            async for _message in client.receive_response():
                pass
        finally:
            await client.disconnect()
        return
    if entry == "client-connect-stream":
        client = ClaudeSDKClient(options=options)
        await client.connect(_user_messages(text))
        try:
            async for _message in client.receive_response():
                pass
        finally:
            await client.disconnect()
        return
    if entry == "client-query":
        async with ClaudeSDKClient(options=options) as client:
            await client.query(text)
            async for _message in client.receive_response():
                pass
        return
    if entry == "client-query-stream":
        async with ClaudeSDKClient(options=options) as client:
            await client.query(_user_messages(text))
            async for _message in client.receive_response():
                pass
        return
    raise AssertionError(f"unknown entry {entry}")


def _assert_marker_not_sent(capture: _Capture, marker: str, stderr: list[str]) -> None:
    assert capture.bodies, (
        "Claude Code made no model request, so this test cannot tell whether "
        "the file was attached.\n" + "\n".join(stderr[-40:])
    )
    leaked = [body for body in capture.bodies if marker.encode() in body]
    assert not leaked, (
        "the @-mentioned file's bytes were attached to the model request "
        f"({len(leaked)} request body/bodies)"
    )


async def _run_entry(entry: _Entry, layout: _Layout, **option_kwargs: Any) -> None:
    stderr: list[str] = []
    with _capturing_model_api() as capture:
        options = _options(layout, capture, stderr=stderr, **option_kwargs)
        with anyio.fail_after(60):
            await _drive(entry, layout.prompt(), options)
        _assert_marker_not_sent(capture, layout.marker, stderr)


_ENTRIES = [
    "query-string",
    "query-stream",
    "query-blocks",
    "client-connect",
    "client-connect-stream",
    "client-query",
    "client-query-stream",
]


@pytest.mark.e2e
@pytest.mark.anyio
@pytest.mark.parametrize("entry", _ENTRIES)
async def test_at_absolute_path_is_not_attached(entry: _Entry, tmp_path: Path) -> None:
    """Every SDK send path, tools disabled, path outside cwd."""
    await _run_entry(entry, _outside_file(tmp_path))


@pytest.mark.e2e
@pytest.mark.anyio
async def test_quoted_absolute_at_path_is_not_attached(tmp_path: Path) -> None:
    marker = _marker()
    secret = tmp_path / "outside" / "secret.txt"
    layout = _layout(tmp_path, f'@"{secret}"', secret, marker)
    await _run_entry("query-string", layout)


@pytest.mark.e2e
@pytest.mark.anyio
async def test_parent_relative_at_path_is_not_attached(tmp_path: Path) -> None:
    marker = _marker()
    secret = tmp_path / "outside" / "secret.txt"
    layout = _layout(tmp_path, "@../outside/secret.txt", secret, marker)
    await _run_entry("query-string", layout)


@pytest.mark.e2e
@pytest.mark.anyio
async def test_tilde_config_at_path_is_not_attached(tmp_path: Path) -> None:
    """A file under the user config directory, named from ``~``, outside cwd."""
    marker = _marker()
    secret = tmp_path / "home" / ".claude" / "notes.txt"
    # ``_layout`` creates home/.claude, then this writes the marker file.
    layout = _layout(tmp_path, "@~/.claude/notes.txt", secret, marker)
    await _run_entry("query-string", layout)


@pytest.mark.e2e
@pytest.mark.anyio
async def test_disallowed_read_still_must_not_attach(tmp_path: Path) -> None:
    """``disallowed_tools=["Read"]`` does not authorize the @-mention read."""
    await _run_entry(
        "query-string",
        _outside_file(tmp_path),
        include_tools=False,
        disallowed_tools=["Read"],
    )


@pytest.mark.e2e
@pytest.mark.anyio
async def test_default_tools_deny_callback_must_not_attach(tmp_path: Path) -> None:
    """Default tools plus a deny-all permission callback.

    The @-mention read has to show up as a denied Read of this file, or not
    happen. Bytes from the file must not reach the model request either way.
    """
    await _run_entry("query-string", _outside_file(tmp_path), include_tools=False)


@pytest.mark.e2e
@pytest.mark.anyio
async def test_verbatim_prompts_does_not_attach_file_bytes(tmp_path: Path) -> None:
    """Control: the existing opt-in already keeps the file off the request."""
    await _run_entry(
        "query-string",
        _outside_file(tmp_path),
        verbatim_prompts=True,
    )
