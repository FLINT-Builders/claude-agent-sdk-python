"""End-to-end tests for the verbatim_prompts option with real Claude API calls.

Claude Code expands an ``@/absolute/path`` token in prompt text into the
contents of that file, outside the working directory and without a tool call.
The SDK marks such prompts ``client_composed`` by default, and
``verbatim_prompts=True`` marks every prompt. These tests plant a random marker
in a temp file and check whether the model can see it.
"""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)


def _options(cwd: Path, verbatim_prompts: bool) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        verbatim_prompts=verbatim_prompts,
        cwd=str(cwd),
        model="haiku",
        # No tools: the only way the model can learn the marker is the
        # prompt's @-mention expansion, never a Read call.
        tools=[],
        setting_sources=[],
        extra_args={"strict-mcp-config": None},
        max_turns=1,
    )


def _plant_marker(tmp_path: Path) -> tuple[Path, str]:
    marker = f"MARKER-{uuid.uuid4().hex[:12]}"
    secret = tmp_path / "outside" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text(f"The marker is {marker}.\n")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    return secret, marker


def _prompt(secret: Path) -> str:
    return (
        "What marker does this file contain? Reply with only the marker, or "
        f"UNKNOWN if you cannot see it. @{secret}"
    )


def _text_of(message: AssistantMessage) -> str:
    assert not any(isinstance(b, ToolUseBlock) for b in message.content)
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))


async def _reply_via_query(prompt: str, options: ClaudeAgentOptions) -> str:
    reply = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            reply += _text_of(message)
    return reply


async def _reply_via_client(prompt: str, options: ClaudeAgentOptions) -> str:
    reply = ""
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                reply += _text_of(message)
            elif isinstance(message, ResultMessage):
                assert not message.is_error, message
    return reply


async def _opted_in_stream(prompt: str) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "user",
        "message": {"role": "user", "content": prompt},
        "parent_tool_use_id": None,
        "client_composed": False,
    }


@pytest.mark.e2e
@pytest.mark.anyio
async def test_at_path_is_expanded_on_explicit_opt_in(tmp_path):
    """Control: a message that explicitly sets ``client_composed: false`` has
    the CLI read the @-mentioned file, so the tests below are meaningful."""
    secret, marker = _plant_marker(tmp_path)
    reply = ""
    async for message in query(
        prompt=_opted_in_stream(_prompt(secret)),
        options=_options(tmp_path / "cwd", verbatim_prompts=False),
    ):
        if isinstance(message, AssistantMessage):
            reply += _text_of(message)
    assert marker in reply, reply


@pytest.mark.e2e
@pytest.mark.anyio
async def test_at_path_is_not_expanded_by_default(tmp_path):
    secret, marker = _plant_marker(tmp_path)
    reply = await _reply_via_query(
        _prompt(secret), _options(tmp_path / "cwd", verbatim_prompts=False)
    )
    assert marker not in reply, reply


@pytest.mark.e2e
@pytest.mark.anyio
async def test_query_string_prompt_is_not_expanded(tmp_path):
    secret, marker = _plant_marker(tmp_path)
    reply = await _reply_via_query(
        _prompt(secret), _options(tmp_path / "cwd", verbatim_prompts=True)
    )
    assert marker not in reply, reply


@pytest.mark.e2e
@pytest.mark.anyio
async def test_client_query_prompt_is_not_expanded(tmp_path):
    secret, marker = _plant_marker(tmp_path)
    reply = await _reply_via_client(
        _prompt(secret), _options(tmp_path / "cwd", verbatim_prompts=True)
    )
    assert marker not in reply, reply
