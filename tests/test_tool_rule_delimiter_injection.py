"""Tool-list entries must stay one rule when the CLI tokenizes them.

Claude Code 2.1.278 (the CLI this SDK bundles) splits every string in
``--tools``, ``--allowedTools``, and ``--disallowedTools`` on commas and on
spaces that sit outside parentheses, then treats a bare tool name as
permission for the whole tool. The same tokenizer is applied to each
``tools`` and ``disallowedTools`` string of an agent sent on the initialize
request. One list entry therefore has to be one rule: a comma or a space
outside parentheses in that entry adds a second rule the caller did not pass
as its own element.

``_build_command`` is the path ``connect()`` uses to build those flags.
``query()`` is the path that sends agent definitions. Both must refuse the
entry before the combined string is handed to the CLI. Separate safe entries
are still joined with commas; that joined value is two rules on purpose.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest

from claude_agent_sdk import query
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from claude_agent_sdk.types import AgentDefinition, ClaudeAgentOptions

_CLI = "/usr/bin/claude"

# One caller-supplied entry that the CLI tokenizer splits into a second rule.
_SPLIT_ENTRIES = (
    "Bash(npm test:*),Bash",
    "Bash(npm test:*),Bash(*)",
    "Read Bash",
    "Bash(npm test:*) Bash",
    "Read,Bash",
    "!Edit,Bash",
)


def _options(**kwargs: Any) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(cli_path=_CLI, **kwargs)


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


class _RecordingTransport:
    """In-process stand-in that answers the initialize handshake and records writes."""

    def __init__(self) -> None:
        self.written: list[str] = []
        self._pending: list[dict[str, Any]] = []
        self._ready_event: anyio.Event | None = None

    async def connect(self) -> None:
        # Created here, inside the running loop, before read_messages or write.
        self._ready_event = anyio.Event()

    def is_ready(self) -> bool:
        return True

    async def write(self, data: str) -> None:
        self.written.append(data)
        message = json.loads(data)
        if message.get("type") != "control_request":
            return
        request_id = message["request_id"]
        self._pending.append(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": request_id,
                    "response": {},
                },
            }
        )
        assert self._ready_event is not None
        self._ready_event.set()

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        assert self._ready_event is not None
        await self._ready_event.wait()
        for message in self._pending:
            yield message
        yield {
            "type": "result",
            "subtype": "success",
            "duration_ms": 0,
            "duration_api_ms": 0,
            "is_error": False,
            "num_turns": 1,
            "session_id": "sess",
            "result": "ok",
        }

    async def close(self) -> None:
        return None

    async def end_input(self) -> None:
        return None


def _initialize_agents(written: list[str]) -> dict[str, Any]:
    for line in written:
        message = json.loads(line)
        request = message.get("request")
        if isinstance(request, dict) and request.get("subtype") == "initialize":
            agents = request.get("agents")
            assert isinstance(agents, dict)
            return agents
    raise AssertionError("initialize request was not written")


class TestToolRuleDelimiters:
    """Reject tool-list entries the CLI would split into extra rules."""

    @pytest.mark.parametrize("entry", _SPLIT_ENTRIES)
    @pytest.mark.parametrize(
        "field",
        ("allowed_tools", "disallowed_tools", "tools"),
    )
    def test_build_command_rejects_entry_the_cli_would_split(
        self, field: str, entry: str
    ) -> None:
        """A single list element must not survive into the CLI argument."""
        options = _options(
            **{field: ["Edit", entry]},
            permission_mode="bypassPermissions",
        )
        transport = SubprocessCLITransport(prompt="test", options=options)
        with pytest.raises(ValueError, match=re.escape(entry)):
            transport._build_command()

    def test_separate_entries_are_still_one_comma_separated_argument(self) -> None:
        """Two safe entries stay two rules; a space inside parentheses stays in one."""
        transport = SubprocessCLITransport(
            prompt="test",
            options=_options(
                allowed_tools=["Bash(npm test:*)", "Read"],
                disallowed_tools=["Edit", "Write"],
                tools=["Read", "Edit", "Bash(git commit -m 'hi there')"],
                permission_mode="bypassPermissions",
            ),
        )
        cmd = transport._build_command()
        assert _flag(cmd, "--allowedTools") == "Bash(npm test:*),Read"
        assert _flag(cmd, "--disallowedTools") == "Edit,Write"
        assert _flag(cmd, "--tools") == "Read,Edit,Bash(git commit -m 'hi there')"
        assert _flag(cmd, "--permission-mode") == "bypassPermissions"

    def test_allowed_tools_injection_is_rejected_when_skills_are_also_set(
        self,
    ) -> None:
        """Skill-name checks do not cover an allowed_tools entry merged beside them."""
        entry = "Bash(npm test:*),Bash"
        transport = SubprocessCLITransport(
            prompt="test",
            options=_options(allowed_tools=[entry], skills=["pdf"]),
        )
        with pytest.raises(ValueError, match=re.escape(entry)):
            transport._build_command()

    @pytest.mark.parametrize("field", ("tools", "disallowedTools"))
    @pytest.mark.parametrize("entry", _SPLIT_ENTRIES)
    def test_query_rejects_agent_entry_before_initialize(
        self, field: str, entry: str
    ) -> None:
        """Agent tool lists ride the initialize request, not the CLI argv flags."""
        agent = AgentDefinition(
            description="helper",
            prompt="help",
            **{field: ["Edit", entry]},
        )
        options = _options(agents={"helper": agent})
        transport = _RecordingTransport()

        async def _run() -> None:
            with pytest.raises(ValueError, match=re.escape(entry)):
                async for _ in query(prompt="hi", options=options, transport=transport):
                    pass

        anyio.run(_run)
        for line in transport.written:
            assert entry not in line

    def test_query_sends_separate_agent_tool_entries_unchanged(self) -> None:
        """Intentionally separate agent tool names are not glued into one rule."""
        options = _options(
            agents={
                "helper": AgentDefinition(
                    description="helper",
                    prompt="help",
                    tools=["Read", "Bash(npm test:*)"],
                    disallowedTools=["Edit", "Write"],
                )
            }
        )
        transport = _RecordingTransport()

        async def _run() -> None:
            async for _ in query(prompt="hi", options=options, transport=transport):
                pass

        anyio.run(_run)
        agents = _initialize_agents(transport.written)
        assert agents["helper"]["tools"] == ["Read", "Bash(npm test:*)"]
        assert agents["helper"]["disallowedTools"] == ["Edit", "Write"]
