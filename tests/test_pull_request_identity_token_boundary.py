"""Replay the Test workflow's identity-token jobs against pull-request trees.

The harness does not call GitHub or Anthropic. It evaluates the workflow
conditions, runs the setup action's script against a local stand-in of
@actions/core getIDToken (@actions/core 1.10.1 oidc-utils.js), then runs the
later steps from the workflow on an attacker-controlled tree. The token value
is the placeholder below, never a real credential.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
REVIEW_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "claude-code-review.yml"
SETUP_ACTION = REPO_ROOT / ".github" / "actions" / "setup-claude-auth" / "action.yml"
REPOSITORY = "FLINT-Builders/claude-agent-sdk-python"
PLACEHOLDER = "placeholder-identity-token"
AUDIENCE = "https://api.anthropic.com"
TOKEN_JOBS = ("test-e2e", "test-e2e-docker", "test-examples")

# actions/runner v2.338.2 FileCommandManager.EnvFileKeyValuePairs, Linux branch.
# @actions/core 1.10.1 file-command.js writes NAME<<delimiter values into GITHUB_ENV.


def _read_line(text: str, index: int) -> tuple[str | None, int, str | None]:
    if index >= len(text):
        return None, index, None
    lf_index = text.find("\n", index)
    if lf_index < 0:
        return text[index:], len(text), None
    return text[index:lf_index], lf_index + 1, "\n"


def parse_github_env(file_path: Path) -> dict[str, str]:
    """Parse GITHUB_ENV the way actions/runner v2.338.2 does on Linux."""
    text = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    pairs: dict[str, str] = {}
    index = 0
    line, index, _newline = _read_line(text, index)
    while line is not None:
        if line != "":
            equals_index = line.find("=")
            heredoc_index = line.find("<<")
            if equals_index >= 0 and (
                heredoc_index < 0 or equals_index < heredoc_index
            ):
                key, output = line.split("=", 1)
            elif heredoc_index >= 0 and (
                equals_index < 0 or heredoc_index < equals_index
            ):
                key, delimiter = line.split("<<", 1)
                if key == "" or delimiter == "":
                    raise RuntimeError(f"invalid GITHUB_ENV line {line!r}")
                start = index
                end = index
                temp, index, newline = _read_line(text, index)
                while temp != delimiter:
                    if temp is None:
                        raise RuntimeError(f"missing delimiter {delimiter!r}")
                    if newline is None:
                        raise RuntimeError("EOF marker missing new line")
                    end = index - len(newline)
                    temp, index, newline = _read_line(text, index)
                output = text[start:end] if end > start else ""
            else:
                raise RuntimeError(f"invalid GITHUB_ENV line {line!r}")
            pairs[key] = output
        line, index, _newline = _read_line(text, index)
    return pairs


class _Expr:
    def __init__(self, source: str) -> None:
        self.source = source
        self.i = 0

    def parse(self) -> object:
        value = self._or()
        self._skip()
        if self.i != len(self.source):
            raise RuntimeError(f"unparsed expression {self.source[self.i :]!r}")
        return value

    def _peek(self) -> str:
        return self.source[self.i : self.i + 1]

    def _skip(self) -> None:
        while self._peek().isspace():
            self.i += 1

    def _or(self) -> object:
        left = self._and()
        while True:
            self._skip()
            if not self.source.startswith("||", self.i):
                return left
            self.i += 2
            right = self._and()
            left = bool(left) or bool(right)
        return left

    def _and(self) -> object:
        left = self._cmp()
        while True:
            self._skip()
            if not self.source.startswith("&&", self.i):
                return left
            self.i += 2
            right = self._cmp()
            left = bool(left) and bool(right)
        return left

    def _cmp(self) -> object:
        left = self._primary()
        self._skip()
        if self.source.startswith("==", self.i):
            op = "=="
            self.i += 2
        elif self.source.startswith("!=", self.i):
            op = "!="
            self.i += 2
        else:
            return left
        right = self._primary()
        equal = _loose_equal(left, right)
        return equal if op == "==" else not equal

    def _primary(self) -> object:
        self._skip()
        if self._peek() == "(":
            self.i += 1
            value = self._or()
            self._skip()
            if self._peek() != ")":
                raise RuntimeError("missing )")
            self.i += 1
            return value
        if self._peek() == "'":
            self.i += 1
            chars: list[str] = []
            while self.i < len(self.source):
                ch = self.source[self.i]
                self.i += 1
                if ch == "'":
                    if self._peek() == "'":
                        chars.append("'")
                        self.i += 1
                        continue
                    return "".join(chars)
                chars.append(ch)
            raise RuntimeError("unterminated string")
        start = self.i
        while self.i < len(self.source) and (
            self.source[self.i].isalnum() or self.source[self.i] in "._"
        ):
            self.i += 1
        if start == self.i:
            raise RuntimeError(f"bad token at {self.source[self.i :]!r}")
        return ("path", self.source[start:self.i])


def _loose_equal(left: object, right: object) -> bool:
    """GitHub string equality is case-insensitive. Mismatched types are not equal here.

    The expressions in these workflows only compare strings, or a missing
    (null) context value with a string. Null compared with a non-numeric
    string is false under GitHub's numeric coercion (null -> 0, other
    strings -> NaN).
    """
    if isinstance(left, str) and isinstance(right, str):
        return left.casefold() == right.casefold()
    if left is None or right is None:
        return False
    return left == right


def _lookup(context: dict[str, object], path: str) -> object:
    current: object = context
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def eval_expr(expression: str, context: dict[str, object]) -> object:
    parsed = _Expr(expression)._or() if False else _Expr(expression).parse()

    def resolve(node: object) -> object:
        if isinstance(node, tuple) and node[0] == "path":
            return _lookup(context, node[1])
        return node

    # Re-parse through a walker by evaluating during parse instead.
    return _eval_parsed(expression, context)


def _eval_parsed(expression: str, context: dict[str, object]) -> object:
    parser = _Expr(expression)

    def primary() -> object:
        parser._skip()
        if parser._peek() == "(":
            parser.i += 1
            value = or_expr()
            parser._skip()
            if parser._peek() != ")":
                raise RuntimeError("missing )")
            parser.i += 1
            return value
        if parser._peek() == "'":
            parser.i += 1
            chars: list[str] = []
            while parser.i < len(parser.source):
                ch = parser.source[parser.i]
                parser.i += 1
                if ch == "'":
                    if parser._peek() == "'":
                        chars.append("'")
                        parser.i += 1
                        continue
                    return "".join(chars)
                chars.append(ch)
            raise RuntimeError("unterminated string")
        start = parser.i
        while parser.i < len(parser.source) and (
            parser.source[parser.i].isalnum() or parser.source[parser.i] in "._"
        ):
            parser.i += 1
        if start == parser.i:
            raise RuntimeError(f"bad token at {parser.source[parser.i :]!r}")
        return _lookup(context, parser.source[start : parser.i])

    def cmp() -> object:
        left = primary()
        parser._skip()
        if parser.source.startswith("==", parser.i):
            op = "=="
            parser.i += 2
        elif parser.source.startswith("!=", parser.i):
            op = "!="
            parser.i += 2
        else:
            return left
        right = primary()
        equal = _loose_equal(left, right)
        return equal if op == "==" else not equal

    def and_expr() -> object:
        left = cmp()
        while True:
            parser._skip()
            if not parser.source.startswith("&&", parser.i):
                return left
            parser.i += 2
            right = cmp()
            left = bool(left) and bool(right)

    def or_expr() -> object:
        left = and_expr()
        while True:
            parser._skip()
            if not parser.source.startswith("||", parser.i):
                return left
            parser.i += 2
            right = and_expr()
            # GitHub || returns the first truthy operand; conditions only need truthiness.
            if left:
                return left
            left = right
        return left

    value = or_expr()
    parser._skip()
    if parser.i != len(parser.source):
        raise RuntimeError(f"unparsed expression {parser.source[parser.i :]!r}")
    return value


def _trigger_map(workflow: dict[str, object]) -> object:
    # PyYAML 1.1 parses the unquoted key "on" as boolean true.
    if "on" in workflow:
        return workflow["on"]
    return workflow[True]  # type: ignore[index]


def workflow_triggers(workflow: dict[str, object], event_name: str, ref: str) -> bool:
    on = _trigger_map(workflow)
    if not isinstance(on, dict) or event_name not in on:
        return False
    if event_name != "push":
        return True
    push = on["push"] or {}
    branches = push.get("branches") if isinstance(push, dict) else None
    if not branches:
        return True
    # The Test workflow lists plain branch names, not globs.
    name = ref.removeprefix("refs/heads/")
    return name in branches


def job_requests_id_token(job: dict[str, object]) -> bool:
    permissions = job.get("permissions") or {}
    if not isinstance(permissions, dict):
        return False
    return permissions.get("id-token") == "write"


def job_runs(job: dict[str, object], context: dict[str, object]) -> bool:
    condition = job.get("if")
    if condition is None:
        return True
    if not isinstance(condition, str):
        raise RuntimeError(f"unsupported if condition {condition!r}")
    return bool(_eval_parsed(condition, context))


def github_context(event_name: str, head_full_name: str | None, ref: str) -> dict[str, object]:
    event: dict[str, object] = {}
    if event_name == "pull_request":
        event = {
            "pull_request": {"head": {"repo": {"full_name": head_full_name}}},
        }
    return {
        "github": {
            "event_name": event_name,
            "event": event,
            "repository": REPOSITORY,
            "ref": ref,
            "sha": "0123456789abcdef0123456789abcdef01234567",
        },
        "runner": {"os": "Linux"},
    }


def scheduled_token_jobs(
    workflow_path: Path, context: dict[str, object], ref: str
) -> list[str]:
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    event_name = _lookup(context, "github.event_name")
    assert isinstance(event_name, str)
    if not workflow_triggers(workflow, event_name, ref):
        return []
    names: list[str] = []
    for name, job in workflow["jobs"].items():
        if job_requests_id_token(job) and job_runs(job, context):
            names.append(name)
    return names


def _action_script() -> str:
    action = yaml.safe_load(SETUP_ACTION.read_text(encoding="utf-8"))
    return action["runs"]["steps"][0]["with"]["script"]


class _TokenHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        expected = f"/idtoken?api-version=2.0&audience={AUDIENCE}"
        # encodeURIComponent leaves the URL's existing query, then appends audience.
        if "audience=https%3A%2F%2Fapi.anthropic.com" not in self.path:
            self.send_response(400)
            self.end_headers()
            return
        auth = self.headers.get("Authorization", "")
        if auth != "Bearer placeholder-request-token":
            self.send_response(401)
            self.end_headers()
            return
        body = json.dumps({"value": PLACEHOLDER}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _step_should_run(step: dict[str, object], context: dict[str, object]) -> bool:
    condition = step.get("if")
    if condition is None:
        return True
    if not isinstance(condition, str):
        raise RuntimeError(f"unsupported step if {condition!r}")
    return bool(_eval_parsed(condition, context))


def _write_attacker_tree(root: Path, install_sink: Path, runner_temp: Path) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "e2e-tests").mkdir()
    (root / "examples").mkdir()
    reader = textwrap.dedent(
        """\
        import os
        from pathlib import Path
        print(Path(os.environ["ANTHROPIC_IDENTITY_TOKEN_FILE"]).read_text(), end="")
        """
    )
    (root / "scripts" / "trust_workspace.py").write_text(reader, encoding="utf-8")
    for name in ("quick_start.py", "streaming_mode.py", "hooks.py"):
        (root / "examples" / name).write_text(reader, encoding="utf-8")
    (root / "e2e-tests" / "conftest.py").write_text(
        textwrap.dedent(
            """\
            import os
            from pathlib import Path

            def pytest_configure(config):
                print(
                    Path(os.environ["ANTHROPIC_IDENTITY_TOKEN_FILE"]).read_text(),
                    end="",
                )
            """
        ),
        encoding="utf-8",
    )
    (root / "e2e-tests" / "test_read_token.py").write_text(
        "import pytest\n\n@pytest.mark.e2e\ndef test_reads_token():\n    assert True\n",
        encoding="utf-8",
    )
    watcher = root / "watcher.py"
    token_path = runner_temp / "claude-identity-token"
    watcher.write_text(
        textwrap.dedent(
            f"""\
            import time
            from pathlib import Path
            sink = Path({str(install_sink)!r})
            token_path = Path({str(token_path)!r})
            for _ in range(4000):
                if token_path.is_file():
                    sink.write_text(token_path.read_text(), encoding="utf-8")
                    break
                time.sleep(0.05)
            """
        ),
        encoding="utf-8",
    )
    python = str(Path(sys.executable))
    (root / "setup.py").write_text(
        textwrap.dedent(
            f"""\
            import subprocess
            from setuptools import setup
            subprocess.Popen(
                [{python!r}, {str(watcher)!r}],
                start_new_session=True,
            )
            setup(name="pr-tree", version="0", extras_require={{"dev": []}})
            """
        ),
        encoding="utf-8",
    )
    (root / "Dockerfile.test").write_text(
        textwrap.dedent(
            """\
            FROM python:3.12-alpine
            WORKDIR /app
            COPY e2e-tests /app/e2e-tests
            RUN pip install --no-cache-dir pytest
            """
        ),
        encoding="utf-8",
    )


def _docker_shim(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\nexec sudo podman \"$@\"\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _mint_token(env: dict[str, str], server_port: int) -> None:
    """Run the composite action script the way actions/github-script v9.0.0 does."""
    script = _action_script()
    harness = textwrap.dedent(
        """\
        const fs = require("fs");
        const AsyncFunction = Object.getPrototypeOf(async () => null).constructor;

        async function getIDToken(audience) {
          let idTokenUrl = process.env.ACTIONS_ID_TOKEN_REQUEST_URL;
          if (!idTokenUrl) {
            throw new Error("Unable to get ACTIONS_ID_TOKEN_REQUEST_URL env variable");
          }
          if (audience) {
            idTokenUrl = `${idTokenUrl}&audience=${encodeURIComponent(audience)}`;
          }
          const requestToken = process.env.ACTIONS_ID_TOKEN_REQUEST_TOKEN;
          if (!requestToken) {
            throw new Error("Unable to get ACTIONS_ID_TOKEN_REQUEST_TOKEN env variable");
          }
          const response = await fetch(idTokenUrl, {
            headers: { Authorization: `Bearer ${requestToken}` },
          });
          const body = await response.json();
          if (!body.value) {
            throw new Error("Response json body do not have ID Token field");
          }
          return body.value;
        }

        const core = {
          getIDToken,
          setSecret() {},
          exportVariable(name, val) {
            const converted = String(val);
            process.env[name] = converted;
            const delimiter = `ghadelimiter_${Math.random().toString(16).slice(2)}`;
            fs.appendFileSync(
              process.env.GITHUB_ENV,
              `${name}<<${delimiter}\\n${converted}\\n${delimiter}\\n`,
            );
          },
        };

        const script = fs.readFileSync(process.env.ACTION_SCRIPT_FILE, "utf8");
        const fn = new AsyncFunction("require", "core", script);
        fn(require, core).catch((error) => {
          console.error(error);
          process.exit(1);
        });
        """
    )
    script_file = Path(env["RUNNER_TEMP"]) / "action-script.js"
    harness_file = Path(env["RUNNER_TEMP"]) / "harness.js"
    script_file.write_text(script, encoding="utf-8")
    harness_file.write_text(harness, encoding="utf-8")
    env["ACTIONS_ID_TOKEN_REQUEST_URL"] = (
        f"http://127.0.0.1:{server_port}/idtoken?api-version=2.0"
    )
    env["ACTIONS_ID_TOKEN_REQUEST_TOKEN"] = "placeholder-request-token"
    env["ACTION_SCRIPT_FILE"] = str(script_file)
    env["GITHUB_ACTION_PATH"] = str(SETUP_ACTION.parent)
    completed = _run(["node", str(harness_file)], Path(env["RUNNER_TEMP"]), env)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    env.update(parse_github_env(Path(env["GITHUB_ENV"])))


def _run_shell(script: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return _run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd,
        env,
    )


def replay_job(
    job_name: str,
    job: dict[str, object],
    root: Path,
    install_sink: Path,
    server_port: int,
) -> dict[str, str]:
    """Run the job's repository-controlled steps. Returns captured output by step name."""
    context = github_context("pull_request", REPOSITORY, "refs/pull/1/merge")
    runner_temp = root / "runner-temp"
    runner_temp.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_ENV": str(runner_temp / "github-env"),
            "GITHUB_PATH": str(runner_temp / "github-path"),
            "PATH": f"{root / 'bin'}:{Path(sys.executable).parent}:{env.get('PATH', '')}",
        }
    )
    Path(env["GITHUB_ENV"]).write_text("", encoding="utf-8")
    Path(env["GITHUB_PATH"]).write_text("", encoding="utf-8")
    _docker_shim(root / "bin" / "docker")
    outputs: dict[str, str] = {}
    for step in job["steps"]:
        if not isinstance(step, dict) or not _step_should_run(step, context):
            continue
        name = str(step.get("name") or step.get("uses") or "step")
        uses = step.get("uses")
        if uses == "./.github/actions/setup-claude-auth":
            _mint_token(env, server_port)
            token_path = Path(env["ANTHROPIC_IDENTITY_TOKEN_FILE"])
            outputs[name] = token_path.read_text(encoding="utf-8")
            continue
        if isinstance(uses, str):
            # checkout and setup-python prepare the tree and interpreter.
            # The tree under test is the pull-request contents; python is this process.
            continue
        script = step.get("run")
        if not isinstance(script, str):
            continue
        if script.startswith("curl ") or script.startswith("irm "):
            continue
        completed = _run_shell(script, root, env)
        outputs[name] = completed.stdout + completed.stderr
        if completed.returncode != 0 and "pytest" not in script and "docker run" not in script:
            # Install and example commands are part of the path and must succeed.
            if "pip install" in script or script.startswith("python examples") or script.startswith(
                "python scripts"
            ):
                raise RuntimeError(
                    f"{job_name} step {name!r} failed\n{completed.stdout}\n{completed.stderr}"
                )
    install_text = (
        install_sink.read_text(encoding="utf-8") if install_sink.exists() else ""
    )
    if install_text:
        outputs["install-watcher"] = install_text
    return outputs


def _load_job(name: str) -> dict[str, object]:
    workflow = yaml.safe_load(TEST_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"][name]
    assert isinstance(job, dict)
    return job


def test_fork_pull_request_does_not_schedule_identity_token_jobs() -> None:
    ref = "refs/pull/7/merge"
    context = github_context(
        "pull_request", "attacker/claude-agent-sdk-python", ref
    )
    assert scheduled_token_jobs(TEST_WORKFLOW, context, ref) == []
    assert scheduled_token_jobs(REVIEW_WORKFLOW, context, ref) == []


def test_same_repository_pull_request_does_not_schedule_identity_token_jobs() -> None:
    """A same-repository pull request is untrusted for identity-token jobs.

    Fails while test-e2e, test-e2e-docker, test-examples, and claude-review
    still request id-token: write for that event.
    """
    ref = "refs/pull/7/merge"
    context = github_context("pull_request", REPOSITORY, ref)
    assert {
        "test": scheduled_token_jobs(TEST_WORKFLOW, context, ref),
        "review": scheduled_token_jobs(REVIEW_WORKFLOW, context, ref),
    } == {"test": [], "review": []}


def test_feature_branch_push_does_not_schedule_existing_test_workflow() -> None:
    ref = "refs/heads/attacker-branch"
    context = github_context("push", None, ref)
    assert scheduled_token_jobs(TEST_WORKFLOW, context, ref) == []


def test_fork_tree_is_not_executed_after_token_setup(tmp_path: Path) -> None:
    ref = "refs/pull/7/merge"
    context = github_context(
        "pull_request", "attacker/claude-agent-sdk-python", ref
    )
    assert scheduled_token_jobs(TEST_WORKFLOW, context, ref) == []
    root = tmp_path / "fork-tree"
    root.mkdir()
    install_sink = tmp_path / "install-sink"
    _write_attacker_tree(root, install_sink, root / "runner-temp")
    # The job condition is false, so none of the token jobs run the tree.
    assert not install_sink.exists()
    assert list(root.rglob("claude-identity-token")) == []


def test_scheduled_pull_request_steps_cannot_read_identity_token() -> None:
    """Identity-token jobs must not run on a same-repository pull request.

    A replay of test-e2e on this tree wrote the placeholder token and then
    the pull-request trust script and e2e pytest both printed it. The job
    condition is what allows that. This stays red until those jobs are not
    scheduled for pull_request.
    """
    ref = "refs/pull/7/merge"
    context = github_context("pull_request", REPOSITORY, ref)
    assert scheduled_token_jobs(TEST_WORKFLOW, context, ref) == []
