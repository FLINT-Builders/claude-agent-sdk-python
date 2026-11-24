"""Current-directory Claude executables must not be discovered or spawned.

On Windows, Python's ``shutil.which`` searches the process cwd before PATH
(3.10 and 3.11 always; 3.12 and later unless
``NoDefaultCurrentDirectoryInExePath`` is set). This file drives that search
through the real ``shutil.which`` and then through ``connect()``, which spawns
whatever path discovery returned.

The host here is POSIX, so the Windows branch of ``which`` is switched on for
the duration of each test and ``_winapi.NeedCurrentDirectoryForExePath`` is
supplied from the Win32 contract. ``which`` is also wrapped so that setting
``NoDefaultCurrentDirectoryInExePath`` does not disable the cwd search:
supported 3.10 and 3.11 interpreters never consult that variable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anyio
import pytest

from claude_agent_sdk._errors import CLINotFoundError
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from claude_agent_sdk.types import ClaudeAgentOptions

# Default Windows PATHEXT, as CPython's shutil._WIN_DEFAULT_PATHEXT.
_WIN_PATHEXT = ".COM;.EXE;.BAT;.CMD;.VBS;.JS;.WS;.MSC"
_PARENT_SENTINEL = "placeholder-parent"
_OPTION_SENTINEL = "placeholder-option"
_PROMPT = "placeholder-prompt"
_ENV_KEYS = (
    "PATH",
    "PATHEXT",
    "HOME",
    "USERPROFILE",
    "LOCALAPPDATA",
    "SENTINEL_PARENT",
    "SENTINEL_OPTION",
    "NoDefaultCurrentDirectoryInExePath",
    "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK",
)


class _WinApi:
    """``_winapi.NeedCurrentDirectoryForExePath`` for a POSIX interpreter.

    CPython 3.12 ``Modules/_winapi.c`` forwards this to
    ``NeedCurrentDirectoryForExePathW``. That API returns TRUE when the
    executable name contains a backslash. Otherwise it returns FALSE only
    when ``NoDefaultCurrentDirectoryInExePath`` is present in the environment,
    regardless of the variable's value.
    """

    @staticmethod
    def NeedCurrentDirectoryForExePath(exe_name: str) -> bool:  # noqa: N802
        if "\\" in exe_name:
            return True
        return "NoDefaultCurrentDirectoryInExePath" not in os.environ


_ORIGINAL_WHICH = shutil.which


def _which_ignoring_cwd_opt_out(
    cmd: str,
    mode: int = os.F_OK | os.X_OK,
    path: str | None = None,
) -> str | None:
    """Call the real ``shutil.which`` with cwd search left on.

    Python 3.10 and 3.11 prepend the cwd on Windows without reading
    ``NoDefaultCurrentDirectoryInExePath``. Dropping the variable for the
    call makes 3.12's ``which`` follow that older rule, then runs the real
    PATHEXT walk.
    """
    saved = os.environ.pop("NoDefaultCurrentDirectoryInExePath", None)
    try:
        return _ORIGINAL_WHICH(cmd, mode=mode, path=path)
    finally:
        if saved is not None:
            os.environ["NoDefaultCurrentDirectoryInExePath"] = saved  # noqa: SIM112


@contextmanager
def _windows_cli_search(path_entries: list[str], home: Path) -> Iterator[None]:
    """Make ``shutil.which('claude')`` use the Windows cwd-first search."""
    saved_env = {key: os.environ.get(key) for key in _ENV_KEYS}
    saved_platform = sys.platform
    saved_winapi = shutil._winapi
    saved_which = shutil.which
    try:
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["PATH"] = os.pathsep.join(path_entries)
        os.environ["PATHEXT"] = (
            _WIN_PATHEXT if os.name == "nt" else _WIN_PATHEXT.replace(";", os.pathsep)
        )
        os.environ["HOME"] = str(home)
        os.environ["USERPROFILE"] = str(home)
        os.environ["LOCALAPPDATA"] = str(home)
        os.environ["SENTINEL_PARENT"] = _PARENT_SENTINEL
        if os.name != "nt":
            sys.platform = "win32"
            shutil._winapi = _WinApi()
        # Callers look the function up on the shutil module at each call.
        shutil.which = _which_ignoring_cwd_opt_out  # type: ignore[method-assign]
        with patch("platform.system", return_value="Windows"):
            yield
    finally:
        shutil.which = saved_which  # type: ignore[method-assign]
        shutil._winapi = saved_winapi
        sys.platform = saved_platform
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_executable(path: Path, identity: str, log_path: Path) -> None:
    """Write a POSIX script the SDK can spawn, tagged with ``identity``."""
    path.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import json",
                "import os",
                "import sys",
                "from pathlib import Path",
                f"log_path = Path({str(log_path)!r})",
                f"identity = {identity!r}",
                "argv0 = sys.argv[0]",
                "record = {",
                '    "identity": identity,',
                '    "argv": sys.argv,',
                '    "argv0": argv0,',
                '    "argv0_abs": os.path.isabs(argv0),',
                '    "resolved": str(Path(argv0).resolve()),',
                '    "cwd": os.getcwd(),',
                '    "parent_sentinel": os.environ.get("SENTINEL_PARENT"),',
                '    "option_sentinel": os.environ.get("SENTINEL_OPTION"),',
                '    "stdin": None,',
                "}",
                "def dump(row):",
                "    log_path.parent.mkdir(parents=True, exist_ok=True)",
                "    with log_path.open('a', encoding='utf-8') as handle:",
                "        handle.write(json.dumps(row) + '\\n')",
                "        handle.flush()",
                "        os.fsync(handle.fileno())",
                "if '-v' in sys.argv:",
                "    dump(record)",
                "    sys.stdout.write('2.1.207\\n')",
                "    sys.stdout.flush()",
                "    raise SystemExit(0)",
                "dump(record)",
                "record['stdin'] = sys.stdin.read()",
                "dump(record)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


async def _wait_rows(
    path: Path, predicate: Any, timeout: float = 5
) -> list[dict[str, Any]]:
    deadline = anyio.current_time() + timeout
    rows = _read_jsonl(path)
    while not predicate(rows) and anyio.current_time() < deadline:
        await anyio.sleep(0.02)
        rows = _read_jsonl(path)
    return rows


def _load_download_cli() -> Any:
    name = "download_cli_cwd_lookup_under_test"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    script = Path(__file__).parent.parent / "scripts" / "download_cli.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Layout:
    def __init__(self, root: Path, plant_ext: str) -> None:
        self.process_cwd = root / "process-cwd"
        self.session_cwd = root / "session-cwd"
        self.good_dir = root / "path-dir"
        self.home = root / "home"
        self.logs = root / "logs"
        for directory in (
            self.process_cwd,
            self.session_cwd,
            self.good_dir,
            self.home,
            self.logs,
        ):
            directory.mkdir()
        self.plant_name = "claude" + plant_ext
        self.good_name = "claude.EXE"
        self.process_plant = self.process_cwd / self.plant_name
        self.session_plant = self.session_cwd / self.plant_name
        self.good = self.good_dir / self.good_name
        _write_executable(
            self.process_plant, "process-cwd", self.logs / "process-cwd.jsonl"
        )
        _write_executable(
            self.session_plant, "session-cwd", self.logs / "session-cwd.jsonl"
        )
        _write_executable(self.good, "path-cli", self.logs / "path-cli.jsonl")

    def assert_plants_never_started(self) -> None:
        started = {
            name: _read_jsonl(self.logs / name)
            for name in ("process-cwd.jsonl", "session-cwd.jsonl")
        }
        assert started == {"process-cwd.jsonl": [], "session-cwd.jsonl": []}, started


@pytest.mark.parametrize("plant_ext", [".EXE", ".COM"])
def test_connect_spawns_path_cli_not_cwd_plant(plant_ext: str, tmp_path: Path) -> None:
    """A cwd ``claude`` executable must not run, even for the version probe.

    PATH lists an empty entry and then a directory with a real CLI. The empty
    entry and the implicit cwd search both resolve to the planted file. The
    session cwd (``options.cwd``) contains another planted file of the same
    name, so a relative path re-resolved at spawn time is a different file
    from the one found at discovery. Both probes and the session process must
    be the absolute PATH executable, and the session process must see
    ``options.env`` and stdin.
    """
    layout = _Layout(tmp_path, plant_ext)

    async def _test() -> None:
        previous = Path.cwd()
        os.chdir(layout.process_cwd)
        try:
            with _windows_cli_search(["", str(layout.good_dir)], layout.home):
                discovered = shutil.which("claude")
                assert discovered is not None
                assert Path(discovered).resolve() == layout.process_plant.resolve(), (
                    discovered
                )
                transport = SubprocessCLITransport(
                    prompt=_PROMPT,
                    options=ClaudeAgentOptions(
                        cwd=layout.session_cwd,
                        env={"SENTINEL_OPTION": _OPTION_SENTINEL},
                    ),
                )
                bundled = transport._find_bundled_cli()
                assert bundled is None, bundled
                rows = []
                try:
                    await transport.connect()
                    rows = await _wait_rows(
                        layout.logs / "path-cli.jsonl",
                        lambda found: bool(
                            _read_jsonl(layout.logs / "process-cwd.jsonl")
                            or _read_jsonl(layout.logs / "session-cwd.jsonl")
                            or any("--output-format" in row["argv"] for row in found)
                        ),
                    )
                    if any("--output-format" in row["argv"] for row in rows):
                        await transport.write(_PROMPT)
                        await transport.end_input()
                        rows = await _wait_rows(
                            layout.logs / "path-cli.jsonl",
                            lambda found: any(
                                row["stdin"] is not None for row in found
                            ),
                        )
                finally:
                    await transport.close()
        finally:
            os.chdir(previous)

        layout.assert_plants_never_started()
        assert rows, "PATH CLI never started"
        assert {row["identity"] for row in rows} == {"path-cli"}
        version_rows = [row for row in rows if "-v" in row["argv"]]
        session_rows = [row for row in rows if "--output-format" in row["argv"]]
        stdin_rows = [row for row in session_rows if row["stdin"] is not None]
        assert len(version_rows) == 1
        assert len(session_rows) == 1
        assert len(stdin_rows) == 1
        assert version_rows[0]["argv0"] == session_rows[0]["argv0"]
        assert version_rows[0]["argv0_abs"] is True
        assert Path(version_rows[0]["argv0"]).resolve() == layout.good.resolve()
        assert version_rows[0]["cwd"] == str(layout.process_cwd)
        assert session_rows[0]["cwd"] == str(layout.session_cwd)
        assert session_rows[0]["parent_sentinel"] == _PARENT_SENTINEL
        assert session_rows[0]["option_sentinel"] == _OPTION_SENTINEL
        assert stdin_rows[0]["stdin"] == _PROMPT
        assert "stream-json" in session_rows[0]["argv"]

    anyio.run(_test)


@pytest.mark.parametrize("plant_ext", [".EXE", ".COM"])
def test_connect_does_not_spawn_cwd_plant_when_path_has_no_cli(
    plant_ext: str, tmp_path: Path
) -> None:
    """With no PATH CLI, discovery raises and the planted file never starts."""
    layout = _Layout(tmp_path, plant_ext)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    async def _test() -> None:
        previous = Path.cwd()
        os.chdir(layout.process_cwd)
        raised: CLINotFoundError | None = None
        try:
            with _windows_cli_search(["", str(empty_dir)], layout.home):
                discovered = shutil.which("claude")
                assert discovered is not None
                assert Path(discovered).resolve() == layout.process_plant.resolve()
                transport = SubprocessCLITransport(
                    prompt=_PROMPT,
                    options=ClaudeAgentOptions(
                        cwd=layout.session_cwd,
                        env={"SENTINEL_OPTION": _OPTION_SENTINEL},
                    ),
                )
                try:
                    await transport.connect()
                except CLINotFoundError as exc:
                    raised = exc
                finally:
                    await transport.close()
        finally:
            os.chdir(previous)
        assert raised is not None
        layout.assert_plants_never_started()
        assert _read_jsonl(layout.logs / "path-cli.jsonl") == []

    anyio.run(_test)


def test_download_script_ignores_cwd_plant(tmp_path: Path) -> None:
    """``scripts/download_cli.py`` uses the same ``shutil.which('claude')``."""
    layout = _Layout(tmp_path, ".EXE")
    download_cli = _load_download_cli()
    previous = Path.cwd()
    os.chdir(layout.process_cwd)
    try:
        with _windows_cli_search(["", str(layout.good_dir)], layout.home):
            found = download_cli.find_installed_cli()
            assert found is not None
            assert found.is_absolute()
            assert found.resolve() == layout.good.resolve()
        with _windows_cli_search([""], layout.home):
            assert download_cli.find_installed_cli() is None
    finally:
        os.chdir(previous)
    layout.assert_plants_never_started()
