"""E2E lifecycle test for the OpenCode provider.

Spawns a real ``opencode --pure`` TUI in a PTY inside an isolated
``XDG_DATA_HOME`` (so the real user database is never touched), drives a
prompt through the interactive input box, and verifies that the OpenCode
provider discovers the session, reads the event-sourced transcript, and
parses user / assistant / tool messages from it.

Local only: requires an ``opencode`` binary on PATH, a configured model
provider (the isolated config pins ``deepseek/deepseek-v4-flash``) and
credentials (``~/.local/share/opencode/auth.json``). Skips when those are
unavailable.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import pty
import select
import signal
import sqlite3
import struct
import subprocess
import termios
import time
from pathlib import Path

import pytest

from ccgram.providers.opencode import OpenCodeProvider

_SMOKE_PROMPT = "Reply with exactly the word DONE"
_DB_POLL_SECONDS = 120
_BOX_READY_SECONDS = 30


def _opencode_binary() -> str | None:
    import shutil

    return shutil.which("opencode")


def _has_auth() -> bool:
    return (Path.home() / ".local" / "share" / "opencode" / "auth.json").exists()


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        _opencode_binary() is None, reason="opencode binary not on PATH"
    ),
    pytest.mark.skipif(not _has_auth(), reason="opencode auth.json not found"),
]


def _strip_ansi(text: str) -> str:
    import re

    text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)
    text = re.sub(r"\x1b[()][A-Z0-9]", "", text)
    text = re.sub(r"\x1b\][^\x07]*\x07", "", text)
    return text


class _TuiProcess:
    """A real opencode TUI running in a PTY with an accumulating capture."""

    def __init__(self, workdir: Path, xdg_data: Path, config_dir: Path) -> None:
        self.master_fd, slave_fd = pty.openpty()
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        env = dict(os.environ)
        env["TERM"] = "xterm-256color"
        env["XDG_DATA_HOME"] = str(xdg_data)
        env["OPENCODE_CONFIG_DIR"] = str(config_dir)
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(
            {
                "model": "deepseek/deepseek-v4-flash",
                "autoupdate": False,
                "default_agent": "build",
            }
        )
        self.proc = subprocess.Popen(
            ["opencode", "--pure"],
            cwd=str(workdir),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave_fd)
        self._buffer = b""

    def read_available(self, timeout: float = 0.2) -> None:
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.master_fd], [], [], 0.1)
            if not r:
                continue
            try:
                chunk = os.read(self.master_fd, 65536)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    return
                raise
            if not chunk:
                return
            self._buffer += chunk

    def text(self) -> str:
        return _strip_ansi(self._buffer.decode("utf-8", "replace"))

    def type(self, text: str) -> None:
        os.write(self.master_fd, text.encode())

    def close(self) -> None:
        with contextlib.suppress(OSError):
            os.write(self.master_fd, b"\x03")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.proc.wait()
        with contextlib.suppress(OSError):
            os.close(self.master_fd)


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated opencode DB + config + mirror root, all in tmp_path."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    xdg_data = tmp_path / "xdg"
    xdg_data.mkdir()
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    mirror_root = tmp_path / "mirrors"
    monkeypatch.setenv("CCGRAM_OPENCODE_DB", str(xdg_data / "opencode" / "opencode.db"))
    monkeypatch.setenv("CCGRAM_OPENCODE_DATA_DIR", str(mirror_root))
    return workdir, xdg_data, config_dir, mirror_root


def _wait_for_assistant_text(db_path: Path, deadline: float) -> str | None:
    """Poll the isolated DB until an assistant text part appears."""
    while time.time() < deadline:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error:
            time.sleep(1)
            continue
        try:
            row = conn.execute(
                "SELECT p.data FROM part p JOIN message m ON p.message_id = m.id "
                "WHERE json_extract(p.data, '$.type') = 'text' "
                "AND json_extract(m.data, '$.role') = 'assistant' "
                "AND length(json_extract(p.data, '$.text')) > 0 "
                "ORDER BY p.time_created LIMIT 1"
            ).fetchone()
        except sqlite3.Error:
            row = None
        finally:
            conn.close()
        if row:
            try:
                return str(json.loads(row[0]).get("text") or "")
            except json.JSONDecodeError, TypeError:
                return ""
        time.sleep(1)
    return None


def test_opencode_lifecycle_poll_and_parse(isolated_env: tuple) -> None:
    workdir, xdg_data, config_dir, mirror_root = isolated_env
    tui = _TuiProcess(workdir, xdg_data, config_dir)
    try:
        deadline = time.time() + _BOX_READY_SECONDS
        while time.time() < deadline:
            tui.read_available()
            if "Ask anything" in tui.text():
                break
            time.sleep(0.3)
        else:
            pytest.fail(
                "opencode TUI input box never appeared; output: " + tui.text()[-400:]
            )

        tui.type(_SMOKE_PROMPT)
        time.sleep(1.5)
        tui.read_available()
        if _SMOKE_PROMPT not in tui.text():
            pytest.fail("typed prompt was not echoed by the TUI")
        tui.type("\r")

        db_path = xdg_data / "opencode" / "opencode.db"
        text = _wait_for_assistant_text(db_path, time.time() + _DB_POLL_SECONDS)
        if not text:
            pytest.fail(
                "opencode never wrote an assistant text part; tail: "
                + tui.text()[-400:]
            )

        # ── Provider round-trip ────────────────────────────────────────────
        provider = OpenCodeProvider()
        event = provider.discover_transcript(str(workdir), "ccgram:@e2e")
        assert event is not None, "provider could not discover the live session"
        assert event.cwd == str(workdir.resolve())

        mirror = Path(event.transcript_path)
        assert mirror.exists()

        entries, offset = provider.read_transcript_file(str(mirror), 0)
        assert offset > 0
        messages, _ = provider.parse_transcript_entries(entries, {})
        texts = [(m.role, m.text) for m in messages]
        assert any(role == "user" and _SMOKE_PROMPT in t for role, t in texts)
        assert any(role == "assistant" and "DONE" in t for role, t in texts)
        # reasoning parts stay hidden
        assert not any("The user wants me" in t for _role, t in texts)

        # incremental read does not duplicate
        entries2, offset2 = provider.read_transcript_file(str(mirror), offset)
        assert offset2 >= offset
        messages2, _ = provider.parse_transcript_entries(entries2, {})
        assert messages2 == []

        # resume picker sees the session
        resumable = provider.discover_resumable_sessions(cwd=str(workdir.resolve()))
        assert any(r.session_id == event.session_id for r in resumable)

        # history parsing works for the user prompt
        hist = [
            provider.parse_history_entry(e)
            for e in entries
            if provider.is_user_transcript_entry(e)
        ]
        assert any(m is not None and _SMOKE_PROMPT in m.text for m in hist)
    finally:
        tui.close()
