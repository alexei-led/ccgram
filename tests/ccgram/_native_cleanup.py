"""Ownership-preserving cleanup for native test resources."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Any


def _socket_identity(path: Path) -> tuple[int, int, int] | None:
    try:
        info = path.lstat()
        if stat.S_ISSOCK(info.st_mode):
            return info.st_dev, info.st_ino, info.st_ctime_ns
    except OSError:
        pass
    return None


def tmux_window_cleanup(server: Any, window_id: str) -> Callable[[], None] | None:
    """Capture the actual endpoint before a fixture can change its context."""
    try:
        result = server.cmd("display-message", "-p", "#{socket_path}")
        if result.returncode != 0 or result.stderr or len(result.stdout) != 1:
            return None
        endpoint = Path(result.stdout[0]).absolute()
        if endpoint.is_symlink():
            return None
        endpoint = endpoint.resolve()
        identity = _socket_identity(endpoint)
        if identity is None:
            return None
    except OSError, ValueError, AttributeError, TypeError:
        return None

    def close() -> None:
        if _socket_identity(endpoint) == identity:
            subprocess.run(
                ["tmux", "-S", str(endpoint), "kill-window", "-t", window_id],
                capture_output=True,
                timeout=5,
            )

    return close


def herdr_window_cleanup(manager: Any, target: str) -> Callable[[], None] | None:
    """A created session does not grant ownership of its workspace/checkout."""
    from ccgram.herdr_targets import is_herdr_session_target

    if not is_herdr_session_target(target):
        return None

    def close() -> None:
        asyncio.run(manager.kill_window(target))

    return close


def tmux_test_namespace() -> tuple[Path, Callable[[], None]]:
    root = Path(tempfile.mkdtemp(prefix="ccg-pytest-", dir="/tmp"))
    initial = root.lstat()
    identity = initial.st_dev, initial.st_ino

    def close() -> None:
        try:
            current = root.lstat()
        except FileNotFoundError:
            return
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != identity
        ):
            return
        directory = root / f"tmux-{os.getuid()}"
        if not directory.is_symlink() and directory.is_dir():
            for endpoint in directory.iterdir():
                if _socket_identity(endpoint) is not None:
                    subprocess.run(
                        ["tmux", "-S", str(endpoint), "kill-server"],
                        capture_output=True,
                        timeout=5,
                    )
        shutil.rmtree(root)

    return root, close
