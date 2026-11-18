"""Exclusive ownership of one state snapshot, shared by daemon and local CLI."""

from __future__ import annotations

import fcntl
import os
import stat
from pathlib import Path
from types import TracebackType

STATE_WRITER_LEASE_KEY = "state_writer_lease"


class StateWriterLease:
    """The lock inode is stable; atomic replacement applies only to state.json."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path.parent.resolve() / state_path.name
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def try_acquire(self) -> bool:
        if self._fd is not None:
            raise RuntimeError("State writer lease is already held")
        if self.state_path.is_symlink():
            raise RuntimeError(
                "State writer ownership cannot follow a state-file symlink"
            )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.state_path.parent.stat()
        # A 0600 lock cannot stop directory writers replacing its inode.
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.getuid()
            or parent.st_mode & 0o022
        ):
            raise RuntimeError("Unsafe state directory ownership or write permissions")
        path = self.state_path.with_name(self.state_path.name + ".lock")
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
            ):
                raise RuntimeError("Unsafe state writer lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            self._fd = fd
            return True
        finally:
            if self._fd != fd:
                os.close(fd)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> StateWriterLease:
        if not self.try_acquire():
            raise RuntimeError("Another process owns the state writer lease")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()
