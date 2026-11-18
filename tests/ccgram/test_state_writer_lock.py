"""Only one process may own a loaded state snapshot, including offline CLI."""

from __future__ import annotations

import os
from pathlib import Path
import select
import subprocess
import sys

import pytest

import ccgram
from ccgram.state_writer_lock import StateWriterLease
from ccgram.utils import atomic_write_json

assert ccgram.__file__ is not None
_ROOT = Path(ccgram.__file__).resolve().parents[2]


def _child_env(tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "PYTHONPATH": str(_ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _child_acquires(path: Path) -> bool:
    code = (
        "import sys; from pathlib import Path; "
        "from ccgram.state_writer_lock import StateWriterLease; "
        "lease = StateWriterLease(Path(sys.argv[1])); "
        "print(int(lease.try_acquire())); lease.close()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        env=_child_env(path.parent),
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return result.stdout.strip() == "1"


def test_separate_process_lease_survives_atomic_state_replacement(tmp_path):
    path = tmp_path / "state.json"
    lock_path = tmp_path / "state.json.lock"

    with StateWriterLease(path) as lease:
        identity = lock_path.stat()
        assert lease.state_path == path.resolve()
        assert not _child_acquires(path)
        atomic_write_json(path, {"version": 1})
        atomic_write_json(path, {"version": 2})
        assert not _child_acquires(path)
        current = lock_path.stat()
        assert (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino)

    assert lock_path.exists()
    assert _child_acquires(path)
    current = lock_path.stat()
    assert (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino)


def test_writer_process_death_releases_lease_without_removing_lock(tmp_path):
    path = tmp_path / "state.json"
    code = (
        "import sys; from pathlib import Path; "
        "from ccgram.state_writer_lock import StateWriterLease; "
        "lease = StateWriterLease(Path(sys.argv[1])); "
        "assert lease.try_acquire(); print('held', flush=True); sys.stdin.read()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        env=_child_env(tmp_path),
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        readable, _, _ = select.select([process.stdout], [], [], 5)
        assert readable, "writer process did not acquire its lease"
        assert process.stdout.readline() == "held\n"
        lease = StateWriterLease(path)
        assert not lease.try_acquire()
        process.kill()
        process.wait(timeout=5)
        assert lease.try_acquire()
        lease.close()
        assert (tmp_path / "state.json.lock").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "permissions", "directory"])
def test_unsafe_lock_path_is_refused_without_modifying_state(tmp_path, unsafe):
    path = tmp_path / "state.json"
    contents = b'{"unrelated":{"keep":true}}\n'
    path.write_bytes(contents)
    lock_path = tmp_path / "state.json.lock"
    target = tmp_path / "unrelated-file"
    target.write_bytes(b"keep")
    target.chmod(0o600)
    if unsafe == "symlink":
        lock_path.symlink_to(target)
    elif unsafe == "hardlink":
        lock_path.hardlink_to(target)
    elif unsafe == "directory":
        lock_path.mkdir()
    else:
        lock_path.write_bytes(b"")
        lock_path.chmod(0o644)

    lease = StateWriterLease(path)
    try:
        with pytest.raises((OSError, RuntimeError)):
            lease.try_acquire()
    finally:
        lease.close()

    assert path.read_bytes() == contents
    assert target.read_bytes() == b"keep"


def test_symlinked_state_path_cannot_split_writer_ownership(tmp_path):
    victim = tmp_path / "real-state.json"
    contents = b'{"unrelated":{"keep":true}}\n'
    victim.write_bytes(contents)
    path = tmp_path / "state.json"
    path.symlink_to(victim)
    lease = StateWriterLease(path)
    try:
        with pytest.raises((OSError, RuntimeError)):
            lease.try_acquire()
    finally:
        lease.close()

    assert path.is_symlink()
    assert path.readlink() == victim
    assert victim.read_bytes() == contents
    assert path.read_bytes() == contents


@pytest.mark.parametrize(
    "mode", [0o775, 0o777], ids=["group-writable", "world-writable"]
)
def test_writable_state_parent_cannot_split_writer_ownership(tmp_path, mode):
    directory = tmp_path / "unsafe"
    directory.mkdir()
    directory.chmod(mode)
    path = directory / "state.json"
    contents = b'{"unrelated":{"keep":true}}\n'
    path.write_bytes(contents)
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep")
    lease = StateWriterLease(path)
    try:
        with pytest.raises((OSError, RuntimeError)):
            lease.try_acquire()
    finally:
        lease.close()

    assert path.read_bytes() == contents
    assert victim.read_bytes() == b"keep"
    assert directory.stat().st_mode & 0o777 == mode


@pytest.mark.parametrize("mode", [0o700, 0o755])
def test_owned_nonwritable_state_parent_remains_supported(tmp_path, mode):
    directory = tmp_path / "owned"
    directory.mkdir()
    directory.chmod(mode)
    path = directory / "state.json"
    contents = b'{"unrelated":{"keep":true}}\n'
    path.write_bytes(contents)

    with StateWriterLease(path) as lease:
        assert lease.held
        assert path.read_bytes() == contents

    assert path.read_bytes() == contents
    assert directory.stat().st_mode & 0o777 == mode
