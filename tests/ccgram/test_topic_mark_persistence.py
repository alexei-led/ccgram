"""State corruption must not make possibly-dispatched marker work disappear."""

import json
import os
from pathlib import Path
import stat

import pytest

from ccgram.state_persistence import StatePersistence
from ccgram.utils import atomic_write_json


@pytest.mark.parametrize("contents", [b"{", b"[]", b"null", b"\xff"])
def test_corrupt_state_is_not_an_empty_safe_marker_journal(tmp_path: Path, contents):
    path = tmp_path / "state.json"
    path.write_bytes(contents)
    persistence = StatePersistence(path, lambda: {"replacement": "unsafe"})

    with pytest.raises(ValueError):
        persistence.load()

    assert path.read_bytes() == contents


def test_unreadable_state_cannot_become_an_empty_marker_journal(tmp_path):
    persistence = StatePersistence(tmp_path, lambda: {"replacement": "unsafe"})

    with pytest.raises(OSError):
        persistence.load()

    assert tmp_path.is_dir()


def test_atomic_marker_checkpoint_syncs_its_containing_directory(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    synced = []
    real_fsync = os.fsync

    def fsync(descriptor):
        info = os.fstat(descriptor)
        synced.append((stat.S_ISDIR(info.st_mode), info.st_dev, info.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync)

    atomic_write_json(path, {"topic_mark_operations": []})

    directory = path.parent.stat()
    assert json.loads(path.read_text(encoding="utf-8")) == {"topic_mark_operations": []}
    assert synced[0][0] is False
    assert synced[-1] == (True, directory.st_dev, directory.st_ino)
