"""The native-write witness is a strict, credential-free boundary."""

from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ccgram.topic_mark_types import TopicMarkWriteGuard, TopicMarkWriteWitness


def witness_row() -> dict[str, str | int | None]:
    return {
        "dispatch_id": "dispatch-1",
        "backend": "tmux",
        "host_id": "host-1",
        "boot_id": "boot-1",
        "namespace_id": "darwin-local",
        "native_pid": 123,
        "native_start": "darwin:100:200",
        "code_id": "cdhash:abcdef",
        "socket_dev": 0,
        "socket_ino": 456,
        "pin_path": "/private/tmp/owned/pin.sock",
    }


@pytest.mark.parametrize("backend", ["agterm", "herdr", "tmux"])
@pytest.mark.parametrize(
    "pin_path", [None, "/private/tmp/owned/pin.sock", "/tmp/with space/pin"]
)
def test_witness_round_trips_json(backend: str, pin_path: str | None):
    row = witness_row() | {"backend": backend, "pin_path": pin_path}

    witness = TopicMarkWriteWitness.from_dict(json.loads(json.dumps(row)))

    assert witness.to_dict() == row
    assert TopicMarkWriteWitness.from_dict(witness.to_dict()) == witness


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        (field, invalid)
        for field in (
            "dispatch_id",
            "backend",
            "host_id",
            "boot_id",
            "namespace_id",
            "native_start",
            "code_id",
        )
        for invalid in (
            "",
            " leading",
            "trailing ",
            "two words",
            "tab\there",
            "nul\0here",
            "line\nhere",
            None,
            1,
            True,
        )
    ]
    + [("backend", "auto"), ("backend", "TMUX"), ("backend", "socat")]
    + [
        (field, invalid)
        for field in ("native_pid", "socket_dev", "socket_ino")
        for invalid in (True, False, 1.0, "1", None, -1)
    ]
    + [("native_pid", 0), ("socket_ino", 0)]
    + [
        ("pin_path", invalid)
        for invalid in (
            "",
            "/",
            "relative.sock",
            "/tmp/../pin",
            "/tmp/./pin",
            "/tmp//pin",
            "//tmp/pin",
            "/tmp/pin/",
            "/tmp/pin\0",
            "/tmp/pin\n",
            True,
            1,
        )
    ],
)
def test_witness_rejects_malformed_field(field: str, invalid: object):
    row: dict[str, object] = dict(witness_row())
    row[field] = invalid

    with pytest.raises(ValueError):
        TopicMarkWriteWitness.from_dict(row)


@pytest.mark.parametrize("raw", [None, True, [], "witness", 1])
def test_witness_requires_an_object(raw: object):
    with pytest.raises(ValueError):
        TopicMarkWriteWitness.from_dict(raw)


@pytest.mark.parametrize("missing", list(witness_row()))
def test_witness_rejects_missing_key(missing: str):
    row = witness_row()
    del row[missing]

    with pytest.raises(ValueError):
        TopicMarkWriteWitness.from_dict(row)


def test_witness_rejects_extra_keys():
    with pytest.raises(ValueError):
        TopicMarkWriteWitness.from_dict(witness_row() | {"selected": True})


def test_witness_constructor_validates_and_freezes():
    witness = TopicMarkWriteWitness.from_dict(witness_row())

    with pytest.raises(ValueError):
        replace(witness, native_pid=True)
    with pytest.raises(FrozenInstanceError):
        setattr(witness, "native_pid", 999)
    assert not hasattr(witness, "__dict__")


def test_guard_accepts_witness_and_exact_dispatch():
    events: list[object] = []

    class Recorder:
        def arm(self, witness: TopicMarkWriteWitness) -> None:
            events.append(witness)

        def terminal(self, dispatch_id: str) -> None:
            events.append(dispatch_id)

    guard: TopicMarkWriteGuard = Recorder()
    witness = TopicMarkWriteWitness.from_dict(witness_row())
    guard.arm(witness)
    guard.terminal(witness.dispatch_id)

    assert events == [witness, "dispatch-1"]


def test_witness_loads_without_package_or_credentials():
    spec = importlib.util.find_spec("ccgram.topic_mark_types")
    assert spec is not None and spec.origin is not None
    path = Path(spec.origin).resolve()
    script = """
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location("isolated_witness", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
row = json.loads(sys.argv[2])
assert module.TopicMarkWriteWitness.from_dict(row).to_dict() == row
assert not any(name == "ccgram" or name.startswith("ccgram.") for name in sys.modules)
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CCGRAM_", "AGTERM_", "HERDR_", "TMUX"))
        and key not in {"TELEGRAM_BOT_TOKEN", "ALLOWED_USERS"}
    }

    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(path), json.dumps(witness_row())],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
