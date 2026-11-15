"""Saved native process identities cannot fabricate a retirement barrier."""

from dataclasses import replace
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import socket
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ccgram.multiplexer import native_process as native
from ccgram.topic_mark_types import TopicMarkWriteWitness

HOST = "11111111-1111-4111-8111-111111111111"
BOOT = "22222222-2222-4222-8222-222222222222"
START = "darwin:100:200"
CODE = "darwin:tmux:3.8:cdhash:" + "a" * 40


def witness() -> TopicMarkWriteWitness:
    return TopicMarkWriteWitness(
        "1" * 32,
        "tmux",
        HOST,
        BOOT,
        "darwin-local",
        123,
        START,
        CODE,
        0,
        456,
        "/tmp/ccgram-mark-" + "1" * 32 + "/s",
    )


@pytest.fixture
def mac_scope(monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "darwin")
    monkeypatch.setattr(native, "_local_scope", lambda: (HOST, BOOT, "darwin-local"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dispatch_id", "dispatch-1"),
        ("pin_path", None),
        ("native_pid", 2**31),
        ("native_start", "darwin:100"),
        ("native_start", "darwin:0100:200"),
        ("native_start", "darwin:-1:200"),
        ("native_start", "darwin:100:1000000"),
        ("native_start", "darwin:100:NaN"),
        ("native_start", "darwin:99999999999999999999:0"),
        ("native_start", "linux:100"),
        ("socket_ino", 2**64),
        ("socket_dev", 2**64),
        ("host_id", "host"),
        ("boot_id", "boot"),
        ("namespace_id", "foreign"),
        ("code_id", "cdhash:" + "a" * 40),
        ("code_id", "darwin:tmux:3.9:cdhash:" + "a" * 40),
        ("code_id", "darwin:herdr:0.9.5:cdhash:" + "a" * 40),
        ("code_id", "darwin:tmux:3.8:cdhash:" + "a" * 39),
    ],
)
def test_malformed_saved_identity_never_reaches_process_comparison(
    monkeypatch, mac_scope, field: str, value: object
):
    probe = Mock(side_effect=AssertionError("unvalidated process probe"))
    monkeypatch.setattr(native, "_process_start", probe)

    assert native.process_retired(replace(witness(), **{field: value})) is None
    probe.assert_not_called()


@pytest.mark.parametrize(
    ("current", "retired"), [(START, False), ("darwin:101:200", True)]
)
def test_retirement_compares_valid_birth_only(monkeypatch, mac_scope, current, retired):
    monkeypatch.setattr(native, "_process_start", lambda _: current)

    assert native.process_retired(witness()) is retired


def test_missing_process_requires_independent_absence(monkeypatch, mac_scope):
    monkeypatch.setattr(native, "_process_start", lambda _: None)
    monkeypatch.setattr(native.os, "kill", lambda *_: None)
    assert native.process_retired(witness()) is None

    def missing(*_):
        raise ProcessLookupError

    monkeypatch.setattr(native.os, "kill", missing)
    assert native.process_retired(witness()) is True


def test_process_permission_failure_is_unknown(monkeypatch, mac_scope):
    def denied(_):
        raise PermissionError

    monkeypatch.setattr(native, "_process_start", denied)
    assert native.process_retired(witness()) is None


@pytest.mark.parametrize(
    ("scope", "retired"),
    [
        ((BOOT, BOOT, "darwin-local"), None),
        ((HOST, HOST, "darwin-local"), True),
        ((HOST, BOOT, "foreign"), None),
    ],
)
def test_retirement_rejects_foreign_host_or_namespace(
    monkeypatch, mac_scope, scope, retired
):
    monkeypatch.setattr(native, "_local_scope", lambda: scope)
    monkeypatch.setattr(
        native, "_process_start", Mock(side_effect=AssertionError("foreign scope"))
    )

    assert native.process_retired(witness()) is retired


def test_native_code_mismatch_refuses_capture(monkeypatch, mac_scope):
    connection = Mock(spec=socket.socket)
    connection.getsockopt.side_effect = [
        123,
        struct.pack("8I", 0, os.getuid(), 0, 0, 0, 123, 0, 1),
    ]
    monkeypatch.setattr(native, "_native_binary", lambda *_: "/trusted/native")
    monkeypatch.setattr(native, "_process_start", lambda _: START)

    def mismatch(*_):
        raise ValueError("Native code identity refused: -67034")

    monkeypatch.setattr(native, "_MacCode", lambda: SimpleNamespace(verify=mismatch))

    assert (
        native.capture_native_process(connection, backend="tmux", binary="/trusted")
        is None
    )


def test_capture_rejects_birth_change_across_code_check(monkeypatch, mac_scope):
    connection = Mock(spec=socket.socket)
    connection.getsockopt.side_effect = [
        123,
        struct.pack("8I", 0, os.getuid(), 0, 0, 0, 123, 0, 1),
    ]
    monkeypatch.setattr(native, "_native_binary", lambda *_: "/trusted/native")
    monkeypatch.setattr(
        native, "_process_start", Mock(side_effect=[START, "darwin:101:200"])
    )
    monkeypatch.setattr(
        native, "_MacCode", lambda: SimpleNamespace(verify=lambda *_: "a" * 40)
    )

    assert (
        native.capture_native_process(connection, backend="tmux", binary="/trusted")
        is None
    )


def test_capture_preserves_kernel_native_identity(monkeypatch, mac_scope):
    connection = Mock(spec=socket.socket)
    connection.getsockopt.side_effect = [
        123,
        struct.pack("8I", 0, os.getuid(), 0, 0, 0, 123, 0, 1),
    ]
    monkeypatch.setattr(native, "_native_binary", lambda *_: "/trusted/native")
    monkeypatch.setattr(native, "_process_start", lambda _: START)
    monkeypatch.setattr(
        native, "_MacCode", lambda: SimpleNamespace(verify=lambda *_: "a" * 40)
    )

    identity = native.capture_native_process(
        connection, backend="tmux", binary="/trusted"
    )

    assert identity is not None
    assert identity == native.NativeProcessIdentity(
        HOST, BOOT, "darwin-local", 123, START, CODE
    )
    assert identity.scope != replace(identity, start="darwin:101:200").scope


def test_real_own_process_birth_is_canonical_and_stable():
    start = native._process_start(os.getpid())

    assert start is not None and native._valid_start(start, native.sys.platform)
    assert native._process_start(os.getpid()) == start


@pytest.mark.skipif(sys.platform != "darwin", reason="public macOS Security APIs")
def test_real_macos_code_primitive_matches_exact_kernel_peer():
    with TemporaryDirectory(prefix="ccg-code-", dir="/tmp") as directory:
        path = Path(directory) / "s"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            listener.listen(1)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(path))
                token = connection.getsockopt(0, 6, 32)
                code = native._MacCode().verify(token, Path(sys.executable).resolve())
                assert re.fullmatch("[0-9a-f]{40}", code)
                with pytest.raises(ValueError, match="Native code identity refused"):
                    native._MacCode().verify(token, Path("/usr/bin/true"))


@pytest.mark.skipif(sys.platform != "darwin", reason="public macOS host/boot APIs")
def test_real_macos_host_boot_scope_is_canonical():
    host, boot, namespace = native._local_scope()

    assert native._canonical_uuid(host)
    assert native._canonical_uuid(boot)
    assert namespace == "darwin-local"


def test_malformed_current_boot_cannot_fabricate_retirement(monkeypatch, mac_scope):
    monkeypatch.setattr(
        native, "_local_scope", lambda: (HOST, "not-a-boot-id", "darwin-local")
    )
    probe = Mock(side_effect=AssertionError("unproved current boot"))
    monkeypatch.setattr(native, "_process_start", probe)

    assert native.process_retired(witness()) is None
    probe.assert_not_called()


@pytest.fixture
def scope_endpoints(monkeypatch):
    with TemporaryDirectory(prefix="ccg-scope-", dir="/tmp") as directory:
        root = Path(directory).resolve()
        identity = native.NativeProcessIdentity(
            HOST, BOOT, "darwin-local", 123, START, CODE
        )
        monkeypatch.setattr(
            native, "capture_native_process", lambda *_a, **_k: identity
        )
        endpoints = [root / "a", root / "b"]
        listeners = []
        try:
            for path in endpoints:
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                listener.bind(str(path))
                listener.listen(16)
                listeners.append(listener)
            yield endpoints, identity
        finally:
            for listener in listeners:
                listener.close()


def test_scope_distinguishes_endpoints_with_same_executor(scope_endpoints):
    (a, b), _identity = scope_endpoints
    first = native.inspect_endpoint_scope(str(a), backend="tmux", binary="/forbidden")
    second = native.inspect_endpoint_scope(str(b), backend="tmux", binary="/forbidden")

    assert first is not None and second is not None and first != second


def test_scope_alias_and_mutable_metadata_do_not_change_identity(scope_endpoints):
    (a, _b), _identity = scope_endpoints
    original = native.inspect_endpoint_scope(
        str(a), backend="tmux", binary="/forbidden"
    )
    alias = a.parent / "alias"
    os.link(a, alias)
    os.utime(a, None)

    assert original is not None
    assert (
        native.inspect_endpoint_scope(str(alias), backend="tmux", binary="/forbidden")
        == original
    )
    assert (
        native.inspect_endpoint_scope(str(a), backend="tmux", binary="/forbidden")
        == original
    )


def test_readonly_scope_equals_dispatch_descriptor_without_creating_files(
    scope_endpoints,
):
    (a, _b), identity = scope_endpoints
    before = a.lstat()
    entries = set(a.parent.iterdir())

    actual = native.inspect_endpoint_scope(str(a), backend="tmux", binary="/forbidden")

    after = a.lstat()
    assert set(a.parent.iterdir()) == entries
    assert (before.st_nlink, before.st_ctime_ns) == (after.st_nlink, after.st_ctime_ns)
    witness = identity.witness(
        "tmux", after, dispatch_id="a" * 32, pin_path="/tmp/ignored-pin"
    )
    assert actual == witness.endpoint_scope


def test_scope_unknown_peer_or_missing_endpoint_is_not_global(
    scope_endpoints, monkeypatch
):
    (a, _b), _identity = scope_endpoints
    monkeypatch.setattr(native, "capture_native_process", lambda *_a, **_k: None)

    assert (
        native.inspect_endpoint_scope(str(a), backend="tmux", binary="/forbidden")
        is None
    )
    assert (
        native.inspect_endpoint_scope(
            str(a.parent / "missing"), backend="tmux", binary="/forbidden"
        )
        is None
    )


def test_scope_rejects_endpoint_replacement_during_capture(
    scope_endpoints, monkeypatch
):
    (a, _b), identity = scope_endpoints
    replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def changed(*_a, **_k):
        a.unlink()
        replacement.bind(str(a))
        return identity

    monkeypatch.setattr(native, "inspect_native_process", changed)
    try:
        assert (
            native.inspect_endpoint_scope(str(a), backend="tmux", binary="/forbidden")
            is None
        )
    finally:
        replacement.close()
