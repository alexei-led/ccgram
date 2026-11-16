"""Local native executor identity; an unavailable proof is never retirement."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import logging
import os
from pathlib import Path
import plistlib
import re
import socket
import struct
import stat
import subprocess
import sys
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from ..topic_mark_types import TopicMarkWriteWitness

logger = logging.getLogger(__name__)
_MAX_PID = 2**31 - 1
_MICROSECONDS = 1_000_000
_MAX_UINT64 = 2**64 - 1
_AUDIT_TOKEN_BYTES = 32
_CDHASH_BYTES = 20
_LINUX_NAMESPACE_FIELDS = 5
# These are the versions whose local handler/FD-transfer contracts were reviewed.
_NATIVE_VERSIONS = {"agterm": "0.42.3", "herdr": "0.9.5", "tmux": "3.8"}
_NONCE_RE = re.compile(r"[0-9a-f]{32}")
_DECIMAL_RE = re.compile(r"0|[1-9][0-9]{0,19}")


@dataclass(frozen=True, slots=True)
class NativeProcessIdentity:
    host_id: str
    boot_id: str
    namespace_id: str
    pid: int
    start: str
    code_id: str

    @property
    def scope(self) -> str:
        raw = (self.host_id, self.boot_id, self.namespace_id, self.pid, self.start)
        return hashlib.sha256(
            json.dumps(raw, separators=(",", ":")).encode()
        ).hexdigest()

    def witness(
        self,
        backend: str,
        info: os.stat_result,
        *,
        dispatch_id: str,
        pin_path: str | None,
    ) -> TopicMarkWriteWitness:
        return TopicMarkWriteWitness(
            dispatch_id=dispatch_id,
            backend=backend,
            host_id=self.host_id,
            boot_id=self.boot_id,
            namespace_id=self.namespace_id,
            native_pid=self.pid,
            native_start=self.start,
            code_id=self.code_id,
            socket_dev=info.st_dev,
            socket_ino=info.st_ino,
            pin_path=pin_path,
        )


class _BSDInfo(ctypes.Structure):
    _fields_ = (
        [
            (name, ctypes.c_uint32)
            for name in (
                "flags",
                "status",
                "xstatus",
                "pid",
                "ppid",
                "uid",
                "gid",
                "ruid",
                "rgid",
                "svuid",
                "svgid",
                "reserved",
            )
        ]
        + [
            ("comm", ctypes.c_char * 16),
            ("name", ctypes.c_char * 32),
        ]
        + [
            (name, ctypes.c_uint32)
            for name in ("nfiles", "pgid", "pjobc", "tdev", "tpgid")
        ]
        + [
            ("nice", ctypes.c_int32),
            ("startsec", ctypes.c_uint64),
            ("startusec", ctypes.c_uint64),
        ]
    )


def _bind(library: ctypes.CDLL, name: str, result: Any, *arguments: Any) -> Any:
    function = getattr(library, name)
    function.restype = result
    function.argtypes = list(arguments)
    return function


def _canonical_uuid(value: str) -> bool:
    try:
        return type(value) is str and str(UUID(value)) == value
    except ValueError, AttributeError:
        return False


def _valid_namespace(value: str, platform: str) -> bool:
    if type(value) is not str:
        return False
    if platform == "darwin":
        return value == "darwin-local"
    pieces = value.split(":")
    return (
        platform == "linux"
        and len(pieces) == _LINUX_NAMESPACE_FIELDS
        and pieces[0] == "linux"
        and all(
            _DECIMAL_RE.fullmatch(part) and int(part) <= _MAX_UINT64
            for part in pieces[1:]
        )
        and int(pieces[2]) > 0
        and int(pieces[4]) > 0
    )


def _canonical_scope(host: str, boot: str, namespace: str) -> bool:
    return (
        _canonical_uuid(host)
        and _canonical_uuid(boot)
        and _valid_namespace(namespace, sys.platform)
    )


def _local_scope() -> tuple[str, str, str]:
    if sys.platform == "darwin":
        system = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        host = (ctypes.c_ubyte * 16)()
        deadline = (ctypes.c_long * 2)(1, 0)
        get_host = _bind(
            system, "gethostuuid", ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
        )
        if get_host(host, deadline) != 0:
            raise OSError(ctypes.get_errno(), "gethostuuid failed")
        size = ctypes.c_size_t(128)
        boot = ctypes.create_string_buffer(size.value)
        sysctl = _bind(
            system,
            "sysctlbyname",
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        if sysctl(b"kern.bootsessionuuid", boot, ctypes.byref(size), None, 0) != 0:
            raise OSError(ctypes.get_errno(), "boot identity unavailable")
        return str(UUID(bytes=bytes(host))), boot.value.decode().lower(), "darwin-local"
    if sys.platform == "linux":
        host = str(UUID(Path("/etc/machine-id").read_text(encoding="ascii").strip()))
        boot = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        )
        identities = [Path(f"/proc/self/ns/{name}").stat() for name in ("pid", "net")]
        namespace = "linux:" + ":".join(
            str(value) for info in identities for value in (info.st_dev, info.st_ino)
        )
        return host, boot, namespace
    raise ValueError("Unsupported native executor platform")


def _process_start(pid: int) -> str | None:
    if type(pid) is not int or not 0 < pid <= _MAX_PID:
        raise ValueError("Invalid native PID")
    if sys.platform == "darwin":
        info = _BSDInfo()
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        read = _bind(
            library,
            "proc_pidinfo",
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        ctypes.set_errno(0)
        count = read(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
        if count == 0:
            error = ctypes.get_errno()
            if error == errno.ESRCH:
                return None
            raise OSError(error, "Native process birth unavailable")
        if count != ctypes.sizeof(info) or info.pid != pid:
            raise ValueError("Malformed native process birth")
        return f"darwin:{info.startsec}:{info.startusec}"
    if sys.platform == "linux":
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        head, tail = raw.rsplit(") ", 1)
        if head.split(" (", 1)[0] != str(pid):
            raise ValueError("Native process PID changed")
        return "linux:" + tail.split()[19]
    raise ValueError("Unsupported native executor platform")


def _valid_start(value: str, platform: str) -> bool:
    pieces = value.split(":")
    if not pieces or pieces[0] != platform:
        return False
    expected = 3 if platform == "darwin" else 2
    if len(pieces) != expected or not all(_DECIMAL_RE.fullmatch(x) for x in pieces[1:]):
        return False
    numbers = [int(x) for x in pieces[1:]]
    return (
        all(number <= _MAX_UINT64 for number in numbers)
        and numbers[0] > 0
        and (platform != "darwin" or numbers[1] < _MICROSECONDS)
    )


def _native_binary(backend: str, binary: str) -> Path:
    executable = Path(binary).resolve(strict=True)
    if backend == "agterm":
        if sys.platform != "darwin":
            raise ValueError("Standalone agterm server does not serve native flags")
        package = executable.parent.parent
        with (package / "Info.plist").open("rb") as stream:
            info = plistlib.load(stream)
        if (
            executable.parent.name != "MacOS"
            or info.get("CFBundleIdentifier") != "com.umputun.agterm"
            or info.get("CFBundleShortVersionString") != _NATIVE_VERSIONS[backend]
        ):
            raise ValueError("Unreviewed agterm native package")
        name = info.get("CFBundleExecutable")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid native app executable")
        return (executable.parent / name).resolve(strict=True)
    option = "-V" if backend == "tmux" else "--version"
    version = subprocess.run(
        [str(executable), option],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    if (
        version.returncode
        or version.stdout.strip() != f"{backend} {_NATIVE_VERSIONS[backend]}"
    ):
        raise ValueError("Unreviewed native executor version")
    return executable


class _MacCode:
    def __init__(self) -> None:
        self.cf = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        self.sec = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
        ref, index, flags, status = (
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_uint32,
            ctypes.c_int32,
        )
        self.release = _bind(self.cf, "CFRelease", None, ref)
        self.data = _bind(self.cf, "CFDataCreate", ref, ref, ref, index)
        self.length = _bind(self.cf, "CFDataGetLength", index, ref)
        self.pointer = _bind(self.cf, "CFDataGetBytePtr", ref, ref)
        self.dictionary = _bind(
            self.cf, "CFDictionaryCreate", ref, ref, ref, ref, index, ref, ref
        )
        self.get = _bind(self.cf, "CFDictionaryGetValue", ref, ref, ref)
        self.url = _bind(
            self.cf,
            "CFURLCreateFromFileSystemRepresentation",
            ref,
            ref,
            ref,
            index,
            ctypes.c_uint8,
        )
        self.string = _bind(
            self.cf,
            "CFStringCreateWithCString",
            ref,
            ref,
            ctypes.c_char_p,
            ctypes.c_uint32,
        )
        self.static = _bind(
            self.sec,
            "SecStaticCodeCreateWithPath",
            status,
            ref,
            flags,
            ctypes.POINTER(ref),
        )
        self.static_valid = _bind(
            self.sec, "SecStaticCodeCheckValidity", status, ref, flags, ref
        )
        self.guest = _bind(
            self.sec,
            "SecCodeCopyGuestWithAttributes",
            status,
            ref,
            ref,
            flags,
            ctypes.POINTER(ref),
        )
        self.dynamic_valid = _bind(
            self.sec, "SecCodeCheckValidity", status, ref, flags, ref
        )
        self.signing = _bind(
            self.sec,
            "SecCodeCopySigningInformation",
            status,
            ref,
            flags,
            ctypes.POINTER(ref),
        )
        self.requirement = _bind(
            self.sec,
            "SecRequirementCreateWithString",
            status,
            ref,
            flags,
            ctypes.POINTER(ref),
        )

    def verify(self, token: bytes, binary: Path) -> str:
        ref = ctypes.c_void_p
        owned: list[int] = []

        def keep(value: int | None) -> int:
            if not value:
                raise ValueError("Native code identity allocation failed")
            owned.append(value)
            return value

        def check(status: int) -> None:
            if status != 0:
                raise ValueError(f"Native code identity refused: {status}")

        try:
            token_bytes = ctypes.create_string_buffer(token)
            data = keep(self.data(None, token_bytes, len(token)))
            key = ref.in_dll(self.sec, "kSecGuestAttributeAudit").value
            attributes = keep(
                self.dictionary(None, (ref * 1)(key), (ref * 1)(data), 1, None, None)
            )
            guest = ref()
            check(self.guest(None, attributes, 0, ctypes.byref(guest)))
            keep(guest.value)
            raw_path = os.fsencode(binary)
            path_buffer = ctypes.create_string_buffer(raw_path)
            url = keep(self.url(None, path_buffer, len(raw_path), 0))
            static = ref()
            check(self.static(url, 0, ctypes.byref(static)))
            keep(static.value)
            check(self.static_valid(static, 0, None))
            check(self.dynamic_valid(guest, 0, None))
            info = ref()
            check(self.signing(static, 0, ctypes.byref(info)))
            keep(info.value)
            digest_key = ref.in_dll(self.sec, "kSecCodeInfoUnique").value
            digest = self.get(info, digest_key)
            if not digest or self.length(digest) != _CDHASH_BYTES:
                raise ValueError("Missing native code hash")
            code_hash = ctypes.string_at(
                self.pointer(digest), self.length(digest)
            ).hex()
            text = keep(
                self.string(None, f'cdhash H"{code_hash}"'.encode(), 0x08000100)
            )
            requirement = ref()
            check(self.requirement(text, 0, ctypes.byref(requirement)))
            keep(requirement.value)
            check(self.dynamic_valid(guest, 0, requirement))
            return code_hash
        finally:
            for value in reversed(owned):
                self.release(value)


def _linux_code(pid: int, binary: Path) -> str:
    for name in ("pid", "net"):
        left = Path(f"/proc/{pid}/ns/{name}").stat()
        right = Path(f"/proc/self/ns/{name}").stat()
        if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
            raise ValueError("Foreign native executor namespace")
    with Path(f"/proc/{pid}/exe").open("rb") as loaded, binary.open("rb") as trusted:
        actual = hashlib.file_digest(loaded, "sha256").hexdigest()
        expected = hashlib.file_digest(trusted, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Native executor is a forwarding or unknown process")
    return actual


def _peer_credentials(connection: socket.socket) -> tuple[int, bytes | None]:
    token = None
    if sys.platform == "darwin":
        pid = connection.getsockopt(0, 2)
        token = connection.getsockopt(0, 6, _AUDIT_TOKEN_BYTES)
        if len(token) != _AUDIT_TOKEN_BYTES:
            raise ValueError("Malformed native audit token")
        audit = struct.unpack("8I", token)
        if audit[5] != pid or audit[1] != os.getuid():
            raise ValueError("Foreign native peer credentials")
    elif sys.platform == "linux":
        pid, uid, _gid = struct.unpack(
            "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        if uid != os.getuid():
            raise ValueError("Foreign native peer credentials")
    else:
        raise ValueError("Unsupported native peer credentials")
    if type(pid) is not int or not 0 < pid <= _MAX_PID:
        raise ValueError("Unproved native peer PID")
    return pid, token


def capture_native_process(
    connection: socket.socket, *, backend: str, binary: str
) -> NativeProcessIdentity | None:
    """Injectable OS boundary; call on the actual mutation connection/pinned inode."""
    try:
        if backend not in _NATIVE_VERSIONS:
            raise ValueError("Unsupported native marker backend")
        trusted = _native_binary(backend, binary)
        pid, token = _peer_credentials(connection)
        before = _process_start(pid)
        if before is None or not _valid_start(before, sys.platform):
            return None
        code = (
            _MacCode().verify(token, trusted)
            if token is not None
            else _linux_code(pid, trusted)
        )
        if _process_start(pid) != before:
            return None
        host, boot, namespace = _local_scope()
        if not _canonical_scope(host, boot, namespace):
            raise ValueError("Unproved native host/boot/namespace")
        kind = "cdhash" if sys.platform == "darwin" else "sha256"
        code_id = f"{sys.platform}:{backend}:{_NATIVE_VERSIONS[backend]}:{kind}:{code}"
        return NativeProcessIdentity(host, boot, namespace, pid, before, code_id)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        IndexError,
        subprocess.SubprocessError,
    ) as exc:
        logger.debug("Native marker executor proof refused: %s", exc)
        return None


def inspect_native_process(
    socket_path: str, *, backend: str, binary: str
) -> NativeProcessIdentity | None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(1)
            connection.connect(socket_path)
            return capture_native_process(connection, backend=backend, binary=binary)
    except OSError:
        return None


def inspect_endpoint_scope(
    socket_path: str, *, backend: str, binary: str
) -> str | None:
    """Read a trusted actual endpoint; a scope-only descriptor never authorizes I/O."""
    try:
        endpoint = Path(socket_path).resolve(strict=True)
        before = endpoint.lstat()
        if not stat.S_ISSOCK(before.st_mode) or before.st_uid != os.getuid():
            return None
        identity = inspect_native_process(str(endpoint), backend=backend, binary=binary)
        after = endpoint.lstat()
        if (
            identity is None
            or not stat.S_ISSOCK(after.st_mode)
            or after.st_uid != os.getuid()
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        ):
            return None
        # Dispatch/pin fields are excluded from the shared pure fingerprint.
        return identity.witness(
            backend, after, dispatch_id=uuid4().hex, pin_path=None
        ).endpoint_scope
    except OSError, ValueError, RuntimeError:
        return None


def valid_saved_witness(witness: TopicMarkWriteWitness) -> bool:
    """Parse every opaque OS identity before comparing generations or touching pins."""
    platform = sys.platform
    if (
        platform not in {"darwin", "linux"}
        or (platform == "linux" and witness.backend == "agterm")
        or witness.backend not in _NATIVE_VERSIONS
        or ((witness.backend == "herdr") != (witness.pin_path is None))
        or not _NONCE_RE.fullmatch(witness.dispatch_id)
        or type(witness.native_pid) is not int
        or not 0 < witness.native_pid <= _MAX_PID
        or not 0 <= witness.socket_dev <= _MAX_UINT64
        or not 0 < witness.socket_ino <= _MAX_UINT64
        or not _canonical_uuid(witness.host_id)
        or not _canonical_uuid(witness.boot_id)
        or not _valid_start(witness.native_start, platform)
    ):
        return False
    namespace_ok = _valid_namespace(witness.namespace_id, platform)
    kind, width = ("cdhash", 40) if platform == "darwin" else ("sha256", 64)
    prefix = f"{platform}:{witness.backend}:{_NATIVE_VERSIONS[witness.backend]}:{kind}:"
    digest = witness.code_id.removeprefix(prefix)
    return bool(
        namespace_ok
        and witness.code_id.startswith(prefix)
        and re.fullmatch(rf"[0-9a-f]{{{width}}}", digest)
    )


def process_retired(witness: TopicMarkWriteWitness) -> bool | None:
    if not valid_saved_witness(witness):
        return None
    try:
        host, boot, namespace = _local_scope()
        if not _canonical_scope(host, boot, namespace) or host != witness.host_id:
            return None
        if boot != witness.boot_id:
            return True
        if namespace != witness.namespace_id:
            return None
        current = _process_start(witness.native_pid)
        if current is None:
            return _missing_process(witness.native_pid)
        if not _valid_start(current, sys.platform):
            return None
        return current != witness.native_start
    except OSError, ValueError, RuntimeError, IndexError:
        return None


def _missing_process(pid: int) -> bool | None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        pass
    return None
