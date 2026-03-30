"""File-based mailbox for inter-agent messaging.

Provides per-window inboxes with atomic writes, TTL-based expiration,
and FIFO ordering via timestamp-prefixed filenames. Mailbox IDs use
qualified window IDs (e.g. ``ccgram:@0``) matching session_map convention.

Key class: Mailbox.
"""

import contextlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

_BODY_SIZE_LIMIT = 10 * 1024  # 10 KB

_DEFAULT_TTL: dict[str, int] = {
    "request": 60,
    "reply": 120,
    "notify": 240,
    "broadcast": 480,
}

_VALID_TYPES = frozenset(_DEFAULT_TTL)

_SWEEPABLE_STATUSES = frozenset({"replied", "expired"})


@dataclass
class Message:
    """A single inter-agent message."""

    id: str
    from_id: str
    to_id: str
    type: str
    body: str
    subject: str = ""
    reply_to: str | None = None
    file_path: str | None = None
    context: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    delivered_at: str | None = None
    read_at: str | None = None
    status: str = "pending"
    ttl_minutes: int = 60

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != "" and v != {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            id=data.get("id", ""),
            from_id=data.get("from_id", data.get("from", "")),
            to_id=data.get("to_id", data.get("to", "")),
            type=data.get("type", "request"),
            body=data.get("body", ""),
            subject=data.get("subject", ""),
            reply_to=data.get("reply_to"),
            file_path=data.get("file_path"),
            context=data.get("context", {}),
            created_at=data.get("created_at", ""),
            delivered_at=data.get("delivered_at"),
            read_at=data.get("read_at"),
            status=data.get("status", "pending"),
            ttl_minutes=data.get("ttl_minutes", 60),
        )

    def is_expired(self) -> bool:
        if not self.created_at:
            return False
        created = datetime.fromisoformat(self.created_at)
        elapsed = datetime.now(timezone.utc) - created
        return elapsed.total_seconds() > self.ttl_minutes * 60


def _validate_no_traversal(value: str, label: str) -> None:
    """Reject values containing path traversal sequences."""
    if ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {label}: must not contain path separators or '..'")


def _sanitize_dir_name(qualified_id: str) -> str:
    """Convert a qualified window ID to a safe directory name.

    Replaces colons with ``=`` so that IDs like ``ccgram:@0`` become
    ``ccgram=@0`` (filesystem-safe on all platforms).
    """
    _validate_no_traversal(qualified_id, "window ID")
    return qualified_id.replace(":", "=")


def _unsanitize_dir_name(dir_name: str) -> str:
    """Reverse ``_sanitize_dir_name``: ``ccgram=@0`` → ``ccgram:@0``."""
    return dir_name.replace("=", ":")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_id() -> str:
    ts_ns = time.time_ns()
    short_uuid = uuid.uuid4().hex[:8]
    return f"{ts_ns}-{short_uuid}"


def _atomic_write_message(path: Path, data: dict[str, Any]) -> None:
    """Write message JSON atomically using tmp dir inside the inbox."""
    tmp_dir = path.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=str(tmp_dir), suffix=".json", prefix=".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


class Mailbox:
    """File-based message mailbox with per-window inboxes.

    Directory layout::

        base_dir/
          ccgram=@0/           # sanitized qualified ID
            tmp/               # atomic write staging
            1743250000-abc123.json
          ccgram=@5/
            tmp/
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _inbox_dir(self, window_id: str) -> Path:
        return self.base_dir / _sanitize_dir_name(window_id)

    def send(
        self,
        from_id: str,
        to_id: str,
        body: str,
        *,
        msg_type: str = "request",
        subject: str = "",
        ttl_minutes: int | None = None,
        reply_to: str | None = None,
        file_path: str | None = None,
        context: dict[str, str] | None = None,
    ) -> Message:
        """Write a message to the recipient's inbox. Returns the created message."""
        if msg_type not in _VALID_TYPES:
            raise ValueError(
                f"Invalid message type: {msg_type!r} (must be one of {sorted(_VALID_TYPES)})"
            )

        if file_path:
            fp = Path(file_path)
            if not fp.is_file():
                raise FileNotFoundError(f"File not found: {file_path}")
            body = f"[file:{file_path}]"
        elif len(body.encode("utf-8")) > _BODY_SIZE_LIMIT:
            raise ValueError(
                f"Body exceeds {_BODY_SIZE_LIMIT} bytes. Use --file for larger payloads."
            )

        if ttl_minutes is None:
            ttl_minutes = _DEFAULT_TTL[msg_type]

        msg_id = _make_id()
        msg = Message(
            id=msg_id,
            from_id=from_id,
            to_id=to_id,
            type=msg_type,
            body=body,
            subject=subject,
            reply_to=reply_to,
            file_path=file_path,
            context=context or {},
            created_at=_now_iso(),
            status="pending",
            ttl_minutes=ttl_minutes,
        )

        inbox = self._inbox_dir(to_id)
        inbox.mkdir(parents=True, exist_ok=True)
        msg_path = inbox / f"{msg_id}.json"
        _atomic_write_message(msg_path, msg.to_dict())

        logger.debug(
            "Message sent", msg_id=msg_id, from_id=from_id, to_id=to_id, type=msg_type
        )
        return msg

    def inbox(self, window_id: str) -> list[Message]:
        """List pending/delivered messages for a window, FIFO order, filtering expired."""
        inbox_dir = self._inbox_dir(window_id)
        if not inbox_dir.is_dir():
            return []

        messages: list[Message] = []
        for entry in sorted(os.scandir(str(inbox_dir)), key=lambda e: e.name):
            if not entry.name.endswith(".json") or entry.name.startswith("."):
                continue
            if entry.is_dir():
                continue
            try:
                with open(entry.path, encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError, OSError:
                continue
            msg = Message.from_dict(data)
            if msg.is_expired():
                continue
            if msg.status in ("pending", "delivered"):
                messages.append(msg)
        return messages

    def all_messages(self, window_id: str) -> list[Message]:
        """List all non-expired messages for a window, including read/replied."""
        inbox_dir = self._inbox_dir(window_id)
        if not inbox_dir.is_dir():
            return []

        messages: list[Message] = []
        for entry in sorted(os.scandir(str(inbox_dir)), key=lambda e: e.name):
            if not entry.name.endswith(".json") or entry.name.startswith("."):
                continue
            if entry.is_dir():
                continue
            try:
                with open(entry.path, encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError, OSError:
                continue
            msg = Message.from_dict(data)
            if msg.is_expired():
                continue
            messages.append(msg)
        return messages

    def read(self, msg_id: str, window_id: str) -> Message | None:
        """Mark a message as read. Returns the updated message or None if not found."""
        msg, path = self._find_message(msg_id, window_id)
        if msg is None or path is None:
            return None
        msg.status = "read"
        msg.read_at = _now_iso()
        _atomic_write_message(path, msg.to_dict())
        return msg

    def reply(self, msg_id: str, window_id: str, body: str) -> Message | None:
        """Create a reply to an existing message.

        Marks the original as ``replied`` and returns the new reply message.
        Returns None if the original is not found.
        """
        original, original_path = self._find_message(msg_id, window_id)
        if original is None or original_path is None:
            return None

        original.status = "replied"
        _atomic_write_message(original_path, original.to_dict())

        return self.send(
            from_id=window_id,
            to_id=original.from_id,
            body=body,
            msg_type="reply",
            reply_to=msg_id,
        )

    def broadcast(
        self,
        from_id: str,
        recipient_ids: list[str],
        body: str,
        *,
        subject: str = "",
        ttl_minutes: int | None = None,
        file_path: str | None = None,
        context: dict[str, str] | None = None,
    ) -> list[Message]:
        """Send a broadcast message to multiple recipients.

        Writes one message file per recipient inbox. Type is always
        ``broadcast`` with 480 min TTL by default. Returns list of
        successfully sent messages.
        """
        sent: list[Message] = []
        for to_id in recipient_ids:
            try:
                msg = self.send(
                    from_id=from_id,
                    to_id=to_id,
                    body=body,
                    msg_type="broadcast",
                    subject=subject,
                    ttl_minutes=ttl_minutes,
                    file_path=file_path,
                    context=context,
                )
                sent.append(msg)
            except ValueError:
                logger.warning("Broadcast send failed", from_id=from_id, to_id=to_id)
        return sent

    def mark_delivered(self, msg_id: str, window_id: str) -> Message | None:
        """Set delivered_at timestamp on a message."""
        msg, path = self._find_message(msg_id, window_id)
        if msg is None or path is None:
            return None
        msg.status = "delivered"
        msg.delivered_at = _now_iso()
        _atomic_write_message(path, msg.to_dict())
        return msg

    def sweep(self, window_id: str | None = None) -> int:
        """Remove expired and old read/replied messages.

        Returns the number of files removed. Uses contextlib.suppress on
        every file operation for race-safety with concurrent sweeps.
        """
        dirs = (
            [self._inbox_dir(window_id)]
            if window_id
            else [
                self.base_dir / d.name
                for d in os.scandir(str(self.base_dir))
                if d.is_dir() and d.name not in ("tmp", "spawns")
            ]
        )
        removed = sum(self._sweep_dir(d) for d in dirs if d.is_dir())
        logger.debug("Sweep completed", removed=removed, window_id=window_id)
        return removed

    def _sweep_dir(self, inbox_dir: Path) -> int:
        """Sweep a single inbox directory. Returns number of files removed."""
        removed = 0
        try:
            entries = list(os.scandir(str(inbox_dir)))
        except FileNotFoundError:
            return 0
        for entry in entries:
            if not entry.name.endswith(".json") or entry.name.startswith("."):
                continue
            if entry.is_dir():
                continue
            removed += self._sweep_entry(entry)
        return removed

    def _sweep_entry(self, entry: os.DirEntry[str]) -> int:
        """Check one message file; remove if expired/replied/corrupt. Returns 1 if removed."""
        try:
            with open(entry.path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return 0
        except json.JSONDecodeError, OSError:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(entry.path)
            return 1

        msg = Message.from_dict(data)
        if msg.is_expired() or msg.status in _SWEEPABLE_STATUSES:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(entry.path)
            return 1
        return 0

    def clear_inbox(self, window_id: str) -> int:
        """Remove all messages from a window's inbox (regardless of status).

        Used on topic close to prevent orphaned messages being delivered
        to a future window that reuses the same ID.
        """
        inbox_dir = self._inbox_dir(window_id)
        if not inbox_dir.is_dir():
            return 0
        removed = 0
        for entry in os.scandir(str(inbox_dir)):
            if not entry.name.endswith(".json") or entry.is_dir():
                continue
            with contextlib.suppress(OSError):
                os.unlink(entry.path)
                removed += 1
        return removed

    def migrate_ids(self, old_to_new: dict[str, str]) -> None:
        """Rename mailbox directories when window IDs are remapped."""
        for old_id, new_id in old_to_new.items():
            old_dir = self._inbox_dir(old_id)
            new_dir = self._inbox_dir(new_id)
            if old_dir.is_dir() and not new_dir.exists():
                old_dir.rename(new_dir)
                logger.info("Migrated mailbox", old_id=old_id, new_id=new_id)

                for entry in os.scandir(str(new_dir)):
                    if not entry.name.endswith(".json") or entry.is_dir():
                        continue
                    try:
                        with open(entry.path, encoding="utf-8") as f:
                            data = json.load(f)
                        changed = False
                        if data.get("to_id") == old_id or data.get("to") == old_id:
                            data["to_id"] = new_id
                            changed = True
                        if data.get("from_id") == old_id or data.get("from") == old_id:
                            data["from_id"] = new_id
                            changed = True
                        if changed:
                            _atomic_write_message(Path(entry.path), data)
                    except json.JSONDecodeError, OSError, FileNotFoundError:
                        continue

    def prune_dead(self, live_ids: set[str]) -> int:
        """Remove mailbox directories for windows not in live_ids.

        Preserves foreign (emdash) windows — only prunes dirs whose
        unsanitized ID starts with the local tmux session prefix.
        """
        removed = 0
        if not self.base_dir.is_dir():
            return removed

        for entry in os.scandir(str(self.base_dir)):
            if not entry.is_dir() or entry.name in ("tmp", "spawns"):
                continue
            qualified_id = _unsanitize_dir_name(entry.name)
            if qualified_id in live_ids:
                continue
            # Only prune local windows (foreign windows managed externally)
            if "emdash-" in qualified_id:
                continue
            try:
                self._remove_inbox_dir(Path(entry.path))
                removed += 1
                logger.info("Pruned dead mailbox", window_id=qualified_id)
            except OSError:
                logger.warning("Failed to prune mailbox", window_id=qualified_id)
        return removed

    def pending_undelivered(self, min_age_seconds: float = 5.0) -> list[Message]:
        """Find messages without delivered_at older than min_age_seconds.

        Used for crash recovery re-injection.
        """
        cutoff = time.time() - min_age_seconds
        results: list[Message] = []

        if not self.base_dir.is_dir():
            return results

        for inbox_entry in os.scandir(str(self.base_dir)):
            if not inbox_entry.is_dir() or inbox_entry.name in ("tmp", "spawns"):
                continue
            self._collect_undelivered(inbox_entry.path, cutoff, results)
        return results

    def _collect_undelivered(
        self, inbox_path: str, cutoff: float, out: list[Message]
    ) -> None:
        """Scan one inbox dir for pending undelivered messages older than cutoff."""
        try:
            entries = list(os.scandir(inbox_path))
        except FileNotFoundError:
            return
        for entry in entries:
            if not entry.name.endswith(".json") or entry.name.startswith("."):
                continue
            if entry.is_dir():
                continue
            msg = self._read_if_undelivered(entry.path, cutoff)
            if msg is not None:
                out.append(msg)

    @staticmethod
    def _read_if_undelivered(path: str, cutoff: float) -> Message | None:
        """Read a message file and return it only if pending+undelivered+old enough."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError, OSError, FileNotFoundError:
            return None
        msg = Message.from_dict(data)
        if msg.status != "pending" or msg.delivered_at is not None:
            return None
        if msg.is_expired() or not msg.created_at:
            return None
        created_ts = datetime.fromisoformat(msg.created_at).timestamp()
        if created_ts < cutoff:
            return msg
        return None

    def _find_message(
        self, msg_id: str, window_id: str
    ) -> tuple[Message | None, Path | None]:
        _validate_no_traversal(msg_id, "message ID")
        inbox_dir = self._inbox_dir(window_id)
        msg_path = inbox_dir / f"{msg_id}.json"
        if not msg_path.is_file():
            return None, None
        try:
            with open(msg_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError, OSError:
            return None, None
        return Message.from_dict(data), msg_path

    def _remove_inbox_dir(self, inbox_dir: Path) -> None:
        """Remove an inbox directory and all its contents."""
        tmp_dir = inbox_dir / "tmp"
        if tmp_dir.is_dir():
            for entry in os.scandir(str(tmp_dir)):
                with contextlib.suppress(OSError):
                    os.unlink(entry.path)
            with contextlib.suppress(OSError):
                tmp_dir.rmdir()
        for entry in os.scandir(str(inbox_dir)):
            if entry.is_dir():
                continue
            with contextlib.suppress(OSError):
                os.unlink(entry.path)
        inbox_dir.rmdir()
