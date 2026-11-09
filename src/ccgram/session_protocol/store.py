"""Durable metadata without a transcript database.

Persisted action errors can include snapshots containing request tool input.

SQLite FULL commits reserve action IDs before I/O. An unfinished reservation is
permanently unknown: replay must never turn a crash into a second key sequence.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Action, ActionResult, ProtocolError, wire

_PAIR_TTL = 120
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id(prefix: str) -> str:
    value = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    encoded = ""
    for _ in range(26):
        encoded = _ALPHABET[value & 31] + encoded
        value >>= 5
    return prefix + encoded


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class TranscriptCursor:
    """Identity of the transcript bytes last observed for one source.

    `marker` digests the last 4 KiB before `observed`. A row written before this
    shape stored a digest of the whole content, so its first check fails and bumps
    the generation once when the file is at least 4 KiB. A smaller file's whole
    content is its tail, so that legacy digest still matches and the generation
    is kept.
    """

    inode: int
    observed: int
    marker: str
    generation: int


class ProtocolStore:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(path, 0o600)
        self.path = path
        self.clock = clock
        self.db = sqlite3.connect(path, timeout=10)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, target TEXT UNIQUE NOT NULL,
                binding TEXT NOT NULL, version INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS devices (
                id TEXT PRIMARY KEY, name TEXT NOT NULL,
                token_hash TEXT UNIQUE NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS pairing (
                code_hash TEXT PRIMARY KEY, expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS actions (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                result TEXT NOT NULL, status INTEGER NOT NULL,
                created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS deliveries (
                action_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                conversation TEXT NOT NULL, text_hash TEXT NOT NULL,
                sent_at REAL NOT NULL, message_id TEXT UNIQUE);
            CREATE TABLE IF NOT EXISTS transcripts (
                key TEXT PRIMARY KEY, inode INTEGER NOT NULL, size INTEGER NOT NULL,
                prefix_size INTEGER NOT NULL, prefix_hash TEXT NOT NULL,
                generation INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('fingerprint', ?)",
                (secrets.token_hex(32),),
            )

    def close(self) -> None:
        self.db.close()

    @property
    def fingerprint(self) -> str:
        return self.db.execute(
            "SELECT value FROM metadata WHERE key='fingerprint'"
        ).fetchone()[0]

    def bind(
        self, target: str, binding: str, aliases: Sequence[str] = ()
    ) -> tuple[str, int]:
        """Retain identity only for backend-attested, unambiguous aliases."""
        target = target.casefold()
        with self.db:
            row = self.db.execute(
                "SELECT id, binding, version FROM sessions WHERE target=?", (target,)
            ).fetchone()
            if row is None and aliases:
                candidates = [
                    r
                    for alias in set(aliases)
                    if (
                        r := self.db.execute(
                            "SELECT id, binding, version FROM sessions WHERE target=?",
                            (alias.casefold(),),
                        ).fetchone()
                    )
                    is not None
                ]
                if len(candidates) == 1:
                    row = candidates[0]
                    self.db.execute(
                        "UPDATE sessions SET target=? WHERE id=?", (target, row[0])
                    )
            if row is None:
                sid = new_id("s_")
                self.db.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, 1)", (sid, target, binding)
                )
                return sid, 1
            sid, old, version = row
            if old != binding:
                version += 1
                self.db.execute(
                    "UPDATE sessions SET binding=?, version=? WHERE id=?",
                    (binding, version, sid),
                )
            return sid, version

    def make_pairing_code(self) -> str:
        code = secrets.token_urlsafe(32)
        with self.db:
            self.db.execute("DELETE FROM pairing WHERE expires<=?", (self.clock(),))
            self.db.execute(
                "INSERT INTO pairing VALUES (?, ?)",
                (_digest(code), self.clock() + _PAIR_TTL),
            )
        return code

    def pair(self, code: str, name: str) -> dict[str, str]:
        token = secrets.token_urlsafe(32)
        device_id = new_id("d_")
        with self.db:
            removed = self.db.execute(
                "DELETE FROM pairing WHERE code_hash=? AND expires>? RETURNING code_hash",
                (_digest(code), self.clock()),
            ).fetchone()
            if removed is None:
                raise ProtocolError(
                    401, "unauthorized", "Pairing code expired or already used"
                )
            self.db.execute(
                "INSERT INTO devices VALUES (?, ?, ?, ?)",
                (device_id, name, _digest(token), self.clock()),
            )
        return {
            "deviceId": device_id,
            "deviceToken": token,
            "fingerprint": self.fingerprint,
        }

    def authenticate(self, token: str) -> str:
        row = self.db.execute(
            "SELECT id FROM devices WHERE token_hash=?", (_digest(token),)
        ).fetchone()
        if row is None:
            raise ProtocolError(
                401, "unauthorized", "Missing, invalid, or revoked device token"
            )
        return row[0]

    def devices(self) -> list[dict[str, Any]]:
        return [
            {"id": r[0], "name": r[1], "createdAt": r[2]}
            for r in self.db.execute(
                "SELECT id, name, created FROM devices ORDER BY created"
            )
        ]

    def revoke(self, device_id: str) -> None:
        with self.db:
            if not self.db.execute(
                "DELETE FROM devices WHERE id=?", (device_id,)
            ).rowcount:
                raise ProtocolError(404, "not_found", "Device not found")

    @staticmethod
    def action_fingerprint(session_id: str, action: Action) -> str:
        return _digest(
            json.dumps(
                [session_id, wire(action)], sort_keys=True, separators=(",", ":")
            )
        )

    def action(
        self, action_id: str, fingerprint: str
    ) -> tuple[ActionResult, int] | None:
        row = self.db.execute(
            "SELECT fingerprint, result, status FROM actions WHERE id=?", (action_id,)
        ).fetchone()
        if row is None:
            return None
        if row[0] != fingerprint:
            raise ProtocolError(
                400,
                "invalid_request",
                "Action ID was already used for a different action",
            )
        body = json.loads(row[1])
        return (
            ActionResult(
                body["actionId"],
                body["outcome"],
                body["error"],
                body.get("data"),
            ),
            row[2],
        )

    def reserve(self, action_id: str, fingerprint: str) -> None:
        result = ActionResult(action_id, "unknown")
        with self.db:
            self.db.execute(
                "INSERT INTO actions VALUES (?, ?, ?, 200, ?)",
                (action_id, fingerprint, json.dumps(wire(result)), self.clock()),
            )

    def finish(self, result: ActionResult, status: int = 200) -> None:
        with self.db:
            self.db.execute(
                "UPDATE actions SET result=?, status=? WHERE id=?",
                (json.dumps(wire(result)), status, result.action_id),
            )

    def remember_delivery(
        self,
        action_id: str,
        session_id: str,
        conversation: str,
        text: str,
        sent_at: float,
    ) -> None:
        with self.db:
            self.db.execute(
                "DELETE FROM deliveries WHERE message_id IS NULL AND sent_at<?",
                (self.clock() - 60,),
            )
            self.db.execute(
                "INSERT INTO deliveries VALUES (?, ?, ?, ?, ?, NULL)",
                (
                    action_id,
                    session_id,
                    conversation,
                    _digest(" ".join(text.split())),
                    sent_at,
                ),
            )

    def match_delivery(
        self,
        session_id: str,
        conversation: str,
        message_id: str,
        text: str,
        timestamp: float,
    ) -> tuple[str | None, bool]:
        row = self.db.execute(
            "SELECT action_id FROM deliveries WHERE message_id=?", (message_id,)
        ).fetchone()
        if row:
            return row[0], False
        row = self.db.execute(
            "SELECT action_id FROM deliveries WHERE session_id=? AND conversation=? AND text_hash=? AND message_id IS NULL AND sent_at<=? AND sent_at>? ORDER BY sent_at LIMIT 1",
            (
                session_id,
                conversation,
                _digest(" ".join(text.split())),
                timestamp,
                timestamp - 60,
            ),
        ).fetchone()
        if row is None:
            return None, False
        with self.db:
            self.db.execute(
                "UPDATE deliveries SET message_id=? WHERE action_id=?",
                (message_id, row[0]),
            )
        return row[0], True

    def transcript_cursor(self, key: str) -> TranscriptCursor | None:
        row = self.db.execute(
            "SELECT inode, prefix_size, prefix_hash, generation FROM transcripts WHERE key=?",
            (key,),
        ).fetchone()
        return None if row is None else TranscriptCursor(*row)

    def save_transcript_cursor(self, key: str, cursor: TranscriptCursor) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO transcripts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    cursor.inode,
                    cursor.observed,
                    cursor.observed,
                    cursor.marker,
                    cursor.generation,
                ),
            )

    def has_transcript_observation(self, key: str) -> bool:
        """Whether this source has ever had a persisted filesystem observation."""
        return (
            self.db.execute("SELECT 1 FROM transcripts WHERE key=?", (key,)).fetchone()
            is not None
        )

    def retarget(self, previous: str, current: str) -> None:
        """Apply a recovery mapping already attested by the existing resolver."""
        with self.db:
            exists = self.db.execute(
                "SELECT id FROM sessions WHERE target=?", (current.casefold(),)
            ).fetchone()
            if exists is None:
                self.db.execute(
                    "UPDATE sessions SET target=? WHERE target=?",
                    (current.casefold(), previous.casefold()),
                )
