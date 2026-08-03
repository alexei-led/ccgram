"""Pure validation for persisted Herdr guarded-session targets."""

from __future__ import annotations

import re

HERDR_SESSION_TARGET_PREFIX = "herdr-session-v1-"
_HERDR_SESSION_TARGET_RE = re.compile(
    rf"{re.escape(HERDR_SESSION_TARGET_PREFIX)}[0-9a-f]{{64}}\Z"
)


def is_herdr_session_target(value: str) -> bool:
    """Return whether *value* is an exact v1 opaque SHA-256 target."""
    return bool(_HERDR_SESSION_TARGET_RE.fullmatch(value))
