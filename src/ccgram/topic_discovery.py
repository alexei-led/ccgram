"""Discovery mode validation and automatic topic admission policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .multiplexer.base import WindowRef

TOPIC_DISCOVERY_MODES = ("auto", "marked")
DEFAULT_TOPIC_DISCOVERY_MODE = "auto"
TOPIC_DISCOVERY_ENV = "CCGRAM_TOPIC_DISCOVERY"


def parse_topic_discovery_mode(value: str) -> str:
    """Validate discovery configuration without loading bot credentials."""
    mode = value.strip().lower()
    if mode not in TOPIC_DISCOVERY_MODES:
        raise ValueError(
            f"{TOPIC_DISCOVERY_ENV} must be {' or '.join(TOPIC_DISCOVERY_MODES)}"
        )
    return mode


def should_adopt_window(
    window: WindowRef, *, mode: str, detach_pending: bool, quarantined: bool
) -> bool:
    """Automatic admission only; never filter liveness or existing bindings.

    An unfinished native write blocks admission even in auto mode. A pending
    clear remains a marked-mode detach, not another selection preference.
    """
    mode = parse_topic_discovery_mode(mode)
    if quarantined:
        return False
    if mode == "auto":
        return window.topic_eligible
    return (
        window.topic_eligible
        and window.topic_markable
        and window.topic_marked is True
        and bool(window.topic_mark_owner)
        and not detach_pending
    )
