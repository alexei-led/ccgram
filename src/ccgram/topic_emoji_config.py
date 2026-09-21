"""Topic emoji customization — load user overrides from a TOML file.

The Telegram forum-topic status emoji (active / idle / done / dead) are
configurable per state, and the two built-in color schemes ("system" and
"user") each map their own set of states to glyphs. Users can override
the defaults by placing a TOML file at ``~/.ccgram/topic_emoji.toml``
(auto-detected) or at ``$CCGRAM_TOPIC_EMOJI_CONFIG`` (explicit path).

This module is **pure data + loader** — it imports nothing from Telegram,
PTB, or tmux, so it is trivially testable in isolation. ``topic_emoji.py``
consumes ``TopicEmojiConfig`` via its ``get_topic_emoji_config()`` singleton
and treats user overrides as a layered patch over the built-in defaults.

TOML schema::

    [topic_emoji.system]
    active = "🟢"
    idle   = "🟡"
    done   = "✅"
    dead   = "💥"

    [topic_emoji.user]
    active = "🟡"
    idle   = "🟢"
    done   = "✅"
    dead   = "💥"

    # Optional: extend the legacy strip list (used by strip_emoji_prefix()
    # to clean up titles left by older versions of ccgram).
    [topic_emoji.legacy_dead]
    emojis = ["⚫", "❌"]

State keys are ``active``, ``idle``, ``done``, ``dead`` — any other key
is logged and ignored. Missing keys fall back to the built-in default for
that state. Malformed entries are logged as warnings and skipped — the
loader never raises.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger()

# State names accepted in the per-mode tables. ``active`` is the working
# state, ``idle`` is paused/waiting, ``done`` is a normal exit, ``dead``
# is a crashed/closed session. These four are the only states ever
# passed to ``update_topic_emoji``; the table validator only accepts
# keys from this set, so unknown state names surface as warnings.
_VALID_STATES: frozenset[str] = frozenset({"active", "idle", "done", "dead"})


# ──────────────────────────────────────────────────────────────────────
# Built-in defaults
#
# The constants are color-named (not state-named) so that
# ``strip_emoji_prefix`` and tests work regardless of mode:
#
#   system mode (default): green=active, yellow=idle ("system POV:
#                         green means working")
#   user mode:             green=idle, yellow=active ("user POV:
#                         green means ready for me")
#
# ``done`` and ``dead`` are shared across both modes — green/yellow only
# describe working vs. waiting.
# ──────────────────────────────────────────────────────────────────────

EMOJI_GREEN_CIRCLE = "\U0001f7e2"
EMOJI_YELLOW_CIRCLE = "\U0001f7e1"
EMOJI_DONE = "\u2705"  # Check mark (agent exited normally)
EMOJI_DEAD = "\U0001f4a5"  # Collision / crash
EMOJI_YOLO = "\U0001f3b2"  # Dice (risk/gamble — auto-approve mode)
EMOJI_RC = "\U0001f4e1"  # Satellite dish (Remote Control active)

# Legacy dead emoji left by older versions of ccgram. ``strip_emoji_prefix``
# consults this list so that names left by a pre-2026-02 (black circle) or
# pre-2026-03 (cross mark) install are cleaned up correctly. Users rarely
# need to extend it, but the TOML hook is here in case the format changes
# again or a user has a stale Telegram name from a custom scheme.
LEGACY_DEAD_EMOJI: tuple[str, ...] = (
    "\u26ab",  # ⚫ — pre-2026-02 dead indicator
    "\u274c",  # ❌ — pre-2026-03 dead indicator
)

DEFAULT_SYSTEM_EMOJI: dict[str, str] = {
    "active": EMOJI_GREEN_CIRCLE,
    "idle": EMOJI_YELLOW_CIRCLE,
    "done": EMOJI_DONE,
    "dead": EMOJI_DEAD,
}

DEFAULT_USER_EMOJI: dict[str, str] = {
    "active": EMOJI_YELLOW_CIRCLE,
    "idle": EMOJI_GREEN_CIRCLE,
    "done": EMOJI_DONE,
    "dead": EMOJI_DEAD,
}


# ──────────────────────────────────────────────────────────────────────
# Resolved config
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TopicEmojiConfig:
    """Resolved topic-emoji configuration.

    Both mapping tables always contain the full set of valid states
    (``active``, ``idle``, ``done``, ``dead``) — missing TOML keys fall
    back to the built-in default for that state. ``legacy_dead`` is the
    extra strip list consulted by ``strip_emoji_prefix``; it always
    starts with the built-in legacy set and is appended to by user
    overrides (so the loader never silently removes the platform defaults).
    """

    system_emoji: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SYSTEM_EMOJI)
    )
    user_emoji: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_USER_EMOJI))
    legacy_dead: tuple[str, ...] = LEGACY_DEAD_EMOJI

    def for_mode(self, mode: str) -> dict[str, str]:
        """Return the state→emoji table for ``mode``.

        Unknown modes fall back to the ``system`` table so behavior is
        identical to the pre-customization code path.
        """
        if mode == "user":
            return self.user_emoji
        return self.system_emoji


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────


def _coerce_emoji(raw: object, *, context: str) -> str | None:
    """Return a non-empty emoji string, or ``None`` if the value is invalid.

    ``context`` is a short label for log messages — e.g. ``"system.active"``.
    Accepts strings only; numbers, lists, and other TOML scalars are
    rejected. Empty / whitespace-only strings are rejected (the loader
    must not silently drop a glyph). Strings containing ASCII whitespace
    in the middle are also rejected — the prefix strip in
    ``strip_emoji_prefix`` uses ``f"{emoji} "`` as the match key, so an
    emoji with an internal space would match a prefix that isn't really
    a prefix and corrupt the cached clean name (e.g. ``active="A B"``
    vs ``idle="A B C"`` strips ``"A B "`` from an idle title and leaves
    ``"C project"``).
    """
    if not isinstance(raw, str):
        logger.warning(
            "Topic emoji config: %s must be a string, got %s, ignoring",
            context,
            type(raw).__name__,
        )
        return None
    stripped = raw.strip()
    if not stripped:
        logger.warning(
            "Topic emoji config: %s is empty, ignoring (use the default)",
            context,
        )
        return None
    if any(char.isspace() for char in stripped):
        logger.warning(
            "Topic emoji config: %s must not contain whitespace, ignoring",
            context,
        )
        return None
    return stripped


def _parse_mode_table(raw: object, *, mode: str) -> dict[str, str]:
    """Parse one ``[topic_emoji.<mode>]`` table. Returns overrides only.

    Unknown state keys are logged and skipped — they never appear in the
    result. Missing entries keep the built-in default. Returns an empty
    dict when the table is absent or malformed.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "Topic emoji config: [topic_emoji.%s] must be a table, ignoring",
            mode,
        )
        return {}
    out: dict[str, str] = {}
    for state, value in raw.items():
        if state not in _VALID_STATES:
            logger.warning(
                "Topic emoji config: [topic_emoji.%s] state=%r invalid "
                "(use one of %s), ignoring",
                mode,
                state,
                sorted(_VALID_STATES),
            )
            continue
        emoji = _coerce_emoji(value, context=f"[topic_emoji.{mode}].{state}")
        if emoji is not None:
            out[state] = emoji
    return out


def _parse_legacy_dead(raw: object) -> tuple[str, ...]:
    """Parse the optional ``[topic_emoji.legacy_dead]`` table.

    Only the ``emojis`` key is recognized. Values must be a list of
    non-empty strings. Bad entries are logged and skipped; the result
    is appended to (not replacing) the built-in legacy set so platform
    defaults are never silently lost.
    """
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        logger.warning(
            "Topic emoji config: [topic_emoji.legacy_dead] must be a table, ignoring"
        )
        return ()
    raw_emojis = raw.get("emojis", [])
    if not isinstance(raw_emojis, list):
        logger.warning(
            "Topic emoji config: [topic_emoji.legacy_dead].emojis must be a list, ignoring"
        )
        return ()
    out: list[str] = []
    for value in raw_emojis:
        if not isinstance(value, str) or not value.strip():
            logger.warning(
                "Topic emoji config: legacy_dead entry %r must be a non-empty string, skipping",
                value,
            )
            continue
        if value in out:
            continue
        out.append(value)
    return tuple(out)


def _read_toml(path: Path) -> dict | None:
    """Read and parse a TOML file. Returns None on any error."""
    if not path.exists():
        # Optional config — its absence is the normal default case, not a warning.
        logger.debug("Topic emoji config file not found: %s — using defaults", path)
        return None
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        logger.warning(
            "Topic emoji config %s unreadable (%s) — using defaults", path, e
        )
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "Topic emoji config %s root must be a table — using defaults", path
        )
        return None
    return raw


def load_topic_emoji_config(path: str | Path | None = None) -> TopicEmojiConfig:
    """Load topic-emoji overrides from a TOML file or fall back to defaults.

    Missing or malformed config falls back to defaults with a warning.
    Missing entries within a present file keep their built-in defaults;
    the loader never raises.
    """
    system_emoji = dict(DEFAULT_SYSTEM_EMOJI)
    user_emoji = dict(DEFAULT_USER_EMOJI)
    legacy_dead: tuple[str, ...] = LEGACY_DEAD_EMOJI

    if not path:
        return TopicEmojiConfig(
            system_emoji=system_emoji,
            user_emoji=user_emoji,
            legacy_dead=legacy_dead,
        )

    # ``Path.expanduser`` raises ``RuntimeError`` (or ``KeyError`` on
    # some platforms / older Python) when the leading ``~user`` cannot
    # be resolved. The loader contract is "never raises", so catch and
    # fall back to defaults with a warning instead of letting the
    # raise escape into the status poll cycle.
    try:
        expanded_path = Path(path).expanduser()
    except (RuntimeError, KeyError) as exc:
        logger.warning(
            "Topic emoji config path %s cannot be expanded (%s) — using defaults",
            path,
            exc,
        )
        return TopicEmojiConfig(
            system_emoji=system_emoji,
            user_emoji=user_emoji,
            legacy_dead=legacy_dead,
        )

    raw = _read_toml(expanded_path)
    if raw is None:
        return TopicEmojiConfig(
            system_emoji=system_emoji,
            user_emoji=user_emoji,
            legacy_dead=legacy_dead,
        )

    section = raw.get("topic_emoji") or {}
    if not isinstance(section, dict):
        logger.warning(
            "Topic emoji config: [topic_emoji] must be a table — using defaults"
        )
        return TopicEmojiConfig(
            system_emoji=system_emoji,
            user_emoji=user_emoji,
            legacy_dead=legacy_dead,
        )

    # Per-mode overrides are merged over the built-in defaults. The
    # merger always starts from a fresh copy of the default so unknown
    # state keys in the TOML do not silently survive.
    system_overrides = _parse_mode_table(section.get("system"), mode="system")
    user_overrides = _parse_mode_table(section.get("user"), mode="user")
    system_emoji.update(system_overrides)
    user_emoji.update(user_overrides)

    extra_legacy = _parse_legacy_dead(section.get("legacy_dead"))
    if extra_legacy:
        legacy_dead = legacy_dead + extra_legacy

    return TopicEmojiConfig(
        system_emoji=system_emoji,
        user_emoji=user_emoji,
        legacy_dead=legacy_dead,
    )
