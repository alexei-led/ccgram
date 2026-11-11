"""Desk/away presence: decides whether Telegram delivery should be quiet.

Two modes per user. ``away`` (the default, also after a restart) is today's
real-time delivery. ``desk`` means the user is working in the terminal, so
Telegram only gets a periodic digest (see ``digest.py``).

- away -> desk: a prompt typed in the terminal reaches the transcript
  (``note_desk_prompt``), unless its text was sent from Telegram in the last
  ten minutes or the computer has been idle for more than two minutes.
- desk -> away: any Telegram update from the user
  (``note_telegram_activity``), or the computer idle for
  ``config.away_idle_minutes`` (``refresh_idle``).

Keyboard activity alone never means "desk": the user may be typing in
Telegram Desktop on the same computer.

The module keeps state only; callers react to the returned transitions.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import structlog

from ...config import config

logger = structlog.get_logger()

DESK = "desk"
AWAY = "away"

# A Telegram message can reach the transcript late (Claude Code queues input
# while busy), long after the 30 s injection match expired; remember sent
# texts this long so a late one is never mistaken for terminal input.
_TELEGRAM_TEXT_MEMORY_S = 600.0
_TELEGRAM_TEXT_MAX = 50
# A terminal prompt is only believed while the computer was touched recently.
_DESK_PROMPT_MAX_IDLE_S = 120.0
_IDLE_CACHE_S = 15.0
_IDLE_COMMAND = ("ioreg", "-c", "IOHIDSystem", "-r", "-d", "1", "-k", "HIDIdleTime")
_IDLE_TIMEOUT_S = 3.0
# Transcript user messages the agent writes itself, not the user's prompts.
_SYNTHETIC_PREFIXES = ("[Request interrupted",)

IdleReader = Callable[[], Awaitable[float | None]]


@dataclass(slots=True)
class _UserPresence:
    mode: str = AWAY
    changed_at: float = 0.0
    telegram_texts: deque[tuple[str, float]] = field(
        default_factory=lambda: deque(maxlen=_TELEGRAM_TEXT_MAX)
    )


_users: dict[int, _UserPresence] = {}
_idle_cache: tuple[float, float | None] | None = None


def _clock() -> float:
    return time.monotonic()


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _state(user_id: int) -> _UserPresence:
    state = _users.get(user_id)
    if state is None:
        state = _UserPresence()
        _users[user_id] = state
    return state


async def _read_mac_idle_seconds() -> float | None:
    """Seconds since the last keyboard/mouse input, or None if unknown."""
    if sys.platform != "darwin":
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            *_IDLE_COMMAND,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), _IDLE_TIMEOUT_S)
    except (OSError, asyncio.TimeoutError) as exc:
        logger.debug("Idle time read failed", error=str(exc))
        return None
    for line in stdout.decode(errors="replace").splitlines():
        if '"HIDIdleTime"' in line:
            try:
                return int(line.rsplit("=", 1)[1].strip()) / 1e9
            except ValueError, IndexError:
                return None
    return None


_idle_reader: IdleReader = _read_mac_idle_seconds


async def idle_seconds() -> float | None:
    """Computer idle time, cached briefly (one subprocess per ~15 s)."""
    global _idle_cache
    now = _clock()
    if _idle_cache is not None and now - _idle_cache[0] < _IDLE_CACHE_S:
        cached = _idle_cache[1]
        return None if cached is None else cached + (now - _idle_cache[0])
    value = await _idle_reader()
    _idle_cache = (now, value)
    return value


def enabled() -> bool:
    return config.desk_mode


def is_quiet(user_id: int, chat_id: int | None = None) -> bool:
    """True while Telegram delivery for this user should be held back.

    A guest chat (``CCGRAM_GUEST_CHATS``) is never held back: the guest is not
    at the owner's desk and expects real-time replies.
    """
    if not config.desk_mode:
        return False
    if chat_id is not None and chat_id in config.guest_chats:
        return False
    state = _users.get(user_id)
    return state is not None and state.mode == DESK


def mode(user_id: int) -> str:
    state = _users.get(user_id)
    return state.mode if state is not None else AWAY


def _set_mode(user_id: int, new_mode: str, reason: str) -> bool:
    state = _state(user_id)
    if state.mode == new_mode:
        return False
    state.mode = new_mode
    state.changed_at = _clock()
    logger.info("Presence changed", user_id=user_id, mode=new_mode, reason=reason)
    return True


def remember_telegram_text(user_id: int, text: str) -> None:
    """Record text typed into a session from Telegram."""
    normalized = _normalize(text)
    if normalized:
        _state(user_id).telegram_texts.append((normalized, _clock()))


def _sent_from_telegram(state: _UserPresence, text: str) -> bool:
    now = _clock()
    while state.telegram_texts and (
        now - state.telegram_texts[0][1] > _TELEGRAM_TEXT_MEMORY_S
    ):
        state.telegram_texts.popleft()
    return any(sent == text for sent, _ in state.telegram_texts)


def note_telegram_activity(user_id: int) -> bool:
    """Any Telegram update from the user: away. True if the mode changed."""
    if not config.desk_mode:
        return False
    return _set_mode(user_id, AWAY, "telegram")


async def note_desk_prompt(user_id: int, text: str) -> bool:
    """A transcript user message that did not come from Telegram.

    Returns True if the user switched to desk mode.
    """
    if not config.desk_mode:
        return False
    normalized = _normalize(text)
    if not normalized or normalized.startswith(_SYNTHETIC_PREFIXES):
        return False
    state = _state(user_id)
    if state.mode == DESK:
        return False
    if _sent_from_telegram(state, normalized):
        return False
    idle = await idle_seconds()
    if idle is not None and idle > _DESK_PROMPT_MAX_IDLE_S:
        logger.debug("Terminal prompt ignored: computer idle", idle_s=round(idle))
        return False
    return _set_mode(user_id, DESK, "terminal prompt")


async def refresh_idle() -> list[int]:
    """Switch desk users to away once the computer has sat idle long enough.

    Returns the users that switched.
    """
    if not config.desk_mode or config.away_idle_minutes <= 0:
        return []
    desk_users = [uid for uid, state in _users.items() if state.mode == DESK]
    if not desk_users:
        return []
    idle = await idle_seconds()
    if idle is None or idle < config.away_idle_minutes * 60:
        return []
    return [uid for uid in desk_users if _set_mode(uid, AWAY, "idle")]


def _reset_idle_cache_for_testing() -> None:
    global _idle_cache
    _idle_cache = None


def _reset_presence_for_testing(idle_reader: IdleReader | None = None) -> None:
    global _idle_cache, _idle_reader
    _users.clear()
    _idle_cache = None
    _idle_reader = idle_reader or _read_mac_idle_seconds
