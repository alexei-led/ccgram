"""Guest access list: who may talk to the owner's session in a guest chat.

A guest is a Telegram user outside ``ALLOWED_USERS``. They are trusted only
inside the group chats named by ``CCGRAM_GUEST_CHATS`` and only after the
owner allowed them there with ``/guest allow``. Everywhere else (private
chat, other groups, commands, buttons) they stay unknown to the bot.

State lives in ``~/.ccgram/guests.json`` so the emergency switch
(``/guest off``) survives a restart:

    {"enabled": true,
     "guests": [{"user_id": 1, "chat_id": -100, "name": "QFY", "owner_id": 2}]}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import structlog

from ...config import config
from ...utils import atomic_write_json, ccgram_dir

logger = structlog.get_logger()

_STATE_FILE = "guests.json"


@dataclass(frozen=True, slots=True)
class Guest:
    user_id: int
    chat_id: int
    name: str
    owner_id: int


@dataclass(slots=True)
class _State:
    enabled: bool = True
    guests: tuple[Guest, ...] = ()


_state: _State | None = None


def _path():
    return ccgram_dir() / _STATE_FILE


def _load() -> _State:
    global _state
    if _state is not None:
        return _state
    state = _State()
    try:
        raw = json.loads(_path().read_text())
        state.enabled = bool(raw.get("enabled", True))
        state.guests = tuple(
            Guest(
                user_id=int(item["user_id"]),
                chat_id=int(item["chat_id"]),
                name=str(item.get("name", "")),
                owner_id=int(item["owner_id"]),
            )
            for item in raw.get("guests", [])
        )
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Fail closed: an unreadable file means no guests and guests off.
        logger.warning("Guest list unreadable; guests disabled", error=str(exc))
        state = _State(enabled=False)
    _state = state
    return state


def _save(state: _State) -> None:
    atomic_write_json(
        _path(),
        {"enabled": state.enabled, "guests": [asdict(g) for g in state.guests]},
    )


def is_guest_chat(chat_id: int | None) -> bool:
    return chat_id is not None and chat_id in config.guest_chats


def chat_label(chat_id: int) -> str:
    return config.guest_chats.get(chat_id, str(chat_id))


def is_enabled() -> bool:
    return _load().enabled


def owner_for(user_id: int, chat_id: int | None) -> int | None:
    """The owner a guest acts for in this chat, or None if not a guest here."""
    if not is_guest_chat(chat_id) or config.is_user_allowed(user_id):
        return None
    state = _load()
    if not state.enabled:
        return None
    for guest in state.guests:
        if guest.user_id == user_id and guest.chat_id == chat_id:
            if not config.is_user_allowed(guest.owner_id):
                return None
            return guest.owner_id
    return None


def guests() -> tuple[Guest, ...]:
    return _load().guests


def allow(owner_id: int, user_id: int, chat_id: int, name: str) -> None:
    state = _load()
    kept = tuple(
        g for g in state.guests if not (g.user_id == user_id and g.chat_id == chat_id)
    )
    state.guests = (*kept, Guest(user_id, chat_id, name, owner_id))
    _save(state)
    logger.info("Guest allowed", user_id=user_id, chat_id=chat_id, owner_id=owner_id)


def remove(user_id: int, chat_id: int | None = None) -> int:
    """Remove a guest (from one chat, or every chat). Returns entries removed."""
    state = _load()
    kept = tuple(
        g
        for g in state.guests
        if not (g.user_id == user_id and (chat_id is None or g.chat_id == chat_id))
    )
    removed = len(state.guests) - len(kept)
    if removed:
        state.guests = kept
        _save(state)
        logger.info("Guest removed", user_id=user_id, chat_id=chat_id)
    return removed


def set_enabled(enabled: bool) -> None:
    state = _load()
    state.enabled = enabled
    _save(state)
    logger.warning("Guest access switched", enabled=enabled)


def _reset_guest_access_for_testing() -> None:
    global _state
    _state = None
