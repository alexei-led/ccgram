"""TelegramClient Protocol + adapters.

Protocol exposing exactly the bot API surface used across ``handlers/`` and
top-level modules. Lets handlers depend on a narrow seam instead of importing
``telegram.Bot`` directly, so:

  - tests can pass a ``FakeTelegramClient`` that records calls
  - ``PTBTelegramClient`` wraps a real PTB ``Bot`` for production

Method names match PTB's ``Bot`` method names verbatim so the adapter is a
straight delegation. The Protocol covers only methods grep'd from the
codebase — no aspirational additions. Add new methods here if and when a
handler needs them.

Public API:
  - ``TelegramClient``       — Protocol the handlers depend on
  - ``PTBTelegramClient``    — adapter wrapping ``telegram.Bot``
  - ``FakeTelegramClient``   — recording fake for tests
  - ``unwrap_bot``           — escape hatch for PTB-only helpers (DraftStream)
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

from telegram import Bot, BotCommand, ChatFullInfo, File, ForumTopic, Message, User
from telegram._botcommandscope import BotCommandScope
from telegram._files.inputmedia import InputMedia
from telegram._reaction import ReactionType


@runtime_checkable
class TelegramClient(Protocol):
    """Narrow seam over the PTB ``Bot`` methods used in this codebase.

    Each method mirrors the corresponding ``Bot`` method's name and primary
    arguments. ``**kwargs`` is accepted on every method so callers can pass
    additional PTB-supported parameters (entities, parse_mode, link_preview,
    request timeouts, etc.) without forcing this Protocol to enumerate them
    all.
    """

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        **kwargs: Any,
    ) -> Message: ...

    async def edit_message_text(
        self,
        text: str,
        chat_id: int | str | None = None,
        message_id: int | None = None,
        **kwargs: Any,
    ) -> Message | bool: ...

    async def edit_message_media(
        self,
        media: InputMedia,
        chat_id: int | str | None = None,
        message_id: int | None = None,
        **kwargs: Any,
    ) -> Message | bool: ...

    async def edit_message_caption(
        self,
        chat_id: int | str | None = None,
        message_id: int | None = None,
        **kwargs: Any,
    ) -> Message | bool: ...

    async def delete_message(
        self,
        chat_id: int | str,
        message_id: int,
        **kwargs: Any,
    ) -> bool: ...

    async def send_photo(
        self,
        chat_id: int | str,
        photo: Any,
        **kwargs: Any,
    ) -> Message: ...

    async def send_document(
        self,
        chat_id: int | str,
        document: Any,
        **kwargs: Any,
    ) -> Message: ...

    async def send_chat_action(
        self,
        chat_id: int | str,
        action: str,
        **kwargs: Any,
    ) -> bool: ...

    async def send_voice(
        self,
        chat_id: int | str,
        voice: Any,
        **kwargs: Any,
    ) -> Message: ...

    async def set_message_reaction(
        self,
        chat_id: int | str,
        message_id: int,
        reaction: Sequence[ReactionType | str] | ReactionType | str | None = None,
        **kwargs: Any,
    ) -> bool: ...

    async def get_chat(self, chat_id: int | str, **kwargs: Any) -> ChatFullInfo: ...

    async def get_me(self, **kwargs: Any) -> User: ...

    async def get_file(self, file_id: str, **kwargs: Any) -> File: ...

    async def create_forum_topic(
        self,
        chat_id: int | str,
        name: str,
        **kwargs: Any,
    ) -> ForumTopic: ...

    async def edit_forum_topic(
        self,
        chat_id: int | str,
        message_thread_id: int,
        **kwargs: Any,
    ) -> bool: ...

    async def close_forum_topic(
        self,
        chat_id: int | str,
        message_thread_id: int,
        **kwargs: Any,
    ) -> bool: ...

    async def delete_forum_topic(
        self,
        chat_id: int | str,
        message_thread_id: int,
        **kwargs: Any,
    ) -> bool: ...

    async def unpin_all_forum_topic_messages(
        self,
        chat_id: int | str,
        message_thread_id: int,
        **kwargs: Any,
    ) -> bool: ...

    async def delete_my_commands(
        self,
        scope: BotCommandScope | None = None,
        language_code: str | None = None,
        **kwargs: Any,
    ) -> bool: ...

    async def set_my_commands(
        self,
        commands: Sequence[BotCommand | tuple[str, str]],
        scope: BotCommandScope | None = None,
        language_code: str | None = None,
        **kwargs: Any,
    ) -> bool: ...


class PTBTelegramClient:
    """Adapter that delegates ``TelegramClient`` calls to a PTB ``Bot``.

    Constructed once per process from ``application.bot`` in ``bootstrap``
    and threaded into handlers as a ``TelegramClient``.
    """

    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    @property
    def bot(self) -> Bot:
        """Underlying PTB Bot — escape hatch for PTB-only helpers.

        Used by ``unwrap_bot`` for ``DraftStream`` / ``do_api_request`` and a
        handful of helpers that need PTB internals. Handlers should depend on
        the ``TelegramClient`` Protocol instead.
        """
        return self._bot

    async def send_message(
        self, chat_id: int | str, text: str, **kwargs: Any
    ) -> Message:
        return await self._bot.send_message(
            chat_id=chat_id,
            text=text,
            **kwargs,
        )

    async def edit_message_text(
        self,
        text: str,
        chat_id: int | str | None = None,
        message_id: int | None = None,
        **kwargs: Any,
    ) -> Message | bool:
        return await self._bot.edit_message_text(
            text=text, chat_id=chat_id, message_id=message_id, **kwargs
        )

    async def edit_message_media(
        self,
        media: InputMedia,
        chat_id: int | str | None = None,
        message_id: int | None = None,
        **kwargs: Any,
    ) -> Message | bool:
        return await self._bot.edit_message_media(
            media=media, chat_id=chat_id, message_id=message_id, **kwargs
        )

    async def edit_message_caption(
        self,
        chat_id: int | str | None = None,
        message_id: int | None = None,
        **kwargs: Any,
    ) -> Message | bool:
        return await self._bot.edit_message_caption(
            chat_id=chat_id, message_id=message_id, **kwargs
        )

    async def delete_message(
        self, chat_id: int | str, message_id: int, **kwargs: Any
    ) -> bool:
        return await self._bot.delete_message(
            chat_id=chat_id, message_id=message_id, **kwargs
        )

    async def send_photo(
        self, chat_id: int | str, photo: Any, **kwargs: Any
    ) -> Message:
        return await self._bot.send_photo(
            chat_id=chat_id,
            photo=photo,
            **kwargs,
        )

    async def send_document(
        self, chat_id: int | str, document: Any, **kwargs: Any
    ) -> Message:
        return await self._bot.send_document(
            chat_id=chat_id,
            document=document,
            **kwargs,
        )

    async def send_chat_action(
        self, chat_id: int | str, action: str, **kwargs: Any
    ) -> bool:
        return await self._bot.send_chat_action(
            chat_id=chat_id, action=action, **kwargs
        )

    async def send_voice(
        self, chat_id: int | str, voice: Any, **kwargs: Any
    ) -> Message:
        return await self._bot.send_voice(
            chat_id=chat_id,
            voice=voice,
            **kwargs,
        )

    async def set_message_reaction(
        self,
        chat_id: int | str,
        message_id: int,
        reaction: Sequence[ReactionType | str] | ReactionType | str | None = None,
        **kwargs: Any,
    ) -> bool:
        return await self._bot.set_message_reaction(
            chat_id=chat_id, message_id=message_id, reaction=reaction, **kwargs
        )

    async def get_chat(self, chat_id: int | str, **kwargs: Any) -> ChatFullInfo:
        return await self._bot.get_chat(chat_id=chat_id, **kwargs)

    async def get_me(self, **kwargs: Any) -> User:
        return await self._bot.get_me(**kwargs)

    async def get_file(self, file_id: str, **kwargs: Any) -> File:
        return await self._bot.get_file(file_id=file_id, **kwargs)

    async def create_forum_topic(
        self, chat_id: int | str, name: str, **kwargs: Any
    ) -> ForumTopic:
        return await self._bot.create_forum_topic(chat_id=chat_id, name=name, **kwargs)

    async def edit_forum_topic(
        self, chat_id: int | str, message_thread_id: int, **kwargs: Any
    ) -> bool:
        return await self._bot.edit_forum_topic(
            chat_id=chat_id, message_thread_id=message_thread_id, **kwargs
        )

    async def close_forum_topic(
        self, chat_id: int | str, message_thread_id: int, **kwargs: Any
    ) -> bool:
        return await self._bot.close_forum_topic(
            chat_id=chat_id, message_thread_id=message_thread_id, **kwargs
        )

    async def delete_forum_topic(
        self, chat_id: int | str, message_thread_id: int, **kwargs: Any
    ) -> bool:
        return await self._bot.delete_forum_topic(
            chat_id=chat_id, message_thread_id=message_thread_id, **kwargs
        )

    async def unpin_all_forum_topic_messages(
        self, chat_id: int | str, message_thread_id: int, **kwargs: Any
    ) -> bool:
        return await self._bot.unpin_all_forum_topic_messages(
            chat_id=chat_id, message_thread_id=message_thread_id, **kwargs
        )

    async def delete_my_commands(
        self,
        scope: BotCommandScope | None = None,
        language_code: str | None = None,
        **kwargs: Any,
    ) -> bool:
        return await self._bot.delete_my_commands(
            scope=scope, language_code=language_code, **kwargs
        )

    async def set_my_commands(
        self,
        commands: Sequence[BotCommand | tuple[str, str]],
        scope: BotCommandScope | None = None,
        language_code: str | None = None,
        **kwargs: Any,
    ) -> bool:
        return await self._bot.set_my_commands(
            commands=commands, scope=scope, language_code=language_code, **kwargs
        )


@dataclass
class _FakeCall:
    """Single recorded call on ``FakeTelegramClient``."""

    method: str
    kwargs: dict[str, Any]


@dataclass
class FakeTelegramClient:
    """Recording fake for tests.

    Every ``await client.send_message(...)`` records an entry on ``calls``.
    Tests can pre-seed ``returns[method_name]`` with a callable producing the
    return value (deterministic) or rely on the default — a no-op return that
    matches the Protocol's nominal type.

    Deliberately *not* a subclass of ``PTBTelegramClient`` — duck-typing to
    ``TelegramClient`` keeps the seam honest.
    """

    calls: list[_FakeCall] = field(default_factory=list)
    returns: dict[str, Any] = field(default_factory=dict)

    def _record(self, method: str, kwargs: dict[str, Any]) -> Any:
        self.calls.append(_FakeCall(method=method, kwargs=dict(kwargs)))
        if method in self.returns:
            spec = self.returns[method]
            if inspect.isfunction(spec) or inspect.ismethod(spec):
                return spec(**kwargs)
            return spec
        return _DEFAULT_RETURNS.get(method, True)

    def call_count(self, method: str) -> int:
        """Count how many times ``method`` was called."""
        return sum(1 for c in self.calls if c.method == method)

    def last_call(self, method: str) -> _FakeCall | None:
        """Return the most recent recorded call for ``method`` (or None)."""
        for call in reversed(self.calls):
            if call.method == method:
                return call
        return None

    def set_side_effect(self, method: str, effects: Sequence[Any]) -> None:
        """Configure ``method`` to return / raise the items of ``effects`` in order.

        Any element that is a ``BaseException`` instance is raised; everything
        else is returned as the call result. Mirrors ``unittest.mock.Mock.side_effect``
        for the iterable case.
        """
        iterator = iter(effects)

        def step(**_kwargs: Any) -> Any:
            value = next(iterator)
            if isinstance(value, BaseException):
                raise value
            return value

        self.returns[method] = step

    async def send_message(
        self, chat_id: int | str, text: str, **kwargs: Any
    ) -> Message:
        return self._record(
            "send_message",
            {
                "chat_id": chat_id,
                "text": text,
                **kwargs,
            },
        )

    async def edit_message_text(
        self,
        text: str,
        chat_id: int | str | None = None,
        message_id: int | None = None,
        **kwargs: Any,
    ) -> Message | bool:
        return self._record(
            "edit_message_text",
            {"text": text, "chat_id": chat_id, "message_id": message_id, **kwargs},
        )

    async def edit_message_media(
        self,
        media: InputMedia,
        chat_id: int | str | None = None,
        message_id: int | None = None,
        **kwargs: Any,
    ) -> Message | bool:
        return self._record(
            "edit_message_media",
            {"media": media, "chat_id": chat_id, "message_id": message_id, **kwargs},
        )

    async def edit_message_caption(
        self,
        chat_id: int | str | None = None,
        message_id: int | None = None,
        **kwargs: Any,
    ) -> Message | bool:
        return self._record(
            "edit_message_caption",
            {"chat_id": chat_id, "message_id": message_id, **kwargs},
        )

    async def delete_message(
        self, chat_id: int | str, message_id: int, **kwargs: Any
    ) -> bool:
        return self._record(
            "delete_message",
            {"chat_id": chat_id, "message_id": message_id, **kwargs},
        )

    async def send_photo(
        self, chat_id: int | str, photo: Any, **kwargs: Any
    ) -> Message:
        return self._record(
            "send_photo",
            {
                "chat_id": chat_id,
                "photo": photo,
                **kwargs,
            },
        )

    async def send_document(
        self, chat_id: int | str, document: Any, **kwargs: Any
    ) -> Message:
        return self._record(
            "send_document",
            {
                "chat_id": chat_id,
                "document": document,
                **kwargs,
            },
        )

    async def send_chat_action(
        self, chat_id: int | str, action: str, **kwargs: Any
    ) -> bool:
        return self._record(
            "send_chat_action", {"chat_id": chat_id, "action": action, **kwargs}
        )

    async def send_voice(
        self, chat_id: int | str, voice: Any, **kwargs: Any
    ) -> Message:
        return self._record(
            "send_voice",
            {
                "chat_id": chat_id,
                "voice": voice,
                **kwargs,
            },
        )

    async def set_message_reaction(
        self,
        chat_id: int | str,
        message_id: int,
        reaction: Sequence[ReactionType | str] | ReactionType | str | None = None,
        **kwargs: Any,
    ) -> bool:
        return self._record(
            "set_message_reaction",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reaction": reaction,
                **kwargs,
            },
        )

    async def get_chat(self, chat_id: int | str, **kwargs: Any) -> ChatFullInfo:
        return self._record("get_chat", {"chat_id": chat_id, **kwargs})

    async def get_me(self, **kwargs: Any) -> User:
        return self._record("get_me", kwargs)

    async def get_file(self, file_id: str, **kwargs: Any) -> File:
        return self._record("get_file", {"file_id": file_id, **kwargs})

    async def create_forum_topic(
        self, chat_id: int | str, name: str, **kwargs: Any
    ) -> ForumTopic:
        return self._record(
            "create_forum_topic", {"chat_id": chat_id, "name": name, **kwargs}
        )

    async def edit_forum_topic(
        self, chat_id: int | str, message_thread_id: int, **kwargs: Any
    ) -> bool:
        return self._record(
            "edit_forum_topic",
            {"chat_id": chat_id, "message_thread_id": message_thread_id, **kwargs},
        )

    async def close_forum_topic(
        self, chat_id: int | str, message_thread_id: int, **kwargs: Any
    ) -> bool:
        return self._record(
            "close_forum_topic",
            {"chat_id": chat_id, "message_thread_id": message_thread_id, **kwargs},
        )

    async def delete_forum_topic(
        self, chat_id: int | str, message_thread_id: int, **kwargs: Any
    ) -> bool:
        return self._record(
            "delete_forum_topic",
            {"chat_id": chat_id, "message_thread_id": message_thread_id, **kwargs},
        )

    async def unpin_all_forum_topic_messages(
        self, chat_id: int | str, message_thread_id: int, **kwargs: Any
    ) -> bool:
        return self._record(
            "unpin_all_forum_topic_messages",
            {"chat_id": chat_id, "message_thread_id": message_thread_id, **kwargs},
        )

    async def delete_my_commands(
        self,
        scope: BotCommandScope | None = None,
        language_code: str | None = None,
        **kwargs: Any,
    ) -> bool:
        return self._record(
            "delete_my_commands",
            {"scope": scope, "language_code": language_code, **kwargs},
        )

    async def set_my_commands(
        self,
        commands: Sequence[BotCommand | tuple[str, str]],
        scope: BotCommandScope | None = None,
        language_code: str | None = None,
        **kwargs: Any,
    ) -> bool:
        return self._record(
            "set_my_commands",
            {
                "commands": commands,
                "scope": scope,
                "language_code": language_code,
                **kwargs,
            },
        )


_DEFAULT_RETURNS: dict[str, Any] = {
    # bool-returning methods default to True; tests that need a Message can
    # pre-seed `returns["send_message"] = my_fake_message`.
    "delete_message": True,
    "send_chat_action": True,
    "set_message_reaction": True,
    "edit_forum_topic": True,
    "close_forum_topic": True,
    "delete_forum_topic": True,
    "unpin_all_forum_topic_messages": True,
    "delete_my_commands": True,
    "set_my_commands": True,
}


def unwrap_bot(client: TelegramClient) -> Bot:
    """Return the underlying PTB ``Bot`` from a ``TelegramClient``.

    Escape hatch for PTB-only helpers (e.g. ``DraftStream`` uses
    ``do_api_request``, which is not on the Protocol). Production passes
    ``PTBTelegramClient(bot)``; tests pass an ``AsyncMock`` shaped like
    ``Bot`` and get it back unchanged.
    """
    if isinstance(client, PTBTelegramClient):
        return client.bot
    return cast("Bot", client)


__all__ = [
    "FakeTelegramClient",
    "PTBTelegramClient",
    "TelegramClient",
    "unwrap_bot",
]
