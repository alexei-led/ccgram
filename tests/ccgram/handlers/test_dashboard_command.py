"""Tests for the topic-scoped /dashboard entry point."""

import hashlib
import hmac
import time
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlencode, urlparse

import pytest
from telegram.error import Forbidden

import ccgram.handlers.dashboard_command as dashboard_module
from ccgram import session_query
from ccgram.miniapp.auth import (
    InvalidTokenError,
    authorize_api_request,
    verify_token,
)
from ccgram.thread_router import ThreadRouter


USER_ID = 42
THREAD_ID = 77
BOT_TOKEN = "1234:abcdef"


def _make_update(
    *, user_id: int = USER_ID, chat_id: int = -1001, thread_id: int | None = THREAD_ID
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    update.effective_chat = MagicMock(id=chat_id, type="supergroup")
    update.message = AsyncMock()
    update.message.chat.id = chat_id
    update.message.message_thread_id = thread_id
    return update


def _make_context() -> MagicMock:
    context = MagicMock()
    context.bot.send_message = AsyncMock()
    return context


def _make_init_data(bot_token: str, user_id: int) -> str:
    params = {
        "auth_date": str(int(time.time())),
        "user": f'{{"id":{user_id},"first_name":"Test"}}',
    }
    data_check = "\n".join(f"{key}={value}" for key, value in sorted(params.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    signature = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
    return urlencode({**params, "hash": signature})


@pytest.fixture
def dashboard_setup(monkeypatch: pytest.MonkeyPatch) -> ThreadRouter:
    router = ThreadRouter(
        schedule_save=lambda: None, has_window_state=lambda _wid: False
    )
    monkeypatch.setattr(
        session_query, "resolve_window_for_topic", router.resolve_window_for_thread
    )
    monkeypatch.setattr(
        dashboard_module.config, "miniapp_base_url", "https://example.test"
    )
    monkeypatch.setattr(dashboard_module.config, "telegram_bot_token", BOT_TOKEN)
    monkeypatch.setattr(
        type(dashboard_module.config),
        "is_user_allowed",
        lambda _self, user_id: user_id == USER_ID,
    )
    return router


async def test_same_topic_id_in_different_chats_opens_each_chat_binding(
    dashboard_setup: ThreadRouter,
) -> None:
    dashboard_setup.bind_thread(USER_ID, THREAD_ID, "window-one", chat_id=-1001)
    dashboard_setup.bind_thread(USER_ID, THREAD_ID, "window-two", chat_id=-1002)

    resolved: list[str] = []
    for chat_id in (-1001, -1002):
        context = _make_context()
        await dashboard_module.dashboard_command(_make_update(chat_id=chat_id), context)
        button = context.bot.send_message.await_args.kwargs[
            "reply_markup"
        ].inline_keyboard[0][0]
        token = urlparse(button.web_app.url).path.removeprefix("/app/")
        resolved.append(verify_token(token, bot_token=BOT_TOKEN).window_id)

    assert resolved == ["window-one", "window-two"]


async def test_button_token_is_signed_and_scoped_to_effective_user(
    dashboard_setup: ThreadRouter,
) -> None:
    dashboard_setup.bind_thread(USER_ID, THREAD_ID, "window-one", chat_id=-1001)
    context = _make_context()

    await dashboard_module.dashboard_command(_make_update(), context)

    send_args = context.bot.send_message.await_args.kwargs
    assert send_args["chat_id"] == USER_ID
    button = send_args["reply_markup"].inline_keyboard[0][0]
    assert button.web_app is not None
    token = urlparse(button.web_app.url).path.removeprefix("/app/")
    payload = verify_token(token, bot_token=BOT_TOKEN)
    assert payload.window_id == "window-one"
    assert payload.user_id == USER_ID

    body, signature = token.split(".", 1)
    changed_signature = ("A" if signature[0] != "A" else "B") + signature[1:]
    tampered = f"{body}.{changed_signature}"
    with pytest.raises(InvalidTokenError, match="signature"):
        verify_token(tampered, bot_token=BOT_TOKEN)
    with pytest.raises(InvalidTokenError, match="user mismatch"):
        authorize_api_request(
            bot_token=BOT_TOKEN,
            token=token,
            init_data=_make_init_data(BOT_TOKEN, USER_ID + 1),
        )


async def test_dashboard_button_is_built_for_every_command(
    dashboard_setup: ThreadRouter,
) -> None:
    dashboard_setup.bind_thread(USER_ID, THREAD_ID, "window-one", chat_id=-1001)
    from ccgram.handlers.status import status_bar_actions

    with patch.object(
        status_bar_actions,
        "build_dashboard_button",
        wraps=status_bar_actions.build_dashboard_button,
    ) as build_button:
        for _ in range(2):
            await dashboard_module.dashboard_command(_make_update(), _make_context())

    assert build_button.call_count == 2


async def test_disabled_mini_app_replies_in_topic(
    dashboard_setup: ThreadRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    dashboard_setup.bind_thread(USER_ID, THREAD_ID, "window-one", chat_id=-1001)
    monkeypatch.setattr(dashboard_module.config, "miniapp_base_url", "")
    update = _make_update()
    context = _make_context()

    await dashboard_module.dashboard_command(update, context)

    assert "disabled" in update.message.reply_text.await_args.args[0].lower()
    context.bot.send_message.assert_not_awaited()


async def test_unbound_topic_replies_in_topic(dashboard_setup: ThreadRouter) -> None:
    update = _make_update()
    context = _make_context()

    await dashboard_module.dashboard_command(update, context)

    assert "not bound" in update.message.reply_text.await_args.args[0].lower()
    context.bot.send_message.assert_not_awaited()


async def test_forbidden_dm_gives_start_then_retry_hint(
    dashboard_setup: ThreadRouter,
) -> None:
    dashboard_setup.bind_thread(USER_ID, THREAD_ID, "window-one", chat_id=-1001)
    context = _make_context()
    context.bot.send_message.side_effect = Forbidden("bot can't initiate conversation")
    update = _make_update()

    await dashboard_module.dashboard_command(update, context)

    response = update.message.reply_text.await_args.args[0].lower()
    assert "/start" in response
    assert "retry" in response


async def test_unauthorized_user_does_not_receive_dashboard(
    dashboard_setup: ThreadRouter,
) -> None:
    dashboard_setup.bind_thread(USER_ID, THREAD_ID, "window-one", chat_id=-1001)
    context = _make_context()
    update = _make_update(user_id=USER_ID + 1)

    await dashboard_module.dashboard_command(update, context)

    context.bot.send_message.assert_not_awaited()
    update.message.reply_text.assert_not_awaited()


async def test_private_start_greeting_for_allowed_user(
    dashboard_setup: ThreadRouter,
) -> None:
    update = _make_update(chat_id=USER_ID, thread_id=None)
    update.effective_chat.type = "private"

    await dashboard_module.private_start_greeting(update, _make_context())

    assert "dashboard" in update.message.reply_text.await_args.args[0].lower()


async def test_private_start_greeting_silently_ignores_denied_user(
    dashboard_setup: ThreadRouter,
) -> None:
    update = _make_update(user_id=USER_ID + 1, chat_id=USER_ID + 1, thread_id=None)
    update.effective_chat.type = "private"

    await dashboard_module.private_start_greeting(update, _make_context())

    update.message.reply_text.assert_not_awaited()
