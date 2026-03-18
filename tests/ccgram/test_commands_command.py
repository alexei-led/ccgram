"""Tests for /commands handler and scoped provider command menus."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import BotCommandScopeChat, BotCommandScopeChatMember

import ccgram.bot as bot_mod
from ccgram.bot import (
    _chat_scoped_provider_menu,
    _scoped_provider_menu,
    _sync_scoped_provider_menu,
    commands_command,
)
from ccgram.cc_commands import CCCommand


def _make_update(
    *,
    user_id: int = 100,
    thread_id: int = 42,
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    msg = AsyncMock()
    msg.message_thread_id = thread_id
    msg.chat.type = "supergroup"
    msg.chat.id = -100999
    msg.chat.is_forum = True
    msg.is_topic_message = True
    update.message = msg
    return update


@pytest.fixture(autouse=True)
def _allow_user():
    with patch("ccgram.bot.is_user_allowed", return_value=True):
        yield


@pytest.fixture(autouse=True)
def _clean_scoped_caches():
    _scoped_provider_menu.clear()
    _chat_scoped_provider_menu.clear()
    bot_mod._global_provider_menu = None
    yield
    _scoped_provider_menu.clear()
    _chat_scoped_provider_menu.clear()
    bot_mod._global_provider_menu = None


class TestCommandsCommand:
    async def test_unauthorized_user_returns_early(self) -> None:
        with (
            patch("ccgram.bot.is_user_allowed", return_value=False),
            patch("ccgram.bot.session_manager") as mock_sm,
        ):
            await commands_command(_make_update(), MagicMock())

        mock_sm.resolve_window_for_thread.assert_not_called()

    async def test_no_message_returns_early(self) -> None:
        update = _make_update()
        update.message = None
        with patch("ccgram.bot.session_manager") as mock_sm:
            await commands_command(update, MagicMock())
        mock_sm.resolve_window_for_thread.assert_not_called()

    async def test_unbound_topic_reports_error(self) -> None:
        update = _make_update()
        with (
            patch("ccgram.bot.session_manager") as mock_sm,
            patch("ccgram.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            mock_sm.resolve_window_for_thread.return_value = None
            await commands_command(update, MagicMock())

        mock_reply.assert_called_once()
        assert "No session bound" in mock_reply.call_args.args[1]

    async def test_no_discoverable_commands_reports_provider(self) -> None:
        update = _make_update()
        provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
        with (
            patch("ccgram.bot.session_manager") as mock_sm,
            patch("ccgram.bot.get_provider_for_window", return_value=provider),
            patch("ccgram.bot._sync_scoped_provider_menu", new_callable=AsyncMock),
            patch("ccgram.bot.discover_provider_commands", return_value=[]),
            patch("ccgram.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            mock_sm.resolve_window_for_thread.return_value = "@1"
            await commands_command(update, MagicMock())

        mock_reply.assert_called_once()
        text = mock_reply.call_args.args[1]
        assert "Provider: `codex`" in text
        assert "No discoverable commands" in text

    async def test_lists_provider_commands_with_original_mapping(self) -> None:
        update = _make_update()
        provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
        discovered = [
            CCCommand(
                name="spec:work",
                telegram_name="spec_work",
                description="↗ work",
                source="command",
            ),
            CCCommand(
                name="/status",
                telegram_name="status",
                description="↗ status",
                source="builtin",
            ),
            CCCommand(
                name="ignored",
                telegram_name="",
                description="↗ ignored",
                source="command",
            ),
        ]
        with (
            patch("ccgram.bot.session_manager") as mock_sm,
            patch("ccgram.bot.get_provider_for_window", return_value=provider),
            patch(
                "ccgram.bot._sync_scoped_provider_menu", new_callable=AsyncMock
            ) as mock_sync,
            patch("ccgram.bot.discover_provider_commands", return_value=discovered),
            patch("ccgram.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            mock_sm.resolve_window_for_thread.return_value = "@1"
            await commands_command(update, MagicMock())

        mock_sync.assert_called_once_with(update.message, 100, provider)
        mock_reply.assert_called_once()
        text = mock_reply.call_args.args[1]
        assert "Provider: `codex`" in text
        assert "`/spec_work`" in text and "`/spec:work`" in text
        assert "`/status`" in text
        assert "ignored" not in text


class TestScopedProviderMenuSync:
    async def test_caches_provider_menu_per_chat_user(self) -> None:
        _scoped_provider_menu.clear()
        _chat_scoped_provider_menu.clear()
        bot_mod._global_provider_menu = None
        try:
            message = AsyncMock()
            message.chat.id = -100999
            message.get_bot.return_value = object()
            provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))

            with patch(
                "ccgram.bot.register_commands", new_callable=AsyncMock
            ) as mock_reg:
                await _sync_scoped_provider_menu(message, 100, provider)
                await _sync_scoped_provider_menu(message, 100, provider)

            mock_reg.assert_called_once()
            assert _scoped_provider_menu[(-100999, 100)] == "codex"
        finally:
            _scoped_provider_menu.clear()
            _chat_scoped_provider_menu.clear()
            bot_mod._global_provider_menu = None

    async def test_cache_updates_when_provider_changes(self) -> None:
        _scoped_provider_menu.clear()
        _chat_scoped_provider_menu.clear()
        bot_mod._global_provider_menu = None
        try:
            message = AsyncMock()
            message.chat.id = -100999
            message.get_bot.return_value = object()
            codex = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
            claude = SimpleNamespace(capabilities=SimpleNamespace(name="claude"))

            with patch(
                "ccgram.bot.register_commands", new_callable=AsyncMock
            ) as mock_reg:
                await _sync_scoped_provider_menu(message, 100, codex)
                await _sync_scoped_provider_menu(message, 100, claude)

            assert mock_reg.call_count == 2
            assert _scoped_provider_menu[(-100999, 100)] == "claude"
        finally:
            _scoped_provider_menu.clear()
            _chat_scoped_provider_menu.clear()
            bot_mod._global_provider_menu = None

    async def test_register_failure_does_not_update_cache(self) -> None:
        _scoped_provider_menu.clear()
        _chat_scoped_provider_menu.clear()
        bot_mod._global_provider_menu = None
        try:
            message = AsyncMock()
            message.chat.id = -100999
            message.get_bot.return_value = object()
            provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))

            with patch(
                "ccgram.bot.register_commands",
                new_callable=AsyncMock,
                side_effect=OSError("boom"),
            ):
                await _sync_scoped_provider_menu(message, 100, provider)

            assert (-100999, 100) not in _scoped_provider_menu
        finally:
            _scoped_provider_menu.clear()
            _chat_scoped_provider_menu.clear()
            bot_mod._global_provider_menu = None

    async def test_falls_back_to_chat_scope_when_member_scope_fails(self) -> None:
        _scoped_provider_menu.clear()
        _chat_scoped_provider_menu.clear()
        bot_mod._global_provider_menu = None
        try:
            message = AsyncMock()
            message.chat.id = -100999
            message.get_bot.return_value = object()
            provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))

            with patch(
                "ccgram.bot.register_commands",
                new_callable=AsyncMock,
                side_effect=[OSError("member"), None],
            ) as mock_reg:
                await _sync_scoped_provider_menu(message, 100, provider)

            assert mock_reg.call_count == 2
            first_scope = mock_reg.call_args_list[0].kwargs["scope"]
            second_scope = mock_reg.call_args_list[1].kwargs["scope"]
            assert isinstance(first_scope, BotCommandScopeChatMember)
            assert isinstance(second_scope, BotCommandScopeChat)
            assert _chat_scoped_provider_menu[-100999] == "codex"
            assert _scoped_provider_menu[(-100999, 100)] == "codex"
        finally:
            _scoped_provider_menu.clear()
            _chat_scoped_provider_menu.clear()
            bot_mod._global_provider_menu = None

    async def test_falls_back_to_global_when_member_and_chat_scope_fail(self) -> None:
        _scoped_provider_menu.clear()
        _chat_scoped_provider_menu.clear()
        bot_mod._global_provider_menu = None
        try:
            message = AsyncMock()
            message.chat.id = -100999
            message.get_bot.return_value = object()
            provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))

            with patch(
                "ccgram.bot.register_commands",
                new_callable=AsyncMock,
                side_effect=[OSError("member"), OSError("chat"), None],
            ) as mock_reg:
                await _sync_scoped_provider_menu(message, 100, provider)

            assert mock_reg.call_count == 3
            assert "scope" in mock_reg.call_args_list[0].kwargs
            assert "scope" in mock_reg.call_args_list[1].kwargs
            assert "scope" not in mock_reg.call_args_list[2].kwargs
            assert _scoped_provider_menu[(-100999, 100)] == "codex"
        finally:
            _scoped_provider_menu.clear()
            _chat_scoped_provider_menu.clear()
            bot_mod._global_provider_menu = None

    async def test_scoped_menu_cache_is_bounded(self) -> None:
        _scoped_provider_menu.clear()
        _chat_scoped_provider_menu.clear()
        bot_mod._global_provider_menu = None
        try:
            message = AsyncMock()
            message.chat.id = -100999
            message.get_bot.return_value = object()
            provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))

            with (
                patch("ccgram.bot._MAX_SCOPED_PROVIDER_MENU_ENTRIES", 1),
                patch("ccgram.bot.register_commands", new_callable=AsyncMock),
            ):
                await _sync_scoped_provider_menu(message, 100, provider)
                await _sync_scoped_provider_menu(message, 101, provider)

            assert len(_scoped_provider_menu) == 1
        finally:
            _scoped_provider_menu.clear()
            _chat_scoped_provider_menu.clear()
            bot_mod._global_provider_menu = None
