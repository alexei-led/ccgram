"""Tests for the /start welcome command."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.bot import create_bot
from ccgram.handlers.topics import new_command


_NC = "ccgram.handlers.topics.new_command"


def _make_update(user_id: int, thread_id: int | None = None) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    update.message = AsyncMock()
    update.message.message_thread_id = thread_id
    return update


def _make_context() -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = {}
    return ctx


@pytest.fixture(autouse=True)
def _allow_user():
    with (
        patch(f"{_NC}.config.is_user_allowed", return_value=True),
        patch(f"{_NC}.commands_command", new_callable=AsyncMock) as panel,
    ):
        yield panel


class TestNewCommand:
    async def test_sends_welcome(self, _allow_user: AsyncMock) -> None:
        update = _make_update(100)
        ctx = _make_context()

        await new_command(update, ctx)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "CCGram" in text
        _allow_user.assert_awaited_once_with(update, ctx)

    async def test_clears_browse_state(self) -> None:
        update = _make_update(100)
        ctx = _make_context()

        with patch(f"{_NC}.clear_browse_state") as mock_clear:
            await new_command(update, ctx)
            mock_clear.assert_called_once_with(ctx.user_data)

    async def test_clears_stranded_worktree_and_thread_state(self) -> None:
        update = _make_update(100)
        ctx = _make_context()
        ctx.user_data.update(
            {
                "_pending_worktree_creating": True,
                "_pending_worktree_repo": "/repo",
                "_pending_worktree_branch": "ccg/x",
                "_pending_thread_id": 42,
                "_pending_thread_text": "hi",
                "state": "selecting_window",
                "unbound_windows": ["@5"],
            }
        )

        await new_command(update, ctx)

        assert ctx.user_data == {}

    async def test_unauthorized_user(self) -> None:
        update = _make_update(999)
        ctx = _make_context()

        with patch(f"{_NC}.config.is_user_allowed", return_value=False):
            await new_command(update, ctx)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "not authorized" in text

    async def test_no_message(self) -> None:
        update = _make_update(100)
        update.message = None
        ctx = _make_context()

        await new_command(update, ctx)

    async def test_no_user(self) -> None:
        update = _make_update(100)
        update.effective_user = None
        ctx = _make_context()

        await new_command(update, ctx)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "not authorized" in text


class TestCommandRegistration:
    @patch("ccgram.bot.config")
    def test_start_registered_and_new_is_provider_forwardable(
        self, mock_config: MagicMock
    ) -> None:
        mock_config.telegram_bot_token = "fake:token"
        app = create_bot()

        handler_commands: list[str] = []
        for group_handlers in app.handlers.values():
            for handler in group_handlers:
                if hasattr(handler, "commands"):
                    handler_commands.extend(handler.commands)  # type: ignore[union-attr]

        assert "start" in handler_commands
        assert "new" not in handler_commands

    @patch("ccgram.bot.config")
    def test_start_uses_welcome_command(self, mock_config: MagicMock) -> None:
        mock_config.telegram_bot_token = "fake:token"
        app = create_bot()

        start_handler = None
        for group_handlers in app.handlers.values():
            for handler in group_handlers:
                if hasattr(handler, "commands") and "start" in handler.commands:  # type: ignore[union-attr]
                    start_handler = handler

        assert start_handler is not None
        assert start_handler.callback is new_command
