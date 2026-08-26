"""Integration tests for PTB Application handler dispatch.

Tests that handlers are correctly registered and PTB routes updates
to the right handler functions. Uses a real PTB Application (``dispatch_app``)
with mocked external dependencies (Bot API, TmuxManager, SessionManager).
"""

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from telegram import Chat

pytestmark = pytest.mark.integration

TEST_USER_ID = 12345
TEST_THREAD_ID = 42


class TestTextRouting:
    @pytest.mark.parametrize(
        "text",
        ["hello world", "!ls -la"],
        ids=["plain", "bang-prefix"],
    )
    async def test_text_reaches_text_handler_verbatim(
        self, dispatch_app, make_text_update, text: str
    ) -> None:
        update = make_text_update(text, bot=dispatch_app.bot)

        with (
            patch(
                "ccgram.handlers.text.text_handler.handle_text_message",
                new_callable=AsyncMock,
            ) as mock_handler,
            patch(
                "ccgram.handlers.text.text_handler.config.is_user_allowed",
                return_value=True,
            ),
        ):
            await dispatch_app.process_update(update)

        mock_handler.assert_awaited_once()
        forwarded = mock_handler.call_args[0][0].message
        assert forwarded.text == text
        assert forwarded.message_thread_id == TEST_THREAD_ID

    async def test_unauthorized_user_rejected(
        self, dispatch_app, make_text_update
    ) -> None:
        update = make_text_update("hello", bot=dispatch_app.bot, user_id=99999)

        with (
            patch(
                "ccgram.handlers.text.text_handler.handle_text_message",
                new_callable=AsyncMock,
            ) as mock_handler,
            patch(
                "ccgram.handlers.text.text_handler.config.is_user_allowed",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.text.text_handler.safe_reply", new_callable=AsyncMock
            ),
        ):
            await dispatch_app.process_update(update)

        mock_handler.assert_not_awaited()


class TestCommandRouting:
    async def test_start_command_dispatched(
        self, dispatch_app, make_text_update
    ) -> None:
        update = make_text_update("/start", bot=dispatch_app.bot)

        with (
            patch(
                "ccgram.handlers.topics.new_command.safe_reply", new_callable=AsyncMock
            ) as mock_reply,
            patch(
                "ccgram.handlers.topics.new_command.config.is_user_allowed",
                return_value=True,
            ),
        ):
            await dispatch_app.process_update(update)

        mock_reply.assert_awaited_once()

    async def test_registered_command_wins_over_text_handler(
        self, dispatch_app, make_text_update
    ) -> None:
        """/history goes to its CommandHandler, never to the TEXT fallback."""
        update = make_text_update("/history", bot=dispatch_app.bot)

        with (
            patch(
                "ccgram.handlers.recovery.history.config.is_user_allowed",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.text.text_handler.handle_text_message",
                new_callable=AsyncMock,
            ) as mock_text,
            patch(
                "ccgram.handlers.recovery.history.thread_router.resolve_window_for_thread",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.recovery.history.safe_reply", new_callable=AsyncMock
            ) as mock_reply,
        ):
            await dispatch_app.process_update(update)

        mock_reply.assert_awaited_once()
        mock_text.assert_not_awaited()

    @pytest.mark.parametrize(
        "command",
        ["/new", "/sometool"],
        ids=["provider-command", "unknown-command"],
    )
    async def test_unregistered_command_forwarded_to_window(
        self, dispatch_app, make_text_update, command: str
    ) -> None:
        update = make_text_update(command, bot=dispatch_app.bot)

        with (
            patch(
                "ccgram.handlers.commands.forward.config.is_user_allowed",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.commands.forward.thread_router.resolve_window_for_thread",
                return_value="@0",
            ),
            patch(
                "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
                new_callable=AsyncMock,
                return_value=MagicMock(window_id="@0"),
            ),
            patch(
                "ccgram.handlers.commands.forward.send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, "Sent"),
            ) as mock_send,
            patch(
                "ccgram.handlers.commands.forward.thread_router.get_display_name",
                return_value="test-win",
            ),
            patch(
                "ccgram.handlers.commands.forward.safe_reply", new_callable=AsyncMock
            ),
            patch.object(Chat, "send_action", new_callable=AsyncMock),
        ):
            await dispatch_app.process_update(update)

        mock_send.assert_awaited_once_with(
            TEST_USER_ID, "@0", TEST_THREAD_ID, command, ANY
        )
