"""Tests for window picker callback handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Bot, CallbackQuery, Update
from telegram.ext import ContextTypes

from ccgram.handlers.callback_data import CB_WIN_BIND, CB_WIN_CANCEL, CB_WIN_NEW
from ccgram.handlers.topics.directory_browser import UNBOUND_WINDOWS_KEY
from ccgram.handlers.user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT
from ccgram.handlers.topics.window_callbacks import handle_window_callback


def _make_query_update_context(
    thread_id: int = 42,
    user_data: dict | None = None,
) -> tuple[AsyncMock, MagicMock, MagicMock]:
    query = AsyncMock(spec=CallbackQuery)
    query.answer = AsyncMock()

    msg = MagicMock()
    msg.message_thread_id = thread_id
    msg.chat.type = "supergroup"
    msg.chat.id = -100999

    update = MagicMock(spec=Update)
    update.message = None
    update.callback_query = MagicMock()
    update.callback_query.message = msg

    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.user_data = user_data if user_data is not None else {}
    context.bot = AsyncMock()
    return query, update, context


class TestBindWindowCallback:
    async def test_bind_existing_window(self) -> None:
        user_data = {
            UNBOUND_WINDOWS_KEY: ["@5"],
            PENDING_THREAD_ID: 42,
        }
        query, update, context = _make_query_update_context(user_data=user_data)

        mock_window = MagicMock()
        mock_window.window_name = "my-project"

        with (
            patch("ccgram.handlers.topics.window_callbacks.session_manager") as mock_sm,
            patch("ccgram.handlers.topics.window_callbacks.thread_router") as mock_tr,
            patch(
                "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
                new_callable=AsyncMock,
                return_value=mock_window,
            ),
            patch("ccgram.handlers.topics.window_callbacks.safe_edit") as mock_edit,
            patch("ccgram.handlers.topics.window_callbacks.format_topic_name_for_mode"),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_sm.get_approval_mode.return_value = "normal"
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

            mock_tr.bind_thread.assert_called_once_with(
                100, 42, "@5", window_name="my-project"
            )
            mock_tr.set_group_chat_id.assert_called_once_with(100, 42, -100999)
            mock_edit.assert_called_once()
            assert "my-project" in mock_edit.call_args[0][1]

    async def test_bind_invalid_index(self) -> None:
        user_data = {UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
        query, update, context = _make_query_update_context(user_data=user_data)

        await handle_window_callback(query, 100, f"{CB_WIN_BIND}abc", update, context)
        query.answer.assert_called_once_with("Invalid data")

    async def test_bind_out_of_range_index(self) -> None:
        user_data = {UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
        query, update, context = _make_query_update_context(user_data=user_data)

        await handle_window_callback(query, 100, f"{CB_WIN_BIND}5", update, context)
        query.answer.assert_called_once_with(
            "Window list changed, please retry", show_alert=True
        )

    async def test_bind_stale_topic_mismatch(self) -> None:
        user_data = {UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 99}
        query, update, context = _make_query_update_context(
            thread_id=42, user_data=user_data
        )

        await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)
        query.answer.assert_called_once_with(
            "Stale picker (topic mismatch)", show_alert=True
        )

    async def test_bind_rejected_after_new_command_reset(self) -> None:
        from ccgram.handlers.topics.new_command import new_command

        user_data = {UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
        query, update, context = _make_query_update_context(user_data=user_data)

        nc_update = MagicMock()
        nc_update.effective_user = MagicMock(id=100)
        nc_update.message = AsyncMock()
        with patch(
            "ccgram.handlers.topics.new_command.config.is_user_allowed",
            return_value=True,
        ):
            await new_command(nc_update, context)

        with patch(
            "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
            new_callable=AsyncMock,
        ) as mock_find:
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        mock_find.assert_not_called()
        query.answer.assert_called_once_with(
            "Window list changed, please retry", show_alert=True
        )

    async def test_bind_forwards_pending_text(self) -> None:
        user_data = {
            UNBOUND_WINDOWS_KEY: ["@5"],
            PENDING_THREAD_ID: 42,
            PENDING_THREAD_TEXT: "hello agent",
        }
        query, update, context = _make_query_update_context(user_data=user_data)

        mock_window = MagicMock()
        mock_window.window_name = "proj"

        with (
            patch("ccgram.handlers.topics.window_callbacks.session_manager") as mock_sm,
            patch("ccgram.handlers.topics.window_callbacks.thread_router") as mock_tr,
            patch(
                "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
                new_callable=AsyncMock,
                return_value=mock_window,
            ),
            patch("ccgram.handlers.topics.window_callbacks.safe_edit"),
            patch("ccgram.handlers.topics.window_callbacks.format_topic_name_for_mode"),
            patch(
                "ccgram.handlers.topics.window_callbacks.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, "ok"),
            ) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_sm.get_approval_mode.return_value = "normal"
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

            mock_send.assert_called_once_with("@5", "hello agent")
            assert PENDING_THREAD_TEXT not in context.user_data


class TestNewWindowCallback:
    async def test_transitions_to_directory_browser(self) -> None:
        user_data = {PENDING_THREAD_ID: 42}
        query, update, context = _make_query_update_context(user_data=user_data)

        with (
            patch(
                "ccgram.handlers.topics.window_callbacks.build_directory_browser",
                return_value=("Browse:", MagicMock(), ["/a", "/b"]),
            ),
            patch("ccgram.handlers.topics.window_callbacks.safe_edit") as mock_edit,
            patch("ccgram.handlers.topics.window_callbacks.clear_window_picker_state"),
        ):
            await handle_window_callback(query, 100, CB_WIN_NEW, update, context)

            mock_edit.assert_called_once()
            query.answer.assert_called_once_with()

    async def test_new_stale_topic_mismatch(self) -> None:
        user_data = {PENDING_THREAD_ID: 99}
        query, update, context = _make_query_update_context(
            thread_id=42, user_data=user_data
        )

        await handle_window_callback(query, 100, CB_WIN_NEW, update, context)
        query.answer.assert_called_once_with(
            "Stale picker (topic mismatch)", show_alert=True
        )

    async def test_new_rejected_after_new_command_reset(self) -> None:
        from ccgram.handlers.topics.new_command import new_command
        from ccgram.handlers.topics.directory_browser import (
            BROWSE_PATH_KEY,
            STATE_KEY,
        )

        user_data = {PENDING_THREAD_ID: 42}
        query, update, context = _make_query_update_context(user_data=user_data)

        nc_update = MagicMock()
        nc_update.effective_user = MagicMock(id=100)
        nc_update.message = AsyncMock()
        with patch(
            "ccgram.handlers.topics.new_command.config.is_user_allowed",
            return_value=True,
        ):
            await new_command(nc_update, context)

        with patch(
            "ccgram.handlers.topics.window_callbacks.build_directory_browser"
        ) as mock_build:
            await handle_window_callback(query, 100, CB_WIN_NEW, update, context)

        mock_build.assert_not_called()
        query.answer.assert_called_once_with(
            "Stale picker (flow reset)", show_alert=True
        )
        assert STATE_KEY not in context.user_data
        assert BROWSE_PATH_KEY not in context.user_data


class TestCancelCallback:
    async def test_cancel_clears_state(self) -> None:
        user_data = {
            PENDING_THREAD_ID: 42,
            PENDING_THREAD_TEXT: "some text",
        }
        query, update, context = _make_query_update_context(user_data=user_data)

        with (
            patch("ccgram.handlers.topics.window_callbacks.safe_edit") as mock_edit,
            patch("ccgram.handlers.topics.window_callbacks.clear_window_picker_state"),
        ):
            await handle_window_callback(query, 100, CB_WIN_CANCEL, update, context)

            mock_edit.assert_called_once_with(query, "Cancelled")
            query.answer.assert_called_once_with("Cancelled")
            assert PENDING_THREAD_ID not in context.user_data
            assert PENDING_THREAD_TEXT not in context.user_data

    async def test_cancel_stale_topic_mismatch(self) -> None:
        user_data = {PENDING_THREAD_ID: 99}
        query, update, context = _make_query_update_context(
            thread_id=42, user_data=user_data
        )

        await handle_window_callback(query, 100, CB_WIN_CANCEL, update, context)
        query.answer.assert_called_once_with(
            "Stale picker (topic mismatch)", show_alert=True
        )


class TestBindProviderDetection:
    async def test_bind_shell_window_offers_prompt_setup(self) -> None:
        user_data = {UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
        query, update, context = _make_query_update_context(user_data=user_data)

        mock_window = MagicMock()
        mock_window.window_name = "my-shell"
        mock_window.pane_current_command = "fish"
        mock_window.pane_tty = "/dev/ttys003"

        with (
            patch("ccgram.handlers.topics.window_callbacks.session_manager") as mock_sm,
            patch("ccgram.handlers.topics.window_callbacks.thread_router") as mock_tr,
            patch(
                "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
                new_callable=AsyncMock,
                return_value=mock_window,
            ),
            patch("ccgram.handlers.topics.window_callbacks.safe_edit"),
            patch("ccgram.handlers.topics.window_callbacks.format_topic_name_for_mode"),
            patch(
                "ccgram.providers.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="shell",
            ),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ) as mock_ensure,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_sm.get_approval_mode.return_value = "normal"
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        mock_sm.set_window_provider.assert_called_once_with("@5", "shell")
        mock_ensure.assert_awaited_once()
        call_args = mock_ensure.call_args
        assert call_args[0] == ("@5", "external_bind")

    async def test_bind_bare_herdr_shell_still_detects_via_foreground(self) -> None:
        # herdr leaves pane_current_command empty for a bare shell pane; the
        # bind path must still run detection (which consults the foreground
        # process via window_id) instead of skipping it.
        user_data = {UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
        query, update, context = _make_query_update_context(user_data=user_data)

        mock_window = MagicMock()
        mock_window.window_name = "bare-shell"
        mock_window.pane_current_command = ""

        with (
            patch("ccgram.handlers.topics.window_callbacks.session_manager") as mock_sm,
            patch("ccgram.handlers.topics.window_callbacks.thread_router") as mock_tr,
            patch(
                "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
                new_callable=AsyncMock,
                return_value=mock_window,
            ),
            patch("ccgram.handlers.topics.window_callbacks.safe_edit"),
            patch("ccgram.handlers.topics.window_callbacks.format_topic_name_for_mode"),
            patch(
                "ccgram.providers.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="shell",
            ) as mock_detect,
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ) as mock_ensure,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_sm.get_approval_mode.return_value = "normal"
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        mock_detect.assert_awaited_once_with("", window_id="@5")
        mock_sm.set_window_provider.assert_called_once_with("@5", "shell")
        mock_ensure.assert_awaited_once()

    async def test_bind_claude_window_does_not_offer_prompt_setup(self) -> None:
        user_data = {UNBOUND_WINDOWS_KEY: ["@5"], PENDING_THREAD_ID: 42}
        query, update, context = _make_query_update_context(user_data=user_data)

        mock_window = MagicMock()
        mock_window.window_name = "my-project"
        mock_window.pane_current_command = "claude"

        with (
            patch("ccgram.handlers.topics.window_callbacks.session_manager") as mock_sm,
            patch("ccgram.handlers.topics.window_callbacks.thread_router") as mock_tr,
            patch(
                "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
                new_callable=AsyncMock,
                return_value=mock_window,
            ),
            patch("ccgram.handlers.topics.window_callbacks.safe_edit"),
            patch("ccgram.handlers.topics.window_callbacks.format_topic_name_for_mode"),
            patch(
                "ccgram.providers.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="claude",
            ),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ) as mock_ensure,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_sm.get_approval_mode.return_value = "normal"
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        mock_ensure.assert_not_awaited()

    async def test_bind_shell_pending_text_routes_through_shell_handler(self) -> None:
        user_data = {
            UNBOUND_WINDOWS_KEY: ["@5"],
            PENDING_THREAD_ID: 42,
            PENDING_THREAD_TEXT: "ls -la",
        }
        query, update, context = _make_query_update_context(user_data=user_data)

        mock_window = MagicMock()
        mock_window.window_name = "my-shell"
        mock_window.pane_current_command = "bash"

        with (
            patch("ccgram.handlers.topics.window_callbacks.session_manager") as mock_sm,
            patch("ccgram.handlers.topics.window_callbacks.thread_router") as mock_tr,
            patch(
                "ccgram.multiplexer.tmux.tmux_manager.find_window_by_id",
                new_callable=AsyncMock,
                return_value=mock_window,
            ),
            patch("ccgram.handlers.topics.window_callbacks.safe_edit"),
            patch("ccgram.handlers.topics.window_callbacks.format_topic_name_for_mode"),
            patch(
                "ccgram.providers.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="shell",
            ),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topics.window_callbacks._forward_pending_text",
                new_callable=AsyncMock,
            ) as mock_forward,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_sm.get_approval_mode.return_value = "normal"
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

        mock_forward.assert_awaited_once()
        forward_args = mock_forward.call_args.args
        assert forward_args[1:] == (100, 42, "@5", "ls -la", "shell")
        assert forward_args[0].bot is context.bot
        assert mock_forward.call_args.kwargs == {"is_existing_window": True}


class TestForwardPendingText:
    async def test_existing_shell_window_sends_raw(self) -> None:
        from ccgram.handlers.topics.window_callbacks import _forward_pending_text

        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.topics.window_callbacks.session_manager"),
            patch(
                "ccgram.handlers.shell.shell_commands.handle_shell_message",
                new_callable=AsyncMock,
            ) as mock_shell,
            patch(
                "ccgram.handlers.topics.window_callbacks.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as mock_send,
        ):
            await _forward_pending_text(
                bot, 1, 42, "@5", "list files", "shell", is_existing_window=True
            )

        mock_shell.assert_not_awaited()
        mock_send.assert_called_once_with("@5", "list files")

    async def test_new_shell_window_routes_through_handler(self) -> None:
        from ccgram.handlers.topics.window_callbacks import _forward_pending_text

        bot = AsyncMock(spec=Bot)
        with patch(
            "ccgram.handlers.shell.shell_commands.handle_shell_message",
            new_callable=AsyncMock,
        ) as mock_shell:
            await _forward_pending_text(
                bot, 1, 42, "@5", "list files", "shell", is_existing_window=False
            )

        mock_shell.assert_awaited_once()
        args = mock_shell.call_args.args
        assert args[1:] == (1, 42, "@5", "list files")
        assert args[0] is bot
