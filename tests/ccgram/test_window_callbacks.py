"""Tests for window picker callback handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

from telegram import CallbackQuery, Update
from telegram.ext import ContextTypes

from ccgram.handlers.callback_data import CB_WIN_BIND, CB_WIN_CANCEL, CB_WIN_NEW
from ccgram.handlers.directory_browser import UNBOUND_WINDOWS_KEY
from ccgram.handlers.user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT
from ccgram.handlers.window_callbacks import handle_window_callback


def _make_query_update_context(
    thread_id: int = 42,
    user_data: dict | None = None,
) -> tuple[AsyncMock, MagicMock, MagicMock]:
    query = AsyncMock(spec=CallbackQuery)
    query.answer = AsyncMock()

    msg = MagicMock()
    msg.message_thread_id = thread_id

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
            patch("ccgram.handlers.window_callbacks.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.window_callbacks.tmux_manager.find_window_by_id",
                new_callable=AsyncMock,
                return_value=mock_window,
            ),
            patch("ccgram.handlers.window_callbacks.safe_edit") as mock_edit,
            patch("ccgram.handlers.window_callbacks.format_topic_name_for_mode"),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_approval_mode.return_value = "normal"
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

            mock_sm.bind_thread.assert_called_once_with(
                100, 42, "@5", window_name="my-project"
            )
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
            patch("ccgram.handlers.window_callbacks.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.window_callbacks.tmux_manager.find_window_by_id",
                new_callable=AsyncMock,
                return_value=mock_window,
            ),
            patch("ccgram.handlers.window_callbacks.safe_edit"),
            patch("ccgram.handlers.window_callbacks.format_topic_name_for_mode"),
        ):
            mock_sm.resolve_chat_id.return_value = -100
            mock_sm.get_approval_mode.return_value = "normal"
            mock_sm.send_to_window = AsyncMock(return_value=(True, "ok"))
            await handle_window_callback(query, 100, f"{CB_WIN_BIND}0", update, context)

            mock_sm.send_to_window.assert_called_once_with("@5", "hello agent")
            assert PENDING_THREAD_TEXT not in context.user_data


class TestNewWindowCallback:
    async def test_transitions_to_directory_browser(self) -> None:
        user_data = {PENDING_THREAD_ID: 42}
        query, update, context = _make_query_update_context(user_data=user_data)

        with (
            patch(
                "ccgram.handlers.window_callbacks.build_directory_browser",
                return_value=("Browse:", MagicMock(), ["/a", "/b"]),
            ),
            patch("ccgram.handlers.window_callbacks.safe_edit") as mock_edit,
            patch("ccgram.handlers.window_callbacks.clear_window_picker_state"),
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


class TestCancelCallback:
    async def test_cancel_clears_state(self) -> None:
        user_data = {
            PENDING_THREAD_ID: 42,
            PENDING_THREAD_TEXT: "some text",
        }
        query, update, context = _make_query_update_context(user_data=user_data)

        with (
            patch("ccgram.handlers.window_callbacks.safe_edit") as mock_edit,
            patch("ccgram.handlers.window_callbacks.clear_window_picker_state"),
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
