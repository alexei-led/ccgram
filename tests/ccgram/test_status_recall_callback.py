"""Tests for status recall callback handling."""

from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.handlers.callback_data import CB_STATUS_RECALL
from ccgram.handlers.screenshot_callbacks import handle_screenshot_callback


def _make_query() -> AsyncMock:
    query = AsyncMock()
    query.answer = AsyncMock()
    return query


async def test_status_recall_sends_selected_history_command() -> None:
    query = _make_query()
    update = MagicMock()
    context = MagicMock()

    with (
        patch(
            "ccgram.handlers.screenshot_callbacks.user_owns_window", return_value=True
        ),
        patch("ccgram.handlers.screenshot_callbacks.get_thread_id", return_value=42),
        patch(
            "ccgram.handlers.screenshot_callbacks.session_manager.resolve_window_for_thread",
            return_value="@0",
        ),
        patch(
            "ccgram.handlers.command_history.get_history",
            return_value=["/status", "/clear"],
        ),
        patch(
            "ccgram.handlers.screenshot_callbacks.session_manager.send_to_window",
            new_callable=AsyncMock,
            return_value=(True, ""),
        ) as mock_send,
        patch("ccgram.handlers.command_history.record_command") as mock_record,
    ):
        await handle_screenshot_callback(
            query, 100, f"{CB_STATUS_RECALL}@0:1", update, context
        )

    mock_send.assert_awaited_once_with("@0", "/clear")
    mock_record.assert_called_once_with(100, 42, "/clear")
    query.answer.assert_awaited_once_with("\u21a9 Sent")


async def test_status_recall_rejects_stale_topic_binding() -> None:
    query = _make_query()
    update = MagicMock()
    context = MagicMock()

    with (
        patch(
            "ccgram.handlers.screenshot_callbacks.user_owns_window", return_value=True
        ),
        patch("ccgram.handlers.screenshot_callbacks.get_thread_id", return_value=42),
        patch(
            "ccgram.handlers.screenshot_callbacks.session_manager.resolve_window_for_thread",
            return_value="@9",
        ),
        patch(
            "ccgram.handlers.screenshot_callbacks.session_manager.send_to_window",
            new_callable=AsyncMock,
        ) as mock_send,
    ):
        await handle_screenshot_callback(
            query, 100, f"{CB_STATUS_RECALL}@0:0", update, context
        )

    mock_send.assert_not_called()
    query.answer.assert_awaited_once_with("Stale status button", show_alert=True)


async def test_status_recall_handles_missing_history_entry() -> None:
    query = _make_query()
    update = MagicMock()
    context = MagicMock()

    with (
        patch(
            "ccgram.handlers.screenshot_callbacks.user_owns_window", return_value=True
        ),
        patch("ccgram.handlers.screenshot_callbacks.get_thread_id", return_value=42),
        patch(
            "ccgram.handlers.screenshot_callbacks.session_manager.resolve_window_for_thread",
            return_value="@0",
        ),
        patch("ccgram.handlers.command_history.get_history", return_value=["/status"]),
        patch(
            "ccgram.handlers.screenshot_callbacks.session_manager.send_to_window",
            new_callable=AsyncMock,
        ) as mock_send,
    ):
        await handle_screenshot_callback(
            query, 100, f"{CB_STATUS_RECALL}@0:1", update, context
        )

    mock_send.assert_not_called()
    query.answer.assert_awaited_once_with("Command not found", show_alert=True)
