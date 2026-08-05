"""Regression coverage for lossless and authorized history callbacks."""

from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.handlers.callback_data import CB_HISTORY_NEXT
from ccgram.handlers.callback_tokens import resolve_callback_data
from ccgram.handlers.recovery.history import _build_history_keyboard
from ccgram.handlers.recovery.history_callbacks import handle_history_callback


def test_history_keyboard_uses_lossless_token_for_herdr_target() -> None:
    window_id = "herdr-session-v1-" + "a" * 64
    keyboard = _build_history_keyboard(window_id, page_index=0, total_pages=2)
    assert keyboard is not None
    callback_data = keyboard.inline_keyboard[0][-1].callback_data
    assert isinstance(callback_data, str)
    assert len(callback_data.encode("utf-8")) <= 64
    assert (
        resolve_callback_data(callback_data, 1, lambda _uid, wid: wid == window_id)
        == f"{CB_HISTORY_NEXT}1:{window_id}:0:0"
    )


async def test_colon_containing_legacy_history_id_stays_legacy() -> None:
    query = AsyncMock()
    legacy_id = "w2:t1:agent"
    data = f"{CB_HISTORY_NEXT}1:{legacy_id}"
    with (
        patch(
            "ccgram.handlers.recovery.history_callbacks.user_owns_window",
            return_value=True,
        ),
        patch("ccgram.handlers.recovery.history_callbacks.tmux_manager") as mux,
        patch(
            "ccgram.handlers.recovery.history_callbacks.send_history",
            new_callable=AsyncMock,
        ) as send,
    ):
        mux.find_window_by_id = AsyncMock(return_value=object())
        await handle_history_callback(query, 7, data, MagicMock(), MagicMock())

    mux.find_window_by_id.assert_awaited_once_with(legacy_id)
    call = send.await_args
    assert call is not None
    assert call.kwargs["start_byte"] == 0
    assert call.kwargs["end_byte"] == 0


async def test_raw_history_callback_checks_ownership_before_window_lookup() -> None:
    query = AsyncMock()
    data = f"{CB_HISTORY_NEXT}1:someone-elses-window:0:0"
    with (
        patch(
            "ccgram.handlers.recovery.history_callbacks.user_owns_window",
            return_value=False,
        ) as owns,
        patch("ccgram.handlers.recovery.history_callbacks.tmux_manager") as mux,
        patch(
            "ccgram.handlers.recovery.history_callbacks.send_history",
            new_callable=AsyncMock,
        ) as send,
    ):
        await handle_history_callback(query, 7, data, MagicMock(), MagicMock())

    owns.assert_called_once_with(7, "someone-elses-window")
    mux.find_window_by_id.assert_not_called()
    send.assert_not_awaited()
    query.answer.assert_awaited_once_with("Not your session", show_alert=True)
