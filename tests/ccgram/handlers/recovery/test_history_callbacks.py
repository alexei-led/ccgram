"""Regression coverage for lossless history callback targets."""

from ccgram.handlers.callback_data import CB_HISTORY_NEXT
from ccgram.handlers.callback_tokens import resolve_callback_data
from ccgram.handlers.recovery.history import _build_history_keyboard


def test_history_keyboard_uses_lossless_token_for_herdr_target() -> None:
    window_id = "herdr-session-v1-" + "a" * 64
    keyboard = _build_history_keyboard(window_id, page_index=0, total_pages=2)
    assert keyboard is not None
    callback_data = keyboard.inline_keyboard[0][-1].callback_data
    assert isinstance(callback_data, str)
    assert len(callback_data.encode("utf-8")) <= 64
    assert resolve_callback_data(
        callback_data, 1, lambda _uid, wid: wid == window_id
    ) == f"{CB_HISTORY_NEXT}1:{window_id}:0:0"
