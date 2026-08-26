"""Regression coverage for lossless and authorized history callbacks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_data import CB_HISTORY_NEXT
from ccgram.handlers.callback_tokens import resolve_callback_data
from ccgram.handlers.recovery.history import _build_history_keyboard
from ccgram.handlers.recovery.history_callbacks import (
    _dispatch,
    handle_history_callback,
)

_HC = "ccgram.handlers.recovery.history_callbacks"


@pytest.fixture()
def env():
    """Owned window, alive, with history rendering stubbed out."""
    with (
        patch(f"{_HC}.user_owns_window", return_value=True) as owns,
        patch(f"{_HC}.tmux_manager") as mux,
        patch(f"{_HC}.safe_edit", new_callable=AsyncMock) as edit,
        patch(f"{_HC}.send_history", new_callable=AsyncMock) as send,
    ):
        mux.find_window_by_id = AsyncMock(return_value=object())
        yield MagicMock(owns=owns, mux=mux, edit=edit, send=send)


async def _handle(data: str, user_id: int = 7) -> AsyncMock:
    query = AsyncMock()
    await handle_history_callback(query, user_id, data, MagicMock(), MagicMock())
    return query


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


class TestCallbackDataParsing:
    @pytest.mark.parametrize(
        ("payload", "window_id", "offset", "start_byte", "end_byte"),
        [
            ("1:@7:100:250", "@7", 1, 100, 250),
            ("0:@7:0:0", "@7", 0, 0, 0),
            ("2:@7", "@7", 2, 0, 0),
            ("1:w2:t1:agent", "w2:t1:agent", 1, 0, 0),
            ("1:w2:t1:0:0", "w2:t1", 1, 0, 0),
        ],
        ids=[
            "byte-range",
            "explicit-zero-range",
            "legacy-no-range",
            "legacy-colon-id",
            "colon-id-with-range",
        ],
    )
    async def test_window_id_and_byte_range(
        self,
        env,
        payload: str,
        window_id: str,
        offset: int,
        start_byte: int,
        end_byte: int,
    ) -> None:
        await _handle(f"{CB_HISTORY_NEXT}{payload}")

        env.mux.find_window_by_id.assert_awaited_once_with(window_id)
        call = env.send.await_args
        assert call is not None
        assert call.kwargs["offset"] == offset
        assert call.kwargs["start_byte"] == start_byte
        assert call.kwargs["end_byte"] == end_byte
        assert call.kwargs["edit"] is True

    @pytest.mark.parametrize(
        "payload",
        ["notanumber:@7", "1", ""],
        ids=["non-numeric-offset", "missing-window-id", "empty"],
    )
    async def test_malformed_data_rejected(self, env, payload: str) -> None:
        query = await _handle(f"{CB_HISTORY_NEXT}{payload}")

        query.answer.assert_awaited_once_with("Invalid data")
        env.send.assert_not_awaited()


class TestOwnershipAndLiveness:
    async def test_ownership_is_checked_before_window_lookup(self) -> None:
        data = f"{CB_HISTORY_NEXT}1:someone-elses-window:0:0"
        with (
            patch(f"{_HC}.user_owns_window", return_value=False) as owns,
            patch(f"{_HC}.tmux_manager") as mux,
            patch(f"{_HC}.send_history", new_callable=AsyncMock) as send,
        ):
            query = await _handle(data)

        owns.assert_called_once_with(7, "someone-elses-window")
        mux.find_window_by_id.assert_not_called()
        send.assert_not_awaited()
        query.answer.assert_awaited_once_with("Not your session", show_alert=True)

    async def test_dead_window_reports_instead_of_paging(self, env) -> None:
        env.mux.find_window_by_id = AsyncMock(return_value=None)

        query = await _handle(f"{CB_HISTORY_NEXT}1:@7:0:0")

        env.send.assert_not_awaited()
        assert env.edit.call_args.args[1] == "Window no longer exists."
        query.answer.assert_awaited_with("Page updated")


class TestDispatch:
    async def test_expired_token_reports_and_stops(self) -> None:
        update = MagicMock()
        update.callback_query = AsyncMock()
        update.callback_query.data = f"{CB_HISTORY_NEXT}expired"
        update.effective_user = MagicMock(id=7)

        with (
            patch(f"{_HC}.resolve_callback_data", return_value=None),
            patch(f"{_HC}.handle_history_callback", new_callable=AsyncMock) as handler,
        ):
            await _dispatch(update, MagicMock())

        handler.assert_not_awaited()
        update.callback_query.answer.assert_awaited_once_with(
            "This button has expired", show_alert=True
        )

    async def test_resolved_token_reaches_the_handler(self) -> None:
        update = MagicMock()
        update.callback_query = AsyncMock()
        update.callback_query.data = "token"
        update.effective_user = MagicMock(id=7)
        resolved = f"{CB_HISTORY_NEXT}1:@7:0:0"

        with (
            patch(f"{_HC}.resolve_callback_data", return_value=resolved),
            patch(f"{_HC}.handle_history_callback", new_callable=AsyncMock) as handler,
        ):
            await _dispatch(update, MagicMock())

        handler.assert_awaited_once()
        assert handler.await_args_list[-1].args[2] == resolved
