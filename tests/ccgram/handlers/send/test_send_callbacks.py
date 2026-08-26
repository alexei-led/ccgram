"""Tests for src/ccgram/handlers/send/send_callbacks.py."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import CallbackQuery, InlineKeyboardMarkup, Message, Update, User
from telegram.error import TelegramError

from ccgram.handlers.callback_data import (
    CB_SEND_CANCEL,
    CB_SEND_DIR,
    CB_SEND_FILE,
    CB_SEND_PAGE,
    CB_SEND_UP,
)
from ccgram.handlers.reactions import REACT_DONE
from ccgram.handlers.send.send_callbacks import _dispatch
from ccgram.handlers.user_state import (
    SEND_CWD_KEY,
    SEND_ITEMS_KEY,
    SEND_PAGE_KEY,
    SEND_PATH_KEY,
    SEND_WINDOW_ID_KEY,
)

_MOD = "ccgram.handlers.send.send_callbacks"
_THREAD_ID = 456
_CHAT_ID = 123

_BROWSER_RESULT = (
    "Browse /tmp/test",
    MagicMock(spec=InlineKeyboardMarkup),
    [Path("/tmp/test/a.txt")],
)


@pytest.fixture(autouse=True)
def _allow_all_users() -> Iterator[None]:
    with patch("ccgram.config.config.is_user_allowed", return_value=True):
        yield


@pytest.fixture(autouse=True)
def bound_topic() -> Iterator[MagicMock]:
    """Topic 456 is bound to window @0 — the common precondition."""
    with (
        patch(f"{_MOD}.thread_router") as router,
        patch(f"{_MOD}.get_thread_id", return_value=_THREAD_ID),
    ):
        router.resolve_window_for_thread.return_value = "@0"
        router.resolve_chat_id.return_value = _CHAT_ID
        yield router


def _make_query(data: str, user_id: int = 789) -> AsyncMock:
    msg = AsyncMock(spec=Message)
    msg.chat_id = _CHAT_ID
    msg.message_thread_id = _THREAD_ID

    query = AsyncMock(spec=CallbackQuery)
    query.data = data
    query.message = msg
    query.from_user = MagicMock(spec=User)
    query.from_user.id = user_id
    return query


def _make_update(query: AsyncMock, user_id: int = 789) -> MagicMock:
    user = MagicMock(spec=User)
    user.id = user_id

    update = MagicMock(spec=Update)
    update.callback_query = query
    update.effective_user = user
    update.effective_message = MagicMock()
    update.effective_message.message_thread_id = _THREAD_ID
    return update


def _make_context(tmp_path: Path, window_id: str = "@0") -> MagicMock:
    ctx = MagicMock()
    ctx.bot = AsyncMock()
    ctx.user_data = {
        SEND_ITEMS_KEY: [tmp_path / "file.txt", tmp_path / "subdir"],
        SEND_PATH_KEY: str(tmp_path),
        SEND_CWD_KEY: str(tmp_path),
        SEND_WINDOW_ID_KEY: window_id,
        SEND_PAGE_KEY: 0,
    }
    return ctx


class TestStaleGuard:
    async def test_outside_topic_answers_error(self, tmp_path: Path) -> None:
        query = _make_query(CB_SEND_CANCEL)
        update = _make_update(query)
        update.effective_message.message_thread_id = None

        with patch(f"{_MOD}.get_thread_id", return_value=None):
            await _dispatch(update, _make_context(tmp_path))

        query.answer.assert_awaited_once_with("Not in a topic", show_alert=True)

    async def test_window_rebound_clears_state_and_alerts(
        self, tmp_path: Path, bound_topic: MagicMock
    ) -> None:
        bound_topic.resolve_window_for_thread.return_value = "@99"
        query = _make_query(CB_SEND_CANCEL)
        ctx = _make_context(tmp_path, window_id="@0")

        await _dispatch(_make_update(query), ctx)

        query.answer.assert_awaited_once_with(
            "Browser expired — use /send to restart", show_alert=True
        )
        assert SEND_WINDOW_ID_KEY not in ctx.user_data


class TestHandleFile:
    @staticmethod
    def _context_with_file(tmp_path: Path, name: str = "report.txt") -> MagicMock:
        path = tmp_path / name
        path.write_text("data", encoding="utf-8")
        ctx = _make_context(tmp_path)
        ctx.user_data[SEND_ITEMS_KEY] = [path]
        return ctx

    async def test_valid_file_uploads_and_clears_state(self, tmp_path: Path) -> None:
        query = _make_query(f"{CB_SEND_FILE}0")
        ctx = self._context_with_file(tmp_path)

        with (
            patch(f"{_MOD}.validate_sendable", return_value=None),
            patch(f"{_MOD}.upload_file", new_callable=AsyncMock) as mock_upload,
        ):
            await _dispatch(_make_update(query), ctx)

        mock_upload.assert_awaited_once()
        # Toast replaced with a persistent reaction on the uploaded file.
        query.answer.assert_awaited_once_with()
        query.message.delete.assert_awaited_once()
        assert SEND_WINDOW_ID_KEY not in ctx.user_data

    async def test_upload_success_reacts_on_uploaded_message(
        self, tmp_path: Path
    ) -> None:
        query = _make_query(f"{CB_SEND_FILE}0")
        ctx = self._context_with_file(tmp_path)
        sent_msg = MagicMock(chat_id=_CHAT_ID, message_id=8800)

        with (
            patch(f"{_MOD}.validate_sendable", return_value=None),
            patch(f"{_MOD}.upload_file", new_callable=AsyncMock, return_value=sent_msg),
            patch(f"{_MOD}.react", new_callable=AsyncMock) as mock_react,
        ):
            await _dispatch(_make_update(query), ctx)

        assert mock_react.call_args.args[1:4] == (_CHAT_ID, 8800, REACT_DONE)

    async def test_upload_returning_no_message_skips_reaction(
        self, tmp_path: Path
    ) -> None:
        query = _make_query(f"{CB_SEND_FILE}0")
        ctx = self._context_with_file(tmp_path)

        with (
            patch(f"{_MOD}.validate_sendable", return_value=None),
            patch(f"{_MOD}.upload_file", new_callable=AsyncMock, return_value=None),
            patch(f"{_MOD}.react", new_callable=AsyncMock) as mock_react,
        ):
            await _dispatch(_make_update(query), ctx)

        mock_react.assert_not_awaited()

    async def test_denied_file_shows_error(self, tmp_path: Path) -> None:
        query = _make_query(f"{CB_SEND_FILE}0")
        ctx = self._context_with_file(tmp_path, "secret.txt")

        with (
            patch(f"{_MOD}.validate_sendable", return_value="access denied"),
            patch(f"{_MOD}.upload_file", new_callable=AsyncMock) as mock_upload,
        ):
            await _dispatch(_make_update(query), ctx)

        mock_upload.assert_not_awaited()
        query.answer.assert_awaited_once_with(
            "Cannot send: access denied", show_alert=True
        )

    @pytest.mark.parametrize(
        ("payload", "expected_answer"),
        [
            pytest.param("99", ("Item not found",), id="index-out-of-range"),
            pytest.param("notanint", ("Invalid selection",), id="non-numeric-index"),
        ],
    )
    async def test_bad_index_rejected(
        self, tmp_path: Path, payload: str, expected_answer: tuple[str]
    ) -> None:
        query = _make_query(f"{CB_SEND_FILE}{payload}")
        ctx = _make_context(tmp_path)
        ctx.user_data[SEND_ITEMS_KEY] = [tmp_path / "file.txt"]

        await _dispatch(_make_update(query), ctx)

        kwargs = {"show_alert": True} if expected_answer[0] == "Item not found" else {}
        query.answer.assert_awaited_once_with(*expected_answer, **kwargs)


class TestHandleDir:
    async def test_valid_dir_builds_browser_and_edits_message(
        self, tmp_path: Path
    ) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        query = _make_query(f"{CB_SEND_DIR}0")
        ctx = _make_context(tmp_path)
        ctx.user_data[SEND_ITEMS_KEY] = [subdir]

        with (
            patch(f"{_MOD}.is_path_contained", return_value=True),
            patch(
                f"{_MOD}.build_file_browser", return_value=_BROWSER_RESULT
            ) as mock_browser,
        ):
            await _dispatch(_make_update(query), ctx)

        mock_browser.assert_called_once_with(subdir, Path(str(tmp_path)), 0)
        query.message.edit_text.assert_awaited_once()
        assert ctx.user_data[SEND_PATH_KEY] == str(subdir)
        assert ctx.user_data[SEND_PAGE_KEY] == 0

    async def test_dir_outside_cwd_shows_error(self, tmp_path: Path) -> None:
        query = _make_query(f"{CB_SEND_DIR}0")
        ctx = _make_context(tmp_path)
        ctx.user_data[SEND_ITEMS_KEY] = [Path("/tmp/evil")]

        with patch(f"{_MOD}.is_path_contained", return_value=False):
            await _dispatch(_make_update(query), ctx)

        query.answer.assert_awaited_once_with(
            "Directory is outside project root", show_alert=True
        )

    async def test_out_of_bounds_dir_index_shows_error(self, tmp_path: Path) -> None:
        query = _make_query(f"{CB_SEND_DIR}5")
        ctx = _make_context(tmp_path)
        ctx.user_data[SEND_ITEMS_KEY] = []

        await _dispatch(_make_update(query), ctx)

        query.answer.assert_awaited_once_with("Item not found", show_alert=True)


class TestHandlePage:
    async def test_valid_page_rebuilds_browser(self, tmp_path: Path) -> None:
        query = _make_query(f"{CB_SEND_PAGE}2")
        ctx = _make_context(tmp_path)

        with patch(
            f"{_MOD}.build_file_browser", return_value=_BROWSER_RESULT
        ) as mock_browser:
            await _dispatch(_make_update(query), ctx)

        mock_browser.assert_called_once_with(
            Path(str(tmp_path)), Path(str(tmp_path)), 2
        )
        assert ctx.user_data[SEND_PAGE_KEY] == 2
        query.message.edit_text.assert_awaited_once()


class TestHandleUp:
    async def test_at_cwd_answers_already_at_root(self, tmp_path: Path) -> None:
        query = _make_query(CB_SEND_UP)

        await _dispatch(_make_update(query), _make_context(tmp_path))

        query.answer.assert_awaited_once_with("Already at project root")

    async def test_below_cwd_navigates_to_parent(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        query = _make_query(CB_SEND_UP)
        ctx = _make_context(tmp_path)
        ctx.user_data[SEND_PATH_KEY] = str(subdir)

        with (
            patch(f"{_MOD}.is_path_contained", return_value=True),
            patch(
                f"{_MOD}.build_file_browser", return_value=_BROWSER_RESULT
            ) as mock_browser,
        ):
            await _dispatch(_make_update(query), ctx)

        mock_browser.assert_called_once_with(tmp_path, Path(str(tmp_path)), 0)
        assert ctx.user_data[SEND_PATH_KEY] == str(tmp_path)


class TestBrowserStateLost:
    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(f"{CB_SEND_PAGE}0", id="page"),
            pytest.param(CB_SEND_UP, id="up"),
        ],
    )
    async def test_missing_cached_path_shows_error(
        self, tmp_path: Path, data: str
    ) -> None:
        query = _make_query(data)
        ctx = _make_context(tmp_path)
        ctx.user_data[SEND_PATH_KEY] = ""

        await _dispatch(_make_update(query), ctx)

        query.answer.assert_awaited_once_with("Browser state lost", show_alert=True)


class TestHandleCancel:
    async def test_cancel_clears_state_and_deletes_message(
        self, tmp_path: Path
    ) -> None:
        query = _make_query(CB_SEND_CANCEL)
        ctx = _make_context(tmp_path)

        await _dispatch(_make_update(query), ctx)

        query.answer.assert_awaited_once_with("Cancelled")
        query.message.delete.assert_awaited_once()
        assert SEND_WINDOW_ID_KEY not in ctx.user_data
        assert SEND_ITEMS_KEY not in ctx.user_data

    async def test_cancel_edits_message_when_delete_fails(self, tmp_path: Path) -> None:
        query = _make_query(CB_SEND_CANCEL)
        query.message.delete.side_effect = TelegramError("gone")

        await _dispatch(_make_update(query), _make_context(tmp_path))

        query.message.edit_text.assert_awaited_once_with("Cancelled")
