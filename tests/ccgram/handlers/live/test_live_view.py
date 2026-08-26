import time
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot
from telegram.error import RetryAfter, TelegramError

from ccgram.handlers.callback_data import (
    CB_KEYS_PREFIX,
    CB_LIVE_START,
    CB_LIVE_STOP,
    CB_SCREENSHOT_REFRESH,
)
from ccgram.handlers.live.live_view import (
    LiveViewState,
    _active_views,
    _edit_caption,
    build_live_keyboard,
    content_hash,
    get_live_view,
    is_live,
    start_live_view,
    stop_live_view,
    tick_live_views,
)
from ccgram.handlers.live.screenshot_callbacks import (
    _handle_live_start,
    _handle_live_stop,
    build_screenshot_keyboard,
)
from ccgram.handlers.toolbar import build_toolbar_keyboard


@pytest.fixture(autouse=True)
def _clear_views():
    _active_views.clear()
    yield
    _active_views.clear()


def _make_view(
    chat_id: int = 100,
    message_id: int = 200,
    thread_id: int = 42,
    user_id: int = 1,
    window_id: str = "@0",
    pane_id: str | None = None,
    last_hash: str = "",
) -> LiveViewState:
    return LiveViewState(
        chat_id=chat_id,
        message_id=message_id,
        thread_id=thread_id,
        user_id=user_id,
        window_id=window_id,
        pane_id=pane_id,
        last_hash=last_hash,
    )


# ── State lifecycle ──────────────────────────────────────────────────────


class TestLifecycle:
    def test_start_and_get(self):
        view = _make_view()
        start_live_view(view)
        assert get_live_view(1, 42) is view

    def test_stop_returns_view(self):
        view = _make_view()
        start_live_view(view)
        result = stop_live_view(1, 42)
        assert result is view
        assert get_live_view(1, 42) is None

    def test_stop_returns_none_when_not_active(self):
        assert stop_live_view(1, 42) is None

    def test_is_live_true(self):
        start_live_view(_make_view())
        assert is_live(1, 42) is True

    def test_is_live_false(self):
        assert is_live(1, 42) is False

    def test_start_replaces_existing(self):
        v1 = _make_view(message_id=100)
        v2 = _make_view(message_id=200)
        start_live_view(v1)
        start_live_view(v2)
        assert get_live_view(1, 42) is v2

    def test_multiple_topics(self):
        v1 = _make_view(user_id=1, thread_id=10)
        v2 = _make_view(user_id=1, thread_id=20)
        start_live_view(v1)
        start_live_view(v2)
        assert is_live(1, 10)
        assert is_live(1, 20)
        stop_live_view(1, 10)
        assert not is_live(1, 10)
        assert is_live(1, 20)


# ── Content hash ─────────────────────────────────────────────────────────


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_input(self):
        assert content_hash("hello") != content_hash("world")

    def test_empty_string(self):
        h = content_hash("")
        assert isinstance(h, str)
        assert len(h) == 32


# ── Cleanup registry ─────────────────────────────────────────────────────


class TestCleanup:
    def test_topic_cleanup_removes_view(self):
        from ccgram.handlers.live.live_view import _clear_live_view

        start_live_view(_make_view())
        assert is_live(1, 42)
        _clear_live_view(1, 42)
        assert not is_live(1, 42)

    def test_topic_cleanup_noop_when_not_active(self):
        from ccgram.handlers.live.live_view import _clear_live_view

        _clear_live_view(1, 42)


# ── Keyboard builders ────────────────────────────────────────────────────


def _buttons(kb) -> list:
    return [btn for row in kb.inline_keyboard for btn in row]


def _labels(kb) -> list[str]:
    return [btn.text for btn in _buttons(kb)]


def _button_labelled(kb, needle: str):
    return next(btn for btn in _buttons(kb) if needle in btn.text)


class TestBuildLiveKeyboard:
    def test_has_stop_button(self):
        assert any("Stop" in label for label in _labels(build_live_keyboard("@0")))

    def test_no_refresh_or_live_button(self):
        labels = _labels(build_live_keyboard("@0"))
        assert not any("Refresh" in label for label in labels)
        assert not any(label == "\U0001f4fa Live" for label in labels)

    @pytest.mark.parametrize("key", ["Esc", "^C", "Enter"])
    def test_has_quick_key(self, key: str):
        assert any(key in label for label in _labels(build_live_keyboard("@0")))

    def test_stop_callback_data_format(self):
        stop_btn = _button_labelled(build_live_keyboard("@0"), "Stop")
        assert isinstance(stop_btn.callback_data, str)
        assert stop_btn.callback_data.startswith(CB_LIVE_STOP)

    def test_pane_id_in_callback_data(self):
        stop_btn = _button_labelled(build_live_keyboard("@0", pane_id="%3"), "Stop")
        assert isinstance(stop_btn.callback_data, str)
        assert "@0|%3" in stop_btn.callback_data

    def test_key_callbacks_use_keys_prefix(self):
        key_btns = [
            btn for btn in _buttons(build_live_keyboard("@0")) if "Stop" not in btn.text
        ]
        for btn in key_btns:
            assert isinstance(btn.callback_data, str)
            assert btn.callback_data.startswith(CB_KEYS_PREFIX)


class TestBuildScreenshotKeyboard:
    @pytest.mark.parametrize(
        ("label", "prefix"),
        [("Live", CB_LIVE_START), ("Refresh", CB_SCREENSHOT_REFRESH)],
    )
    def test_button_present_with_callback_prefix(self, label: str, prefix: str):
        btn = _button_labelled(build_screenshot_keyboard("@0"), label)
        assert isinstance(btn.callback_data, str)
        assert btn.callback_data.startswith(prefix)

    def test_pane_id_propagated(self):
        live_btn = _button_labelled(
            build_screenshot_keyboard("@0", pane_id="%5"), "Live"
        )
        assert isinstance(live_btn.callback_data, str)
        assert "@0|%5" in live_btn.callback_data


class TestBuildToolbarKeyboard:
    def test_live_replaces_esc_in_row1(self):
        with patch(
            "ccgram.handlers.polling.polling_state.terminal_screen_buffer.is_rc_active",
            return_value=False,
        ):
            kb = build_toolbar_keyboard("@0")
        assert any("Live" in btn.text for btn in kb.inline_keyboard[0])

    def test_live_callback_data(self):
        # After the toolbar refactor, all buttons use the single CB_TOOLBAR
        # prefix; the suffix is "<window_id>:<action_name>". The live action
        # dispatches internally to the screenshot handler with CB_LIVE_START.
        from ccgram.handlers.callback_data import CB_TOOLBAR

        with patch(
            "ccgram.handlers.polling.polling_state.terminal_screen_buffer.is_rc_active",
            return_value=False,
        ):
            kb = build_toolbar_keyboard("@0")
        live_btn = _button_labelled(kb, "Live")
        assert isinstance(live_btn.callback_data, str)
        assert live_btn.callback_data == f"{CB_TOOLBAR}@0:live"


# ── Tick function ────────────────────────────────────────────────────────


@contextmanager
def _tick_env(*, capture: str | None = "new text", window_alive: bool = True):
    """Patch the multiplexer + renderer used by ``tick_live_views``."""
    with (
        patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux,
        patch(
            "ccgram.handlers.live.live_view.text_to_image",
            new_callable=AsyncMock,
            return_value=b"PNG",
        ) as mock_img,
    ):
        mock_tmux.find_window_by_id = AsyncMock(
            return_value=MagicMock(window_id="@0") if window_alive else None
        )
        mock_tmux.capture_pane = AsyncMock(return_value=capture)
        mock_tmux.capture_pane_by_id = AsyncMock(return_value=capture)
        yield mock_tmux, mock_img


class TestTickLiveViews:
    @pytest.fixture(autouse=True)
    def _patch_rate_limit(self):
        with patch(
            "ccgram.handlers.live.live_view.rate_limit_send", new_callable=AsyncMock
        ):
            yield

    async def test_skip_when_hash_unchanged(self):
        view = _make_view(last_hash=content_hash("same text"))
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        with _tick_env(capture="same text"):
            await tick_live_views(bot)
        bot.edit_message_media.assert_not_awaited()

    async def test_edit_when_hash_changed(self):
        view = _make_view(last_hash="old_hash")
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        with _tick_env(capture="new text"):
            await tick_live_views(bot)
        bot.edit_message_media.assert_awaited_once()
        assert view.last_hash == content_hash("new text")

    async def test_auto_stop_on_timeout(self):
        view = _make_view()
        view.start_time = time.monotonic() - 999
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        await tick_live_views(bot)
        assert not is_live(1, 42)
        bot.edit_message_caption.assert_awaited_once()
        assert "timeout" in bot.edit_message_caption.call_args.kwargs["caption"]

    async def test_auto_stop_on_dead_window(self):
        start_live_view(_make_view())
        bot = AsyncMock(spec=Bot)
        with _tick_env(window_alive=False):
            await tick_live_views(bot)
        assert not is_live(1, 42)
        bot.edit_message_caption.assert_awaited_once()
        assert "window closed" in bot.edit_message_caption.call_args.kwargs["caption"]

    async def test_telegram_error_stops_view(self):
        start_live_view(_make_view(last_hash="old"))
        bot = AsyncMock(spec=Bot)
        bot.edit_message_media = AsyncMock(side_effect=TelegramError("gone"))
        with _tick_env():
            await tick_live_views(bot)
        assert not is_live(1, 42)

    async def test_skip_when_capture_returns_none(self):
        start_live_view(_make_view(last_hash="old"))
        bot = AsyncMock(spec=Bot)
        with _tick_env(capture=None) as (_tmux, mock_img):
            await tick_live_views(bot)
        mock_img.assert_not_awaited()
        assert is_live(1, 42)

    async def test_pane_id_uses_capture_pane_by_id(self):
        start_live_view(_make_view(pane_id="%3", last_hash="old"))
        bot = AsyncMock(spec=Bot)
        with _tick_env(capture="pane text") as (mock_tmux, _img):
            await tick_live_views(bot)
        mock_tmux.capture_pane_by_id.assert_awaited_once_with(
            "%3", with_ansi=True, window_id="@0"
        )
        bot.edit_message_media.assert_awaited_once()

    async def test_multiple_views_ticked(self):
        start_live_view(_make_view(user_id=1, thread_id=10, last_hash="old1"))
        start_live_view(
            _make_view(user_id=1, thread_id=20, last_hash="old2", message_id=300)
        )
        bot = AsyncMock(spec=Bot)
        with _tick_env(capture="changed"):
            await tick_live_views(bot)
        assert bot.edit_message_media.await_count == 2

    async def test_noop_when_no_active_views(self):
        bot = AsyncMock(spec=Bot)
        await tick_live_views(bot)
        bot.edit_message_media.assert_not_awaited()

    @pytest.mark.parametrize(
        "retry_after",
        [30, timedelta(seconds=30)],
        ids=["seconds", "timedelta"],
    )
    async def test_retry_after_pauses_view(self, retry_after):
        view = _make_view(last_hash="old")
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        bot.edit_message_media = AsyncMock(side_effect=RetryAfter(retry_after))
        with _tick_env():
            await tick_live_views(bot)
        assert is_live(1, 42)
        assert view.next_edit_after > time.monotonic()
        assert view.last_hash == "old"

    async def test_backoff_skips_tick(self):
        view = _make_view(last_hash="old")
        view.next_edit_after = time.monotonic() + 999
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        with _tick_env():
            await tick_live_views(bot)
        bot.edit_message_media.assert_not_awaited()
        assert is_live(1, 42)


# ── _edit_caption ────────────────────────────────────────────────────────


class TestEditCaption:
    async def test_edits_caption_with_keyboard(self):
        view = _make_view()
        bot = AsyncMock(spec=Bot)
        await _edit_caption(bot, view, "Done")
        bot.edit_message_caption.assert_awaited_once()
        kwargs = bot.edit_message_caption.call_args.kwargs
        assert kwargs["caption"] == "Done"
        assert kwargs["reply_markup"] is not None

    async def test_suppresses_telegram_error(self):
        view = _make_view()
        bot = AsyncMock(spec=Bot)
        bot.edit_message_caption = AsyncMock(side_effect=TelegramError("gone"))
        await _edit_caption(bot, view, "Done")


# ── Callback handlers ───────────────────────────────────────────────────


def _make_query(
    message_id: int = 200,
) -> tuple[AsyncMock, MagicMock]:
    query = AsyncMock()
    message = MagicMock()
    message.message_id = message_id
    query.message = message
    query.get_bot.return_value = AsyncMock()
    update = MagicMock()
    return query, update


_SC = "ccgram.handlers.live.screenshot_callbacks"


@contextmanager
def _callback_env(
    *,
    owns: bool = True,
    thread_id: int | None = 42,
    window_alive: bool = True,
    capture: str | None = "terminal text",
):
    """Patch the screenshot-callback collaborators; yield the tmux mock."""
    with (
        patch(f"{_SC}.user_owns_window", return_value=owns),
        patch(f"{_SC}.get_thread_id", return_value=thread_id),
        patch(f"{_SC}.tmux_manager") as mock_tmux,
        patch(f"{_SC}.text_to_image", new_callable=AsyncMock, return_value=b"PNG"),
        patch(f"{_SC}.thread_router") as mock_router,
    ):
        mock_tmux.find_window_by_id = AsyncMock(
            return_value=MagicMock(window_id="@0") if window_alive else None
        )
        mock_tmux.capture_pane = AsyncMock(return_value=capture)
        mock_tmux.capture_pane_by_id = AsyncMock(return_value=capture)
        mock_router.resolve_chat_id.return_value = 100
        yield mock_tmux


def _answer_text(query: AsyncMock) -> str:
    call = query.answer.call_args
    return call.kwargs.get("text", call.args[0] if call.args else "")


class TestHandleLiveStart:
    async def test_rejects_non_owner(self):
        query, update = _make_query()
        with _callback_env(owns=False):
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        query.answer.assert_awaited_once()
        assert "Not your session" in _answer_text(query)

    async def test_rejects_already_live(self):
        start_live_view(_make_view(user_id=1, thread_id=42))
        query, update = _make_query()
        with _callback_env():
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        query.answer.assert_awaited_once()
        assert "already" in _answer_text(query).lower()

    async def test_rejects_no_thread(self):
        query, update = _make_query()
        with _callback_env(thread_id=None):
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        assert "topic" in _answer_text(query).lower()

    async def test_rejects_dead_window(self):
        query, update = _make_query()
        with _callback_env(window_alive=False):
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        assert "not found" in _answer_text(query).lower()

    async def test_rejects_empty_capture(self):
        query, update = _make_query()
        with _callback_env(capture=None):
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        assert "capture" in _answer_text(query).lower()
        assert not is_live(1, 42)

    async def test_success_starts_live_view(self):
        query, update = _make_query()
        with _callback_env():
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        query.edit_message_media.assert_awaited_once()
        view = get_live_view(1, 42)
        assert view is not None
        assert view.window_id == "@0"
        assert view.last_hash == content_hash("terminal text")

    async def test_success_with_pane_id(self):
        query, update = _make_query()
        with _callback_env(capture="pane text") as mock_tmux:
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0|%3", update)
        mock_tmux.capture_pane_by_id.assert_awaited_once_with(
            "%3", with_ansi=True, window_id="@0"
        )
        query.edit_message_media.assert_awaited_once()
        view = get_live_view(1, 42)
        assert view is not None
        assert view.pane_id == "%3"


class TestHandleLiveStop:
    async def test_rejects_non_owner(self):
        query, update = _make_query()
        with _callback_env(owns=False):
            await _handle_live_stop(query, 1, f"{CB_LIVE_STOP}@0", update)
        query.answer.assert_awaited_once()
        assert "Not your session" in _answer_text(query)

    async def test_rejects_no_thread(self):
        query, update = _make_query()
        with _callback_env(thread_id=None):
            await _handle_live_stop(query, 1, f"{CB_LIVE_STOP}@0", update)
        assert "topic" in _answer_text(query).lower()

    async def test_stop_when_not_active(self):
        assert not is_live(1, 42)
        query, update = _make_query()
        with _callback_env():
            await _handle_live_stop(query, 1, f"{CB_LIVE_STOP}@0", update)
        assert "Stopped" in _answer_text(query)

    async def test_success_stops_live_view(self):
        start_live_view(_make_view())
        query, update = _make_query()
        with _callback_env():
            await _handle_live_stop(query, 1, f"{CB_LIVE_STOP}@0", update)
        assert not is_live(1, 42)
        query.edit_message_caption.assert_awaited_once()
        assert "Screenshot" in query.edit_message_caption.call_args.kwargs["caption"]
        assert "Stopped" in _answer_text(query)


# ── _handle_keys live view guard ────────────────────────────────────────


class TestHandleKeysLiveGuard:
    async def test_skips_refresh_when_live_view_active(self):
        from ccgram.handlers.status.status_bar_actions import _handle_keys

        start_live_view(_make_view(user_id=1, thread_id=42))
        query = AsyncMock()
        query.message = MagicMock(message_id=200, message_thread_id=42)
        update = MagicMock()
        update.callback_query = query
        update.message = None

        with (
            patch(
                "ccgram.handlers.status.status_bar_actions.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.status.status_bar_actions.get_thread_id",
                return_value=42,
            ),
            patch(
                "ccgram.handlers.status.status_bar_actions.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.status.status_bar_actions.text_to_image",
                new_callable=AsyncMock,
            ) as mock_img,
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.send_keys = AsyncMock()
            await _handle_keys(query, 1, f"{CB_KEYS_PREFIX}ent:@0", update)
        mock_img.assert_not_awaited()

    async def test_refreshes_when_no_live_view(self):
        from ccgram.handlers.status.status_bar_actions import _handle_keys

        assert not is_live(1, 42)
        query = AsyncMock()
        query.message = MagicMock(message_id=200, message_thread_id=42)
        update = MagicMock()
        update.callback_query = query
        update.message = None

        with (
            patch(
                "ccgram.handlers.status.status_bar_actions.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.status.status_bar_actions.get_thread_id",
                return_value=42,
            ),
            patch(
                "ccgram.handlers.status.status_bar_actions.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.status.status_bar_actions.text_to_image",
                new_callable=AsyncMock,
                return_value=b"PNG",
            ) as mock_img,
            patch("ccgram.handlers.status.status_bar_actions._KEY_REFRESH_DELAY", 0),
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane = AsyncMock(return_value="terminal text")
            mock_tmux.send_keys = AsyncMock()
            await _handle_keys(query, 1, f"{CB_KEYS_PREFIX}ent:@0", update)
            import asyncio

            await asyncio.sleep(0.05)
        mock_img.assert_awaited_once()
