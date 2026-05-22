import time
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


class TestBuildLiveKeyboard:
    def test_has_stop_button(self):
        kb = build_live_keyboard("@0")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        labels = [btn.text for btn in flat]
        assert any("Stop" in label for label in labels)

    def test_no_refresh_or_live_button(self):
        kb = build_live_keyboard("@0")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        labels = [btn.text for btn in flat]
        assert not any("Refresh" in label for label in labels)
        assert not any(label == "\U0001f4fa Live" for label in labels)

    def test_has_quick_keys(self):
        kb = build_live_keyboard("@0")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        labels = [btn.text for btn in flat]
        assert any("Esc" in label for label in labels)
        assert any("^C" in label for label in labels)
        assert any("Enter" in label for label in labels)

    def test_stop_callback_data_format(self):
        kb = build_live_keyboard("@0")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        stop_btn = [btn for btn in flat if "Stop" in btn.text][0]
        assert isinstance(stop_btn.callback_data, str)
        assert stop_btn.callback_data.startswith(CB_LIVE_STOP)

    def test_pane_id_in_callback_data(self):
        kb = build_live_keyboard("@0", pane_id="%3")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        stop_btn = [btn for btn in flat if "Stop" in btn.text][0]
        assert isinstance(stop_btn.callback_data, str)
        assert "@0:%3" in stop_btn.callback_data

    def test_key_callbacks_use_keys_prefix(self):
        kb = build_live_keyboard("@0")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        key_btns = [btn for btn in flat if "Stop" not in btn.text]
        for btn in key_btns:
            assert isinstance(btn.callback_data, str)
            assert btn.callback_data.startswith(CB_KEYS_PREFIX)


class TestBuildScreenshotKeyboard:
    def test_has_live_button(self):
        kb = build_screenshot_keyboard("@0")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        labels = [btn.text for btn in flat]
        assert any("Live" in label for label in labels)

    def test_has_refresh_button(self):
        kb = build_screenshot_keyboard("@0")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        labels = [btn.text for btn in flat]
        assert any("Refresh" in label for label in labels)

    def test_live_callback_data_format(self):
        kb = build_screenshot_keyboard("@0")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        live_btn = [btn for btn in flat if "Live" in btn.text][0]
        assert isinstance(live_btn.callback_data, str)
        assert live_btn.callback_data.startswith(CB_LIVE_START)

    def test_refresh_callback_data_format(self):
        kb = build_screenshot_keyboard("@0")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        refresh_btn = [btn for btn in flat if "Refresh" in btn.text][0]
        assert isinstance(refresh_btn.callback_data, str)
        assert refresh_btn.callback_data.startswith(CB_SCREENSHOT_REFRESH)

    def test_pane_id_propagated(self):
        kb = build_screenshot_keyboard("@0", pane_id="%5")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        live_btn = [btn for btn in flat if "Live" in btn.text][0]
        assert isinstance(live_btn.callback_data, str)
        assert "@0:%5" in live_btn.callback_data


class TestBuildToolbarKeyboard:
    def test_has_live_button(self):
        with patch(
            "ccgram.handlers.polling.polling_state.terminal_screen_buffer.is_rc_active",
            return_value=False,
        ):
            kb = build_toolbar_keyboard("@0")
        flat = [btn for row in kb.inline_keyboard for btn in row]
        labels = [btn.text for btn in flat]
        assert any("Live" in label for label in labels)

    def test_live_replaces_esc_in_row1(self):
        with patch(
            "ccgram.handlers.polling.polling_state.terminal_screen_buffer.is_rc_active",
            return_value=False,
        ):
            kb = build_toolbar_keyboard("@0")
        row1_labels = [btn.text for btn in kb.inline_keyboard[0]]
        assert any("Live" in label for label in row1_labels)

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
        flat = [btn for row in kb.inline_keyboard for btn in row]
        live_btn = [btn for btn in flat if "Live" in btn.text][0]
        assert isinstance(live_btn.callback_data, str)
        assert live_btn.callback_data == f"{CB_TOOLBAR}@0:live"


# ── Tick function ────────────────────────────────────────────────────────


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
        with (
            patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux,
            patch(
                "ccgram.handlers.live.live_view.text_to_image",
                new_callable=AsyncMock,
            ),
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane = AsyncMock(return_value="same text")
            await tick_live_views(bot)
        bot.edit_message_media.assert_not_awaited()

    async def test_edit_when_hash_changed(self):
        view = _make_view(last_hash="old_hash")
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux,
            patch(
                "ccgram.handlers.live.live_view.text_to_image",
                new_callable=AsyncMock,
                return_value=b"PNG",
            ),
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane = AsyncMock(return_value="new text")
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

    async def test_auto_stop_on_dead_window(self):
        view = _make_view()
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux:
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            await tick_live_views(bot)
        assert not is_live(1, 42)

    async def test_telegram_error_stops_view(self):
        view = _make_view(last_hash="old")
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        bot.edit_message_media = AsyncMock(side_effect=TelegramError("gone"))
        with (
            patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux,
            patch(
                "ccgram.handlers.live.live_view.text_to_image",
                new_callable=AsyncMock,
                return_value=b"PNG",
            ),
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane = AsyncMock(return_value="new text")
            await tick_live_views(bot)
        assert not is_live(1, 42)

    async def test_skip_when_capture_returns_none(self):
        view = _make_view(last_hash="old")
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux,
            patch(
                "ccgram.handlers.live.live_view.text_to_image",
                new_callable=AsyncMock,
            ) as mock_img,
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane = AsyncMock(return_value=None)
            await tick_live_views(bot)
        mock_img.assert_not_awaited()
        assert is_live(1, 42)

    async def test_pane_id_uses_capture_pane_by_id(self):
        view = _make_view(pane_id="%3", last_hash="old")
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux,
            patch(
                "ccgram.handlers.live.live_view.text_to_image",
                new_callable=AsyncMock,
                return_value=b"PNG",
            ),
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane_by_id = AsyncMock(return_value="pane text")
            await tick_live_views(bot)
        mock_tmux.capture_pane_by_id.assert_awaited_once_with(
            "%3", with_ansi=True, window_id="@0"
        )
        bot.edit_message_media.assert_awaited_once()

    async def test_multiple_views_ticked(self):
        v1 = _make_view(user_id=1, thread_id=10, last_hash="old1")
        v2 = _make_view(user_id=1, thread_id=20, last_hash="old2", message_id=300)
        start_live_view(v1)
        start_live_view(v2)
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux,
            patch(
                "ccgram.handlers.live.live_view.text_to_image",
                new_callable=AsyncMock,
                return_value=b"PNG",
            ),
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane = AsyncMock(return_value="changed")
            await tick_live_views(bot)
        assert bot.edit_message_media.await_count == 2

    async def test_noop_when_no_active_views(self):
        bot = AsyncMock(spec=Bot)
        await tick_live_views(bot)
        bot.edit_message_media.assert_not_awaited()

    async def test_timeout_edits_caption(self):
        view = _make_view()
        view.start_time = time.monotonic() - 999
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        await tick_live_views(bot)
        bot.edit_message_caption.assert_awaited_once()
        call_kwargs = bot.edit_message_caption.call_args.kwargs
        assert "timeout" in call_kwargs["caption"]

    async def test_retry_after_pauses_view(self):
        view = _make_view(last_hash="old")
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        bot.edit_message_media = AsyncMock(side_effect=RetryAfter(30))
        with (
            patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux,
            patch(
                "ccgram.handlers.live.live_view.text_to_image",
                new_callable=AsyncMock,
                return_value=b"PNG",
            ),
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane = AsyncMock(return_value="new text")
            await tick_live_views(bot)
        assert is_live(1, 42)
        assert view.next_edit_after > time.monotonic()
        assert view.last_hash == "old"

    async def test_backoff_skips_tick(self):
        view = _make_view(last_hash="old")
        view.next_edit_after = time.monotonic() + 999
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux:
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            await tick_live_views(bot)
        bot.edit_message_media.assert_not_awaited()
        assert is_live(1, 42)

    async def test_dead_window_edits_caption(self):
        view = _make_view()
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux:
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            await tick_live_views(bot)
        bot.edit_message_caption.assert_awaited_once()
        call_kwargs = bot.edit_message_caption.call_args.kwargs
        assert "window closed" in call_kwargs["caption"]


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


class TestHandleLiveStart:
    async def test_rejects_non_owner(self):
        query, update = _make_query()
        with patch(
            "ccgram.handlers.live.screenshot_callbacks.user_owns_window",
            return_value=False,
        ):
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        query.answer.assert_awaited_once()
        assert "Not your session" in query.answer.call_args.kwargs.get(
            "text", query.answer.call_args.args[0]
        )

    async def test_rejects_already_live(self):
        view = _make_view(user_id=1, thread_id=42)
        start_live_view(view)
        query, update = _make_query()
        with (
            patch(
                "ccgram.handlers.live.screenshot_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.get_thread_id",
                return_value=42,
            ),
        ):
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        query.answer.assert_awaited_once()
        assert "already" in query.answer.call_args.args[0].lower()

    async def test_rejects_no_thread(self):
        query, update = _make_query()
        with (
            patch(
                "ccgram.handlers.live.screenshot_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.get_thread_id",
                return_value=None,
            ),
        ):
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        query.answer.assert_awaited()
        assert "topic" in query.answer.call_args.args[0].lower()

    async def test_rejects_dead_window(self):
        query, update = _make_query()
        with (
            patch(
                "ccgram.handlers.live.screenshot_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.get_thread_id",
                return_value=42,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.tmux_manager"
            ) as mock_tmux,
        ):
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        query.answer.assert_awaited()
        assert "not found" in query.answer.call_args.args[0].lower()

    async def test_rejects_empty_capture(self):
        query, update = _make_query()
        with (
            patch(
                "ccgram.handlers.live.screenshot_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.get_thread_id",
                return_value=42,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.tmux_manager"
            ) as mock_tmux,
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane = AsyncMock(return_value=None)
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        query.answer.assert_awaited()
        assert "capture" in query.answer.call_args.args[0].lower()
        assert not is_live(1, 42)

    async def test_success_starts_live_view(self):
        query, update = _make_query()
        with (
            patch(
                "ccgram.handlers.live.screenshot_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.get_thread_id",
                return_value=42,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.live.screenshot_callbacks.text_to_image",
                new_callable=AsyncMock,
                return_value=b"PNG",
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.thread_router"
            ) as mock_router,
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane = AsyncMock(return_value="terminal text")
            mock_router.resolve_chat_id.return_value = 100
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0", update)
        query.edit_message_media.assert_awaited_once()
        assert is_live(1, 42)
        view = get_live_view(1, 42)
        assert view is not None
        assert view.window_id == "@0"
        assert view.last_hash == content_hash("terminal text")

    async def test_success_with_pane_id(self):
        query, update = _make_query()
        with (
            patch(
                "ccgram.handlers.live.screenshot_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.get_thread_id",
                return_value=42,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.live.screenshot_callbacks.text_to_image",
                new_callable=AsyncMock,
                return_value=b"PNG",
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.thread_router"
            ) as mock_router,
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane_by_id = AsyncMock(return_value="pane text")
            mock_router.resolve_chat_id.return_value = 100
            await _handle_live_start(query, 1, f"{CB_LIVE_START}@0:%3", update)
        mock_tmux.capture_pane_by_id.assert_awaited_once_with(
            "%3", with_ansi=True, window_id="@0"
        )
        query.edit_message_media.assert_awaited_once()
        assert is_live(1, 42)
        view = get_live_view(1, 42)
        assert view is not None
        assert view.pane_id == "%3"


class TestHandleLiveStop:
    async def test_rejects_non_owner(self):
        query, update = _make_query()
        with patch(
            "ccgram.handlers.live.screenshot_callbacks.user_owns_window",
            return_value=False,
        ):
            await _handle_live_stop(query, 1, f"{CB_LIVE_STOP}@0", update)
        query.answer.assert_awaited_once()
        assert "Not your session" in query.answer.call_args.kwargs.get(
            "text", query.answer.call_args.args[0]
        )

    async def test_rejects_no_thread(self):
        query, update = _make_query()
        with (
            patch(
                "ccgram.handlers.live.screenshot_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.get_thread_id",
                return_value=None,
            ),
        ):
            await _handle_live_stop(query, 1, f"{CB_LIVE_STOP}@0", update)
        query.answer.assert_awaited()
        assert "topic" in query.answer.call_args.args[0].lower()

    async def test_stop_when_not_active(self):
        assert not is_live(1, 42)
        query, update = _make_query()
        with (
            patch(
                "ccgram.handlers.live.screenshot_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.get_thread_id",
                return_value=42,
            ),
        ):
            await _handle_live_stop(query, 1, f"{CB_LIVE_STOP}@0", update)
        query.answer.assert_awaited()
        assert "Stopped" in query.answer.call_args.args[0]

    async def test_success_stops_live_view(self):
        start_live_view(_make_view())
        assert is_live(1, 42)
        query, update = _make_query()
        with (
            patch(
                "ccgram.handlers.live.screenshot_callbacks.user_owns_window",
                return_value=True,
            ),
            patch(
                "ccgram.handlers.live.screenshot_callbacks.get_thread_id",
                return_value=42,
            ),
        ):
            await _handle_live_stop(query, 1, f"{CB_LIVE_STOP}@0", update)
        assert not is_live(1, 42)
        query.edit_message_caption.assert_awaited_once()
        assert "Screenshot" in query.edit_message_caption.call_args.kwargs["caption"]
        query.answer.assert_awaited()
        assert "Stopped" in query.answer.call_args.args[0]


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


# ── RetryAfter timedelta branch ─────────────────────────────────────────


class TestRetryAfterTimedelta:
    @pytest.fixture(autouse=True)
    def _patch_rate_limit(self):
        with patch(
            "ccgram.handlers.live.live_view.rate_limit_send", new_callable=AsyncMock
        ):
            yield

    async def test_retry_after_timedelta_pauses_view(self):
        view = _make_view(last_hash="old")
        start_live_view(view)
        bot = AsyncMock(spec=Bot)
        bot.edit_message_media = AsyncMock(
            side_effect=RetryAfter(timedelta(seconds=30))
        )
        with (
            patch("ccgram.handlers.live.live_view.tmux_manager") as mock_tmux,
            patch(
                "ccgram.handlers.live.live_view.text_to_image",
                new_callable=AsyncMock,
                return_value=b"PNG",
            ),
        ):
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id="@0")
            )
            mock_tmux.capture_pane = AsyncMock(return_value="new text")
            await tick_live_views(bot)
        assert is_live(1, 42)
        assert view.next_edit_after > time.monotonic()
