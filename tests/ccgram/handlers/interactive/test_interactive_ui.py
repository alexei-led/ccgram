"""Tests for interactive UI rendering."""

import pytest
from telegram import InlineKeyboardMarkup

from ccgram.handlers.callback_data import (
    CB_ASK_DOWN,
    CB_ASK_ENTER,
    CB_ASK_ESC,
    CB_ASK_LEFT,
    CB_ASK_REFRESH,
    CB_ASK_RIGHT,
    CB_ASK_SPACE,
    CB_ASK_TAB,
    CB_ASK_UP,
)
from ccgram.handlers.interactive.interactive_ui import (
    INTERACTIVE_INSTRUCTION_LINE,
    _build_interactive_keyboard,
    format_interactive_message,
)


def _cb_data(kb: InlineKeyboardMarkup, row: int | None = None) -> list[str]:
    rows = [kb.inline_keyboard[row]] if row is not None else kb.inline_keyboard
    return [str(btn.callback_data) for r in rows for btn in r if btn.callback_data]


class TestBuildInteractiveKeyboard:
    def test_default_layout_has_left_right(self) -> None:
        data = _cb_data(_build_interactive_keyboard("@0"), row=1)
        assert any(d.startswith(CB_ASK_LEFT) for d in data)
        assert any(d.startswith(CB_ASK_RIGHT) for d in data)

    def test_restore_checkpoint_omits_left_right(self) -> None:
        data = _cb_data(
            _build_interactive_keyboard("@0", ui_name="RestoreCheckpoint"), row=1
        )
        assert not any(d.startswith(CB_ASK_LEFT) for d in data)
        assert not any(d.startswith(CB_ASK_RIGHT) for d in data)

    def test_restore_checkpoint_has_down_only(self) -> None:
        data = _cb_data(
            _build_interactive_keyboard("@0", ui_name="RestoreCheckpoint"), row=1
        )
        assert len(data) == 1
        assert data[0].startswith(CB_ASK_DOWN)

    def test_all_direction_keys_present(self) -> None:
        kb = _build_interactive_keyboard("@0")
        assert len(kb.inline_keyboard) == 3
        data = _cb_data(kb)
        for prefix in (
            CB_ASK_UP,
            CB_ASK_DOWN,
            CB_ASK_LEFT,
            CB_ASK_RIGHT,
            CB_ASK_SPACE,
            CB_ASK_TAB,
        ):
            assert any(d.startswith(prefix) for d in data), f"Missing {prefix}"

    def test_action_keys_present(self) -> None:
        data = _cb_data(_build_interactive_keyboard("@0"), row=2)
        assert any(d.startswith(CB_ASK_ESC) for d in data)
        assert any(d.startswith(CB_ASK_ENTER) for d in data)
        assert any(d.startswith(CB_ASK_REFRESH) for d in data)

    def test_callback_data_contains_window_id(self) -> None:
        data = _cb_data(_build_interactive_keyboard("@12"))
        assert all("@12" in d for d in data)

    def test_pane_id_appended_to_target(self) -> None:
        data = _cb_data(_build_interactive_keyboard("@12", pane_id="%5"))
        assert all("@12|%5" in d for d in data)

    def test_callback_data_truncated_to_64_bytes(self) -> None:
        data = _cb_data(
            _build_interactive_keyboard("@" + "9" * 60, pane_id="%" + "1" * 60)
        )
        assert all(len(d) <= 64 for d in data)


class TestFormatInteractiveMessage:
    def test_prepends_instruction_line(self) -> None:
        out = format_interactive_message("Pick one:")
        assert out.startswith(INTERACTIVE_INSTRUCTION_LINE)
        assert "Pick one:" in out

    def test_instruction_describes_keys(self) -> None:
        for token in ("↑↓", "Enter", "Esc"):
            assert token in INTERACTIVE_INSTRUCTION_LINE

    def test_pane_prefix_with_pane_id(self) -> None:
        out = format_interactive_message("Body", pane_id="%5")
        assert "Pane (%5):" in out
        assert "Body" in out
        assert out.startswith(INTERACTIVE_INSTRUCTION_LINE)

    def test_no_pane_prefix_without_pane_id(self) -> None:
        out = format_interactive_message("Body")
        assert "Pane (" not in out

    def test_short_text_unchanged(self) -> None:
        out = format_interactive_message("hi")
        assert out == f"{INTERACTIVE_INSTRUCTION_LINE}\nhi"

    def test_oversized_text_truncated_within_4096(self) -> None:
        huge = "x" * 5000
        out = format_interactive_message(huge)
        assert len(out) <= 4096
        assert out.startswith(INTERACTIVE_INSTRUCTION_LINE)
        # Tail of the input must survive (most recent terminal lines)
        assert out.endswith("x")

    def test_oversized_with_pane_prefix_within_4096(self) -> None:
        huge = "y" * 5000
        out = format_interactive_message(huge, pane_id="%9")
        assert len(out) <= 4096
        assert "Pane (%9):" in out

    def test_pane_name_replaces_generic_label(self) -> None:
        out = format_interactive_message("Body", pane_id="%5", pane_name="api-gateway")
        assert "api-gateway (%5):" in out
        # Generic "Pane" word must NOT appear when a name is set.
        assert "Pane (%5):" not in out

    def test_blank_pane_name_falls_back_to_generic(self) -> None:
        out = format_interactive_message("Body", pane_id="%5", pane_name="   ")
        assert "Pane (%5):" in out

    def test_none_pane_name_falls_back_to_generic(self) -> None:
        out = format_interactive_message("Body", pane_id="%5", pane_name=None)
        assert "Pane (%5):" in out

    def test_pane_name_ignored_without_pane_id(self) -> None:
        out = format_interactive_message("Body", pane_name="api-gateway")
        assert "api-gateway" not in out
        assert "Pane (" not in out


class TestInteractiveModeTracking:
    @pytest.fixture(autouse=True)
    def _clear_interactive_mode(self) -> None:
        from ccgram.handlers.interactive.interactive_ui import _interactive_mode

        _interactive_mode.clear()

    def test_set_and_get(self) -> None:
        from ccgram.handlers.interactive.interactive_ui import (
            get_interactive_window,
            set_interactive_mode,
        )

        set_interactive_mode(100, "@0", thread_id=42)
        assert get_interactive_window(100, 42) == "@0"

    def test_clear(self) -> None:
        from ccgram.handlers.interactive.interactive_ui import (
            clear_interactive_mode,
            get_interactive_window,
            set_interactive_mode,
        )

        set_interactive_mode(100, "@0", thread_id=42)
        clear_interactive_mode(100, thread_id=42)
        assert get_interactive_window(100, 42) is None

    def test_none_thread_uses_zero(self) -> None:
        from ccgram.handlers.interactive.interactive_ui import (
            get_interactive_window,
            set_interactive_mode,
        )

        set_interactive_mode(100, "@0", thread_id=None)
        assert get_interactive_window(100, None) == "@0"


class TestDeadTopicCooldown:
    """Verify longer backoff when topic is deleted (thread not found)."""

    @pytest.fixture(autouse=True)
    def _clear_state(self) -> None:
        from ccgram.handlers.interactive.interactive_ui import (
            _interactive_mode,
            _interactive_msgs,
            _send_cooldowns,
        )

        _interactive_mode.clear()
        _interactive_msgs.clear()
        _send_cooldowns.clear()

    async def test_dead_topic_applies_longer_cooldown(self) -> None:
        from unittest.mock import AsyncMock, patch

        from telegram.error import BadRequest

        from ccgram.handlers.interactive.interactive_ui import (
            _DEAD_TOPIC_RETRY_INTERVAL,
            _send_cooldowns,
            handle_interactive_ui,
        )

        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = BadRequest("Message thread not found")

        with (
            patch(
                "ccgram.handlers.interactive.interactive_ui._capture_interactive_content",
                new_callable=AsyncMock,
                return_value=("AskUserQuestion", "Pick one:"),
            ),
            patch(
                "ccgram.handlers.interactive.interactive_ui.thread_router"
            ) as mock_sm,
            patch(
                "ccgram.handlers.interactive.interactive_ui.rate_limit_send",
                new_callable=AsyncMock,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -999

            result = await handle_interactive_ui(mock_bot, 100, "@2", thread_id=42)
            assert result is False

            # Cooldown should be set to ~60s, not the default 5s
            ikey = (100, 42)
            assert ikey in _send_cooldowns
            import time

            cooldown_remaining = _send_cooldowns[ikey] - time.monotonic()
            assert cooldown_remaining > 30  # well above the default 5s
            assert cooldown_remaining <= _DEAD_TOPIC_RETRY_INTERVAL

    async def test_non_dead_topic_error_uses_normal_cooldown(self) -> None:
        from unittest.mock import AsyncMock, patch

        from telegram.error import BadRequest

        from ccgram.handlers.interactive.interactive_ui import (
            _SEND_RETRY_INTERVAL,
            _send_cooldowns,
            handle_interactive_ui,
        )

        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = BadRequest("Chat not found")

        with (
            patch(
                "ccgram.handlers.interactive.interactive_ui._capture_interactive_content",
                new_callable=AsyncMock,
                return_value=("AskUserQuestion", "Pick one:"),
            ),
            patch(
                "ccgram.handlers.interactive.interactive_ui.thread_router"
            ) as mock_sm,
            patch(
                "ccgram.handlers.interactive.interactive_ui.rate_limit_send",
                new_callable=AsyncMock,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -999

            result = await handle_interactive_ui(mock_bot, 100, "@2", thread_id=42)
            assert result is False

            # Normal cooldown — should be around now, not 60s into the future
            ikey = (100, 42)
            assert ikey in _send_cooldowns
            import time

            cooldown_remaining = _send_cooldowns[ikey] - time.monotonic()
            assert cooldown_remaining <= _SEND_RETRY_INTERVAL


class TestLookupPaneName:
    @pytest.fixture(autouse=True)
    def _isolated_window_store(self):  # type: ignore[no-untyped-def]
        from ccgram.window_state_store import window_store

        saved = dict(window_store.window_states)
        window_store.window_states.clear()
        try:
            yield
        finally:
            window_store.window_states.clear()
            window_store.window_states.update(saved)

    def test_returns_name_when_pane_recorded(self) -> None:
        from ccgram.handlers.interactive.interactive_ui import _lookup_pane_name
        from ccgram.window_state_store import PaneInfo, WindowState, window_store

        state = WindowState()
        state.panes["%5"] = PaneInfo(pane_id="%5", name="api-gateway")
        window_store.window_states["@0"] = state

        assert _lookup_pane_name("@0", "%5") == "api-gateway"

    def test_returns_none_when_pane_missing(self) -> None:
        from ccgram.handlers.interactive.interactive_ui import _lookup_pane_name

        assert _lookup_pane_name("@0", "%99") is None

    def test_returns_none_when_pane_has_no_name(self) -> None:
        from ccgram.handlers.interactive.interactive_ui import _lookup_pane_name
        from ccgram.window_state_store import PaneInfo, WindowState, window_store

        state = WindowState()
        state.panes["%5"] = PaneInfo(pane_id="%5", name=None)
        window_store.window_states["@0"] = state

        assert _lookup_pane_name("@0", "%5") is None


class TestHandleInteractiveUIPaneName:
    @pytest.fixture(autouse=True)
    def _clear_state(self):  # type: ignore[no-untyped-def]
        from ccgram.handlers.interactive.interactive_ui import (
            _interactive_mode,
            _interactive_msgs,
            _send_cooldowns,
        )
        from ccgram.window_state_store import window_store

        _interactive_mode.clear()
        _interactive_msgs.clear()
        _send_cooldowns.clear()
        saved = dict(window_store.window_states)
        window_store.window_states.clear()
        yield
        window_store.window_states.clear()
        window_store.window_states.update(saved)

    async def test_named_pane_label_in_sent_message(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from ccgram.handlers.interactive.interactive_ui import handle_interactive_ui
        from ccgram.window_state_store import PaneInfo, WindowState, window_store

        state = WindowState()
        state.panes["%5"] = PaneInfo(pane_id="%5", name="api-gateway")
        window_store.window_states["@2"] = state

        mock_bot = AsyncMock()
        sent = MagicMock()
        sent.message_id = 42
        mock_bot.send_message.return_value = sent

        with (
            patch(
                "ccgram.handlers.interactive.interactive_ui._capture_interactive_content",
                new_callable=AsyncMock,
                return_value=("AskUserQuestion", "Pick one:"),
            ),
            patch(
                "ccgram.handlers.interactive.interactive_ui.thread_router"
            ) as mock_sm,
            patch(
                "ccgram.handlers.interactive.interactive_ui.rate_limit_send",
                new_callable=AsyncMock,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -999
            ok = await handle_interactive_ui(
                mock_bot, 100, "@2", thread_id=42, pane_id="%5"
            )

        assert ok is True
        sent_text = mock_bot.send_message.call_args.kwargs["text"]
        assert "api-gateway (%5):" in sent_text
        assert "Pane (%5):" not in sent_text

    async def test_unnamed_pane_falls_back_to_generic_label(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from ccgram.handlers.interactive.interactive_ui import handle_interactive_ui

        mock_bot = AsyncMock()
        sent = MagicMock()
        sent.message_id = 42
        mock_bot.send_message.return_value = sent

        with (
            patch(
                "ccgram.handlers.interactive.interactive_ui._capture_interactive_content",
                new_callable=AsyncMock,
                return_value=("AskUserQuestion", "Pick one:"),
            ),
            patch(
                "ccgram.handlers.interactive.interactive_ui.thread_router"
            ) as mock_sm,
            patch(
                "ccgram.handlers.interactive.interactive_ui.rate_limit_send",
                new_callable=AsyncMock,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -999
            ok = await handle_interactive_ui(
                mock_bot, 100, "@2", thread_id=42, pane_id="%5"
            )

        assert ok is True
        sent_text = mock_bot.send_message.call_args.kwargs["text"]
        assert "Pane (%5):" in sent_text


class TestHandleInteractiveUITransientRetry:
    @pytest.fixture(autouse=True)
    def _clear_state(self):
        from ccgram.handlers.interactive.interactive_ui import (
            _interactive_mode,
            _interactive_msgs,
            _send_cooldowns,
        )

        _interactive_mode.clear()
        _interactive_msgs.clear()
        _send_cooldowns.clear()
        yield
        _interactive_mode.clear()
        _interactive_msgs.clear()
        _send_cooldowns.clear()

    async def test_timed_out_retries_then_succeeds(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from telegram.error import TimedOut

        from ccgram.handlers.interactive.interactive_ui import handle_interactive_ui

        mock_bot = AsyncMock()
        sent = MagicMock()
        sent.message_id = 42
        mock_bot.send_message.side_effect = [TimedOut("blip"), sent]

        with (
            patch(
                "ccgram.handlers.interactive.interactive_ui._capture_interactive_content",
                new_callable=AsyncMock,
                return_value=("AskUserQuestion", "Pick one:"),
            ),
            patch(
                "ccgram.handlers.interactive.interactive_ui.thread_router"
            ) as mock_sm,
            patch(
                "ccgram.handlers.interactive.interactive_ui.rate_limit_send",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.interactive.interactive_ui.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -999
            ok = await handle_interactive_ui(mock_bot, 100, "@2", thread_id=42)

        assert ok is True
        assert mock_bot.send_message.call_count == 2

    async def test_timed_out_exhausts_retries(self) -> None:
        from unittest.mock import AsyncMock, patch

        from telegram.error import TimedOut

        from ccgram.handlers.interactive.interactive_ui import handle_interactive_ui

        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = TimedOut("persistent")

        with (
            patch(
                "ccgram.handlers.interactive.interactive_ui._capture_interactive_content",
                new_callable=AsyncMock,
                return_value=("AskUserQuestion", "Pick one:"),
            ),
            patch(
                "ccgram.handlers.interactive.interactive_ui.thread_router"
            ) as mock_sm,
            patch(
                "ccgram.handlers.interactive.interactive_ui.rate_limit_send",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.interactive.interactive_ui.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_sm.resolve_chat_id.return_value = -999
            ok = await handle_interactive_ui(mock_bot, 100, "@2", thread_id=42)

        assert ok is False
        assert mock_bot.send_message.call_count == 2
