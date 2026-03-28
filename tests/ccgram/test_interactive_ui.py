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
from ccgram.handlers.interactive_ui import _build_interactive_keyboard


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
        assert all("@12:%5" in d for d in data)

    def test_callback_data_truncated_to_64_bytes(self) -> None:
        data = _cb_data(
            _build_interactive_keyboard("@" + "9" * 60, pane_id="%" + "1" * 60)
        )
        assert all(len(d) <= 64 for d in data)


class TestInteractiveModeTracking:
    @pytest.fixture(autouse=True)
    def _clear_interactive_mode(self) -> None:
        from ccgram.handlers.interactive_ui import _interactive_mode

        _interactive_mode.clear()

    def test_set_and_get(self) -> None:
        from ccgram.handlers.interactive_ui import (
            get_interactive_window,
            set_interactive_mode,
        )

        set_interactive_mode(100, "@0", thread_id=42)
        assert get_interactive_window(100, 42) == "@0"

    def test_clear(self) -> None:
        from ccgram.handlers.interactive_ui import (
            clear_interactive_mode,
            get_interactive_window,
            set_interactive_mode,
        )

        set_interactive_mode(100, "@0", thread_id=42)
        clear_interactive_mode(100, thread_id=42)
        assert get_interactive_window(100, 42) is None

    def test_none_thread_uses_zero(self) -> None:
        from ccgram.handlers.interactive_ui import (
            get_interactive_window,
            set_interactive_mode,
        )

        set_interactive_mode(100, "@0", thread_id=None)
        assert get_interactive_window(100, None) == "@0"


class TestDeadTopicCooldown:
    """Verify longer backoff when topic is deleted (thread not found)."""

    @pytest.fixture(autouse=True)
    def _clear_state(self) -> None:
        from ccgram.handlers.interactive_ui import (
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

        from ccgram.handlers.interactive_ui import (
            _DEAD_TOPIC_RETRY_INTERVAL,
            _send_cooldowns,
            handle_interactive_ui,
        )

        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = BadRequest("Message thread not found")

        with (
            patch(
                "ccgram.handlers.interactive_ui._capture_interactive_content",
                new_callable=AsyncMock,
                return_value=("AskUserQuestion", "Pick one:"),
            ),
            patch("ccgram.handlers.interactive_ui.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.interactive_ui.rate_limit_send",
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

        from ccgram.handlers.interactive_ui import (
            _SEND_RETRY_INTERVAL,
            _send_cooldowns,
            handle_interactive_ui,
        )

        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = BadRequest("Chat not found")

        with (
            patch(
                "ccgram.handlers.interactive_ui._capture_interactive_content",
                new_callable=AsyncMock,
                return_value=("AskUserQuestion", "Pick one:"),
            ),
            patch("ccgram.handlers.interactive_ui.session_manager") as mock_sm,
            patch(
                "ccgram.handlers.interactive_ui.rate_limit_send",
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
