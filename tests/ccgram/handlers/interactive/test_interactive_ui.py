"""Tests for interactive UI rendering."""

import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import InlineKeyboardMarkup
from telegram.error import BadRequest, TimedOut

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
    _DEAD_TOPIC_RETRY_INTERVAL,
    _SEND_RETRY_INTERVAL,
    _build_interactive_keyboard,
    _interactive_mode,
    _interactive_msgs,
    _lookup_pane_name,
    _send_cooldowns,
    clear_interactive_mode,
    format_interactive_message,
    get_interactive_window,
    handle_interactive_ui,
    INTERACTIVE_INSTRUCTION_LINE,
    set_interactive_mode,
)
from ccgram.window_state_store import PaneInfo, WindowState, window_store


_UI = "ccgram.handlers.interactive.interactive_ui"


@contextmanager
def _interactive_env(bot: AsyncMock):
    """Patch the terminal capture + routing collaborators of handle_interactive_ui."""
    with (
        patch(
            f"{_UI}._capture_interactive_content",
            new_callable=AsyncMock,
            return_value=("AskUserQuestion", "Pick one:"),
        ),
        patch(f"{_UI}.thread_router") as mock_router,
        patch(f"{_UI}.rate_limit_send", new_callable=AsyncMock),
        patch(f"{_UI}.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_router.resolve_chat_id.return_value = -999
        yield bot


def _sending_bot(*, message_id: int = 42) -> AsyncMock:
    bot = AsyncMock()
    sent = MagicMock()
    sent.message_id = message_id
    bot.send_message.return_value = sent
    return bot


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


class TestFormatInteractiveMessage:
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

    @pytest.mark.parametrize("pane_name", ["   ", None], ids=["blank", "unset"])
    def test_unusable_pane_name_falls_back_to_generic(
        self, pane_name: str | None
    ) -> None:
        out = format_interactive_message("Body", pane_id="%5", pane_name=pane_name)
        assert "Pane (%5):" in out

    def test_pane_name_ignored_without_pane_id(self) -> None:
        out = format_interactive_message("Body", pane_name="api-gateway")
        assert "api-gateway" not in out
        assert "Pane (" not in out


class TestInteractiveModeTracking:
    @pytest.fixture(autouse=True)
    def _clear_interactive_mode(self) -> None:
        _interactive_mode.clear()

    def test_set_and_get(self) -> None:
        set_interactive_mode(100, "@0", thread_id=42, chat_id=-999)
        assert get_interactive_window(100, 42, chat_id=-999) == "@0"

    def test_clear(self) -> None:
        set_interactive_mode(100, "@0", thread_id=42, chat_id=-999)
        clear_interactive_mode(100, thread_id=42, chat_id=-999)
        assert get_interactive_window(100, 42, chat_id=-999) is None

    def test_none_thread_uses_zero(self) -> None:
        set_interactive_mode(100, "@0", thread_id=None, chat_id=-999)
        assert get_interactive_window(100, None, chat_id=-999) == "@0"


@pytest.fixture
def _clear_send_state():
    for state in (_interactive_mode, _interactive_msgs, _send_cooldowns):
        state.clear()
    yield
    for state in (_interactive_mode, _interactive_msgs, _send_cooldowns):
        state.clear()


@pytest.fixture
def _isolated_window_store():
    saved = dict(window_store.window_states)
    window_store.window_states.clear()
    try:
        yield
    finally:
        window_store.window_states.clear()
        window_store.window_states.update(saved)


class TestSendCooldown:
    """A topic that is gone must back off far longer than a transient failure."""

    async def test_dead_topic_applies_longer_cooldown(self, _clear_send_state) -> None:
        bot = AsyncMock()
        bot.send_message.side_effect = BadRequest("Message thread not found")

        with _interactive_env(bot):
            assert await handle_interactive_ui(bot, 100, "@2", thread_id=42) is False

        remaining = _send_cooldowns[(100, -999, 42)] - time.monotonic()
        assert remaining > _SEND_RETRY_INTERVAL
        assert remaining <= _DEAD_TOPIC_RETRY_INTERVAL

    async def test_cooldown_suppresses_the_next_send(self, _clear_send_state) -> None:
        bot = _sending_bot()
        _send_cooldowns[(100, -999, 42)] = time.monotonic()

        with _interactive_env(bot):
            assert await handle_interactive_ui(bot, 100, "@2", thread_id=42) is False

        bot.send_message.assert_not_called()

    async def test_other_error_uses_normal_cooldown(self, _clear_send_state) -> None:
        bot = AsyncMock()
        bot.send_message.side_effect = BadRequest("Chat not found")

        with _interactive_env(bot):
            assert await handle_interactive_ui(bot, 100, "@2", thread_id=42) is False

        remaining = _send_cooldowns[(100, -999, 42)] - time.monotonic()
        assert remaining <= _SEND_RETRY_INTERVAL


class TestPaneLabel:
    async def test_named_pane_label_in_sent_message(
        self, _clear_send_state, _isolated_window_store
    ) -> None:
        state = WindowState()
        state.panes["%5"] = PaneInfo(pane_id="%5", name="api-gateway")
        window_store.window_states["@2"] = state

        bot = _sending_bot()
        with _interactive_env(bot):
            ok = await handle_interactive_ui(bot, 100, "@2", thread_id=42, pane_id="%5")

        assert ok is True
        sent_text = bot.send_message.call_args.kwargs["text"]
        assert "api-gateway (%5):" in sent_text
        assert "Pane (%5):" not in sent_text

    async def test_unnamed_pane_falls_back_to_generic_label(
        self, _clear_send_state, _isolated_window_store
    ) -> None:
        bot = _sending_bot()
        with _interactive_env(bot):
            ok = await handle_interactive_ui(bot, 100, "@2", thread_id=42, pane_id="%5")

        assert ok is True
        assert "Pane (%5):" in bot.send_message.call_args.kwargs["text"]

    def test_lookup_returns_recorded_name(self, _isolated_window_store) -> None:
        state = WindowState()
        state.panes["%5"] = PaneInfo(pane_id="%5", name="api-gateway")
        window_store.window_states["@0"] = state

        assert _lookup_pane_name("@0", "%5") == "api-gateway"

    def test_lookup_returns_none_for_unknown_pane(self, _isolated_window_store) -> None:
        assert _lookup_pane_name("@0", "%99") is None

    def test_lookup_returns_none_when_pane_has_no_name(
        self, _isolated_window_store
    ) -> None:
        state = WindowState()
        state.panes["%5"] = PaneInfo(pane_id="%5", name=None)
        window_store.window_states["@0"] = state

        assert _lookup_pane_name("@0", "%5") is None


class TestTransientRetry:
    async def test_timed_out_retries_then_succeeds(self, _clear_send_state) -> None:
        bot = _sending_bot()
        sent = bot.send_message.return_value
        bot.send_message.side_effect = [TimedOut("blip"), sent]

        with _interactive_env(bot):
            ok = await handle_interactive_ui(bot, 100, "@2", thread_id=42)

        assert ok is True
        assert bot.send_message.call_count == 2

    async def test_timed_out_exhausts_retries(self, _clear_send_state) -> None:
        bot = AsyncMock()
        bot.send_message.side_effect = TimedOut("persistent")

        with _interactive_env(bot):
            ok = await handle_interactive_ui(bot, 100, "@2", thread_id=42)

        assert ok is False
        assert bot.send_message.call_count == 2
