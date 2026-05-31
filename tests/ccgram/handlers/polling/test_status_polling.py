import time

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from telegram import Bot
from telegram.error import BadRequest, TelegramError

from _helpers import make_mock_provider

from ccgram.handlers.topics.topic_lifecycle import (
    check_autoclose_timers,
    probe_topic_existence,
    prune_stale_state,
)
from ccgram.handlers.polling.window_tick import (
    _handle_dead_window_notification,
    _parse_with_pyte,
    _scan_window_panes,
    decide_tick,
)
from ccgram.handlers.polling.polling_state import (
    interactive_strategy,
    lifecycle_strategy,
    terminal_poll_state,
    terminal_screen_buffer,
)
from ccgram.handlers.polling.polling_types import (
    MAX_PROBE_FAILURES,
    TickContext,
    is_shell_prompt,
)
from ccgram.telegram_client import PTBTelegramClient
from ccgram.tmux_manager import PaneInfo


def _assert_handle_called_once_with_client(mock_handle, bot, *args, **kwargs):
    mock_handle.assert_called_once()
    client_arg = mock_handle.call_args.args[0]
    assert isinstance(client_arg, PTBTelegramClient)
    assert client_arg.bot is bot
    assert mock_handle.call_args.args[1:] == args
    assert mock_handle.call_args.kwargs == kwargs


def _assert_clear_called_once_with_client(mock_clear, user_id, bot, thread_id):
    mock_clear.assert_called_once()
    args = mock_clear.call_args.args
    assert args[0] == user_id
    assert isinstance(args[1], PTBTelegramClient)
    assert args[1].bot is bot
    assert args[2] == thread_id


_window_poll_state = terminal_poll_state._states
_topic_poll_state = lifecycle_strategy._states
_dead_notified = lifecycle_strategy._dead_notified
_pane_alert_hashes = interactive_strategy._pane_alert_hashes
_start_autoclose_timer = lifecycle_strategy.start_autoclose_timer
_clear_autoclose_timer = lifecycle_strategy.clear_autoclose_timer


def _has_autoclose(user_id: int, thread_id: int) -> bool:
    ts = _topic_poll_state.get((user_id, thread_id))
    return ts is not None and ts.autoclose is not None


def _get_autoclose(user_id: int, thread_id: int) -> tuple[str, float] | None:
    ts = _topic_poll_state.get((user_id, thread_id))
    return ts.autoclose if ts else None


@pytest.fixture(autouse=True)
def _reset():
    _window_poll_state.clear()
    _topic_poll_state.clear()
    _dead_notified.clear()
    yield
    _window_poll_state.clear()
    _topic_poll_state.clear()
    _dead_notified.clear()


class TestIsShellPrompt:
    @pytest.mark.parametrize(
        "cmd",
        ["bash", "zsh", "fish", "sh", "/usr/bin/zsh", "  bash  ", "dash", "ksh"],
    )
    def test_shell_detected(self, cmd: str) -> None:
        assert is_shell_prompt(cmd) is True

    @pytest.mark.parametrize("cmd", ["node", "claude", "npx", ""])
    def test_non_shell_rejected(self, cmd: str) -> None:
        assert is_shell_prompt(cmd) is False


class TestAutocloseTimers:
    def test_start_timer(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        assert _get_autoclose(1, 42) == ("done", 100.0)

    def test_start_timer_preserves_existing_same_state(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        _start_autoclose_timer(1, 42, "done", 200.0)
        assert _get_autoclose(1, 42) == ("done", 100.0)

    def test_start_timer_resets_on_state_change(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        _start_autoclose_timer(1, 42, "dead", 200.0)
        assert _get_autoclose(1, 42) == ("dead", 200.0)

    def test_clear_on_active(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        _clear_autoclose_timer(1, 42)
        assert not _has_autoclose(1, 42)

    def test_clear_timer(self) -> None:
        _start_autoclose_timer(1, 42, "done", 100.0)
        lifecycle_strategy.clear_autoclose_timer(1, 42)
        assert not _has_autoclose(1, 42)

    def test_clear_nonexistent_is_noop(self) -> None:
        lifecycle_strategy.clear_autoclose_timer(1, 42)

    @pytest.mark.parametrize(
        ("state", "minutes", "elapsed"),
        [("done", 30, 30 * 60 + 1), ("dead", 10, 10 * 60 + 1)],
        ids=["done", "dead"],
    )
    async def test_check_expired(
        self, state: str, minutes: int, elapsed: float
    ) -> None:
        _start_autoclose_timer(1, 42, state, 0.0)
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.topics.topic_lifecycle.config") as mock_config,
            patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as mock_tr,
            patch("ccgram.handlers.topics.topic_lifecycle.time") as mock_time,
            patch("ccgram.handlers.topics.topic_lifecycle.clear_topic_state"),
        ):
            mock_config.autoclose_done_minutes = 30
            mock_config.autoclose_dead_minutes = minutes
            mock_time.monotonic.return_value = elapsed
            mock_tr.resolve_chat_id.return_value = -100
            await check_autoclose_timers(bot)
        bot.delete_forum_topic.assert_called_once_with(
            chat_id=-100, message_thread_id=42
        )
        mock_tr.unbind_thread.assert_called_once_with(1, 42)
        assert not _has_autoclose(1, 42)

    async def test_check_not_expired_yet(self) -> None:
        _start_autoclose_timer(1, 42, "done", 0.0)
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.topics.topic_lifecycle.config") as mock_config,
            patch("ccgram.handlers.topics.topic_lifecycle.time") as mock_time,
        ):
            mock_config.autoclose_done_minutes = 30
            mock_config.autoclose_dead_minutes = 10
            mock_time.monotonic.return_value = 29 * 60
            await check_autoclose_timers(bot)
        bot.close_forum_topic.assert_not_called()
        assert _has_autoclose(1, 42)

    async def test_check_disabled_when_zero(self) -> None:
        _start_autoclose_timer(1, 42, "done", 0.0)
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.topics.topic_lifecycle.config") as mock_config,
            patch("ccgram.handlers.topics.topic_lifecycle.time") as mock_time,
        ):
            mock_config.autoclose_done_minutes = 0
            mock_config.autoclose_dead_minutes = 0
            mock_time.monotonic.return_value = 999999
            await check_autoclose_timers(bot)
        bot.close_forum_topic.assert_not_called()

    async def test_check_telegram_error_handled(self) -> None:
        _start_autoclose_timer(1, 42, "done", 0.0)
        bot = AsyncMock(spec=Bot)
        bot.close_forum_topic.side_effect = TelegramError("fail")
        with (
            patch("ccgram.handlers.topics.topic_lifecycle.config") as mock_config,
            patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as mock_tr,
            patch("ccgram.handlers.topics.topic_lifecycle.time") as mock_time,
        ):
            mock_config.autoclose_done_minutes = 30
            mock_config.autoclose_dead_minutes = 10
            mock_time.monotonic.return_value = 30 * 60 + 1
            mock_tr.resolve_chat_id.return_value = -100
            await check_autoclose_timers(bot)
        assert not _has_autoclose(1, 42)

    async def test_check_treats_missing_topic_as_removed(self) -> None:
        _start_autoclose_timer(1, 42, "done", 0.0)
        bot = AsyncMock(spec=Bot)
        bot.delete_forum_topic.side_effect = BadRequest("Topic_id_invalid")
        with (
            patch("ccgram.handlers.topics.topic_lifecycle.config") as mock_config,
            patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as mock_tr,
            patch("ccgram.handlers.topics.topic_lifecycle.time") as mock_time,
            patch(
                "ccgram.handlers.topics.topic_lifecycle.clear_topic_state",
                new_callable=AsyncMock,
            ) as mock_clear,
        ):
            mock_config.autoclose_done_minutes = 30
            mock_config.autoclose_dead_minutes = 10
            mock_time.monotonic.return_value = 30 * 60 + 1
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_window_for_thread.return_value = "@0"

            await check_autoclose_timers(bot)

        bot.close_forum_topic.assert_not_called()
        mock_tr.unbind_thread.assert_called_once_with(1, 42)
        mock_clear.assert_awaited_once()
        assert not _has_autoclose(1, 42)


class TestTranscriptActivityHeuristic:
    def test_active_when_recent_transcript(self) -> None:
        now = time.monotonic()
        result = terminal_poll_state.is_recently_active("@0", now - 5.0)
        assert result is True
        assert _window_poll_state.get("@0") and _window_poll_state["@0"].has_seen_status

    def test_inactive_when_stale_transcript(self) -> None:
        now = time.monotonic()
        result = terminal_poll_state.is_recently_active("@0", now - 20.0)
        assert result is False
        assert not (
            _window_poll_state.get("@0") and _window_poll_state["@0"].has_seen_status
        )

    def test_inactive_when_no_activity_ts(self) -> None:
        result = terminal_poll_state.is_recently_active("@0", None)
        assert result is False

    def test_clears_startup_timer_on_activity(self) -> None:
        now = time.monotonic()
        terminal_poll_state.get_state("@0").startup_time = now - 15.0
        result = terminal_poll_state.is_recently_active("@0", now - 3.0)
        assert result is True
        assert (
            _window_poll_state.get("@0") is None
            or _window_poll_state["@0"].startup_time is None
        )


def _make_ctx(
    window_id: str = "@0",
    resolved_status_text: str | None = None,
    is_shell_prompt: bool = False,
    has_seen_status: bool = False,
    is_recently_active: bool = False,
    startup_time: float | None = None,
    is_dead_window: bool = False,
    supports_hook: bool = True,
) -> TickContext:
    return TickContext(
        window_id=window_id,
        resolved_status_text=resolved_status_text,
        is_shell_prompt=is_shell_prompt,
        has_seen_status=has_seen_status,
        is_recently_active=is_recently_active,
        startup_time=startup_time,
        is_dead_window=is_dead_window,
        supports_hook=supports_hook,
    )


class TestStartupTimeout:
    def test_first_poll_no_startup_time_yields_starting(self) -> None:
        ctx = _make_ctx(startup_time=None)
        decision = decide_tick(ctx)
        assert decision.transition == "starting"

    def test_startup_timeout_transitions_to_idle(self) -> None:
        ctx = _make_ctx(startup_time=time.monotonic() - 31.0)
        decision = decide_tick(ctx)
        assert decision.transition == "idle"

    def test_startup_grace_period_stays_starting(self) -> None:
        ctx = _make_ctx(startup_time=time.monotonic())
        decision = decide_tick(ctx)
        assert decision.transition == "starting"


@pytest.fixture()
def _reset_pyte():
    terminal_screen_buffer.reset_screen_buffer_state()
    interactive_strategy.clear_all_alerts()
    yield
    terminal_screen_buffer.reset_screen_buffer_state()
    interactive_strategy.clear_all_alerts()


_SEP = "─" * 30


@pytest.mark.usefixtures("_reset_pyte")
class TestParseWithPyte:
    @pytest.mark.parametrize(
        ("spinner", "text", "expected_raw"),
        [
            ("✻", "Reading file src/main.py", "Reading file src/main.py"),
            ("⠋", "Thinking about things", "Thinking about things"),
        ],
        ids=["unicode-spinner", "braille-spinner"],
    )
    def test_detects_spinner(self, spinner: str, text: str, expected_raw: str) -> None:
        pane_text = f"Output\n{spinner} {text}\n{_SEP}\n"
        result = _parse_with_pyte("@0", pane_text)
        assert result is not None
        assert result.raw_text == expected_raw
        assert result.is_interactive is False

    def test_detects_interactive_ui(self) -> None:
        pane_text = (
            "  Would you like to proceed?\n"
            f"  {_SEP}\n"
            "  Yes     No\n"
            f"  {_SEP}\n"
            "  ctrl-g to edit in vim\n"
        )
        result = _parse_with_pyte("@0", pane_text)
        assert result is not None
        assert result.is_interactive is True
        assert result.ui_type == "ExitPlanMode"

    def test_returns_none_for_plain_text(self) -> None:
        result = _parse_with_pyte("@0", "$ echo hello\nhello\n$\n")
        assert result is None

    def test_screen_buffer_cached_per_window(self) -> None:
        pane_text = f"Output\n✻ Working\n{_SEP}\n"
        _parse_with_pyte("@0", pane_text)
        _parse_with_pyte("@1", pane_text)
        assert _window_poll_state["@0"].screen_buffer is not None
        assert _window_poll_state["@1"].screen_buffer is not None

    def test_interactive_takes_precedence_over_status(self) -> None:
        pane_text = (
            f"✻ Working on task\n{_SEP}\n"
            "  Do you want to proceed?\n"
            "  Allow write to /tmp/foo\n"
            "  Esc to cancel\n"
        )
        result = _parse_with_pyte("@0", pane_text)
        assert result is not None
        assert result.is_interactive is True
        assert result.ui_type == "PermissionPrompt"


@pytest.mark.usefixtures("_reset_pyte")
class TestPyteContentHashCaching:
    def test_cache_hit_returns_same_result(self) -> None:
        pane_text = f"Output\n✻ Working on task\n{_SEP}\n"
        result1 = _parse_with_pyte("@0", pane_text)
        result2 = _parse_with_pyte("@0", pane_text)
        assert result1 is not None
        assert result2 is result1

    def test_cache_miss_on_changed_content(self) -> None:
        result1 = _parse_with_pyte("@0", f"Output\n✻ Reading file\n{_SEP}\n")
        result2 = _parse_with_pyte("@0", f"Output\n✻ Writing file\n{_SEP}\n")
        assert result1 is not None
        assert result2 is not None
        assert result1 is not result2
        assert result1.raw_text != result2.raw_text

    def test_cache_miss_on_dimension_change(self) -> None:
        pane_text = f"Output\n✻ Working\n{_SEP}\n"
        result1 = _parse_with_pyte("@0", pane_text, columns=80, rows=24)
        result2 = _parse_with_pyte("@0", pane_text, columns=120, rows=40)
        assert result1 is not None
        assert result2 is not None
        assert result2 is not result1

    def test_cache_none_result(self) -> None:
        pane_text = "$ echo hello\nhello\n$\n"
        result1 = _parse_with_pyte("@0", pane_text)
        result2 = _parse_with_pyte("@0", pane_text)
        assert result1 is None
        assert result2 is None
        assert terminal_poll_state.get_state("@0").last_pane_hash != 0

    def test_interactive_ui_not_cached(self) -> None:
        pane_text = (
            "  Would you like to proceed?\n"
            "  Yes / No\n"
            f"  {_SEP}\n"
            "  ctrl-g to edit in vim\n"
        )
        result1 = _parse_with_pyte("@0", pane_text)
        result2 = _parse_with_pyte("@0", pane_text)
        assert result1 is not None
        assert result1.is_interactive is True
        assert result2 is not result1

    def test_clear_screen_buffer_resets_cache(self) -> None:
        _parse_with_pyte("@0", f"Output\n✻ Working\n{_SEP}\n")
        ws = terminal_poll_state.get_state("@0")
        assert ws.last_pane_hash is not None

        terminal_screen_buffer.clear_screen_buffer("@0")
        assert ws.last_pane_hash is None
        assert ws.last_pyte_result is None


@pytest.mark.usefixtures("_reset_pyte")
class TestPyteDimensionPassthrough:
    def test_custom_dimensions_used(self) -> None:
        _parse_with_pyte("@0", f"Output\n✻ Working\n{_SEP}\n", columns=80, rows=24)
        buf = terminal_poll_state.get_state("@0").screen_buffer
        assert buf is not None
        assert buf.columns == 80
        assert buf.rows == 24

    def test_zero_dimensions_fall_back_to_default(self) -> None:
        _parse_with_pyte("@0", f"Output\n✻ Working\n{_SEP}\n", columns=0, rows=0)
        buf = terminal_poll_state.get_state("@0").screen_buffer
        assert buf is not None
        assert buf.columns == 200
        assert buf.rows == 50

    def test_resize_reuses_buffer(self) -> None:
        pane_text = f"Output\n✻ Working\n{_SEP}\n"
        _parse_with_pyte("@0", pane_text, columns=80, rows=24)
        buf1 = terminal_poll_state.get_state("@0").screen_buffer
        assert buf1 is not None

        _parse_with_pyte("@0", pane_text + " changed", columns=120, rows=40)
        buf2 = terminal_poll_state.get_state("@0").screen_buffer
        assert buf2 is buf1
        assert buf2 is not None
        assert buf2.columns == 120
        assert buf2.rows == 40


@pytest.mark.usefixtures("_reset_pyte")
class TestAnsiCapturePyteParsing:
    def test_ansi_spinner_detected(self) -> None:
        pane_text = f"Some output\n\x1b[36m✻ Reading file src/main.py\x1b[0m\n{_SEP}\n"
        result = _parse_with_pyte("@0", pane_text)
        assert result is not None
        assert result.raw_text == "Reading file src/main.py"
        assert result.is_interactive is False

    def test_ansi_interactive_ui_detected(self) -> None:
        pane_text = (
            "  \x1b[1mWould you like to proceed?\x1b[0m\n"
            f"  {_SEP}\n"
            "  Yes     No\n"
            f"  {_SEP}\n"
            "  ctrl-g to edit in vim\n"
        )
        result = _parse_with_pyte("@0", pane_text)
        assert result is not None
        assert result.is_interactive is True

    def test_last_rendered_text_populated(self) -> None:
        _parse_with_pyte("@0", "\x1b[32mHello\x1b[0m\nWorld\n")
        ws = terminal_poll_state.get_state("@0")
        assert ws.last_rendered_text is not None
        assert "\x1b" not in ws.last_rendered_text
        assert "Hello" in ws.last_rendered_text
        assert "World" in ws.last_rendered_text

    def test_last_rendered_text_cached_on_hash_hit(self) -> None:
        pane_text = "$ echo hello\nhello\n"
        _parse_with_pyte("@0", pane_text)
        rendered_first = terminal_poll_state.get_state("@0").last_rendered_text
        _parse_with_pyte("@0", pane_text)
        assert terminal_poll_state.get_state("@0").last_rendered_text is rendered_first

    def test_last_rendered_text_cleared_by_clear_screen_buffer(self) -> None:
        _parse_with_pyte("@0", "Hello\nWorld\n")
        ws = terminal_poll_state.get_state("@0")
        assert ws.last_rendered_text is not None
        terminal_screen_buffer.clear_screen_buffer("@0")
        assert ws.last_rendered_text is None

    def test_empty_screen_renders_as_empty_string(self) -> None:
        _parse_with_pyte("@0", "\n\n\n")
        assert terminal_poll_state.get_state("@0").last_rendered_text == ""


def _mock_update_status_patches(*, pyte_result, provider):
    from contextlib import ExitStack

    stack = ExitStack()
    mocks: dict[str, MagicMock] = {}
    mocks["tm"] = stack.enter_context(
        patch("ccgram.handlers.polling.window_tick.apply.tmux_manager")
    )
    mocks["sm"] = stack.enter_context(
        patch("ccgram.handlers.polling.window_tick.apply.window_query")
    )
    mocks["tr"] = stack.enter_context(
        patch("ccgram.handlers.polling.window_tick.apply.thread_router")
    )
    stack.enter_context(
        patch("ccgram.handlers.polling.window_tick.apply.update_topic_emoji")
    )
    mocks["enqueue"] = stack.enter_context(
        patch("ccgram.handlers.polling.window_tick.apply.enqueue_status_update")
    )
    stack.enter_context(
        patch(
            "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
            return_value=None,
        )
    )
    mocks["provider"] = stack.enter_context(
        patch(
            "ccgram.handlers.polling.window_tick.apply.get_provider_for_window",
            return_value=provider,
        )
    )
    # observe.py has its own bindings — patch those too so _resolve_status
    # sees the mocked provider/tmux_manager/window_query.
    stack.enter_context(
        patch(
            "ccgram.handlers.polling.window_tick.observe.get_provider_for_window",
            return_value=provider,
        )
    )
    observe_tm = stack.enter_context(
        patch("ccgram.handlers.polling.window_tick.observe.tmux_manager")
    )
    observe_tm.get_pane_title = AsyncMock(return_value="")
    stack.enter_context(
        patch("ccgram.handlers.polling.window_tick.observe.window_query")
    )
    stack.enter_context(
        patch(
            "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
            return_value=pyte_result,
        )
    )

    mock_window = MagicMock()
    mock_window.window_id = "@0"
    mock_window.window_name = "project"
    mock_window.pane_current_command = "node"
    mock_window.pane_width = 80
    mock_window.pane_height = 24
    mocks["tm"].find_window_by_id = AsyncMock(return_value=mock_window)
    mocks["tm"].capture_pane = AsyncMock(return_value="\x1b[1msome ansi output\x1b[0m")
    mocks["tm"].get_pane_title = AsyncMock(return_value="")
    mocks["tr"].resolve_chat_id.return_value = -100
    mocks["tr"].get_display_name.return_value = "project"

    return stack, mocks


class TestPyteFallbackInUpdateStatus:
    async def test_empty_rendered_text_does_not_fall_back_to_raw_ansi(self) -> None:
        stack, mocks = _mock_update_status_patches(
            pyte_result=None, provider=make_mock_provider(has_status=False)
        )
        with stack:
            from ccgram.handlers.polling.window_tick import (
                _update_status as update_status_message,
            )

            terminal_poll_state.get_state("@0").last_rendered_text = ""
            await update_status_message(AsyncMock(spec=Bot), 1, "@0", thread_id=42)

            call_args = mocks["provider"].return_value.parse_terminal_status.call_args
            assert call_args[0][0] == ""

    async def test_falls_back_to_provider_with_rendered_text(self) -> None:
        stack, mocks = _mock_update_status_patches(
            pyte_result=None, provider=make_mock_provider(has_status=True)
        )
        with stack:
            from ccgram.handlers.polling.window_tick import (
                _update_status as update_status_message,
            )

            terminal_poll_state.get_state(
                "@0"
            ).last_rendered_text = "clean rendered text"
            await update_status_message(AsyncMock(spec=Bot), 1, "@0", thread_id=42)

            provider_mock = mocks["provider"].return_value
            provider_mock.parse_terminal_status.assert_called_once()
            assert (
                provider_mock.parse_terminal_status.call_args[0][0]
                == "clean rendered text"
            )

    async def test_uses_pyte_result_when_available(self) -> None:
        from ccgram.providers.base import StatusUpdate

        pyte_status = StatusUpdate(
            raw_text="Reading file",
            display_label="\U0001f4d6 reading\u2026",
        )
        stack, mocks = _mock_update_status_patches(
            pyte_result=pyte_status, provider=make_mock_provider(has_status=True)
        )
        with stack:
            from ccgram.handlers.polling.window_tick import (
                _update_status as update_status_message,
            )

            await update_status_message(AsyncMock(spec=Bot), 1, "@0", thread_id=42)

            provider_mock = mocks["provider"].return_value
            provider_mock.parse_terminal_status.assert_not_called()
            mocks["enqueue"].assert_called_once()
            assert mocks["enqueue"].call_args[0][3] == "\U0001f4d6 Reading file"


class TestClearSeenStatus:
    def test_clears_seen_status_and_startup(self) -> None:
        terminal_poll_state.get_state("@0").has_seen_status = True
        terminal_poll_state.get_state("@0").startup_time = 100.0
        terminal_poll_state.clear_seen_status("@0")
        assert not (
            _window_poll_state.get("@0") and _window_poll_state["@0"].has_seen_status
        )
        assert (
            _window_poll_state.get("@0") is None
            or _window_poll_state["@0"].startup_time is None
        )


class TestTransitionToIdle:
    async def test_sends_idle_text(self) -> None:
        from ccgram.handlers.callback_data import IDLE_STATUS_TEXT
        from ccgram.handlers.polling.window_tick import _transition_to_idle

        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.update_topic_emoji"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.enqueue_status_update"
            ) as mock_enqueue,
            patch("ccgram.handlers.polling.window_tick.apply.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 100.0
            await _transition_to_idle(bot, 1, "@0", 42, -100, "project")
        mock_enqueue.assert_called_once()
        assert mock_enqueue.call_args[0][3] == IDLE_STATUS_TEXT
        assert mock_enqueue.call_args[1]["thread_id"] == 42


class TestShellPromptClearsStatus:
    def test_shell_prompt_with_hook_yields_done(self) -> None:
        ctx = _make_ctx(is_shell_prompt=True, supports_hook=True)
        decision = decide_tick(ctx)
        assert decision.transition == "done"

    def test_hookless_shell_prompt_yields_idle(self) -> None:
        ctx = _make_ctx(is_shell_prompt=True, supports_hook=False)
        decision = decide_tick(ctx)
        assert decision.transition == "idle"


class TestProbeFailures:
    async def test_probe_skips_suspended_windows(self) -> None:
        terminal_poll_state.get_state("@5").probe_failures = MAX_PROBE_FAILURES
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as mock_tr:
            mock_tr.iter_thread_bindings.return_value = [(1, 42, "@5")]
            await probe_topic_existence(bot)
        bot.unpin_all_forum_topic_messages.assert_not_called()

    async def test_probe_success_resets_counter(self) -> None:
        terminal_poll_state.get_state("@5").probe_failures = 2
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as mock_tr:
            mock_tr.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_tr.resolve_chat_id.return_value = -100
            await probe_topic_existence(bot)
        assert (
            _window_poll_state.get("@5") is None
            or _window_poll_state["@5"].probe_failures == 0
        )
        bot.unpin_all_forum_topic_messages.assert_called_once_with(
            chat_id=-100, message_thread_id=42
        )

    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(TelegramError("Timed out"), id="telegram-error"),
            pytest.param(BadRequest("Permission denied"), id="bad-request-other"),
        ],
    )
    async def test_probe_error_increments_counter(self, exc: TelegramError) -> None:
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages.side_effect = exc
        with patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as mock_tr:
            mock_tr.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_tr.resolve_chat_id.return_value = -100
            await probe_topic_existence(bot)
        assert _window_poll_state["@5"].probe_failures == 1

    async def test_probe_suspends_after_max_failures(self) -> None:
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages.side_effect = TelegramError("Timed out")
        with patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as mock_tr:
            mock_tr.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_tr.resolve_chat_id.return_value = -100
            for _ in range(MAX_PROBE_FAILURES + 1):
                await probe_topic_existence(bot)
        assert bot.unpin_all_forum_topic_messages.call_count == MAX_PROBE_FAILURES
        assert _window_poll_state["@5"].probe_failures == MAX_PROBE_FAILURES

    @pytest.mark.parametrize(
        "window_alive",
        [
            pytest.param(True, id="window-alive"),
            pytest.param(False, id="window-already-gone"),
        ],
    )
    async def test_topic_deleted_cleans_up(self, window_alive: bool) -> None:
        terminal_poll_state.get_state("@5").probe_failures = 1
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages.side_effect = BadRequest("Topic_id_invalid")
        mock_window = MagicMock()
        mock_window.window_id = "@5"
        with (
            patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as mock_tr,
            patch("ccgram.handlers.topics.topic_lifecycle.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.topics.topic_lifecycle.clear_topic_state",
                new_callable=AsyncMock,
            ) as mock_cleanup,
        ):
            mock_tr.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.find_window_by_id = AsyncMock(
                return_value=mock_window if window_alive else None
            )
            mock_tm.kill_window = AsyncMock()
            await probe_topic_existence(bot)
        mock_tm.kill_window.assert_not_called()
        mock_cleanup.assert_called_once_with(1, 42, bot, window_id="@5")
        mock_tr.unbind_thread.assert_called_once_with(1, 42)
        assert (
            _window_poll_state.get("@5") is None
            or _window_poll_state["@5"].probe_failures == 0
        )


class TestPruneStaleStatePolling:
    async def test_calls_sync_and_prune(self) -> None:
        mock_win = MagicMock()
        mock_win.window_id = "@1"
        mock_win.window_name = "proj"
        with patch("ccgram.handlers.topics.topic_lifecycle.session_manager") as mock_sm:
            mock_sm.sync_display_names.return_value = False
            mock_sm.prune_stale_state.return_value = False
            await prune_stale_state([mock_win])
        mock_sm.sync_display_names.assert_called_once_with([("@1", "proj")])
        mock_sm.prune_stale_state.assert_called_once_with({"@1"})

    async def test_empty_window_list(self) -> None:
        with patch("ccgram.handlers.topics.topic_lifecycle.session_manager") as mock_sm:
            mock_sm.sync_display_names.return_value = False
            mock_sm.prune_stale_state.return_value = False
            await prune_stale_state([])
        mock_sm.sync_display_names.assert_called_once_with([])
        mock_sm.prune_stale_state.assert_called_once_with(set())


class TestProviderSwitchPromptSetup:
    async def test_switch_to_shell_offers_prompt_setup(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="shell",
            ),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ) as mock_ensure,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(
                    session_id="",
                    cwd="/proj",
                    provider_name="claude",
                    transcript_path="",
                )
            }
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="fish", cwd="/proj")
            )
            await discover_and_register_transcript(
                "@7", client=bot, user_id=1, thread_id=42
            )

        mock_sm.set_window_provider.assert_called_once_with("@7", "shell", cwd="/proj")
        mock_ensure.assert_awaited_once()
        assert mock_ensure.call_args[0] == ("@7", "provider_switch")

    async def test_switch_to_claude_does_not_offer_prompt_setup(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = True

        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="claude",
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ) as mock_ensure,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(
                    session_id="",
                    cwd="/proj",
                    provider_name="shell",
                    transcript_path="/path/to/claude.jsonl",
                )
            }
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="claude", cwd="/proj")
            )
            await discover_and_register_transcript(
                "@7", client=bot, user_id=1, thread_id=42
            )

        mock_ensure.assert_not_awaited()

    async def test_fallback_shell_assignment_offers_prompt_setup(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="",
            ),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ) as mock_ensure,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(
                    session_id="",
                    cwd="/proj",
                    provider_name="",
                    transcript_path="",
                )
            }
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bash", cwd="/proj")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            mock_config.tmux_session_name = "ccgram"
            await discover_and_register_transcript(
                "@7", client=bot, user_id=1, thread_id=42
            )

        mock_sm.set_window_provider.assert_called_once_with("@7", "shell")
        mock_ensure.assert_awaited_once()
        assert mock_ensure.call_args[0] == ("@7", "provider_switch")

    async def test_fallback_shell_assignment_sets_up_prompt_without_bot(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="",
            ),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ) as mock_ensure,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.should_probe_pane_title_for_provider_detection",
                return_value=False,
            ),
        ):
            mock_ws.window_states = {
                "@7": MagicMock(
                    session_id="",
                    cwd="/proj",
                    provider_name="",
                    transcript_path="",
                )
            }
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bash", cwd="/proj")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            mock_config.tmux_session_name = "ccgram"
            await discover_and_register_transcript("@7")

        mock_ensure.assert_awaited_once()
        assert mock_ensure.call_args[0] == ("@7", "provider_switch")


class TestProviderSwitchChain:
    async def test_claude_to_shell_to_gemini_to_shell(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.window_state_store import WindowState

        state = WindowState(
            cwd="/proj", provider_name="claude", transcript_path="/tx/claude.jsonl"
        )

        def _set_provider(window_id: str, name: str, cwd: str | None = None) -> None:
            state.provider_name = name
            if cwd is not None:
                state.cwd = cwd

        claude_caps = MagicMock()
        claude_caps.capabilities.supports_hook = True
        claude_caps.capabilities.chat_first_command_path = False
        shell_caps = MagicMock()
        shell_caps.capabilities.supports_hook = False
        shell_caps.capabilities.chat_first_command_path = True
        gemini_caps = MagicMock()
        gemini_caps.capabilities.supports_hook = False
        gemini_caps.capabilities.chat_first_command_path = False
        gemini_caps.capabilities.name = "gemini"
        gemini_caps.discover_transcript.return_value = None

        provider_map: dict[str, MagicMock] = {
            "claude": claude_caps,
            "shell": shell_caps,
            "gemini": gemini_caps,
        }

        def _provider_for(window_id: str, name: str | None = None) -> MagicMock:
            return provider_map.get(name or "", claude_caps)

        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
                new_callable=AsyncMock,
            ) as mock_detect,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                side_effect=_provider_for,
            ),
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.ensure_setup",
                new_callable=AsyncMock,
            ) as mock_ensure,
            patch(
                "ccgram.handlers.shell.shell_capture.clear_shell_monitor_state"
            ) as mock_clear_capture,
            patch(
                "ccgram.handlers.shell.shell_prompt_orchestrator.clear_state"
            ) as mock_clear_orch,
        ):
            mock_ws.window_states = {"@7": state}

            def _clear_transcript(window_id: str) -> None:
                s = mock_ws.window_states.get(window_id)
                if s is not None:
                    s.transcript_path = ""

            mock_ws.clear_transcript_path.side_effect = _clear_transcript
            mock_sm.set_window_provider.side_effect = _set_provider
            mock_config.tmux_session_name = "ccgram"

            # Step 1: claude → shell. User exits claude, pane shows fish.
            mock_detect.return_value = "shell"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(
                    pane_current_command="fish", cwd="/proj", pane_tty=""
                )
            )
            await discover_and_register_transcript(
                "@7", client=bot, user_id=1, thread_id=42
            )
            assert state.provider_name == "shell"
            assert state.transcript_path == ""
            assert mock_ensure.await_count == 1
            assert mock_ensure.call_args is not None
            assert mock_ensure.call_args.args == ("@7", "provider_switch")
            mock_clear_capture.assert_not_called()
            mock_clear_orch.assert_not_called()

            # Step 2: shell → gemini. User runs `gemini` in shell.
            mock_detect.return_value = "gemini"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(
                    pane_current_command="gemini", cwd="/proj", pane_tty=""
                )
            )
            await discover_and_register_transcript(
                "@7", client=bot, user_id=1, thread_id=42
            )
            assert state.provider_name == "gemini"
            mock_clear_capture.assert_called_once_with("@7")
            mock_clear_orch.assert_called_once_with("@7")
            # No new ensure_setup call — we're leaving shell, not entering it.
            assert mock_ensure.await_count == 1

            # Step 3: gemini → shell. User exits gemini, pane shows fish.
            mock_detect.return_value = "shell"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(
                    pane_current_command="fish", cwd="/proj", pane_tty=""
                )
            )
            await discover_and_register_transcript(
                "@7", client=bot, user_id=1, thread_id=42
            )
            assert state.provider_name == "shell"
            assert state.transcript_path == ""
            assert mock_ensure.await_count == 2
            assert mock_ensure.call_args is not None
            assert mock_ensure.call_args.args == ("@7", "provider_switch")


class TestMaybeDiscoverTranscript:
    async def test_noop_when_discovered_session_matches_current(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.chat_first_command_path = False
        mock_provider.capabilities.name = "codex"
        mock_provider.discover_transcript.return_value = SessionStartEvent(
            session_id="existing-id",
            cwd="/proj",
            transcript_path="/path/existing.jsonl",
            window_key="ccgram:@7",
        )

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(
                    session_id="existing-id",
                    cwd="/proj",
                    transcript_path="/path/existing.jsonl",
                    provider_name="codex",
                )
            }
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bun")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            mock_config.tmux_session_name = "ccgram"
            await discover_and_register_transcript("@7")

        mock_sm.register_hookless_session.assert_not_called()
        mock_sm.write_hookless_session_map.assert_not_called()

    async def test_skips_when_no_cwd_and_no_tmux_window(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
        ):
            mock_ws.window_states = {"@7": MagicMock(session_id="", cwd="")}
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            await discover_and_register_transcript("@7")
        mock_sm.register_hookless_session.assert_not_called()

    async def test_falls_back_to_tmux_cwd_when_state_cwd_empty(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.chat_first_command_path = False
        mock_provider.capabilities.name = "codex"
        event = SessionStartEvent(
            session_id="uuid-xyz",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            window_key="ccgram:@7",
        )
        mock_provider.discover_transcript.return_value = event

        mock_state = MagicMock(session_id="", cwd="", provider_name="codex")
        mock_window = MagicMock(cwd="/my/project", pane_current_command="bun")

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
        ):
            mock_ws.window_states = {"@7": mock_state}
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            mock_config.tmux_session_name = "ccgram"
            await discover_and_register_transcript("@7")

        mock_sm.set_window_provider.assert_called_once_with(
            "@7", "codex", cwd="/my/project"
        )
        mock_sms.register_hookless_session.assert_called_once()

    async def test_skips_when_provider_has_hooks(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = True
        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                return_value=mock_provider,
            ),
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="claude")
            }
            await discover_and_register_transcript("@7")
        mock_sm.register_hookless_session.assert_not_called()

    async def test_skips_when_window_not_tracked(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
        ):
            mock_ws.window_states = {}
            await discover_and_register_transcript("@7")
        mock_sm.register_hookless_session.assert_not_called()

    async def test_registers_when_transcript_found(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.chat_first_command_path = False
        mock_provider.capabilities.name = "codex"
        event = SessionStartEvent(
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            window_key="ccgram:@7",
        )
        mock_provider.discover_transcript.return_value = event

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/my/project", provider_name="codex")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_window = MagicMock(pane_current_command="bun")
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            await discover_and_register_transcript("@7")

        mock_sms.register_hookless_session.assert_called_once_with(
            window_id="@7",
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            provider_name="codex",
        )
        mock_sms.write_hookless_session_map.assert_called_once_with(
            window_id="@7",
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            provider_name="codex",
        )

    async def test_skips_session_if_already_bound_to_other_window(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.chat_first_command_path = False
        mock_provider.capabilities.name = "codex"
        event = SessionStartEvent(
            session_id="shared-session",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            window_key="ccgram:@7",
        )
        mock_provider.discover_transcript.return_value = event

        mock_router = MagicMock()
        mock_router.iter_thread_bindings.return_value = [
            (123, 42, "@7"),
            (321, 7, "@9"),
        ]

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch("ccgram.thread_router.thread_router", mock_router),
        ):
            mock_ws.window_states = {
                "@7": MagicMock(
                    session_id="", cwd="/my/project", provider_name="codex"
                ),
                "@9": MagicMock(
                    session_id="shared-session",
                    cwd="/my/project",
                    provider_name="codex",
                ),
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bun")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            await discover_and_register_transcript("@7")

        mock_sms.register_hookless_session.assert_not_called()
        mock_sms.write_hookless_session_map.assert_not_called()

    async def test_updates_when_new_session_discovered_for_same_window(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.chat_first_command_path = False
        mock_provider.capabilities.name = "codex"
        event = SessionStartEvent(
            session_id="uuid-new",
            cwd="/my/project",
            transcript_path="/path/to/new.jsonl",
            window_key="ccgram:@7",
        )
        mock_provider.discover_transcript.return_value = event

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(
                    session_id="uuid-old",
                    cwd="/my/project",
                    transcript_path="/path/to/old.jsonl",
                    provider_name="codex",
                )
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bun")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            await discover_and_register_transcript("@7")

        mock_sms.register_hookless_session.assert_called_once_with(
            window_id="@7",
            session_id="uuid-new",
            cwd="/my/project",
            transcript_path="/path/to/new.jsonl",
            provider_name="codex",
        )

    async def test_noop_when_discovery_returns_none(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.chat_first_command_path = False
        mock_provider.discover_transcript.return_value = None

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="codex")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bun")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            await discover_and_register_transcript("@7")

        mock_sm.register_hookless_session.assert_not_called()
        mock_sm.write_hookless_session_map.assert_not_called()

    async def test_session_map_write_runs_in_background_thread(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.chat_first_command_path = False
        mock_provider.capabilities.name = "codex"
        event = SessionStartEvent(
            session_id="uuid-abc",
            cwd="/my/project",
            transcript_path="/path/to/transcript.jsonl",
            window_key="ccgram:@7",
        )
        mock_provider.discover_transcript.return_value = event

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.asyncio"
            ) as mock_asyncio,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/my/project", provider_name="codex")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_window = MagicMock(pane_current_command="bun")
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            mock_asyncio.to_thread = AsyncMock(side_effect=[event, None])
            await discover_and_register_transcript("@7")

        assert mock_asyncio.to_thread.call_count == 2
        discover_call = mock_asyncio.to_thread.call_args_list[0]
        assert discover_call.args[0] == mock_provider.discover_transcript
        write_call = mock_asyncio.to_thread.call_args_list[1]
        assert write_call.args[0] == mock_sms.write_hookless_session_map
        mock_sms.register_hookless_session.assert_called_once()

    async def test_tries_hookless_providers_when_provider_name_empty(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent

        event = SessionStartEvent(
            session_id="uuid-found",
            cwd="/proj",
            transcript_path="/path/to/transcript.jsonl",
            window_key="ccgram:@7",
        )

        mock_codex = MagicMock()
        mock_codex.capabilities.supports_hook = False
        mock_codex.capabilities.chat_first_command_path = False
        mock_codex.capabilities.name = "codex"
        mock_codex.discover_transcript.return_value = event

        mock_gemini = MagicMock()
        mock_gemini.capabilities.supports_hook = False
        mock_gemini.capabilities.chat_first_command_path = False
        mock_gemini.capabilities.name = "gemini"
        mock_gemini.discover_transcript.return_value = None

        mock_claude = MagicMock()
        mock_claude.capabilities.supports_hook = True
        mock_claude.capabilities.name = "claude"

        mock_registry = MagicMock()
        mock_registry.provider_names.return_value = ["claude", "codex", "gemini"]

        def mock_get(name: str) -> MagicMock:
            return {"claude": mock_claude, "codex": mock_codex, "gemini": mock_gemini}[
                name
            ]

        mock_registry.get = mock_get

        mock_window = MagicMock(pane_current_command="bun")

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch("ccgram.providers.registry", mock_registry),
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            await discover_and_register_transcript("@7")

        mock_sms.register_hookless_session.assert_called_once_with(
            window_id="@7",
            session_id="uuid-found",
            cwd="/proj",
            transcript_path="/path/to/transcript.jsonl",
            provider_name="codex",
        )

    async def test_tries_next_provider_when_session_conflicts_with_bound_window(
        self,
    ) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent

        conflicting_event = SessionStartEvent(
            session_id="uuid-in-use",
            cwd="/proj",
            transcript_path="/path/to/in-use.jsonl",
            window_key="ccgram:@7",
        )
        alternative_event = SessionStartEvent(
            session_id="uuid-new",
            cwd="/proj",
            transcript_path="/path/to/alt.jsonl",
            window_key="ccgram:@7",
        )

        mock_codex = MagicMock()
        mock_codex.capabilities.supports_hook = False
        mock_codex.capabilities.chat_first_command_path = False
        mock_codex.capabilities.name = "codex"
        mock_codex.discover_transcript.return_value = conflicting_event

        mock_gemini = MagicMock()
        mock_gemini.capabilities.supports_hook = False
        mock_gemini.capabilities.chat_first_command_path = False
        mock_gemini.capabilities.name = "gemini"
        mock_gemini.discover_transcript.return_value = alternative_event

        mock_registry = MagicMock()
        mock_registry.provider_names.return_value = ["codex", "gemini"]

        def mock_get(name: str) -> MagicMock:
            return {"codex": mock_codex, "gemini": mock_gemini}[name]

        mock_registry.get = mock_get
        mock_router = MagicMock()
        mock_router.iter_thread_bindings.return_value = [(123, 42, "@9")]

        mock_bound_state = MagicMock(
            session_id="uuid-in-use",
            cwd="/proj",
            provider_name="codex",
        )

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch("ccgram.providers.registry", mock_registry),
            patch("ccgram.thread_router.thread_router", mock_router),
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name=""),
                "@9": mock_bound_state,
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bun")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            await discover_and_register_transcript("@7")

        mock_codex.discover_transcript.assert_called_once()
        mock_gemini.discover_transcript.assert_called_once()
        mock_sms.register_hookless_session.assert_called_once_with(
            window_id="@7",
            session_id="uuid-new",
            cwd="/proj",
            transcript_path="/path/to/alt.jsonl",
            provider_name="gemini",
        )

    async def test_skips_hookless_fallback_when_pane_is_shell(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        mock_window = MagicMock(pane_current_command="bash")

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="")
            }
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            await discover_and_register_transcript("@7")

        mock_sm.register_hookless_session.assert_not_called()

    async def test_passes_max_age_zero_when_pane_is_alive(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.chat_first_command_path = False
        mock_provider.capabilities.name = "codex"
        mock_provider.discover_transcript.return_value = None

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.asyncio"
            ) as mock_asyncio,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="codex")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command="bun")
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            mock_asyncio.to_thread = AsyncMock(return_value=None)
            await discover_and_register_transcript("@7")

        discover_call = mock_asyncio.to_thread.call_args_list[0]
        assert discover_call.args[0] == mock_provider.discover_transcript
        assert discover_call.kwargs["max_age"] == 0

    async def test_passes_max_age_none_when_pane_not_alive(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = False
        mock_provider.capabilities.chat_first_command_path = False
        mock_provider.capabilities.name = "codex"
        mock_provider.discover_transcript.return_value = None

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.asyncio"
            ) as mock_asyncio,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="codex")
            }
            mock_config.tmux_session_name = "ccgram"
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            mock_asyncio.to_thread = AsyncMock(return_value=None)
            await discover_and_register_transcript("@7")

        discover_call = mock_asyncio.to_thread.call_args_list[0]
        assert discover_call.args[0] == mock_provider.discover_transcript
        assert discover_call.kwargs["max_age"] is None

    async def test_rebinds_stale_codex_window_to_gemini_from_pane_title(self) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent

        mock_codex = MagicMock()
        mock_codex.capabilities.supports_hook = False
        mock_codex.capabilities.chat_first_command_path = False
        mock_codex.capabilities.name = "codex"
        mock_codex.discover_transcript.return_value = None

        gemini_event = SessionStartEvent(
            session_id="gemini-uuid",
            cwd="/Users/alexei/Workspace/ccgram",
            transcript_path="/Users/alexei/.gemini/tmp/ccgram/chats/session.json",
            window_key="ccgram:@7",
        )
        mock_gemini = MagicMock()
        mock_gemini.capabilities.supports_hook = False
        mock_gemini.capabilities.chat_first_command_path = False
        mock_gemini.capabilities.name = "gemini"
        mock_gemini.discover_transcript.return_value = gemini_event

        mock_state = MagicMock(
            session_id="old-codex-id",
            cwd="/Users/alexei",
            transcript_path="/Users/alexei/.codex/sessions/old.jsonl",
            provider_name="codex",
        )

        def _provider_for_window(_wid: str, _name: str | None = None) -> MagicMock:
            if mock_state.provider_name == "gemini":
                return mock_gemini
            return mock_codex

        def _set_window_provider(
            window_id: str, provider_name: str, *, cwd: str | None = None
        ) -> None:
            assert window_id == "@7"
            mock_state.provider_name = provider_name
            if cwd:
                mock_state.cwd = cwd

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                side_effect=_provider_for_window,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="",
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
        ):
            mock_ws.window_states = {"@7": mock_state}
            mock_sm.set_window_provider.side_effect = _set_window_provider
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(
                    pane_current_command="bun",
                    cwd="/Users/alexei/Workspace/ccgram",
                )
            )
            mock_tmux.get_pane_title = AsyncMock(return_value="◇  Ready (ccbot)")
            mock_config.tmux_session_name = "ccgram"
            await discover_and_register_transcript("@7")

        mock_codex.discover_transcript.assert_not_called()
        mock_gemini.discover_transcript.assert_called_once()
        mock_sm.set_window_provider.assert_called_once_with(
            "@7",
            "gemini",
            cwd="/Users/alexei/Workspace/ccgram",
        )
        mock_sms.register_hookless_session.assert_called_once_with(
            window_id="@7",
            session_id="gemini-uuid",
            cwd="/Users/alexei/Workspace/ccgram",
            transcript_path="/Users/alexei/.gemini/tmp/ccgram/chats/session.json",
            provider_name="gemini",
        )

    async def test_rebinds_stale_claude_window_to_codex_from_transcript_path(
        self,
    ) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent

        codex_event = SessionStartEvent(
            session_id="codex-uuid",
            cwd="/Users/alexei/Workspace/ccgram",
            transcript_path="/Users/alexei/.codex/sessions/2026/03/23/test.jsonl",
            window_key="ccgram:@7",
        )
        mock_codex = MagicMock()
        mock_codex.capabilities.supports_hook = False
        mock_codex.capabilities.chat_first_command_path = False
        mock_codex.capabilities.name = "codex"
        mock_codex.discover_transcript.return_value = codex_event

        mock_claude = MagicMock()
        mock_claude.capabilities.supports_hook = True
        mock_claude.capabilities.name = "claude"

        mock_state = MagicMock(
            session_id="old-claude-id",
            cwd="/Users/alexei/Workspace/ccgram",
            transcript_path="/Users/alexei/.codex/sessions/old.jsonl",
            provider_name="claude",
        )

        def _provider_for_window(_wid: str, _name: str | None = None) -> MagicMock:
            if mock_state.provider_name == "codex":
                return mock_codex
            return mock_claude

        def _set_window_provider(
            window_id: str, provider_name: str, *, cwd: str | None = None
        ) -> None:
            assert window_id == "@7"
            mock_state.provider_name = provider_name
            if cwd:
                mock_state.cwd = cwd

        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,  # noqa: F841
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,  # noqa: F841
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
            ) as mock_sms,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                side_effect=_provider_for_window,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.detect_provider_from_pane",
                new_callable=AsyncMock,
                return_value="",
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.should_probe_pane_title_for_provider_detection",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.recovery.transcript_discovery.config"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
        ):
            mock_ws.window_states = {"@7": mock_state}
            mock_sm.set_window_provider.side_effect = _set_window_provider
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(
                    pane_current_command="node",
                    cwd="/Users/alexei/Workspace/ccgram",
                )
            )
            mock_config.tmux_session_name = "ccgram"
            await discover_and_register_transcript("@7")

        mock_sm.set_window_provider.assert_called_once_with(
            "@7",
            "codex",
            cwd="/Users/alexei/Workspace/ccgram",
        )
        mock_codex.discover_transcript.assert_called_once()
        mock_sms.register_hookless_session.assert_called_once_with(
            window_id="@7",
            session_id="codex-uuid",
            cwd="/Users/alexei/Workspace/ccgram",
            transcript_path="/Users/alexei/.codex/sessions/2026/03/23/test.jsonl",
            provider_name="codex",
        )


class TestDeadWindowNotification:
    async def test_marks_notified_even_when_send_fails(self) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.window_query") as mock_sm,
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.render_banner",
                return_value=("⚠ Session ended", None),
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_display_name.return_value = "test"
            mock_sm.view_window.return_value = MagicMock(cwd="/proj")
            await _handle_dead_window_notification(bot, 1, 42, "@5")

        assert (1, 42, "@5") in _dead_notified

    async def test_no_retry_after_failed_send(self) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.window_query") as mock_sm,
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_send,
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.render_banner",
                return_value=("⚠ Session ended", None),
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_display_name.return_value = "test"
            mock_sm.view_window.return_value = MagicMock(cwd="/proj")
            await _handle_dead_window_notification(bot, 1, 42, "@5")
            await _handle_dead_window_notification(bot, 1, 42, "@5")

        mock_send.assert_called_once()

    @pytest.mark.parametrize(
        "error_msg",
        [
            pytest.param("Message thread not found", id="capitalized"),
            pytest.param("message thread not found", id="lowercase"),
            pytest.param("Bad Request: Thread not found", id="thread-variant"),
        ],
    )
    async def test_probe_cleans_up_on_thread_not_found(self, error_msg: str) -> None:
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages.side_effect = BadRequest(error_msg)
        mock_window = MagicMock()
        mock_window.window_id = "@5"
        with (
            patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as mock_tr,
            patch("ccgram.handlers.topics.topic_lifecycle.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.topics.topic_lifecycle.clear_topic_state",
                new_callable=AsyncMock,
            ) as mock_cleanup,
        ):
            mock_tr.iter_thread_bindings.return_value = [(1, 42, "@5")]
            mock_tr.resolve_chat_id.return_value = -100
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.kill_window = AsyncMock()
            await probe_topic_existence(bot)

        mock_tm.kill_window.assert_not_called()
        mock_cleanup.assert_called_once_with(1, 42, bot, window_id="@5")
        mock_tr.unbind_thread.assert_called_once_with(1, 42)


class TestPaneAlertHelpers:
    def test_has_pane_alert_true_when_present(self) -> None:
        _pane_alert_hashes["%1"] = ("prompt text", 100.0, "@0")
        assert interactive_strategy.has_pane_alert("%1") is True

    def test_has_pane_alert_false_when_absent(self) -> None:
        assert interactive_strategy.has_pane_alert("%99") is False

    def test_clear_pane_alerts_removes_for_window(self) -> None:
        _pane_alert_hashes["%1"] = ("prompt A", 100.0, "@0")
        _pane_alert_hashes["%2"] = ("prompt B", 100.0, "@0")
        _pane_alert_hashes["%3"] = ("prompt C", 100.0, "@5")
        interactive_strategy.clear_pane_alerts("@0")
        assert "%1" not in _pane_alert_hashes
        assert "%2" not in _pane_alert_hashes
        assert "%3" in _pane_alert_hashes


def _make_pane(pane_id: str = "%1", *, active: bool = True, index: int = 0) -> PaneInfo:
    return PaneInfo(
        pane_id=pane_id,
        index=index,
        active=active,
        command="claude",
        path="/tmp",
        width=80,
        height=24,
    )


class TestScanWindowPanes:
    async def test_skips_single_pane_window(self) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.tmux_manager.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            mock_tm.list_panes = AsyncMock(return_value=[_make_pane()])
            await _scan_window_panes(bot, 1, "@0", 42)
        mock_handle.assert_not_called()

    async def test_detects_interactive_prompt_in_non_active_pane(self) -> None:
        from ccgram.providers.base import StatusUpdate

        bot = AsyncMock(spec=Bot)
        interactive = StatusUpdate(
            raw_text="Allow?",
            display_label="Allow?",
            is_interactive=True,
            ui_type="PermissionPrompt",
        )
        mock_provider = MagicMock()
        mock_provider.parse_terminal_status.return_value = interactive
        with (
            patch("ccgram.tmux_manager.tmux_manager") as mock_tm,
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[_make_pane(), _make_pane("%2", active=False, index=1)]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="Allow?\nEsc\n")
            await _scan_window_panes(bot, 1, "@0", 42)
        _assert_handle_called_once_with_client(
            mock_handle, bot, 1, "@0", 42, pane_id="%2"
        )

    async def test_skips_active_pane(self) -> None:
        bot = AsyncMock(spec=Bot)
        mock_provider = MagicMock()
        mock_provider.parse_terminal_status.return_value = None
        with (
            patch("ccgram.tmux_manager.tmux_manager") as mock_tm,
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[_make_pane(), _make_pane("%2", active=False, index=1)]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="some text")
            await _scan_window_panes(bot, 1, "@0", 42)
        mock_handle.assert_not_called()
        mock_tm.capture_pane_by_id.assert_called_once_with("%2", window_id="@0")

    async def test_deduplicates_same_prompt(self) -> None:
        from ccgram.providers.base import StatusUpdate

        bot = AsyncMock(spec=Bot)
        interactive = StatusUpdate(
            raw_text="Allow write?",
            display_label="Allow write?",
            is_interactive=True,
            ui_type="PermissionPrompt",
        )
        mock_provider = MagicMock()
        mock_provider.parse_terminal_status.return_value = interactive
        with (
            patch("ccgram.tmux_manager.tmux_manager") as mock_tm,
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[_make_pane(), _make_pane("%2", active=False, index=1)]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="Allow write?\nEsc\n")
            await _scan_window_panes(bot, 1, "@0", 42)
            await _scan_window_panes(bot, 1, "@0", 42)
        mock_handle.assert_called_once()

    async def test_clears_stale_alert_when_pane_disappears(self) -> None:
        _pane_alert_hashes["%2"] = ("old prompt", 100.0, "@0")
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.tmux_manager.tmux_manager") as mock_tm:
            mock_tm.list_panes = AsyncMock(return_value=[_make_pane()])
            await _scan_window_panes(bot, 1, "@0", 42)
        assert "%2" not in _pane_alert_hashes

    async def test_clears_alert_when_interactive_ui_gone(self) -> None:
        _pane_alert_hashes["%2"] = ("old prompt", 100.0, "@0")
        bot = AsyncMock(spec=Bot)
        mock_provider = MagicMock()
        mock_provider.parse_terminal_status.return_value = None
        with (
            patch("ccgram.tmux_manager.tmux_manager") as mock_tm,
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[_make_pane(), _make_pane("%2", active=False, index=1)]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="normal output")
            await _scan_window_panes(bot, 1, "@0", 42)
        assert "%2" not in _pane_alert_hashes
        mock_handle.assert_not_called()

    async def test_cached_pane_count_skips_subprocess(self) -> None:
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.tmux_manager.tmux_manager") as mock_tm:
            mock_tm.list_panes = AsyncMock(return_value=[_make_pane()])
            await _scan_window_panes(bot, 1, "@0", 42)
            await _scan_window_panes(bot, 1, "@0", 42)
        mock_tm.list_panes.assert_called_once()


@pytest.mark.usefixtures("_reset_pyte")
class TestUpdateStatusMessageEdgeCases:
    async def test_window_gone_enqueues_clear(self) -> None:
        from ccgram.handlers.polling.window_tick import (
            _update_status as update_status_message,
        )

        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.enqueue_status_update",
                new_callable=AsyncMock,
            ) as mock_enqueue,
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            await update_status_message(bot, 1, "@0", thread_id=42)
        mock_enqueue.assert_called_once_with(ANY, 1, "@0", None, thread_id=42)

    async def test_empty_capture_keeps_existing_status(self) -> None:
        from ccgram.handlers.polling.window_tick import (
            _update_status as update_status_message,
        )

        bot = AsyncMock(spec=Bot)
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.pane_width = 80
        mock_window.pane_height = 24
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.enqueue_status_update",
                new_callable=AsyncMock,
            ) as mock_enqueue,
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value=None)
            await update_status_message(bot, 1, "@0", thread_id=42)
        mock_enqueue.assert_not_called()

    async def test_vim_insert_detected_from_rendered_text(self) -> None:
        from ccgram.handlers.polling.window_tick import (
            _update_status as update_status_message,
        )
        from ccgram.providers.base import StatusUpdate

        terminal_poll_state.get_state(
            "@0"
        ).last_rendered_text = "some code\n-- INSERT --\n"
        pyte_status = StatusUpdate(raw_text="Working", display_label="...working")
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.pane_width = 80
        mock_window.pane_height = 24
        mock_window.pane_current_command = "node"
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch("ccgram.handlers.polling.window_tick.apply.update_topic_emoji"),
            patch("ccgram.handlers.polling.window_tick.apply.enqueue_status_update"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=pyte_status,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe.notify_vim_insert_seen"
            ) as mock_vim,
            patch(
                "ccgram.handlers.polling.window_tick.observe.has_insert_indicator",
                return_value=True,
            ),
            patch("ccgram.handlers.polling.window_tick.apply._send_typing_throttled"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_subagent_names",
                return_value=[],
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="\x1b[1mansi\x1b[0m")
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_display_name.return_value = "project"
            await update_status_message(bot, 1, "@0", thread_id=42)
        mock_vim.assert_called_once_with("@0")

    async def test_status_includes_subagent_names(self) -> None:
        from ccgram.handlers.polling.window_tick import (
            _update_status as update_status_message,
        )
        from ccgram.providers.base import StatusUpdate

        pyte_status = StatusUpdate(
            raw_text="Working", display_label="\u23f3 Working\u2026"
        )
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.pane_width = 80
        mock_window.pane_height = 24
        mock_window.pane_current_command = "node"
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch("ccgram.handlers.polling.window_tick.apply.update_topic_emoji"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.enqueue_status_update",
                new_callable=AsyncMock,
            ) as mock_enqueue,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=pyte_status,
            ),
            patch("ccgram.tmux_manager.has_insert_indicator", return_value=False),
            patch("ccgram.tmux_manager.notify_vim_insert_seen"),
            patch("ccgram.handlers.polling.window_tick.apply._send_typing_throttled"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_subagent_names",
                return_value=["write-tests"],
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="some output")
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_display_name.return_value = "project"

            await update_status_message(bot, 1, "@0", thread_id=42)
        status_text = mock_enqueue.call_args[0][3]
        assert "write-tests" in status_text
        assert "\U0001f916" in status_text

    async def test_status_prefers_multiline_raw_task_block(self) -> None:
        from ccgram.handlers.polling.window_tick import (
            _update_status as update_status_message,
        )
        from ccgram.providers.base import StatusUpdate

        pyte_status = StatusUpdate(
            raw_text=(
                "Running py-idioms review…\n"
                "✔ Detect languages and scope\n"
                "◼ Spawn review agents\n"
                "◻ Collect agent results"
            ),
            display_label="\u26a1 running\u2026",
        )
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.pane_width = 80
        mock_window.pane_height = 24
        mock_window.pane_current_command = "node"
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch("ccgram.handlers.polling.window_tick.apply.update_topic_emoji"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.enqueue_status_update",
                new_callable=AsyncMock,
            ) as mock_enqueue,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=pyte_status,
            ),
            patch("ccgram.tmux_manager.has_insert_indicator", return_value=False),
            patch("ccgram.tmux_manager.notify_vim_insert_seen"),
            patch("ccgram.handlers.polling.window_tick.apply._send_typing_throttled"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_subagent_names",
                return_value=[],
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="some output")
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_display_name.return_value = "project"

            await update_status_message(bot, 1, "@0", thread_id=42)
        status_text = mock_enqueue.call_args[0][3]
        assert status_text.startswith("Running py-idioms review…")
        assert "✔ Detect languages and scope" in status_text
        assert "◻ Collect agent results" in status_text

    async def test_interactive_window_clears_when_ui_disappears(self) -> None:
        from ccgram.handlers.polling.window_tick import (
            _update_status as update_status_message,
        )
        from ccgram.providers.base import StatusUpdate

        non_interactive = StatusUpdate(raw_text="Working", display_label="...working")
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.pane_width = 80
        mock_window.pane_height = 24
        mock_window.pane_current_command = "node"
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch("ccgram.handlers.polling.window_tick.apply.update_topic_emoji"),
            patch("ccgram.handlers.polling.window_tick.apply.enqueue_status_update"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value="@0",
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=non_interactive,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.clear_interactive_msg",
                new_callable=AsyncMock,
            ) as mock_clear,
            patch("ccgram.tmux_manager.has_insert_indicator", return_value=False),
            patch("ccgram.tmux_manager.notify_vim_insert_seen"),
            patch("ccgram.handlers.polling.window_tick.apply._send_typing_throttled"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_subagent_names",
                return_value=[],
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="some output")
            mock_tr.resolve_chat_id.return_value = -100
            mock_tr.get_display_name.return_value = "project"

            await update_status_message(bot, 1, "@0", thread_id=42)
        _assert_clear_called_once_with_client(mock_clear, 1, bot, 42)

    async def test_new_interactive_ui_enters_interactive_mode(self) -> None:
        from ccgram.handlers.polling.window_tick import (
            _update_status as update_status_message,
        )
        from ccgram.providers.base import StatusUpdate

        interactive_status = StatusUpdate(
            raw_text="Allow?",
            display_label="Allow?",
            is_interactive=True,
            ui_type="PermissionPrompt",
        )
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.pane_width = 80
        mock_window.pane_height = 24
        mock_window.pane_current_command = "node"
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.update_topic_emoji"),
            patch("ccgram.handlers.polling.window_tick.apply.enqueue_status_update"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=interactive_status,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
            patch("ccgram.tmux_manager.has_insert_indicator", return_value=False),
            patch("ccgram.tmux_manager.notify_vim_insert_seen"),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tm.capture_pane = AsyncMock(return_value="Allow?\nEsc\n")
            await update_status_message(bot, 1, "@0", thread_id=42)
        _assert_handle_called_once_with_client(mock_handle, bot, 1, "@0", 42)


@pytest.mark.usefixtures("_reset_pyte")
class TestCheckInteractiveOnly:
    @pytest.mark.parametrize(
        "interactive_window",
        [
            pytest.param(None, id="no_active_ui"),
            pytest.param("@1", id="different_window_active"),
        ],
    )
    async def test_detects_interactive_ui(self, interactive_window: str | None) -> None:
        from ccgram.handlers.polling.window_tick import _check_interactive_only
        from ccgram.providers.base import StatusUpdate

        interactive_status = StatusUpdate(
            raw_text="Allow?",
            display_label="Allow?",
            is_interactive=True,
            ui_type="PermissionPrompt",
        )
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.pane_width = 80
        mock_window.pane_height = 24
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=interactive_window,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=interactive_status,
            ) as mock_pyte,
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
            patch(
                "ccgram.handlers.polling.window_tick.apply.set_interactive_mode",
            ) as mock_set,
        ):
            mock_tm.capture_pane = AsyncMock(return_value="Allow?\nEsc\n")
            await _check_interactive_only(bot, 1, "@0", 42, _window=mock_window)
        mock_pyte.assert_called_once_with(
            "@0", "Allow?\nEsc\n", columns=80, rows=24, parse_claude_chrome=True
        )
        mock_set.assert_called_once_with(1, "@0", 42)
        _assert_handle_called_once_with_client(mock_handle, bot, 1, "@0", 42)

    async def test_clears_interactive_mode_on_handle_failure(self) -> None:
        from ccgram.handlers.polling.window_tick import _check_interactive_only
        from ccgram.providers.base import StatusUpdate

        interactive_status = StatusUpdate(
            raw_text="Allow?",
            display_label="Allow?",
            is_interactive=True,
            ui_type="PermissionPrompt",
        )
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.pane_width = 80
        mock_window.pane_height = 24
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=interactive_status,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.set_interactive_mode",
            ) as mock_set,
            patch(
                "ccgram.handlers.polling.window_tick.apply.clear_interactive_mode",
            ) as mock_clear,
        ):
            mock_tm.capture_pane = AsyncMock(return_value="Allow?\nEsc\n")
            await _check_interactive_only(bot, 1, "@0", 42, _window=mock_window)
        mock_set.assert_called_once_with(1, "@0", 42)
        mock_clear.assert_called_once_with(1, 42)

    async def test_skips_when_already_interactive(self) -> None:
        from ccgram.handlers.polling.window_tick import _check_interactive_only

        mock_window = MagicMock()
        mock_window.window_id = "@0"
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value="@0",
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
            ) as mock_pyte,
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            mock_tm.capture_pane = AsyncMock()
            await _check_interactive_only(bot, 1, "@0", 42, _window=mock_window)
        mock_tm.capture_pane.assert_not_called()
        mock_pyte.assert_not_called()
        mock_handle.assert_not_called()

    async def test_no_action_when_not_interactive(self) -> None:
        from ccgram.handlers.polling.window_tick import _check_interactive_only
        from ccgram.providers.base import StatusUpdate

        normal_status = StatusUpdate(
            raw_text="Reading file", display_label="reading..."
        )
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.pane_width = 80
        mock_window.pane_height = 24
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=normal_status,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            mock_tm.capture_pane = AsyncMock(return_value="some output")
            await _check_interactive_only(bot, 1, "@0", 42, _window=mock_window)
        mock_handle.assert_not_called()

    async def test_no_action_when_window_gone(self) -> None:
        from ccgram.handlers.polling.window_tick import _check_interactive_only

        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            await _check_interactive_only(bot, 1, "@0", 42)
        mock_handle.assert_not_called()

    async def test_no_action_on_empty_capture(self) -> None:
        from ccgram.handlers.polling.window_tick import _check_interactive_only

        mock_window = MagicMock()
        mock_window.window_id = "@0"
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
            ) as mock_pyte,
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            mock_tm.capture_pane = AsyncMock(return_value="")
            await _check_interactive_only(bot, 1, "@0", 42, _window=mock_window)
        mock_pyte.assert_not_called()
        mock_handle.assert_not_called()

    @pytest.mark.parametrize(
        ("uses_pane_title", "expected_title"),
        [
            pytest.param(False, "", id="no_pane_title"),
            pytest.param(True, "gemini-title", id="with_pane_title"),
        ],
    )
    async def test_falls_back_to_provider_regex(
        self, uses_pane_title: bool, expected_title: str
    ) -> None:
        from ccgram.handlers.polling.window_tick import _check_interactive_only
        from ccgram.providers.base import StatusUpdate

        interactive_status = StatusUpdate(
            raw_text="Allow?",
            display_label="Allow?",
            is_interactive=True,
            ui_type="PermissionPrompt",
        )
        mock_provider = MagicMock()
        mock_provider.capabilities.uses_pane_title = uses_pane_title
        mock_provider.parse_terminal_status.return_value = interactive_status
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.pane_width = 80
        mock_window.pane_height = 24
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.polling.window_tick.observe.tmux_manager"
            ) as observe_tm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe.get_provider_for_window",
                return_value=mock_provider,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_handle,
            patch("ccgram.handlers.polling.window_tick.apply.set_interactive_mode"),
        ):
            mock_tm.capture_pane = AsyncMock(return_value="Allow?\nEsc\n")
            observe_tm.get_pane_title = AsyncMock(return_value="gemini-title")
            await _check_interactive_only(bot, 1, "@0", 42, _window=mock_window)
        mock_provider.parse_terminal_status.assert_called_once_with(
            "Allow?\nEsc\n", pane_title=expected_title
        )
        _assert_handle_called_once_with_client(mock_handle, bot, 1, "@0", 42)
        if uses_pane_title:
            observe_tm.get_pane_title.assert_called_once_with("@0")
