import time

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from telegram import Bot
from telegram.error import BadRequest, TelegramError

from ccgram.handlers.topics.topic_lifecycle import (
    check_autoclose_timers,
    probe_topic_existence,
    prune_stale_state,
    reset_probe_schedule,
)
from ccgram.handlers.polling.window_tick import (
    _check_interactive_only,
    _handle_dead_window_notification,
    _parse_with_pyte,
    _scan_window_panes,
    _update_status,
    decide_tick,
)
from ccgram.handlers.polling.polling_runtime import PollingRuntime
from ccgram.handlers.polling.polling_state import (
    interactive_strategy,
    lifecycle_strategy,
    terminal_poll_state,
    terminal_screen_buffer,
)
from ccgram.handlers.polling.polling_types import MAX_PROBE_FAILURES, TickContext
from ccgram.providers.base import StatusUpdate
from ccgram.telegram_client import PTBTelegramClient
from ccgram.multiplexer.base import ForegroundInfo, PaneInfo

_INTERACTIVE_STATUS = StatusUpdate(
    raw_text="Allow?",
    display_label="Allow?",
    is_interactive=True,
    ui_type="PermissionPrompt",
)


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


def _has_autoclose(user_id: int, thread_id: int) -> bool:
    ts = _topic_poll_state.get((user_id, thread_id))
    return ts is not None and ts.autoclose is not None


@pytest.fixture(autouse=True)
def _reset():
    reset_probe_schedule()
    _window_poll_state.clear()
    _topic_poll_state.clear()
    _dead_notified.clear()
    yield
    reset_probe_schedule()
    _window_poll_state.clear()
    _topic_poll_state.clear()
    _dead_notified.clear()


class TestAutocloseTimers:
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
        bot.close_forum_topic.assert_called_once_with(
            chat_id=-100, message_thread_id=42
        )
        bot.delete_forum_topic.assert_not_called()
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

    async def test_check_telegram_error_does_not_clear_timer(self) -> None:
        """A non-fatal TelegramError leaves the timer so the next cycle retries."""
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
        # Timer must stay so the next cycle can retry.
        assert _has_autoclose(1, 42)

    async def test_check_treats_missing_topic_as_removed(self) -> None:
        """A gone topic is cleaned up whether close_forum_topic says it's gone."""
        _start_autoclose_timer(1, 42, "done", 0.0)
        bot = AsyncMock(spec=Bot)
        bot.close_forum_topic.side_effect = BadRequest("Topic_id_invalid")
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

        bot.close_forum_topic.assert_called_once_with(
            chat_id=-100, message_thread_id=42
        )
        bot.delete_forum_topic.assert_not_called()
        mock_tr.unbind_thread.assert_called_once_with(1, 42)
        mock_clear.assert_awaited_once()
        assert not _has_autoclose(1, 42)


class TestTranscriptActivityHeuristic:
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
    startup_quietly_settled: bool = False,
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
        startup_quietly_settled=startup_quietly_settled,
        is_dead_window=is_dead_window,
        supports_hook=supports_hook,
    )


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


_APPLY = "ccgram.handlers.polling.window_tick.apply."
_OBSERVE = "ccgram.handlers.polling.window_tick.observe."
_UNSET = object()


def _tick_window(window_id: str = "@0", *, pane_current_command: str = "node"):
    w = MagicMock()
    w.window_id = window_id
    w.window_name = "project"
    w.pane_current_command = pane_current_command
    w.pane_width = 80
    w.pane_height = 24
    return w


def _provider(
    *,
    terminal_status: StatusUpdate | None = None,
    uses_pane_title: bool = False,
    uses_pyte_status_parsing: bool = True,
) -> MagicMock:
    provider = MagicMock()
    provider.capabilities.uses_pane_title = uses_pane_title
    provider.capabilities.uses_pyte_status_parsing = uses_pyte_status_parsing
    provider.parse_terminal_status.return_value = terminal_status
    return provider


@contextmanager
def _tick_env(
    *,
    window: Any = _UNSET,
    capture: str | None = "some output",
    pyte_result: StatusUpdate | None = None,
    provider: MagicMock | None = None,
    interactive_window: str | None = None,
    subagents: tuple[str, ...] = (),
):
    """Patch every collaborator ``_update_status``/``_check_interactive_only`` touch.

    ``observe`` binds ``tmux_manager``, ``get_provider_for_window`` and the vim
    helpers at import time, so those need patching separately from ``apply``'s.
    """
    provider = provider if provider is not None else _provider()
    window = _tick_window() if window is _UNSET else window

    with ExitStack() as stack:

        def _patch(target: str, **kwargs: Any) -> MagicMock:
            return stack.enter_context(patch(target, **kwargs))

        mocks = SimpleNamespace()
        mocks.provider = provider
        mocks.window = window

        mocks.tmux = _patch(_APPLY + "tmux_manager")
        mocks.tmux.find_window_by_id = AsyncMock(return_value=window)
        mocks.tmux.capture_pane = AsyncMock(return_value=capture)
        mocks.thread_router = _patch(_APPLY + "thread_router")
        mocks.thread_router.resolve_chat_id.return_value = -100
        mocks.thread_router.get_display_name.return_value = "project"
        mocks.enqueue = _patch(_APPLY + "enqueue_status_update", new_callable=AsyncMock)
        mocks.handle_ui = _patch(
            _APPLY + "handle_interactive_ui", new_callable=AsyncMock
        )
        mocks.clear_interactive_msg = _patch(
            _APPLY + "clear_interactive_msg", new_callable=AsyncMock
        )
        mocks.set_interactive_mode = _patch(_APPLY + "set_interactive_mode")
        mocks.clear_interactive_mode = _patch(_APPLY + "clear_interactive_mode")
        _patch(_APPLY + "window_query")
        _patch(_APPLY + "update_topic_emoji")
        _patch(_APPLY + "_send_typing_throttled")
        _patch(_APPLY + "get_interactive_window", return_value=interactive_window)
        _patch(_APPLY + "get_subagent_names", return_value=list(subagents))
        _patch(_APPLY + "get_provider_for_window", return_value=provider)

        mocks.observe_tmux = _patch(_OBSERVE + "tmux_manager")
        mocks.observe_tmux.get_pane_title = AsyncMock(return_value="")
        mocks.observe_tmux.capabilities.native_agent_status = False
        mocks.pyte = _patch(_OBSERVE + "_parse_with_pyte", return_value=pyte_result)
        mocks.vim_notify = _patch(_OBSERVE + "notify_vim_insert_seen")
        _patch(_OBSERVE + "get_provider_for_window", return_value=provider)
        _patch(_OBSERVE + "window_query")

        yield mocks


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


class TestQuietStartupSettlement:
    async def test_startup_expiry_stays_quiet_and_runs_idle_cleanup(self) -> None:
        """A dormant window must never turn its quiet expiry into Ready later."""
        from ccgram.handlers.polling.window_tick import _apply_tick_decision

        runtime = PollingRuntime.create()
        runtime.poll_state.begin_startup_timer("@0", time.monotonic() - 31.0)
        runtime.lifecycle.start_autoclose_timer(1, 42, "done", 0.0)
        runtime.lifecycle.record_typing_sent(1, 42)
        bot = AsyncMock(spec=Bot)

        first = decide_tick(
            _make_ctx(
                startup_time=runtime.poll_state.get_state("@0").startup_time,
                startup_quietly_settled=False,
            )
        )
        assert first.transition == "idle"
        assert first.send_status is False

        with (
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji",
                new_callable=AsyncMock,
            ) as mock_emoji,
            patch(
                "ccgram.handlers.polling.window_tick.apply.enqueue_status_update",
                new_callable=AsyncMock,
            ) as mock_enqueue,
            patch(
                "ccgram.handlers.polling.window_tick.apply.thread_router"
            ) as mock_router,
        ):
            mock_router.resolve_chat_id.return_value = -100
            mock_router.get_display_name.return_value = "project"
            await _apply_tick_decision(bot, 1, "@0", 42, first, runtime=runtime)

        state = runtime.poll_state.get_state("@0")
        assert state.startup_time is None
        assert state.has_seen_status is False
        assert state.startup_quietly_settled is True
        assert runtime.lifecycle.get_state(1, 42).autoclose is None
        assert runtime.lifecycle.get_state(1, 42).last_typing_sent is None
        mock_emoji.assert_awaited_once()
        mock_enqueue.assert_not_awaited()

        second = decide_tick(
            _make_ctx(
                startup_time=state.startup_time,
                has_seen_status=state.has_seen_status,
                startup_quietly_settled=state.startup_quietly_settled,
            )
        )
        assert second.transition == "idle"
        assert second.send_status is False

    def test_real_status_clears_quiet_settlement(self) -> None:
        runtime = PollingRuntime.create()
        runtime.poll_state.mark_startup_quietly_settled("@0")

        runtime.poll_state.mark_seen_status("@0")

        state = runtime.poll_state.get_state("@0")
        assert state.has_seen_status is True
        assert state.startup_quietly_settled is False


class TestSettledWindowsStaySettled:
    """A quiet window must not fall back into the startup grace.

    ``starting`` paints the topic active and sends a typing indicator, so a
    window that keeps re-entering it looks busy forever: green topic, typing
    indicator that never clears, and a 30s sawtooth back through idle.
    """

    async def _run(self, transition: str, *, supports_hook: bool) -> str | None:
        from ccgram.handlers.polling.window_tick import (
            _apply_done_transition,
            _transition_to_idle,
        )

        runtime = PollingRuntime.create()
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.update_topic_emoji"),
            patch("ccgram.handlers.polling.window_tick.apply.enqueue_status_update"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.thread_router"
            ) as mock_router,
        ):
            mock_router.resolve_chat_id.return_value = -100
            mock_router.get_display_name.return_value = "project"
            if transition == "idle":
                await _transition_to_idle(
                    bot, 1, "@0", 42, -100, "project", runtime=runtime
                )
            else:
                await _apply_done_transition(bot, 1, "@0", 42, runtime=runtime)

        poll_state = runtime.poll_state.peek_state("@0")
        assert poll_state is not None
        return decide_tick(
            _make_ctx(
                has_seen_status=runtime.poll_state.check_seen_status("@0"),
                startup_time=poll_state.startup_time,
                supports_hook=supports_hook,
            )
        ).transition

    async def test_idle_window_stays_idle_on_the_next_tick(self) -> None:
        assert await self._run("idle", supports_hook=True) == "idle"

    async def test_done_window_does_not_restart_its_grace(self) -> None:
        assert await self._run("done", supports_hook=True) == "idle"

    async def test_hookless_done_window_also_settles(self) -> None:
        assert await self._run("done", supports_hook=False) == "idle"


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
                # Stands in for PROBE_INTERVAL elapsing between poll cycles.
                reset_probe_schedule()
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
        mock_cleanup.assert_called_once_with(1, 42, bot, window_id="@5", chat_id=-100)
        mock_tr.unbind_thread.assert_called_once_with(1, 42, chat_id=-100)
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
    @pytest.mark.parametrize("pane_command", ["fish", ""])
    async def test_agent_origin_returning_to_shell_requests_recovery(
        self, pane_command: str
    ) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )

        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.recovery.transcript_discovery.session_manager"
            ) as mock_sm,
            patch("ccgram.window_state_ports.identity_state.window_store") as mock_ws,
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
                    initial_provider_name="claude",
                    transcript_path="",
                )
            }
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(pane_current_command=pane_command, cwd="/proj")
            )
            exited = await discover_and_register_transcript(
                "@7", client=bot, user_id=1, thread_id=42
            )

        assert exited is True
        mock_sm.set_window_provider.assert_not_called()
        mock_ensure.assert_not_awaited()

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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
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
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
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
            mock_config.return_value = "ccgram:"
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
            cwd="/proj",
            provider_name="claude",
            initial_provider_name="shell",
            transcript_path="/tx/claude.jsonl",
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
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
            mock_config.return_value = "ccgram:"

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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
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
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
            ) as mock_config,
        ):
            mock_ws.window_states = {"@7": mock_state}
            mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
            mock_tmux.get_pane_title = AsyncMock(return_value="")
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/my/project", provider_name="codex")
            }
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
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
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
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
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
            ) as mock_config,
            patch(
                "ccgram.handlers.recovery.transcript_discovery.tmux_manager"
            ) as mock_tmux,
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="codex")
            }
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
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
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
            ) as mock_config,
            patch("ccgram.providers.registry", mock_registry),
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name="")
            }
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
            ) as mock_config,
            patch("ccgram.providers.registry", mock_registry),
            patch("ccgram.thread_router.thread_router", mock_router),
        ):
            mock_ws.window_states = {
                "@7": MagicMock(session_id="", cwd="/proj", provider_name=""),
                "@9": mock_bound_state,
            }
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
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
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
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
            mock_config.return_value = "ccgram:"
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            mock_asyncio.to_thread = AsyncMock(return_value=None)
            await discover_and_register_transcript("@7")

        discover_call = mock_asyncio.to_thread.call_args_list[0]
        assert discover_call.args[0] == mock_provider.discover_transcript
        assert discover_call.kwargs["max_age"] is None

    async def test_rebinds_hookful_provider_when_foreground_agent_process_restarted(
        self,
    ) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent
        from ccgram.providers.process_detection import _pgid_cache

        event = SessionStartEvent(
            session_id="new-codex-id",
            cwd="/proj",
            transcript_path="/path/to/new-codex.jsonl",
            window_key="ccgram:@7",
        )
        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = True
        mock_provider.capabilities.chat_first_command_path = False
        mock_provider.capabilities.name = "codex"
        mock_provider.discover_transcript.return_value = event

        old_state = MagicMock(
            session_id="old-codex-id",
            cwd="/proj",
            transcript_path="/path/to/old-codex.jsonl",
            provider_name="codex",
        )

        _pgid_cache.clear()
        _pgid_cache["@7"] = (111, "codex")
        try:
            with (
                patch(
                    "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
                ) as mock_sms,
                patch(
                    "ccgram.window_state_ports.identity_state.window_store"
                ) as mock_ws,  # noqa: F841
                patch(
                    "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                    return_value=mock_provider,
                ),
                patch(
                    "ccgram.handlers.recovery.transcript_discovery.session_map_prefix",
                    return_value="ccgram:",
                ),
                patch("ccgram.multiplexer.multiplexer") as mock_mux,
            ):
                mock_ws.window_states = {"@7": old_state}
                mock_mux.foreground = AsyncMock(
                    return_value=ForegroundInfo(
                        pid=222,
                        pgid=222,
                        argv=["node", "/opt/@openai/codex/bin/codex.js"],
                        cwd="/proj",
                    )
                )

                await discover_and_register_transcript(
                    "@7",
                    _window=MagicMock(
                        pane_current_command="node",
                        pane_tty="/dev/ttys007",
                        cwd="/proj",
                    ),
                )
        finally:
            _pgid_cache.clear()

        mock_provider.discover_transcript.assert_called_once()
        assert mock_provider.discover_transcript.call_args.kwargs["max_age"] == 0
        mock_sms.register_hookless_session.assert_called_once_with(
            window_id="@7",
            session_id="new-codex-id",
            cwd="/proj",
            transcript_path="/path/to/new-codex.jsonl",
            provider_name="codex",
        )

    async def test_does_not_rebind_hookful_provider_when_agent_process_unchanged(
        self,
    ) -> None:
        from ccgram.handlers.recovery.transcript_discovery import (
            discover_and_register_transcript,
        )
        from ccgram.providers.base import SessionStartEvent
        from ccgram.providers.process_detection import _pgid_cache

        event = SessionStartEvent(
            session_id="new-codex-id",
            cwd="/proj",
            transcript_path="/path/to/new-codex.jsonl",
            window_key="ccgram:@7",
        )
        mock_provider = MagicMock()
        mock_provider.capabilities.supports_hook = True
        mock_provider.capabilities.chat_first_command_path = False
        mock_provider.capabilities.name = "codex"
        mock_provider.discover_transcript.return_value = event

        old_state = MagicMock(
            session_id="old-codex-id",
            cwd="/proj",
            transcript_path="/path/to/old-codex.jsonl",
            provider_name="codex",
        )

        _pgid_cache.clear()
        _pgid_cache["@7"] = (111, "codex")
        try:
            with (
                patch(
                    "ccgram.handlers.recovery.transcript_discovery.session_map_sync"
                ) as mock_sms,
                patch(
                    "ccgram.window_state_ports.identity_state.window_store"
                ) as mock_ws,  # noqa: F841
                patch(
                    "ccgram.handlers.recovery.transcript_discovery.get_provider_for_window",
                    return_value=mock_provider,
                ),
                patch(
                    "ccgram.handlers.recovery.transcript_discovery.session_map_prefix",
                    return_value="ccgram:",
                ),
                patch("ccgram.multiplexer.multiplexer") as mock_mux,
            ):
                mock_ws.window_states = {"@7": old_state}
                mock_mux.foreground = AsyncMock(
                    return_value=ForegroundInfo(
                        pid=111,
                        pgid=111,
                        argv=["node", "/opt/@openai/codex/bin/codex.js"],
                        cwd="/proj",
                    )
                )

                await discover_and_register_transcript(
                    "@7",
                    _window=MagicMock(
                        pane_current_command="node",
                        pane_tty="/dev/ttys007",
                        cwd="/proj",
                    ),
                )
        finally:
            _pgid_cache.clear()

        mock_provider.discover_transcript.assert_not_called()
        mock_sms.register_hookless_session.assert_not_called()

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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
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
            mock_tmux.get_pane_title = AsyncMock(return_value="◇  Ready (ccgram)")
            mock_config.return_value = "ccgram:"
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
                "ccgram.handlers.recovery.transcript_discovery.session_map_prefix"
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
            mock_config.return_value = "ccgram:"
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
        mock_cleanup.assert_called_once_with(1, 42, bot, window_id="@5", chat_id=-100)
        mock_tr.unbind_thread.assert_called_once_with(1, 42, chat_id=-100)


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
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
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
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
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
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
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
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
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
        with patch("ccgram.multiplexer.multiplexer") as mock_tm:
            mock_tm.list_panes = AsyncMock(return_value=[_make_pane()])
            await _scan_window_panes(bot, 1, "@0", 42)
        assert "%2" not in _pane_alert_hashes

    async def test_clears_alert_when_interactive_ui_gone(self) -> None:
        _pane_alert_hashes["%2"] = ("old prompt", 100.0, "@0")
        bot = AsyncMock(spec=Bot)
        mock_provider = MagicMock()
        mock_provider.parse_terminal_status.return_value = None
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
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
        with patch("ccgram.multiplexer.multiplexer") as mock_tm:
            mock_tm.list_panes = AsyncMock(return_value=[_make_pane()])
            await _scan_window_panes(bot, 1, "@0", 42)
            await _scan_window_panes(bot, 1, "@0", 42)
        mock_tm.list_panes.assert_called_once()


@pytest.mark.usefixtures("_reset_pyte")
class TestUpdateStatusMessage:
    async def test_window_gone_enqueues_clear(self) -> None:
        bot = AsyncMock(spec=Bot)
        with _tick_env(window=None) as env:
            await _update_status(bot, 1, "@0", thread_id=42)
        env.enqueue.assert_called_once_with(ANY, 1, "@0", None, thread_id=42)

    async def test_empty_capture_keeps_existing_status(self) -> None:
        bot = AsyncMock(spec=Bot)
        with _tick_env(capture=None) as env:
            await _update_status(bot, 1, "@0", thread_id=42)
        env.enqueue.assert_not_called()

    async def test_uses_pyte_result_instead_of_the_provider_regex(self) -> None:
        bot = AsyncMock(spec=Bot)
        pyte_status = StatusUpdate(raw_text="Reading file", display_label="📖 reading…")
        with _tick_env(pyte_result=pyte_status) as env:
            await _update_status(bot, 1, "@0", thread_id=42)
        env.provider.parse_terminal_status.assert_not_called()
        env.enqueue.assert_called_once()
        assert env.enqueue.call_args[0][3] == "📖 Reading file"

    async def test_falls_back_to_the_provider_with_pyte_rendered_text(self) -> None:
        bot = AsyncMock(spec=Bot)
        provider = _provider(
            terminal_status=StatusUpdate(
                raw_text="Working...", display_label="…working"
            )
        )
        with _tick_env(pyte_result=None, provider=provider) as env:
            terminal_poll_state.get_state(
                "@0"
            ).last_rendered_text = "clean rendered text"
            await _update_status(bot, 1, "@0", thread_id=42)
        env.provider.parse_terminal_status.assert_called_once()
        assert env.provider.parse_terminal_status.call_args[0][0] == (
            "clean rendered text"
        )

    async def test_empty_rendered_text_does_not_fall_back_to_raw_ansi(self) -> None:
        bot = AsyncMock(spec=Bot)
        with _tick_env(
            pyte_result=None, capture="\x1b[1msome ansi output\x1b[0m"
        ) as env:
            terminal_poll_state.get_state("@0").last_rendered_text = ""
            await _update_status(bot, 1, "@0", thread_id=42)
        assert env.provider.parse_terminal_status.call_args[0][0] == ""

    async def test_vim_insert_detected_from_rendered_text(self) -> None:
        bot = AsyncMock(spec=Bot)
        pyte_status = StatusUpdate(raw_text="Working", display_label="...working")
        with _tick_env(pyte_result=pyte_status) as env:
            terminal_poll_state.get_state(
                "@0"
            ).last_rendered_text = "some code\n-- INSERT --\n"
            await _update_status(bot, 1, "@0", thread_id=42)
        env.vim_notify.assert_called_once_with("@0")

    async def test_status_includes_subagent_names(self) -> None:
        bot = AsyncMock(spec=Bot)
        pyte_status = StatusUpdate(raw_text="Working", display_label="⏳ Working…")
        with _tick_env(pyte_result=pyte_status, subagents=("write-tests",)) as env:
            await _update_status(bot, 1, "@0", thread_id=42)
        status_text = env.enqueue.call_args[0][3]
        assert "write-tests" in status_text
        assert "\U0001f916" in status_text

    async def test_status_prefers_multiline_raw_task_block(self) -> None:
        bot = AsyncMock(spec=Bot)
        pyte_status = StatusUpdate(
            raw_text=(
                "Running py-idioms review…\n"
                "✔ Detect languages and scope\n"
                "◼ Spawn review agents\n"
                "◻ Collect agent results"
            ),
            display_label="⚡ running…",
        )
        with _tick_env(pyte_result=pyte_status) as env:
            await _update_status(bot, 1, "@0", thread_id=42)
        status_text = env.enqueue.call_args[0][3]
        assert status_text.startswith("Running py-idioms review…")
        assert "✔ Detect languages and scope" in status_text
        assert "◻ Collect agent results" in status_text

    async def test_interactive_window_clears_when_ui_disappears(self) -> None:
        bot = AsyncMock(spec=Bot)
        non_interactive = StatusUpdate(raw_text="Working", display_label="...working")
        with _tick_env(pyte_result=non_interactive, interactive_window="@0") as env:
            await _update_status(bot, 1, "@0", thread_id=42)
        _assert_clear_called_once_with_client(env.clear_interactive_msg, 1, bot, 42)

    async def test_new_interactive_ui_enters_interactive_mode(self) -> None:
        bot = AsyncMock(spec=Bot)
        with _tick_env(pyte_result=_INTERACTIVE_STATUS, capture="Allow?\nEsc\n") as env:
            await _update_status(bot, 1, "@0", thread_id=42)
        _assert_handle_called_once_with_client(env.handle_ui, bot, 1, "@0", 42)
        env.enqueue.assert_not_called()


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
        bot = AsyncMock(spec=Bot)
        with _tick_env(
            pyte_result=_INTERACTIVE_STATUS,
            capture="Allow?\nEsc\n",
            interactive_window=interactive_window,
        ) as env:
            await _check_interactive_only(bot, 1, "@0", 42, _window=env.window)

        env.pyte.assert_called_once()
        args, kwargs = env.pyte.call_args
        assert args == ("@0", "Allow?\nEsc\n")
        assert kwargs.get("columns") == 80
        assert kwargs.get("rows") == 24
        assert kwargs.get("parse_claude_chrome") is True
        env.set_interactive_mode.assert_called_once_with(1, "@0", 42)
        _assert_handle_called_once_with_client(env.handle_ui, bot, 1, "@0", 42)

    async def test_clears_interactive_mode_on_handle_failure(self) -> None:
        bot = AsyncMock(spec=Bot)
        with _tick_env(pyte_result=_INTERACTIVE_STATUS, capture="Allow?\nEsc\n") as env:
            env.handle_ui.return_value = False
            await _check_interactive_only(bot, 1, "@0", 42, _window=env.window)

        env.set_interactive_mode.assert_called_once_with(1, "@0", 42)
        env.clear_interactive_mode.assert_called_once_with(1, 42)

    async def test_skips_when_already_interactive(self) -> None:
        bot = AsyncMock(spec=Bot)
        with _tick_env(interactive_window="@0") as env:
            await _check_interactive_only(bot, 1, "@0", 42, _window=env.window)

        env.tmux.capture_pane.assert_not_called()
        env.pyte.assert_not_called()
        env.handle_ui.assert_not_called()

    async def test_no_action_when_not_interactive(self) -> None:
        bot = AsyncMock(spec=Bot)
        normal = StatusUpdate(raw_text="Reading file", display_label="reading...")
        with _tick_env(pyte_result=normal) as env:
            await _check_interactive_only(bot, 1, "@0", 42, _window=env.window)

        env.handle_ui.assert_not_called()
        env.set_interactive_mode.assert_not_called()

    async def test_no_action_when_window_gone(self) -> None:
        bot = AsyncMock(spec=Bot)
        with _tick_env(window=None) as env:
            await _check_interactive_only(bot, 1, "@0", 42)

        env.tmux.capture_pane.assert_not_called()
        env.handle_ui.assert_not_called()

    async def test_no_action_on_empty_capture(self) -> None:
        bot = AsyncMock(spec=Bot)
        with _tick_env(capture="") as env:
            await _check_interactive_only(bot, 1, "@0", 42, _window=env.window)

        env.pyte.assert_not_called()
        env.handle_ui.assert_not_called()

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
        bot = AsyncMock(spec=Bot)
        provider = _provider(
            terminal_status=_INTERACTIVE_STATUS, uses_pane_title=uses_pane_title
        )
        with _tick_env(
            pyte_result=None, capture="Allow?\nEsc\n", provider=provider
        ) as env:
            env.observe_tmux.get_pane_title = AsyncMock(return_value="gemini-title")
            await _check_interactive_only(bot, 1, "@0", 42, _window=env.window)

        provider.parse_terminal_status.assert_called_once_with(
            "Allow?\nEsc\n", pane_title=expected_title
        )
        _assert_handle_called_once_with_client(env.handle_ui, bot, 1, "@0", 42)
        if uses_pane_title:
            env.observe_tmux.get_pane_title.assert_called_once_with("@0")
        else:
            env.observe_tmux.get_pane_title.assert_not_called()
