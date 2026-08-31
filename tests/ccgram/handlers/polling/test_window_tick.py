import ast
import inspect
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot
from telegram.error import TelegramError

from ccgram.handlers.polling import window_tick
from ccgram.handlers.polling.polling_state import (
    interactive_strategy,
    lifecycle_strategy,
    terminal_poll_state,
    terminal_screen_buffer,
)
from ccgram.handlers.polling.window_tick import (
    _forward_pane_output,
    _handle_dead_window_notification,
    _maybe_check_passive_shell,
    _scan_window_panes,
    _send_typing_throttled,
    _update_status,
    tick_window,
)
from ccgram.handlers.polling.polling_types import PaneTransition
from ccgram.handlers.polling.window_tick.apply import _PANE_OUTPUT_PREVIEW_LINES
from ccgram.providers.base import StatusUpdate
from ccgram.window_state_store import window_store


@pytest.fixture(autouse=True)
def _reset():
    terminal_poll_state._states.clear()  # no public clear_all method
    lifecycle_strategy.reset_autoclose_state()
    lifecycle_strategy.reset_typing_state()
    lifecycle_strategy.reset_dead_notification_state()
    interactive_strategy.clear_all_alerts()
    terminal_screen_buffer.reset_screen_buffer_state()
    yield
    terminal_poll_state._states.clear()
    lifecycle_strategy.reset_autoclose_state()
    lifecycle_strategy.reset_typing_state()
    lifecycle_strategy.reset_dead_notification_state()
    interactive_strategy.clear_all_alerts()
    terminal_screen_buffer.reset_screen_buffer_state()


def _make_window(
    window_id="@0", pane_width=120, pane_height=40, pane_current_command="claude"
):
    w = MagicMock()
    w.window_id = window_id
    w.pane_width = pane_width
    w.pane_height = pane_height
    w.pane_current_command = pane_current_command
    return w


def _make_status(raw_text="Working...", is_interactive=False, display_label=""):
    return StatusUpdate(
        raw_text=raw_text, display_label=display_label, is_interactive=is_interactive
    )


class TestTickWindowDeadWindow:
    async def test_dead_window_calls_handle_dead(self):
        bot = AsyncMock(spec=Bot)
        with patch.object(
            window_tick, "_handle_dead_window_notification", new_callable=AsyncMock
        ) as mock_dead:
            await tick_window(bot, 1, 100, "@0", None)
            mock_dead.assert_called_once()
            args, kwargs = mock_dead.call_args
            assert args == (bot, 1, 100, "@0")

    async def test_dead_window_skips_other_work(self):
        bot = AsyncMock(spec=Bot)
        with (
            patch.object(
                window_tick, "_handle_dead_window_notification", new_callable=AsyncMock
            ),
            patch.object(
                window_tick, "_update_status", new_callable=AsyncMock
            ) as mock_status,
            patch.object(
                window_tick, "_scan_window_panes", new_callable=AsyncMock
            ) as mock_scan,
        ):
            await tick_window(bot, 1, 100, "@0", None)
            mock_status.assert_not_called()
            mock_scan.assert_not_called()

    async def test_already_dead_notified_returns_early(self):
        bot = AsyncMock(spec=Bot)
        lifecycle_strategy.mark_dead_notified(1, 100, "@0")
        with patch.object(
            window_tick, "_handle_dead_window_notification", new_callable=AsyncMock
        ) as mock_dead:
            await tick_window(bot, 1, 100, "@0", None)
            mock_dead.assert_not_called()

    async def test_agent_exit_kills_managed_window_and_enters_recovery(self):
        bot = AsyncMock(spec=Bot)
        window = _make_window()
        with (
            patch.object(
                window_tick,
                "discover_and_register_transcript",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                window_tick.lifecycle_state,
                "get_origin",
                return_value="ccgram_created",
            ),
            patch.object(
                window_tick,
                "tmux_manager",
                MagicMock(kill_window=AsyncMock()),
            ) as mock_tmux,
            patch.object(
                window_tick, "_handle_dead_window_notification", new_callable=AsyncMock
            ) as mock_dead,
            patch.object(
                window_tick, "_update_status", new_callable=AsyncMock
            ) as mock_status,
        ):
            await tick_window(bot, 1, 100, "@0", window)

        mock_tmux.kill_window.assert_awaited_once_with("@0")
        mock_dead.assert_awaited_once_with(
            bot, 1, 100, "@0", runtime=window_tick.get_default_runtime()
        )
        mock_status.assert_not_awaited()

    async def test_agent_exit_keeps_manually_discovered_window(self):
        bot = AsyncMock(spec=Bot)
        window = _make_window()
        with (
            patch.object(
                window_tick,
                "discover_and_register_transcript",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                window_tick.lifecycle_state,
                "get_origin",
                return_value="manual_discovered",
            ),
            patch.object(
                window_tick,
                "tmux_manager",
                MagicMock(kill_window=AsyncMock()),
            ) as mock_tmux,
            patch.object(
                window_tick, "_handle_dead_window_notification", new_callable=AsyncMock
            ) as mock_dead,
        ):
            await tick_window(bot, 1, 100, "@0", window)

        mock_tmux.kill_window.assert_not_awaited()
        mock_dead.assert_awaited_once()


class TestTickWindowPendingQueue:
    async def test_pending_queue_skips_status_update(self):
        bot = AsyncMock(spec=Bot)
        w = _make_window()
        mock_queue = MagicMock()
        mock_queue.empty.return_value = False

        with (
            patch.object(
                window_tick, "discover_and_register_transcript", new_callable=AsyncMock
            ),
            patch.object(window_tick, "get_message_queue", return_value=mock_queue),
            patch.object(
                window_tick, "_check_interactive_only", new_callable=AsyncMock
            ) as mock_interactive,
            patch.object(
                window_tick, "_update_status", new_callable=AsyncMock
            ) as mock_status,
            patch.object(
                window_tick, "_scan_window_panes", new_callable=AsyncMock
            ) as mock_scan,
            patch.object(
                window_tick, "_maybe_check_passive_shell", new_callable=AsyncMock
            ) as mock_shell,
        ):
            await tick_window(bot, 1, 100, "@0", w)
            mock_interactive.assert_called_once()
            mock_status.assert_not_called()
            mock_scan.assert_called_once()
            mock_shell.assert_called_once()


class TestTickWindowEmptyQueue:
    async def test_empty_queue_runs_status_update(self):
        bot = AsyncMock(spec=Bot)
        w = _make_window()
        mock_queue = MagicMock()
        mock_queue.empty.return_value = True

        with (
            patch.object(
                window_tick, "discover_and_register_transcript", new_callable=AsyncMock
            ),
            patch.object(window_tick, "get_message_queue", return_value=mock_queue),
            patch.object(
                window_tick, "_update_status", new_callable=AsyncMock
            ) as mock_status,
            patch.object(
                window_tick, "_scan_window_panes", new_callable=AsyncMock
            ) as mock_scan,
            patch.object(
                window_tick, "_maybe_check_passive_shell", new_callable=AsyncMock
            ) as mock_shell,
        ):
            await tick_window(bot, 1, 100, "@0", w)
            mock_status.assert_called_once()
            mock_scan.assert_called_once()
            mock_shell.assert_called_once()

    async def test_no_queue_runs_status_update(self):
        bot = AsyncMock(spec=Bot)
        w = _make_window()

        with (
            patch.object(
                window_tick, "discover_and_register_transcript", new_callable=AsyncMock
            ),
            patch.object(window_tick, "get_message_queue", return_value=None),
            patch.object(
                window_tick, "_update_status", new_callable=AsyncMock
            ) as mock_status,
            patch.object(window_tick, "_scan_window_panes", new_callable=AsyncMock),
            patch.object(
                window_tick, "_maybe_check_passive_shell", new_callable=AsyncMock
            ),
        ):
            await tick_window(bot, 1, 100, "@0", w)
            mock_status.assert_called_once()


class TestUpdateStatusInteractive:
    async def test_interactive_ui_wins_over_status(self):
        bot = AsyncMock(spec=Bot)
        w = _make_window()
        interactive_status = _make_status(raw_text="Accept?", is_interactive=True)

        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.thread_router"),
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
            patch(
                "ccgram.handlers.polling.window_tick.apply.enqueue_status_update",
                new_callable=AsyncMock,
            ) as mock_enqueue,
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji",
                new_callable=AsyncMock,
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=w)
            mock_tm.capture_pane = AsyncMock(return_value="pane text")
            await _update_status(bot, 1, "@0", thread_id=100, _window=w)
            mock_handle.assert_called_once()
            mock_enqueue.assert_not_called()


class TestUpdateStatusActiveLine:
    async def test_active_status_enqueues_and_sets_emoji(self):
        bot = AsyncMock(spec=Bot)
        w = _make_window()
        status = _make_status(raw_text="Working on task", is_interactive=False)

        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=status,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.enqueue_status_update",
                new_callable=AsyncMock,
            ) as mock_enqueue,
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji",
                new_callable=AsyncMock,
            ) as mock_emoji,
            patch(
                "ccgram.handlers.polling.window_tick.apply._send_typing_throttled",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.claude_task_state"
            ) as mock_cts,
            patch("ccgram.handlers.polling.window_tick.apply.get_provider_for_window"),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=w)
            mock_tm.capture_pane = AsyncMock(return_value="pane text")
            mock_tr.resolve_chat_id.return_value = 42
            mock_tr.get_display_name.return_value = "test"
            mock_cts.get_subagent_names = MagicMock(return_value=[])
            mock_cts.build_subagent_label = MagicMock()
            await _update_status(bot, 1, "@0", thread_id=100, _window=w)
            mock_enqueue.assert_called_once()
            mock_emoji.assert_called()
            mock_cts.set_last_status.assert_called_once()

    async def test_subagent_label_appended(self):
        bot = AsyncMock(spec=Bot)
        w = _make_window()
        status = _make_status(raw_text="Working", is_interactive=False)

        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch("ccgram.handlers.polling.window_tick.apply.window_query"),
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.observe._parse_with_pyte",
                return_value=status,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.enqueue_status_update",
                new_callable=AsyncMock,
            ) as mock_enqueue,
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply._send_typing_throttled",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.claude_task_state"
            ) as mock_cts,
            patch("ccgram.handlers.polling.window_tick.apply.get_provider_for_window"),
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_subagent_names",
                return_value=["subagent-1"],
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.build_subagent_label",
                return_value="1 subagent",
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=w)
            mock_tm.capture_pane = AsyncMock(return_value="pane text")
            mock_tr.resolve_chat_id.return_value = 42
            mock_tr.get_display_name.return_value = "test"
            mock_cts.clear_wait_header = MagicMock()
            mock_cts.set_last_status = MagicMock()
            await _update_status(bot, 1, "@0", thread_id=100, _window=w)
            enqueue_call = mock_enqueue.call_args
            assert "1 subagent" in str(enqueue_call)


class TestMaybeCheckPassiveShell:
    async def test_non_shell_provider_is_skipped(self):
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_provider_for_window"
            ) as mock_prov,
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.shell.shell_capture.check_passive_shell_output",
                new_callable=AsyncMock,
            ) as mock_check,
        ):
            mock_prov.return_value.capabilities.chat_first_command_path = False
            mock_tm.capture_pane = AsyncMock()
            await _maybe_check_passive_shell(bot, 1, "@0", 100)

        mock_check.assert_not_called()
        mock_tm.capture_pane.assert_not_called()

    async def test_shell_provider_passes_rendered_text_through(self):
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_provider_for_window"
            ) as mock_prov,
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.shell.shell_capture.check_passive_shell_output",
                new_callable=AsyncMock,
            ) as mock_check,
        ):
            mock_prov.return_value.capabilities.chat_first_command_path = True
            terminal_poll_state.get_state("@0").last_rendered_text = "$ output here"
            mock_tm.capture_pane = AsyncMock()
            await _maybe_check_passive_shell(bot, 1, "@0", 100)

        mock_tm.capture_pane.assert_not_called()
        assert mock_check.call_args[0][4] == "$ output here"

    @pytest.mark.parametrize(
        ("captured", "expected"),
        [
            pytest.param("$ raw capture", "$ raw capture", id="capture-succeeds"),
            pytest.param("", None, id="capture-empty"),
        ],
    )
    async def test_falls_back_to_a_live_capture_when_nothing_is_rendered_yet(
        self, captured, expected
    ):
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_provider_for_window"
            ) as mock_prov,
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.shell.shell_capture.check_passive_shell_output",
                new_callable=AsyncMock,
            ) as mock_check,
        ):
            mock_prov.return_value.capabilities.chat_first_command_path = True
            mock_tm.capture_pane = AsyncMock(return_value=captured)
            await _maybe_check_passive_shell(bot, 1, "@0", 100)

        if expected is None:
            mock_check.assert_not_called()
        else:
            assert mock_check.call_args[0][4] == expected


class TestContractTests:
    def test_tick_window_is_coroutine_function(self):
        assert inspect.iscoroutinefunction(window_tick.tick_window)

    def test_tick_window_is_sole_async_public_function(self):
        public_async = [
            name
            for name in dir(window_tick)
            if not name.startswith("_")
            and inspect.iscoroutinefunction(getattr(window_tick, name))
            and getattr(getattr(window_tick, name), "__module__", None)
            == "ccgram.handlers.polling.window_tick"
        ]
        assert public_async == ["tick_window"]

    def test_decide_tick_is_public_pure_function(self):
        assert hasattr(window_tick, "decide_tick")
        assert not inspect.iscoroutinefunction(window_tick.decide_tick)
        assert callable(window_tick.decide_tick)

    def test_polling_coordinator_imports_only_tick_window(self):
        import ccgram.handlers.polling.polling_coordinator as pc

        source = inspect.getsource(pc)
        tree = ast.parse(source)
        window_tick_imports = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module and "window_tick" in node.module:
                for alias in node.names:
                    window_tick_imports.append(alias.name)
            elif node.level and node.level > 0 and node.module is None:
                for alias in node.names:
                    if alias.name == "window_tick":
                        window_tick_imports.append(alias.name)
        assert window_tick_imports == [] or all(
            name == "window_tick" for name in window_tick_imports
        ), f"Unexpected imports from window_tick: {window_tick_imports}"


class TestDeadWindowTopicDeleted:
    @pytest.mark.parametrize(
        "error_msg",
        ["thread not found", "TOPIC_ID_INVALID"],
        ids=["thread_not_found", "topic_id_invalid"],
    )
    async def test_thread_not_found_unbinds_and_clears(self, error_msg):
        from telegram.error import BadRequest

        bot = AsyncMock(spec=["unpin_all_forum_topic_messages"])
        bot.unpin_all_forum_topic_messages = AsyncMock(
            side_effect=BadRequest(error_msg)
        )

        with (
            patch("ccgram.handlers.polling.window_tick.apply.thread_router") as mock_tr,
            patch("ccgram.handlers.polling.window_tick.apply.window_query") as mock_sm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.update_topic_emoji",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.clear_tool_msg_ids_for_topic"
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.rate_limit_send_message",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.clear_topic_state",
                new_callable=AsyncMock,
            ) as mock_clear,
        ):
            mock_tr.resolve_chat_id.return_value = 42
            mock_tr.get_display_name.return_value = "test"
            mock_sm.get_window_state.return_value = MagicMock(cwd="/tmp")

            await _handle_dead_window_notification(bot, 1, 100, "@0")

            mock_clear.assert_awaited_once()
            _, kwargs = mock_clear.call_args
            assert kwargs.get("window_dead") is True
            mock_tr.unbind_thread.assert_called_once_with(
                1, 100, retirement_reason="remote_deleted"
            )


class TestPaneLifecycleNotify:
    @pytest.fixture(autouse=True)
    def _reset_store(self):
        window_store.reset()
        saved_schedule = window_store._schedule_save
        window_store._schedule_save = lambda: None
        yield
        window_store.reset()
        window_store._schedule_save = saved_schedule

    @pytest.fixture
    def transitions(self):
        return [
            PaneTransition(pane_id="%5", prev_state=None, new_state="active"),
            PaneTransition(pane_id="%6", prev_state=None, new_state="dead"),
            PaneTransition(pane_id="%7", prev_state="idle", new_state="active"),
        ]

    @staticmethod
    @contextmanager
    def _notify_env(*, global_default: bool, send_error: Exception | None = None):
        with (
            patch.object(window_tick.apply, "thread_router") as mock_tr,
            patch.object(
                window_tick.apply,
                "config",
                MagicMock(pane_lifecycle_notify=global_default),
            ),
            patch.object(
                window_tick.apply,
                "safe_send",
                new_callable=AsyncMock,
                side_effect=send_error,
            ) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = 42
            yield mock_send

    @pytest.mark.parametrize(
        ("global_default", "override", "expect_sends"),
        [
            pytest.param(False, None, False, id="off-by-default"),
            pytest.param(True, None, True, id="on-by-default"),
            pytest.param(False, True, True, id="override-enables"),
            pytest.param(True, False, False, id="override-disables"),
        ],
    )
    async def test_notify_gate(
        self, transitions, global_default, override, expect_sends
    ):
        if override is not None:
            window_store.set_pane_lifecycle_notify("@0", override)
        bot = AsyncMock(spec=Bot)
        with self._notify_env(global_default=global_default) as mock_send:
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, transitions)

        # Only created (%5) and closed (%6) are announced — a state change
        # between two live states (%7) is not a lifecycle event.
        assert mock_send.await_count == (2 if expect_sends else 0)

    async def test_created_and_closed_lines_name_the_pane(self, transitions):
        window_store.set_pane_lifecycle_notify("@0", True)
        bot = AsyncMock(spec=Bot)
        with self._notify_env(global_default=False) as mock_send:
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, transitions)

        texts = [call.args[2] for call in mock_send.call_args_list]
        assert any("➕" in t and "%5" in t and "created" in t for t in texts)
        assert any("➖" in t and "%6" in t and "closed" in t for t in texts)

    async def test_named_pane_used_in_label(self):
        window_store.set_pane_lifecycle_notify("@0", True)
        window_store.upsert_pane("@0", "%5", name="api-gateway", state="active")
        bot = AsyncMock(spec=Bot)
        transitions = [
            PaneTransition(pane_id="%5", prev_state=None, new_state="active")
        ]
        with self._notify_env(global_default=False) as mock_send:
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, transitions)

        text = mock_send.call_args.args[2]
        assert "api-gateway" in text
        assert "%5" in text

    async def test_one_failed_send_does_not_abort_the_rest(self):
        window_store.set_pane_lifecycle_notify("@0", True)
        bot = AsyncMock(spec=Bot)
        transitions = [
            PaneTransition(pane_id="%5", prev_state=None, new_state="active"),
            PaneTransition(pane_id="%6", prev_state=None, new_state="active"),
        ]
        with self._notify_env(
            global_default=False, send_error=TelegramError("boom")
        ) as mock_send:
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, transitions)

        assert mock_send.await_count == 2

    async def test_empty_transitions_short_circuits(self):
        window_store.set_pane_lifecycle_notify("@0", True)
        bot = AsyncMock(spec=Bot)
        with self._notify_env(global_default=True) as mock_send:
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, [])

        mock_send.assert_not_called()

    @pytest.mark.parametrize(
        "scan_transitions",
        [
            pytest.param(
                [PaneTransition(pane_id="%5", prev_state=None, new_state="active")],
                id="transitions",
            ),
            pytest.param([], id="no-transitions"),
        ],
    )
    async def test_scan_panes_forwards_transitions_to_lifecycle(self, scan_transitions):
        bot = AsyncMock(spec=Bot)
        with (
            patch.object(window_tick.apply, "pane_status_strategy") as mock_strategy,
            patch.object(
                window_tick.apply, "_notify_pane_lifecycle", new_callable=AsyncMock
            ) as mock_notify,
        ):
            mock_strategy.scan_window = AsyncMock(return_value=scan_transitions)
            await _scan_window_panes(bot, 1, "@0", 100)

        if scan_transitions:
            mock_notify.assert_awaited_once()
            assert mock_notify.call_args.args[4] == scan_transitions
        else:
            mock_notify.assert_not_called()


class TestSendTypingThrottled:
    @pytest.fixture(autouse=True)
    def _typing_env(self):
        with patch.object(window_tick.apply, "thread_router") as mock_tr:
            mock_tr.resolve_chat_id.return_value = -100
            yield mock_tr

    async def test_no_thread_means_no_chat_action(self):
        bot = AsyncMock(spec=Bot)
        await _send_typing_throttled(bot, 1, None)
        bot.send_chat_action.assert_not_called()

    async def test_first_call_sends_and_the_next_is_throttled(self):
        bot = AsyncMock(spec=Bot)
        await _send_typing_throttled(bot, 1, 42)
        await _send_typing_throttled(bot, 1, 42)
        bot.send_chat_action.assert_awaited_once()
        assert bot.send_chat_action.call_args.kwargs["chat_id"] == -100
        assert bot.send_chat_action.call_args.kwargs["message_thread_id"] == 42

    async def test_throttle_is_per_topic(self):
        bot = AsyncMock(spec=Bot)
        await _send_typing_throttled(bot, 1, 42)
        await _send_typing_throttled(bot, 1, 43)
        assert bot.send_chat_action.await_count == 2

    async def test_telegram_error_is_swallowed(self):
        bot = AsyncMock(spec=Bot)
        bot.send_chat_action.side_effect = TelegramError("boom")
        await _send_typing_throttled(bot, 1, 42)
        bot.send_chat_action.assert_awaited_once()


class TestForwardPaneOutput:
    @pytest.fixture(autouse=True)
    def _reset_store(self):
        window_store.reset()
        saved_schedule = window_store._schedule_save
        window_store._schedule_save = lambda: None
        yield
        window_store.reset()
        window_store._schedule_save = saved_schedule

    @staticmethod
    @contextmanager
    def _send_env():
        with (
            patch.object(window_tick.apply, "thread_router") as mock_tr,
            patch.object(
                window_tick.apply, "safe_send", new_callable=AsyncMock
            ) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            yield mock_send

    @pytest.mark.parametrize(
        ("subscribed", "text"),
        [
            pytest.param(False, "output", id="unsubscribed-pane"),
            pytest.param(True, "   \n\n", id="blank-capture"),
        ],
    )
    async def test_nothing_is_forwarded(self, subscribed, text):
        window_store.upsert_pane("@0", "%1", state="active", subscribed=subscribed)
        bot = AsyncMock(spec=Bot)
        with self._send_env() as mock_send:
            await _forward_pane_output(bot, 1, "@0", 42, "%1", text)
        mock_send.assert_not_called()

    async def test_unknown_pane_is_skipped(self):
        bot = AsyncMock(spec=Bot)
        with self._send_env() as mock_send:
            await _forward_pane_output(bot, 1, "@0", 42, "%missing", "output")
        mock_send.assert_not_called()

    async def test_named_pane_is_labelled_and_fenced(self):
        window_store.upsert_pane(
            "@0", "%1", state="active", subscribed=True, name="api-gateway"
        )
        bot = AsyncMock(spec=Bot)
        with self._send_env() as mock_send:
            await _forward_pane_output(bot, 1, "@0", 42, "%1", "line-a\nline-b\n")

        text = mock_send.call_args.args[2]
        assert "api-gateway (%1)" in text
        assert text.endswith("```")
        assert "line-a\nline-b" in text
        assert mock_send.call_args.kwargs["message_thread_id"] == 42

    async def test_only_the_tail_of_a_long_capture_is_forwarded(self):
        window_store.upsert_pane("@0", "%1", state="active", subscribed=True)
        bot = AsyncMock(spec=Bot)
        capture = "\n".join(f"line{i}" for i in range(30))
        with self._send_env() as mock_send:
            await _forward_pane_output(bot, 1, "@0", 42, "%1", capture)

        lines = mock_send.call_args.args[2].splitlines()[2:-1]
        assert lines == [f"line{i}" for i in range(30 - _PANE_OUTPUT_PREVIEW_LINES, 30)]

    async def test_telegram_error_is_logged_not_raised(self):
        window_store.upsert_pane("@0", "%1", state="active", subscribed=True)
        bot = AsyncMock(spec=Bot)
        with self._send_env() as mock_send:
            mock_send.side_effect = TelegramError("boom")
            await _forward_pane_output(bot, 1, "@0", 42, "%1", "output")
        mock_send.assert_awaited_once()
