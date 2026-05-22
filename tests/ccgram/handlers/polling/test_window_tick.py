import ast
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot

from ccgram.handlers.polling import window_tick
from ccgram.handlers.polling.polling_state import (
    interactive_strategy,
    lifecycle_strategy,
    terminal_poll_state,
    terminal_screen_buffer,
)
from ccgram.handlers.polling.polling_types import TickContext
from ccgram.handlers.polling.window_tick import (
    _check_interactive_only,
    _handle_dead_window_notification,
    _maybe_check_passive_shell,
    _scan_window_panes,
    _update_status,
    decide_tick,
    tick_window,
)
from ccgram.providers.base import StatusUpdate


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
            mock_dead.assert_called_once_with(bot, 1, 100, "@0")

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


class TestDecideTickActiveTranscript:
    def test_recently_active_yields_active_transition(self):
        ctx = _make_ctx(is_recently_active=True)
        decision = decide_tick(ctx)
        assert decision.transition == "active"
        assert decision.send_status is False


class TestDecideTickShellPrompt:
    def test_claude_provider_yields_done(self):
        ctx = _make_ctx(is_shell_prompt=True, supports_hook=True)
        decision = decide_tick(ctx)
        assert decision.transition == "done"

    def test_shell_provider_yields_idle(self):
        ctx = _make_ctx(is_shell_prompt=True, supports_hook=False)
        decision = decide_tick(ctx)
        assert decision.transition == "idle"

    def test_no_startup_time_yields_starting(self):
        ctx = _make_ctx(startup_time=None)
        decision = decide_tick(ctx)
        assert decision.transition == "starting"


class TestScanPanes:
    async def test_single_pane_cache_fast_path(self):
        bot = AsyncMock(spec=Bot)
        terminal_screen_buffer.update_pane_count_cache("@0", 1)

        with patch("ccgram.tmux_manager.tmux_manager") as mock_tm:
            await _scan_window_panes(bot, 1, "@0", 100)
            mock_tm.list_panes.assert_not_called()

    async def test_surfaces_interactive_alert(self):
        bot = AsyncMock(spec=Bot)
        pane_active = MagicMock(pane_id="%0", active=True, command="claude")
        pane_blocked = MagicMock(pane_id="%1", active=False, command="claude")
        interactive_status = _make_status(raw_text="Permission?", is_interactive=True)

        with (
            patch("ccgram.tmux_manager.tmux_manager") as mock_tm,
            patch("ccgram.providers.get_provider_for_window") as mock_prov,
            patch(
                "ccgram.handlers.polling.window_tick.apply.handle_interactive_ui",
                new_callable=AsyncMock,
            ) as mock_ui,
        ):
            mock_tm.list_panes = AsyncMock(return_value=[pane_active, pane_blocked])
            mock_tm.capture_pane_by_id = AsyncMock(return_value="pane text")
            mock_prov.return_value.parse_terminal_status.return_value = (
                interactive_status
            )
            await _scan_window_panes(bot, 1, "@0", 100)
            mock_ui.assert_called_once()
            assert mock_ui.call_args.kwargs.get("pane_id") == "%1"


class TestMaybeCheckPassiveShell:
    async def test_non_shell_noop(self):
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.polling.window_tick.apply.window_query") as mock_sm,
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager"),
        ):
            mock_sm.get_window_state.return_value = MagicMock(provider_name="claude")
            await _maybe_check_passive_shell(bot, 1, "@0", 100)

    async def test_shell_provider_calls_passive_check(self):
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
            ws = terminal_poll_state.get_state("@0")
            ws.last_rendered_text = "$ output here"
            mock_tm.capture_pane = AsyncMock(return_value="$ output here")
            await _maybe_check_passive_shell(bot, 1, "@0", 100)
            mock_check.assert_called_once()


class TestCheckInteractiveOnly:
    async def test_already_interactive_returns_early(self):
        bot = AsyncMock(spec=Bot)
        w = _make_window()

        with (
            patch("ccgram.handlers.polling.window_tick.apply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.polling.window_tick.apply.get_interactive_window",
                return_value="@0",
            ),
        ):
            mock_tm.find_window_by_id = AsyncMock(return_value=w)
            mock_tm.capture_pane = AsyncMock()
            await _check_interactive_only(bot, 1, "@0", 100, _window=w)
            mock_tm.capture_pane.assert_not_called()


class TestDeadWindowNotification:
    async def test_sends_once(self):
        bot = AsyncMock(spec=Bot)
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
            ) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = 42
            mock_tr.get_display_name.return_value = "test"
            mock_sm.get_window_state.return_value = MagicMock(cwd="/tmp")
            mock_send.return_value = MagicMock()
            await _handle_dead_window_notification(bot, 1, 100, "@0")
            assert lifecycle_strategy.is_dead_notified(1, 100, "@0")
            mock_send.reset_mock()
            await _handle_dead_window_notification(bot, 1, 100, "@0")
            mock_send.assert_not_called()


class TestContractTests:
    def test_tick_window_exists_and_is_callable(self):
        assert hasattr(window_tick, "tick_window")
        assert callable(window_tick.tick_window)

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

    def test_polling_coordinator_does_not_import_per_window_collaborators(self):
        import ccgram.handlers.polling.polling_coordinator as pc

        source = inspect.getsource(pc)
        tree = ast.parse(source)
        forbidden = {
            "claude_task_state",
            "providers.base",
            "session_monitor",
            "cleanup",
            "interactive_ui",
            "message_queue",
            "message_sender",
            "recovery_callbacks",
            "topic_emoji",
            "transcript_discovery",
            "polling_state",
        }
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        violations = {m for m in imported_modules if any(f in m for f in forbidden)}
        assert not violations, (
            f"polling_coordinator imports per-window collaborators: {violations}"
        )


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
            mock_tr.unbind_thread.assert_called_once_with(1, 100)


class TestPaneLifecycleNotify:
    @pytest.fixture(autouse=True)
    def _reset_store(self):
        from ccgram.window_state_store import window_store

        window_store.reset()
        saved_schedule = window_store._schedule_save
        window_store._schedule_save = lambda: None
        yield
        window_store.reset()
        window_store._schedule_save = saved_schedule

    @pytest.fixture
    def transitions(self):
        from ccgram.handlers.polling.polling_types import PaneTransition

        return [
            PaneTransition(pane_id="%5", prev_state=None, new_state="active"),
            PaneTransition(pane_id="%6", prev_state=None, new_state="dead"),
            PaneTransition(pane_id="%7", prev_state="idle", new_state="active"),
        ]

    async def test_disabled_globally_and_per_window_no_notify(self, transitions):
        from ccgram.handlers.polling import window_tick

        bot = AsyncMock(spec=Bot)
        with (
            patch.object(window_tick.apply, "thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.config",
                MagicMock(pane_lifecycle_notify=False),
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = 42
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, transitions)
        mock_send.assert_not_called()

    async def test_per_window_override_enables(self, transitions):
        from ccgram.handlers.polling import window_tick
        from ccgram.window_state_store import window_store

        window_store.set_pane_lifecycle_notify("@0", True)
        bot = AsyncMock(spec=Bot)
        with (
            patch.object(window_tick.apply, "thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.config",
                MagicMock(pane_lifecycle_notify=False),
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = 42
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, transitions)
        # Created (%5) and closed (%6) only — state-change (%7) is skipped
        assert mock_send.await_count == 2
        texts = [call.args[2] for call in mock_send.call_args_list]
        assert any("➕" in t and "%5" in t and "created" in t for t in texts)
        assert any("➖" in t and "%6" in t and "closed" in t for t in texts)

    async def test_per_window_override_disables(self, transitions):
        from ccgram.handlers.polling import window_tick
        from ccgram.window_state_store import window_store

        # Global default ON, but per-window override OFF wins.
        window_store.set_pane_lifecycle_notify("@0", False)
        bot = AsyncMock(spec=Bot)
        with (
            patch.object(window_tick.apply, "thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.config",
                MagicMock(pane_lifecycle_notify=True),
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = 42
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, transitions)
        mock_send.assert_not_called()

    async def test_global_default_enables_when_no_override(self, transitions):
        from ccgram.handlers.polling import window_tick

        bot = AsyncMock(spec=Bot)
        with (
            patch.object(window_tick.apply, "thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.config",
                MagicMock(pane_lifecycle_notify=True),
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = 42
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, transitions)
        assert mock_send.await_count == 2

    async def test_named_pane_used_in_label(self):
        from ccgram.handlers.polling import window_tick
        from ccgram.handlers.polling.polling_types import PaneTransition
        from ccgram.window_state_store import window_store

        window_store.set_pane_lifecycle_notify("@0", True)
        window_store.upsert_pane("@0", "%5", name="api-gateway", state="active")
        bot = AsyncMock(spec=Bot)
        transitions = [
            PaneTransition(pane_id="%5", prev_state=None, new_state="active"),
        ]
        with (
            patch.object(window_tick.apply, "thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.config",
                MagicMock(pane_lifecycle_notify=False),
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = 42
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, transitions)
        text = mock_send.call_args.args[2]
        assert "api-gateway" in text
        assert "%5" in text

    async def test_telegram_error_logged_not_raised(self):
        from telegram.error import TelegramError

        from ccgram.handlers.polling import window_tick
        from ccgram.handlers.polling.polling_types import PaneTransition
        from ccgram.window_state_store import window_store

        window_store.set_pane_lifecycle_notify("@0", True)
        bot = AsyncMock(spec=Bot)
        transitions = [
            PaneTransition(pane_id="%5", prev_state=None, new_state="active"),
        ]
        with (
            patch.object(window_tick.apply, "thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.config",
                MagicMock(pane_lifecycle_notify=False),
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.safe_send",
                new_callable=AsyncMock,
                side_effect=TelegramError("boom"),
            ),
        ):
            mock_tr.resolve_chat_id.return_value = 42
            # Should swallow and log
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, transitions)

    async def test_empty_transitions_short_circuits(self):
        from ccgram.handlers.polling import window_tick
        from ccgram.window_state_store import window_store

        window_store.set_pane_lifecycle_notify("@0", True)
        bot = AsyncMock(spec=Bot)
        with (
            patch.object(window_tick.apply, "thread_router") as mock_tr,
            patch(
                "ccgram.handlers.polling.window_tick.apply.config",
                MagicMock(pane_lifecycle_notify=True),
            ),
            patch(
                "ccgram.handlers.polling.window_tick.apply.safe_send",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            mock_tr.resolve_chat_id.return_value = 42
            await window_tick._notify_pane_lifecycle(bot, 1, "@0", 100, [])
        mock_send.assert_not_called()

    async def test_scan_panes_invokes_lifecycle(self):
        from ccgram.handlers.polling import window_tick
        from ccgram.handlers.polling.polling_types import PaneTransition

        bot = AsyncMock(spec=Bot)
        scan_transitions = [
            PaneTransition(pane_id="%5", prev_state=None, new_state="active"),
        ]
        with (
            patch.object(window_tick.apply, "pane_status_strategy") as mock_strategy,
            patch.object(
                window_tick.apply, "_notify_pane_lifecycle", new_callable=AsyncMock
            ) as mock_notify,
        ):
            mock_strategy.scan_window = AsyncMock(return_value=scan_transitions)
            await _scan_window_panes(bot, 1, "@0", 100)
        mock_notify.assert_awaited_once()
        passed_transitions = mock_notify.call_args.args[4]
        assert passed_transitions == scan_transitions

    async def test_scan_panes_skips_lifecycle_on_empty(self):
        from ccgram.handlers.polling import window_tick

        bot = AsyncMock(spec=Bot)
        with (
            patch.object(window_tick.apply, "pane_status_strategy") as mock_strategy,
            patch.object(
                window_tick.apply, "_notify_pane_lifecycle", new_callable=AsyncMock
            ) as mock_notify,
        ):
            mock_strategy.scan_window = AsyncMock(return_value=[])
            await _scan_window_panes(bot, 1, "@0", 100)
        mock_notify.assert_not_called()
