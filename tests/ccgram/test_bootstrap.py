from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from ccgram import bootstrap


@pytest.fixture(autouse=True)
def _reset_bootstrap_state():
    bootstrap.reset_for_testing()
    yield
    bootstrap.reset_for_testing()


def _make_app() -> MagicMock:
    app = MagicMock()
    app.bot = AsyncMock()
    app.job_queue = None
    return app


class TestBootstrapApplicationOrdering:
    async def test_start_session_monitor_raises_when_callbacks_unwired(self):
        app = _make_app()
        with pytest.raises(
            RuntimeError, match="wire_runtime_callbacks.*before.*start_session_monitor"
        ):
            await bootstrap.start_session_monitor(app)

    async def test_start_session_monitor_succeeds_after_wire(self):
        app = _make_app()
        bootstrap.wire_runtime_callbacks()
        with patch("ccgram.bootstrap.SessionMonitor") as monitor_cls:
            instance = MagicMock()
            instance.start = MagicMock()
            monitor_cls.return_value = instance
            with patch("ccgram.bootstrap.set_active_monitor"):
                result = await bootstrap.start_session_monitor(app)

        assert result is instance
        instance.start.assert_called_once()
        assert bootstrap.session_monitor is instance


class TestEnsureMultiplexerSession:
    async def test_forwards_to_active_backend(self):
        from ccgram.multiplexer import install_multiplexer

        backend = MagicMock()
        backend.ensure_session = AsyncMock()
        install_multiplexer(backend)

        await bootstrap.ensure_multiplexer_session()

        backend.ensure_session.assert_awaited_once_with()

    async def test_exits_cleanly_when_backend_unavailable(self):
        from ccgram.multiplexer import install_multiplexer

        backend = MagicMock()
        backend.ensure_session = AsyncMock(side_effect=RuntimeError("socket down"))
        install_multiplexer(backend)

        # An unreachable backend exits via SystemExit (caught by PTB's
        # run_polling → graceful shutdown, no traceback), not a raw exception.
        with pytest.raises(SystemExit) as exc_info:
            await bootstrap.ensure_multiplexer_session()

        assert exc_info.value.code == 1
        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestWireRuntimeCallbacks:
    def test_wires_approval_callback(self):
        from ccgram.handlers.shell import shell_capture

        bootstrap.wire_runtime_callbacks()

        assert shell_capture._approval_callback_registered is True
        assert bootstrap._callbacks_wired is True

    def test_double_wire_is_idempotent(self):
        bootstrap.wire_runtime_callbacks()
        bootstrap.wire_runtime_callbacks()

        assert bootstrap._callbacks_wired is True

    def test_wires_and_resets_exact_pending_creation_ownership(self):
        from ccgram.handlers.topics.topic_orchestration import (
            clear_pending_creation,
            register_pending_creation,
        )
        import ccgram.session_map as session_map_module

        bootstrap.wire_runtime_callbacks()
        register_pending_creation("@owned")
        try:
            predicate = session_map_module._in_flight_window_predicate
            assert predicate is not None
            assert predicate("@owned")
            assert not predicate("@unrelated")

            bootstrap.reset_for_testing()

            assert session_map_module._in_flight_window_predicate is None
        finally:
            clear_pending_creation("@owned")


class TestSettlePreexistingWindows:
    def test_preserves_chat_identity_for_same_numbered_threads(self) -> None:
        from ccgram.handlers.polling.polling_state import terminal_poll_state

        bindings = [
            (7, -1001, 42, "@3"),
            (7, -1002, 42, "@4"),
            (7, None, 43, "@5"),
        ]
        with (
            patch("ccgram.bootstrap.thread_router") as mock_router,
            patch("ccgram.bootstrap.mark_awaiting_first_paint") as mark_first,
        ):
            mock_router.iter_thread_bindings_with_chat.return_value = bindings
            mock_router.resolve_chat_id.return_value = -1003

            bootstrap._settle_preexisting_windows()

        try:
            assert all(
                terminal_poll_state.check_seen_status(window_id)
                for *_, window_id in bindings
            )
            assert mark_first.call_args_list == [
                call(-1001, 42),
                call(-1002, 42),
                call(-1003, 43),
            ]
            mock_router.resolve_chat_id.assert_called_once_with(7, 43)
        finally:
            for *_, window_id in bindings:
                terminal_poll_state.clear_seen_status(window_id)


class TestBootstrapApplication:
    async def test_runs_full_sequence_in_order(self):
        app = _make_app()

        order: list[str] = []

        with (
            patch(
                "ccgram.bootstrap.install_global_exception_handler",
                side_effect=lambda: order.append("exc_handler"),
            ),
            patch(
                "ccgram.bootstrap.ensure_multiplexer_session",
                new=AsyncMock(side_effect=lambda: order.append("ensure_session")),
            ),
            patch(
                "ccgram.bootstrap.register_provider_commands",
                new=AsyncMock(side_effect=lambda _app: order.append("commands")),
            ),
            patch("ccgram.bootstrap.session_manager") as sm,
            patch(
                "ccgram.bootstrap._adopt_unbound_windows",
                new=AsyncMock(side_effect=lambda _bot: order.append("adopt")),
            ),
            patch(
                "ccgram.bootstrap.verify_hooks_installed",
                side_effect=lambda: order.append("hooks"),
            ),
            patch(
                "ccgram.bootstrap.wire_runtime_callbacks",
                side_effect=lambda: order.append("wire"),
            ),
            patch(
                "ccgram.bootstrap.start_session_monitor",
                new=AsyncMock(side_effect=lambda _app: order.append("monitor")),
            ),
            patch(
                "ccgram.bootstrap.start_status_polling",
                side_effect=lambda _app: order.append("polling"),
            ),
            patch(
                "ccgram.main.start_miniapp_if_enabled",
                new=AsyncMock(side_effect=lambda: order.append("miniapp")),
            ),
        ):
            sm.resolve_stale_ids = AsyncMock(
                side_effect=lambda: order.append("stale_ids")
            )
            await bootstrap.bootstrap_application(app)

        assert order == [
            "exc_handler",
            "ensure_session",
            "commands",
            "stale_ids",
            "adopt",
            "hooks",
            "wire",
            "monitor",
            "polling",
            "miniapp",
        ]


class TestShutdownRuntime:
    async def test_cancels_polling_task_and_stops_monitor(self):
        import asyncio

        async def _noop():
            return None

        bootstrap._status_poll_task = asyncio.create_task(_noop())  # type: ignore[assignment]
        monitor = MagicMock()
        monitor.stop_and_wait = AsyncMock()
        bootstrap.session_monitor = monitor

        with (
            patch(
                "ccgram.bootstrap.shutdown_workers", new_callable=AsyncMock
            ) as workers,
        ):
            await bootstrap.stop_delivery_runtime()

        monitor.stop_and_wait.assert_awaited_once()
        monitor.commit_delivered_watermarks.assert_called_once()
        workers.assert_awaited_once()
        assert bootstrap.session_monitor is None
        assert bootstrap._status_poll_task is None

        # Phase 2 (post_shutdown): no HTTP needed, flushes state.
        with (
            patch(
                "ccgram.main.stop_miniapp_if_enabled", new_callable=AsyncMock
            ) as stop_mini,
            patch("ccgram.bootstrap.session_manager") as sm,
        ):
            await bootstrap.shutdown_runtime()
        stop_mini.assert_awaited_once()
        sm.flush_state.assert_called_once()

    async def test_awaits_all_producers_before_draining_workers(self):
        order: list[str] = []
        monitor = MagicMock()
        monitor.stop_and_wait = AsyncMock(side_effect=lambda: order.append("monitor"))
        monitor.commit_delivered_watermarks.side_effect = lambda: order.append("commit")
        stream = MagicMock()
        stream.stop_and_wait = AsyncMock(side_effect=lambda: order.append("stream"))
        bootstrap.session_monitor = monitor

        with (
            patch(
                "ccgram.event_stream_monitor.get_active_event_stream",
                return_value=stream,
            ),
            patch("ccgram.event_stream_monitor.set_active_event_stream"),
            patch(
                "ccgram.bootstrap.shutdown_workers",
                new=AsyncMock(side_effect=lambda: order.append("drain")),
            ),
        ):
            await bootstrap.stop_delivery_runtime()

        assert order == ["monitor", "stream", "drain", "commit"]

    async def test_handles_no_running_components(self):
        bootstrap._status_poll_task = None
        bootstrap.session_monitor = None

        with (
            patch("ccgram.bootstrap.shutdown_workers", new_callable=AsyncMock),
            patch("ccgram.main.stop_miniapp_if_enabled", new_callable=AsyncMock),
            patch("ccgram.bootstrap.session_manager"),
        ):
            await bootstrap.shutdown_runtime()


class TestResetForTesting:
    def test_clears_module_state(self):
        bootstrap.wire_runtime_callbacks()
        bootstrap.session_monitor = MagicMock()
        bootstrap._status_poll_task = MagicMock()

        bootstrap.reset_for_testing()

        assert bootstrap._callbacks_wired is False
        assert bootstrap.session_monitor is None
        assert bootstrap._status_poll_task is None

    def test_clears_global_active_monitor_singleton(self):
        from ccgram import session_monitor as sm_mod

        monitor = MagicMock()
        sm_mod.set_active_monitor(monitor)
        bootstrap.session_monitor = monitor
        assert sm_mod.get_active_monitor() is monitor

        bootstrap.reset_for_testing()

        assert sm_mod.get_active_monitor() is None

    async def test_shutdown_runtime_clears_global_active_monitor_singleton(self):
        from ccgram import session_monitor as sm_mod

        monitor = MagicMock()
        monitor.stop_and_wait = AsyncMock()
        sm_mod.set_active_monitor(monitor)
        bootstrap.session_monitor = monitor

        with (
            patch("ccgram.bootstrap.shutdown_workers", new_callable=AsyncMock),
            patch("ccgram.main.stop_miniapp_if_enabled", new_callable=AsyncMock),
            patch("ccgram.bootstrap.session_manager"),
        ):
            # The singleton is cleared by phase 1 (producers stop).
            await bootstrap.stop_delivery_runtime()
            await bootstrap.shutdown_runtime()

        assert sm_mod.get_active_monitor() is None

    def test_resets_inner_callback_registrations(self):
        from ccgram.handlers.shell import shell_capture

        bootstrap.wire_runtime_callbacks()
        bootstrap.reset_for_testing()

        # After reset, re-wiring must succeed (i.e., the F2.6 fail-loud
        # double-registration guard sees a clean slate).
        assert shell_capture._approval_callback_registered is False

        bootstrap.wire_runtime_callbacks()
        assert shell_capture._approval_callback_registered is True


class TestVerifyHooksInstalled:
    def test_skips_when_provider_does_not_support_hooks(self):
        provider = MagicMock()
        provider.capabilities.supports_hook = False

        with patch("ccgram.bootstrap.get_provider", return_value=provider):
            bootstrap.verify_hooks_installed()

    def test_warns_when_settings_file_missing(self, tmp_path):
        provider = MagicMock()
        provider.capabilities.supports_hook = True
        provider.capabilities.name = "claude"

        missing = tmp_path / "missing.json"

        with (
            patch("ccgram.bootstrap.get_provider", return_value=provider),
            patch("ccgram.bootstrap.logger") as logger,
            patch("ccgram.hook._claude_settings_file", return_value=missing),
        ):
            bootstrap.verify_hooks_installed()

        logger.warning.assert_called_once()

    def test_logs_install_hint_for_non_claude_managed_provider(self):
        provider = MagicMock()
        provider.capabilities.supports_hook = True
        provider.capabilities.name = "codex"
        provider.capabilities.hook_install_managed_by_ccgram = True

        with (
            patch("ccgram.bootstrap.get_provider", return_value=provider),
            patch("ccgram.bootstrap.logger") as logger,
        ):
            bootstrap.verify_hooks_installed()

        # DEBUG, not INFO: an opt-in latency tip should not greet every startup.
        logger.debug.assert_called_once()
        # Message includes the provider name and the install command.
        args = logger.debug.call_args[0]
        assert "codex" in args

    def test_no_hint_for_non_managed_provider(self):
        provider = MagicMock()
        provider.capabilities.supports_hook = True
        provider.capabilities.name = "pi"
        provider.capabilities.hook_install_managed_by_ccgram = False

        with (
            patch("ccgram.bootstrap.get_provider", return_value=provider),
            patch("ccgram.bootstrap.logger") as logger,
        ):
            bootstrap.verify_hooks_installed()

        logger.info.assert_not_called()
        logger.warning.assert_not_called()


class TestNewWindowCallbackChecksEligibility:
    """The monitor's automatic creation sink re-reads the verdict per window.

    Discovery emits a batch and this callback awaits each topic creation in
    turn, so without a fresh read the last window in a batch is judged on a
    listing taken before the first one's topic existed. Explicit binds go
    straight to handle_new_window and are not affected.
    """

    @staticmethod
    async def _callback(app):
        with patch("ccgram.bootstrap.SessionMonitor") as monitor_cls:
            instance = MagicMock()
            instance.start = MagicMock()
            monitor_cls.return_value = instance
            await bootstrap.start_session_monitor(app)
        return instance.set_new_window_callback.call_args[0][0]

    async def _run(self, *, adoptable: bool):
        from ccgram.session_monitor import NewWindowEvent

        app = _make_app()
        bootstrap.wire_runtime_callbacks()
        callback = await self._callback(app)

        with (
            patch(
                "ccgram.bootstrap._still_adoptable",
                new_callable=AsyncMock,
                return_value=adoptable,
            ),
            patch(
                "ccgram.bootstrap._handle_new_window",
                new_callable=AsyncMock,
            ) as handle,
        ):
            await callback(
                NewWindowEvent(
                    window_id="@5", session_id="s", window_name="proj", cwd="/proj"
                )
            )
        return handle

    async def test_adoptable_window_gets_a_topic(self):
        handle = await self._run(adoptable=True)
        handle.assert_awaited_once()

    async def test_window_that_is_no_longer_adoptable_gets_none(self):
        handle = await self._run(adoptable=False)
        handle.assert_not_awaited()
