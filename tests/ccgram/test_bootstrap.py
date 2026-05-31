from unittest.mock import AsyncMock, MagicMock, patch

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
        monitor.stop = MagicMock()
        bootstrap.session_monitor = monitor

        with (
            patch(
                "ccgram.bootstrap.shutdown_workers", new_callable=AsyncMock
            ) as workers,
            patch(
                "ccgram.main.stop_miniapp_if_enabled", new_callable=AsyncMock
            ) as stop_mini,
            patch("ccgram.bootstrap.session_manager") as sm,
        ):
            await bootstrap.shutdown_runtime()

        monitor.stop.assert_called_once()
        workers.assert_awaited_once()
        stop_mini.assert_awaited_once()
        sm.flush_state.assert_called_once()
        assert bootstrap.session_monitor is None
        assert bootstrap._status_poll_task is None

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
        monitor.stop = MagicMock()
        sm_mod.set_active_monitor(monitor)
        bootstrap.session_monitor = monitor

        with (
            patch("ccgram.bootstrap.shutdown_workers", new_callable=AsyncMock),
            patch("ccgram.main.stop_miniapp_if_enabled", new_callable=AsyncMock),
            patch("ccgram.bootstrap.session_manager"),
        ):
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
