"""Tests for the /upgrade restart stop path."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch


class _FakeProcess:
    returncode = 0

    def __init__(self, output: bytes) -> None:
        self._output = output

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._output, b""

    def kill(self) -> None:  # pragma: no cover - only on timeout
        raise AssertionError("kill must not be called")


async def test_upgrade_restart_arms_the_shutdown_watchdog() -> None:
    """A wedge in this stop path must still exit for the supervisor."""
    from ccgram.handlers import upgrade as upgrade_module

    update = MagicMock()
    update.effective_user.id = 42
    update.message.reply_text = AsyncMock(return_value=MagicMock())
    context = MagicMock()
    operations: list[str] = []
    request_restart = Mock(side_effect=lambda: operations.append("request"))
    context.bot_data = {"request_restart": request_restart}

    with (
        patch.object(upgrade_module.config, "is_user_allowed", return_value=True),
        patch.object(upgrade_module, "safe_edit", new_callable=AsyncMock),
        patch.object(
            upgrade_module.asyncio,
            "create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=_FakeProcess(b"Upgraded ccgram v1.0.0 -> v1.0.1"),
        ),
        patch("ccgram.bot.arm_shutdown_watchdog") as arm_watchdog,
    ):
        arm_watchdog.side_effect = lambda: operations.append("watchdog")
        context.application.stop_running.side_effect = lambda: operations.append("stop")

        await upgrade_module.upgrade_command(update, context)

        request_restart.assert_called_once_with()
        arm_watchdog.assert_called_once_with()
        context.application.stop_running.assert_called_once_with()
        assert operations == ["request", "watchdog", "stop"]


async def test_upgrade_without_restart_wiring_reports_manual_restart() -> None:
    from ccgram.handlers import upgrade as upgrade_module

    update = MagicMock()
    update.effective_user.id = 42
    update.message.reply_text = AsyncMock(return_value=MagicMock())
    context = MagicMock()
    context.bot_data = {}

    with (
        patch.object(upgrade_module.config, "is_user_allowed", return_value=True),
        patch.object(upgrade_module, "safe_edit", new_callable=AsyncMock) as safe_edit,
        patch.object(
            upgrade_module.asyncio,
            "create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=_FakeProcess(b"Upgraded ccgram v1.0.0 -> v1.0.1"),
        ),
        patch("ccgram.bot.arm_shutdown_watchdog") as arm_watchdog,
    ):
        await upgrade_module.upgrade_command(update, context)

    safe_edit.assert_awaited_once()
    edit_call = safe_edit.await_args
    assert edit_call is not None
    assert "restart ccgram manually" in edit_call.args[1].lower()
    arm_watchdog.assert_not_called()
    context.application.stop_running.assert_not_called()


async def test_upgrade_no_update_does_not_request_restart() -> None:
    from ccgram.handlers import upgrade as upgrade_module

    update = MagicMock()
    update.effective_user.id = 42
    update.message.reply_text = AsyncMock(return_value=MagicMock())
    context = MagicMock()
    context.bot_data = {}

    with (
        patch.object(upgrade_module.config, "is_user_allowed", return_value=True),
        patch.object(upgrade_module, "safe_edit", new_callable=AsyncMock) as safe_edit,
        patch.object(
            upgrade_module.asyncio,
            "create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=_FakeProcess(b"Nothing to upgrade"),
        ),
        patch("ccgram.bot.arm_shutdown_watchdog") as arm_watchdog,
    ):
        await upgrade_module.upgrade_command(update, context)

    edit_call = safe_edit.await_args
    assert edit_call is not None
    assert "Already up to date" in edit_call.args[1]
    arm_watchdog.assert_not_called()
    context.application.stop_running.assert_not_called()


async def test_upgrade_failure_does_not_request_restart() -> None:
    from ccgram.handlers import upgrade as upgrade_module

    update = MagicMock()
    update.effective_user.id = 42
    update.message.reply_text = AsyncMock(return_value=MagicMock())
    context = MagicMock()
    context.bot_data = {}
    process = _FakeProcess(b"permission denied")
    process.returncode = 1

    with (
        patch.object(upgrade_module.config, "is_user_allowed", return_value=True),
        patch.object(upgrade_module, "safe_edit", new_callable=AsyncMock) as safe_edit,
        patch.object(
            upgrade_module.asyncio,
            "create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ),
        patch("ccgram.bot.arm_shutdown_watchdog") as arm_watchdog,
    ):
        await upgrade_module.upgrade_command(update, context)

    edit_call = safe_edit.await_args
    assert edit_call is not None
    assert "Upgrade failed (exit 1)" in edit_call.args[1]
    arm_watchdog.assert_not_called()
    context.application.stop_running.assert_not_called()
