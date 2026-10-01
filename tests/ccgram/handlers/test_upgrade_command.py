"""Tests for the /upgrade restart stop path."""

from unittest.mock import AsyncMock, MagicMock, patch


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
        patch("ccgram.main._restart_requested", False),
    ):
        await upgrade_module.upgrade_command(update, context)

        arm_watchdog.assert_called_once_with()
        context.application.stop_running.assert_called_once_with()

        from ccgram import main as main_module

        assert main_module._restart_requested is True
