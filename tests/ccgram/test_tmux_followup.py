from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

from ccgram.multiplexer.window_ops import send_followup_to_window


async def test_send_followup_to_window_sends_text_then_alt_enter() -> None:
    with (
        patch("ccgram.multiplexer.window_ops.thread_router") as mock_router,
        patch("ccgram.multiplexer.window_ops.multiplexer") as mock_tmux,
        patch(
            "ccgram.multiplexer.window_ops.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
    ):
        mock_router.get_display_name.return_value = "project"
        mock_tmux.find_window_by_id = AsyncMock(
            return_value=SimpleNamespace(window_id="@1")
        )
        mock_tmux.send_keys = AsyncMock(return_value=True)

        success, message = await send_followup_to_window("@1", "run tests")

    assert success is True
    assert message == "Follow-up queued for project"
    sleep.assert_awaited_once_with(0.5)
    mock_tmux.send_keys.assert_has_awaits(
        [
            call("@1", "run tests", enter=False, literal=True),
            call("@1", "M-Enter", enter=False, literal=False),
        ]
    )


async def test_send_followup_to_window_reports_missing_window() -> None:
    with (
        patch("ccgram.multiplexer.window_ops.thread_router") as mock_router,
        patch("ccgram.multiplexer.window_ops.multiplexer") as mock_tmux,
    ):
        mock_router.get_display_name.return_value = "project"
        mock_tmux.find_window_by_id = AsyncMock(return_value=None)
        mock_tmux.send_keys = AsyncMock(return_value=True)

        success, message = await send_followup_to_window("@missing", "run tests")

    assert success is False
    assert message == "Window not found (may have been closed)"
    mock_tmux.send_keys.assert_not_called()
