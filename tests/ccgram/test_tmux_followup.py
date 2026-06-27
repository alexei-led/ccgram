import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from ccgram.multiplexer.window_ops import send_followup_to_window, send_to_window


async def test_send_to_window_times_out_when_tmux_send_hangs(monkeypatch) -> None:
    never = asyncio.Event()

    async def hang_send_keys(*_args, **_kwargs) -> bool:
        await never.wait()
        return True

    monkeypatch.setattr(
        "ccgram.multiplexer.window_ops.SEND_KEYS_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    mock_router = SimpleNamespace(get_display_name=MagicMock(return_value="project"))
    with (
        patch("ccgram.multiplexer.window_ops.thread_router", new=mock_router),
        patch("ccgram.multiplexer.window_ops.multiplexer") as mock_tmux,
    ):
        mock_tmux.find_window_by_id = AsyncMock(
            return_value=SimpleNamespace(window_id="@1")
        )
        mock_tmux.send_keys = AsyncMock(side_effect=hang_send_keys)

        success, message = await asyncio.wait_for(
            send_to_window("@1", "run tests"), timeout=0.2
        )

    assert success is False
    assert message == "Timed out sending keys to project"


async def test_send_to_window_times_out_when_window_lookup_hangs(monkeypatch) -> None:
    never = asyncio.Event()

    async def hang_find_window(*_args, **_kwargs):
        await never.wait()
        return SimpleNamespace(window_id="@1")

    monkeypatch.setattr(
        "ccgram.multiplexer.window_ops.SEND_KEYS_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    mock_router = SimpleNamespace(get_display_name=MagicMock(return_value="project"))
    with (
        patch("ccgram.multiplexer.window_ops.thread_router", new=mock_router),
        patch("ccgram.multiplexer.window_ops.multiplexer") as mock_tmux,
    ):
        mock_tmux.find_window_by_id = AsyncMock(side_effect=hang_find_window)
        mock_tmux.send_keys = AsyncMock(return_value=True)

        success, message = await asyncio.wait_for(
            send_to_window("@1", "run tests"), timeout=0.2
        )

    assert success is False
    assert message == "Timed out sending keys to project"
    mock_tmux.send_keys.assert_not_called()


async def test_send_followup_to_window_sends_text_then_alt_enter() -> None:
    mock_router = SimpleNamespace(get_display_name=MagicMock(return_value="project"))
    with (
        patch("ccgram.multiplexer.window_ops.thread_router", new=mock_router),
        patch("ccgram.multiplexer.window_ops.multiplexer") as mock_tmux,
        patch(
            "ccgram.multiplexer.window_ops.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
    ):
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
    mock_router = SimpleNamespace(get_display_name=MagicMock(return_value="project"))
    with (
        patch("ccgram.multiplexer.window_ops.thread_router", new=mock_router),
        patch("ccgram.multiplexer.window_ops.multiplexer") as mock_tmux,
    ):
        mock_tmux.find_window_by_id = AsyncMock(return_value=None)
        mock_tmux.send_keys = AsyncMock(return_value=True)

        success, message = await send_followup_to_window("@missing", "run tests")

    assert success is False
    assert message == "Window not found (may have been closed)"
    mock_tmux.send_keys.assert_not_called()
