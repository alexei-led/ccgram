"""Tests for the backend-neutral send helpers (``multiplexer/window_ops.py``)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from ccgram.multiplexer.window_ops import send_followup_to_window, send_to_window


@pytest.fixture
def mux(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """The active multiplexer, answering for window ``@1`` named "project"."""
    monkeypatch.setattr(
        "ccgram.multiplexer.window_ops.thread_router",
        SimpleNamespace(get_display_name=MagicMock(return_value="project")),
    )
    backend = MagicMock()
    backend.find_window_by_id = AsyncMock(return_value=SimpleNamespace(window_id="@1"))
    backend.send_keys = AsyncMock(return_value=True)
    monkeypatch.setattr("ccgram.multiplexer.window_ops.multiplexer", backend)
    return backend


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    sleep = AsyncMock()
    monkeypatch.setattr("ccgram.multiplexer.window_ops.asyncio.sleep", sleep)
    return sleep


class TestSendToWindow:
    @pytest.mark.parametrize("raw", [False, True])
    async def test_sends_text_to_the_resolved_window(
        self, mux: MagicMock, raw: bool
    ) -> None:
        assert await send_to_window("@1", "run tests", raw=raw) == (
            True,
            "Sent to project",
        )
        mux.send_keys.assert_awaited_once_with("@1", "run tests", raw=raw)

    async def test_reports_missing_window_without_sending(self, mux: MagicMock) -> None:
        mux.find_window_by_id = AsyncMock(return_value=None)

        assert await send_to_window("@missing", "run tests") == (
            False,
            "Window not found (may have been closed)",
        )
        mux.send_keys.assert_not_called()

    async def test_reports_a_rejected_send(self, mux: MagicMock) -> None:
        mux.send_keys = AsyncMock(return_value=False)

        assert await send_to_window("@1", "run tests") == (False, "Failed to send keys")

    @pytest.mark.parametrize("hanging_call", ["find_window_by_id", "send_keys"])
    async def test_times_out_instead_of_blocking_the_caller(
        self, mux: MagicMock, monkeypatch: pytest.MonkeyPatch, hanging_call: str
    ) -> None:
        never = asyncio.Event()

        async def hang(*_args, **_kwargs):
            await never.wait()

        monkeypatch.setattr(
            "ccgram.multiplexer.window_ops.SEND_KEYS_TIMEOUT_SECONDS", 0.01
        )
        setattr(mux, hanging_call, AsyncMock(side_effect=hang))

        result = await asyncio.wait_for(send_to_window("@1", "run tests"), timeout=0.2)

        assert result == (False, "Timed out sending keys to project")
        if hanging_call == "find_window_by_id":
            mux.send_keys.assert_not_called()


class TestSendFollowupToWindow:
    async def test_sends_text_then_alt_enter(
        self, mux: MagicMock, no_sleep: AsyncMock
    ) -> None:
        assert await send_followup_to_window("@1", "run tests") == (
            True,
            "Follow-up queued for project",
        )
        no_sleep.assert_awaited_once_with(0.5)
        mux.send_keys.assert_has_awaits(
            [
                call("@1", "run tests", enter=False, literal=True),
                call("@1", "M-Enter", enter=False, literal=False),
            ]
        )

    async def test_reports_missing_window(self, mux: MagicMock) -> None:
        mux.find_window_by_id = AsyncMock(return_value=None)

        assert await send_followup_to_window("@missing", "run tests") == (
            False,
            "Window not found (may have been closed)",
        )
        mux.send_keys.assert_not_called()

    async def test_rejected_text_send_never_queues_the_alt_enter(
        self, mux: MagicMock, no_sleep: AsyncMock
    ) -> None:
        mux.send_keys = AsyncMock(return_value=False)

        assert await send_followup_to_window("@1", "run tests") == (
            False,
            "Failed to send follow-up text",
        )
        mux.send_keys.assert_awaited_once_with(
            "@1", "run tests", enter=False, literal=True
        )
        no_sleep.assert_not_awaited()

    async def test_rejected_alt_enter_is_reported(
        self, mux: MagicMock, no_sleep: AsyncMock
    ) -> None:
        mux.send_keys = AsyncMock(side_effect=[True, False])

        assert await send_followup_to_window("@1", "run tests") == (
            False,
            "Failed to send follow-up key",
        )
