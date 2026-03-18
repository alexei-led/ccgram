import asyncio
import logging

import pytest
import structlog

from ccgram.utils import task_done_callback


@pytest.fixture(autouse=True)
def _configure_structlog_for_caplog():
    """Configure structlog to route through stdlib logging so caplog works."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    yield
    structlog.reset_defaults()


async def test_task_done_callback_logs_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _failing() -> None:
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        task = asyncio.create_task(_failing())
        task.add_done_callback(task_done_callback)
        with pytest.raises(RuntimeError):
            await task
    assert "boom" in caplog.text
    assert "Background task" in caplog.text


async def test_task_done_callback_ignores_cancelled() -> None:
    async def _forever() -> None:
        await asyncio.sleep(999)

    task = asyncio.create_task(_forever())
    task.add_done_callback(task_done_callback)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_task_done_callback_ignores_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _ok() -> None:
        pass

    with caplog.at_level(logging.ERROR):
        task = asyncio.create_task(_ok())
        task.add_done_callback(task_done_callback)
        await task
    assert "Background task" not in caplog.text


async def test_backoff_constants_session_monitor() -> None:
    from ccgram.session_monitor import _BACKOFF_MAX, _BACKOFF_MIN

    assert _BACKOFF_MIN == 2.0
    assert _BACKOFF_MAX == 30.0
    for streak in range(10):
        delay = min(_BACKOFF_MAX, _BACKOFF_MIN * (2**streak))
        assert delay <= _BACKOFF_MAX


async def test_backoff_constants_status_polling() -> None:
    from ccgram.handlers.status_polling import _BACKOFF_MAX, _BACKOFF_MIN

    assert _BACKOFF_MIN == 2.0
    assert _BACKOFF_MAX == 30.0


async def test_backoff_doubles_on_consecutive_errors() -> None:
    from ccgram.session_monitor import _BACKOFF_MAX, _BACKOFF_MIN

    delays = []
    for streak in range(5):
        delays.append(min(_BACKOFF_MAX, _BACKOFF_MIN * (2**streak)))
    assert delays[0] == 2.0
    assert delays[1] == 4.0
    assert delays[2] == 8.0
    assert delays[3] == 16.0
    assert delays[4] == 30.0
