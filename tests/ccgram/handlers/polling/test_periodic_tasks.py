"""Time-gating of the poll loop's periodic and per-tick lifecycle tasks."""

from contextlib import contextmanager
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.polling.periodic_tasks import (
    TOPIC_CHECK_INTERVAL,
    run_lifecycle_tasks,
    run_periodic_tasks,
)
from ccgram.multiplexer.base import WindowRef

_MODULE = "ccgram.handlers.polling.periodic_tasks."
_LIVE_VIEW_INTERVAL = 5.0


@contextmanager
def _tasks(now: float):
    with (
        patch(_MODULE + "time") as mock_time,
        patch(_MODULE + "config") as mock_config,
        patch(_MODULE + "tick_live_views", new_callable=AsyncMock) as live,
        patch(_MODULE + "prune_stale_state", new_callable=AsyncMock) as prune,
        patch(_MODULE + "probe_topic_existence", new_callable=AsyncMock) as probe,
        patch(_MODULE + "check_unbound_window_ttl", new_callable=AsyncMock) as unbound,
        patch(_MODULE + "cleanup_retired_topics", new_callable=AsyncMock) as cleanup,
        patch(
            _MODULE + "recover_topic_provisioning",
            new_callable=AsyncMock,
            return_value={},
        ) as recover,
        patch(_MODULE + "log_throttle_sweep") as sweep,
    ):
        mock_time.monotonic.return_value = now
        mock_config.live_view_interval = _LIVE_VIEW_INTERVAL
        yield MagicMock(
            live=live,
            prune=prune,
            probe=probe,
            unbound=unbound,
            cleanup=cleanup,
            recover=recover,
            sweep=sweep,
        )


class TestRunPeriodicTasks:
    @pytest.mark.parametrize(
        ("elapsed", "expected"),
        [
            pytest.param(_LIVE_VIEW_INTERVAL - 0.1, False, id="within-interval"),
            pytest.param(_LIVE_VIEW_INTERVAL, True, id="exactly-at-interval"),
            pytest.param(_LIVE_VIEW_INTERVAL + 1.0, True, id="past-interval"),
        ],
    )
    async def test_live_view_tick_is_interval_gated(self, elapsed, expected):
        timers = {"live_view": 0.0, "topic_check": 1e9}
        with _tasks(now=elapsed) as tasks:
            await run_periodic_tasks(MagicMock(), [], timers)

        assert tasks.live.await_count == (1 if expected else 0)
        assert timers["live_view"] == (elapsed if expected else 0.0)

    @pytest.mark.parametrize(
        ("elapsed", "expected"),
        [
            pytest.param(TOPIC_CHECK_INTERVAL - 0.1, False, id="within-interval"),
            pytest.param(TOPIC_CHECK_INTERVAL, True, id="exactly-at-interval"),
        ],
    )
    async def test_topic_check_is_interval_gated(self, elapsed, expected):
        timers = {"live_view": 1e9, "topic_check": 0.0}
        windows = cast(list[WindowRef], [MagicMock()])
        with _tasks(now=elapsed) as tasks:
            await run_periodic_tasks(MagicMock(), windows, timers)

        assert tasks.prune.await_count == (1 if expected else 0)
        assert tasks.recover.await_count == (1 if expected else 0)
        assert tasks.probe.await_count == (1 if expected else 0)
        assert tasks.cleanup.await_count == (1 if expected else 0)
        assert tasks.sweep.call_count == (1 if expected else 0)
        assert timers["topic_check"] == (elapsed if expected else 0.0)

    async def test_topic_check_receives_the_live_window_list(self):
        timers = {"live_view": 1e9, "topic_check": 0.0}
        windows = cast(list[WindowRef], [MagicMock(), MagicMock()])
        client = MagicMock()
        with _tasks(now=TOPIC_CHECK_INTERVAL) as tasks:
            await run_periodic_tasks(client, windows, timers)

        tasks.prune.assert_awaited_once_with(windows)
        tasks.probe.assert_awaited_once_with(client)
        tasks.cleanup.assert_awaited_once_with(client, exclude_reasons=None)
        tasks.recover.assert_awaited_once_with(client)

    async def test_recovery_rate_limit_defers_other_telegram_maintenance(self):
        timers = {"live_view": 1e9, "topic_check": 0.0}
        with _tasks(now=TOPIC_CHECK_INTERVAL) as tasks:
            tasks.recover.return_value = {"rate_limited": 1}
            await run_periodic_tasks(MagicMock(), [], timers)
        tasks.prune.assert_awaited_once()
        tasks.probe.assert_not_awaited()
        tasks.cleanup.assert_not_awaited()


class TestRunLifecycleTasks:
    async def test_runs_both_lifecycle_checks_every_tick(self):
        client = MagicMock()
        windows = cast(list[WindowRef], [MagicMock()])
        with _tasks(now=0.0) as tasks:
            await run_lifecycle_tasks(client, windows)

        tasks.unbound.assert_awaited_once_with(windows)
