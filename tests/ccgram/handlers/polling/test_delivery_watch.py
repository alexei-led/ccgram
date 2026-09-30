"""Silent delivery-wedge early warning: pure decision core and wiring."""

import asyncio

from ccgram.handlers.polling import delivery_watch
from ccgram.handlers.polling.delivery_watch import (
    DeliveryGapWatch,
    check_delivery_wedges,
)
from ccgram.session_state_ports import DeliveryWatermark
from ccgram.telegram_client import FakeTelegramClient


async def _flush_alert_tasks() -> None:
    """Let fire-and-forget alert sends run to completion."""
    while delivery_watch._alert_tasks:
        await asyncio.wait_for(
            asyncio.gather(*delivery_watch._alert_tasks, return_exceptions=True),
            timeout=5,
        )


def _wire(monkeypatch, tmp_path, topics, deliveries) -> FakeTelegramClient:
    """Point the watch at synthetic bindings and delivery projections."""
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(b"x" * 4096)
    watch = DeliveryGapWatch(gap_threshold=1024, stuck_grace_s=300.0)
    monkeypatch.setattr(delivery_watch, "_watch", watch)
    monkeypatch.setattr(delivery_watch, "iter_bound_topics", lambda: topics)

    def projection(window_id):
        spec = deliveries.get(window_id)
        if spec is None:
            return None
        watermark, fenced = spec
        return DeliveryWatermark(
            watermark=watermark,
            transcript_path=str(transcript),
            fenced=fenced,
        )

    monkeypatch.setattr(delivery_watch, "get_delivery_watermark", projection)
    return FakeTelegramClient()


class TestDeliveryGapWatch:
    def watch(self) -> DeliveryGapWatch:
        return DeliveryGapWatch(gap_threshold=1000, stuck_grace_s=300.0)

    def test_small_gap_never_alerts(self) -> None:
        w = self.watch()
        assert not w.observe("@1", offset=0, size=999, now=0.0)
        assert not w.observe("@1", offset=0, size=999, now=1000.0)

    def test_alert_fires_once_after_grace(self) -> None:
        w = self.watch()
        assert not w.observe("@1", offset=0, size=2000, now=0.0)
        assert not w.observe("@1", offset=0, size=2000, now=299.0)
        assert w.observe("@1", offset=0, size=2100, now=301.0)
        assert not w.observe("@1", offset=0, size=2200, now=602.0)

    def test_offset_advance_rearms(self) -> None:
        w = self.watch()
        w.observe("@1", offset=0, size=2000, now=0.0)
        assert w.observe("@1", offset=0, size=2000, now=301.0)
        assert not w.observe("@1", offset=2000, size=2100, now=400.0)
        # New incident: first stuck observation starts the clock, the
        # next one past the grace fires again.
        assert not w.observe("@1", offset=2000, size=4000, now=701.0)
        assert w.observe("@1", offset=2000, size=4000, now=1002.0)

    def test_gap_shrink_below_threshold_rearms(self) -> None:
        w = self.watch()
        w.observe("@1", offset=0, size=2000, now=0.0)
        assert w.observe("@1", offset=0, size=2000, now=301.0)
        # Delivery catches up: gap small again, watch quiet.
        assert not w.observe("@1", offset=1900, size=2000, now=400.0)
        assert not w.observe("@1", offset=1900, size=2000, now=900.0)

    def test_disarm_rearms_the_one_shot(self) -> None:
        w = self.watch()
        w.observe("@1", offset=0, size=2000, now=0.0)
        assert w.observe("@1", offset=0, size=2000, now=301.0)
        w.disarm("@1")
        assert w.observe("@1", offset=0, size=2000, now=361.0)

    def test_forget_prunes_unbound_windows(self) -> None:
        w = self.watch()
        w.observe("@1", offset=0, size=2000, now=0.0)
        w.observe("@2", offset=0, size=2000, now=0.0)
        w.forget({"@2"})
        assert set(w._windows) == {"@2"}


class TestCheckDeliveryWedges:
    async def test_stalled_window_alerts_once_via_direct_send(
        self, monkeypatch, tmp_path
    ) -> None:
        client = _wire(
            monkeypatch,
            tmp_path,
            topics=[(7, -100200, 42, "@1")],
            deliveries={"@1": (0, False)},
        )

        # First pass: starts the stuck clock, no alert.
        monkeypatch.setattr(delivery_watch.time, "monotonic", lambda: 0.0)
        await check_delivery_wedges(client)
        await _flush_alert_tasks()
        assert client.call_count("send_message") == 0

        # Same watermark 301s later: one alert, direct (never queued).
        monkeypatch.setattr(delivery_watch.time, "monotonic", lambda: 301.0)
        await check_delivery_wedges(client)
        await _flush_alert_tasks()
        assert client.call_count("send_message") == 1
        call = client.last_call("send_message")
        assert call is not None
        assert call.kwargs["message_thread_id"] == 42
        assert "Delivery stall" in call.kwargs["text"]

        # Still stuck later: no repeat.
        monkeypatch.setattr(delivery_watch.time, "monotonic", lambda: 602.0)
        await check_delivery_wedges(client)
        await _flush_alert_tasks()
        assert client.call_count("send_message") == 1

    async def test_fenced_freeze_is_not_a_wedge(self, monkeypatch, tmp_path) -> None:
        client = _wire(
            monkeypatch,
            tmp_path,
            topics=[(7, -100200, 42, "@1")],
            deliveries={"@1": (0, True)},
        )
        monkeypatch.setattr(delivery_watch.time, "monotonic", lambda: 0.0)
        for now in (0.0, 301.0, 602.0):
            monkeypatch.setattr(delivery_watch.time, "monotonic", lambda n=now: n)
            await check_delivery_wedges(client)
        await _flush_alert_tasks()
        assert client.call_count("send_message") == 0

    async def test_failed_alert_send_retries_next_pass(
        self, monkeypatch, tmp_path
    ) -> None:
        client = _wire(
            monkeypatch,
            tmp_path,
            topics=[(7, -100200, 42, "@1")],
            deliveries={"@1": (0, False)},
        )
        attempts = {"n": 0}

        def flaky(**kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("telegram degraded")
            return True

        client.returns["send_message"] = flaky
        monkeypatch.setattr(delivery_watch.time, "monotonic", lambda: 0.0)
        await check_delivery_wedges(client)
        monkeypatch.setattr(delivery_watch.time, "monotonic", lambda: 301.0)
        await check_delivery_wedges(client)
        await _flush_alert_tasks()
        assert attempts["n"] == 1  # first alert failed...
        monkeypatch.setattr(delivery_watch.time, "monotonic", lambda: 361.0)
        await check_delivery_wedges(client)
        await _flush_alert_tasks()
        assert attempts["n"] == 2  # ...and was retried next pass

    async def test_unresolved_chat_id_does_not_consume_the_alert(
        self, monkeypatch, tmp_path
    ) -> None:
        topics: list[tuple[int, int | None, int, str]] = [(7, None, 42, "@1")]
        client = _wire(
            monkeypatch,
            tmp_path,
            topics=topics,
            deliveries={"@1": (0, False)},
        )
        monkeypatch.setattr(delivery_watch.time, "monotonic", lambda: 0.0)
        await check_delivery_wedges(client)
        monkeypatch.setattr(delivery_watch.time, "monotonic", lambda: 301.0)
        await check_delivery_wedges(client)
        await _flush_alert_tasks()
        assert client.call_count("send_message") == 0
        # The chat mapping resolves; the alert fires on the next pass.
        topics[0] = (7, -100200, 42, "@1")
        monkeypatch.setattr(delivery_watch.time, "monotonic", lambda: 361.0)
        await check_delivery_wedges(client)
        await _flush_alert_tasks()
        assert client.call_count("send_message") == 1

    async def test_unmeasurable_windows_skip(self, monkeypatch, tmp_path) -> None:
        client = _wire(
            monkeypatch,
            tmp_path,
            topics=[(7, -100200, 42, "@1")],
            deliveries={"@1": None},
        )
        monkeypatch.setattr(delivery_watch.time, "monotonic", lambda: 0.0)
        await check_delivery_wedges(client)
        await _flush_alert_tasks()
        assert client.call_count("send_message") == 0
        assert delivery_watch._watch._windows == {}
