import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.msg_broker import (
    BROKER_CYCLE_INTERVAL,
    SWEEP_INTERVAL,
    DeliveryState,
    MessageDeliveryStrategy,
    _INJECTION_CHAR_LIMIT,
    _LOOP_THRESHOLD,
    _RATE_WINDOW_SECONDS,
    _collect_eligible,
    _pair_key,
    broker_delivery_cycle,
    clear_delivery_state,
    delivery_strategy,
    format_file_reference,
    format_injection_text,
    merge_injection_texts,
    reset_delivery_state,
    write_delivery_file,
)
from ccgram.mailbox import Mailbox


@pytest.fixture(autouse=True)
def _clean_strategy():
    reset_delivery_state()
    yield
    reset_delivery_state()


@pytest.fixture()
def mailbox(tmp_path: Path) -> Mailbox:
    return Mailbox(tmp_path / "mailbox")


class TestMessageDeliveryStrategy:
    def setup_method(self):
        self.strategy = MessageDeliveryStrategy()

    def test_get_state_creates_new(self):
        state = self.strategy.get_state("ccgram:@0")
        assert isinstance(state, DeliveryState)
        assert state.delivery_timestamps == []

    def test_get_state_returns_same_instance(self):
        s1 = self.strategy.get_state("ccgram:@0")
        s2 = self.strategy.get_state("ccgram:@0")
        assert s1 is s2

    def test_clear_state_removes(self):
        self.strategy.get_state("ccgram:@0")
        self.strategy.clear_state("ccgram:@0")
        assert "ccgram:@0" not in self.strategy._states

    def test_clear_state_nonexistent_is_noop(self):
        self.strategy.clear_state("ccgram:@999")

    def test_reset_all_state(self):
        self.strategy.get_state("ccgram:@0")
        self.strategy.get_state("ccgram:@5")
        self.strategy.reset_all_state()
        assert len(self.strategy._states) == 0


class TestRateLimiting:
    def setup_method(self):
        self.strategy = MessageDeliveryStrategy()

    def test_within_rate_limit(self):
        assert self.strategy.check_rate_limit("ccgram:@0", max_rate=10)

    def test_exceeds_rate_limit(self):
        for _ in range(10):
            self.strategy.record_delivery("ccgram:@0")
        assert not self.strategy.check_rate_limit("ccgram:@0", max_rate=10)

    def test_rate_limit_window_expires(self):
        state = self.strategy.get_state("ccgram:@0")
        old_time = time.monotonic() - _RATE_WINDOW_SECONDS - 1
        state.delivery_timestamps = [old_time] * 10
        assert self.strategy.check_rate_limit("ccgram:@0", max_rate=10)

    def test_record_delivery_sets_timestamp(self):
        self.strategy.record_delivery("ccgram:@0")
        state = self.strategy.get_state("ccgram:@0")
        assert len(state.delivery_timestamps) == 1
        assert state.delivery_timestamps[0] > 0


class TestLoopDetection:
    def setup_method(self):
        self.strategy = MessageDeliveryStrategy()

    def test_no_loop_initially(self):
        assert not self.strategy.check_loop("ccgram:@0", "ccgram:@5")

    def test_loop_detected_at_threshold(self):
        for _ in range(_LOOP_THRESHOLD):
            self.strategy.record_exchange("ccgram:@0", "ccgram:@5")
        assert self.strategy.check_loop("ccgram:@0", "ccgram:@5")

    def test_loop_detection_is_symmetric(self):
        for _ in range(_LOOP_THRESHOLD):
            self.strategy.record_exchange("ccgram:@0", "ccgram:@5")
        assert self.strategy.check_loop("ccgram:@5", "ccgram:@0")

    def test_loop_below_threshold(self):
        for _ in range(_LOOP_THRESHOLD - 1):
            self.strategy.record_exchange("ccgram:@0", "ccgram:@5")
        assert not self.strategy.check_loop("ccgram:@0", "ccgram:@5")

    def test_pause_and_unpause(self):
        self.strategy.pause_peer("ccgram:@0", "ccgram:@5")
        assert self.strategy.is_paused("ccgram:@0", "ccgram:@5")
        self.strategy.unpause_peer("ccgram:@0", "ccgram:@5")
        assert not self.strategy.is_paused("ccgram:@0", "ccgram:@5")

    def test_allow_more_clears_loop_state(self):
        for _ in range(_LOOP_THRESHOLD):
            self.strategy.record_exchange("ccgram:@0", "ccgram:@5")
        self.strategy.pause_peer("ccgram:@0", "ccgram:@5")
        self.strategy.allow_more("ccgram:@0", "ccgram:@5")
        assert not self.strategy.is_paused("ccgram:@0", "ccgram:@5")
        assert not self.strategy.check_loop("ccgram:@0", "ccgram:@5")


class TestPairKey:
    def test_order_independent(self):
        assert _pair_key("ccgram:@0", "ccgram:@5") == _pair_key(
            "ccgram:@5", "ccgram:@0"
        )

    def test_format(self):
        key = _pair_key("ccgram:@0", "ccgram:@5")
        assert "|" in key


class TestFormatInjectionText:
    def test_basic_request(self):
        text = format_injection_text(
            msg_id="123-abc",
            from_id="ccgram:@0",
            from_name="payment-svc",
            branch="feat/refund",
            subject="API query",
            body="What is your API?",
            msg_type="request",
        )
        assert "[MSG 123-abc from ccgram:@0" in text
        assert "payment-svc" in text
        assert "feat/refund" in text
        assert "API query:" in text
        assert "What is your API?" in text
        assert "REPLY WITH:" in text

    def test_notify_no_reply_hint(self):
        text = format_injection_text(
            msg_id="456-def",
            from_id="ccgram:@0",
            from_name="svc",
            branch="",
            subject="",
            body="FYI done",
            msg_type="notify",
        )
        assert "REPLY WITH:" not in text

    def test_newlines_replaced(self):
        text = format_injection_text(
            msg_id="1",
            from_id="ccgram:@0",
            from_name="svc",
            branch="",
            subject="",
            body="line1\n\nline2\nline3",
            msg_type="notify",
        )
        assert "\n" not in text
        assert "line1 | line2 line3" in text

    def test_truncation_at_limit(self):
        long_body = "x" * 1000
        text = format_injection_text(
            msg_id="1",
            from_id="ccgram:@0",
            from_name="svc",
            branch="",
            subject="",
            body=long_body,
            msg_type="notify",
        )
        assert len(text) <= _INJECTION_CHAR_LIMIT
        assert text.endswith("...")

    def test_no_subject(self):
        text = format_injection_text(
            msg_id="1",
            from_id="ccgram:@0",
            from_name="svc",
            branch="",
            subject="",
            body="hello",
            msg_type="notify",
        )
        assert "svc)]" in text
        assert text.count(":") == 1  # only the colon in "ccgram:@0"


class TestFormatFileReference:
    def test_format(self):
        ref = format_file_reference("123-abc", "/tmp/deliver-123-abc.txt")
        assert "[MSG 123-abc]" in ref
        assert "/tmp/deliver-123-abc.txt" in ref


class TestMergeInjectionTexts:
    def test_single(self):
        assert merge_injection_texts(["hello"]) == "hello"

    def test_multiple(self):
        result = merge_injection_texts(["msg1", "msg2", "msg3"])
        assert result == "msg1 --- msg2 --- msg3"


class TestWriteDeliveryFile:
    def test_writes_file(self, tmp_path):
        path = write_delivery_file(tmp_path, "ccgram:@0", "123-abc", "long body text")
        assert path.exists()
        assert path.read_text() == "long body text"
        assert "deliver-123-abc.txt" in path.name

    def test_creates_directories(self, tmp_path):
        mailbox_dir = tmp_path / "mailbox"
        path = write_delivery_file(mailbox_dir, "ccgram:@0", "456", "body")
        assert path.exists()
        assert path.parent.name == "tmp"


class TestCollectEligible:
    def test_returns_pending_non_broadcast(self, mailbox):
        mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        eligible, loops = _collect_eligible(mailbox, "ccgram:@5", msg_rate_limit=10)
        assert len(eligible) == 1
        assert eligible[0].type == "request"
        assert loops == []

    def test_skips_broadcast(self, mailbox):
        mailbox.send("ccgram:@0", "ccgram:@5", "broadcast msg", msg_type="broadcast")
        eligible, _loops = _collect_eligible(mailbox, "ccgram:@5", msg_rate_limit=10)
        assert len(eligible) == 0

    def test_skips_paused_peer(self, mailbox):
        mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        delivery_strategy.pause_peer("ccgram:@5", "ccgram:@0")
        eligible, _loops = _collect_eligible(mailbox, "ccgram:@5", msg_rate_limit=10)
        assert len(eligible) == 0

    def test_respects_rate_limit(self, mailbox):
        mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        for _ in range(10):
            delivery_strategy.record_delivery("ccgram:@5")
        eligible, _loops = _collect_eligible(mailbox, "ccgram:@5", msg_rate_limit=10)
        assert len(eligible) == 0

    def test_loop_detection_pauses_peer(self, mailbox):
        mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        for _ in range(_LOOP_THRESHOLD):
            delivery_strategy.record_exchange("ccgram:@5", "ccgram:@0")
        eligible, loops = _collect_eligible(mailbox, "ccgram:@5", msg_rate_limit=10)
        assert len(eligible) == 0
        assert len(loops) == 1
        assert loops[0] == ("ccgram:@5", "ccgram:@0")

    def test_empty_inbox(self, mailbox):
        eligible, loops = _collect_eligible(mailbox, "ccgram:@5", msg_rate_limit=10)
        assert eligible == []
        assert loops == []


class TestBrokerDeliveryCycle:
    @pytest.fixture()
    def mock_tmux(self):
        mgr = AsyncMock()
        mgr.send_keys = AsyncMock(return_value=True)
        return mgr

    @pytest.fixture()
    def mock_provider(self):
        provider = MagicMock()
        provider.capabilities.name = "claude"
        provider.capabilities.supports_hook = True
        return provider

    @pytest.fixture()
    def shell_provider(self):
        provider = MagicMock()
        provider.capabilities.name = "shell"
        return provider

    async def test_delivers_pending_message(
        self, mailbox, mock_tmux, mock_provider, tmp_path
    ):
        mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        window_states = {"@5": MagicMock(provider_name="claude")}

        with patch(
            "ccgram.providers.get_provider_for_window",
            return_value=mock_provider,
        ):
            count = await broker_delivery_cycle(
                mailbox,
                mock_tmux,
                window_states,
                "ccgram",
                10,
                tmp_path,
                idle_windows=frozenset({"ccgram:@5"}),
            )

        assert count == 1
        mock_tmux.send_keys.assert_called_once()
        msgs = mailbox.inbox("ccgram:@5")
        assert len(msgs) == 0 or msgs[0].status == "delivered"

    async def test_skips_shell_windows(
        self, mailbox, mock_tmux, shell_provider, tmp_path
    ):
        mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        window_states = {"@5": MagicMock(provider_name="shell")}

        with patch(
            "ccgram.providers.get_provider_for_window",
            return_value=shell_provider,
        ):
            count = await broker_delivery_cycle(
                mailbox, mock_tmux, window_states, "ccgram", 10, tmp_path
            )

        assert count == 0
        mock_tmux.send_keys.assert_not_called()

    async def test_skips_broadcast_messages(
        self, mailbox, mock_tmux, mock_provider, tmp_path
    ):
        mailbox.send("ccgram:@0", "ccgram:@5", "broadcast", msg_type="broadcast")
        window_states = {"@5": MagicMock(provider_name="claude")}

        with patch(
            "ccgram.providers.get_provider_for_window",
            return_value=mock_provider,
        ):
            count = await broker_delivery_cycle(
                mailbox,
                mock_tmux,
                window_states,
                "ccgram",
                10,
                tmp_path,
                idle_windows=frozenset({"ccgram:@5"}),
            )

        assert count == 0
        mock_tmux.send_keys.assert_not_called()

    async def test_sets_delivered_at(self, mailbox, mock_tmux, mock_provider, tmp_path):
        msg = mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        window_states = {"@5": MagicMock(provider_name="claude")}

        with patch(
            "ccgram.providers.get_provider_for_window",
            return_value=mock_provider,
        ):
            await broker_delivery_cycle(
                mailbox,
                mock_tmux,
                window_states,
                "ccgram",
                10,
                tmp_path,
                idle_windows=frozenset({"ccgram:@5"}),
            )

        all_msgs = mailbox.all_messages("ccgram:@5")
        delivered = [m for m in all_msgs if m.id == msg.id]
        assert len(delivered) == 1
        assert delivered[0].delivered_at is not None
        assert delivered[0].status == "delivered"

    async def test_rate_limiting_enforcement(
        self, mailbox, mock_tmux, mock_provider, tmp_path
    ):
        mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        window_states = {"@5": MagicMock(provider_name="claude")}
        for _ in range(10):
            delivery_strategy.record_delivery("ccgram:@5")

        with patch(
            "ccgram.providers.get_provider_for_window",
            return_value=mock_provider,
        ):
            count = await broker_delivery_cycle(
                mailbox,
                mock_tmux,
                window_states,
                "ccgram",
                10,
                tmp_path,
                idle_windows=frozenset({"ccgram:@5"}),
            )

        assert count == 0

    async def test_merges_multiple_messages(
        self, mailbox, mock_tmux, mock_provider, tmp_path
    ):
        mailbox.send("ccgram:@0", "ccgram:@5", "msg1", msg_type="request")
        mailbox.send("ccgram:@0", "ccgram:@5", "msg2", msg_type="notify")
        window_states = {"@5": MagicMock(provider_name="claude")}

        with patch(
            "ccgram.providers.get_provider_for_window",
            return_value=mock_provider,
        ):
            count = await broker_delivery_cycle(
                mailbox,
                mock_tmux,
                window_states,
                "ccgram",
                10,
                tmp_path,
                idle_windows=frozenset({"ccgram:@5"}),
            )

        assert count == 2
        mock_tmux.send_keys.assert_called_once()
        call_text = mock_tmux.send_keys.call_args[0][1]
        assert "---" in call_text

    async def test_file_reference_for_long_body(
        self, mailbox, mock_tmux, mock_provider, tmp_path
    ):
        long_body = "x" * 600
        mailbox.send("ccgram:@0", "ccgram:@5", long_body, msg_type="notify")
        window_states = {"@5": MagicMock(provider_name="claude")}

        with patch(
            "ccgram.providers.get_provider_for_window",
            return_value=mock_provider,
        ):
            count = await broker_delivery_cycle(
                mailbox,
                mock_tmux,
                window_states,
                "ccgram",
                10,
                tmp_path,
                idle_windows=frozenset({"ccgram:@5"}),
            )

        assert count == 1
        call_text = mock_tmux.send_keys.call_args[0][1]
        assert "See:" in call_text

    async def test_loop_detection_pauses_delivery(
        self, mailbox, mock_tmux, mock_provider, tmp_path
    ):
        mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        for _ in range(_LOOP_THRESHOLD):
            delivery_strategy.record_exchange("ccgram:@5", "ccgram:@0")
        window_states = {"@5": MagicMock(provider_name="claude")}

        with patch(
            "ccgram.providers.get_provider_for_window",
            return_value=mock_provider,
        ):
            count = await broker_delivery_cycle(
                mailbox,
                mock_tmux,
                window_states,
                "ccgram",
                10,
                tmp_path,
                idle_windows=frozenset({"ccgram:@5"}),
            )

        assert count == 0

    async def test_failed_send_keys_no_delivery(self, mailbox, mock_provider, tmp_path):
        mock_tmux = AsyncMock()
        mock_tmux.send_keys = AsyncMock(return_value=False)
        mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        window_states = {"@5": MagicMock(provider_name="claude")}

        with patch(
            "ccgram.providers.get_provider_for_window",
            return_value=mock_provider,
        ):
            count = await broker_delivery_cycle(
                mailbox,
                mock_tmux,
                window_states,
                "ccgram",
                10,
                tmp_path,
                idle_windows=frozenset({"ccgram:@5"}),
            )

        assert count == 0
        pending = mailbox.inbox("ccgram:@5")
        assert len(pending) == 1
        assert pending[0].status == "pending"


class TestCrashRecovery:
    def test_pending_undelivered_found_on_startup(self, mailbox):
        msg = mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        undelivered = mailbox.pending_undelivered(min_age_seconds=0)
        assert len(undelivered) == 1
        assert undelivered[0].id == msg.id

    def test_delivered_messages_excluded(self, mailbox):
        msg = mailbox.send("ccgram:@0", "ccgram:@5", "hello", msg_type="request")
        mailbox.mark_delivered(msg.id, "ccgram:@5")
        undelivered = mailbox.pending_undelivered(min_age_seconds=0)
        assert len(undelivered) == 0


class TestModuleLevelFunctions:
    def test_clear_delivery_state(self):
        delivery_strategy.get_state("ccgram:@0")
        clear_delivery_state("ccgram:@0")
        assert "ccgram:@0" not in delivery_strategy._states

    def test_reset_delivery_state(self):
        delivery_strategy.get_state("ccgram:@0")
        delivery_strategy.get_state("ccgram:@5")
        reset_delivery_state()
        assert len(delivery_strategy._states) == 0


class TestConstants:
    def test_broker_cycle_interval(self):
        assert BROKER_CYCLE_INTERVAL > 0

    def test_sweep_interval(self):
        assert SWEEP_INTERVAL > 0

    def test_injection_char_limit(self):
        assert _INJECTION_CHAR_LIMIT == 500
