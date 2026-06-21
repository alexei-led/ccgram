"""Tests for PaneStatusStrategy: classification, transitions, dead-pane reconciliation,
state upserts, and the async scan_window orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot

from ccgram.handlers.polling.polling_state import (
    InteractiveUIStrategy,
    PaneStatusStrategy,
    TerminalPollState,
    TerminalScreenBuffer,
)
from ccgram.handlers.polling.polling_types import PaneTransition
from ccgram.providers.base import StatusUpdate
from ccgram.multiplexer.base import PaneInfo as TmuxPaneInfo
from ccgram.window_state_store import PaneInfo, window_store


def _require_pane(window_id: str, pane_id: str) -> PaneInfo:
    pane = window_store.get_pane(window_id, pane_id)
    assert pane is not None, f"expected pane {pane_id} in window {window_id}"
    return pane


def _pane(
    pane_id: str = "%1",
    *,
    active: bool = True,
    index: int = 0,
    command: str = "claude",
) -> TmuxPaneInfo:
    return TmuxPaneInfo(
        pane_id=pane_id,
        index=index,
        active=active,
        command=command,
        path="/tmp",
        width=80,
        height=24,
    )


def _interactive_status(text: str = "Allow?") -> StatusUpdate:
    return StatusUpdate(
        raw_text=text,
        display_label=text,
        is_interactive=True,
        ui_type="PermissionPrompt",
    )


def _idle_provider(window_name: str = "claude") -> MagicMock:
    provider = MagicMock()
    provider.capabilities.name = window_name
    provider.parse_terminal_status.return_value = None
    return provider


def _interactive_provider(prompt: str = "Allow?") -> MagicMock:
    provider = MagicMock()
    provider.capabilities.name = "claude"
    provider.parse_terminal_status.return_value = _interactive_status(prompt)
    return provider


@pytest.fixture
def strategy() -> PaneStatusStrategy:
    poll_state = TerminalPollState()
    return PaneStatusStrategy(TerminalScreenBuffer(poll_state), InteractiveUIStrategy())


@pytest.fixture(autouse=True)
def _reset_window_store():
    window_store.reset()
    saved_schedule = window_store._schedule_save
    window_store._schedule_save = lambda: None
    yield
    window_store.reset()
    window_store._schedule_save = saved_schedule


class TestClassifyPane:
    def test_blocked_when_status_is_interactive(self) -> None:
        assert (
            PaneStatusStrategy.classify_pane(False, _interactive_status()) == "blocked"
        )

    def test_blocked_wins_over_active(self) -> None:
        assert (
            PaneStatusStrategy.classify_pane(True, _interactive_status()) == "blocked"
        )

    def test_active_when_active_and_no_status(self) -> None:
        assert PaneStatusStrategy.classify_pane(True, None) == "active"

    def test_idle_when_inactive_and_no_status(self) -> None:
        assert PaneStatusStrategy.classify_pane(False, None) == "idle"

    def test_idle_when_inactive_and_non_interactive_status(self) -> None:
        non_interactive = StatusUpdate(
            raw_text="working...", display_label="working", is_interactive=False
        )
        assert PaneStatusStrategy.classify_pane(False, non_interactive) == "idle"


class TestRecordPaneState:
    def test_returns_none_for_first_record(self, strategy: PaneStatusStrategy) -> None:
        prev = strategy.record_pane_state("@0", "%1", "active", provider="claude")
        assert prev is None
        pane = window_store.get_pane("@0", "%1")
        assert pane is not None
        assert pane.state == "active"
        assert pane.provider == "claude"

    def test_returns_prev_state_on_update(self, strategy: PaneStatusStrategy) -> None:
        strategy.record_pane_state("@0", "%1", "idle", provider="claude")
        prev = strategy.record_pane_state("@0", "%1", "blocked", provider="claude")
        assert prev == "idle"
        assert _require_pane("@0", "%1").state == "blocked"

    def test_updates_last_active_ts(self, strategy: PaneStatusStrategy) -> None:
        strategy.record_pane_state(
            "@0", "%1", "active", provider="claude", last_active_ts=1234.5
        )
        pane = _require_pane("@0", "%1")
        assert pane.last_active_ts == 1234.5


class TestReconcileDeadPanes:
    def test_returns_empty_when_window_unknown(
        self, strategy: PaneStatusStrategy
    ) -> None:
        assert strategy.reconcile_dead_panes("@unknown", {"%1"}) == []

    def test_drops_panes_not_in_live_set(self, strategy: PaneStatusStrategy) -> None:
        strategy.record_pane_state("@0", "%1", "active", provider="claude")
        strategy.record_pane_state("@0", "%2", "idle", provider="claude")
        strategy.record_pane_state("@0", "%3", "idle", provider="claude")
        gone = strategy.reconcile_dead_panes("@0", {"%1"})
        # reconcile_dead_panes returns (pane_id, name) tuples so callers
        # can preserve the user-assigned name in lifecycle notifications
        # after the PaneInfo is removed.
        assert sorted(pid for pid, _ in gone) == ["%2", "%3"]
        assert window_store.get_pane("@0", "%2") is None
        assert window_store.get_pane("@0", "%3") is None
        assert window_store.get_pane("@0", "%1") is not None

    def test_purges_alerts_for_gone_panes(self, strategy: PaneStatusStrategy) -> None:
        strategy.record_pane_state("@0", "%2", "blocked", provider="claude")
        strategy._interactive.set_pane_alert("%2", "Allow?", 100.0, "@0")
        strategy.reconcile_dead_panes("@0", set())
        assert not strategy._interactive.has_pane_alert("%2")


class TestScanWindowSinglePane:
    async def test_records_active_single_pane(
        self, strategy: PaneStatusStrategy
    ) -> None:
        # Mark the window scanned so first-scan suppression doesn't drop
        # the "created" transition this test is verifying.
        strategy._scanned_windows.add("@0")
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        with patch("ccgram.multiplexer.multiplexer") as mock_tm:
            mock_tm.list_panes = AsyncMock(return_value=[_pane("%1", active=True)])
            transitions = await strategy.scan_window(
                bot, 1, "@0", 42, on_blocked=on_blocked
            )
        on_blocked.assert_not_called()
        pane = window_store.get_pane("@0", "%1")
        assert pane is not None
        assert pane.state == "active"
        assert any(t.pane_id == "%1" and t.new_state == "active" for t in transitions)

    async def test_first_scan_suppresses_created_transition(
        self, strategy: PaneStatusStrategy
    ) -> None:
        # First scan must not announce an existing pane as freshly born;
        # this prevents bot restart from spamming "pane created" for
        # single-pane windows that were already alive.
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        with patch("ccgram.multiplexer.multiplexer") as mock_tm:
            mock_tm.list_panes = AsyncMock(return_value=[_pane("%1", active=True)])
            transitions = await strategy.scan_window(
                bot, 1, "@0", 42, on_blocked=on_blocked
            )
        # The pane state still gets recorded for future transition tracking,
        # but no transition is surfaced for the "created" event.
        assert window_store.get_pane("@0", "%1") is not None
        assert all(
            not (t.pane_id == "%1" and t.prev_state is None) for t in transitions
        )

    async def test_returns_no_transitions_when_state_unchanged(
        self, strategy: PaneStatusStrategy
    ) -> None:
        strategy.record_pane_state("@0", "%1", "active", provider="claude")
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        with patch("ccgram.multiplexer.multiplexer") as mock_tm:
            mock_tm.list_panes = AsyncMock(return_value=[_pane("%1", active=True)])
            transitions = await strategy.scan_window(
                bot, 1, "@0", 42, on_blocked=on_blocked
            )
        assert all(t.pane_id != "%1" for t in transitions)


class TestScanWindowMultiPane:
    async def test_active_pane_recorded_and_skips_capture(
        self, strategy: PaneStatusStrategy
    ) -> None:
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        provider = _idle_provider()
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[
                    _pane("%1", active=True),
                    _pane("%2", active=False, index=1),
                ]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="text")
            await strategy.scan_window(bot, 1, "@0", 42, on_blocked=on_blocked)
        # Active pane is never captured
        mock_tm.capture_pane_by_id.assert_called_once_with("%2", window_id="@0")
        assert _require_pane("@0", "%1").state == "active"
        assert _require_pane("@0", "%2").state == "idle"

    async def test_blocked_pane_surfaces_alert(
        self, strategy: PaneStatusStrategy
    ) -> None:
        # Mark window already scanned so first-scan suppression doesn't
        # drop the "blocked" transition this test verifies.
        strategy._scanned_windows.add("@0")
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        provider = _interactive_provider("Allow?")
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[
                    _pane("%1", active=True),
                    _pane("%2", active=False, index=1),
                ]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="Allow?\nEsc\n")
            transitions = await strategy.scan_window(
                bot, 1, "@0", 42, on_blocked=on_blocked
            )
        on_blocked.assert_called_once_with(bot, 1, "@0", 42, "%2")
        assert _require_pane("@0", "%2").state == "blocked"
        assert any(t.pane_id == "%2" and t.new_state == "blocked" for t in transitions)

    async def test_blocked_alert_dedup_within_two_scans(
        self, strategy: PaneStatusStrategy
    ) -> None:
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        provider = _interactive_provider("Allow?")
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[
                    _pane("%1", active=True),
                    _pane("%2", active=False, index=1),
                ]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="Allow?\nEsc\n")
            await strategy.scan_window(bot, 1, "@0", 42, on_blocked=on_blocked)
            await strategy.scan_window(bot, 1, "@0", 42, on_blocked=on_blocked)
        on_blocked.assert_called_once()

    async def test_blocked_alert_resurfaces_when_prompt_changes(
        self, strategy: PaneStatusStrategy
    ) -> None:
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        first = _interactive_provider("Allow read?")
        second = _interactive_provider("Allow write?")
        providers = iter([first, second])
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch(
                "ccgram.providers.get_provider_for_window",
                side_effect=lambda *_a, **_k: next(providers),
            ),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[
                    _pane("%1", active=True),
                    _pane("%2", active=False, index=1),
                ]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="prompt\nEsc\n")
            await strategy.scan_window(bot, 1, "@0", 42, on_blocked=on_blocked)
            await strategy.scan_window(bot, 1, "@0", 42, on_blocked=on_blocked)
        assert on_blocked.call_count == 2

    async def test_dead_pane_removed_from_state(
        self, strategy: PaneStatusStrategy
    ) -> None:
        strategy.record_pane_state("@0", "%2", "active", provider="claude")
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        provider = _idle_provider()
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(return_value=[_pane("%1", active=True)])
            transitions = await strategy.scan_window(
                bot, 1, "@0", 42, on_blocked=on_blocked
            )
        assert window_store.get_pane("@0", "%2") is None
        assert any(t.pane_id == "%2" and t.new_state == "dead" for t in transitions)

    async def test_active_to_idle_transition_emitted(
        self, strategy: PaneStatusStrategy
    ) -> None:
        strategy.record_pane_state("@0", "%2", "active", provider="claude")
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        provider = _idle_provider()
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[
                    _pane("%1", active=True),
                    _pane("%2", active=False, index=1),
                ]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="text")
            transitions = await strategy.scan_window(
                bot, 1, "@0", 42, on_blocked=on_blocked
            )
        assert PaneTransition(pane_id="%2", prev_state="active", new_state="idle") in (
            transitions
        )

    async def test_per_pane_provider_detection(
        self, strategy: PaneStatusStrategy
    ) -> None:
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        window_provider = _idle_provider("claude")
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=window_provider,
            ),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[
                    _pane("%1", active=True, command="claude"),
                    _pane("%2", active=False, index=1, command="codex"),
                    _pane("%3", active=False, index=2, command="bash"),
                ]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="output")
            await strategy.scan_window(bot, 1, "@0", 42, on_blocked=on_blocked)
        assert _require_pane("@0", "%1").provider == "claude"
        assert _require_pane("@0", "%2").provider == "codex"
        assert _require_pane("@0", "%3").provider == "shell"

    async def test_capture_failure_falls_back_to_idle(
        self, strategy: PaneStatusStrategy
    ) -> None:
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        provider = _idle_provider()
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[
                    _pane("%1", active=True),
                    _pane("%2", active=False, index=1),
                ]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value=None)
            await strategy.scan_window(bot, 1, "@0", 42, on_blocked=on_blocked)
        assert _require_pane("@0", "%2").state == "idle"
        on_blocked.assert_not_called()


class TestScanWindowFastPath:
    async def test_single_pane_cached_skips_subprocess(
        self, strategy: PaneStatusStrategy
    ) -> None:
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        with patch("ccgram.multiplexer.multiplexer") as mock_tm:
            mock_tm.list_panes = AsyncMock(return_value=[_pane("%1", active=True)])
            await strategy.scan_window(bot, 1, "@0", 42, on_blocked=on_blocked)
            transitions = await strategy.scan_window(
                bot, 1, "@0", 42, on_blocked=on_blocked
            )
        mock_tm.list_panes.assert_called_once()
        assert transitions == []
