"""Tests for pane subscribe / rename callbacks and pane-output forwarding.

Covers:
  - subscribe / unsubscribe toggle persists on PaneInfo.subscribed
  - rename prompt + apply_pane_rename round-trip
  - rename clears with "-" or empty input
  - rename truncates over-long names
  - keyboard builder shapes
  - PaneStatusStrategy.scan_window forwards subscribed pane output via callback
  - subscription cleared automatically when pane dies (reconcile drops PaneInfo)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot

from ccgram.handlers.live import pane_callbacks
from ccgram.handlers.callback_data import (
    CB_PANE_LIFECYCLE_TOGGLE,
    CB_PANE_RENAME,
    CB_PANE_SUBSCRIBE,
    CB_PANE_UNSUBSCRIBE,
)
from ccgram.handlers.live.pane_callbacks import (
    apply_pane_rename,
    build_pane_buttons,
    build_pane_lifecycle_button,
)
from ccgram.handlers.polling.polling_state import (
    InteractiveUIStrategy,
    PaneStatusStrategy,
    TerminalPollState,
    TerminalScreenBuffer,
)
from ccgram.handlers.user_state import (
    PANE_RENAME_PANE_ID,
    PANE_RENAME_THREAD_ID,
    PANE_RENAME_WINDOW_ID,
)
from ccgram.thread_router import thread_router
from ccgram.multiplexer.base import PaneInfo as TmuxPaneInfo
from ccgram.window_state_store import window_store


@pytest.fixture(autouse=True)
def _reset_state():
    window_store.reset()
    thread_router.reset()
    saved_schedule = window_store._schedule_save
    window_store._schedule_save = lambda: None
    yield
    window_store.reset()
    thread_router.reset()
    window_store._schedule_save = saved_schedule


def _bind(user_id: int, thread_id: int, window_id: str) -> None:
    thread_router.thread_bindings.setdefault(user_id, {})[thread_id] = window_id
    thread_router._rebuild_reverse_index()


def _query(data: str, *, user_id: int = 1, thread_id: int = 99) -> MagicMock:
    query = MagicMock()
    query.data = data
    query.answer = AsyncMock()
    query.from_user.id = user_id
    msg = MagicMock()
    msg.message_thread_id = thread_id
    query.message = msg
    bot = AsyncMock(spec=Bot)
    bot.send_message = AsyncMock()
    query.get_bot = MagicMock(return_value=bot)
    return query


def _update(query: MagicMock, *, user_id: int = 1) -> MagicMock:
    upd = MagicMock()
    upd.callback_query = query
    upd.effective_user.id = user_id
    upd.message = None
    return upd


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = {}
    return ctx


class TestBuildPaneButtons:
    def test_unsubscribed_shows_sub_button(self) -> None:
        row = build_pane_buttons("@0", "%1", subscribed=False)
        assert len(row) == 3
        labels = [btn.text for btn in row]
        assert any("Sub" in lbl and "Unsub" not in lbl for lbl in labels)
        assert any("Rename" in lbl for lbl in labels)

    def test_subscribed_shows_unsub_button(self) -> None:
        row = build_pane_buttons("@0", "%1", subscribed=True)
        labels = [btn.text for btn in row]
        assert any("Unsub" in lbl for lbl in labels)

    def test_callback_data_under_64_bytes(self) -> None:
        row = build_pane_buttons("ccgram:@1234", "%99", subscribed=False)
        for btn in row:
            data = btn.callback_data
            assert isinstance(data, str)
            assert len(data) <= 64


class TestSubscribeToggle:
    async def test_subscribe_marks_pane(self) -> None:
        _bind(1, 99, "@0")
        window_store.upsert_pane("@0", "%5", state="idle")
        query = _query(f"{CB_PANE_SUBSCRIBE}@0|%5")
        update = _update(query)
        ctx = _ctx()
        await pane_callbacks._dispatch(update, ctx)
        pane = window_store.get_pane("@0", "%5")
        assert pane is not None and pane.subscribed is True
        query.answer.assert_called_once()

    async def test_unsubscribe_clears_flag(self) -> None:
        _bind(1, 99, "@0")
        window_store.upsert_pane("@0", "%5", state="idle", subscribed=True)
        query = _query(f"{CB_PANE_UNSUBSCRIBE}@0|%5")
        await pane_callbacks._dispatch(_update(query), _ctx())
        pane = window_store.get_pane("@0", "%5")
        assert pane is not None and pane.subscribed is False

    async def test_subscribe_rejects_non_owner(self) -> None:
        _bind(1, 99, "@0")
        window_store.upsert_pane("@0", "%5", state="idle")
        # User 2 has no binding to @0
        query = _query(f"{CB_PANE_SUBSCRIBE}@0|%5", user_id=2)
        await pane_callbacks._dispatch(_update(query, user_id=2), _ctx())
        pane = window_store.get_pane("@0", "%5")
        assert pane is not None and pane.subscribed is False
        # Alert toast was answered
        query.answer.assert_called_once()
        assert query.answer.call_args.kwargs.get("show_alert") is True

    async def test_subscribe_rejects_unknown_pane(self) -> None:
        _bind(1, 99, "@0")
        query = _query(f"{CB_PANE_SUBSCRIBE}@0|%9")
        with patch(
            "ccgram.multiplexer.tmux.tmux_manager.list_panes",
            AsyncMock(return_value=[]),
        ):
            await pane_callbacks._dispatch(_update(query), _ctx())
        # No pane was created
        assert window_store.get_pane("@0", "%9") is None

    async def test_subscribe_hydrates_pane_from_tmux(self) -> None:
        _bind(1, 99, "@0")
        query = _query(f"{CB_PANE_SUBSCRIBE}@0|%9")
        live = [
            TmuxPaneInfo(
                pane_id="%9",
                index=2,
                active=False,
                command="claude",
                path="/tmp",
                width=80,
                height=24,
            )
        ]
        with patch(
            "ccgram.multiplexer.tmux.tmux_manager.list_panes",
            AsyncMock(return_value=live),
        ):
            await pane_callbacks._dispatch(_update(query), _ctx())
        pane = window_store.get_pane("@0", "%9")
        assert pane is not None and pane.subscribed is True

    async def test_subscribe_rejects_malformed_data(self) -> None:
        _bind(1, 99, "@0")
        query = _query(f"{CB_PANE_SUBSCRIBE}@0")  # missing :%pane
        await pane_callbacks._dispatch(_update(query), _ctx())
        query.answer.assert_called_once_with("Invalid pane")


class TestRenamePrompt:
    async def test_rename_records_pending_state_and_sends_prompt(self) -> None:
        _bind(1, 99, "@0")
        window_store.upsert_pane("@0", "%5", state="idle")
        query = _query(f"{CB_PANE_RENAME}@0|%5")
        ctx = _ctx()
        await pane_callbacks._dispatch(_update(query), ctx)
        assert ctx.user_data[PANE_RENAME_WINDOW_ID] == "@0"
        assert ctx.user_data[PANE_RENAME_PANE_ID] == "%5"
        assert ctx.user_data[PANE_RENAME_THREAD_ID] == 99
        bot = query.get_bot.return_value
        bot.send_message.assert_awaited_once()

    async def test_rename_rejects_non_owner(self) -> None:
        _bind(1, 99, "@0")
        query = _query(f"{CB_PANE_RENAME}@0|%5", user_id=2)
        ctx = _ctx()
        await pane_callbacks._dispatch(_update(query, user_id=2), ctx)
        assert PANE_RENAME_WINDOW_ID not in ctx.user_data


class TestApplyPaneRename:
    async def test_applies_name_clears_pending_state(self) -> None:
        window_store.upsert_pane("@0", "%5", state="idle")
        ud = {
            PANE_RENAME_WINDOW_ID: "@0",
            PANE_RENAME_PANE_ID: "%5",
            PANE_RENAME_THREAD_ID: 99,
        }
        msg = MagicMock()
        msg.reply_text = AsyncMock()
        handled = await apply_pane_rename(ud, 99, "api-gateway", msg)
        assert handled is True
        pane = window_store.get_pane("@0", "%5")
        assert pane is not None and pane.name == "api-gateway"
        # Pending state cleared so the next message routes normally.
        assert PANE_RENAME_WINDOW_ID not in ud

    async def test_dash_clears_existing_name(self) -> None:
        window_store.upsert_pane("@0", "%5", state="idle", name="old")
        ud = {
            PANE_RENAME_WINDOW_ID: "@0",
            PANE_RENAME_PANE_ID: "%5",
            PANE_RENAME_THREAD_ID: 99,
        }
        msg = MagicMock()
        msg.reply_text = AsyncMock()
        handled = await apply_pane_rename(ud, 99, "-", msg)
        assert handled is True
        pane = window_store.get_pane("@0", "%5")
        assert pane is not None and pane.name is None

    async def test_rejects_over_long_name(self) -> None:
        window_store.upsert_pane("@0", "%5", state="idle")
        ud = {
            PANE_RENAME_WINDOW_ID: "@0",
            PANE_RENAME_PANE_ID: "%5",
            PANE_RENAME_THREAD_ID: 99,
        }
        msg = MagicMock()
        msg.reply_text = AsyncMock()
        handled = await apply_pane_rename(ud, 99, "x" * 200, msg)
        assert handled is True
        pane = window_store.get_pane("@0", "%5")
        # Name not assigned — user must resend a shorter version.
        assert pane is not None and pane.name is None
        msg.reply_text.assert_awaited_once()
        reply_text = msg.reply_text.call_args.args[0]
        assert "Name too long" in reply_text or "too long" in reply_text.lower()

    async def test_skips_when_no_pending_rename(self) -> None:
        msg = MagicMock()
        handled = await apply_pane_rename({}, 99, "anything", msg)
        assert handled is False

    async def test_skips_for_different_thread(self) -> None:
        ud = {
            PANE_RENAME_WINDOW_ID: "@0",
            PANE_RENAME_PANE_ID: "%5",
            PANE_RENAME_THREAD_ID: 99,
        }
        msg = MagicMock()
        handled = await apply_pane_rename(ud, 100, "anything", msg)
        assert handled is False
        # Pending state preserved for the original thread
        assert ud[PANE_RENAME_THREAD_ID] == 99

    async def test_handles_none_user_data(self) -> None:
        msg = MagicMock()
        handled = await apply_pane_rename(None, 99, "anything", msg)
        assert handled is False


def _pane(
    pane_id: str = "%2",
    *,
    active: bool = False,
    index: int = 1,
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


@pytest.fixture
def strategy() -> PaneStatusStrategy:
    return PaneStatusStrategy(
        TerminalScreenBuffer(TerminalPollState()), InteractiveUIStrategy()
    )


class TestSubscribedOutputForwarding:
    async def test_callback_invoked_for_subscribed_pane(
        self, strategy: PaneStatusStrategy
    ) -> None:
        window_store.upsert_pane("@0", "%2", state="idle", subscribed=True)
        bot = AsyncMock(spec=Bot)
        on_blocked = AsyncMock()
        on_pane_output = AsyncMock()
        provider = MagicMock()
        provider.parse_terminal_status.return_value = None
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[_pane("%1", active=True, index=0), _pane("%2")]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="hello world\n")
            await strategy.scan_window(
                bot,
                1,
                "@0",
                42,
                on_blocked=on_blocked,
                on_pane_output=on_pane_output,
            )
        on_pane_output.assert_awaited_once()
        await_args = on_pane_output.await_args
        assert await_args is not None
        args = await_args.args
        assert args[2] == "@0"
        assert args[4] == "%2"
        assert "hello world" in args[5]

    async def test_callback_skipped_when_unsubscribed(
        self, strategy: PaneStatusStrategy
    ) -> None:
        window_store.upsert_pane("@0", "%2", state="idle", subscribed=False)
        bot = AsyncMock(spec=Bot)
        on_pane_output = AsyncMock()
        provider = MagicMock()
        provider.parse_terminal_status.return_value = None
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[_pane("%1", active=True, index=0), _pane("%2")]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="hello\n")
            await strategy.scan_window(
                bot,
                1,
                "@0",
                42,
                on_blocked=AsyncMock(),
                on_pane_output=on_pane_output,
            )
        on_pane_output.assert_not_called()

    async def test_dedupes_when_content_unchanged(
        self, strategy: PaneStatusStrategy
    ) -> None:
        window_store.upsert_pane("@0", "%2", state="idle", subscribed=True)
        bot = AsyncMock(spec=Bot)
        on_pane_output = AsyncMock()
        provider = MagicMock()
        provider.parse_terminal_status.return_value = None
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[_pane("%1", active=True, index=0), _pane("%2")]
            )
            mock_tm.capture_pane_by_id = AsyncMock(return_value="same\n")
            await strategy.scan_window(
                bot,
                1,
                "@0",
                42,
                on_blocked=AsyncMock(),
                on_pane_output=on_pane_output,
            )
            await strategy.scan_window(
                bot,
                1,
                "@0",
                42,
                on_blocked=AsyncMock(),
                on_pane_output=on_pane_output,
            )
        assert on_pane_output.await_count == 1

    async def test_resends_when_content_changes(
        self,
        strategy: PaneStatusStrategy,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Disable per-pane forward rate limit so back-to-back scans both
        # forward when content changes. The rate limit is exercised
        # separately in `test_rate_limits_back_to_back_forwards`.
        monkeypatch.setattr(PaneStatusStrategy, "PANE_FORWARD_MIN_INTERVAL", 0.0)
        window_store.upsert_pane("@0", "%2", state="idle", subscribed=True)
        bot = AsyncMock(spec=Bot)
        on_pane_output = AsyncMock()
        provider = MagicMock()
        provider.parse_terminal_status.return_value = None
        outputs = iter(["first\n", "second\n"])
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[_pane("%1", active=True, index=0), _pane("%2")]
            )
            mock_tm.capture_pane_by_id = AsyncMock(
                side_effect=lambda *_a, **_k: next(outputs)
            )
            await strategy.scan_window(
                bot,
                1,
                "@0",
                42,
                on_blocked=AsyncMock(),
                on_pane_output=on_pane_output,
            )
            await strategy.scan_window(
                bot,
                1,
                "@0",
                42,
                on_blocked=AsyncMock(),
                on_pane_output=on_pane_output,
            )
        assert on_pane_output.await_count == 2

    async def test_rate_limits_back_to_back_forwards(
        self, strategy: PaneStatusStrategy
    ) -> None:
        # With the default 5s minimum interval, the second scan within
        # the same monotonic window must NOT forward even when content
        # genuinely changed — protects Telegram from busy-pane floods.
        window_store.upsert_pane("@0", "%2", state="idle", subscribed=True)
        bot = AsyncMock(spec=Bot)
        on_pane_output = AsyncMock()
        provider = MagicMock()
        provider.parse_terminal_status.return_value = None
        outputs = iter(["first\n", "second\n"])
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[_pane("%1", active=True, index=0), _pane("%2")]
            )
            mock_tm.capture_pane_by_id = AsyncMock(
                side_effect=lambda *_a, **_k: next(outputs)
            )
            await strategy.scan_window(
                bot,
                1,
                "@0",
                42,
                on_blocked=AsyncMock(),
                on_pane_output=on_pane_output,
            )
            await strategy.scan_window(
                bot,
                1,
                "@0",
                42,
                on_blocked=AsyncMock(),
                on_pane_output=on_pane_output,
            )
        assert on_pane_output.await_count == 1

    async def test_dead_pane_drops_subscription(
        self, strategy: PaneStatusStrategy
    ) -> None:
        # Subscribed pane that disappears in the next scan
        window_store.upsert_pane("@0", "%2", state="idle", subscribed=True)
        bot = AsyncMock(spec=Bot)
        on_pane_output = AsyncMock()
        provider = MagicMock()
        provider.parse_terminal_status.return_value = None
        with (
            patch("ccgram.multiplexer.multiplexer") as mock_tm,
            patch("ccgram.providers.get_provider_for_window", return_value=provider),
        ):
            mock_tm.list_panes = AsyncMock(
                return_value=[_pane("%1", active=True, index=0)]
            )
            await strategy.scan_window(
                bot,
                1,
                "@0",
                42,
                on_blocked=AsyncMock(),
                on_pane_output=on_pane_output,
            )
        # PaneInfo for %2 — and its subscribed flag — is gone
        assert window_store.get_pane("@0", "%2") is None
        # Hash cache also evicted
        assert "%2" not in strategy._pane_content_hash


class TestBuildPaneLifecycleButton:
    def test_disabled_state_label_and_callback(self) -> None:
        btn = build_pane_lifecycle_button("@0", enabled=False)
        assert "off" in btn.text.lower()
        data = btn.callback_data
        assert isinstance(data, str)
        assert data.startswith(CB_PANE_LIFECYCLE_TOGGLE)
        assert data.endswith("@0")

    def test_enabled_state_label(self) -> None:
        btn = build_pane_lifecycle_button("@0", enabled=True)
        assert "on" in btn.text.lower()

    def test_callback_data_under_64_bytes(self) -> None:
        btn = build_pane_lifecycle_button("ccgram:@1234567890", enabled=True)
        data = btn.callback_data
        assert isinstance(data, str)
        assert len(data) <= 64


class TestLifecycleToggle:
    async def test_toggle_off_to_on_persists(self) -> None:
        _bind(1, 99, "@0")
        query = _query(f"{CB_PANE_LIFECYCLE_TOGGLE}@0")
        await pane_callbacks._dispatch(_update(query), _ctx())
        ws = window_store.get_window_state("@0")
        assert ws.pane_lifecycle_notify is True
        query.answer.assert_called_once()
        assert "on" in query.answer.call_args.args[0].lower()

    async def test_toggle_on_to_off_persists(self) -> None:
        _bind(1, 99, "@0")
        window_store.set_pane_lifecycle_notify("@0", True)
        query = _query(f"{CB_PANE_LIFECYCLE_TOGGLE}@0")
        await pane_callbacks._dispatch(_update(query), _ctx())
        ws = window_store.get_window_state("@0")
        assert ws.pane_lifecycle_notify is False
        assert "off" in query.answer.call_args.args[0].lower()

    async def test_toggle_rejects_non_owner(self) -> None:
        _bind(1, 99, "@0")
        query = _query(f"{CB_PANE_LIFECYCLE_TOGGLE}@0", user_id=2)
        await pane_callbacks._dispatch(_update(query, user_id=2), _ctx())
        ws = window_store.get_window_state("@0")
        assert ws.pane_lifecycle_notify is None
        assert query.answer.call_args.kwargs.get("show_alert") is True

    async def test_toggle_rejects_missing_window_id(self) -> None:
        _bind(1, 99, "@0")
        query = _query(CB_PANE_LIFECYCLE_TOGGLE)  # no window_id suffix
        await pane_callbacks._dispatch(_update(query), _ctx())
        query.answer.assert_called_once_with("Invalid window")
