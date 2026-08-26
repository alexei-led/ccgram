import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot
from telegram.error import BadRequest, RetryAfter

from ccgram.window_view import WindowView

from ccgram.handlers.topics.topic_lifecycle import (
    PROBE_MAX_PER_CYCLE,
    check_autoclose_timers,
    check_unbound_window_ttl,
    probe_topic_existence,
    prune_stale_state,
    reset_probe_schedule,
    rollback_legacy_herdr_binding,
)
from ccgram.handlers.polling.polling_state import (
    lifecycle_strategy,
    terminal_poll_state,
)


@pytest.fixture(autouse=True)
def _clean_strategy_state():
    reset_probe_schedule()
    terminal_poll_state._states.clear()
    lifecycle_strategy._states.clear()
    lifecycle_strategy._dead_notified.clear()
    yield
    reset_probe_schedule()
    terminal_poll_state._states.clear()
    lifecycle_strategy._states.clear()
    lifecycle_strategy._dead_notified.clear()


class TestLegacyHerdrMigration:
    def test_rollback_restores_binding_but_remains_action_blocked(self) -> None:
        with (
            patch(
                "ccgram.handlers.topics.topic_lifecycle.legacy_state"
            ) as legacy_state,
            patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as router,
        ):
            legacy_state.rollback_legacy_herdr_archive.return_value = True
            assert rollback_legacy_herdr_binding(1, 100, "w2:t1") is True
            router.bind_thread.assert_called_once_with(1, 100, "w2:t1")

    def test_rollback_rejects_non_archived_record(self) -> None:
        with patch(
            "ccgram.handlers.topics.topic_lifecycle.legacy_state"
        ) as legacy_state:
            legacy_state.rollback_legacy_herdr_archive.return_value = False
            assert rollback_legacy_herdr_binding(1, 100, "w2:t1") is False


class TestCheckAutocloseTimers:
    async def test_no_topics_is_noop(self):
        bot = AsyncMock(spec=Bot)
        await check_autoclose_timers(bot)
        bot.close_forum_topic.assert_not_called()
        bot.delete_forum_topic.assert_not_called()

    async def test_expired_done_topic_is_closed_not_deleted(self):
        """Autoclose closes the topic; it never deletes (irreversible data loss)."""
        bot = AsyncMock(spec=Bot)
        user_id, thread_id = 1, 100
        lifecycle_strategy.start_autoclose_timer(
            user_id, thread_id, "done", time.monotonic() - 99999
        )
        with (
            patch("ccgram.handlers.topics.topic_lifecycle.config") as mock_config,
            patch(
                "ccgram.handlers.topics.topic_lifecycle.thread_router"
            ) as mock_router,
            patch(
                "ccgram.handlers.topics.topic_lifecycle.clear_topic_state",
                new_callable=AsyncMock,
            ),
        ):
            mock_config.autoclose_done_minutes = 1
            mock_router.resolve_chat_id.return_value = 42
            mock_router.get_window_for_thread.return_value = "@0"
            await check_autoclose_timers(bot)
        bot.close_forum_topic.assert_called_once()
        bot.delete_forum_topic.assert_not_called()

    async def test_not_yet_expired_topic_stays(self):
        bot = AsyncMock(spec=Bot)
        user_id, thread_id = 1, 100
        lifecycle_strategy.start_autoclose_timer(
            user_id, thread_id, "done", time.monotonic()
        )
        with patch("ccgram.handlers.topics.topic_lifecycle.config") as mock_config:
            mock_config.autoclose_done_minutes = 60
            await check_autoclose_timers(bot)
        bot.close_forum_topic.assert_not_called()
        bot.delete_forum_topic.assert_not_called()

    async def test_expired_dead_topic_stays_when_window_is_live(self):
        bot = AsyncMock(spec=Bot)
        user_id, thread_id = 1, 100
        lifecycle_strategy.start_autoclose_timer(
            user_id, thread_id, "dead", time.monotonic() - 99999
        )
        with (
            patch("ccgram.handlers.topics.topic_lifecycle.config") as mock_config,
            patch(
                "ccgram.handlers.topics.topic_lifecycle.thread_router"
            ) as mock_router,
            patch("ccgram.handlers.topics.topic_lifecycle.tmux_manager") as mock_tmux,
        ):
            mock_config.autoclose_done_minutes = 30
            mock_config.autoclose_dead_minutes = 10
            mock_router.get_window_for_thread.return_value = "@0"
            mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
            await check_autoclose_timers(bot)
        bot.delete_forum_topic.assert_not_called()
        assert lifecycle_strategy.get_state(user_id, thread_id).autoclose is None


HERDR_TARGET = "herdr-session-v1-" + "a" * 64


def _window_view(origin: str, window_id: str = "@0") -> WindowView:
    return WindowView(
        window_id=window_id,
        cwd="/tmp",
        provider_name="claude",
        approval_mode="normal",
        batch_mode="batched",
        tool_call_visibility="default",
        transcript_path=None,
        window_name="test",
        session_id="s1",
        origin=origin,
    )


@dataclass
class _TtlSweep:
    mux: MagicMock
    revoke: MagicMock
    unbound_timer: float | None


async def _sweep_expired_unbound_window(
    window_id: str = "@0",
    *,
    origin: str = "ccgram_created",
    kill_ok: bool = True,
    bindings: list[tuple[int, int, str]] | None = None,
) -> _TtlSweep:
    """Age *window_id* past the unbound TTL, run one sweep, report what happened."""
    state = terminal_poll_state.get_state(window_id)
    state.unbound_timer = time.monotonic() - 100
    window = MagicMock(window_id=window_id, window_name="test")
    with (
        patch("ccgram.handlers.topics.topic_lifecycle.config") as mock_config,
        patch("ccgram.handlers.topics.topic_lifecycle.thread_router") as mock_router,
        patch("ccgram.handlers.topics.topic_lifecycle.window_query") as mock_wq,
        patch("ccgram.handlers.topics.topic_lifecycle.tmux_manager") as mock_tmux,
        patch("ccgram.handlers.topics.topic_lifecycle.revoke_window_tokens") as revoke,
    ):
        mock_config.autoclose_done_minutes = 1
        mock_router.iter_thread_bindings.return_value = bindings or []
        mock_wq.view_window.return_value = _window_view(origin, window_id)
        mock_tmux.kill_window = AsyncMock(return_value=kill_ok)
        await check_unbound_window_ttl([window])
    return _TtlSweep(
        mux=mock_tmux,
        revoke=revoke,
        unbound_timer=terminal_poll_state.get_state(window_id).unbound_timer,
    )


class TestCheckUnboundWindowTtl:
    async def test_no_timeout_is_noop(self):
        with patch("ccgram.handlers.topics.topic_lifecycle.config") as mock_config:
            mock_config.autoclose_done_minutes = 0
            await check_unbound_window_ttl([])

    async def test_bound_window_timer_cleared(self):
        sweep = await _sweep_expired_unbound_window(bindings=[(1, 100, "@0")])
        assert sweep.unbound_timer is None
        sweep.mux.kill_window.assert_not_called()

    async def test_manually_discovered_window_is_not_killed(self):
        sweep = await _sweep_expired_unbound_window(origin="manual_discovered")
        assert sweep.unbound_timer is None
        sweep.mux.kill_window.assert_not_called()

    @pytest.mark.parametrize(
        "window_id",
        ["@0", HERDR_TARGET],
        ids=["tmux_window", "guarded_herdr_target"],
    )
    async def test_ccgram_created_window_is_killed_through_the_proxy(
        self, window_id: str
    ):
        sweep = await _sweep_expired_unbound_window(window_id)
        sweep.mux.kill_window.assert_called_once_with(window_id)
        sweep.revoke.assert_called_once_with(window_id)

    async def test_failed_kill_keeps_tokens_alive(self):
        """Tokens stay valid while the window might still be running."""
        sweep = await _sweep_expired_unbound_window(kill_ok=False)
        sweep.revoke.assert_not_called()


class TestPruneStaleState:
    async def test_syncs_display_names(self):
        mock_window = MagicMock(window_id="@0", window_name="test")
        with patch("ccgram.handlers.topics.topic_lifecycle.session_manager") as mock_sm:
            await prune_stale_state([mock_window])
            mock_sm.sync_display_names.assert_called_once_with([("@0", "test")])
            mock_sm.prune_stale_state.assert_called_once_with({"@0"})


class TestProbeTopicExistence:
    @pytest.mark.parametrize(
        ("window_id", "origin", "expect_kill"),
        [
            ("@0", "manual_discovered", False),
            ("@0", "ccgram_created", True),
            (HERDR_TARGET, "ccgram_created", True),
        ],
        ids=["manual_window_survives", "ccgram_window_killed", "herdr_target_killed"],
    )
    async def test_deleted_topic_unbinds_and_kills_only_what_ccgram_created(
        self, window_id: str, origin: str, expect_kill: bool
    ):
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages = AsyncMock(
            side_effect=BadRequest("Topic_id_invalid")
        )
        with (
            patch(
                "ccgram.handlers.topics.topic_lifecycle.thread_router"
            ) as mock_router,
            patch("ccgram.handlers.topics.topic_lifecycle.tmux_manager") as mock_tmux,
            patch("ccgram.handlers.topics.topic_lifecycle.window_query") as mock_wq,
            patch(
                "ccgram.handlers.topics.topic_lifecycle.clear_topic_state",
                new_callable=AsyncMock,
            ),
        ):
            mock_router.iter_thread_bindings.return_value = [(1, 100, window_id)]
            mock_router.resolve_chat_id.return_value = 42
            mock_tmux.find_window_by_id = AsyncMock(
                return_value=MagicMock(window_id=window_id)
            )
            mock_wq.view_window.return_value = _window_view(origin, window_id)
            mock_tmux.kill_window = AsyncMock()
            await probe_topic_existence(bot)

        mock_router.unbind_thread.assert_called_once_with(1, 100, chat_id=42)
        if expect_kill:
            mock_tmux.kill_window.assert_called_once_with(window_id)
        else:
            mock_tmux.kill_window.assert_not_called()

    async def test_flood_control_backs_off_without_suspending(self):
        """RetryAfter is chat-wide: it must not disable deleted-topic detection."""
        from ccgram.handlers.topics import topic_lifecycle as tl

        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages = AsyncMock(
            side_effect=[RetryAfter(3), None]
        )
        with (
            patch.object(tl, "thread_router") as mock_router,
            patch.object(tl, "lifecycle_strategy") as mock_strategy,
            patch.object(tl.time, "monotonic", return_value=100.0),
        ):
            mock_router.iter_thread_bindings.return_value = [(1, 100, "@0")]
            mock_router.resolve_chat_id.return_value = 42
            mock_strategy.should_skip_probe.return_value = False

            await probe_topic_existence(bot)
            mock_strategy.record_probe_failure.assert_not_called()

            # Whole pass is paused while the chat is flood-limited.
            bot.unpin_all_forum_topic_messages.reset_mock()
            await probe_topic_existence(bot)
            bot.unpin_all_forum_topic_messages.assert_not_called()

            # Once the chat backoff expires, retry the failed topic instead of
            # waiting for the normal five-minute probe interval.
            tl._probe_backoff_until[42] = 0.0
            await probe_topic_existence(bot)
            bot.unpin_all_forum_topic_messages.assert_called_once()

    async def test_flood_control_is_scoped_to_the_affected_chat(self):
        from ccgram.handlers.topics import topic_lifecycle as tl

        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages = AsyncMock(
            side_effect=[RetryAfter(3), None, None]
        )
        bindings = [
            (1, 42, 100, "@0"),
            (1, 43, 100, "@1"),
        ]
        with (
            patch.object(tl, "thread_router") as mock_router,
            patch.object(tl.time, "monotonic", return_value=100.0),
        ):
            mock_router.iter_thread_bindings_with_chat.return_value = bindings
            await probe_topic_existence(bot)

            assert bot.unpin_all_forum_topic_messages.call_count == 2
            assert (
                bot.unpin_all_forum_topic_messages.call_args_list[0].kwargs["chat_id"]
                == 42
            )
            assert (
                bot.unpin_all_forum_topic_messages.call_args_list[1].kwargs["chat_id"]
                == 43
            )

            bot.unpin_all_forum_topic_messages.reset_mock()
            await probe_topic_existence(bot)
            bot.unpin_all_forum_topic_messages.assert_not_called()

            # Thread IDs are chat-local. Once chat 42's backoff expires, retry
            # its failed topic even though chat 43 probed thread 100.
            tl._probe_backoff_until[42] = 0.0
            await probe_topic_existence(bot)
            bot.unpin_all_forum_topic_messages.assert_called_once()
            assert bot.unpin_all_forum_topic_messages.call_args.kwargs["chat_id"] == 42

    async def test_probe_budget_allows_only_one_topic_per_chat(self):
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages = AsyncMock()
        bindings = [
            (1, 42, 100, "@0"),
            (1, 42, 101, "@1"),
            (1, 43, 102, "@2"),
        ]
        with patch(
            "ccgram.handlers.topics.topic_lifecycle.thread_router"
        ) as mock_router:
            mock_router.iter_thread_bindings_with_chat.return_value = bindings
            await probe_topic_existence(bot)

        assert bot.unpin_all_forum_topic_messages.call_count == 2
        assert {
            call.kwargs["chat_id"]
            for call in bot.unpin_all_forum_topic_messages.call_args_list
        } == {42, 43}

    async def test_probes_run_shortly_after_boot(self):
        """time.monotonic() starts near zero: never-probed must still be due."""
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages = AsyncMock()
        with (
            patch(
                "ccgram.handlers.topics.topic_lifecycle.thread_router"
            ) as mock_router,
            patch(
                "ccgram.handlers.topics.topic_lifecycle.time.monotonic",
                return_value=5.0,
            ),
        ):
            mock_router.iter_thread_bindings.return_value = [(1, 100, "@0")]
            mock_router.resolve_chat_id.return_value = 42
            await probe_topic_existence(bot)

        bot.unpin_all_forum_topic_messages.assert_called_once()

    async def test_probe_budget_rotates_across_cycles(self):
        """Bounded admin calls per cycle, least-recently-probed topic first."""
        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages = AsyncMock()
        topics = [
            (1, 1000 + i, 100 + i, f"@{i}") for i in range(3 * PROBE_MAX_PER_CYCLE)
        ]
        with patch(
            "ccgram.handlers.topics.topic_lifecycle.thread_router"
        ) as mock_router:
            mock_router.iter_thread_bindings_with_chat.return_value = topics

            probed = []
            for _ in range(3):
                bot.unpin_all_forum_topic_messages.reset_mock()
                await probe_topic_existence(bot)
                assert (
                    bot.unpin_all_forum_topic_messages.call_count == PROBE_MAX_PER_CYCLE
                )
                probed += [
                    c.kwargs["message_thread_id"]
                    for c in bot.unpin_all_forum_topic_messages.call_args_list
                ]

        # Every topic probed exactly once — no repeats while others are due.
        assert sorted(probed) == [t[2] for t in topics]

    async def test_unprobeable_windows_do_not_hold_cycle_slots(self):
        from ccgram.handlers.topics import topic_lifecycle as tl

        bot = AsyncMock(spec=Bot)
        bot.unpin_all_forum_topic_messages = AsyncMock()
        blocked = [f"@blocked{i}" for i in range(PROBE_MAX_PER_CYCLE)]
        tl._probe_pin_disabled.update(blocked)
        try:
            with patch.object(tl, "thread_router") as mock_router:
                mock_router.iter_thread_bindings.return_value = [
                    *[(1, 200 + i, wid) for i, wid in enumerate(blocked)],
                    (1, 300, "@live"),
                ]
                mock_router.resolve_chat_id.return_value = 42
                await probe_topic_existence(bot)
        finally:
            tl._probe_pin_disabled.difference_update(blocked)

        bot.unpin_all_forum_topic_messages.assert_called_once()
        assert (
            bot.unpin_all_forum_topic_messages.call_args.kwargs["message_thread_id"]
            == 300
        )

    async def test_suspended_probe_skipped(self):
        bot = AsyncMock(spec=Bot)
        ws = terminal_poll_state.get_state("@0")
        ws.probe_failures = 999
        with patch(
            "ccgram.handlers.topics.topic_lifecycle.thread_router"
        ) as mock_router:
            mock_router.iter_thread_bindings.return_value = [(1, 100, "@0")]
            await probe_topic_existence(bot)
        bot.unpin_all_forum_topic_messages.assert_not_called()

    async def test_missing_pin_rights_disables_probe_without_suspending(self):
        from ccgram.handlers.topics import topic_lifecycle as tl

        bot = AsyncMock(spec=Bot)
        # Real Telegram error for unpin without can_pin_messages is lowercase.
        bot.unpin_all_forum_topic_messages = AsyncMock(
            side_effect=BadRequest("not enough rights to manage pinned messages")
        )
        wid = "@probe-pin"
        tl._probe_pin_disabled.discard(wid)
        try:
            with (
                patch.object(tl, "thread_router") as mock_router,
                patch.object(tl, "lifecycle_strategy") as mock_strategy,
            ):
                mock_router.iter_thread_bindings.return_value = [(1, 100, wid)]
                mock_router.resolve_chat_id.return_value = 42
                mock_strategy.should_skip_probe.return_value = False

                await probe_topic_existence(bot)

                # Permission error must not count as a probe failure (no suspend).
                mock_strategy.record_probe_failure.assert_not_called()
                assert wid in tl._probe_pin_disabled

                # Next tick skips the window entirely — no further API call.
                bot.unpin_all_forum_topic_messages.reset_mock()
                await probe_topic_existence(bot)
                bot.unpin_all_forum_topic_messages.assert_not_called()
        finally:
            tl._probe_pin_disabled.discard(wid)
