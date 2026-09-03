import contextlib
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest, TelegramError

from ccgram.multiplexer.base import WindowRef
from ccgram.session import SessionManager
from ccgram.thread_router import thread_router
from ccgram.handlers.sync_command import _run_audit
from ccgram.handlers.callback_data import CB_SYNC_DISMISS, CB_SYNC_FIX
from ccgram.handlers.sync_command import (
    _cleanup_retired_topics,
    _close_ghost_topics,
    _format_report,
    _probe_dead_topics,
    _recreate_dead_topics,
    _sync_live_topic_names,
    _dispatch,
    handle_sync_dismiss,
    handle_sync_fix,
    sync_command,
)
from ccgram.session import AuditIssue, AuditResult
from ccgram.thread_router import RetiredTopic


@pytest.fixture(autouse=True)
def _patch_deps():
    with (
        patch("ccgram.handlers.sync_command.session_manager") as mock_sm,
        patch("ccgram.handlers.sync_command.session_map_sync") as mock_sms,
        patch("ccgram.handlers.sync_command.window_query") as mock_wq,
        patch("ccgram.handlers.sync_command.thread_router") as mock_tr,
        patch("ccgram.handlers.sync_command.tmux_manager") as mock_tm,
        patch(
            "ccgram.handlers.sync_command.list_windows_for_reconciliation"
        ) as mock_listing,
        # still_adoptable lives in topic_orchestration and reads its own
        # backend reference, so the same listing has to answer there.
        patch(
            "ccgram.handlers.topics.topic_orchestration.tmux_manager"
        ) as mock_tm_topics,
        patch("ccgram.handlers.sync_command.config") as mock_cfg,
    ):
        mock_sm.audit_state.return_value = AuditResult(
            issues=[], total_bindings=0, live_binding_count=0
        )
        mock_tr.iter_thread_bindings.return_value = []
        mock_sm.window_states = {}
        mock_tm.list_windows = AsyncMock(return_value=[])
        mock_tm.list_windows_for_reconciliation = mock_listing
        mock_tm_topics.list_windows_for_reconciliation = mock_listing
        mock_listing.return_value = []
        mock_cfg.is_user_allowed.return_value = True
        yield mock_sm, mock_sms, mock_wq, mock_tr, mock_tm, mock_cfg


def _audit(*issues: AuditIssue, total: int = 3, live: int = 3) -> AuditResult:
    return AuditResult(
        issues=list(issues), total_bindings=total, live_binding_count=live
    )


class TestFormatReport:
    @pytest.mark.parametrize(
        ("audit", "expected_text"),
        [
            pytest.param(_audit(), "3 topics bound, all windows alive", id="all-alive"),
            pytest.param(_audit(), "No orphaned entries", id="all-clear"),
            pytest.param(
                _audit(total=0, live=0), "No topic bindings", id="no-bindings"
            ),
        ],
    )
    def test_clean_report_has_no_keyboard(
        self, audit: AuditResult, expected_text: str
    ) -> None:
        text, keyboard = _format_report(audit)
        assert expected_text in text
        assert keyboard is None

    @pytest.mark.parametrize(
        ("audit", "expected_text"),
        [
            pytest.param(
                _audit(
                    AuditIssue(
                        "ghost_binding",
                        "user:100 thread:42 window:@7 (dead)",
                        fixable=True,
                    ),
                    live=2,
                ),
                ["ghost binding"],
                id="ghost-binding",
            ),
            pytest.param(
                _audit(AuditIssue("orphaned_display_name", "@7 (old)", fixable=True)),
                ["1 orphaned display name"],
                id="orphaned-display-name",
            ),
            pytest.param(
                _audit(
                    AuditIssue("orphaned_window", "@5 (stray)", fixable=True),
                    total=1,
                    live=1,
                ),
                ["unbound window"],
                id="orphaned-window",
            ),
            pytest.param(
                _audit(
                    AuditIssue(
                        "dead_topic",
                        "user:100 thread:42 window:@2 (qmd-go)",
                        fixable=True,
                    )
                ),
                ["1 dead topic", "deleted in Telegram"],
                id="dead-topic",
            ),
        ],
    )
    def test_fixable_issue_offers_fix_and_dismiss(
        self, audit: AuditResult, expected_text: list[str]
    ) -> None:
        text, keyboard = _format_report(audit)

        for fragment in expected_text:
            assert fragment in text
        assert keyboard is not None
        assert "Fix 1 issue" in keyboard.inline_keyboard[0][0].text
        buttons = [btn.callback_data for row in keyboard.inline_keyboard for btn in row]
        assert CB_SYNC_FIX in buttons
        assert CB_SYNC_DISMISS in buttons

    def test_known_retired_candidate_reports_reason_and_offers_fix(self) -> None:
        text, keyboard = _format_report(
            _audit(
                AuditIssue(
                    "retired_topic",
                    "reason:system_replacement",
                    fixable=True,
                ),
                total=0,
                live=0,
            )
        )

        assert "known retired topic cleanup candidate" in text
        assert "system_replacement" in text
        assert keyboard is not None
        assert "Fix 1 issue" in keyboard.inline_keyboard[0][0].text

    def test_retired_cleanup_outcomes_are_distinguished_in_report(self) -> None:
        text, _keyboard = _format_report(
            _audit(total=0, live=0),
            retired_outcomes={
                "deleted": 1,
                "closed": 2,
                "already_gone": 3,
                "failed": 4,
            },
        )

        assert "Deleted 1 known retired topic" in text
        assert "Closed 2 known retired topic" in text
        assert "Already gone 3 known retired topic" in text
        assert "Could not remove 4 known retired topic" in text

    def test_fix_button_counts_every_issue(self) -> None:
        _text, keyboard = _format_report(
            _audit(
                AuditIssue("orphaned_display_name", "@7 (old)", fixable=True),
                AuditIssue("stale_offset", "user 100, window @9", fixable=True),
                AuditIssue(
                    "display_name_drift", "@1: stored='a' tmux='b'", fixable=True
                ),
            )
        )

        assert keyboard is not None
        assert "Fix 3 issues" in keyboard.inline_keyboard[0][0].text

    def test_legacy_herdr_binding_is_reported_but_not_fixable(self) -> None:
        text, keyboard = _format_report(
            _audit(
                AuditIssue(
                    "legacy_herdr",
                    "w2:t1 is blocked; archive or explicitly rebind to a listed session target",
                    fixable=False,
                ),
                total=1,
                live=0,
            )
        )

        assert "legacy Herdr binding" in text
        assert "explicitly rebind" in text
        assert keyboard is None

    @pytest.mark.parametrize(
        ("kwargs", "expected_text"),
        [
            pytest.param({"fixed_count": 2}, "✅ Fixed 2 issues", id="fixed-header"),
            pytest.param(
                {"fixed_count": 1, "closed_topic_count": 1},
                "Removed 1 stale topic",
                id="closed-singular",
            ),
            pytest.param(
                {"fixed_count": 1, "closed_topic_count": 2},
                "Removed 2 stale topics",
                id="closed-plural",
            ),
            pytest.param(
                {"fixed_count": 1, "recreated_topic_count": 1},
                "Recreated 1 topic",
                id="recreated-singular",
            ),
            pytest.param(
                {"fixed_count": 2, "recreated_topic_count": 2},
                "Recreated 2 topics",
                id="recreated-plural",
            ),
        ],
    )
    def test_fixed_mode_summary(self, kwargs: dict, expected_text: str) -> None:
        text, _keyboard = _format_report(_audit(total=0, live=0), **kwargs)
        assert expected_text in text


class TestRetiredTopicCleanup:
    @staticmethod
    def _topic() -> RetiredTopic:
        return RetiredTopic(
            user_id=100,
            chat_id=-999,
            thread_id=42,
            reason="system_replacement",
            cleanup_eligible=True,
            sequence=1,
        )

    async def test_deleted_outcome_removes_known_retired_topic(
        self, _patch_deps
    ) -> None:
        *_, mock_tr, _, _ = _patch_deps
        topic = self._topic()
        mock_tr.iter_retired_topics.return_value = [topic]
        mock_tr.get_window_for_chat_thread.return_value = None
        client = AsyncMock()

        outcomes = await _cleanup_retired_topics(client)

        assert outcomes == {"deleted": 1}
        client.delete_forum_topic.assert_awaited_once_with(-999, 42)
        client.close_forum_topic.assert_not_awaited()
        mock_tr.discard_retired_topic.assert_called_once_with(topic)

    async def test_close_fallback_is_reported_separately(self, _patch_deps) -> None:
        *_, mock_tr, _, _ = _patch_deps
        topic = self._topic()
        mock_tr.iter_retired_topics.return_value = [topic]
        mock_tr.get_window_for_chat_thread.return_value = None
        client = AsyncMock()
        client.delete_forum_topic.side_effect = TelegramError("not permitted")

        outcomes = await _cleanup_retired_topics(client)

        assert outcomes == {"closed": 1}
        client.close_forum_topic.assert_awaited_once_with(-999, 42)
        mock_tr.discard_retired_topic.assert_called_once_with(topic)

    async def test_already_gone_is_terminal_not_a_failed_delete(
        self, _patch_deps
    ) -> None:
        *_, mock_tr, _, _ = _patch_deps
        topic = self._topic()
        mock_tr.iter_retired_topics.return_value = [topic]
        mock_tr.get_window_for_chat_thread.return_value = None
        client = AsyncMock()
        client.delete_forum_topic.side_effect = BadRequest("Message thread not found")

        outcomes = await _cleanup_retired_topics(client)

        assert outcomes == {"already_gone": 1}
        client.close_forum_topic.assert_not_awaited()
        mock_tr.discard_retired_topic.assert_called_once_with(topic)

    async def test_failed_calls_remain_in_registry_for_later_fix(
        self, _patch_deps
    ) -> None:
        *_, mock_tr, _, _ = _patch_deps
        topic = self._topic()
        mock_tr.iter_retired_topics.return_value = [topic]
        mock_tr.get_window_for_chat_thread.return_value = None
        client = AsyncMock()
        client.delete_forum_topic.side_effect = TelegramError("not permitted")
        client.close_forum_topic.side_effect = TelegramError("not permitted")

        outcomes = await _cleanup_retired_topics(client)

        assert outcomes == {"failed": 1}
        mock_tr.discard_retired_topic.assert_not_called()

    async def test_active_or_rebound_topic_is_protected_before_api_call(
        self, _patch_deps
    ) -> None:
        *_, mock_tr, _, _ = _patch_deps
        topic = self._topic()
        mock_tr.iter_retired_topics.return_value = [topic]
        # Simulates a rebind after the Fix snapshot but before its API call.
        mock_tr.get_window_for_chat_thread.return_value = "@rebound"
        client = AsyncMock()

        outcomes = await _cleanup_retired_topics(client)

        assert outcomes == {"protected_active": 1}
        client.delete_forum_topic.assert_not_awaited()
        client.close_forum_topic.assert_not_awaited()
        mock_tr.discard_retired_topic.assert_not_called()


class TestSyncDismiss:
    async def test_dismiss_deletes_message(self, _patch_deps) -> None:
        query = AsyncMock()
        query.delete_message = AsyncMock()
        msg = MagicMock()
        msg.text = "some report text"
        query.message = msg

        with patch("ccgram.handlers.sync_command.safe_edit") as mock_edit:
            await handle_sync_dismiss(query)
            query.delete_message.assert_awaited_once()
            mock_edit.assert_not_called()

    async def test_dismiss_fallback_when_delete_fails(self, _patch_deps) -> None:
        query = AsyncMock()
        query.delete_message = AsyncMock(side_effect=TelegramError("Forbidden"))
        msg = MagicMock()
        msg.text = None
        query.message = msg

        with patch("ccgram.handlers.sync_command.safe_edit") as mock_edit:
            await handle_sync_dismiss(query)
            query.delete_message.assert_awaited_once()
            mock_edit.assert_called_once_with(query, "Dismissed", reply_markup=None)


class TestSyncLiveTopicNames:
    async def test_limits_concurrent_telegram_calls_to_five(self, _patch_deps) -> None:
        _, _, _, mock_tr, _, _ = _patch_deps
        bindings = [(100, thread_id, f"window-{thread_id}") for thread_id in range(8)]
        mock_tr.iter_thread_bindings.return_value = bindings
        mock_tr.resolve_chat_id.return_value = -999
        mock_tr.get_display_name.side_effect = lambda window_id: window_id

        active = 0
        peak = 0
        started = 0
        first_batch_started = asyncio.Event()
        release = asyncio.Event()

        async def hold_call(*_args) -> None:
            nonlocal active, peak, started
            active += 1
            peak = max(peak, active)
            started += 1
            if started == 5:
                first_batch_started.set()
            await release.wait()
            active -= 1

        with patch(
            "ccgram.handlers.sync_command.sync_topic_name",
            new=AsyncMock(side_effect=hold_call),
        ) as mock_sync:
            task = asyncio.create_task(
                _sync_live_topic_names(
                    MagicMock(), {window_id for _, _, window_id in bindings}
                )
            )
            await asyncio.wait_for(first_batch_started.wait(), timeout=1)
            assert peak == 5
            assert not task.done()
            release.set()
            await asyncio.wait_for(task, timeout=1)

        assert mock_sync.await_count == len(bindings)

    async def test_waits_for_other_topic_syncs_after_unexpected_error(
        self, _patch_deps
    ) -> None:
        _, _, _, mock_tr, _, _ = _patch_deps
        bindings = [(100, 1, "window-1"), (100, 2, "window-2")]
        mock_tr.iter_thread_bindings.return_value = bindings
        mock_tr.resolve_chat_id.return_value = -999
        mock_tr.get_display_name.side_effect = lambda window_id: window_id

        blocked_started = asyncio.Event()
        release = asyncio.Event()

        async def fail_or_block(_client, _chat_id, thread_id, _name) -> None:
            if thread_id == 1:
                raise RuntimeError("unexpected")
            blocked_started.set()
            await release.wait()

        with (
            patch(
                "ccgram.handlers.sync_command.sync_topic_name",
                new=AsyncMock(side_effect=fail_or_block),
            ),
            patch("ccgram.handlers.sync_command.logger") as mock_logger,
        ):
            task = asyncio.create_task(
                _sync_live_topic_names(
                    MagicMock(), {window_id for _, _, window_id in bindings}
                )
            )
            await asyncio.wait_for(blocked_started.wait(), timeout=1)
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            await asyncio.wait_for(task, timeout=1)

        mock_logger.error.assert_called_once()


class TestSyncCommand:
    async def test_unauthorized_user_rejected(self, _patch_deps) -> None:
        _, _, _, _, _, mock_cfg = _patch_deps
        mock_cfg.is_user_allowed.return_value = False

        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()

        with patch("ccgram.handlers.sync_command.safe_reply") as mock_reply:
            await sync_command(update, MagicMock())
            mock_reply.assert_called_once()
            assert "not authorized" in mock_reply.call_args[0][1]

    async def test_unauthorized_fix_callback_cannot_start_destructive_cleanup(
        self, _patch_deps
    ) -> None:
        *_, mock_cfg = _patch_deps
        mock_cfg.is_user_allowed.return_value = False
        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.callback_query = AsyncMock()
        update.callback_query.data = CB_SYNC_FIX

        with patch(
            "ccgram.handlers.sync_command.handle_sync_fix", new_callable=AsyncMock
        ) as fix:
            await _dispatch(update, MagicMock())

        fix.assert_not_awaited()
        update.callback_query.answer.assert_awaited_once_with(
            "You are not authorized", show_alert=True
        )

    async def test_no_user_returns_early(self, _patch_deps) -> None:
        update = MagicMock()
        update.effective_user = None
        update.message = AsyncMock()

        with patch("ccgram.handlers.sync_command.safe_reply") as mock_reply:
            await sync_command(update, MagicMock())
            mock_reply.assert_not_called()

    async def test_calls_audit_and_replies(self, _patch_deps) -> None:
        mock_sm, _, _, _, _, _ = _patch_deps
        mock_sm.audit_state.return_value = AuditResult(
            issues=[], total_bindings=2, live_binding_count=2
        )

        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()
        update.message.chat.id = -999
        update.message.message_thread_id = None

        with (
            patch(
                "ccgram.handlers.sync_command.safe_reply",
                new_callable=AsyncMock,
            ) as mock_reply,
            patch(
                "ccgram.handlers.sync_command.safe_edit",
                new_callable=AsyncMock,
            ) as mock_edit,
            patch("ccgram.handlers.sync_command.logger") as mock_logger,
        ):
            await sync_command(update, MagicMock())
            mock_reply.assert_awaited_once_with(update.message, "🔍 State audit…")
            mock_sm.audit_state.assert_called_once()
            mock_edit.assert_awaited_once()
            assert "2 topics bound" in mock_edit.call_args.args[1]

        assert mock_logger.info.call_args_list[0].args == (
            "State audit command started",
        )
        assert mock_logger.info.call_args_list[0].kwargs == {
            "chat_id": -999,
            "thread_id": None,
        }
        assert mock_logger.info.call_args_list[-1].args == (
            "State audit command completed",
        )

    async def test_topic_probe_timeout_still_returns_audit_report(
        self, _patch_deps
    ) -> None:
        mock_sm, _, _, _, _, _ = _patch_deps
        mock_sm.audit_state.return_value = AuditResult(
            issues=[], total_bindings=2, live_binding_count=2
        )

        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()
        update.message.chat.id = -999
        update.message.message_thread_id = None
        update.get_bot.return_value = AsyncMock()
        status_message = MagicMock()

        async def never_finishes(_client) -> list[AuditIssue]:
            await asyncio.Event().wait()
            return []

        with (
            patch(
                "ccgram.handlers.sync_command.safe_reply",
                new_callable=AsyncMock,
                return_value=status_message,
            ),
            patch(
                "ccgram.handlers.sync_command.safe_edit",
                new_callable=AsyncMock,
            ) as mock_edit,
            patch(
                "ccgram.handlers.sync_command._probe_dead_topics",
                new_callable=AsyncMock,
                side_effect=never_finishes,
            ),
            patch("ccgram.handlers.sync_command._TELEGRAM_PROBE_TIMEOUT_S", 0.01),
        ):
            await sync_command(update, MagicMock())

        mock_edit.assert_awaited_once()
        assert "Telegram topic check incomplete" in mock_edit.call_args.args[1]

    async def test_audit_does_not_mutate_live_topic_names(self, _patch_deps) -> None:
        mock_sm, _, _, mock_tr, mock_tm, _ = _patch_deps
        mock_sm.audit_state.return_value = AuditResult(
            issues=[], total_bindings=1, live_binding_count=1
        )
        mock_tm.list_windows.return_value = [
            MagicMock(window_id="@7", window_name="ccgram-codex")
        ]
        mock_tr.iter_thread_bindings.return_value = [(100, 42, "@7")]
        mock_tr.resolve_chat_id.return_value = -999
        mock_tr.get_display_name.return_value = "ccgram-codex"

        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()
        bot = AsyncMock()
        bot.send_message.return_value = MagicMock(message_id=999)
        update.get_bot.return_value = bot

        with (
            patch("ccgram.handlers.sync_command.safe_reply"),
            patch(
                "ccgram.handlers.sync_command.sync_topic_name",
                new_callable=AsyncMock,
            ) as mock_sync_topic_name,
        ):
            await sync_command(update, MagicMock())

        mock_sync_topic_name.assert_not_awaited()


class TestSyncFix:
    async def test_unavailable_listing_does_not_change_state(self, _patch_deps) -> None:
        mock_sm, mock_sms, _, _, mock_tm, _ = _patch_deps
        mock_tm.list_windows_for_reconciliation.return_value = None
        query = MagicMock()

        with patch("ccgram.handlers.sync_command.safe_edit") as mock_edit:
            await handle_sync_fix(query)

        mock_sm.sync_display_names.assert_not_called()
        mock_sm.prune_stale_state.assert_not_called()
        mock_sms.prune_session_map.assert_not_called()
        mock_sm.prune_stale_window_states.assert_not_called()
        assert mock_edit.call_count == 2
        assert "No state changes were made" in mock_edit.call_args.args[1]

    async def test_fix_runs_cleanup_and_re_audits(self, _patch_deps) -> None:
        mock_sm, mock_sms, _, _, _, _ = _patch_deps
        mock_sm.audit_state.side_effect = [
            AuditResult(
                issues=[
                    AuditIssue("orphaned_display_name", "@7 (old)", fixable=True),
                ],
                total_bindings=2,
                live_binding_count=2,
            ),
            AuditResult(issues=[], total_bindings=2, live_binding_count=2),
        ]

        query = MagicMock()

        with (
            patch("ccgram.handlers.sync_command.safe_edit") as mock_edit,
            patch(
                "ccgram.handlers.sync_command._sync_live_topic_names",
                new_callable=AsyncMock,
            ) as mock_sync_topic_names,
        ):
            await handle_sync_fix(query)
            mock_sm.sync_display_names.assert_called_once_with([])
            mock_sm.prune_stale_state.assert_called_once_with(set())
            mock_sms.prune_session_map.assert_called_once_with(set())
            mock_sm.prune_stale_window_states.assert_called_once_with(set())
            mock_sync_topic_names.assert_awaited_once()
            assert mock_sync_topic_names.call_args.args[1] == set()
            assert mock_sm.audit_state.call_count == 2
            assert mock_edit.call_count == 2
            assert "🔧 Fixing…" in mock_edit.call_args_list[0].args[1]
            assert "\u2705 Fixed 1 issue" in mock_edit.call_args_list[1].args[1]

    async def test_fix_computes_actual_fixed_count(self, _patch_deps) -> None:
        mock_sm, _, _, _, _, _ = _patch_deps
        mock_sm.audit_state.side_effect = [
            AuditResult(
                issues=[
                    AuditIssue("orphaned_display_name", "@7", fixable=True),
                    AuditIssue("stale_offset", "user 1, @9", fixable=True),
                ],
                total_bindings=1,
                live_binding_count=1,
            ),
            AuditResult(
                issues=[
                    AuditIssue("stale_offset", "user 1, @9", fixable=True),
                ],
                total_bindings=1,
                live_binding_count=1,
            ),
        ]

        query = MagicMock()

        with patch("ccgram.handlers.sync_command.safe_edit") as mock_edit:
            await handle_sync_fix(query)
            assert "\u2705 Fixed 1 issue" in mock_edit.call_args[0][1]

    async def test_fix_closes_ghost_topics(self, _patch_deps) -> None:
        mock_sm, _, _, mock_tr, _, _ = _patch_deps
        mock_sm.audit_state.side_effect = [
            AuditResult(
                issues=[
                    AuditIssue(
                        "ghost_binding",
                        "user:100 thread:42 window:w2:t1 (dead)",
                        fixable=True,
                    ),
                ],
                total_bindings=1,
                live_binding_count=0,
            ),
            AuditResult(issues=[], total_bindings=0, live_binding_count=0),
        ]
        mock_tr.resolve_chat_id.return_value = -999
        mock_tr.get_window_for_thread.return_value = "w2:t1"

        query = MagicMock()
        mock_bot = AsyncMock()
        query.get_bot.return_value = mock_bot

        with (
            patch("ccgram.handlers.sync_command.safe_edit") as mock_edit,
            patch("ccgram.handlers.sync_command.clear_topic_state") as mock_cleanup,
        ):
            await handle_sync_fix(query)
            mock_bot.delete_forum_topic.assert_called_once_with(
                chat_id=-999, message_thread_id=42
            )
            mock_cleanup.assert_called_once()
            cleanup_args = mock_cleanup.call_args
            assert cleanup_args.args[:2] == (100, 42)
            assert cleanup_args.kwargs["window_id"] == "w2:t1"
            assert cleanup_args.kwargs["client"].bot is mock_bot
            mock_tr.unbind_thread.assert_called_once_with(
                100, 42, retirement_reason="remote_removed"
            )
            report_text = mock_edit.call_args[0][1]
            assert "Removed 1 stale topic" in report_text

    async def test_fix_skips_unbind_when_close_fails(self, _patch_deps) -> None:
        mock_sm, _, _, mock_tr, _, _ = _patch_deps
        mock_sm.audit_state.side_effect = [
            AuditResult(
                issues=[
                    AuditIssue(
                        "ghost_binding",
                        "user:100 thread:42 window:w2:t1 (dead)",
                        fixable=True,
                    ),
                ],
                total_bindings=1,
                live_binding_count=0,
            ),
            AuditResult(
                issues=[
                    AuditIssue(
                        "ghost_binding",
                        "user:100 thread:42 window:w2:t1 (dead)",
                        fixable=True,
                    ),
                ],
                total_bindings=1,
                live_binding_count=0,
            ),
        ]
        mock_tr.resolve_chat_id.return_value = -999
        mock_tr.get_window_for_thread.return_value = "w2:t1"

        query = MagicMock()
        mock_bot = AsyncMock()
        mock_bot.delete_forum_topic.side_effect = TelegramError("Forbidden")
        mock_bot.close_forum_topic.side_effect = TelegramError("Forbidden")
        query.get_bot = MagicMock(return_value=mock_bot)

        with (
            patch("ccgram.handlers.sync_command.safe_edit") as mock_edit,
            patch("ccgram.handlers.sync_command.clear_topic_state") as mock_cleanup,
        ):
            await handle_sync_fix(query)
            mock_bot.delete_forum_topic.assert_called_once()
            mock_bot.close_forum_topic.assert_called_once()
            mock_cleanup.assert_not_called()
            mock_tr.unbind_thread.assert_not_called()
            report_text = mock_edit.call_args[0][1]
            assert "safe to close manually" in report_text

    async def test_fix_skips_close_when_no_group_chat(self, _patch_deps) -> None:
        mock_sm, _, _, mock_tr, _, _ = _patch_deps
        mock_sm.audit_state.side_effect = [
            AuditResult(
                issues=[
                    AuditIssue(
                        "ghost_binding",
                        "user:100 thread:42 window:@7 (dead)",
                        fixable=True,
                    ),
                ],
                total_bindings=1,
                live_binding_count=0,
            ),
            AuditResult(issues=[], total_bindings=0, live_binding_count=0),
        ]
        mock_tr.resolve_chat_id.return_value = 100
        mock_tr.get_window_for_thread.return_value = "@7"

        query = MagicMock()
        mock_bot = AsyncMock()
        query.get_bot.return_value = mock_bot

        with (
            patch("ccgram.handlers.sync_command.safe_edit"),
            patch("ccgram.handlers.sync_command.clear_topic_state") as mock_cleanup,
        ):
            await handle_sync_fix(query)
            mock_bot.close_forum_topic.assert_not_called()
            mock_cleanup.assert_called_once()
            cleanup_args = mock_cleanup.call_args
            assert cleanup_args.args[:2] == (100, 42)
            assert cleanup_args.kwargs["window_id"] == "@7"
            assert cleanup_args.kwargs["client"].bot is mock_bot
            mock_tr.unbind_thread.assert_called_once_with(
                100, 42, retirement_reason="remote_removed"
            )

    @staticmethod
    def _listing(mock_tm, *refs) -> None:
        """Set the confirmed listing both the audit and adoption read.

        An orphaned_window issue means a live unbound window, so the window
        must be in the listing: an issue with an empty listing is a state no
        backend produces.
        """
        mock_tm.list_windows_for_reconciliation.return_value = list(refs)

    @staticmethod
    def _ref(window_id: str, name: str, *, eligible: bool = True):
        from ccgram.multiplexer.base import WindowRef

        return WindowRef(
            window_id=window_id,
            window_name=name,
            cwd="/tmp",
            topic_eligible=eligible,
        )

    async def test_fix_adopts_orphaned_windows(self, _patch_deps) -> None:
        mock_sm, _, mock_wq, _, mock_tm, _ = _patch_deps
        self._listing(mock_tm, self._ref("w2:t5", "stray-proj"))
        mock_sm.audit_state.side_effect = [
            AuditResult(
                issues=[
                    AuditIssue("orphaned_window", "w2:t5 (stray)", fixable=True),
                ],
                total_bindings=1,
                live_binding_count=1,
            ),
            AuditResult(issues=[], total_bindings=1, live_binding_count=1),
        ]
        mock_wq.view_window.return_value = MagicMock(
            session_id="s1", cwd="/tmp", window_name="stray-proj"
        )

        query = MagicMock()

        with (
            patch("ccgram.handlers.sync_command.safe_edit"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.handle_new_window",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            await handle_sync_fix(query)
            mock_handle.assert_called_once()
            event = mock_handle.call_args[0][0]
            assert event.window_id == "w2:t5"
            assert event.window_name == "stray-proj"

    async def test_fix_does_not_adopt_a_window_that_became_ineligible(
        self, _patch_deps
    ) -> None:
        """The audit's verdict is re-read at the point of adoption.

        /sync Fix probes every bound topic over the network between the audit
        and the adoption, so the listing behind an orphaned_window issue can be
        seconds old. A window that moved out of scope, or went away, in that
        gap must not be handed a topic.
        """
        mock_sm, _, mock_wq, _, mock_tm, _ = _patch_deps
        mock_sm.audit_state.side_effect = [
            AuditResult(
                issues=[AuditIssue("orphaned_window", "w2:t5 (stray)", fixable=True)],
                total_bindings=1,
                live_binding_count=1,
            ),
            AuditResult(issues=[], total_bindings=1, live_binding_count=1),
        ]
        mock_wq.view_window.return_value = MagicMock(
            session_id="s1", cwd="/tmp", window_name="stray-proj"
        )
        # First read (the audit) sees it adoptable; by the adoption read it has
        # moved out of scope.
        mock_tm.list_windows_for_reconciliation.side_effect = [
            [self._ref("w2:t5", "stray-proj")],
            [self._ref("w2:t5", "stray-proj", eligible=False)],
            [self._ref("w2:t5", "stray-proj", eligible=False)],
        ]

        query = MagicMock()

        with (
            patch("ccgram.handlers.sync_command.safe_edit"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.handle_new_window",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            await handle_sync_fix(query)

        mock_handle.assert_not_called()

    async def test_second_orphan_is_judged_after_the_first_is_created(
        self, _patch_deps
    ) -> None:
        """The re-read is per candidate, not once for the batch.

        Creating a topic is several Telegram round-trips on its own, so a
        batch-wide snapshot judges the second orphan on a listing taken before
        the first one's topic existed. Here B leaves scope while A is being
        created, and must not reach handle_new_window.
        """
        mock_sm, _, mock_wq, _, mock_tm, _ = _patch_deps
        mock_sm.audit_state.side_effect = [
            AuditResult(
                issues=[
                    AuditIssue("orphaned_window", "w2:tA (a)", fixable=True),
                    AuditIssue("orphaned_window", "w2:tB (b)", fixable=True),
                ],
                total_bindings=2,
                live_binding_count=2,
            ),
            AuditResult(issues=[], total_bindings=2, live_binding_count=2),
        ]
        mock_wq.view_window.return_value = MagicMock(
            session_id="s1", cwd="/tmp", window_name="proj"
        )

        both_live = [self._ref("w2:tA", "a"), self._ref("w2:tB", "b")]
        b_left_scope = [
            self._ref("w2:tA", "a"),
            self._ref("w2:tB", "b", eligible=False),
        ]

        # Creating A's topic is what takes B out of scope, so the flip happens
        # exactly where a batch-wide snapshot would already have been taken.
        state = {"windows": both_live}

        async def _listing(*_a, **_kw):
            return state["windows"]

        # Mutate, never rebind: the fixture aliases this attribute to the
        # module-level helper patch, and replacing it breaks the alias.
        mock_tm.list_windows_for_reconciliation.side_effect = _listing

        query = MagicMock()

        with (
            patch("ccgram.handlers.sync_command.safe_edit"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.handle_new_window",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):

            async def _create(*_a, **_kw):
                state["windows"] = b_left_scope

            mock_handle.side_effect = _create
            await handle_sync_fix(query)

        adopted = [call[0][0].window_id for call in mock_handle.call_args_list]
        assert adopted == ["w2:tA"]

    async def test_fix_adopts_nothing_when_the_listing_goes_away(
        self, _patch_deps
    ) -> None:
        """Adoption is the one step here that is safe to skip and retry."""
        mock_sm, _, mock_wq, _, mock_tm, _ = _patch_deps
        mock_sm.audit_state.side_effect = [
            AuditResult(
                issues=[AuditIssue("orphaned_window", "w2:t5 (stray)", fixable=True)],
                total_bindings=1,
                live_binding_count=1,
            ),
            AuditResult(issues=[], total_bindings=1, live_binding_count=1),
        ]
        mock_wq.view_window.return_value = MagicMock(
            session_id="s1", cwd="/tmp", window_name="stray-proj"
        )
        mock_tm.list_windows_for_reconciliation.side_effect = [
            [self._ref("w2:t5", "stray-proj")],
            None,
            [self._ref("w2:t5", "stray-proj")],
        ]

        query = MagicMock()

        with (
            patch("ccgram.handlers.sync_command.safe_edit"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.handle_new_window",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            await handle_sync_fix(query)

        mock_handle.assert_not_called()


class TestDeadTopicDetection:
    async def test_probe_detects_dead_topic(self, _patch_deps) -> None:
        _, _, _, mock_tr, _, _ = _patch_deps
        mock_tr.iter_thread_bindings.return_value = [(100, 42, "@2")]
        mock_tr.resolve_chat_id.return_value = -999
        mock_tr.get_display_name.return_value = "qmd-go"

        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = BadRequest("Message thread not found")

        issues = await _probe_dead_topics(mock_bot)
        assert len(issues) == 1
        assert issues[0].category == "dead_topic"
        assert "window:@2" in issues[0].detail
        assert issues[0].fixable is True

    async def test_probe_skips_alive_topic(self, _patch_deps) -> None:
        _, _, _, mock_tr, _, _ = _patch_deps
        mock_tr.iter_thread_bindings.return_value = [(100, 42, "@2")]
        mock_tr.resolve_chat_id.return_value = -999

        mock_bot = AsyncMock()
        mock_bot.send_message.return_value = MagicMock(message_id=999)

        issues = await _probe_dead_topics(mock_bot)

        assert issues == []
        mock_bot.send_message.assert_awaited_once_with(
            -999,
            ".",
            message_thread_id=42,
            disable_notification=True,
        )
        mock_bot.delete_message.assert_awaited_once_with(-999, 999)

    async def test_probe_skips_network_errors(self, _patch_deps) -> None:
        _, _, _, mock_tr, _, _ = _patch_deps
        mock_tr.iter_thread_bindings.return_value = [(100, 42, "@2")]
        mock_tr.resolve_chat_id.return_value = -999

        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = TelegramError("Network error")

        issues = await _probe_dead_topics(mock_bot)
        assert issues == []

    async def test_probe_includes_private_chat_binding(self, _patch_deps) -> None:
        _, _, _, mock_tr, _, _ = _patch_deps
        mock_tr.iter_thread_bindings.return_value = [(100, 42, "@2")]
        mock_tr.resolve_chat_id.return_value = 100
        mock_bot = AsyncMock()
        mock_bot.send_message.return_value = MagicMock(message_id=999)

        issues = await _probe_dead_topics(mock_bot)

        assert issues == []
        mock_bot.send_message.assert_awaited_once_with(
            100,
            ".",
            message_thread_id=42,
            disable_notification=True,
        )
        mock_bot.delete_message.assert_awaited_once_with(100, 999)


class TestPrivateTopicSyncLifecycle:
    async def test_closes_and_unbinds_private_ghost_topic(self, _patch_deps) -> None:
        _, _, _, mock_tr, _, _ = _patch_deps
        mock_tr.get_window_for_thread.return_value = "@2"
        mock_tr.resolve_chat_id.return_value = 100
        issue = AuditIssue(
            "ghost_binding", "user:100 thread:42 window:@2 (private)", fixable=True
        )
        client = AsyncMock()

        with patch(
            "ccgram.handlers.sync_command.clear_topic_state", new_callable=AsyncMock
        ) as clear_state:
            closed, manual_close = await _close_ghost_topics(client, [issue])

        assert (closed, manual_close) == (1, 0)
        client.delete_forum_topic.assert_awaited_once_with(100, 42)
        clear_state.assert_awaited_once_with(100, 42, client=client, window_id="@2")
        mock_tr.unbind_thread.assert_called_once_with(
            100, 42, retirement_reason="remote_removed"
        )


class TestDeadTopicRecreation:
    """A dead topic is a live window whose Telegram topic was deleted.

    So the window is present in the listing throughout: recreation is refused
    unless it is confirmed present, because the repair unbinds the thread
    before creating the replacement.
    """

    @pytest.fixture(autouse=True)
    def _window_is_live(self, _patch_deps):
        """Whatever window a case binds, report it live."""
        from ccgram.multiplexer.base import WindowRef

        _, _, _, mock_tr, mock_tm, _ = _patch_deps

        async def _listing(*_a, **_kw):
            bound = mock_tr.get_window_for_thread.return_value
            if not isinstance(bound, str):
                return []
            return [WindowRef(window_id=bound, window_name="proj", cwd="/tmp/proj")]

        mock_tm.list_windows_for_reconciliation.side_effect = _listing

    async def test_skips_stale_issue_after_topic_was_rebound(self, _patch_deps) -> None:
        _mock_sm, _mock_sms, mock_wq, mock_tr, _mock_tm, _mock_cfg = _patch_deps
        mock_wq.view_window.return_value = MagicMock(
            session_id="old", cwd="/tmp/proj", window_name="reflex-gh"
        )
        mock_tr.get_window_for_thread.return_value = None
        issues = [
            AuditIssue(
                "dead_topic",
                "user:100 thread:42 window:@1 (reflex-gh)",
                fixable=True,
            ),
            AuditIssue(
                "ghost_binding",
                "user:100 thread:42 window:@1 (reflex-gh)",
                fixable=True,
            ),
        ]
        bot = AsyncMock()

        with patch(
            "ccgram.handlers.topics.topic_orchestration.handle_new_window",
            new_callable=AsyncMock,
        ) as mock_handle:
            recreated = await _recreate_dead_topics(bot, issues)
            closed, manual_close = await _close_ghost_topics(bot, issues)

        assert recreated == 0
        assert closed == 0
        assert manual_close == 0
        mock_handle.assert_not_called()
        mock_tr.unbind_thread.assert_not_called()
        bot.delete_forum_topic.assert_not_called()

    async def test_recreate_unbinds_and_creates_topic(self, _patch_deps) -> None:
        mock_sm, _, mock_wq, mock_tr, _, _ = _patch_deps
        mock_wq.view_window.return_value = MagicMock(
            session_id="s1", cwd="/tmp/proj", window_name="qmd-go"
        )
        mock_tr.get_window_for_thread.return_value = "w2:t2"

        issues = [
            AuditIssue(
                "dead_topic",
                "user:100 thread:42 window:w2:t2 (qmd-go)",
                fixable=True,
            ),
        ]

        mock_bot = AsyncMock()

        with patch(
            "ccgram.handlers.topics.topic_orchestration.handle_new_window",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_handle:
            count = await _recreate_dead_topics(mock_bot, issues)
            assert count == 1
            mock_tr.unbind_thread.assert_called_once_with(
                100, 42, retirement_reason="remote_deleted"
            )
            mock_handle.assert_called_once()
            event = mock_handle.call_args[0][0]
            assert event.window_id == "w2:t2"
            assert event.window_name == "qmd-go"
            assert mock_handle.call_args.kwargs == {
                "target_user_id": 100,
                "target_chat_id": mock_tr.resolve_chat_id.return_value,
            }

    async def test_recreate_restores_binding_when_creation_returns_false(
        self, _patch_deps
    ) -> None:
        _, _, mock_wq, mock_tr, _, _ = _patch_deps
        mock_wq.view_window.return_value = MagicMock(
            session_id="s1", cwd="/tmp", window_name="proj"
        )
        mock_tr.get_window_for_thread.return_value = "@2"
        mock_tr.resolve_chat_id.return_value = -999
        issues = [
            AuditIssue(
                "dead_topic",
                "user:100 thread:42 window:@2 (proj)",
                fixable=True,
            ),
        ]

        with patch(
            "ccgram.handlers.topics.topic_orchestration.handle_new_window",
            new_callable=AsyncMock,
            return_value=False,
        ):
            count = await _recreate_dead_topics(AsyncMock(), issues)

        assert count == 0
        mock_tr.bind_thread.assert_called_once_with(
            100, 42, "@2", window_name="proj", chat_id=-999
        )
        mock_tr.set_group_chat_id.assert_called_once_with(100, 42, -999)

    async def test_recreate_restores_binding_when_cancelled(self, _patch_deps) -> None:
        _, _, mock_wq, mock_tr, _, _ = _patch_deps
        mock_wq.view_window.return_value = MagicMock(
            session_id="s1", cwd="/tmp", window_name="proj"
        )
        mock_tr.get_window_for_thread.return_value = "@2"
        mock_tr.resolve_chat_id.return_value = -999
        issues = [
            AuditIssue(
                "dead_topic",
                "user:100 thread:42 window:@2 (proj)",
                fixable=True,
            ),
        ]

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.handle_new_window",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await _recreate_dead_topics(AsyncMock(), issues)

        mock_tr.bind_thread.assert_called_once_with(
            100, 42, "@2", window_name="proj", chat_id=-999
        )
        mock_tr.set_group_chat_id.assert_called_once_with(100, 42, -999)

    async def test_recreate_skips_non_dead_topic_issues(self, _patch_deps) -> None:
        issues = [
            AuditIssue("ghost_binding", "user:100 thread:42 window:@7", fixable=True),
        ]
        mock_bot = AsyncMock()

        with patch(
            "ccgram.handlers.topics.topic_orchestration.handle_new_window",
            new_callable=AsyncMock,
        ) as mock_handle:
            count = await _recreate_dead_topics(mock_bot, issues)
            assert count == 0
            mock_handle.assert_not_called()

    async def test_recreate_handles_telegram_error(self, _patch_deps) -> None:
        mock_sm, _, mock_wq, mock_tr, _, _ = _patch_deps
        mock_wq.view_window.return_value = MagicMock(
            session_id="s1", cwd="/tmp", window_name="proj"
        )
        mock_tr.get_window_for_thread.return_value = "@2"
        mock_tr.resolve_chat_id.return_value = -999

        issues = [
            AuditIssue(
                "dead_topic",
                "user:100 thread:42 window:@2 (proj)",
                fixable=True,
            ),
        ]

        mock_bot = AsyncMock()

        with patch(
            "ccgram.handlers.topics.topic_orchestration.handle_new_window",
            new_callable=AsyncMock,
            side_effect=TelegramError("Failed"),
        ):
            count = await _recreate_dead_topics(mock_bot, issues)
            assert count == 0
            mock_tr.unbind_thread.assert_called_once_with(
                100, 42, retirement_reason="remote_deleted"
            )
            mock_tr.bind_thread.assert_called_once_with(
                100, 42, "@2", window_name="proj", chat_id=-999
            )


class TestSyncFixDeadTopic:
    async def test_fix_recreates_dead_topics(self, _patch_deps) -> None:
        from ccgram.multiplexer.base import WindowRef

        mock_sm, _, mock_wq, mock_tr, mock_tm, _ = _patch_deps
        # The window is live throughout — a dead *topic* is a deleted Telegram
        # topic, not a dead window — and recreation is refused unless the
        # window is confirmed present at that point.
        mock_tm.list_windows_for_reconciliation.return_value = [
            WindowRef(window_id="@2", window_name="qmd-go", cwd="/tmp")
        ]
        mock_sm.audit_state.side_effect = [
            AuditResult(issues=[], total_bindings=1, live_binding_count=1),
            AuditResult(issues=[], total_bindings=1, live_binding_count=1),
        ]
        mock_tr.iter_thread_bindings.side_effect = [
            [(100, 42, "@2")],  # pre-audit probe
            [],  # prune_stale_offsets
            [],  # live topic-name reconciliation
            [],  # post-fix probe (already unbound)
        ]
        mock_tr.resolve_chat_id.return_value = -999
        mock_tr.get_display_name.return_value = "qmd-go"
        mock_tr.get_window_for_thread.return_value = "@2"
        mock_wq.view_window.return_value = MagicMock(
            session_id="s1", cwd="/tmp", window_name="qmd-go"
        )

        query = MagicMock()
        mock_bot = AsyncMock()
        mock_bot.send_message.side_effect = [
            BadRequest("Message thread not found"),  # pre-audit
        ]
        query.get_bot.return_value = mock_bot

        with (
            patch("ccgram.handlers.sync_command.safe_edit") as mock_edit,
            patch(
                "ccgram.handlers.topics.topic_orchestration.handle_new_window",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            await handle_sync_fix(query)
            mock_tr.unbind_thread.assert_called_once_with(
                100, 42, retirement_reason="remote_deleted"
            )
            mock_handle.assert_called_once()
            report_text = mock_edit.call_args[0][1]
            assert "Recreated 1 topic" in report_text


class TestSyncAuditSeparatesLivenessFromAdoption:
    """`/sync` Fix adopts the orphans the audit reports, so adoption takes the
    backend's verdict — but liveness must stay complete, or a live binding to
    an excluded window becomes a fixable ghost that Fix closes.
    """

    @staticmethod
    def _windows() -> list[WindowRef]:
        return [
            WindowRef(
                window_id="REFUSED",
                window_name="elsewhere",
                cwd="/other",
                pane_current_command="claude",
                topic_eligible=False,
            ),
            WindowRef(
                window_id="ALLOWED",
                window_name="agent",
                cwd="/proj",
                pane_current_command="claude",
            ),
        ]

    async def test_audit_gets_every_window_live_and_only_some_adoptable(self):
        """Backend-shaped: the UI listing omits what the backend refuses.

        Mocking ``list_windows`` to return the refused window described a state
        real agterm cannot produce, so it could not expose the defect. The
        audit must take liveness from the reconciliation listing, which keeps
        the refused window, and adoptability from its ``topic_eligible``.
        """
        with (
            patch("ccgram.handlers.sync_command.tmux_manager") as mock_tmux,
            patch(
                "ccgram.handlers.sync_command.list_windows_for_reconciliation",
                new_callable=AsyncMock,
                return_value=self._windows(),
            ),
            patch("ccgram.handlers.sync_command.session_manager") as mock_sm,
        ):
            # what a real backend does: the UI listing hides the refused one
            mock_tmux.list_windows = AsyncMock(
                return_value=[w for w in self._windows() if w.topic_eligible]
            )
            mock_sm.audit_state.return_value = MagicMock(issues=[])

            await _run_audit()

            live_ids, live_pairs, adoptable = mock_sm.audit_state.call_args[0]

        # liveness is not narrowed: the excluded window still exists
        assert live_ids == {"REFUSED", "ALLOWED"}
        assert {wid for wid, _ in live_pairs} == {"REFUSED", "ALLOWED"}
        # adoption is
        assert adoptable == {"ALLOWED"}

    async def test_an_unavailable_listing_does_not_declare_everything_dead(self):
        with (
            patch(
                "ccgram.handlers.sync_command.list_windows_for_reconciliation",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("ccgram.handlers.sync_command.session_manager") as mock_sm,
        ):
            result = await _run_audit()

        assert result is None
        mock_sm.audit_state.assert_not_called()

    def test_a_live_but_ineligible_binding_is_not_a_ghost(self):
        """The regression my first attempt at this introduced.

        Narrowing the live set made an existing topic, whose session merely
        stopped qualifying for new adoption, look like a fixable ghost.
        """
        sm = SessionManager()
        thread_router.bind_thread(1, 2, "REFUSED", chat_id=-100)

        audit = sm.audit_state(
            {"REFUSED", "ALLOWED"},
            [("REFUSED", "elsewhere"), ("ALLOWED", "agent")],
            {"ALLOWED"},
        )

        ghosts = [i for i in audit.issues if i.category == "ghost_binding"]
        assert not ghosts, "a live window is not a ghost merely by being excluded"

    def test_an_ineligible_window_is_not_a_fixable_orphan(self):
        """Both windows must be orphan *candidates*, or this asserts nothing.

        A window becomes an orphan only when ccgram already knows it (session
        map or window state) and no topic is bound to it. With a fresh manager
        that set is empty, no orphans are produced at all, and a check that
        merely says "no orphan mentions REFUSED" passes without running.
        """
        sm = SessionManager()
        with patch.object(
            SessionManager,
            "_get_session_map_window_ids",
            return_value={"REFUSED", "ALLOWED"},
        ):
            audit = sm.audit_state(
                {"REFUSED", "ALLOWED"},
                [("REFUSED", "elsewhere"), ("ALLOWED", "agent")],
                {"ALLOWED"},
            )

        orphans = [i for i in audit.issues if i.category == "orphaned_window"]
        # the eligible one is offered for adoption...
        assert any("ALLOWED" in i.detail for i in orphans), (
            "the eligible window must be a candidate, or the check below is empty"
        )
        # ...and the refused one is not
        assert not any("REFUSED" in i.detail for i in orphans)

    async def test_handle_sync_fix_passes_both_sets_and_never_adopts_the_refused(
        self,
    ):
        """The mutating path. ``handle_sync_fix`` does not call ``_run_audit``,
        so testing that function proved nothing about pressing Fix.
        """
        query = MagicMock()
        query.get_bot.return_value = MagicMock()

        with (
            patch(
                "ccgram.handlers.sync_command.list_windows_for_reconciliation",
                new_callable=AsyncMock,
                return_value=self._windows(),
            ),
            patch("ccgram.handlers.sync_command.session_manager") as mock_sm,
            patch("ccgram.handlers.sync_command.session_map_sync"),
            patch("ccgram.handlers.sync_command.user_preferences"),
            patch("ccgram.handlers.sync_command.window_query"),
            patch("ccgram.handlers.sync_command.safe_edit", new_callable=AsyncMock),
            patch(
                "ccgram.handlers.sync_command._probe_dead_topics",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "ccgram.handlers.sync_command._retired_topic_issues", return_value=[]
            ),
            patch(
                "ccgram.handlers.sync_command._sync_live_topic_names",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.sync_command._adopt_orphaned_windows",
                new_callable=AsyncMock,
            ) as mock_adopt,
            patch(
                "ccgram.handlers.sync_command._close_ghost_topics",
                new_callable=AsyncMock,
                return_value=(0, 0),
            ),
        ):
            mock_sm.audit_state.return_value = MagicMock(issues=[])
            with contextlib.suppress(Exception):
                await handle_sync_fix(query)

            # The later stages are mocked out and can raise on their mocks; the
            # claim is about the audit call the fix path makes.
            audits = [c for c in mock_sm.audit_state.call_args_list if len(c[0]) == 3]
            assert audits, "handle_sync_fix must pass both sets to audit_state"
            live_ids, live_pairs, adoptable = audits[0][0]

        assert live_ids == {"REFUSED", "ALLOWED"}, "pruning needs every window"
        assert adoptable == {"ALLOWED"}
        for call in mock_adopt.call_args_list:
            for issue in call[0][1]:
                assert "REFUSED" not in getattr(issue, "detail", "")
