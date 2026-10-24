import ast
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import TelegramError

from ccgram.delivery_contract import DeliveryReceipt
from ccgram.handlers.messaging_pipeline.message_task import ContentTask
from ccgram.handlers.messaging_pipeline.tool_batch import (
    BATCH_MAX_LENGTH,
    ToolBatch,
    ToolBatchEntry,
    _active_batches,
    _add_tool_use_entry,
    _format_mixed_batch_lines,
    _send_or_edit_batch,
    flush_batch,
    has_active_batch,
    has_ephemeral_active_batch,
    process_tool_event,
    ToolEventOutcome,
)
from ccgram.telegram_draft import (
    DRAFT_LEGACY,
    mark_draft_unavailable,
    reset_draft_state,
)


class TestHasEphemeralActiveBatch:
    """Helper used by the dispatcher to suppress status updates while an
    ephemeral batch owns the bubble (prevents the visible flicker where the
    formatted tool bubble is deleted, a plain status bubble takes its place,
    and the assistant text replaces that)."""

    def setup_method(self) -> None:
        _active_batches.clear()

    def teardown_method(self) -> None:
        _active_batches.clear()

    def test_no_batch_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: True,
        )
        assert has_active_batch(1, 10) is False
        assert has_ephemeral_active_batch(1, 10) is False

    def test_ephemeral_batch_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: True,
        )
        _active_batches[(1, 10)] = ToolBatch(window_id="@0", thread_id=10)
        assert has_active_batch(1, 10) is True
        assert has_ephemeral_active_batch(1, 10) is True

    def test_non_ephemeral_batch_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: False,
        )
        _active_batches[(1, 10)] = ToolBatch(window_id="@0", thread_id=10)
        assert has_active_batch(1, 10) is True
        assert has_ephemeral_active_batch(1, 10) is False


class TestNoImportFromMessageQueue:
    def test_no_import_from_message_queue(self) -> None:
        import ccgram.handlers.messaging_pipeline.tool_batch as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        violations: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "message_queue" in node.module
            ):
                violations.append(f"line {node.lineno}: from {node.module} import ...")
        assert violations == [], f"tool_batch imports from message_queue: {violations}"


class TestDraftStreamIntegration:
    """Verify tool_batch routes send/edit through DraftStream (legacy mode)."""

    @pytest.fixture(autouse=True)
    def _setup_draft_state(self, monkeypatch: pytest.MonkeyPatch):
        reset_draft_state()
        # Force legacy DraftStream mode for deterministic bot.* assertions.
        mark_draft_unavailable("test")
        _active_batches.clear()
        # Avoid real wall-clock rate-limiting in unit tests.
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch._rate_limit_chat",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.thread_router",
            MagicMock(resolve_chat_id=MagicMock(return_value=42)),
        )
        # No status bubble to clear.
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.get_batch_mode",
            lambda _wid: "batched",
        )
        # is_ephemeral_tools is imported directly; with the new ephemeral
        # default it would route through safe_send instead of DraftStream.
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: False,
        )
        yield
        _active_batches.clear()
        reset_draft_state()

    @staticmethod
    def _make_bot(send_id: int = 99):
        bot = AsyncMock()
        sent = MagicMock()
        sent.message_id = send_id
        bot.send_message.return_value = sent
        return bot

    async def test_first_tool_use_starts_draft_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No status bubble to dismiss.
        monkeypatch.setattr(
            "ccgram.handlers.status.status_bubble.clear_status_message",
            AsyncMock(return_value=None),
        )
        bot = self._make_bot(send_id=77)

        batch = ToolBatch(window_id="@0", thread_id=10)
        batch.entries.append(
            ToolBatchEntry(tool_use_id="t1", tool_use_text="Read foo.py")
        )

        await _send_or_edit_batch(
            bot, user_id=1, batch=batch, chat_id=42, raw_thread_id=10, thread_id_or_0=10
        )

        bot.send_message.assert_awaited_once()
        assert batch.draft is not None
        assert batch.draft.mode == DRAFT_LEGACY
        assert batch.telegram_msg_id == 77

    async def test_revocation_after_rate_limit_wait_prevents_tool_batch_send(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ccgram.handlers.messaging_pipeline import tool_batch

        monkeypatch.setattr(
            "ccgram.handlers.status.status_bubble.clear_status_message",
            AsyncMock(return_value=None),
        )
        receipt = DeliveryReceipt()
        receipt.track()
        receipt.close()
        bot = self._make_bot()

        async def revoke_after_rate_limit(_chat_id):
            receipt.revoke()

        monkeypatch.setattr(
            tool_batch,
            "_rate_limit_chat",
            AsyncMock(side_effect=revoke_after_rate_limit),
        )
        task = ContentTask(
            window_id="@0",
            parts=("Read foo.py",),
            content_type="tool_use",
            tool_use_id="t1",
            thread_id=10,
            delivery_receipts=(receipt,),
        )

        result = await process_tool_event(
            bot, user_id=1, task=task, admit=lambda: not receipt.revoked
        )

        assert result.outcome is ToolEventOutcome.INTENTIONALLY_DROPPED
        bot.send_message.assert_not_awaited()

    async def test_current_result_falls_back_when_cached_tool_receipt_is_revoked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale_receipt = DeliveryReceipt()
        stale_receipt.revoke()
        current_receipt = DeliveryReceipt()
        batch = ToolBatch(window_id="@0", thread_id=10)
        batch.entries.append(
            ToolBatchEntry(
                tool_use_id="old-tool",
                tool_use_text="visible old tool",
                delivery_receipts=(stale_receipt,),
            )
        )
        batch.telegram_msg_id = 77
        batch.last_sent_text = "visible old tool"
        _active_batches[(1, 10)] = batch
        task = ContentTask(
            window_id="@0",
            parts=("current result",),
            content_type="tool_result",
            tool_use_id="old-tool",
            thread_id=10,
            delivery_receipts=(current_receipt,),
        )
        bot = self._make_bot()

        result = await process_tool_event(
            bot, user_id=1, task=task, admit=lambda: not current_receipt.revoked
        )

        assert result.followup is task
        assert result.outcome is ToolEventOutcome.DELIVERED
        assert (1, 10) not in _active_batches
        assert batch.last_sent_text == "visible old tool"
        bot.edit_message_text.assert_not_awaited()
        bot.send_message.assert_not_awaited()

    async def test_current_tool_use_starts_fresh_after_published_stale_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale_receipt = DeliveryReceipt()
        stale_receipt.revoke()
        current_receipt = DeliveryReceipt()
        batch = ToolBatch(window_id="@0", thread_id=10)
        batch.entries.append(
            ToolBatchEntry(
                tool_use_id="old-tool",
                tool_use_text="visible old tool",
                delivery_receipts=(stale_receipt,),
            )
        )
        batch.telegram_msg_id = 77
        batch.last_sent_text = "visible old tool"
        _active_batches[(1, 10)] = batch
        task = ContentTask(
            window_id="@0",
            parts=("current tool",),
            content_type="tool_use",
            tool_use_id="new-tool",
            thread_id=10,
            delivery_receipts=(current_receipt,),
        )
        bot = self._make_bot()

        result = await process_tool_event(
            bot, user_id=1, task=task, admit=lambda: not current_receipt.revoked
        )

        assert result.outcome is ToolEventOutcome.DELIVERED
        assert batch.last_sent_text == "visible old tool"
        bot.edit_message_text.assert_not_awaited()
        bot.send_message.assert_awaited_once()
        assert bot.send_message.call_args.kwargs["text"] == "current tool"
        assert _active_batches[(1, 10)] is not batch

    async def test_revoked_batch_entry_does_not_join_current_tool_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.status.status_bubble.clear_status_message",
            AsyncMock(return_value=None),
        )
        bot = self._make_bot()
        stale_receipt = DeliveryReceipt()
        stale_receipt.revoke()
        current_receipt = DeliveryReceipt()
        batch = ToolBatch(window_id="@0", thread_id=10)
        batch.entries.extend(
            [
                ToolBatchEntry(
                    tool_use_id="stale",
                    tool_use_text="stale tool",
                    delivery_receipts=(stale_receipt,),
                ),
                ToolBatchEntry(
                    tool_use_id="current",
                    tool_use_text="current tool",
                    delivery_receipts=(current_receipt,),
                ),
            ]
        )

        delivered = await _send_or_edit_batch(
            bot,
            user_id=1,
            batch=batch,
            chat_id=42,
            raw_thread_id=10,
            thread_id_or_0=10,
            admit=lambda: not current_receipt.revoked,
        )

        assert delivered is True
        sent_text = bot.send_message.call_args.kwargs["text"]
        assert "stale tool" not in sent_text
        assert "current tool" in sent_text

    async def test_failed_initial_draft_is_not_acknowledged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.status.status_bubble.clear_status_message",
            AsyncMock(return_value=None),
        )
        bot = self._make_bot()
        bot.send_message.side_effect = TelegramError("temporary")
        task = ContentTask(
            window_id="@0",
            parts=("Read foo.py",),
            content_type="tool_use",
            tool_use_id="t1",
            thread_id=10,
        )

        result = await process_tool_event(bot, user_id=1, task=task)

        assert result.outcome is ToolEventOutcome.FAILED

    async def test_noop_re_render_does_not_re_edit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Result arrival doesn't change rendered text — must not re-edit.

        Re-editing with identical text would trigger Telegram's "Message is
        not modified" error and the legacy fallback path would strip the
        entities, leaving the bubble visibly unformatted.
        """
        monkeypatch.setattr(
            "ccgram.handlers.status.status_bubble.clear_status_message",
            AsyncMock(return_value=None),
        )
        bot = self._make_bot(send_id=77)

        batch = ToolBatch(window_id="@0", thread_id=10)
        batch.entries.append(
            ToolBatchEntry(tool_use_id="t1", tool_use_text="Read foo.py")
        )
        await _send_or_edit_batch(bot, 1, batch, 42, 10, 10)

        # Standard entries don't render tool_result_text — re-render produces
        # identical text and the edit must be skipped.
        batch.entries[0].tool_result_text = "42 lines"
        await _send_or_edit_batch(bot, 1, batch, 42, 10, 10)

        bot.send_message.assert_awaited_once()
        bot.edit_message_text.assert_not_awaited()

    async def test_second_call_edits_when_text_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second entry arriving genuinely changes the rendered text → real edit."""
        monkeypatch.setattr(
            "ccgram.handlers.status.status_bubble.clear_status_message",
            AsyncMock(return_value=None),
        )
        bot = self._make_bot(send_id=77)

        batch = ToolBatch(window_id="@0", thread_id=10)
        batch.entries.append(
            ToolBatchEntry(tool_use_id="t1", tool_use_text="Read foo.py")
        )
        await _send_or_edit_batch(bot, 1, batch, 42, 10, 10)

        batch.entries.append(ToolBatchEntry(tool_use_id="t2", tool_use_text="Bash ls"))
        await _send_or_edit_batch(bot, 1, batch, 42, 10, 10)

        bot.send_message.assert_awaited_once()
        bot.edit_message_text.assert_awaited_once()

    async def test_failed_legacy_edit_is_not_acknowledged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.status.status_bubble.clear_status_message",
            AsyncMock(return_value=None),
        )
        bot = self._make_bot(send_id=77)
        batch = ToolBatch(window_id="@0", thread_id=10)
        batch.entries.append(
            ToolBatchEntry(tool_use_id="t1", tool_use_text="Read foo.py")
        )
        assert await _send_or_edit_batch(bot, 1, batch, 42, 10, 10) is True
        previous_text = batch.last_sent_text
        batch.entries.append(ToolBatchEntry(tool_use_id="t2", tool_use_text="Bash ls"))
        bot.edit_message_text.side_effect = TelegramError("edit failed")

        with pytest.raises(TelegramError, match="edit failed"):
            await _send_or_edit_batch(bot, 1, batch, 42, 10, 10)

        assert batch.last_sent_text == previous_text

    async def test_flush_batch_finalizes_active_draft(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.status.status_bubble.clear_status_message",
            AsyncMock(return_value=None),
        )
        bot = self._make_bot(send_id=77)
        batch = ToolBatch(window_id="@0", thread_id=10)
        batch.entries.append(
            ToolBatchEntry(tool_use_id="t1", tool_use_text="Read foo.py")
        )
        _active_batches[(1, 10)] = batch

        await _send_or_edit_batch(bot, 1, batch, 42, 10, 10)
        assert batch.draft is not None and not batch.draft.closed

        await flush_batch(bot, user_id=1, thread_id_or_0=10)

        # Draft is closed, removed from active.
        assert batch.draft.closed is True
        assert (1, 10) not in _active_batches

    async def test_flush_batch_no_op_when_no_entries(self) -> None:
        bot = self._make_bot()
        await flush_batch(bot, user_id=1, thread_id_or_0=10)
        bot.send_message.assert_not_called()
        bot.edit_message_text.assert_not_called()


class TestNativeDraftIntegration:
    async def test_tool_batch_forces_persistent_message_when_drafts_available(
        self, monkeypatch
    ) -> None:
        from ccgram.telegram_draft import reset_draft_state

        reset_draft_state()
        monkeypatch.setattr(
            "ccgram.handlers.status.status_bubble.clear_status_message",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch._rate_limit_chat",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.get_batch_mode",
            lambda _wid: "batched",
        )
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: False,
        )
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.thread_router",
            MagicMock(resolve_chat_id=MagicMock(return_value=42)),
        )

        bot = AsyncMock()
        bot.send_message_draft = AsyncMock(return_value=True)
        sent = MagicMock(message_id=77)
        bot.send_message.return_value = sent
        batch = ToolBatch(window_id="@0", thread_id=10)
        batch.entries.append(
            ToolBatchEntry(tool_use_id="t1", tool_use_text="Read foo.py")
        )
        _active_batches[(1, 10)] = batch

        delivered = await _send_or_edit_batch(bot, 1, batch, 42, 10, 10)

        assert delivered is True
        assert batch.draft is not None
        assert batch.draft.mode == DRAFT_LEGACY
        assert batch.telegram_msg_id == 77
        assert batch.last_sent_text == "Read foo.py"
        bot.send_message_draft.assert_not_awaited()
        bot.send_message.assert_awaited_once_with(
            chat_id=42, text="Read foo.py", message_thread_id=10
        )

    async def test_failed_persistent_send_can_be_retried(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch._rate_limit_chat",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.thread_router",
            MagicMock(resolve_chat_id=MagicMock(return_value=42)),
        )
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.get_batch_mode",
            lambda _wid: "batched",
        )
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: False,
        )
        bot = AsyncMock()
        bot.send_message_draft = AsyncMock(return_value=True)
        bot.send_message.side_effect = [
            TelegramError("temporary"),
            MagicMock(message_id=9),
        ]
        batch = ToolBatch(window_id="@0", thread_id=10)
        batch.entries.append(
            ToolBatchEntry(tool_use_id="t1", tool_use_text="Read foo.py")
        )
        _active_batches[(1, 10)] = batch

        first = await _send_or_edit_batch(bot, 1, batch, 42, 10, 10)
        second = await _send_or_edit_batch(bot, 1, batch, 42, 10, 10)

        assert first is False
        assert second is True
        assert batch.telegram_msg_id == 9
        assert _active_batches[(1, 10)] is batch


class TestDedupConsecutiveEntries:
    def _entry(
        self,
        text: str = "📖 **Read** `foo.py`",
        result: str | None = None,
        name: str | None = None,
    ) -> ToolBatchEntry:
        return ToolBatchEntry(
            tool_use_id=None,
            tool_use_text=text,
            tool_result_text=result,
            tool_name=name,
        )

    def test_consecutive_identical_collapse_to_count(self) -> None:
        entries = [
            self._entry("📖 read: x.py", "12 lines"),
            self._entry("📖 read: x.py", "12 lines"),
            self._entry("📖 read: x.py", "12 lines"),
        ]
        lines = _format_mixed_batch_lines(entries)
        assert len(lines) == 1
        assert " ×3" in lines[0]
        # result text is intentionally not rendered in the new format.
        assert "12 lines" not in lines[0]

    def test_mixed_status_same_tool_use_text_not_merged(self) -> None:
        entries = [
            self._entry("📖 **Read** `x.py`", "12 lines"),
            self._entry("📖 **Read** `x.py`", "error: not found"),
        ]
        lines = _format_mixed_batch_lines(entries)
        assert len(lines) == 2
        assert all(" ×" not in line for line in lines)

    def test_non_consecutive_identical_not_merged(self) -> None:
        entries = [
            self._entry("📖 **Read** `x.py`", "12 lines"),
            self._entry("✏️ **Edit** `x.py`", "ok"),
            self._entry("📖 **Read** `x.py`", "12 lines"),
        ]
        lines = _format_mixed_batch_lines(entries)
        assert len(lines) == 3
        assert all(" ×" not in line for line in lines)

    def test_task_create_run_unaffected_by_dedup(self) -> None:
        entries = [
            self._entry("📋 taskcreate: T1", None, "TaskCreate"),
            self._entry("📋 taskcreate: T2", None, "TaskCreate"),
        ]
        lines = _format_mixed_batch_lines(entries)
        assert all(" ×" not in line for line in lines)


class TestOversizedEntryTruncation:
    def test_oversized_entry_truncated(self) -> None:

        batch = ToolBatch(window_id="@0", thread_id=0)
        big_text = "x" * (BATCH_MAX_LENGTH + 100)
        task = ContentTask(
            window_id="@0",
            parts=(big_text,),
            content_type="tool_use",
            tool_use_id="tu1",
            thread_id=0,
        )
        _add_tool_use_entry(task, batch, ephemeral=True)
        assert len(batch.entries) == 1
        assert len(batch.entries[0].tool_use_text) <= BATCH_MAX_LENGTH
        assert batch.entries[0].tool_use_text.endswith("…")
        assert batch.total_length <= BATCH_MAX_LENGTH
