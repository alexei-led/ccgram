import ast
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

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
    flush_if_active,
    has_active_batch,
    has_ephemeral_active_batch,
    process_tool_event,
)
from ccgram.telegram_draft import mark_draft_unavailable, reset_draft_state


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


class TestProcessToolEventSignature:
    def test_accepts_content_task_and_returns_optional(self) -> None:
        sig = inspect.signature(process_tool_event)
        params = list(sig.parameters.values())
        assert params[2].name == "task"
        assert params[2].annotation == "ContentTask"
        assert sig.return_annotation == "ContentTask | None"

    def test_flush_if_active_exists_and_accepts_content_task(self) -> None:
        sig = inspect.signature(flush_if_active)
        params = list(sig.parameters.values())
        assert params[2].name == "task"
        assert params[2].annotation == "ContentTask"


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
        assert batch.telegram_msg_id == 77

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
