from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import TelegramError

from ccgram.handlers.messaging_pipeline.message_task import ContentTask
from ccgram.handlers.messaging_pipeline.tool_batch import (
    BATCH_MAX_ENTRIES,
    BATCH_MAX_LENGTH,
    ToolBatch,
    ToolBatchEntry,
    _active_batches,
    _add_tool_use_entry,
    _handle_tool_result,
    flush_batch,
    process_tool_event,
    ToolEventOutcome,
)
from ccgram.telegram_client import FakeTelegramClient


def _make_tool_use(
    tool_use_id: str = "tu1",
    text: str = "Read src/foo.py",
    window_id: str = "@0",
    thread_id: int | None = 10,
) -> ContentTask:
    return ContentTask(
        window_id=window_id,
        parts=(text,),
        content_type="tool_use",
        tool_use_id=tool_use_id,
        thread_id=thread_id,
    )


def _make_tool_result(
    tool_use_id: str = "tu1",
    text: str = "42 lines",
    window_id: str = "@0",
    thread_id: int | None = 10,
) -> ContentTask:
    return ContentTask(
        window_id=window_id,
        parts=(text,),
        content_type="tool_result",
        tool_use_id=tool_use_id,
        thread_id=thread_id,
    )


@pytest.fixture(autouse=True)
def _clear():
    _active_batches.clear()
    yield
    _active_batches.clear()


@pytest.fixture
def ephemeral_env(monkeypatch):
    monkeypatch.setattr(
        "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
        lambda _wid: True,
    )
    monkeypatch.setattr(
        "ccgram.handlers.messaging_pipeline.tool_batch.get_batch_mode",
        lambda _wid: "ephemeral",
    )
    monkeypatch.setattr(
        "ccgram.handlers.messaging_pipeline.tool_batch._rate_limit_chat",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "ccgram.handlers.messaging_pipeline.tool_batch.thread_router",
        MagicMock(resolve_chat_id=MagicMock(return_value=42)),
    )
    monkeypatch.setattr(
        "ccgram.handlers.status.status_bubble.clear_status_message",
        AsyncMock(return_value=None),
    )


class TestFifoEviction:
    def test_entry_count_overflow_evicts_oldest(self) -> None:
        batch = ToolBatch(window_id="@0", thread_id=0)
        for i in range(BATCH_MAX_ENTRIES):
            task = _make_tool_use(tool_use_id=f"tu{i}", text=f"Tool {i}")
            _add_tool_use_entry(task, batch, ephemeral=True)
        assert len(batch.entries) == BATCH_MAX_ENTRIES

        extra = _make_tool_use(tool_use_id="tu_new", text="New tool")
        overflow = _add_tool_use_entry(extra, batch, ephemeral=True)

        assert overflow is False
        assert len(batch.entries) == BATCH_MAX_ENTRIES
        assert batch.entries[-1].tool_use_id == "tu_new"
        assert all(e.tool_use_id != "tu0" for e in batch.entries)

    def test_char_budget_overflow_evicts_oldest(self) -> None:
        batch = ToolBatch(window_id="@0", thread_id=0)
        long_text = "x" * 400
        for i in range(6):
            task = _make_tool_use(tool_use_id=f"tu{i}", text=long_text)
            _add_tool_use_entry(task, batch, ephemeral=True)

        new_task = _make_tool_use(tool_use_id="tu_final", text=long_text)
        overflow = _add_tool_use_entry(new_task, batch, ephemeral=True)

        assert overflow is False
        assert batch.total_length <= BATCH_MAX_LENGTH
        assert batch.entries[-1].tool_use_id == "tu_final"

    def test_never_signals_overflow_in_ephemeral(self) -> None:
        batch = ToolBatch(window_id="@0", thread_id=0)
        for i in range(BATCH_MAX_ENTRIES + 5):
            task = _make_tool_use(tool_use_id=f"tu{i}", text="x" * 100)
            result = _add_tool_use_entry(task, batch, ephemeral=True)
            assert result is False

    def test_total_length_stays_consistent_after_eviction(self) -> None:
        batch = ToolBatch(window_id="@0", thread_id=0)
        for i in range(BATCH_MAX_ENTRIES):
            task = _make_tool_use(tool_use_id=f"tu{i}", text=f"Tool {i}")
            _add_tool_use_entry(task, batch, ephemeral=True)

        task = _make_tool_use(tool_use_id="tu_new", text="New tool")
        _add_tool_use_entry(task, batch, ephemeral=True)

        expected = sum(len(e.tool_use_text) for e in batch.entries)
        assert batch.total_length == expected


class TestOrphanToolResult:
    async def test_orphan_result_dropped_in_ephemeral(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: True,
        )
        client = FakeTelegramClient()
        batch = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[ToolBatchEntry(tool_use_id="tu1", tool_use_text="Read x")],
        )
        task = _make_tool_result(tool_use_id="tu_unknown", window_id="@0")

        updated_batch, followup = await _handle_tool_result(
            client, user_id=1, task=task, batch=batch, thread_id_or_0=0
        )

        assert followup is None
        assert updated_batch is None
        assert client.call_count("send_message") == 0
        assert client.call_count("edit_message_text") == 0

    async def test_orphan_result_flushes_in_batched_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: False,
        )
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.thread_router",
            MagicMock(resolve_chat_id=MagicMock(return_value=42)),
        )
        client = FakeTelegramClient()
        batch = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[ToolBatchEntry(tool_use_id="tu1", tool_use_text="Read x")],
        )
        _active_batches[(1, 0)] = batch
        task = _make_tool_result(tool_use_id="tu_unknown", window_id="@0")

        with patch(
            "ccgram.handlers.messaging_pipeline.tool_batch.flush_batch",
            new_callable=AsyncMock,
        ) as mock_flush:
            _updated_batch, followup = await _handle_tool_result(
                client, user_id=1, task=task, batch=batch, thread_id_or_0=0
            )

        assert followup is task
        mock_flush.assert_awaited_once()


class TestFlushBatchEphemeral:
    @patch("ccgram.handlers.messaging_pipeline.tool_batch.thread_router")
    async def test_flush_calls_delete_message_on_tracked_id(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42

        client = FakeTelegramClient()

        with patch(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            return_value=True,
        ):
            _active_batches[(1, 10)] = ToolBatch(
                window_id="@0",
                thread_id=10,
                entries=[ToolBatchEntry("t1", "Read x", "ok")],
                telegram_msg_id=77,
            )
            await flush_batch(client, user_id=1, thread_id_or_0=10)

        assert client.call_count("delete_message") == 1
        last = client.last_call("delete_message")
        assert last is not None
        assert last.kwargs["message_id"] == 77
        assert client.call_count("edit_message_text") == 0
        assert (1, 10) not in _active_batches

    @patch("ccgram.handlers.messaging_pipeline.tool_batch.thread_router")
    async def test_flush_noop_when_no_telegram_msg_id(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        client = FakeTelegramClient()

        with patch(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            return_value=True,
        ):
            _active_batches[(1, 10)] = ToolBatch(
                window_id="@0",
                thread_id=10,
                entries=[ToolBatchEntry("t1", "Read x")],
                telegram_msg_id=None,
            )
            await flush_batch(client, user_id=1, thread_id_or_0=10)

        assert client.call_count("delete_message") == 0
        assert (1, 10) not in _active_batches

    @patch("ccgram.handlers.messaging_pipeline.tool_batch.thread_router")
    async def test_flush_batched_mode_does_not_delete(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        client = FakeTelegramClient()

        with patch(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            return_value=False,
        ):
            _active_batches[(1, 10)] = ToolBatch(
                window_id="@0",
                thread_id=10,
                entries=[ToolBatchEntry("t1", "Read x", "ok")],
                telegram_msg_id=99,
            )
            await flush_batch(client, user_id=1, thread_id_or_0=10)

        assert client.call_count("delete_message") == 0
        assert client.call_count("edit_message_text") == 1

    @patch("ccgram.handlers.messaging_pipeline.tool_batch.thread_router")
    async def test_delete_telegram_error_swallowed(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        client = FakeTelegramClient()
        client.set_side_effect("delete_message", [TelegramError("gone")])

        with patch(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            return_value=True,
        ):
            _active_batches[(1, 10)] = ToolBatch(
                window_id="@0",
                thread_id=10,
                entries=[ToolBatchEntry("t1", "Read x", "ok")],
                telegram_msg_id=77,
            )
            await flush_batch(client, user_id=1, thread_id_or_0=10)

        assert (1, 10) not in _active_batches


class TestEphemeralDeliveryNoDraft:
    async def test_first_entry_uses_send_message(self, ephemeral_env) -> None:
        client = FakeTelegramClient()
        sent_msg = MagicMock()
        sent_msg.message_id = 55

        async def fake_safe_send(*_args, **_kwargs):
            return sent_msg

        with patch(
            "ccgram.handlers.messaging_pipeline.message_sender.safe_send",
            side_effect=fake_safe_send,
        ):
            result = await process_tool_event(client, 1, _make_tool_use())

        batch = _active_batches.get((1, 10))
        assert result.outcome is ToolEventOutcome.DELIVERED
        assert batch is not None
        assert batch.draft is None
        assert batch.telegram_msg_id == 55
        assert client.call_count("send_message") == 0

    async def test_initial_send_failure_is_not_acknowledged(
        self, ephemeral_env
    ) -> None:
        client = FakeTelegramClient()

        with patch(
            "ccgram.handlers.messaging_pipeline.message_sender.safe_send",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await process_tool_event(client, 1, _make_tool_use())

        assert result.outcome is ToolEventOutcome.FAILED
        assert _active_batches[(1, 10)].telegram_msg_id is None

    async def test_edit_failure_is_not_acknowledged(self, ephemeral_env) -> None:
        client = FakeTelegramClient()
        sent_msg = MagicMock(message_id=55)

        with (
            patch(
                "ccgram.handlers.messaging_pipeline.message_sender.safe_send",
                new_callable=AsyncMock,
                return_value=sent_msg,
            ),
            patch(
                "ccgram.handlers.messaging_pipeline.message_sender.edit_with_fallback",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await process_tool_event(client, 1, _make_tool_use(tool_use_id="tu1"))
            result = await process_tool_event(
                client, 1, _make_tool_use(tool_use_id="tu2", text="Edit b.py")
            )

        assert result.outcome is ToolEventOutcome.FAILED

    async def test_no_draft_api_used(
        self, ephemeral_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeTelegramClient()

        with (
            patch(
                "ccgram.handlers.messaging_pipeline.message_sender.safe_send",
                new_callable=AsyncMock,
                return_value=MagicMock(message_id=55),
            ),
            patch(
                "ccgram.handlers.messaging_pipeline.message_sender.edit_with_fallback",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            await process_tool_event(client, 1, _make_tool_use(tool_use_id="tu1"))
            await process_tool_event(client, 1, _make_tool_use(tool_use_id="tu2"))

        batch = _active_batches.get((1, 10))
        assert batch is not None
        assert batch.draft is None
        assert client.call_count("send_message") == 0
        assert client.call_count("edit_message_text") == 0
