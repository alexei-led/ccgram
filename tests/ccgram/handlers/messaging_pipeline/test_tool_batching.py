import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import RetryAfter

from ccgram.handlers.messaging_pipeline.message_queue import (
    _dispatch,
    _handle_content_task,
    get_or_create_queue,
    shutdown_workers,
)
from ccgram.handlers.messaging_pipeline.message_task import (
    ContentTask,
    MessageTask,
    StatusUpdateTask,
)
from ccgram.handlers.messaging_pipeline.tool_batch import (
    BATCH_MAX_ENTRIES,
    BATCH_MAX_LENGTH,
    ToolBatch,
    ToolBatchEntry,
    _active_batches,
    clear_batch_for_topic,
    flush_batch,
    format_batch_message,
    is_batch_eligible,
    process_tool_event,
)
from ccgram.session import (
    SessionManager,
    WindowState,
    window_store,
)
from ccgram.window_state_store import BATCH_MODES, DEFAULT_BATCH_MODE
from ccgram.telegram_draft import mark_draft_unavailable, reset_draft_state


@pytest.fixture
def batch_env():
    """Force DraftStream into legacy mode and patch DraftStream collaborators.

    Returns ``(bot, mock_send, mock_clear)`` where ``mock_send`` is the
    bot.send_message AsyncMock that DraftStream.start() drives in legacy mode.
    """
    reset_draft_state()
    mark_draft_unavailable("test")
    try:
        with (
            patch(
                "ccgram.handlers.status.status_bubble.clear_status_message",
                new_callable=AsyncMock,
            ) as mock_clear,
            patch(
                "ccgram.handlers.messaging_pipeline.tool_batch.get_batch_mode",
                return_value="batched",
            ),
            patch(
                "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.messaging_pipeline.tool_batch._rate_limit_chat",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "ccgram.handlers.messaging_pipeline.tool_batch.thread_router"
            ) as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = 42
            bot = AsyncMock()
            sent_msg = MagicMock()
            sent_msg.message_id = 100
            bot.send_message.return_value = sent_msg
            yield bot, bot.send_message, mock_clear
    finally:
        reset_draft_state()
        _active_batches.clear()


class TestFormatBatchMessage:
    def test_single_entry(self) -> None:
        entries = [ToolBatchEntry(tool_use_id="t1", tool_use_text="Read src/foo.py")]
        result = format_batch_message(entries)
        # No code-block fence \u2014 plain text with inline mono on summaries only.
        assert not result.startswith("```")
        assert "Read src/foo.py" in result
        # No status glyphs, no count header.
        assert "\u23f3" not in result
        assert "\u2705" not in result
        assert "\u274c" not in result
        assert "tool call" not in result

    def test_result_text_is_dropped(self) -> None:
        entries = [
            ToolBatchEntry(
                tool_use_id="t1",
                tool_use_text="Read src/foo.py",
                tool_result_text="42 lines",
            )
        ]
        result = format_batch_message(entries)
        assert "Read src/foo.py" in result
        # tool_result_text is intentionally not rendered.
        assert "42 lines" not in result
        assert "\u23bf" not in result

    def test_multiple_entries_one_per_line(self) -> None:
        entries = [
            ToolBatchEntry("t1", "Read src/a.py", "10 lines"),
            ToolBatchEntry("t2", "Edit src/a.py", "+3 -1"),
            ToolBatchEntry("t3", "Bash make test"),
        ]
        result = format_batch_message(entries)
        assert "```" not in result
        assert result.split("\n") == [
            "Read src/a.py",
            "Edit src/a.py",
            "Bash make test",
        ]

    def test_subagent_label_renders_as_first_body_line(self) -> None:
        entries = [ToolBatchEntry("t1", "Read x"), ToolBatchEntry("t2", "Edit y")]
        result = format_batch_message(entries, subagent_label="\U0001f916 write-tests")
        assert "```" not in result
        assert result.split("\n", 1)[0] == "\U0001f916 write-tests"

    def test_task_create_batch_renders_numbered_list(self) -> None:
        entries = [
            ToolBatchEntry(
                "t1",
                "📋 taskcreate: Understand the Problem Domain",
                tool_name="TaskCreate",
            ),
            ToolBatchEntry(
                "t2",
                "📋 taskcreate: Map Integrations",
                tool_name="TaskCreate",
            ),
            ToolBatchEntry(
                "t3",
                "📋 taskcreate: Apply the Balance Rule",
                tool_name="TaskCreate",
            ),
        ]

        result = format_batch_message(entries, subagent_label="\U0001f916 subagent")

        assert result.split("\n")[0] == "\U0001f916 subagent"
        assert "Creating 3 tasks\u2026" in result
        assert "1. Understand the Problem Domain" in result
        assert "2. Map Integrations" in result
        assert "3. Apply the Balance Rule" in result
        assert "tool calls" not in result

    def test_task_create_batch_renders_completed_header_when_results_arrive(
        self,
    ) -> None:
        entries = [
            ToolBatchEntry(
                "t1",
                "📋 taskcreate: Write the Review",
                tool_name="TaskCreate",
                tool_result_text="Done",
            )
        ]

        result = format_batch_message(entries)

        assert result.startswith("Created 1 task\n")
        assert "1. Write the Review" in result
        assert "\u23bf" not in result

    def test_task_create_batch_falls_back_when_tool_name_missing(self) -> None:
        entries = [ToolBatchEntry("t1", "TaskCreate Understand the Problem Domain")]

        result = format_batch_message(entries)
        assert "```" not in result
        assert result == "TaskCreate Understand the Problem Domain"

    def test_mixed_batch_groups_task_create_entries(self) -> None:
        entries = [
            ToolBatchEntry(
                "t0", "🔧 toolsearch: select:TaskCreate,TaskUpdate,TaskList"
            ),
            ToolBatchEntry(
                "t1",
                "📋 taskcreate: Tune regex linter",
                tool_name="TaskCreate",
            ),
            ToolBatchEntry(
                "t2",
                "**TaskCreate**Apply fixes to opus agents",
                tool_name="TaskCreate",
            ),
        ]

        result = format_batch_message(entries, subagent_label="\U0001f916 subagent")
        assert "```" not in result
        lines = result.split("\n")
        assert lines[0] == "\U0001f916 subagent"
        assert "toolsearch" in lines[1].lower()
        assert "Creating 2 tasks\u2026" in result
        assert "1. Tune regex linter" in result
        assert "2. Apply fixes to opus agents" in result

    def test_task_update_entries_render_as_progress_section(self) -> None:
        entries = [
            ToolBatchEntry(
                "t1",
                "📋 taskupdate: Tune regex linter -> in progress",
                tool_name="TaskUpdate",
            ),
            ToolBatchEntry(
                "t2",
                "📋 taskupdate: Apply fixes to opus agents -> completed",
                tool_name="TaskUpdate",
                tool_result_text="Done",
            ),
        ]

        result = format_batch_message(entries)

        assert "Updating 2 tasks\u2026" in result
        assert "- Tune regex linter -> in progress" in result
        assert "- Apply fixes to opus agents -> completed" in result

    def test_task_list_entry_renders_as_task_list_sync(self) -> None:
        entries = [
            ToolBatchEntry(
                "t1",
                "📋 tasklist: refresh",
                tool_name="TaskList",
            )
        ]

        result = format_batch_message(entries)

        assert "```" not in result
        assert result == "Refreshing task list\u2026"


class TestIsBatchEligible:
    @pytest.mark.parametrize("content_type", ["tool_use", "tool_result"])
    @patch(
        "ccgram.handlers.messaging_pipeline.tool_batch.get_batch_mode",
        return_value="batched",
    )
    def test_tool_types_eligible(self, _mock_gbm, content_type: str) -> None:
        task = ContentTask(window_id="@0", parts=("x",), content_type=content_type)  # type: ignore[arg-type]
        assert is_batch_eligible(task) is True

    @pytest.mark.parametrize("content_type", ["text", "thinking", "assistant"])
    @patch(
        "ccgram.handlers.messaging_pipeline.tool_batch.get_batch_mode",
        return_value="batched",
    )
    def test_non_tool_types_not_eligible(self, _mock_gbm, content_type: str) -> None:
        task = ContentTask(window_id="@0", parts=("x",))
        assert is_batch_eligible(task) is False


class TestBatchDataStructures:
    def test_tool_batch_entry_defaults(self) -> None:
        entry = ToolBatchEntry(tool_use_id="t1", tool_use_text="Read x")
        assert entry.tool_result_text is None

    def test_tool_batch_defaults(self) -> None:
        batch = ToolBatch(window_id="@0", thread_id=0)
        assert batch.entries == []
        assert batch.telegram_msg_id is None
        assert batch.total_length == 0

    def test_batch_entry_accumulation(self) -> None:
        batch = ToolBatch(window_id="@0", thread_id=0)
        for i in range(5):
            entry = ToolBatchEntry(f"t{i}", f"Read file{i}.py")
            batch.entries.append(entry)
            batch.total_length += len(entry.tool_use_text)
        assert len(batch.entries) == 5
        assert batch.total_length == sum(len(f"Read file{i}.py") for i in range(5))

    def test_constants(self) -> None:
        assert BATCH_MAX_ENTRIES == 9
        assert BATCH_MAX_LENGTH == 2800


class TestWindowStateBatchMode:
    def test_default_batch_mode(self) -> None:
        ws = WindowState()
        assert ws.batch_mode == DEFAULT_BATCH_MODE
        assert ws.batch_mode == "ephemeral"

    @pytest.mark.parametrize(
        ("mode", "expect_key"),
        [("ephemeral", False), ("batched", True), ("verbose", True)],
    )
    def test_to_dict_batch_mode(self, mode: str, expect_key: bool) -> None:
        ws = WindowState(session_id="s1", cwd="/tmp", batch_mode=mode)
        d = ws.to_dict()
        if expect_key:
            assert d["batch_mode"] == mode
        else:
            assert "batch_mode" not in d

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ({"session_id": "s1", "cwd": "/tmp"}, "ephemeral"),
            ({"session_id": "s1", "cwd": "/tmp", "batch_mode": "verbose"}, "verbose"),
            ({"session_id": "s1", "cwd": "/tmp", "batch_mode": "batched"}, "batched"),
        ],
    )
    def test_from_dict(self, data: dict[str, str], expected: str) -> None:
        assert WindowState.from_dict(data).batch_mode == expected

    @pytest.mark.parametrize("mode", sorted(BATCH_MODES))
    def test_roundtrip(self, mode: str) -> None:
        ws = WindowState(session_id="s1", cwd="/tmp", batch_mode=mode)
        assert WindowState.from_dict(ws.to_dict()).batch_mode == mode


@pytest.fixture
def mgr(monkeypatch) -> SessionManager:
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    return SessionManager()


class TestSessionManagerBatchMode:
    def test_get_default(self, mgr: SessionManager) -> None:
        assert mgr.get_batch_mode("@0") == "ephemeral"

    def test_get_nonexistent_window(self, mgr: SessionManager) -> None:
        assert mgr.get_batch_mode("@999") == "ephemeral"

    def test_set_mode(self, mgr: SessionManager) -> None:
        mgr.set_batch_mode("@0", "verbose")
        assert mgr.get_batch_mode("@0") == "verbose"

    def test_set_mode_validates(self, mgr: SessionManager) -> None:
        with pytest.raises(ValueError, match="Invalid batch mode"):
            mgr.set_batch_mode("@0", "invalid")

    @pytest.mark.parametrize(
        ("start", "expected"),
        [
            ("batched", "ephemeral"),
            ("ephemeral", "verbose"),
            ("verbose", "batched"),
        ],
    )
    def test_cycle(self, mgr: SessionManager, start: str, expected: str) -> None:
        mgr.set_batch_mode("@0", start)
        assert mgr.cycle_batch_mode("@0") == expected
        assert mgr.get_batch_mode("@0") == expected

    def test_cycle_full_circle(self, mgr: SessionManager) -> None:
        # Default is "ephemeral"; cycle goes ephemeral -> verbose -> batched -> ephemeral.
        mgr.cycle_batch_mode("@0")
        assert mgr.get_batch_mode("@0") == "verbose"
        mgr.cycle_batch_mode("@0")
        assert mgr.get_batch_mode("@0") == "batched"
        mgr.cycle_batch_mode("@0")
        assert mgr.get_batch_mode("@0") == "ephemeral"

    def test_set_same_mode_no_save(self, mgr: SessionManager, monkeypatch) -> None:
        mgr.set_batch_mode("@0", "verbose")
        save_calls = []
        monkeypatch.setattr(
            SessionManager, "_save_state", lambda self: save_calls.append(1)
        )
        mgr.set_batch_mode("@0", "verbose")  # same mode
        assert len(save_calls) == 0

    def test_get_invalid_stored_mode_returns_default(self, mgr: SessionManager) -> None:
        state = window_store.get_window_state("@0")
        state.batch_mode = "garbage"
        assert mgr.get_batch_mode("@0") == "ephemeral"


@pytest.fixture(autouse=True)
def _clear_batches():
    _active_batches.clear()
    yield
    _active_batches.clear()


def _make_tool_use(
    window_id: str = "@0",
    tool_use_id: str = "tu1",
    text: str = "Read src/foo.py",
    tool_name: str | None = None,
    thread_id: int | None = 10,
) -> ContentTask:
    return ContentTask(
        window_id=window_id,
        parts=(text,),
        content_type="tool_use",
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        thread_id=thread_id,
    )


def _make_tool_result(
    tool_use_id: str | None = "tu1",
    text: str = "42 lines",
    thread_id: int | None = 10,
    window_id: str = "@0",
) -> ContentTask:
    return ContentTask(
        window_id=window_id,
        parts=(text,),
        content_type="tool_result",
        tool_use_id=tool_use_id,
        thread_id=thread_id,
    )


class TestProcessBatchTask:
    async def test_tool_use_creates_batch(self, batch_env) -> None:
        bot, mock_send, _ = batch_env
        await process_tool_event(bot, 1, _make_tool_use())

        bkey = (1, 10)
        assert bkey in _active_batches
        batch = _active_batches[bkey]
        assert len(batch.entries) == 1
        assert batch.entries[0].tool_use_id == "tu1"
        assert batch.telegram_msg_id == 100

    async def test_task_create_batch_sends_task_list(self, batch_env) -> None:
        bot, mock_send, _ = batch_env
        await process_tool_event(
            bot,
            1,
            _make_tool_use(
                text="📋 taskcreate: Understand the Problem Domain",
                tool_name="TaskCreate",
            ),
        )
        # DraftStream.start (legacy) drives bot.send_message with kwargs.
        sent_text = mock_send.await_args.kwargs["text"]

        assert sent_text.startswith("Creating 1 task\u2026\n")
        assert "1. Understand the Problem Domain" in sent_text
        assert "tool call" not in sent_text

    async def test_tool_result_updates_entry(self, batch_env) -> None:
        bot, _, _ = batch_env
        await process_tool_event(bot, 1, _make_tool_use())
        await process_tool_event(bot, 1, _make_tool_result())

        batch = _active_batches[(1, 10)]
        assert batch.entries[0].tool_result_text == "42 lines"

    async def test_multiple_tool_calls_accumulate(self, batch_env) -> None:
        bot, _, _ = batch_env
        await process_tool_event(
            bot, 1, _make_tool_use(tool_use_id="tu1", text="Read a.py")
        )
        await process_tool_event(
            bot, 1, _make_tool_result(tool_use_id="tu1", text="10 lines")
        )
        await process_tool_event(
            bot, 1, _make_tool_use(tool_use_id="tu2", text="Edit a.py")
        )
        await process_tool_event(
            bot, 1, _make_tool_use(tool_use_id="tu3", text="Bash make test")
        )

        batch = _active_batches[(1, 10)]
        assert len(batch.entries) == 3
        assert batch.entries[0].tool_use_id == "tu1"
        assert batch.entries[0].tool_result_text == "10 lines"
        assert batch.entries[1].tool_use_id == "tu2"
        assert batch.entries[1].tool_result_text is None
        assert batch.entries[2].tool_use_id == "tu3"

    async def test_tool_result_truncates_long_text(self, batch_env) -> None:
        bot, _, _ = batch_env
        await process_tool_event(bot, 1, _make_tool_use())
        long_result = "x" * 200 + "\nsecond line"
        await process_tool_event(bot, 1, _make_tool_result(text=long_result))

        batch = _active_batches[(1, 10)]
        result_text = batch.entries[0].tool_result_text
        assert result_text is not None
        assert len(result_text) <= 200
        assert "\n" not in result_text

    async def test_tool_result_no_matching_entry_flushes(self, batch_env) -> None:
        bot, _, _ = batch_env
        with patch(
            "ccgram.handlers.messaging_pipeline.tool_batch.flush_batch",
            new_callable=AsyncMock,
        ) as mock_flush:
            await process_tool_event(bot, 1, _make_tool_use(tool_use_id="tu1"))
            followup = await process_tool_event(
                bot, 1, _make_tool_result(tool_use_id="tu_unknown")
            )
        mock_flush.assert_awaited_once()
        assert followup is not None

    async def test_different_window_flushes_old_batch(self, batch_env) -> None:
        bot, _, _ = batch_env
        with patch(
            "ccgram.handlers.messaging_pipeline.tool_batch.flush_batch",
            new_callable=AsyncMock,
        ) as mock_flush:
            await process_tool_event(bot, 1, _make_tool_use(window_id="@0"))
            await process_tool_event(
                bot, 1, _make_tool_use(window_id="@1", tool_use_id="tu2")
            )
        mock_flush.assert_awaited_once()

    async def test_batch_overflow_entries_splits(self, batch_env) -> None:
        bot, _, _ = batch_env
        for i in range(BATCH_MAX_ENTRIES + 2):
            await process_tool_event(
                bot, 1, _make_tool_use(tool_use_id=f"tu{i}", text=f"Tool {i}")
            )

        batch = _active_batches[(1, 10)]
        assert len(batch.entries) == 2
        assert batch.entries[0].tool_use_id == f"tu{BATCH_MAX_ENTRIES}"
        assert batch.entries[1].tool_use_id == f"tu{BATCH_MAX_ENTRIES + 1}"

    async def test_batch_clears_status_on_first_send(self, batch_env) -> None:
        bot, _, mock_clear = batch_env
        await process_tool_event(bot, 1, _make_tool_use())
        mock_clear.assert_awaited_once()

    async def test_second_tool_edits_existing_message(self, batch_env) -> None:
        bot, mock_send, _ = batch_env
        await process_tool_event(bot, 1, _make_tool_use(tool_use_id="tu1"))
        mock_send.assert_awaited_once()

        await process_tool_event(
            bot, 1, _make_tool_use(tool_use_id="tu2", text="Edit b.py")
        )
        bot.edit_message_text.assert_awaited()


class TestHandleContentTask:
    @patch(
        "ccgram.handlers.messaging_pipeline.tool_batch.get_batch_mode",
        return_value="batched",
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.process_tool_event",
        new_callable=AsyncMock,
    )
    async def test_batch_eligible_routes_to_batch(
        self, mock_batch, mock_should
    ) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = ContentTask(
            content_type="tool_use",
            window_id="@0",
            parts=("Read x",),
        )
        mock_batch.return_value = None
        with patch(
            "ccgram.handlers.messaging_pipeline.message_queue.is_tool_calls_hidden",
            return_value=False,
        ):
            extra = await _handle_content_task(bot, 1, task, queue, lock)
        assert extra == 0
        mock_batch.assert_awaited_once()

    @patch(
        "ccgram.handlers.messaging_pipeline.tool_batch.get_batch_mode",
        return_value="individual",
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    async def test_verbose_mode_skips_batch(self, mock_process, mock_should) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = ContentTask(
            content_type="tool_use",
            window_id="@0",
            parts=("Read x",),
        )
        with patch(
            "ccgram.handlers.messaging_pipeline.message_queue.is_tool_calls_hidden",
            return_value=False,
        ):
            extra = await _handle_content_task(bot, 1, task, queue, lock)
        assert extra == 0
        mock_process.assert_awaited_once()

    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.flush_if_active",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    async def test_text_flushes_active_batch(self, mock_process, mock_flush) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = ContentTask(
            content_type="text",
            window_id="@0",
            parts=("Hello",),
        )
        await _handle_content_task(bot, 1, task, queue, lock)
        mock_flush.assert_awaited_once_with(bot, 1, task)
        mock_process.assert_awaited_once()

    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.flush_if_active",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    async def test_thinking_flushes_active_batch(
        self, mock_process, mock_flush
    ) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = ContentTask(
            content_type="text",
            window_id="@0",
            parts=("Thinking...",),
            thread_id=5,
        )
        await _handle_content_task(bot, 1, task, queue, lock)
        mock_flush.assert_awaited_once_with(bot, 1, task)

    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._process_content_task",
        new_callable=AsyncMock,
    )
    async def test_no_batch_no_flush(self, mock_process) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = ContentTask(
            content_type="text",
            window_id="@0",
            parts=("Hello",),
        )
        await _handle_content_task(bot, 1, task, queue, lock)
        mock_process.assert_awaited_once()


class TestFlushBatch:
    @pytest.fixture(autouse=True)
    def _force_batched_mode(self, monkeypatch):
        # The new ephemeral default would route flush_batch to delete_message;
        # these tests exercise the batched-mode edit/send path.
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: False,
        )

    @patch("ccgram.handlers.messaging_pipeline.tool_batch.thread_router")
    async def test_flush_removes_batch(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 10)] = ToolBatch(
            window_id="@0",
            thread_id=10,
            entries=[ToolBatchEntry("t1", "Read x", "ok")],
            telegram_msg_id=100,
        )

        bot = AsyncMock()
        await flush_batch(bot, 1, 10)
        assert (1, 10) not in _active_batches

    async def test_flush_noop_when_no_batch(self) -> None:
        bot = AsyncMock()
        await flush_batch(bot, 1, 10)  # should not raise

    @patch("ccgram.handlers.messaging_pipeline.tool_batch.thread_router")
    async def test_flush_edits_final_message(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 0)] = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[
                ToolBatchEntry("t1", "Read a.py", "10 lines"),
                ToolBatchEntry("t2", "Edit a.py", "+1 -1"),
            ],
            telegram_msg_id=200,
        )

        bot = AsyncMock()
        await flush_batch(bot, 1, 0)
        bot.edit_message_text.assert_awaited()

    @patch("ccgram.handlers.messaging_pipeline.tool_batch.thread_router")
    async def test_flush_no_edit_without_telegram_msg_id(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 0)] = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[ToolBatchEntry("t1", "Read x")],
            telegram_msg_id=None,
        )

        bot = AsyncMock()
        await flush_batch(bot, 1, 0)
        bot.edit_message_text.assert_not_awaited()
        assert (1, 0) not in _active_batches

    async def test_flush_empty_entries_noop(self) -> None:
        _active_batches[(1, 0)] = ToolBatch(window_id="@0", thread_id=0, entries=[])
        bot = AsyncMock()
        await flush_batch(bot, 1, 0)
        assert (1, 0) not in _active_batches
        bot.edit_message_text.assert_not_awaited()

    @patch("ccgram.claude_task_state.get_subagent_names", return_value=["researcher"])
    @patch("ccgram.handlers.messaging_pipeline.tool_batch.thread_router")
    async def test_flush_includes_subagent_label(self, mock_tr, _mock_names) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 0)] = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[ToolBatchEntry("t1", "Read x", "ok")],
            telegram_msg_id=100,
        )

        bot = AsyncMock()
        await flush_batch(bot, 1, 0)
        text_sent = bot.edit_message_text.call_args.kwargs["text"]
        assert "researcher" in text_sent

    @patch("ccgram.handlers.messaging_pipeline.tool_batch.thread_router")
    async def test_flush_handles_telegram_error(self, mock_tr) -> None:
        from telegram.error import TelegramError

        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 0)] = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[ToolBatchEntry("t1", "Read x", "ok")],
            telegram_msg_id=100,
        )

        bot = AsyncMock()
        bot.edit_message_text.side_effect = TelegramError("bad markup")
        await flush_batch(bot, 1, 0)
        assert (1, 0) not in _active_batches


class TestBatchIsolation:
    async def test_different_threads_separate_batches(self, batch_env) -> None:
        bot, _, _ = batch_env
        await process_tool_event(
            bot, 1, _make_tool_use(thread_id=10, tool_use_id="tu1")
        )
        await process_tool_event(
            bot, 1, _make_tool_use(thread_id=20, tool_use_id="tu2")
        )

        assert (1, 10) in _active_batches
        assert (1, 20) in _active_batches
        assert len(_active_batches[(1, 10)].entries) == 1
        assert len(_active_batches[(1, 20)].entries) == 1


class TestShutdownClearsBatches:
    async def test_shutdown_clears_active_batches(self) -> None:
        await shutdown_workers()  # ensure clean state from previous tests
        _active_batches[(1, 0)] = ToolBatch(window_id="@0", thread_id=0)
        _active_batches[(2, 5)] = ToolBatch(window_id="@1", thread_id=5)
        await shutdown_workers()
        assert len(_active_batches) == 0


class TestQueueWorkerRetryAfter:
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue.asyncio.sleep",
        new_callable=AsyncMock,
    )
    @patch(
        "ccgram.handlers.messaging_pipeline.message_queue._handle_content_task",
        new_callable=AsyncMock,
    )
    async def test_retry_after_retries_same_task(self, mock_handle, mock_sleep) -> None:
        await shutdown_workers()
        mock_handle.side_effect = [RetryAfter(1), 0]

        bot = AsyncMock()
        queue = get_or_create_queue(bot, 1)
        queue.put_nowait(
            ContentTask(
                window_id="@0",
                parts=("hello",),
                content_type="text",
                thread_id=10,
            )
        )

        try:
            await asyncio.wait_for(queue.join(), timeout=1)
            assert mock_handle.await_count == 2
            mock_sleep.assert_awaited_once()
        finally:
            await shutdown_workers()


class TestToolResultNotDropped:
    async def test_tool_result_no_active_batch_falls_through(self, batch_env) -> None:
        bot, _, _ = batch_env
        task = _make_tool_result(tool_use_id="tu1", text="result text")
        result = await process_tool_event(bot, 1, task)
        assert result == task

    async def test_tool_result_none_tool_use_id_falls_through(self, batch_env) -> None:
        bot, _, _ = batch_env
        await process_tool_event(bot, 1, _make_tool_use(tool_use_id="tu1"))
        task = _make_tool_result(tool_use_id=None, text="result text")
        result = await process_tool_event(bot, 1, task)
        assert result == task
        assert (1, 10) in _active_batches
        assert len(_active_batches[(1, 10)].entries) == 1


class TestBatchLengthOverflow:
    async def test_overflow_on_length(self, batch_env) -> None:
        bot, _, _ = batch_env
        long_text = "x" * 500
        for i in range(8):
            await process_tool_event(
                bot, 1, _make_tool_use(tool_use_id=f"tu{i}", text=long_text)
            )

        batch = _active_batches[(1, 10)]
        assert batch.total_length <= BATCH_MAX_LENGTH
        assert len(batch.entries) < 8


class TestTopicCleanupClearsBatch:
    def test_clear_batch_for_topic(self) -> None:
        _active_batches[(1, 10)] = ToolBatch(window_id="@0", thread_id=10)
        clear_batch_for_topic(1, 10)
        assert (1, 10) not in _active_batches

    def test_clear_batch_for_topic_noop(self) -> None:
        clear_batch_for_topic(1, 999)  # should not raise

    def test_clear_batch_none_thread(self) -> None:
        _active_batches[(1, 0)] = ToolBatch(window_id="@0", thread_id=0)
        clear_batch_for_topic(1, None)
        assert (1, 0) not in _active_batches


class TestFlushSendFallback:
    @pytest.fixture(autouse=True)
    def _force_batched_mode(self, monkeypatch):
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: False,
        )

    @patch(
        "ccgram.handlers.messaging_pipeline.tool_batch._rate_limit_chat",
        new=AsyncMock(return_value=None),
    )
    @patch("ccgram.handlers.messaging_pipeline.tool_batch.thread_router")
    async def test_flush_sends_when_no_telegram_msg_id(self, mock_tr) -> None:
        # No prior message and no draft → flush_batch opens a fresh
        # DraftStream, which uses bot.send_message in legacy mode.
        reset_draft_state()
        mark_draft_unavailable("test")
        try:
            mock_tr.resolve_chat_id.return_value = 42
            _active_batches[(1, 0)] = ToolBatch(
                window_id="@0",
                thread_id=0,
                entries=[ToolBatchEntry("t1", "Read x", "ok")],
                telegram_msg_id=None,
            )

            bot = AsyncMock()
            sent = MagicMock()
            sent.message_id = 100
            bot.send_message.return_value = sent
            await flush_batch(bot, 1, 0)
            bot.send_message.assert_awaited_once()
            assert "Read x" in bot.send_message.call_args.kwargs["text"]
            assert (1, 0) not in _active_batches
        finally:
            reset_draft_state()


class TestDefensiveElseBranch:
    @patch("ccgram.handlers.messaging_pipeline.tool_batch.thread_router")
    async def test_unexpected_content_type_routes_to_normal(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        bot = AsyncMock()
        task = ContentTask(
            window_id="@0",
            parts=("hello",),
            content_type="text",
            thread_id=10,
        )
        result = await process_tool_event(bot, 1, task)
        assert result == task


class TestDifferentUsersIsolation:
    async def test_different_users_same_thread_separate_batches(
        self, batch_env
    ) -> None:
        bot, _, _ = batch_env
        await process_tool_event(
            bot, 1, _make_tool_use(thread_id=10, tool_use_id="tu1")
        )
        await process_tool_event(
            bot, 2, _make_tool_use(thread_id=10, tool_use_id="tu2")
        )

        assert (1, 10) in _active_batches
        assert (2, 10) in _active_batches
        assert _active_batches[(1, 10)].entries[0].tool_use_id == "tu1"
        assert _active_batches[(2, 10)].entries[0].tool_use_id == "tu2"


class TestBatchResultTruncation:
    def test_result_truncated_to_200_chars(self):
        long_text = "x" * 300
        entry = ToolBatchEntry("t1", "Bash cmd")
        entry.tool_result_text = long_text.split("\n", 1)[0][:200]
        assert len(entry.tool_result_text) == 200

    def test_multiline_result_uses_first_line(self):
        result_text = "line one\nline two\nline three"
        first_line = result_text.split("\n", 1)[0][:200]
        assert first_line == "line one"


class TestDispatcherSuppressesStatusOnEphemeralBatch:
    """Status updates must NOT flush an active ephemeral tool batch.

    Repro for the visible-flicker bug: while tool calls are streaming, a
    1s status poll would delete the formatted tool bubble and insert a
    status bubble (with no-markdown text, so it looks plain) in its
    position — appearing as the tool bubble "flipping to plain text".
    The dispatcher must drop the StatusUpdateTask in this case.
    """

    def setup_method(self) -> None:
        _active_batches.clear()

    def teardown_method(self) -> None:
        _active_batches.clear()

    async def test_status_update_dropped_when_ephemeral_batch_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ephemeral batch owns the bubble for (user=1, thread=10).
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: True,
        )
        _active_batches[(1, 10)] = ToolBatch(window_id="@0", thread_id=10)

        flushed = AsyncMock()
        processed = AsyncMock()
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.message_queue.flush_batch",
            flushed,
        )
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.message_queue.process_status_update",
            processed,
        )

        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = StatusUpdateTask(window_id="@0", text="working", thread_id=10)

        result = await _dispatch(bot, 1, task, queue, lock)

        assert result == 0
        flushed.assert_not_called()
        processed.assert_not_called()
        # Batch still on screen.
        assert (1, 10) in _active_batches

    async def test_status_update_processed_when_no_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: True,
        )
        processed = AsyncMock()
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.message_queue.process_status_update",
            processed,
        )

        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = StatusUpdateTask(window_id="@0", text="idle", thread_id=10)

        await _dispatch(bot, 1, task, queue, lock)

        processed.assert_awaited_once()

    async def test_status_update_processed_when_batch_non_ephemeral(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Non-ephemeral (batched/verbose) batches don't get suppression —
        # those modes leave the batch on screen after flush instead of
        # deleting it, so the flicker doesn't happen.
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.tool_batch.is_ephemeral_tools",
            lambda _wid: False,
        )
        _active_batches[(1, 10)] = ToolBatch(window_id="@0", thread_id=10)
        flushed = AsyncMock()
        processed = AsyncMock()
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.message_queue.flush_batch",
            flushed,
        )
        monkeypatch.setattr(
            "ccgram.handlers.messaging_pipeline.message_queue.process_status_update",
            processed,
        )

        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = StatusUpdateTask(window_id="@0", text="working", thread_id=10)

        await _dispatch(bot, 1, task, queue, lock)

        flushed.assert_awaited_once()
        processed.assert_awaited_once()
