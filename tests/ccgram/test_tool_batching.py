import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import RetryAfter

from ccgram.handlers.message_queue import (
    BATCH_MAX_ENTRIES,
    BATCH_MAX_LENGTH,
    MessageTask,
    ToolBatch,
    ToolBatchEntry,
    _active_batches,
    _flush_batch,
    _handle_content_task,
    _is_batch_eligible,
    _process_batch_task,
    clear_batch_for_topic,
    format_batch_message,
    get_or_create_queue,
    shutdown_workers,
)
from ccgram.session import (
    BATCH_MODES,
    DEFAULT_BATCH_MODE,
    SessionManager,
    WindowState,
)


class TestFormatBatchMessage:
    def test_single_entry_pending(self) -> None:
        entries = [ToolBatchEntry(tool_use_id="t1", tool_use_text="Read src/foo.py")]
        result = format_batch_message(entries)
        assert result.startswith("\u26a1 1 tool call")
        assert "Read src/foo.py" in result
        assert "\u23f3" in result

    def test_single_entry_with_result(self) -> None:
        entries = [
            ToolBatchEntry(
                tool_use_id="t1",
                tool_use_text="Read src/foo.py",
                tool_result_text="42 lines",
            )
        ]
        result = format_batch_message(entries)
        assert "1 tool call" in result
        assert "42 lines" in result
        assert "\u23f3" not in result

    def test_multiple_entries(self) -> None:
        entries = [
            ToolBatchEntry("t1", "Read src/a.py", "10 lines"),
            ToolBatchEntry("t2", "Edit src/a.py", "+3 -1"),
            ToolBatchEntry("t3", "Bash make test"),
        ]
        result = format_batch_message(entries)
        assert "3 tool calls" in result
        assert "Read src/a.py" in result
        assert "Edit src/a.py" in result
        assert "Bash make test" in result
        lines = result.split("\n")
        assert "\u23f3" in lines[-1]
        assert "\u23f3" not in lines[1]

    def test_header_pluralization(self) -> None:
        single = format_batch_message([ToolBatchEntry("t1", "Read x")])
        assert "tool call\n" in single

        multi = format_batch_message(
            [ToolBatchEntry("t1", "Read x"), ToolBatchEntry("t2", "Edit y")]
        )
        assert "tool calls\n" in multi

    def test_result_separator(self) -> None:
        entries = [ToolBatchEntry("t1", "Read x", "ok")]
        result = format_batch_message(entries)
        assert "\u23bf" in result  # ⎿ separator between tool_use and result

    def test_all_entries_have_results(self) -> None:
        entries = [
            ToolBatchEntry("t1", "Read a.py", "10 lines"),
            ToolBatchEntry("t2", "Edit a.py", "+1 -1"),
            ToolBatchEntry("t3", "Bash make test", "PASS"),
        ]
        result = format_batch_message(entries)
        assert "\u23f3" not in result

    def test_empty_result_text(self) -> None:
        entries = [ToolBatchEntry("t1", "Read x", "")]
        result = format_batch_message(entries)
        assert "\u23bf" in result
        assert "\u23f3" not in result

    def test_subagent_label_none(self) -> None:
        entries = [ToolBatchEntry("t1", "Read x")]
        result = format_batch_message(entries, subagent_label=None)
        assert "[" not in result.split("\n")[0]

    def test_subagent_label_single(self) -> None:
        entries = [ToolBatchEntry("t1", "Read x"), ToolBatchEntry("t2", "Edit y")]
        result = format_batch_message(entries, subagent_label="\U0001f916 write-tests")
        header = result.split("\n")[0]
        assert "2 tool calls" in header
        assert "[\U0001f916 write-tests]" in header

    def test_subagent_label_multi(self) -> None:
        entries = [ToolBatchEntry("t1", "Read x")]
        label = "\U0001f916 2 subagents: write-tests, refactor"
        result = format_batch_message(entries, subagent_label=label)
        header = result.split("\n")[0]
        assert "[" in header
        assert "2 subagents" in header

    def test_task_create_batch_renders_numbered_list(self) -> None:
        entries = [
            ToolBatchEntry(
                "t1",
                "**TaskCreate** `Understand the Problem Domain`",
                tool_name="TaskCreate",
            ),
            ToolBatchEntry(
                "t2",
                "**TaskCreate** `Map Integrations`",
                tool_name="TaskCreate",
            ),
            ToolBatchEntry(
                "t3",
                "**TaskCreate** `Apply the Balance Rule`",
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
                "**TaskCreate** `Write the Review`",
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

        assert result.startswith("\u26a1 1 tool call")

    def test_mixed_batch_groups_task_create_entries(self) -> None:
        entries = [
            ToolBatchEntry(
                "t0", "**ToolSearch** `select:TaskCreate,TaskUpdate,TaskList`"
            ),
            ToolBatchEntry(
                "t1",
                "**TaskCreate** `Tune regex linter`",
                tool_name="TaskCreate",
            ),
            ToolBatchEntry(
                "t2",
                "**TaskCreate**Apply fixes to opus agents",
                tool_name="TaskCreate",
            ),
        ]

        result = format_batch_message(entries, subagent_label="\U0001f916 subagent")

        lines = result.split("\n")
        assert lines[0] == "\u26a1 3 tool calls"
        assert lines[1] == "\U0001f916 subagent"
        assert "ToolSearch" in lines[2]
        assert "Creating 2 tasks\u2026" in result
        assert "1. Tune regex linter" in result
        assert "2. Apply fixes to opus agents" in result

    def test_task_update_entries_render_as_progress_section(self) -> None:
        entries = [
            ToolBatchEntry(
                "t1",
                "**TaskUpdate** `Tune regex linter -> in progress`",
                tool_name="TaskUpdate",
            ),
            ToolBatchEntry(
                "t2",
                "**TaskUpdate** `Apply fixes to opus agents -> completed`",
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
                "**TaskList** `refresh`",
                tool_name="TaskList",
            )
        ]

        result = format_batch_message(entries)

        assert result == "\u26a1 1 tool call\nRefreshing task list\u2026"


class TestIsBatchEligible:
    @pytest.mark.parametrize("content_type", ["tool_use", "tool_result"])
    def test_tool_types_eligible(self, content_type: str) -> None:
        task = MessageTask(task_type="content", content_type=content_type, parts=["x"])
        assert _is_batch_eligible(task) is True

    @pytest.mark.parametrize("content_type", ["text", "thinking", "assistant"])
    def test_non_tool_types_not_eligible(self, content_type: str) -> None:
        task = MessageTask(task_type="content", content_type=content_type, parts=["x"])
        assert _is_batch_eligible(task) is False

    def test_status_update_not_eligible(self) -> None:
        task = MessageTask(task_type="status_update", text="working")
        assert _is_batch_eligible(task) is False

    def test_status_clear_not_eligible(self) -> None:
        task = MessageTask(task_type="status_clear", text="clear")
        assert _is_batch_eligible(task) is False


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
        assert BATCH_MAX_ENTRIES == 10
        assert BATCH_MAX_LENGTH == 2800


class TestWindowStateBatchMode:
    def test_default_batch_mode(self) -> None:
        ws = WindowState()
        assert ws.batch_mode == DEFAULT_BATCH_MODE
        assert ws.batch_mode == "batched"

    @pytest.mark.parametrize(
        ("mode", "expect_key"),
        [("batched", False), ("verbose", True)],
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
            ({"session_id": "s1", "cwd": "/tmp"}, "batched"),
            ({"session_id": "s1", "cwd": "/tmp", "batch_mode": "verbose"}, "verbose"),
            ({"session_id": "s1", "cwd": "/tmp", "batch_mode": "batched"}, "batched"),
        ],
    )
    def test_from_dict(self, data: dict[str, str], expected: str) -> None:
        assert WindowState.from_dict(data).batch_mode == expected

    @pytest.mark.parametrize("mode", list(BATCH_MODES))
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
        assert mgr.get_batch_mode("@0") == "batched"

    def test_get_nonexistent_window(self, mgr: SessionManager) -> None:
        assert mgr.get_batch_mode("@999") == "batched"

    def test_set_mode(self, mgr: SessionManager) -> None:
        mgr.set_batch_mode("@0", "verbose")
        assert mgr.get_batch_mode("@0") == "verbose"

    def test_set_mode_validates(self, mgr: SessionManager) -> None:
        with pytest.raises(ValueError, match="Invalid batch mode"):
            mgr.set_batch_mode("@0", "invalid")

    @pytest.mark.parametrize(
        ("start", "expected"),
        [("batched", "verbose"), ("verbose", "batched")],
    )
    def test_cycle(self, mgr: SessionManager, start: str, expected: str) -> None:
        mgr.set_batch_mode("@0", start)
        assert mgr.cycle_batch_mode("@0") == expected
        assert mgr.get_batch_mode("@0") == expected

    def test_cycle_full_circle(self, mgr: SessionManager) -> None:
        mgr.cycle_batch_mode("@0")
        assert mgr.get_batch_mode("@0") == "verbose"
        mgr.cycle_batch_mode("@0")
        assert mgr.get_batch_mode("@0") == "batched"

    def test_set_same_mode_no_save(self, mgr: SessionManager, monkeypatch) -> None:
        mgr.set_batch_mode("@0", "verbose")
        save_calls = []
        monkeypatch.setattr(
            SessionManager, "_save_state", lambda self: save_calls.append(1)
        )
        mgr.set_batch_mode("@0", "verbose")  # same mode
        assert len(save_calls) == 0

    def test_get_invalid_stored_mode_returns_default(self, mgr: SessionManager) -> None:
        state = mgr.get_window_state("@0")
        state.batch_mode = "garbage"
        assert mgr.get_batch_mode("@0") == "batched"


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
) -> MessageTask:
    return MessageTask(
        task_type="content",
        content_type="tool_use",
        window_id=window_id,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        text=text,
        parts=[text],
        thread_id=thread_id,
    )


def _make_tool_result(
    tool_use_id: str | None = "tu1",
    text: str = "42 lines",
    thread_id: int | None = 10,
    window_id: str = "@0",
) -> MessageTask:
    return MessageTask(
        task_type="content",
        content_type="tool_result",
        window_id=window_id,
        tool_use_id=tool_use_id,
        text=text,
        parts=[text],
        thread_id=thread_id,
    )


class TestProcessBatchTask:
    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_tool_use_creates_batch(
        self, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(bot, 1, _make_tool_use())

        bkey = (1, 10)
        assert bkey in _active_batches
        batch = _active_batches[bkey]
        assert len(batch.entries) == 1
        assert batch.entries[0].tool_use_id == "tu1"
        assert batch.telegram_msg_id == 100

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_task_create_batch_sends_task_list(
        self, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(
            bot,
            1,
            _make_tool_use(
                text="**TaskCreate** `Understand the Problem Domain`",
                tool_name="TaskCreate",
            ),
        )
        sent_text = mock_send.await_args.args[2]

        assert sent_text.startswith("Creating 1 task\u2026\n")
        assert "1. Understand the Problem Domain" in sent_text
        assert "tool call" not in sent_text

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_tool_result_updates_entry(
        self, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(bot, 1, _make_tool_use())
        await _process_batch_task(bot, 1, _make_tool_result())

        batch = _active_batches[(1, 10)]
        assert batch.entries[0].tool_result_text == "42 lines"

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_multiple_tool_calls_accumulate(
        self, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(
            bot, 1, _make_tool_use(tool_use_id="tu1", text="Read a.py")
        )
        await _process_batch_task(
            bot, 1, _make_tool_result(tool_use_id="tu1", text="10 lines")
        )
        await _process_batch_task(
            bot, 1, _make_tool_use(tool_use_id="tu2", text="Edit a.py")
        )
        await _process_batch_task(
            bot, 1, _make_tool_use(tool_use_id="tu3", text="Bash make test")
        )

        batch = _active_batches[(1, 10)]
        assert len(batch.entries) == 3
        assert batch.entries[0].tool_use_id == "tu1"
        assert batch.entries[0].tool_result_text == "10 lines"
        assert batch.entries[1].tool_use_id == "tu2"
        assert batch.entries[1].tool_result_text is None
        assert batch.entries[2].tool_use_id == "tu3"

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_tool_result_truncates_long_text(
        self, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(bot, 1, _make_tool_use())
        long_result = "x" * 200 + "\nsecond line"
        await _process_batch_task(bot, 1, _make_tool_result(text=long_result))

        batch = _active_batches[(1, 10)]
        result_text = batch.entries[0].tool_result_text
        assert result_text is not None
        assert len(result_text) <= 200
        assert "\n" not in result_text

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.message_queue._flush_batch", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.message_queue._process_content_task", new_callable=AsyncMock
    )
    async def test_tool_result_no_matching_entry_flushes(
        self, mock_process, mock_flush, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(bot, 1, _make_tool_use(tool_use_id="tu1"))
        await _process_batch_task(bot, 1, _make_tool_result(tool_use_id="tu_unknown"))
        mock_flush.assert_awaited_once()
        mock_process.assert_awaited_once()

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    @patch("ccgram.handlers.message_queue._flush_batch", new_callable=AsyncMock)
    async def test_different_window_flushes_old_batch(
        self, mock_flush, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(bot, 1, _make_tool_use(window_id="@0"))
        await _process_batch_task(
            bot, 1, _make_tool_use(window_id="@1", tool_use_id="tu2")
        )
        mock_flush.assert_awaited_once()

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_batch_overflow_entries_splits(
        self, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        for i in range(BATCH_MAX_ENTRIES + 1):
            await _process_batch_task(
                bot, 1, _make_tool_use(tool_use_id=f"tu{i}", text=f"Tool {i}")
            )

        batch = _active_batches[(1, 10)]
        assert len(batch.entries) == 2
        assert batch.entries[0].tool_use_id == f"tu{BATCH_MAX_ENTRIES - 1}"
        assert batch.entries[1].tool_use_id == f"tu{BATCH_MAX_ENTRIES}"

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_batch_clears_status_on_first_send(
        self, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(bot, 1, _make_tool_use())
        mock_clear.assert_awaited_once()

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_second_tool_edits_existing_message(
        self, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(bot, 1, _make_tool_use(tool_use_id="tu1"))
        mock_send.assert_awaited_once()  # First tool_use sends

        await _process_batch_task(
            bot, 1, _make_tool_use(tool_use_id="tu2", text="Edit b.py")
        )
        bot.edit_message_text.assert_awaited()  # Second tool_use edits


class TestHandleContentTask:
    @patch("ccgram.handlers.message_queue.session_manager")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch("ccgram.handlers.message_queue._process_batch_task", new_callable=AsyncMock)
    async def test_batch_eligible_routes_to_batch(
        self, mock_batch, mock_should, mock_sm
    ) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = MessageTask(
            task_type="content",
            content_type="tool_use",
            window_id="@0",
            parts=["Read x"],
        )
        extra = await _handle_content_task(bot, 1, task, queue, lock)
        assert extra == 0
        mock_batch.assert_awaited_once()

    @patch("ccgram.handlers.message_queue.session_manager")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=False)
    @patch(
        "ccgram.handlers.message_queue._process_content_task", new_callable=AsyncMock
    )
    async def test_verbose_mode_skips_batch(
        self, mock_process, mock_should, mock_sm
    ) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = MessageTask(
            task_type="content",
            content_type="tool_use",
            window_id="@0",
            parts=["Read x"],
        )
        extra = await _handle_content_task(bot, 1, task, queue, lock)
        assert extra == 0
        mock_process.assert_awaited_once()

    @patch("ccgram.handlers.message_queue.session_manager")
    @patch("ccgram.handlers.message_queue._flush_batch", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.message_queue._process_content_task", new_callable=AsyncMock
    )
    async def test_text_flushes_active_batch(
        self, mock_process, mock_flush, mock_sm
    ) -> None:
        _active_batches[(1, 0)] = ToolBatch(window_id="@0", thread_id=0, entries=[])

        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = MessageTask(
            task_type="content",
            content_type="text",
            window_id="@0",
            parts=["Hello"],
        )
        await _handle_content_task(bot, 1, task, queue, lock)
        mock_flush.assert_awaited_once_with(bot, 1, 0)
        mock_process.assert_awaited_once()

    @patch("ccgram.handlers.message_queue.session_manager")
    @patch("ccgram.handlers.message_queue._flush_batch", new_callable=AsyncMock)
    @patch(
        "ccgram.handlers.message_queue._process_content_task", new_callable=AsyncMock
    )
    async def test_thinking_flushes_active_batch(
        self, mock_process, mock_flush, mock_sm
    ) -> None:
        _active_batches[(1, 5)] = ToolBatch(window_id="@0", thread_id=5, entries=[])

        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = MessageTask(
            task_type="content",
            content_type="thinking",
            window_id="@0",
            parts=["Thinking..."],
            thread_id=5,
        )
        await _handle_content_task(bot, 1, task, queue, lock)
        mock_flush.assert_awaited_once_with(bot, 1, 5)

    @patch("ccgram.handlers.message_queue.session_manager")
    @patch(
        "ccgram.handlers.message_queue._process_content_task", new_callable=AsyncMock
    )
    async def test_no_batch_no_flush(self, mock_process, mock_sm) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = MessageTask(
            task_type="content",
            content_type="text",
            window_id="@0",
            parts=["Hello"],
        )
        await _handle_content_task(bot, 1, task, queue, lock)
        mock_process.assert_awaited_once()

    @patch("ccgram.handlers.message_queue.session_manager")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch("ccgram.handlers.message_queue._process_batch_task", new_callable=AsyncMock)
    async def test_no_window_id_skips_batch(
        self, mock_batch, mock_should, mock_sm
    ) -> None:
        bot = AsyncMock()
        queue: asyncio.Queue[MessageTask] = asyncio.Queue()
        lock = asyncio.Lock()
        task = MessageTask(
            task_type="content",
            content_type="tool_use",
            window_id=None,
            parts=["Read x"],
        )
        with patch(
            "ccgram.handlers.message_queue._process_content_task",
            new_callable=AsyncMock,
        ) as mock_process:
            await _handle_content_task(bot, 1, task, queue, lock)
            mock_process.assert_awaited_once()
            mock_batch.assert_not_awaited()


class TestFlushBatch:
    @patch("ccgram.handlers.message_queue.thread_router")
    async def test_flush_removes_batch(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 10)] = ToolBatch(
            window_id="@0",
            thread_id=10,
            entries=[ToolBatchEntry("t1", "Read x", "ok")],
            telegram_msg_id=100,
        )

        bot = AsyncMock()
        await _flush_batch(bot, 1, 10)
        assert (1, 10) not in _active_batches

    async def test_flush_noop_when_no_batch(self) -> None:
        bot = AsyncMock()
        await _flush_batch(bot, 1, 10)  # should not raise

    @patch("ccgram.handlers.message_queue.thread_router")
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
        await _flush_batch(bot, 1, 0)
        bot.edit_message_text.assert_awaited()

    @patch("ccgram.handlers.message_queue.thread_router")
    async def test_flush_no_edit_without_telegram_msg_id(self, mock_tr) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 0)] = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[ToolBatchEntry("t1", "Read x")],
            telegram_msg_id=None,
        )

        bot = AsyncMock()
        await _flush_batch(bot, 1, 0)
        bot.edit_message_text.assert_not_awaited()
        assert (1, 0) not in _active_batches

    async def test_flush_empty_entries_noop(self) -> None:
        _active_batches[(1, 0)] = ToolBatch(window_id="@0", thread_id=0, entries=[])
        bot = AsyncMock()
        await _flush_batch(bot, 1, 0)
        assert (1, 0) not in _active_batches
        bot.edit_message_text.assert_not_awaited()

    @patch(
        "ccgram.handlers.hook_events.get_subagent_names", return_value=["researcher"]
    )
    @patch("ccgram.handlers.message_queue.thread_router")
    async def test_flush_includes_subagent_label(self, mock_tr, _mock_names) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 0)] = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[ToolBatchEntry("t1", "Read x", "ok")],
            telegram_msg_id=100,
        )

        bot = AsyncMock()
        await _flush_batch(bot, 1, 0)
        text_sent = bot.edit_message_text.call_args.kwargs["text"]
        assert "researcher" in text_sent

    @patch("ccgram.handlers.message_queue.thread_router")
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
        await _flush_batch(bot, 1, 0)
        assert (1, 0) not in _active_batches


class TestBatchIsolation:
    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_different_threads_separate_batches(
        self, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(
            bot, 1, _make_tool_use(thread_id=10, tool_use_id="tu1")
        )
        await _process_batch_task(
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
    @patch("ccgram.handlers.message_queue.asyncio.sleep", new_callable=AsyncMock)
    @patch("ccgram.handlers.message_queue._handle_content_task", new_callable=AsyncMock)
    async def test_retry_after_retries_same_task(self, mock_handle, mock_sleep) -> None:
        await shutdown_workers()
        mock_handle.side_effect = [RetryAfter(1), 0]

        bot = AsyncMock()
        queue = get_or_create_queue(bot, 1)
        queue.put_nowait(
            MessageTask(
                task_type="content",
                window_id="@0",
                parts=["hello"],
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
    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    @patch(
        "ccgram.handlers.message_queue._process_content_task", new_callable=AsyncMock
    )
    async def test_tool_result_no_active_batch_falls_through(
        self, mock_process, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        bot = AsyncMock()
        task = _make_tool_result(tool_use_id="tu1", text="result text")
        await _process_batch_task(bot, 1, task)
        mock_process.assert_awaited_once_with(bot, 1, task)

    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    @patch(
        "ccgram.handlers.message_queue._process_content_task", new_callable=AsyncMock
    )
    async def test_tool_result_none_tool_use_id_falls_through(
        self, mock_process, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(bot, 1, _make_tool_use(tool_use_id="tu1"))
        task = _make_tool_result(tool_use_id=None, text="result text")
        await _process_batch_task(bot, 1, task)
        mock_process.assert_awaited_once_with(bot, 1, task)
        assert (1, 10) in _active_batches
        assert len(_active_batches[(1, 10)].entries) == 1


class TestBatchLengthOverflow:
    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_overflow_on_length(
        self, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        long_text = "x" * 500
        for i in range(8):
            await _process_batch_task(
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
    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    async def test_flush_sends_when_no_telegram_msg_id(
        self, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        _active_batches[(1, 0)] = ToolBatch(
            window_id="@0",
            thread_id=0,
            entries=[ToolBatchEntry("t1", "Read x", "ok")],
            telegram_msg_id=None,  # first send failed
        )

        bot = AsyncMock()
        await _flush_batch(bot, 1, 0)
        mock_send.assert_awaited_once()
        send_args = mock_send.call_args
        assert "Read x" in send_args.args[2]
        assert (1, 0) not in _active_batches


class TestDefensiveElseBranch:
    @patch("ccgram.handlers.message_queue.thread_router")
    @patch(
        "ccgram.handlers.message_queue._process_content_task", new_callable=AsyncMock
    )
    async def test_unexpected_content_type_routes_to_normal(
        self, mock_process, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        bot = AsyncMock()
        task = MessageTask(
            task_type="content",
            content_type="text",
            window_id="@0",
            parts=["hello"],
            thread_id=10,
        )
        await _process_batch_task(bot, 1, task)
        mock_process.assert_awaited_once_with(bot, 1, task)


class TestDifferentUsersIsolation:
    @patch("ccgram.handlers.message_queue.thread_router")
    @patch("ccgram.handlers.message_queue.rate_limit_send_message")
    @patch("ccgram.handlers.message_queue._should_batch", return_value=True)
    @patch(
        "ccgram.handlers.message_queue._do_clear_status_message", new_callable=AsyncMock
    )
    async def test_different_users_same_thread_separate_batches(
        self, mock_clear, mock_should, mock_send, mock_tr
    ) -> None:
        mock_tr.resolve_chat_id.return_value = 42
        sent_msg = MagicMock()
        sent_msg.message_id = 100
        mock_send.return_value = sent_msg

        bot = AsyncMock()
        await _process_batch_task(
            bot, 1, _make_tool_use(thread_id=10, tool_use_id="tu1")
        )
        await _process_batch_task(
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


class TestBatchResultPrefix:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("error: file not found", "\u274c"),
            ("FAILED test_foo.py::test_bar", "\u274c"),
            ("Exception: KeyError", "\u274c"),
            ("Traceback (most recent call last)", "\u274c"),
            ("exit code 1", "\u274c"),
            ("exit code 127", "\u274c"),
            ("2 failures", "\u274c"),
            ("1 failure", "\u274c"),
            ("test failure in module", "\u274c"),
            ("23 passed", "\u2705"),
            ("Tests passed successfully", "\u2705"),
            ("success", "\u2705"),
            ("exit code 0", "\u2705"),
            ("10 lines read", "\u23bf"),
            ("file written", "\u23bf"),
            ("", "\u23bf"),
        ],
    )
    def test_prefix_detection(self, text, expected):
        from ccgram.handlers.message_queue import _batch_result_prefix

        assert _batch_result_prefix(text) == expected

    def test_error_takes_priority_over_success(self):
        from ccgram.handlers.message_queue import _batch_result_prefix

        text = "3 passed, 1 FAILED"
        assert _batch_result_prefix(text) == "\u274c"


class TestBatchEntryFormatting:
    def test_success_prefix_in_formatted_entry(self):
        entry = ToolBatchEntry("t1", "Bash make test", "23 passed")
        result = format_batch_message([entry])
        assert "\u2705" in result
        assert "23 passed" in result

    def test_error_prefix_in_formatted_entry(self):
        entry = ToolBatchEntry("t1", "Bash make test", "FAILED test_foo")
        result = format_batch_message([entry])
        assert "\u274c" in result
        assert "FAILED test_foo" in result

    def test_neutral_prefix_in_formatted_entry(self):
        entry = ToolBatchEntry("t1", "Read src/foo.py", "42 lines")
        result = format_batch_message([entry])
        assert "\u23bf" in result
        assert "42 lines" in result

    def test_pending_entry_shows_hourglass(self):
        entry = ToolBatchEntry("t1", "Bash make test")
        result = format_batch_message([entry])
        assert "\u23f3" in result
