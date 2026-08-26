import ast
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import TelegramError

from ccgram.expandable_quote import EXPANDABLE_QUOTE_END, EXPANDABLE_QUOTE_START
from ccgram.handlers.messaging_pipeline.message_task import (
    StatusClearTask,
    StatusUpdateTask,
)
from ccgram.handlers.status.status_bubble import (
    _status_msg_info,
    clear_status_message,
    clear_status_msg_info,
    convert_status_to_content,
    format_claude_task_status,
    format_pane_block,
    process_status_clear,
    process_status_update,
    send_status_text,
)
from ccgram.window_state_store import PaneInfo, WindowState, window_store

USER_ID = 1
THREAD_ID = 10
WINDOW_ID = "@0"
CHAT_ID = 42


@pytest.fixture(autouse=True)
def _clear_status_tracking():
    _status_msg_info.clear()
    yield
    _status_msg_info.clear()


def _make_bot(send_id: int = 99) -> AsyncMock:
    """Build an AsyncMock bot that returns a sensible Message on send."""
    bot = AsyncMock()
    sent = MagicMock()
    sent.message_id = send_id
    bot.send_message.return_value = sent
    return bot


class TestSendStatusText:
    @patch("ccgram.handlers.status.status_bubble.config", create=True)
    @patch("ccgram.handlers.status.status_bubble.thread_router")
    async def test_hidden_status_does_not_send_or_track(self, mock_router, mock_config):
        mock_config.hide_status = True
        bot = _make_bot()

        await send_status_text(bot, USER_ID, THREAD_ID, WINDOW_ID, "running...")

        mock_router.resolve_chat_id.assert_not_called()
        bot.send_message.assert_not_awaited()
        assert (USER_ID, THREAD_ID) not in _status_msg_info

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    async def test_sends_new_message(self, mock_router):
        mock_router.resolve_chat_id.return_value = CHAT_ID

        bot = _make_bot(send_id=99)
        await send_status_text(bot, USER_ID, THREAD_ID, WINDOW_ID, "running...")

        # safe_send uses bot.send_message with entity-based formatting.
        bot.send_message.assert_awaited_once()
        assert _status_msg_info[(USER_ID, THREAD_ID)] == (
            99,
            WINDOW_ID,
            "running...",
            CHAT_ID,
        )

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_edits_existing_same_window(self, mock_edit, mock_router):
        mock_router.resolve_chat_id.return_value = CHAT_ID
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, WINDOW_ID, "old", CHAT_ID)
        mock_edit.return_value = True

        bot = _make_bot()
        await send_status_text(bot, USER_ID, THREAD_ID, WINDOW_ID, "new text")

        mock_edit.assert_awaited_once()
        bot.send_message.assert_not_called()
        assert _status_msg_info[(USER_ID, THREAD_ID)] == (
            50,
            WINDOW_ID,
            "new text",
            CHAT_ID,
        )

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_replace_failure_deletes_old_bubble_before_resending(
        self, mock_edit, mock_router
    ):
        # When the in-place edit can't update the existing bubble, the old
        # message must be deleted before a fresh one is created — otherwise
        # the topic ends up with two status bubbles, the first one orphaned.
        mock_router.resolve_chat_id.return_value = CHAT_ID
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, WINDOW_ID, "old", CHAT_ID)
        mock_edit.return_value = False

        bot = _make_bot(send_id=99)
        await send_status_text(bot, USER_ID, THREAD_ID, WINDOW_ID, "new text")

        bot.delete_message.assert_awaited_once_with(chat_id=CHAT_ID, message_id=50)
        bot.send_message.assert_awaited_once()
        assert _status_msg_info[(USER_ID, THREAD_ID)] == (
            99,
            WINDOW_ID,
            "new text",
            CHAT_ID,
        )

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.clear_status_message",
        new_callable=AsyncMock,
    )
    async def test_existing_status_for_other_window_is_cleared_first(
        self, mock_clear, mock_router
    ):
        mock_router.resolve_chat_id.return_value = CHAT_ID
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, "@1", "running...", CHAT_ID)

        bot = _make_bot(send_id=300)
        await send_status_text(bot, USER_ID, THREAD_ID, WINDOW_ID, "running...")

        mock_clear.assert_called_once_with(bot, USER_ID, THREAD_ID)
        bot.send_message.assert_awaited_once()
        assert _status_msg_info[(USER_ID, THREAD_ID)] == (
            300,
            WINDOW_ID,
            "running...",
            CHAT_ID,
        )

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_dedup_identical_content(self, mock_edit, mock_router):
        mock_router.resolve_chat_id.return_value = CHAT_ID
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, WINDOW_ID, "same", CHAT_ID)

        bot = _make_bot()
        await send_status_text(bot, USER_ID, THREAD_ID, WINDOW_ID, "same")

        mock_edit.assert_not_called()
        bot.send_message.assert_not_called()


class TestClearStatusMessage:
    async def test_deletes_tracked_message(self):
        # No DraftStream tracked → falls back to bot.delete_message.
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, WINDOW_ID, "text", CHAT_ID)

        bot = AsyncMock()
        await clear_status_message(bot, USER_ID, THREAD_ID)

        bot.delete_message.assert_called_once_with(chat_id=CHAT_ID, message_id=50)
        assert (USER_ID, THREAD_ID) not in _status_msg_info

    async def test_swallows_delete_failure_and_forgets_tracking(self):
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, WINDOW_ID, "text", CHAT_ID)

        bot = AsyncMock()
        bot.delete_message.side_effect = TelegramError("message to delete not found")
        await clear_status_message(bot, USER_ID, THREAD_ID)

        assert (USER_ID, THREAD_ID) not in _status_msg_info

    async def test_noop_when_no_tracking(self):
        bot = AsyncMock()
        await clear_status_message(bot, USER_ID, THREAD_ID)

        bot.delete_message.assert_not_called()


class TestConvertStatusToContent:
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_converts_status_to_content(self, mock_edit):
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, WINDOW_ID, "old", CHAT_ID)
        mock_edit.return_value = True

        bot = AsyncMock()
        result = await convert_status_to_content(
            bot, USER_ID, THREAD_ID, WINDOW_ID, "content text"
        )

        assert result == 50
        mock_edit.assert_called_once()
        assert (USER_ID, THREAD_ID) not in _status_msg_info

    async def test_returns_none_when_no_status(self):
        bot = AsyncMock()
        result = await convert_status_to_content(
            bot, USER_ID, THREAD_ID, WINDOW_ID, "content"
        )

        assert result is None

    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_returns_none_when_edit_fails(self, mock_edit):
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, WINDOW_ID, "old", CHAT_ID)
        mock_edit.return_value = False

        result = await convert_status_to_content(
            AsyncMock(), USER_ID, THREAD_ID, WINDOW_ID, "content text"
        )

        assert result is None
        # Tracking is dropped either way — the bubble is no longer ours to edit.
        assert (USER_ID, THREAD_ID) not in _status_msg_info

    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_deletes_status_from_different_window(self, mock_edit):
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, "@1", "old", CHAT_ID)

        bot = AsyncMock()
        result = await convert_status_to_content(
            bot, USER_ID, THREAD_ID, WINDOW_ID, "content"
        )

        assert result is None
        bot.delete_message.assert_called_once_with(chat_id=CHAT_ID, message_id=50)


class TestFormatClaudeTaskStatus:
    @patch(
        "ccgram.handlers.status.status_bubble.get_task_snapshot",
        return_value=None,
    )
    @patch("ccgram.handlers.status.status_bubble.get_wait_header", return_value=None)
    def test_no_tasks_returns_base_text(self, mock_wait, mock_snap):
        result = format_claude_task_status(WINDOW_ID, "Running")
        assert result == "Running"

    @patch(
        "ccgram.handlers.status.status_bubble.get_task_snapshot",
        return_value=None,
    )
    @patch(
        "ccgram.handlers.status.status_bubble.get_wait_header",
        return_value="Waiting for input...",
    )
    def test_with_wait_header(self, mock_wait, mock_snap):
        result = format_claude_task_status(WINDOW_ID, "Running")
        assert result == "Waiting for input..."

    @patch("ccgram.handlers.status.status_bubble.get_task_snapshot")
    @patch("ccgram.handlers.status.status_bubble.get_wait_header", return_value=None)
    def test_with_task_list(self, mock_wait, mock_snap):
        item = MagicMock()
        item.status = "in_progress"
        item.active_form = "writing tests"
        item.subject = "Task A"
        item.task_id = 1
        item.owner = None
        item.blocked_by = []
        snapshot = MagicMock()
        snapshot.total_count = 1
        snapshot.done_count = 0
        snapshot.open_count = 1
        snapshot.items = [item]
        mock_snap.return_value = snapshot

        result = format_claude_task_status(WINDOW_ID, "Running")
        assert result is not None
        assert "1 tasks (0 done, 1 open)" in result
        assert "writing tests" in result


class TestClearStatusMsgInfo:
    def test_clears_specific_thread(self):
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, WINDOW_ID, "text", CHAT_ID)
        _status_msg_info[(USER_ID, 20)] = (51, WINDOW_ID, "text", CHAT_ID)

        clear_status_msg_info(USER_ID, THREAD_ID)

        assert (USER_ID, THREAD_ID) not in _status_msg_info
        assert (USER_ID, 20) in _status_msg_info

    def test_noop_when_not_tracked(self):
        clear_status_msg_info(USER_ID, THREAD_ID)


class TestProcessStatusUpdate:
    @patch("ccgram.handlers.status.status_bubble.thread_router")
    async def test_returns_none_when_absorbed(self, mock_router):
        mock_router.resolve_chat_id.return_value = CHAT_ID

        bot = _make_bot(send_id=99)
        task = StatusUpdateTask(
            window_id=WINDOW_ID, text="thinking", thread_id=THREAD_ID
        )
        result = await process_status_update(bot, USER_ID, task)

        assert result is None
        # Legacy DraftStream uses bot.send_message for the initial bubble.
        bot.send_message.assert_awaited_once()

    @patch("ccgram.handlers.status.status_bubble.thread_router")
    async def test_clears_when_no_status_text(self, mock_router):
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, WINDOW_ID, "old", CHAT_ID)

        bot = _make_bot()
        task = StatusUpdateTask(window_id=WINDOW_ID, text=None, thread_id=THREAD_ID)
        with patch(
            "ccgram.handlers.status.status_bubble.format_claude_task_status",
            return_value=None,
        ):
            result = await process_status_update(bot, USER_ID, task)

        assert result is None
        assert (USER_ID, THREAD_ID) not in _status_msg_info


class TestProcessStatusClear:
    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(
        "ccgram.handlers.status.status_bubble.edit_with_fallback",
        new_callable=AsyncMock,
    )
    async def test_re_renders_with_task_snapshot(self, mock_edit, mock_router):
        mock_router.resolve_chat_id.return_value = CHAT_ID
        mock_edit.return_value = True
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, WINDOW_ID, "old", CHAT_ID)

        bot = AsyncMock()
        task = StatusClearTask(window_id=WINDOW_ID, thread_id=THREAD_ID)
        with patch(
            "ccgram.handlers.status.status_bubble.format_claude_task_status",
            return_value="1 tasks (1 done, 0 open)",
        ):
            await process_status_clear(bot, USER_ID, task)

        mock_edit.assert_called_once()

    async def test_deletes_when_no_snapshot(self):
        _status_msg_info[(USER_ID, THREAD_ID)] = (50, WINDOW_ID, "old", CHAT_ID)

        bot = AsyncMock()
        task = StatusClearTask(window_id=WINDOW_ID, thread_id=THREAD_ID)
        with patch(
            "ccgram.handlers.status.status_bubble.format_claude_task_status",
            return_value=None,
        ):
            await process_status_clear(bot, USER_ID, task)

        bot.delete_message.assert_called_once_with(chat_id=CHAT_ID, message_id=50)
        assert (USER_ID, THREAD_ID) not in _status_msg_info


class TestNoImportFromMessageQueue:
    def test_no_import_from_message_queue(self) -> None:
        import ccgram.handlers.status.status_bubble as mod

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
        assert violations == [], (
            f"status_bubble imports from message_queue: {violations}"
        )


@pytest.fixture
def _isolated_window_store():
    saved = dict(window_store.window_states)
    window_store.window_states.clear()
    try:
        yield
    finally:
        window_store.window_states.clear()
        window_store.window_states.update(saved)


def _seed_panes(window_id: str, panes: list[PaneInfo]) -> None:
    state = WindowState()
    state.panes = {p.pane_id: p for p in panes}
    window_store.window_states[window_id] = state


class TestFormatPaneBlock:
    def test_returns_none_for_unknown_window(self, _isolated_window_store):
        assert format_pane_block("@99") is None

    def test_returns_none_for_single_pane(self, _isolated_window_store):
        _seed_panes("@0", [PaneInfo(pane_id="%1", state="active")])
        assert format_pane_block("@0") is None

    def test_returns_none_when_only_dead_panes(self, _isolated_window_store):
        _seed_panes(
            "@0",
            [
                PaneInfo(pane_id="%1", state="dead"),
                PaneInfo(pane_id="%2", state="dead"),
            ],
        )
        assert format_pane_block("@0") is None

    def test_two_panes_inline_list(self, _isolated_window_store):
        _seed_panes(
            "@0",
            [
                PaneInfo(pane_id="%5", state="active"),
                PaneInfo(pane_id="%6", state="blocked"),
            ],
        )
        result = format_pane_block("@0")
        assert result is not None
        assert result.startswith("└ ")
        assert "%5 active" in result
        assert "%6 ⏸ blocked" in result
        # Single line for ≤3 panes — no expandable quote sentinel.
        assert "\n" not in result
        assert EXPANDABLE_QUOTE_START not in result

    def test_three_panes_uses_pane_name_when_set(self, _isolated_window_store):
        _seed_panes(
            "@0",
            [
                PaneInfo(pane_id="%5", name="api-gateway", state="active"),
                PaneInfo(pane_id="%6", state="blocked"),
                PaneInfo(pane_id="%7", state="idle"),
            ],
        )
        result = format_pane_block("@0")
        assert result is not None
        assert "api-gateway active" in result
        assert "%5 active" not in result  # name preferred over pane_id
        assert "%6 ⏸ blocked" in result
        assert " · " in result

    def test_idle_age_renders_minutes(self, _isolated_window_store):
        with patch("ccgram.handlers.status.status_bubble.time") as mock_time:
            mock_time.time.return_value = 1_000_000.0
            _seed_panes(
                "@0",
                [
                    PaneInfo(pane_id="%5", state="active"),
                    PaneInfo(
                        pane_id="%6", state="idle", last_active_ts=1_000_000.0 - 120.0
                    ),
                ],
            )
            result = format_pane_block("@0")
        assert result is not None
        assert "%6 idle 2m" in result

    def test_idle_age_renders_hours(self, _isolated_window_store):
        with patch("ccgram.handlers.status.status_bubble.time") as mock_time:
            mock_time.time.return_value = 1_000_000.0
            _seed_panes(
                "@0",
                [
                    PaneInfo(pane_id="%5", state="active"),
                    PaneInfo(
                        pane_id="%6",
                        state="idle",
                        last_active_ts=1_000_000.0 - 7200.0,
                    ),
                ],
            )
            result = format_pane_block("@0")
        assert result is not None
        assert "%6 idle 2h" in result

    def test_four_panes_wrapped_in_expandable_quote(self, _isolated_window_store):
        _seed_panes(
            "@0",
            [
                PaneInfo(pane_id="%5", state="active"),
                PaneInfo(pane_id="%6", state="idle"),
                PaneInfo(pane_id="%7", state="blocked"),
                PaneInfo(pane_id="%8", state="idle"),
            ],
        )
        result = format_pane_block("@0")
        assert result is not None
        assert result.startswith(EXPANDABLE_QUOTE_START)
        assert result.endswith(EXPANDABLE_QUOTE_END)
        assert "└ %5 active" in result
        assert "└ %8" in result
        # 4 panes → multi-line content inside the quote.
        assert result.count("\n") >= 3

    def test_panes_sorted_for_stable_output(self, _isolated_window_store):
        _seed_panes(
            "@0",
            [
                PaneInfo(pane_id="%10", state="active"),
                PaneInfo(pane_id="%2", state="idle"),
            ],
        )
        result = format_pane_block("@0")
        assert result is not None
        assert result.index("%2") < result.index("%10")

    def test_supports_opaque_herdr_pane_ids(self, _isolated_window_store):
        _seed_panes(
            "wMM:t1",
            [
                PaneInfo(pane_id="wMM:p2", state="idle"),
                PaneInfo(pane_id="wMM:p1", state="active"),
            ],
        )
        result = format_pane_block("wMM:t1")
        assert result is not None
        assert result.index("wMM:p1") < result.index("wMM:p2")

    def test_dead_panes_excluded_when_others_visible(self, _isolated_window_store):
        _seed_panes(
            "@0",
            [
                PaneInfo(pane_id="%5", state="active"),
                PaneInfo(pane_id="%6", state="idle"),
                PaneInfo(pane_id="%7", state="dead"),
            ],
        )
        result = format_pane_block("@0")
        assert result is not None
        assert "%7" not in result


class TestFormatClaudeTaskStatusWithPanes:
    @patch(
        "ccgram.handlers.status.status_bubble.get_task_snapshot",
        return_value=None,
    )
    @patch("ccgram.handlers.status.status_bubble.get_wait_header", return_value=None)
    def test_pane_block_appended_to_base_text(
        self, mock_wait, mock_snap, _isolated_window_store
    ):
        _seed_panes(
            "@0",
            [
                PaneInfo(pane_id="%5", state="active"),
                PaneInfo(pane_id="%6", state="idle"),
            ],
        )
        result = format_claude_task_status("@0", "Running")
        assert result is not None
        assert result.splitlines()[0] == "Running"
        assert "└ %5 active" in result
        assert "%6" in result

    @patch(
        "ccgram.handlers.status.status_bubble.get_task_snapshot",
        return_value=None,
    )
    @patch("ccgram.handlers.status.status_bubble.get_wait_header", return_value=None)
    def test_no_pane_block_for_single_pane(
        self, mock_wait, mock_snap, _isolated_window_store
    ):
        _seed_panes("@0", [PaneInfo(pane_id="%5", state="active")])
        result = format_claude_task_status("@0", "Running")
        assert result == "Running"

    @patch("ccgram.handlers.status.status_bubble.get_task_snapshot")
    @patch("ccgram.handlers.status.status_bubble.get_wait_header", return_value=None)
    def test_pane_block_inserted_between_header_and_tasks(
        self, mock_wait, mock_snap, _isolated_window_store
    ):
        _seed_panes(
            "@0",
            [
                PaneInfo(pane_id="%5", state="active"),
                PaneInfo(pane_id="%6", state="idle"),
            ],
        )
        item = MagicMock()
        item.status = "in_progress"
        item.active_form = "writing tests"
        item.subject = "Task A"
        item.task_id = 1
        item.owner = None
        item.blocked_by = []
        snapshot = MagicMock()
        snapshot.total_count = 1
        snapshot.done_count = 0
        snapshot.open_count = 1
        snapshot.items = [item]
        mock_snap.return_value = snapshot

        result = format_claude_task_status("@0", "Running")
        assert result is not None
        lines = result.splitlines()
        # header → pane block → task summary → task item
        assert lines[0] == "Running"
        assert lines[1].startswith("└ ")
        assert "1 tasks (0 done, 1 open)" in lines[2]


# Patch target for the lazily-imported build_dashboard_button (loaded inside
# build_status_keyboard at call time from status_bar_actions, not at module level).
_DASH_PATCH = "ccgram.handlers.status.status_bar_actions.build_dashboard_button"


class TestBuildStatusKeyboardGroupVsPrivate:
    """#178: web_app (Dashboard) button must not appear in group/topic chats."""

    @patch(_DASH_PATCH)
    def test_private_chat_shows_dashboard_button(self, mock_dash):
        """is_group=False → private chat → build_dashboard_button is consulted."""
        from ccgram.handlers.status.status_bubble import build_status_keyboard

        mock_dash.return_value = MagicMock(web_app=MagicMock(), callback_data=None)
        build_status_keyboard(WINDOW_ID, user_id=USER_ID, is_group=False)
        mock_dash.assert_called_once_with(WINDOW_ID, USER_ID)

    @patch(_DASH_PATCH)
    def test_group_chat_hides_dashboard_button(self, mock_dash):
        """is_group=True → Telegram rejects web_app → build_dashboard_button skipped."""
        from ccgram.handlers.status.status_bubble import build_status_keyboard

        mock_dash.return_value = MagicMock(web_app=MagicMock(), callback_data=None)
        build_status_keyboard(WINDOW_ID, user_id=USER_ID, is_group=True)
        mock_dash.assert_not_called()

    @patch("ccgram.handlers.status.status_bubble.config", create=True)
    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(_DASH_PATCH)
    async def test_send_in_thread_does_not_call_dashboard_button(
        self, mock_dash, mock_router, mock_config
    ):
        """send_status_text with non-zero thread_id suppresses the dashboard button."""
        mock_config.hide_status = False
        mock_router.resolve_chat_id.return_value = CHAT_ID
        import ccgram.handlers.status.status_bubble as bubble_mod

        await bubble_mod.send_status_text(
            _make_bot(77), USER_ID, THREAD_ID, WINDOW_ID, "working"
        )
        mock_dash.assert_not_called()

    @patch("ccgram.handlers.status.status_bubble.config", create=True)
    @patch("ccgram.handlers.status.status_bubble.thread_router")
    @patch(_DASH_PATCH)
    async def test_send_in_private_calls_dashboard_button(
        self, mock_dash, mock_router, mock_config
    ):
        """send_status_text with thread_id==0 (private chat) consults the dashboard button."""
        mock_config.hide_status = False
        mock_router.resolve_chat_id.return_value = CHAT_ID
        mock_dash.return_value = None  # Mini App not configured — button absent
        import ccgram.handlers.status.status_bubble as bubble_mod

        await bubble_mod.send_status_text(
            _make_bot(78), USER_ID, 0, WINDOW_ID, "working"
        )
        mock_dash.assert_called_once_with(WINDOW_ID, USER_ID)
