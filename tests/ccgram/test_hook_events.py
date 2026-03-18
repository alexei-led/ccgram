"""Tests for hook event dispatcher."""

from unittest.mock import ANY, AsyncMock, patch

from telegram import Bot

from ccgram.handlers.hook_events import (
    HookEvent,
    _active_subagents,
    _resolve_users_for_window_key,
    clear_subagents,
    dispatch_hook_event,
    get_subagent_count,
)


def _make_event(
    event_type: str = "Stop",
    window_key: str = "ccgram:@0",
    session_id: str = "test-id",
    data: dict | None = None,
    timestamp: float = 0.0,
) -> HookEvent:
    return HookEvent(
        event_type=event_type,
        window_key=window_key,
        session_id=session_id,
        data=data or {},
        timestamp=timestamp,
    )


class TestResolveUsersForWindowKey:
    def test_extracts_window_id(self, monkeypatch) -> None:
        bindings = [
            (111, 42, "@0"),
            (222, 99, "@5"),
        ]
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter(bindings),
        )
        result = _resolve_users_for_window_key("ccgram:@0")
        assert result == [(111, 42, "@0")]

    def test_no_match(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([]),
        )
        result = _resolve_users_for_window_key("ccgram:@99")
        assert result == []

    def test_invalid_key_format(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([]),
        )
        result = _resolve_users_for_window_key("nocolon")
        assert result == []


class TestSubagentTracking:
    def setup_method(self) -> None:
        _active_subagents.clear()

    def test_start_increments_count(self) -> None:
        _active_subagents["@0"] = {"a1"}
        assert get_subagent_count("@0") == 1

    def test_clear_removes_all(self) -> None:
        _active_subagents["@0"] = {"a1", "a2"}
        clear_subagents("@0")
        assert get_subagent_count("@0") == 0

    def test_count_missing_window(self) -> None:
        assert get_subagent_count("@999") == 0


class TestDispatchHookEvent:
    async def test_unknown_event_ignored(self) -> None:
        event = _make_event(event_type="SomeUnknownEvent")
        await dispatch_hook_event(event, None)  # type: ignore[arg-type]

    async def test_session_start_ignored(self) -> None:
        event = _make_event(event_type="SessionStart")
        await dispatch_hook_event(event, None)  # type: ignore[arg-type]


class TestHandleStop:
    async def test_sends_done_notification(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.session_manager.resolve_chat_id",
                return_value=-100,
            ),
            patch(
                "ccgram.handlers.hook_events.session_manager.get_display_name",
                return_value="project",
            ),
            patch("ccgram.handlers.topic_emoji.update_topic_emoji") as mock_emoji,
            patch(
                "ccgram.handlers.message_queue.enqueue_status_update"
            ) as mock_enqueue,
            patch("ccgram.handlers.status_polling.clear_seen_status") as mock_clear,
            patch(
                "ccgram.handlers.status_polling._start_autoclose_timer"
            ) as mock_timer,
        ):
            event = _make_event(event_type="Stop", data={"stop_reason": "done"})
            await dispatch_hook_event(event, bot)

            mock_clear.assert_called_once_with("@0")
            mock_emoji.assert_called_once_with(bot, -100, 42, "done", "project")
            mock_timer.assert_called_once_with(100, 42, "done", ANY)
            mock_enqueue.assert_called_once_with(bot, 100, "@0", None, thread_id=42)

    async def test_stop_no_users_skips(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([]),
        )
        bot = AsyncMock(spec=Bot)
        with patch(
            "ccgram.handlers.message_queue.enqueue_status_update"
        ) as mock_enqueue:
            event = _make_event(event_type="Stop")
            await dispatch_hook_event(event, bot)
            mock_enqueue.assert_not_called()


class TestHandleNotification:
    async def test_renders_interactive_ui(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.interactive_ui.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.interactive_ui.set_interactive_mode",
            ) as mock_set,
            patch(
                "ccgram.handlers.interactive_ui.handle_interactive_ui",
                return_value=True,
            ) as mock_handle,
            patch("asyncio.sleep"),
        ):
            event = _make_event(
                event_type="Notification",
                data={"tool_name": "AskUserQuestion"},
            )
            await dispatch_hook_event(event, bot)

            mock_set.assert_called_once_with(100, "@0", 42)
            mock_handle.assert_called_once_with(bot, 100, "@0", 42)

    async def test_skips_when_already_interactive(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.interactive_ui.get_interactive_window",
                return_value="@0",
            ),
            patch(
                "ccgram.handlers.interactive_ui.handle_interactive_ui",
            ) as mock_handle,
        ):
            event = _make_event(event_type="Notification")
            await dispatch_hook_event(event, bot)
            mock_handle.assert_not_called()

    async def test_clears_mode_when_handle_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.interactive_ui.get_interactive_window",
                return_value=None,
            ),
            patch("ccgram.handlers.interactive_ui.set_interactive_mode"),
            patch(
                "ccgram.handlers.interactive_ui.handle_interactive_ui",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.interactive_ui.clear_interactive_mode",
            ) as mock_clear,
            patch("asyncio.sleep"),
        ):
            event = _make_event(event_type="Notification")
            await dispatch_hook_event(event, bot)
            mock_clear.assert_called_once_with(100, 42)


class TestHandleSubagentStart:
    def setup_method(self) -> None:
        _active_subagents.clear()

    async def test_tracks_new_subagent(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        bot = AsyncMock(spec=Bot)
        event = _make_event(
            event_type="SubagentStart",
            data={"subagent_id": "sub-1", "name": "researcher"},
        )
        await dispatch_hook_event(event, bot)
        assert get_subagent_count("@0") == 1

    async def test_tracks_multiple_subagents(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        bot = AsyncMock(spec=Bot)
        for sub_id in ("sub-1", "sub-2"):
            event = _make_event(
                event_type="SubagentStart", data={"subagent_id": sub_id}
            )
            await dispatch_hook_event(event, bot)
        assert get_subagent_count("@0") == 2


class TestHandleSubagentStop:
    def setup_method(self) -> None:
        _active_subagents.clear()

    async def test_removes_subagent(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        _active_subagents["@0"] = {"sub-1", "sub-2"}
        bot = AsyncMock(spec=Bot)
        event = _make_event(event_type="SubagentStop", data={"subagent_id": "sub-1"})
        await dispatch_hook_event(event, bot)
        assert get_subagent_count("@0") == 1

    async def test_removes_last_subagent_cleans_dict(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        _active_subagents["@0"] = {"sub-1"}
        bot = AsyncMock(spec=Bot)
        event = _make_event(event_type="SubagentStop", data={"subagent_id": "sub-1"})
        await dispatch_hook_event(event, bot)
        assert get_subagent_count("@0") == 0
        assert "@0" not in _active_subagents


class TestHandleTeammateIdle:
    async def test_sends_idle_notification(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        bot = AsyncMock(spec=Bot)
        with patch(
            "ccgram.handlers.message_queue.enqueue_status_update"
        ) as mock_enqueue:
            event = _make_event(
                event_type="TeammateIdle",
                data={"teammate_name": "reviewer"},
            )
            await dispatch_hook_event(event, bot)
            mock_enqueue.assert_called_once_with(
                bot,
                100,
                "@0",
                "\U0001f4a4 Teammate 'reviewer' went idle",
                thread_id=42,
            )

    async def test_unknown_teammate_name(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        bot = AsyncMock(spec=Bot)
        with patch(
            "ccgram.handlers.message_queue.enqueue_status_update"
        ) as mock_enqueue:
            event = _make_event(event_type="TeammateIdle", data={})
            await dispatch_hook_event(event, bot)
            assert "unknown" in mock_enqueue.call_args[0][3]


class TestHandleTaskCompleted:
    async def test_sends_completion_notification(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        bot = AsyncMock(spec=Bot)
        with patch(
            "ccgram.handlers.message_queue.enqueue_status_update"
        ) as mock_enqueue:
            event = _make_event(
                event_type="TaskCompleted",
                data={"task_subject": "write tests", "teammate_name": "coder"},
            )
            await dispatch_hook_event(event, bot)
            text = mock_enqueue.call_args[0][3]
            assert "\u2705 Task completed: write tests" in text
            assert "(by 'coder')" in text

    async def test_no_teammate_name(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.hook_events.session_manager.iter_thread_bindings",
            lambda: iter([(100, 42, "@0")]),
        )
        bot = AsyncMock(spec=Bot)
        with patch(
            "ccgram.handlers.message_queue.enqueue_status_update"
        ) as mock_enqueue:
            event = _make_event(
                event_type="TaskCompleted",
                data={"task_subject": "deploy"},
            )
            await dispatch_hook_event(event, bot)
            text = mock_enqueue.call_args[0][3]
            assert "\u2705 Task completed: deploy" in text
            assert "(by " not in text
