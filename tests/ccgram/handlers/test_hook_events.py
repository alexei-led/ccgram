from collections.abc import Callable, Iterator
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from telegram import Bot

from ccgram.claude_task_state import claude_task_state
from ccgram.config import config
from ccgram.handlers.callback_data import IDLE_STATUS_TEXT

from ccgram.claude_task_state import (
    _active_subagents,
    build_subagent_label,
    clear_subagents,
    get_subagent_names,
)
from ccgram.handlers.hook_events import (
    HookEvent,
    _resolve_users_for_window_key,
    dispatch_hook_event,
)


_TR_ITER = "ccgram.handlers.hook_events.thread_router.iter_thread_bindings"
_DEFAULT_BINDING = (100, 42, "@0")


@pytest.fixture(autouse=True)
def bindings(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Bind one topic to window @0 by default; call to rebind or unbind."""

    def _set(*rows: tuple[int, int, str]) -> None:
        monkeypatch.setattr(_TR_ITER, lambda: iter(rows))

    _set(_DEFAULT_BINDING)
    return _set


@pytest.fixture(autouse=True)
def _clean_subagents() -> Iterator[None]:
    _active_subagents.clear()
    yield
    _active_subagents.clear()


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
    def test_matches_only_bindings_for_that_window(self, bindings) -> None:
        bindings((111, 42, "@0"), (222, 99, "@5"))

        assert _resolve_users_for_window_key("ccgram:@0") == [(111, 42, "@0")]

    def test_herdr_window_id_keeps_its_own_colon(self, bindings, monkeypatch) -> None:
        """An opaque target can contain colons after its backend prefix."""
        monkeypatch.setattr(config, "multiplexer_name", "herdr")
        bindings((111, 42, "opaque:target:id"))

        assert _resolve_users_for_window_key("herdr:opaque:target:id") == [
            (111, 42, "opaque:target:id")
        ]

    @pytest.mark.parametrize(
        "window_key",
        [
            pytest.param("other-session:@0", id="wrong-tmux-session"),
            pytest.param("herdr:@0", id="wrong-backend"),
        ],
    )
    def test_rejects_wrong_backend_or_session_prefix(
        self, bindings, monkeypatch, window_key: str
    ) -> None:
        monkeypatch.setattr(config, "multiplexer_name", "tmux")
        monkeypatch.setattr(config, "tmux_session_name", "ccgram")
        bindings((111, 42, "@0"))

        assert _resolve_users_for_window_key(window_key) == []

    @pytest.mark.parametrize(
        "window_key",
        [
            pytest.param("ccgram:@99", id="unbound-window"),
            pytest.param("nocolon", id="malformed-key"),
        ],
    )
    def test_returns_no_users(self, bindings, window_key: str) -> None:
        bindings()

        assert _resolve_users_for_window_key(window_key) == []


class TestSubagentTracking:
    def test_get_names_returns_tracked_labels(self) -> None:
        _active_subagents["@0"] = {"a1": "write-tests", "a2": "refactor"}
        assert sorted(get_subagent_names("@0")) == ["refactor", "write-tests"]

    def test_clear_removes_all(self) -> None:
        _active_subagents["@0"] = {"a1": "agent-1", "a2": "agent-2"}
        clear_subagents("@0")
        assert get_subagent_names("@0") == []

    def test_names_missing_window(self) -> None:
        assert get_subagent_names("@999") == []


class TestBuildSubagentLabel:
    @pytest.mark.parametrize(
        ("names", "expected"),
        [
            pytest.param([], None, id="empty"),
            pytest.param(["write-tests"], "🤖 write-tests", id="single-name-verbatim"),
            pytest.param(
                ["write-tests", "refactor"],
                "🤖 2 subagents: write-tests, refactor",
                id="two-names",
            ),
            pytest.param(["a", "b", "c"], "🤖 3 subagents: a, b, c", id="three-names"),
            pytest.param(
                ["a", "b", "c", "d"], "🤖 4 subagents: a, b, c", id="truncates-at-three"
            ),
        ],
    )
    def test_label(self, names: list[str], expected: str | None) -> None:
        assert build_subagent_label(names) == expected


class TestDispatchHookEvent:
    async def test_unknown_event_ignored(self) -> None:
        event = _make_event(event_type="SomeUnknownEvent")
        await dispatch_hook_event(event, None)  # type: ignore[arg-type]

    async def test_session_start_ignored(self) -> None:
        event = _make_event(event_type="SessionStart")
        await dispatch_hook_event(event, None)  # type: ignore[arg-type]


class TestHandleStop:
    async def test_updates_status_without_touching_topic_emoji(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch("ccgram.handlers.hook_events.update_topic_emoji") as mock_emoji,
            patch("ccgram.handlers.hook_events.enqueue_status_update") as mock_enqueue,
        ):
            event = _make_event(event_type="Stop", data={"stop_reason": "done"})
            await dispatch_hook_event(event, bot)

            mock_emoji.assert_not_called()
            mock_enqueue.assert_called_once()
            status_text = mock_enqueue.call_args[0][3]
            assert status_text is not None
            assert IDLE_STATUS_TEXT in status_text

    async def test_stop_no_users_skips(self, bindings) -> None:
        bindings()
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.hook_events.enqueue_status_update") as mock_enqueue:
            event = _make_event(event_type="Stop")
            await dispatch_hook_event(event, bot)
            mock_enqueue.assert_not_called()

    async def test_stop_wrong_prefix_has_no_users_and_skips_dispatch(
        self, bindings, monkeypatch
    ) -> None:
        monkeypatch.setattr(config, "multiplexer_name", "tmux")
        monkeypatch.setattr(config, "tmux_session_name", "ccgram")
        bindings((100, 42, "@0"))
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.hook_events.enqueue_status_update") as mock_enqueue:
            event = _make_event(event_type="Stop", window_key="other-session:@0")
            await dispatch_hook_event(event, bot)
            mock_enqueue.assert_not_called()


class TestEnhanceWithLlmSummary:
    async def test_enhances_ready_with_summary(self, bindings) -> None:
        mock_state = MagicMock()
        mock_state.transcript_path = "/tmp/transcript.jsonl"
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.view_window",
                return_value=mock_state,
            ),
            patch("ccgram.handlers.hook_events.enqueue_status_update") as mock_enqueue,
            patch(
                "ccgram.llm.summarizer.summarize_completion",
                new_callable=AsyncMock,
                return_value="Fixed auth bug, all 5 tests pass",
            ),
        ):
            event = _make_event(
                event_type="Stop",
                data={"stop_reason": "done", "num_turns": 3},
            )
            await dispatch_hook_event(event, bot)

            calls = mock_enqueue.call_args_list
            assert len(calls) == 1
            status_text = calls[0][0][3]
            assert "Done" in status_text
            assert "Fixed auth bug" in status_text

    async def test_no_enhancement_when_no_llm(self, bindings) -> None:
        mock_state = MagicMock()
        mock_state.transcript_path = "/tmp/transcript.jsonl"
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.view_window",
                return_value=mock_state,
            ),
            patch("ccgram.handlers.hook_events.enqueue_status_update") as mock_enqueue,
            patch(
                "ccgram.llm.summarizer.summarize_completion",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            event = _make_event(
                event_type="Stop",
                data={"stop_reason": "done", "num_turns": 3},
            )
            await dispatch_hook_event(event, bot)

            import asyncio

            await asyncio.sleep(0.1)

            assert mock_enqueue.call_count == 1

    async def test_enhancement_error_is_silent(self, bindings) -> None:
        mock_state = MagicMock()
        mock_state.transcript_path = "/tmp/transcript.jsonl"
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.view_window",
                return_value=mock_state,
            ),
            patch("ccgram.handlers.hook_events.enqueue_status_update"),
            patch(
                "ccgram.llm.summarizer.summarize_completion",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API down"),
            ),
        ):
            event = _make_event(
                event_type="Stop",
                data={"stop_reason": "done", "num_turns": 3},
            )
            await dispatch_hook_event(event, bot)

            import asyncio

            await asyncio.sleep(0.1)


class TestHandleNotification:
    async def test_renders_interactive_ui(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.get_interactive_window",
                return_value=None,
            ),
            patch(
                "ccgram.handlers.hook_events.set_interactive_mode",
            ) as mock_set,
            patch(
                "ccgram.handlers.hook_events.handle_interactive_ui",
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
            mock_handle.assert_called_once()
            assert mock_handle.call_args.args[0] is bot
            assert mock_handle.call_args.args[1:] == (100, "@0", 42)

    async def test_skips_when_already_interactive(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.get_interactive_window",
                return_value="@0",
            ),
            patch(
                "ccgram.handlers.hook_events.handle_interactive_ui",
            ) as mock_handle,
        ):
            event = _make_event(event_type="Notification")
            await dispatch_hook_event(event, bot)
            mock_handle.assert_not_called()

    async def test_clears_mode_when_handle_fails(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.get_interactive_window",
                return_value=None,
            ),
            patch("ccgram.handlers.hook_events.set_interactive_mode"),
            patch(
                "ccgram.handlers.hook_events.handle_interactive_ui",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.hook_events.clear_interactive_mode",
            ) as mock_clear,
            patch("asyncio.sleep"),
        ):
            event = _make_event(event_type="Notification")
            await dispatch_hook_event(event, bot)
            mock_clear.assert_called_once_with(100, 42)

    async def test_sets_wait_header_from_notification_message(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.get_interactive_window",
                return_value=None,
            ),
            patch("ccgram.handlers.hook_events.set_interactive_mode"),
            patch(
                "ccgram.handlers.hook_events.handle_interactive_ui",
                return_value=False,
            ),
            patch("ccgram.handlers.hook_events.clear_interactive_mode"),
            patch(
                "ccgram.handlers.hook_events.enqueue_status_update",
                new_callable=AsyncMock,
            ) as mock_enqueue,
            patch("asyncio.sleep"),
        ):
            event = _make_event(
                event_type="Notification",
                data={"message": "Claude needs your permission to use Bash"},
            )
            await dispatch_hook_event(event, bot)

            assert claude_task_state.get_wait_header("@0") == "Approval needed: Bash"
            mock_enqueue.assert_awaited_once_with(ANY, 100, "@0", None, thread_id=42)


class TestHandleSubagentStart:
    @pytest.mark.parametrize(
        ("data", "expected_name"),
        [
            pytest.param(
                {"subagent_id": "sub-1", "name": "researcher"},
                "researcher",
                id="explicit-name",
            ),
            pytest.param(
                {"subagent_id": "sub-1", "description": "explore code"},
                "explore code",
                id="falls-back-to-description",
            ),
            pytest.param(
                {"subagent_id": "sub-1", "name": "   ", "description": "real"},
                "real",
                id="blank-name-falls-back-to-description",
            ),
            pytest.param(
                {"subagent_id": "abcdef123456789"},
                "abcdef123456",
                id="falls-back-to-truncated-id",
            ),
            pytest.param(
                {"subagent_id": "", "name": "", "description": ""},
                "subagent",
                id="nothing-usable-falls-back-to-literal",
            ),
        ],
    )
    async def test_tracked_name_resolution(
        self, bindings, data: dict, expected_name: str
    ) -> None:
        with patch("ccgram.handlers.hook_events.enqueue_status_update"):
            await dispatch_hook_event(
                _make_event(event_type="SubagentStart", data=data),
                AsyncMock(spec=Bot),
            )

        assert get_subagent_names("@0") == [expected_name]

    async def test_tracks_multiple_subagents(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        for sub_id in ("sub-1", "sub-2"):
            event = _make_event(
                event_type="SubagentStart", data={"subagent_id": sub_id}
            )
            await dispatch_hook_event(event, bot)

        assert len(get_subagent_names("@0")) == 2

    async def test_tracked_once_across_multiple_user_bindings(self, bindings) -> None:
        bindings((100, 42, "@0"), (200, 99, "@0"))
        event = _make_event(
            event_type="SubagentStart",
            data={"subagent_id": "sub-1", "name": "researcher"},
        )

        await dispatch_hook_event(event, AsyncMock(spec=Bot))

        assert get_subagent_names("@0") == ["researcher"]

    async def test_unbound_window_does_not_track(self, bindings) -> None:
        bindings()
        event = _make_event(
            event_type="SubagentStart",
            data={"subagent_id": "sub-1", "name": "test"},
        )

        await dispatch_hook_event(event, AsyncMock(spec=Bot))

        assert _active_subagents == {}


class TestHandleSubagentStop:
    async def test_removes_subagent(self, bindings) -> None:
        _active_subagents["@0"] = {"sub-1": "agent-1", "sub-2": "agent-2"}
        bot = AsyncMock(spec=Bot)
        event = _make_event(event_type="SubagentStop", data={"subagent_id": "sub-1"})
        await dispatch_hook_event(event, bot)
        assert len(get_subagent_names("@0")) == 1

    async def test_removes_last_subagent_cleans_dict(self, bindings) -> None:
        _active_subagents["@0"] = {"sub-1": "agent-1"}
        bot = AsyncMock(spec=Bot)
        event = _make_event(event_type="SubagentStop", data={"subagent_id": "sub-1"})
        await dispatch_hook_event(event, bot)
        assert get_subagent_names("@0") == []
        assert "@0" not in _active_subagents

    async def test_unknown_id_is_noop(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        event = _make_event(
            event_type="SubagentStop", data={"subagent_id": "never-seen"}
        )
        await dispatch_hook_event(event, bot)
        assert get_subagent_names("@0") == []


class TestHandleTeammateIdle:
    async def test_sends_idle_notification(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.hook_events.enqueue_status_update") as mock_enqueue:
            event = _make_event(
                event_type="TeammateIdle",
                data={"teammate_name": "reviewer"},
            )
            await dispatch_hook_event(event, bot)
            mock_enqueue.assert_called_once_with(
                ANY,
                100,
                "@0",
                "\U0001f4a4 Teammate 'reviewer' went idle",
                thread_id=42,
            )

    async def test_unknown_teammate_name(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.hook_events.enqueue_status_update") as mock_enqueue:
            event = _make_event(event_type="TeammateIdle", data={})
            await dispatch_hook_event(event, bot)
            assert "unknown" in mock_enqueue.call_args[0][3]


class TestHandleTaskCompleted:
    async def test_sends_completion_notification(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.hook_events.enqueue_status_update") as mock_enqueue:
            event = _make_event(
                event_type="TaskCompleted",
                data={"task_subject": "write tests", "teammate_name": "coder"},
            )
            await dispatch_hook_event(event, bot)
            text = mock_enqueue.call_args[0][3]
            assert "\u2705 Task completed: write tests" in text
            assert "(by 'coder')" in text

    async def test_no_teammate_name(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.hook_events.enqueue_status_update") as mock_enqueue:
            event = _make_event(
                event_type="TaskCompleted",
                data={"task_subject": "deploy"},
            )
            await dispatch_hook_event(event, bot)
            text = mock_enqueue.call_args[0][3]
            assert "\u2705 Task completed: deploy" in text
            assert "(by " not in text

    async def test_tracked_task_refreshes_task_status(self, bindings) -> None:
        claude_task_state.apply_entries(
            "@0",
            "session-1",
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": "TaskCreate",
                                "input": {
                                    "subject": "Write tests",
                                    "description": "",
                                    "activeForm": "",
                                },
                            }
                        ]
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-1",
                                "content": "Task #1 created successfully",
                            }
                        ]
                    },
                    "toolUseResult": {"task": {"id": "1", "subject": "Write tests"}},
                },
            ],
        )
        bot = AsyncMock(spec=Bot)
        with patch(
            "ccgram.handlers.hook_events.enqueue_status_update",
            new_callable=AsyncMock,
        ) as mock_enqueue:
            event = _make_event(
                event_type="TaskCompleted",
                session_id="session-1",
                data={"task_id": "1", "task_subject": "Write tests"},
            )
            await dispatch_hook_event(event, bot)

            snapshot = claude_task_state.get_snapshot("@0")
            assert snapshot is not None
            assert snapshot.done_count == 1
            mock_enqueue.assert_awaited_once_with(ANY, 100, "@0", None, thread_id=42)


class TestHandleStopFailure:
    async def test_sends_error_alert(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.thread_router.resolve_chat_id",
                return_value=-100,
            ),
            patch(
                "ccgram.handlers.messaging_pipeline.message_sender.rate_limit_send_message"
            ) as mock_send,
        ):
            event = _make_event(
                event_type="StopFailure",
                data={"error": "rate_limit", "error_details": "429 Too Many Requests"},
            )
            await dispatch_hook_event(event, bot)
            text = mock_send.call_args[0][2]
            assert "rate_limit" in text
            assert "429" in text

    async def test_no_users_skips(self, bindings) -> None:
        bindings()
        bot = AsyncMock(spec=Bot)
        with patch(
            "ccgram.handlers.messaging_pipeline.message_sender.rate_limit_send_message"
        ) as mock_send:
            event = _make_event(event_type="StopFailure", data={"error": "unknown"})
            await dispatch_hook_event(event, bot)
            mock_send.assert_not_called()


class TestHandleSessionEnd:
    async def test_transitions_to_done(self, bindings) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.thread_router.resolve_chat_id",
                return_value=-100,
            ),
            patch(
                "ccgram.handlers.hook_events.thread_router.get_display_name",
                return_value="project",
            ),
            patch(
                "ccgram.session_lifecycle.window_store.clear_window_session",
            ) as mock_clear_session,
            patch("ccgram.handlers.hook_events.update_topic_emoji") as mock_emoji,
            patch("ccgram.handlers.hook_events.enqueue_status_update") as mock_enqueue,
            patch(
                "ccgram.handlers.polling.polling_state.terminal_poll_state.clear_seen_status"
            ) as mock_clear,
        ):
            event = _make_event(event_type="SessionEnd", data={"reason": "clear"})
            await dispatch_hook_event(event, bot)

            mock_clear.assert_called_once_with("@0")
            mock_emoji.assert_called_once_with(ANY, -100, 42, "done", "project")
            mock_enqueue.assert_called_once_with(ANY, 100, "@0", None, thread_id=42)
            mock_clear_session.assert_called_once_with("@0")

    async def test_clears_claude_task_state(self, bindings) -> None:
        claude_task_state.apply_entries(
            "@0",
            "session-1",
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "todo-1",
                                "name": "TodoWrite",
                                "input": {
                                    "todos": [
                                        {
                                            "content": "Review changes",
                                            "status": "completed",
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                }
            ],
        )
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.thread_router.resolve_chat_id",
                return_value=-100,
            ),
            patch(
                "ccgram.handlers.hook_events.thread_router.get_display_name",
                return_value="project",
            ),
            patch("ccgram.session_lifecycle.window_store.clear_window_session"),
            patch("ccgram.handlers.hook_events.update_topic_emoji"),
            patch("ccgram.handlers.hook_events.enqueue_status_update"),
            patch(
                "ccgram.handlers.polling.polling_state.terminal_poll_state.clear_seen_status"
            ),
        ):
            event = _make_event(event_type="SessionEnd", data={"reason": "clear"})
            await dispatch_hook_event(event, bot)

        assert claude_task_state.get_snapshot("@0") is None

    async def test_clears_subagents_on_session_end(self, bindings) -> None:
        _active_subagents["@0"] = {"sub-1": "researcher"}
        bot = AsyncMock(spec=Bot)
        with (
            patch(
                "ccgram.handlers.hook_events.thread_router.resolve_chat_id",
                return_value=-100,
            ),
            patch(
                "ccgram.handlers.hook_events.thread_router.get_display_name",
                return_value="project",
            ),
            patch(
                "ccgram.session_lifecycle.window_store.clear_window_session",
            ),
            patch("ccgram.handlers.hook_events.update_topic_emoji"),
            patch("ccgram.handlers.hook_events.enqueue_status_update"),
            patch(
                "ccgram.handlers.polling.polling_state.terminal_poll_state.clear_seen_status"
            ),
        ):
            event = _make_event(event_type="SessionEnd", data={"reason": "clear"})
            await dispatch_hook_event(event, bot)
            assert get_subagent_names("@0") == []

    async def test_no_users_skips(self, bindings) -> None:
        bindings()
        bot = AsyncMock(spec=Bot)
        with patch("ccgram.handlers.hook_events.enqueue_status_update") as mock_enqueue:
            event = _make_event(event_type="SessionEnd", data={"reason": "logout"})
            await dispatch_hook_event(event, bot)
            mock_enqueue.assert_not_called()
