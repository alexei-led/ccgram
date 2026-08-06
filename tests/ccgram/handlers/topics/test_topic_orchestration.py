from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest, RetryAfter, TelegramError, TimedOut

from ccgram.handlers.topics.topic_orchestration import (
    collect_target_chats,
    _is_window_already_bound,
    _topic_create_retry_until,
    _window_topic_locks,
    adopt_unbound_windows,
    handle_new_window,
)
from ccgram.session_monitor import NewWindowEvent


@pytest.fixture(autouse=True)
def _clear_retry_state():
    _topic_create_retry_until.clear()
    _window_topic_locks.clear()
    yield
    _topic_create_retry_until.clear()
    _window_topic_locks.clear()


@pytest.fixture(autouse=True)
def _mock_tmux():
    mock_window = MagicMock()
    mock_window.pane_current_command = ""
    with patch("ccgram.handlers.topics.topic_orchestration.tmux_manager") as mock_tmux:
        mock_tmux.find_window_by_id = AsyncMock(return_value=mock_window)
        yield mock_tmux


def _make_event(
    window_id: str = "@10",
    session_id: str = "sess-1",
    window_name: str = "my-project",
    cwd: str = "/home/user/my-project",
) -> NewWindowEvent:
    return NewWindowEvent(
        window_id=window_id,
        session_id=session_id,
        window_name=window_name,
        cwd=cwd,
    )


def _make_topic(thread_id: int = 999) -> MagicMock:
    topic = MagicMock()
    topic.message_thread_id = thread_id
    return topic


class TestIsWindowAlreadyBound:
    def test_bound_window(self):
        with patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router"
        ) as mock_router:
            mock_router.has_window.return_value = True
            assert _is_window_already_bound("@5") is True

    def test_unbound_window(self):
        with patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router"
        ) as mock_router:
            mock_router.has_window.return_value = False
            assert _is_window_already_bound("@5") is False

    def test_no_bindings(self):
        with patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router"
        ) as mock_router:
            mock_router.has_window.return_value = False
            assert _is_window_already_bound("@0") is False


class TestCollectTargetChats:
    def test_from_bindings(self):
        with patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router"
        ) as mock_router:
            mock_router.iter_thread_bindings.return_value = [
                (1, 100, "@0"),
            ]
            mock_router.resolve_chat_id.return_value = -1001
            result = collect_target_chats("@5")
            assert result == {-1001}

    def test_fallback_to_group_chat_ids(self):
        with patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router"
        ) as mock_router:
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {1: -2002}
            result = collect_target_chats("@5")
            assert result == {-2002}

    def test_fallback_to_config_group_id(self):
        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_router,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {}
            mock_config.group_id = -3003
            result = collect_target_chats("@5")
            assert result == {-3003}

    def test_no_chats_available(self):
        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_router,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {}
            mock_config.group_id = None
            result = collect_target_chats("@5")
            assert result == set()

    def test_skips_positive_ids(self):
        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_router,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_router.iter_thread_bindings.return_value = []
            mock_router.group_chat_ids = {"100:5": 100}
            mock_config.group_id = None
            result = collect_target_chats("@5")
            assert result == set()


class TestHandleNewWindow:
    async def test_serializes_concurrent_creation_for_same_window(self) -> None:
        active = 0
        max_active = 0

        async def locked_handler(
            _event: NewWindowEvent, _client: AsyncMock, **_kwargs: object
        ) -> bool:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)
            active -= 1
            return True

        event = _make_event()
        with patch(
            "ccgram.handlers.topics.topic_orchestration._handle_new_window_locked",
            side_effect=locked_handler,
        ):
            assert await asyncio.gather(
                handle_new_window(event, AsyncMock()),
                handle_new_window(event, AsyncMock()),
            ) == [True, True]

        assert max_active == 1
        assert _window_topic_locks == {}

    async def test_targeted_creation_ignores_another_users_binding(self) -> None:
        event = _make_event(window_id="@2", window_name="proj")
        bot = AsyncMock()
        bot.create_forum_topic.return_value = _make_topic(thread_id=77)

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = True
            mock_tr.iter_thread_bindings.return_value = iter([(200, 2, "@2")])
            mock_config.allowed_users = {100, 200}
            created = await handle_new_window(
                event,
                bot,
                target_user_id=100,
                target_chat_id=-100100,
            )

        assert created is True
        mock_tr.bind_thread.assert_called_once_with(
            100, 77, "@2", window_name="proj", chat_id=-100100
        )
        mock_tr.set_group_chat_id.assert_called_once_with(100, 77, -100100)

    async def test_skips_already_bound(self):
        event = NewWindowEvent(
            window_id="@0", session_id="s1", window_name="test", cwd="/tmp"
        )
        bot = AsyncMock()
        with patch(
            "ccgram.handlers.topics.topic_orchestration._is_window_already_bound",
            return_value=True,
        ):
            await handle_new_window(event, bot)
        bot.create_forum_topic.assert_not_called()

    async def test_creates_topic(self):
        event = NewWindowEvent(
            window_id="@5", session_id="s2", window_name="myproject", cwd="/tmp"
        )
        bot = AsyncMock()
        topic = MagicMock()
        topic.message_thread_id = 999
        bot.create_forum_topic.return_value = topic

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration._is_window_already_bound",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration._auto_detect_provider",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration.collect_target_chats",
                return_value={-1001},
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_router,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_router.iter_thread_bindings.return_value = [(1, 100, "@0")]
            mock_router.resolve_chat_id.return_value = -1001
            mock_config.allowed_users = set()
            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(chat_id=-1001, name="myproject")

    async def test_skips_when_no_chats(self):
        event = NewWindowEvent(
            window_id="@5", session_id="s2", window_name="test", cwd="/tmp"
        )
        bot = AsyncMock()

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration._is_window_already_bound",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration._auto_detect_provider",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration.collect_target_chats",
                return_value=set(),
            ),
        ):
            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_not_called()

    async def test_rate_limit_backoff(self):
        event = NewWindowEvent(
            window_id="@5", session_id="s2", window_name="test", cwd="/tmp"
        )
        bot = AsyncMock()
        _topic_create_retry_until[-1001] = time.monotonic() + 60

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration._is_window_already_bound",
                return_value=False,
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration._auto_detect_provider",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration.collect_target_chats",
                return_value={-1001},
            ),
        ):
            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_not_called()

    async def test_creates_topic_with_group_id(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=42))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100500, name="my-project"
        )

    async def test_binds_first_allowed_user(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=42))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_tr.resolve_chat_id.return_value = 12345
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

        mock_tr.bind_thread.assert_called_once_with(
            12345, 42, "@10", window_name="my-project", chat_id=-100500
        )
        mock_tr.set_group_chat_id.assert_called_once_with(12345, 42, -100500)

    async def test_creates_topic_from_bindings(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=77))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config"),
        ):
            bindings = [(100, 5, "@1")]
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.side_effect = [
                iter(bindings),
                iter(bindings),
                iter(bindings),
            ]
            mock_tr.resolve_chat_id.return_value = -100200

            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100200, name="my-project"
        )

    async def test_binds_existing_user(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=77))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config"),
        ):
            bindings = [(100, 5, "@1")]
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.side_effect = [
                iter(bindings),
                iter(bindings),
                iter(bindings),
            ]
            mock_tr.resolve_chat_id.return_value = -100200

            await handle_new_window(event, bot)

        mock_tr.bind_thread.assert_called_once_with(
            100, 77, "@10", window_name="my-project", chat_id=-100200
        )
        mock_tr.set_group_chat_id.assert_called_once_with(100, 77, -100200)

    async def test_topic_name_falls_back_to_cwd_dirname(self) -> None:
        event = _make_event(window_name="", cwd="/home/user/cool-project")
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=42))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100500, name="cool-project"
        )

    async def test_telegram_error_logged_not_raised(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(side_effect=TelegramError("API error"))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

    async def test_retry_after_sets_backoff_and_skips_immediate_retry(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(side_effect=RetryAfter(27))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
            patch(
                "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
                {},
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration.time.monotonic",
                side_effect=[100.0, 100.0, 101.0],
            ),
        ):
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.side_effect = [iter([]) for _ in range(6)]
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)
            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100500, name="my-project"
        )

    async def test_retries_after_backoff_expires(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(
            side_effect=[RetryAfter(3), _make_topic(thread_id=42)]
        )

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
            patch(
                "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
                {},
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration.time.monotonic",
                side_effect=[100.0, 100.0, 106.0],
            ),
        ):
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.side_effect = [iter([]) for _ in range(6)]
            mock_tr.resolve_chat_id.return_value = 12345
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)
            await handle_new_window(event, bot)

        assert bot.create_forum_topic.call_count == 2
        mock_tr.bind_thread.assert_called_once_with(
            12345, 42, "@10", window_name="my-project", chat_id=-100500
        )

    async def test_uses_group_chat_ids_when_no_bindings(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=42))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_tr.group_chat_ids = {"100:5": -100200}
            mock_config.group_id = None
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100200, name="my-project"
        )

    async def test_rebinds_existing_same_name_topic_for_dead_old_window(self) -> None:
        event = _make_event(window_id="@3", window_name="reflex-gh")
        bot = AsyncMock()
        probe_msg = MagicMock()
        probe_msg.message_id = 555
        bot.send_message = AsyncMock(return_value=probe_msg)
        bot.delete_message = AsyncMock()

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch(
                "ccgram.handlers.topics.topic_orchestration.tmux_manager"
            ) as mock_tmux,
        ):
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.return_value = iter([(100, 120014, "@1")])
            mock_tr.get_display_name.return_value = "🟡 reflex-gh"
            mock_tr.resolve_chat_id.return_value = -100200
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)

            await handle_new_window(event, bot)

        mock_tr.bind_thread.assert_called_once_with(
            100, 120014, "@3", window_name="reflex-gh", chat_id=-100200
        )
        mock_tr.set_group_chat_id.assert_called_once_with(100, 120014, -100200)
        bot.send_message.assert_awaited_once_with(
            -100200,
            ".",
            message_thread_id=120014,
            disable_notification=True,
        )
        bot.delete_message.assert_awaited_once_with(-100200, 555)
        bot.create_forum_topic.assert_not_called()

    async def test_dead_same_name_topic_is_unbound_then_new_topic_created(self) -> None:
        event = _make_event(window_id="@3", window_name="reflex-gh")
        bot = AsyncMock()
        bot.send_message = AsyncMock(side_effect=BadRequest("Topic_id_invalid"))
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=77))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch(
                "ccgram.handlers.topics.topic_orchestration.tmux_manager"
            ) as mock_tmux,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
        ):
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.side_effect = [
                iter([(100, 120014, "@1")]),
                iter([]),
                iter([]),
            ]
            mock_tr.get_display_name.return_value = "reflex-gh"
            mock_tr.resolve_chat_id.return_value = -100200
            mock_tmux.find_window_by_id = AsyncMock(return_value=None)
            mock_tr.group_chat_ids = {"100:120014": -100200}
            mock_config.group_id = None
            mock_config.allowed_users = {100}

            await handle_new_window(event, bot)

        mock_tr.unbind_thread.assert_called_once_with(100, 120014)
        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100200, name="reflex-gh"
        )


class TestCreateForumTopicTransientRetry:
    async def test_timed_out_retries_then_succeeds(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(
            side_effect=[TimedOut("blip"), _make_topic(thread_id=42)]
        )

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
            patch(
                "ccgram.handlers.topics.topic_orchestration.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

        assert bot.create_forum_topic.call_count == 2

    async def test_timed_out_exhausts_retries_then_logs(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(side_effect=TimedOut("persistent"))

        with (
            patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
            patch(
                "ccgram.handlers.topics.topic_orchestration.thread_router"
            ) as mock_tr,
            patch("ccgram.handlers.topics.topic_orchestration.config") as mock_config,
            patch(
                "ccgram.handlers.topics.topic_orchestration.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_tr.has_window.return_value = False
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await handle_new_window(event, bot)

        # Original attempt + 1 retry = 2 calls
        assert bot.create_forum_topic.call_count == 2


class TestAdoptUnboundWindows:
    async def test_adopts_orphaned_windows(self):
        bot = AsyncMock()
        mock_window = MagicMock()
        mock_window.window_id = "@0"
        mock_window.window_name = "test"

        mock_audit = MagicMock()
        mock_issue = MagicMock()
        mock_issue.category = "orphaned_window"
        mock_audit.issues = [mock_issue]

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager"
            ) as mock_sm,
            patch(
                "ccgram.handlers.topics.topic_orchestration._adopt_orphaned_windows",
                new_callable=AsyncMock,
                create=True,
            ),
        ):
            mock_tmux.list_windows = AsyncMock(return_value=[mock_window])
            mock_sm.audit_state.return_value = mock_audit

            with patch(
                "ccgram.handlers.sync_command._adopt_orphaned_windows",
                new_callable=AsyncMock,
            ) as mock_adopt:
                await adopt_unbound_windows(bot)
                mock_adopt.assert_called_once()

    async def test_no_orphans_skips(self):
        bot = AsyncMock()
        mock_audit = MagicMock()
        mock_audit.issues = []

        with (
            patch(
                "ccgram.handlers.topics.topic_orchestration.tmux_manager"
            ) as mock_tmux,
            patch(
                "ccgram.handlers.topics.topic_orchestration.session_manager"
            ) as mock_sm,
        ):
            mock_tmux.list_windows = AsyncMock(return_value=[])
            mock_sm.audit_state.return_value = mock_audit
            await adopt_unbound_windows(bot)
