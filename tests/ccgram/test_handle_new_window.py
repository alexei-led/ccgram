"""Tests for _handle_new_window auto-topic creation (TASK-032).

Covers cold-start with CCGRAM_GROUP_ID, cold-start without it,
normal flow with existing bindings, already-bound window skip,
and RetryAfter backoff behavior.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import RetryAfter, TelegramError

from ccgram.handlers.topic_orchestration import handle_new_window as _handle_new_window
from ccgram.session_monitor import NewWindowEvent


@pytest.fixture(autouse=True)
def _mock_tmux():
    """Mock tmux_manager.find_window_by_id for all tests in this module."""
    mock_window = MagicMock()
    mock_window.pane_current_command = ""
    with patch("ccgram.handlers.topic_orchestration.tmux_manager") as mock_tmux:
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


class TestHandleNewWindowColdStart:
    """Cold-start: no existing bindings, CCGRAM_GROUP_ID is the only source."""

    async def test_creates_topic_with_group_id(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=42))

        with (
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
        ):
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await _handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100500, name="my-project"
        )

    async def test_binds_first_allowed_user(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=42))

        with (
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
        ):
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_tr.resolve_chat_id.return_value = 12345
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await _handle_new_window(event, bot)

        mock_tr.bind_thread.assert_called_once_with(
            12345, 42, "@10", window_name="my-project"
        )
        mock_tr.set_group_chat_id.assert_called_once_with(12345, 42, -100500)

    async def test_skips_without_group_id(self) -> None:
        event = _make_event()
        bot = AsyncMock()

        with (
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
        ):
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = None

            await _handle_new_window(event, bot)

        bot.create_forum_topic.assert_not_called()


class TestHandleNewWindowNormalFlow:
    """Normal flow: existing bindings provide the target chat."""

    async def test_creates_topic_from_bindings(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=77))

        with (
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config"),
        ):
            bindings = [(100, 5, "@1")]
            mock_tr.iter_thread_bindings.side_effect = [
                iter(bindings),
                iter(bindings),
                iter(bindings),
            ]
            mock_tr.resolve_chat_id.return_value = -100200

            await _handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100200, name="my-project"
        )

    async def test_binds_existing_user(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=77))

        with (
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config"),
        ):
            bindings = [(100, 5, "@1")]
            # iter_thread_bindings called multiple times: once for already-bound check,
            # once for collecting chats, once for binding loop
            mock_tr.iter_thread_bindings.side_effect = [
                iter(bindings),
                iter(bindings),
                iter(bindings),
            ]
            mock_tr.resolve_chat_id.return_value = -100200

            await _handle_new_window(event, bot)

        mock_tr.bind_thread.assert_called_once_with(
            100, 77, "@10", window_name="my-project"
        )
        mock_tr.set_group_chat_id.assert_called_once_with(100, 77, -100200)


class TestHandleNewWindowAlreadyBound:
    """Window already bound to a topic - should skip."""

    async def test_skips_already_bound_window(self) -> None:
        event = _make_event(window_id="@10")
        bot = AsyncMock()

        with (
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config"),
        ):
            mock_tr.iter_thread_bindings.return_value = iter([(100, 5, "@10")])

            await _handle_new_window(event, bot)

        bot.create_forum_topic.assert_not_called()


class TestHandleNewWindowErrors:
    """Error handling during topic creation."""

    async def test_telegram_error_logged_not_raised(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(side_effect=TelegramError("API error"))

        with (
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
        ):
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await _handle_new_window(event, bot)

    async def test_retry_after_sets_backoff_and_skips_immediate_retry(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(side_effect=RetryAfter(27))

        with (
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
            patch("ccgram.handlers.topic_orchestration._topic_create_retry_until", {}),
            patch(
                "ccgram.handlers.topic_orchestration.time.monotonic",
                side_effect=[100.0, 100.0, 101.0],
            ),
        ):
            mock_tr.iter_thread_bindings.side_effect = [
                iter([]),
                iter([]),
                iter([]),
                iter([]),
            ]
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await _handle_new_window(event, bot)
            await _handle_new_window(event, bot)

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
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
            patch("ccgram.handlers.topic_orchestration._topic_create_retry_until", {}),
            patch(
                "ccgram.handlers.topic_orchestration.time.monotonic",
                side_effect=[100.0, 100.0, 106.0],
            ),
        ):
            mock_tr.iter_thread_bindings.side_effect = [
                iter([]),
                iter([]),
                iter([]),
                iter([]),
                iter([]),
            ]
            mock_tr.resolve_chat_id.return_value = 12345
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await _handle_new_window(event, bot)
            await _handle_new_window(event, bot)

        assert bot.create_forum_topic.call_count == 2
        mock_tr.bind_thread.assert_called_once_with(
            12345, 42, "@10", window_name="my-project"
        )

    async def test_topic_name_falls_back_to_cwd_dirname(self) -> None:
        event = _make_event(window_name="", cwd="/home/user/cool-project")
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=42))

        with (
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
        ):
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_config.group_id = -100500
            mock_config.allowed_users = {12345}

            await _handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100500, name="cool-project"
        )


class TestHandleNewWindowGroupChatIdsFallback:
    """Post-restart: no bindings left, but group_chat_ids preserved."""

    async def test_uses_group_chat_ids_when_no_bindings(self) -> None:
        event = _make_event()
        bot = AsyncMock()
        bot.create_forum_topic = AsyncMock(return_value=_make_topic(thread_id=42))

        with (
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
        ):
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_tr.group_chat_ids = {"100:5": -100200}
            mock_config.group_id = None
            mock_config.allowed_users = {12345}

            await _handle_new_window(event, bot)

        bot.create_forum_topic.assert_called_once_with(
            chat_id=-100200, name="my-project"
        )

    async def test_skips_positive_ids(self) -> None:
        event = _make_event()
        bot = AsyncMock()

        with (
            patch("ccgram.handlers.topic_orchestration.session_manager"),
            patch("ccgram.handlers.topic_orchestration.thread_router") as mock_tr,
            patch("ccgram.handlers.topic_orchestration.config") as mock_config,
        ):
            mock_tr.iter_thread_bindings.return_value = iter([])
            mock_tr.group_chat_ids = {"100:5": 100}  # DM fallback (positive)
            mock_config.group_id = None

            await _handle_new_window(event, bot)

        bot.create_forum_topic.assert_not_called()
