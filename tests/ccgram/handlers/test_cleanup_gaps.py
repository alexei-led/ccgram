import asyncio

from ccgram.handlers.interactive.interactive_ui import (
    _send_cooldowns,
    clear_send_cooldowns,
)
from ccgram.handlers.text.text_handler import (
    _bash_capture_tasks,
    cancel_bash_capture,
)
from ccgram.handlers.status.topic_emoji import (
    _MAX_DISABLED_CHATS,
    _disabled_chats,
    clear_disabled_chat,
)
from ccgram.handlers.topics.topic_orchestration import (
    _topic_create_retry_until,
    clear_topic_create_retry,
)


class TestCancelBashCapture:
    def test_clears_existing_task(self) -> None:
        async def _noop() -> None: ...

        task = asyncio.ensure_future(_noop())
        _bash_capture_tasks[(1, 42)] = task
        cancel_bash_capture(1, 42)
        assert (1, 42) not in _bash_capture_tasks

    def test_missing_key_no_error(self) -> None:
        _bash_capture_tasks.clear()
        cancel_bash_capture(999, 999)


class TestClearSendCooldowns:
    def test_clears_existing_cooldown(self) -> None:
        _send_cooldowns[(1, -100, 42)] = 123.0
        clear_send_cooldowns(1, 42)
        assert (1, -100, 42) not in _send_cooldowns

    def test_missing_key_no_error(self) -> None:
        _send_cooldowns.clear()
        clear_send_cooldowns(999, 999)


class TestClearTopicCreateRetry:
    def test_clears_existing_retry(self) -> None:
        _topic_create_retry_until[-100] = 999.0
        clear_topic_create_retry(-100)
        assert -100 not in _topic_create_retry_until

    def test_missing_key_no_error(self) -> None:
        _topic_create_retry_until.clear()
        clear_topic_create_retry(-999)


class TestClearDisabledChat:
    def test_removes_chat_from_set(self) -> None:
        _disabled_chats.add(-100)
        clear_disabled_chat(-100)
        assert -100 not in _disabled_chats

    def test_missing_chat_no_error(self) -> None:
        _disabled_chats.clear()
        clear_disabled_chat(-999)

    def test_size_guard_clears_all(self) -> None:
        _disabled_chats.clear()
        for i in range(_MAX_DISABLED_CHATS + 2):
            _disabled_chats.add(i)
        clear_disabled_chat(0)
        assert len(_disabled_chats) == 0
