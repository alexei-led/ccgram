"""Topic identity extraction across forum and private direct-message chats."""

from unittest.mock import MagicMock

import pytest

from ccgram.handlers.callback_helpers import get_thread_id


def _update(
    *,
    chat_type: str = "supergroup",
    message_thread_id: int | None = None,
    is_topic_message: bool = False,
) -> MagicMock:
    update = MagicMock()
    message = update.message
    message.message_thread_id = message_thread_id
    message.is_topic_message = is_topic_message
    message.chat.type = chat_type
    message.chat.id = 100
    update.callback_query = None
    return update


class TestGetThreadId:
    @pytest.mark.parametrize(
        ("message_thread_id", "expected"),
        [(42, 42), (1, None), (None, None)],
    )
    def test_supergroup_behavior_is_unchanged(
        self, message_thread_id: int | None, expected: int | None
    ) -> None:
        assert get_thread_id(_update(message_thread_id=message_thread_id)) == expected

    def test_recognizes_private_topic_message(self) -> None:
        update = _update(
            chat_type="private", message_thread_id=42, is_topic_message=True
        )

        assert get_thread_id(update) == 42

    def test_does_not_guess_private_topic_from_thread_id_without_topic_flag(
        self,
    ) -> None:
        update = _update(chat_type="private", message_thread_id=42)

        assert get_thread_id(update) is None

    def test_keeps_unthreaded_private_dm_legacy_behavior(self) -> None:
        update = _update(chat_type="private")

        assert get_thread_id(update) is None

    def test_private_general_topic_is_the_control_lane(self) -> None:
        update = _update(
            chat_type="private", message_thread_id=1, is_topic_message=True
        )

        assert get_thread_id(update) is None
