"""Tests for /sync dead-topic recovery helpers."""

from ccgram.config import config
from ccgram.handlers.sync_command import _resolve_bound_group_chat_id


class TestResolveBoundGroupChatId:
    def test_prefers_bound_group_chat_id(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.sync_command.thread_router.resolve_chat_id",
            lambda user_id, thread_id: -100123,
        )
        monkeypatch.setattr(config, "group_id", -100999)

        assert _resolve_bound_group_chat_id(1, 10) == -100123

    def test_falls_back_to_config_group_id(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.sync_command.thread_router.resolve_chat_id",
            lambda user_id, thread_id: user_id,
        )
        monkeypatch.setattr(config, "group_id", -100999)

        assert _resolve_bound_group_chat_id(1, 10) == -100999

    def test_returns_user_id_without_any_group_context(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "ccgram.handlers.sync_command.thread_router.resolve_chat_id",
            lambda user_id, thread_id: user_id,
        )
        monkeypatch.setattr(config, "group_id", None)

        assert _resolve_bound_group_chat_id(1, 10) == 1
