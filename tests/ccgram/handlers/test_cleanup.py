from unittest.mock import AsyncMock, patch

import pytest

from ccgram.config import config
from ccgram.handlers.cleanup import clear_topic_state, rollback_command


class TestClearTopicState:
    async def test_enqueues_status_clear_when_bot_available(self) -> None:
        bot = AsyncMock()
        with (
            patch("ccgram.handlers.cleanup.enqueue_status_update") as mock_enqueue,
            patch("ccgram.handlers.cleanup.clear_interactive_msg"),
            patch("ccgram.thread_router.thread_router") as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            await clear_topic_state(1, 42, client=bot, window_id="@0")

        mock_enqueue.assert_called_once()
        args = mock_enqueue.call_args
        assert args[0][1] == 1
        assert args[0][2] == "@0"
        assert args[0][3] is None
        assert args[1]["thread_id"] == 42

    async def test_skips_enqueue_when_no_bot(self) -> None:
        with (
            patch("ccgram.handlers.cleanup.enqueue_status_update") as mock_enqueue,
            patch("ccgram.handlers.cleanup.clear_interactive_msg"),
            patch("ccgram.thread_router.thread_router") as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            await clear_topic_state(1, 42, client=None, window_id="@0")

        mock_enqueue.assert_not_called()

    async def test_enqueues_empty_window_id_when_none(self) -> None:
        bot = AsyncMock()
        with (
            patch("ccgram.handlers.cleanup.enqueue_status_update") as mock_enqueue,
            patch("ccgram.handlers.cleanup.clear_interactive_msg"),
            patch("ccgram.thread_router.thread_router") as mock_tr,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            await clear_topic_state(1, 42, client=bot, window_id=None)

        mock_enqueue.assert_called_once()
        assert mock_enqueue.call_args[0][2] == ""

    async def test_revokes_window_callback_tokens(self) -> None:
        with (
            patch("ccgram.handlers.cleanup.clear_interactive_msg"),
            patch("ccgram.thread_router.thread_router") as mock_tr,
            patch("ccgram.handlers.cleanup.revoke_window_tokens") as revoke,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            await clear_topic_state(1, 42, client=None, window_id="@0")

        revoke.assert_called_once_with("@0")


class TestClearTopicStateQualifiedId:
    """qualified_id is built from session_map_prefix(), not hardcoded tmux prefix."""

    @pytest.fixture
    def _common_patches(self):
        with (
            patch("ccgram.handlers.cleanup.enqueue_status_update"),
            patch("ccgram.handlers.cleanup.clear_interactive_msg"),
            patch("ccgram.thread_router.thread_router") as mock_tr,
            patch("ccgram.handlers.cleanup.topic_state") as mock_ts,
        ):
            mock_tr.resolve_chat_id.return_value = -100
            yield mock_ts

    async def test_qualified_id_tmux_prefix(self, monkeypatch, _common_patches) -> None:
        monkeypatch.setattr(config, "multiplexer_name", "tmux")
        monkeypatch.setattr(config, "tmux_session_name", "ccgram")
        mock_ts = _common_patches
        await clear_topic_state(1, 42, client=None, window_id="@5", window_dead=True)
        kwargs = mock_ts.clear_all.call_args[1]
        assert kwargs["qualified_id"] == "ccgram:@5"

    async def test_qualified_id_herdr_prefix(
        self, monkeypatch, _common_patches
    ) -> None:
        monkeypatch.setattr(config, "multiplexer_name", "herdr")
        mock_ts = _common_patches
        target = "herdr-session-v1-" + "a" * 64
        await clear_topic_state(1, 42, client=None, window_id=target, window_dead=True)
        kwargs = mock_ts.clear_all.call_args[1]
        assert kwargs["qualified_id"] == f"herdr:{target}"

    async def test_qualified_id_none_when_window_alive(
        self, monkeypatch, _common_patches
    ) -> None:
        monkeypatch.setattr(config, "multiplexer_name", "herdr")
        mock_ts = _common_patches
        # window_dead defaults to True; pass False to suppress qualified_id
        await clear_topic_state(
            1,
            42,
            client=None,
            window_id="herdr-session-v1-" + "a" * 64,
            window_dead=False,
        )
        kwargs = mock_ts.clear_all.call_args[1]
        assert kwargs["qualified_id"] is None


class TestRollbackCommand:
    async def test_restores_only_this_topics_archived_binding(self) -> None:
        update = AsyncMock()
        update.effective_user.id = 1
        update.message = AsyncMock()
        with (
            patch("ccgram.handlers.cleanup.config") as mock_config,
            patch("ccgram.handlers.cleanup.get_thread_id", return_value=42),
            patch("ccgram.handlers.cleanup.thread_router") as router,
            patch("ccgram.handlers.cleanup.legacy_state") as legacy_state,
            patch(
                "ccgram.handlers.topics.topic_lifecycle.rollback_legacy_herdr_binding",
                return_value=True,
            ) as rollback,
            patch("ccgram.handlers.cleanup.safe_reply", new_callable=AsyncMock),
        ):
            mock_config.is_user_allowed.return_value = True
            router.get_window_for_thread.return_value = None
            legacy_state.get_archived_legacy_herdr_binding.return_value = "w2:t1"

            await rollback_command(update, AsyncMock())

        legacy_state.get_archived_legacy_herdr_binding.assert_called_once_with(1, 42)
        rollback.assert_called_once_with(1, 42, "w2:t1")

    async def test_does_not_restore_another_users_archived_binding(self) -> None:
        update = AsyncMock()
        update.effective_user.id = 2
        update.message = AsyncMock()
        with (
            patch("ccgram.handlers.cleanup.config") as mock_config,
            patch("ccgram.handlers.cleanup.get_thread_id", return_value=42),
            patch("ccgram.handlers.cleanup.thread_router") as router,
            patch("ccgram.handlers.cleanup.legacy_state") as legacy_state,
            patch("ccgram.handlers.cleanup.safe_reply", new_callable=AsyncMock),
        ):
            mock_config.is_user_allowed.return_value = True
            router.get_window_for_thread.return_value = None
            legacy_state.get_archived_legacy_herdr_binding.return_value = None

            await rollback_command(update, AsyncMock())

        legacy_state.get_archived_legacy_herdr_binding.assert_called_once_with(2, 42)
