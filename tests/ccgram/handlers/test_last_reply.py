"""Tests for last_reply.send_last_reply and last_command handler."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.telegram_client import FakeTelegramClient


def _ai_caps(supports_structured=True):
    caps = MagicMock()
    caps.name = "claude"
    caps.supports_structured_transcript = supports_structured
    return caps


def _shell_caps():
    caps = MagicMock()
    caps.name = "shell"
    caps.supports_structured_transcript = False
    return caps


def _make_provider(caps):
    p = MagicMock()
    p.capabilities = caps
    return p


def _msg(role, content_type, text):
    return {"role": role, "content_type": content_type, "text": text}


class TestExtractLastAiReply:
    async def test_assistant_text_after_last_user_turn(self) -> None:
        from ccgram.handlers.last_reply import _extract_last_ai_reply

        messages = [
            _msg("user", "text", "hello"),
            _msg("assistant", "text", "first reply"),
            _msg("user", "text", "second question"),
            _msg("assistant", "text", "second reply"),
        ]

        with patch(
            "ccgram.session_query.get_recent_messages",
            AsyncMock(return_value=(messages, 4)),
        ):
            result = await _extract_last_ai_reply("@0")
        assert result == "second reply"

    async def test_multiple_assistant_blocks_joined(self) -> None:
        from ccgram.handlers.last_reply import _extract_last_ai_reply

        messages = [
            _msg("user", "text", "go"),
            _msg("assistant", "text", "part one"),
            _msg("assistant", "text", "part two"),
        ]
        with patch(
            "ccgram.session_query.get_recent_messages",
            AsyncMock(return_value=(messages, 3)),
        ):
            result = await _extract_last_ai_reply("@0")
        assert result == "part one\n\npart two"

    async def test_fallback_to_most_recent_assistant_when_last_turn_has_none(
        self,
    ) -> None:
        from ccgram.handlers.last_reply import _extract_last_ai_reply

        messages = [
            _msg("assistant", "text", "early reply"),
            _msg("user", "text", "question"),
            _msg("assistant", "tool_use", ""),
        ]
        with patch(
            "ccgram.session_query.get_recent_messages",
            AsyncMock(return_value=(messages, 3)),
        ):
            result = await _extract_last_ai_reply("@0")
        assert result == "early reply"

    async def test_no_reply_yet_when_no_assistant_text(self) -> None:
        from ccgram.handlers.last_reply import _extract_last_ai_reply

        messages = [
            _msg("user", "text", "hello"),
            _msg("assistant", "tool_use", ""),
        ]
        with patch(
            "ccgram.session_query.get_recent_messages",
            AsyncMock(return_value=(messages, 2)),
        ):
            result = await _extract_last_ai_reply("@0")
        assert result == "No reply yet."

    async def test_empty_messages(self) -> None:
        from ccgram.handlers.last_reply import _extract_last_ai_reply

        with patch(
            "ccgram.session_query.get_recent_messages",
            AsyncMock(return_value=([], 0)),
        ):
            result = await _extract_last_ai_reply("@0")
        assert result == "No reply yet."


class TestSendLastReplyShell:
    async def test_shell_returns_block(self) -> None:
        from ccgram.handlers import last_reply

        fake = FakeTelegramClient()
        with (
            patch(
                "ccgram.handlers.last_reply.get_window_provider",
                return_value="shell",
            ),
            patch("ccgram.handlers.last_reply.tmux_manager") as mock_tm,
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=_make_provider(_shell_caps()),
            ),
            patch(
                "ccgram.last_unit.extract_last_shell_block",
                return_value="$ echo hi\nhi",
            ),
        ):
            mock_tm.capture_pane_scrollback = AsyncMock(return_value="scrollback")
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        assert any(c.method == "send_message" for c in fake.calls)
        sent = fake.last_call("send_message")
        assert sent is not None
        assert "$ echo hi" in sent.kwargs["text"]

    async def test_shell_no_block_returns_not_found(self) -> None:
        from ccgram.handlers import last_reply

        fake = FakeTelegramClient()
        with (
            patch(
                "ccgram.handlers.last_reply.get_window_provider",
                return_value="shell",
            ),
            patch("ccgram.handlers.last_reply.tmux_manager") as mock_tm,
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=_make_provider(_shell_caps()),
            ),
            patch(
                "ccgram.last_unit.extract_last_shell_block",
                return_value=None,
            ),
        ):
            mock_tm.capture_pane_scrollback = AsyncMock(return_value="scrollback")
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        sent = fake.last_call("send_message")
        assert sent is not None
        assert "No command output found." in sent.kwargs["text"]

    async def test_shell_no_scrollback_returns_not_found(self) -> None:
        from ccgram.handlers import last_reply

        fake = FakeTelegramClient()
        with (
            patch(
                "ccgram.handlers.last_reply.get_window_provider",
                return_value="shell",
            ),
            patch("ccgram.handlers.last_reply.tmux_manager") as mock_tm,
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=_make_provider(_shell_caps()),
            ),
            patch(
                "ccgram.last_unit.extract_last_shell_block",
                return_value=None,
            ),
        ):
            mock_tm.capture_pane_scrollback = AsyncMock(return_value=None)
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        sent = fake.last_call("send_message")
        assert sent is not None
        assert "No command output found." in sent.kwargs["text"]


class TestSendLastReplyAI:
    async def test_short_text_uses_send_message(self) -> None:
        from ccgram.handlers import last_reply

        fake = FakeTelegramClient()
        messages = [
            _msg("user", "text", "q"),
            _msg("assistant", "text", "short reply"),
        ]
        mock_sq = MagicMock()
        mock_sq.get_recent_messages = AsyncMock(return_value=(messages, 2))

        with (
            patch(
                "ccgram.handlers.last_reply.get_window_provider",
                return_value="claude",
            ),
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=_make_provider(_ai_caps()),
            ),
            patch(
                "ccgram.session_query.get_recent_messages",
                mock_sq.get_recent_messages,
            ),
        ):
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        assert fake.call_count("send_message") == 1
        assert fake.call_count("send_document") == 0

    async def test_text_at_limit_uses_send_message(self) -> None:
        from ccgram.handlers import last_reply

        fake = FakeTelegramClient()
        messages = [
            _msg("user", "text", "q"),
            _msg("assistant", "text", "x" * 4096),
        ]
        mock_sq = MagicMock()
        mock_sq.get_recent_messages = AsyncMock(return_value=(messages, 2))

        with (
            patch(
                "ccgram.handlers.last_reply.get_window_provider",
                return_value="claude",
            ),
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=_make_provider(_ai_caps()),
            ),
            patch(
                "ccgram.session_query.get_recent_messages",
                mock_sq.get_recent_messages,
            ),
        ):
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        assert fake.call_count("send_message") == 1
        assert fake.call_count("send_document") == 0

    async def test_long_text_uses_send_document(self) -> None:
        from ccgram.handlers import last_reply

        fake = FakeTelegramClient()
        long_text = "x" * 5000
        messages = [
            _msg("user", "text", "q"),
            _msg("assistant", "text", long_text),
        ]
        mock_sq = MagicMock()
        mock_sq.get_recent_messages = AsyncMock(return_value=(messages, 2))

        with (
            patch(
                "ccgram.handlers.last_reply.get_window_provider",
                return_value="claude",
            ),
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=_make_provider(_ai_caps()),
            ),
            patch(
                "ccgram.session_query.get_recent_messages",
                mock_sq.get_recent_messages,
            ),
        ):
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        assert fake.call_count("send_document") == 1
        assert fake.call_count("send_message") == 0
        doc_call = fake.last_call("send_document")
        assert doc_call is not None
        assert "last-reply-0.txt" in doc_call.kwargs.get("filename", "")

    async def test_long_text_cleans_temp_file_on_send_failure(self) -> None:
        from ccgram.handlers import last_reply

        fake = FakeTelegramClient()
        fake.set_side_effect("send_document", [RuntimeError("boom")])
        long_text = "x" * 5000
        messages = [
            _msg("user", "text", "q"),
            _msg("assistant", "text", long_text),
        ]
        created: list[str] = []
        real_ntf = tempfile.NamedTemporaryFile

        def _spy(*a, **k):
            fh = real_ntf(*a, **k)
            created.append(fh.name)
            return fh

        with (
            patch(
                "ccgram.handlers.last_reply.get_window_provider",
                return_value="claude",
            ),
            patch(
                "ccgram.providers.get_provider_for_window",
                return_value=_make_provider(_ai_caps()),
            ),
            patch(
                "ccgram.session_query.get_recent_messages",
                AsyncMock(return_value=(messages, 2)),
            ),
            patch("tempfile.NamedTemporaryFile", side_effect=_spy),
            pytest.raises(RuntimeError),
        ):
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        assert created
        assert not Path(created[0]).exists()


class TestLastCommand:
    def _make_update(self, user_id=1, thread_id=42, chat_id=100):
        user = MagicMock()
        user.id = user_id

        message = MagicMock()
        message.message_thread_id = thread_id
        message.get_bot = MagicMock(return_value=MagicMock())

        chat = MagicMock()
        chat.id = chat_id

        update = MagicMock()
        update.effective_user = user
        update.message = message
        update.effective_chat = chat
        return update

    async def test_unbound_window_sends_error(self) -> None:
        from ccgram.handlers.last_reply import last_command

        update = self._make_update()
        ctx = MagicMock()
        mock_reply = AsyncMock()

        with (
            patch("ccgram.config.config") as mock_cfg,
            patch("ccgram.handlers.callback_helpers.get_thread_id", return_value=42),
            patch("ccgram.handlers.last_reply.thread_router") as mock_tr,
            patch(
                "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
                mock_reply,
            ),
            patch("ccgram.utils.is_general_topic", return_value=False),
            patch("ccgram.utils.handle_general_topic_message"),
        ):
            mock_cfg.is_user_allowed.return_value = True
            mock_tr.get_window_for_thread.return_value = None
            await last_command(update, ctx)

        mock_reply.assert_called_once()
        text = mock_reply.call_args[0][1]
        assert "not bound" in text.lower() or "❌" in text

    async def test_dead_window_sends_error(self) -> None:
        from ccgram.handlers.last_reply import last_command

        update = self._make_update()
        ctx = MagicMock()
        mock_reply = AsyncMock()

        with (
            patch("ccgram.config.config") as mock_cfg,
            patch("ccgram.handlers.callback_helpers.get_thread_id", return_value=42),
            patch("ccgram.handlers.last_reply.thread_router") as mock_tr,
            patch("ccgram.handlers.last_reply.tmux_manager") as mock_tm,
            patch(
                "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
                mock_reply,
            ),
            patch("ccgram.utils.is_general_topic", return_value=False),
            patch("ccgram.utils.handle_general_topic_message"),
        ):
            mock_cfg.is_user_allowed.return_value = True
            mock_tr.get_window_for_thread.return_value = "@0"
            mock_tm.find_window_by_id = AsyncMock(return_value=None)
            await last_command(update, ctx)

        mock_reply.assert_called_once()
        text = mock_reply.call_args[0][1]
        assert "no longer exists" in text.lower() or "❌" in text

    async def test_bound_live_window_calls_send_last_reply(self) -> None:
        from ccgram.handlers.last_reply import last_command

        update = self._make_update()
        ctx = MagicMock()
        fake_window = MagicMock()
        mock_slr = AsyncMock()

        with (
            patch("ccgram.config.config") as mock_cfg,
            patch("ccgram.handlers.callback_helpers.get_thread_id", return_value=42),
            patch("ccgram.handlers.last_reply.thread_router") as mock_tr,
            patch("ccgram.handlers.last_reply.tmux_manager") as mock_tm,
            patch("ccgram.utils.is_general_topic", return_value=False),
            patch("ccgram.utils.handle_general_topic_message"),
            patch("ccgram.handlers.last_reply.send_last_reply", mock_slr),
            patch("ccgram.telegram_client.PTBTelegramClient"),
        ):
            mock_cfg.is_user_allowed.return_value = True
            mock_tr.get_window_for_thread.return_value = "@0"
            mock_tr.resolve_chat_id.return_value = 100
            mock_tm.find_window_by_id = AsyncMock(return_value=fake_window)
            await last_command(update, ctx)

        mock_slr.assert_called_once()
        _, call_chat_id, call_thread_id, call_window_id = mock_slr.call_args[0]
        assert call_chat_id == 100
        assert call_thread_id == 42
        assert call_window_id == "@0"
