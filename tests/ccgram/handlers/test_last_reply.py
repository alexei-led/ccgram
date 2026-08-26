"""Tests for last_reply.send_last_reply and last_command handler."""

import contextlib
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers import last_reply
from ccgram.handlers.last_reply import _extract_last_ai_reply, last_command
from ccgram.telegram_client import FakeTelegramClient

_LR = "ccgram.handlers.last_reply"


def _make_provider(name: str, *, structured: bool) -> MagicMock:
    caps = MagicMock()
    caps.name = name
    caps.supports_structured_transcript = structured
    provider = MagicMock()
    provider.capabilities = caps
    return provider


def _msg(role: str, content_type: str, text: str) -> dict:
    return {"role": role, "content_type": content_type, "text": text}


@contextlib.contextmanager
def _ai_window(messages: list[dict]) -> Iterator[None]:
    """Window @0 runs a structured-transcript agent with *messages* recorded."""
    with (
        patch(f"{_LR}.get_window_provider", return_value="claude"),
        patch(
            "ccgram.providers.get_provider_for_window",
            return_value=_make_provider("claude", structured=True),
        ),
        patch(
            "ccgram.session_query.get_recent_messages",
            AsyncMock(return_value=(messages, len(messages))),
        ),
    ):
        yield


@contextlib.contextmanager
def _shell_window(*, scrollback: str | None, block: str | None) -> Iterator[None]:
    with (
        patch(f"{_LR}.get_window_provider", return_value="shell"),
        patch(
            "ccgram.providers.get_provider_for_window",
            return_value=_make_provider("shell", structured=False),
        ),
        patch(f"{_LR}.tmux_manager") as mock_tm,
        patch("ccgram.last_unit.extract_last_shell_block", return_value=block),
    ):
        mock_tm.capture_pane_scrollback = AsyncMock(return_value=scrollback)
        yield


class TestExtractLastAiReply:
    @pytest.mark.parametrize(
        ("messages", "expected"),
        [
            pytest.param(
                [
                    _msg("user", "text", "hello"),
                    _msg("assistant", "text", "first reply"),
                    _msg("user", "text", "second question"),
                    _msg("assistant", "text", "second reply"),
                ],
                "second reply",
                id="last-turn-only",
            ),
            pytest.param(
                [
                    _msg("user", "text", "go"),
                    _msg("assistant", "text", "part one"),
                    _msg("assistant", "text", "part two"),
                ],
                "part one\n\npart two",
                id="blocks-of-one-turn-joined",
            ),
            pytest.param(
                [
                    _msg("assistant", "text", "early reply"),
                    _msg("user", "text", "question"),
                    _msg("assistant", "tool_use", ""),
                ],
                "early reply",
                id="falls-back-to-earlier-turn",
            ),
            pytest.param(
                [_msg("user", "text", "hello"), _msg("assistant", "tool_use", "")],
                "No reply yet.",
                id="no-assistant-text",
            ),
            pytest.param([], "No reply yet.", id="empty-transcript"),
        ],
    )
    async def test_extraction(self, messages: list[dict], expected: str) -> None:
        with patch(
            "ccgram.session_query.get_recent_messages",
            AsyncMock(return_value=(messages, len(messages))),
        ):
            assert await _extract_last_ai_reply("@0") == expected


class TestSendLastReplyShell:
    async def test_relays_the_last_command_block(self) -> None:
        fake = FakeTelegramClient()

        with _shell_window(scrollback="scrollback", block="$ echo hi\nhi"):
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        sent = fake.last_call("send_message")
        assert sent is not None
        assert "$ echo hi" in sent.kwargs["text"]

    @pytest.mark.parametrize(
        "scrollback",
        [
            pytest.param("scrollback", id="scrollback-without-a-block"),
            pytest.param(None, id="no-scrollback-at-all"),
        ],
    )
    async def test_reports_when_nothing_can_be_extracted(
        self, scrollback: str | None
    ) -> None:
        fake = FakeTelegramClient()

        with _shell_window(scrollback=scrollback, block=None):
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        sent = fake.last_call("send_message")
        assert sent is not None
        assert "No command output found." in sent.kwargs["text"]


class TestSendLastReplyAI:
    @pytest.mark.parametrize(
        ("reply_text", "sender"),
        [
            pytest.param("short reply", "send_message", id="short"),
            pytest.param("x" * 4096, "send_message", id="exactly-at-telegram-limit"),
            pytest.param("x" * 5000, "send_document", id="over-limit-becomes-a-file"),
        ],
    )
    async def test_delivery_channel_follows_length(
        self, reply_text: str, sender: str
    ) -> None:
        fake = FakeTelegramClient()
        messages = [_msg("user", "text", "q"), _msg("assistant", "text", reply_text)]

        with _ai_window(messages):
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        other = "send_document" if sender == "send_message" else "send_message"
        assert fake.call_count(sender) == 1
        assert fake.call_count(other) == 0

    async def test_overflow_document_is_named_per_window(self) -> None:
        fake = FakeTelegramClient()
        messages = [_msg("user", "text", "q"), _msg("assistant", "text", "x" * 5000)]

        with _ai_window(messages):
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        sent = fake.last_call("send_document")
        assert sent is not None
        assert "last-reply-0.txt" in sent.kwargs["filename"]

    async def test_temp_file_is_removed_when_upload_fails(self) -> None:
        fake = FakeTelegramClient()
        fake.set_side_effect("send_document", [RuntimeError("boom")])
        messages = [_msg("user", "text", "q"), _msg("assistant", "text", "x" * 5000)]
        created: list[str] = []
        real_ntf = tempfile.NamedTemporaryFile

        def _spy(*args, **kwargs):
            handle = real_ntf(*args, **kwargs)
            created.append(handle.name)
            return handle

        with (
            _ai_window(messages),
            patch("tempfile.NamedTemporaryFile", side_effect=_spy),
            pytest.raises(RuntimeError),
        ):
            await last_reply.send_last_reply(fake, 100, 42, "@0")

        assert created
        assert not Path(created[0]).exists()


class TestLastCommand:
    @staticmethod
    def _update(user_id: int = 1, thread_id: int = 42, chat_id: int = 100) -> MagicMock:
        update = MagicMock()
        update.effective_user = MagicMock(id=user_id)
        update.message = MagicMock(
            message_thread_id=thread_id, get_bot=MagicMock(return_value=MagicMock())
        )
        update.effective_chat = MagicMock(id=chat_id)
        return update

    @pytest.fixture
    def command_env(self) -> Iterator[MagicMock]:
        with (
            patch("ccgram.config.config") as config,
            patch("ccgram.handlers.callback_helpers.get_thread_id", return_value=42),
            patch(f"{_LR}.thread_router") as router,
            patch(f"{_LR}.tmux_manager") as mux,
            patch("ccgram.utils.is_general_topic", return_value=False),
            patch("ccgram.utils.handle_general_topic_message"),
        ):
            config.is_user_allowed.return_value = True
            router.get_window_for_thread.return_value = "@0"
            router.resolve_chat_id.return_value = 100
            mux.find_window_by_id = AsyncMock(return_value=MagicMock())
            yield MagicMock(config=config, router=router, mux=mux)

    @pytest.mark.parametrize(
        ("break_binding", "expected_fragment"),
        [
            pytest.param("unbound", "not bound", id="topic-not-bound"),
            pytest.param("dead", "no longer exists", id="window-gone"),
        ],
    )
    async def test_error_paths_reply_and_do_not_fetch(
        self, command_env: MagicMock, break_binding: str, expected_fragment: str
    ) -> None:
        if break_binding == "unbound":
            command_env.router.get_window_for_thread.return_value = None
        else:
            command_env.mux.find_window_by_id = AsyncMock(return_value=None)

        with (
            patch(
                "ccgram.handlers.messaging_pipeline.message_sender.safe_reply",
                new_callable=AsyncMock,
            ) as mock_reply,
            patch(f"{_LR}.send_last_reply", new_callable=AsyncMock) as mock_send,
        ):
            await last_command(self._update(), MagicMock())

        mock_send.assert_not_called()
        mock_reply.assert_called_once()
        text = mock_reply.call_args[0][1]
        assert expected_fragment in text.lower() or "❌" in text

    async def test_bound_live_window_delegates_to_send_last_reply(
        self, command_env: MagicMock
    ) -> None:
        with (
            patch(f"{_LR}.send_last_reply", new_callable=AsyncMock) as mock_send,
            patch("ccgram.telegram_client.PTBTelegramClient"),
        ):
            await last_command(self._update(), MagicMock())

        mock_send.assert_called_once()
        _client, chat_id, thread_id, window_id = mock_send.call_args[0]
        assert (chat_id, thread_id, window_id) == (100, 42, "@0")
