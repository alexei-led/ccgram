"""Unit tests for voice message handler and voice callbacks."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import CallbackQuery, Message
from telegram.constants import ChatAction
from telegram.error import TelegramError

from ccgram.handlers.user_state import VOICE_PENDING
from ccgram.handlers.voice import voice_callbacks, voice_handler
from ccgram.whisper.base import TranscriptionResult

_VH = "ccgram.handlers.voice.voice_handler"
_VC = "ccgram.handlers.voice.voice_callbacks"

_CHAT_ID = 999
_USER_ID = 100
_THREAD_ID = 42
_TRANSCRIPT = "do the thing"


def _make_update(
    user_id: int = _USER_ID,
    thread_id: int | None = _THREAD_ID,
    message_id: int = 1,
    voice_file_id: str = "voice123",
    voice_file_size: int | None = 1000,
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.message = MagicMock()
    update.message.message_id = message_id
    update.message.voice = MagicMock()
    update.message.voice.file_id = voice_file_id
    update.message.voice.file_size = voice_file_size
    update.message.chat = MagicMock()
    update.message.chat.id = _CHAT_ID
    update.message.get_bot = MagicMock(
        return_value=MagicMock(send_chat_action=AsyncMock())
    )
    update.message.reply_text = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    update.effective_message = update.message
    update.message.message_thread_id = thread_id
    return update


def _make_callback_query(data: str, message_id: int = _THREAD_ID) -> MagicMock:
    query = MagicMock(spec=CallbackQuery)
    query.data = data
    query.from_user = MagicMock()
    query.message = MagicMock(spec=Message)
    query.message.message_id = message_id
    query.message.chat = MagicMock()
    query.message.chat.id = _CHAT_ID
    query.message.delete = AsyncMock()
    bot = MagicMock()
    bot.set_message_reaction = AsyncMock()
    query.message.get_bot = MagicMock(return_value=bot)
    query.answer = AsyncMock()
    return query


def _callback_update(data: str, message_id: int = _THREAD_ID) -> MagicMock:
    update = MagicMock()
    update.callback_query = _make_callback_query(data, message_id=message_id)
    update.effective_user = MagicMock()
    update.effective_user.id = _USER_ID
    return update


def _callback_context(pending: dict | None = None) -> MagicMock:
    context = MagicMock()
    context.user_data = {} if pending is None else {VOICE_PENDING: pending}
    return context


class TestHandleVoiceMessage:
    @pytest.fixture
    def voice_env(self) -> Iterator[SimpleNamespace]:
        """Authorized user, bound topic, working transcriber — the happy path."""
        transcriber = MagicMock(
            transcribe=AsyncMock(
                return_value=TranscriptionResult(text=_TRANSCRIPT, language="en")
            )
        )
        with (
            patch(f"{_VH}.config") as config,
            patch(f"{_VH}.thread_router") as router,
            patch(
                f"{_VH}.get_transcriber", return_value=transcriber
            ) as get_transcriber,
            patch(f"{_VH}.safe_reply", new_callable=AsyncMock) as reply,
            patch(
                f"{_VH}._download_voice",
                new_callable=AsyncMock,
                return_value=b"fake audio bytes",
            ) as download,
        ):
            config.is_user_allowed.return_value = True
            config.voice_autosend = False
            router.resolve_window_for_thread.return_value = "@0"
            reply.return_value = MagicMock(chat=MagicMock(id=_CHAT_ID))
            yield SimpleNamespace(
                config=config,
                router=router,
                get_transcriber=get_transcriber,
                transcriber=transcriber,
                reply=reply,
                download=download,
            )

    async def test_transcription_stores_pending_confirmation(
        self, voice_env: SimpleNamespace
    ) -> None:
        update = _make_update()
        context = MagicMock(user_data={})

        await voice_handler.handle_voice_message(update, context)

        assert context.user_data[VOICE_PENDING][(_CHAT_ID, 1)] == _TRANSCRIPT
        voice_env.transcriber.transcribe.assert_awaited_once_with(
            b"fake audio bytes", "voice.ogg"
        )
        update.message.get_bot.return_value.send_chat_action.assert_awaited_once_with(
            chat_id=_CHAT_ID, message_thread_id=_THREAD_ID, action=ChatAction.TYPING
        )

    async def test_autosend_posts_transcription_without_keyboard(
        self, voice_env: SimpleNamespace
    ) -> None:
        voice_env.config.voice_autosend = True
        update = _make_update()
        context = MagicMock(user_data={})

        with patch(
            f"{_VH}._send_transcribed_text",
            new_callable=AsyncMock,
            return_value=(True, ""),
        ) as mock_send:
            await voice_handler.handle_voice_message(update, context)

        voice_env.reply.assert_awaited_once_with(
            update.message, f"🎤 Transcribed:\n\n{_TRANSCRIPT}"
        )
        mock_send.assert_awaited_once_with(
            _USER_ID, _THREAD_ID, "@0", _TRANSCRIPT, update.message
        )
        assert context.user_data == {}

    async def test_autosend_stops_if_transcription_cannot_be_posted(
        self, voice_env: SimpleNamespace
    ) -> None:
        voice_env.config.voice_autosend = True
        voice_env.reply.return_value = None

        with patch(
            f"{_VH}._send_transcribed_text", new_callable=AsyncMock
        ) as mock_send:
            await voice_handler.handle_voice_message(
                _make_update(), MagicMock(user_data={})
            )

        mock_send.assert_not_awaited()

    async def test_confirm_reply_gone_skips_pending(
        self, voice_env: SimpleNamespace
    ) -> None:
        voice_env.reply.return_value = None
        context = MagicMock(user_data={})

        await voice_handler.handle_voice_message(_make_update(), context)

        assert context.user_data == {}

    async def test_unauthorized_user(self, voice_env: SimpleNamespace) -> None:
        voice_env.config.is_user_allowed.return_value = False

        await voice_handler.handle_voice_message(
            _make_update(), MagicMock(user_data={})
        )

        assert "not authorized" in voice_env.reply.call_args.args[1]
        voice_env.router.resolve_window_for_thread.assert_not_called()

    async def test_no_transcriber_configured(self, voice_env: SimpleNamespace) -> None:
        voice_env.get_transcriber.return_value = None

        await voice_handler.handle_voice_message(
            _make_update(), MagicMock(user_data={})
        )

        assert "not configured" in voice_env.reply.call_args.args[1]
        voice_env.download.assert_not_awaited()

    async def test_transcriber_factory_error_is_surfaced(
        self, voice_env: SimpleNamespace
    ) -> None:
        voice_env.get_transcriber.side_effect = ValueError("missing OPENAI_API_KEY")

        await voice_handler.handle_voice_message(
            _make_update(), MagicMock(user_data={})
        )

        assert "missing openai_api_key" in voice_env.reply.call_args.args[1].lower()

    async def test_unbound_topic_explains_no_queueing(
        self, voice_env: SimpleNamespace
    ) -> None:
        voice_env.router.resolve_window_for_thread.return_value = None

        await voice_handler.handle_voice_message(
            _make_update(), MagicMock(user_data={})
        )

        body = voice_env.reply.call_args.args[1]
        assert "not bound" in body
        assert "Voice messages aren't queued" in body

    async def test_file_too_large(self, voice_env: SimpleNamespace) -> None:
        await voice_handler.handle_voice_message(
            _make_update(voice_file_size=26 * 1024 * 1024), MagicMock(user_data={})
        )

        assert "too large" in voice_env.reply.call_args.args[1]
        voice_env.download.assert_not_awaited()

    async def test_empty_transcription(self, voice_env: SimpleNamespace) -> None:
        voice_env.transcriber.transcribe.return_value = TranscriptionResult(
            text="   ", language="en"
        )
        context = MagicMock(user_data={})

        await voice_handler.handle_voice_message(_make_update(), context)

        assert "empty result" in voice_env.reply.call_args.args[1].lower()
        assert context.user_data == {}

    async def test_transcription_runtime_error(
        self, voice_env: SimpleNamespace
    ) -> None:
        voice_env.transcriber.transcribe.side_effect = RuntimeError(
            "Transcription failed: 401"
        )

        await voice_handler.handle_voice_message(
            _make_update(), MagicMock(user_data={})
        )

        assert "❌" in voice_env.reply.call_args.args[1]

    async def test_failed_download_stops_processing(
        self, voice_env: SimpleNamespace
    ) -> None:
        voice_env.download.return_value = None
        context = MagicMock(user_data={})

        await voice_handler.handle_voice_message(_make_update(), context)

        voice_env.transcriber.transcribe.assert_not_awaited()
        assert context.user_data == {}


class TestDownloadVoice:
    async def test_telegram_error_replies_and_returns_none(self) -> None:
        message = MagicMock()
        message.get_bot.return_value.get_file = AsyncMock(
            side_effect=TelegramError("download failed")
        )

        with patch(f"{_VH}.safe_reply", new_callable=AsyncMock) as mock_reply:
            result = await voice_handler._download_voice(message, "voice123")

        assert result is None
        assert "Failed to download" in mock_reply.call_args.args[1]

    async def test_returns_downloaded_bytes(self) -> None:
        message = MagicMock()
        file = MagicMock()
        file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio"))
        message.get_bot.return_value.get_file = AsyncMock(return_value=file)

        assert await voice_handler._download_voice(message, "voice123") == b"audio"


class TestHandleVoiceCallback:
    @pytest.fixture
    def callback_env(self) -> Iterator[SimpleNamespace]:
        with (
            patch(f"{_VC}.get_thread_id", return_value=_THREAD_ID),
            patch(f"{_VC}.thread_router") as router,
            patch(
                f"{_VC}.send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, None),
            ) as send_to_window,
            patch(f"{_VC}.get_provider_for_window") as get_provider,
            patch(f"{_VC}.ack_reaction", new_callable=AsyncMock) as ack,
        ):
            router.resolve_window_for_thread.return_value = "@0"
            get_provider.return_value.capabilities.name = "claude"
            get_provider.return_value.capabilities.chat_first_command_path = False
            yield SimpleNamespace(
                router=router,
                send_to_window=send_to_window,
                get_provider=get_provider,
                ack=ack,
            )

    @staticmethod
    def _shell_provider(env: SimpleNamespace) -> None:
        env.get_provider.return_value.capabilities.name = "shell"
        env.get_provider.return_value.capabilities.chat_first_command_path = True

    async def test_send_delivers_and_reacts(
        self, callback_env: SimpleNamespace
    ) -> None:
        update = _callback_update("vc:send:42")
        context = _callback_context({(_CHAT_ID, _THREAD_ID): "hello"})

        await voice_callbacks.handle_voice_callback(update, context)

        callback_env.send_to_window.assert_called_once_with(
            _USER_ID, "@0", _THREAD_ID, "hello", _CHAT_ID
        )
        update.callback_query.message.delete.assert_called_once()
        # Toast replaced with persistent reactions: 👀 on receive, 🔥 on delivery.
        update.callback_query.answer.assert_called_once_with()
        bot = update.callback_query.message.get_bot()
        emojis = [
            call.kwargs["reaction"][0].emoji
            for call in bot.set_message_reaction.await_args_list
        ]
        assert {"👀", "🔥"} <= set(emojis)
        assert context.user_data[VOICE_PENDING] == {}

    @pytest.mark.parametrize(
        ("action", "expected_answer"),
        [
            pytest.param("send", (), id="send"),
            pytest.param("drop", ("Discarded",), id="drop"),
        ],
    )
    async def test_delete_failure_still_answers(
        self, callback_env: SimpleNamespace, action: str, expected_answer: tuple
    ) -> None:
        update = _callback_update(f"vc:{action}:42")
        update.callback_query.message.delete = AsyncMock(
            side_effect=TelegramError("gone")
        )

        await voice_callbacks.handle_voice_callback(
            update, _callback_context({(_CHAT_ID, _THREAD_ID): "hello"})
        )

        update.callback_query.answer.assert_called_once_with(*expected_answer)

    @pytest.mark.parametrize(
        "pending",
        [
            pytest.param({(_CHAT_ID, _THREAD_ID): "hello"}, id="with-pending-entry"),
            pytest.param(None, id="without-pending-entry"),
        ],
    )
    async def test_drop_discards(
        self, callback_env: SimpleNamespace, pending: dict | None
    ) -> None:
        update = _callback_update("vc:drop:42")
        context = _callback_context(pending)

        await voice_callbacks.handle_voice_callback(update, context)

        update.callback_query.message.delete.assert_called_once()
        update.callback_query.answer.assert_called_once_with("Discarded")
        assert (_CHAT_ID, _THREAD_ID) not in context.user_data.get(VOICE_PENDING, {})

    async def test_expired_entry(self, callback_env: SimpleNamespace) -> None:
        update = _callback_update("vc:send:99", message_id=99)

        await voice_callbacks.handle_voice_callback(update, _callback_context({}))

        assert "expired" in update.callback_query.answer.call_args.args[0].lower()

    async def test_send_without_bound_window_keeps_pending(
        self, callback_env: SimpleNamespace
    ) -> None:
        callback_env.router.resolve_window_for_thread.return_value = None
        update = _callback_update("vc:send:42")
        context = _callback_context({(_CHAT_ID, _THREAD_ID): "hello"})

        await voice_callbacks.handle_voice_callback(update, context)

        update.callback_query.answer.assert_called_once_with(
            "⚠️ No session bound.", show_alert=True
        )
        assert (_CHAT_ID, _THREAD_ID) in context.user_data[VOICE_PENDING]

    @pytest.mark.parametrize(
        "error_msg",
        [
            pytest.param("tmux down", id="tmux-down"),
            pytest.param("window not found", id="window-not-found"),
        ],
    )
    async def test_send_failure_preserves_pending(
        self, callback_env: SimpleNamespace, error_msg: str
    ) -> None:
        callback_env.send_to_window.return_value = (False, error_msg)
        update = _callback_update("vc:send:42")
        context = _callback_context({(_CHAT_ID, _THREAD_ID): "hello"})

        await voice_callbacks.handle_voice_callback(update, context)

        update.callback_query.answer.assert_called_once_with(
            f"❌ {error_msg}", show_alert=True
        )
        assert (_CHAT_ID, _THREAD_ID) in context.user_data[VOICE_PENDING]

    async def test_invalid_payload(self, callback_env: SimpleNamespace) -> None:
        update = _callback_update("vc:send:not-an-int")

        await voice_callbacks.handle_voice_callback(update, _callback_context())

        update.callback_query.answer.assert_called_once_with("Invalid callback data")

    async def test_inaccessible_message(self) -> None:
        query = MagicMock()
        query.data = "vc:send:42"
        query.message = MagicMock()
        query.answer = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        update.effective_user = MagicMock()

        await voice_callbacks.handle_voice_callback(update, MagicMock())

        query.answer.assert_called_once_with("Message no longer available")

    async def test_shell_provider_routes_through_llm(
        self, callback_env: SimpleNamespace
    ) -> None:
        self._shell_provider(callback_env)
        update = _callback_update("vc:send:42")
        context = _callback_context({(_CHAT_ID, _THREAD_ID): "list files"})

        with patch(
            "ccgram.handlers.shell.shell_commands.handle_shell_message",
            new_callable=AsyncMock,
        ) as mock_shell:
            await voice_callbacks.handle_voice_callback(update, context)

        mock_shell.assert_called_once()
        assert mock_shell.call_args.args[1:] == (
            _USER_ID,
            _THREAD_ID,
            "@0",
            "list files",
        )
        assert mock_shell.call_args.args[0].bot is (
            update.callback_query.message.get_bot()
        )
        callback_env.send_to_window.assert_not_called()
        update.callback_query.message.delete.assert_called_once()
        update.callback_query.answer.assert_called_once_with()
        callback_env.ack.assert_called_once()

    async def test_shell_provider_error_preserves_pending(
        self, callback_env: SimpleNamespace
    ) -> None:
        self._shell_provider(callback_env)
        update = _callback_update("vc:send:42")
        context = _callback_context({(_CHAT_ID, _THREAD_ID): "list files"})

        with patch(
            "ccgram.handlers.shell.shell_commands.handle_shell_message",
            new_callable=AsyncMock,
            side_effect=OSError("tmux died"),
        ):
            await voice_callbacks.handle_voice_callback(update, context)

        assert context.user_data[VOICE_PENDING][(_CHAT_ID, _THREAD_ID)] == "list files"
        update.callback_query.answer.assert_called_once_with(
            "❌ Failed to send", show_alert=True
        )
        callback_env.send_to_window.assert_not_called()
