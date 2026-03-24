"""Unit tests for voice message handler and voice callbacks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.user_state import VOICE_PENDING
from ccgram.whisper.base import TranscriptionResult

_VH = "ccgram.handlers.voice_handler"
_VC = "ccgram.handlers.voice_callbacks"


def _make_update(
    user_id: int = 100,
    thread_id: int | None = 42,
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
    update.message.chat.id = 999
    update.message.get_bot = MagicMock(
        return_value=MagicMock(send_chat_action=AsyncMock())
    )
    update.message.reply_text = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    update.effective_message = update.message
    update.message.message_thread_id = thread_id
    return update


def _make_callback_query(data: str, message_id: int = 42) -> MagicMock:
    from telegram import CallbackQuery, Message

    query = MagicMock(spec=CallbackQuery)
    query.data = data
    query.from_user = MagicMock()
    query.message = MagicMock(spec=Message)
    query.message.message_id = message_id
    query.message.chat = MagicMock()
    query.message.chat.id = 999
    query.message.delete = AsyncMock()
    query.answer = AsyncMock()
    return query


class TestHandleVoiceMessage:
    @patch(f"{_VH}._download_voice", new_callable=AsyncMock)
    @patch(f"{_VH}.session_manager")
    @patch(f"{_VH}.config")
    @patch(f"{_VH}.get_transcriber")
    @patch(f"{_VH}.safe_reply", new_callable=AsyncMock)
    async def test_no_transcriber(
        self,
        mock_reply: AsyncMock,
        mock_get_transcriber: MagicMock,
        mock_config: MagicMock,
        mock_session_manager: MagicMock,
        mock_download: AsyncMock,
    ) -> None:
        from ccgram.handlers import voice_handler

        mock_config.is_user_allowed.return_value = True
        mock_get_transcriber.return_value = None
        mock_session_manager.resolve_window_for_thread.return_value = "@0"
        mock_download.return_value = b"fake audio bytes"

        await voice_handler.handle_voice_message(
            _make_update(), MagicMock(user_data={})
        )

        mock_reply.assert_called_once()
        assert "not configured" in mock_reply.call_args.args[1]
        mock_download.assert_not_awaited()

    @patch(f"{_VH}.session_manager")
    @patch(f"{_VH}.config")
    @patch(f"{_VH}.get_transcriber")
    @patch(f"{_VH}.safe_reply", new_callable=AsyncMock)
    async def test_unauthorized_user(
        self,
        mock_reply: AsyncMock,
        mock_get_transcriber: MagicMock,
        mock_config: MagicMock,
        mock_session_manager: MagicMock,
    ) -> None:
        from ccgram.handlers import voice_handler

        mock_config.is_user_allowed.return_value = False

        await voice_handler.handle_voice_message(
            _make_update(), MagicMock(user_data={})
        )

        mock_reply.assert_called_once()
        assert "not authorized" in mock_reply.call_args.args[1]
        mock_session_manager.resolve_window_for_thread.assert_not_called()

    @patch(f"{_VH}.session_manager")
    @patch(f"{_VH}.config")
    @patch(f"{_VH}.get_transcriber")
    @patch(f"{_VH}.safe_reply", new_callable=AsyncMock)
    async def test_topic_not_bound(
        self,
        mock_reply: AsyncMock,
        mock_get_transcriber: MagicMock,
        mock_config: MagicMock,
        mock_session_manager: MagicMock,
    ) -> None:
        from ccgram.handlers import voice_handler

        mock_config.is_user_allowed.return_value = True
        mock_get_transcriber.return_value = MagicMock()
        mock_session_manager.resolve_window_for_thread.return_value = None

        await voice_handler.handle_voice_message(
            _make_update(), MagicMock(user_data={})
        )

        mock_reply.assert_called_once()
        assert "Bind this topic" in mock_reply.call_args.args[1]

    @patch(f"{_VH}.session_manager")
    @patch(f"{_VH}.config")
    @patch(f"{_VH}.get_transcriber")
    @patch(f"{_VH}.safe_reply", new_callable=AsyncMock)
    async def test_file_too_large(
        self,
        mock_reply: AsyncMock,
        mock_get_transcriber: MagicMock,
        mock_config: MagicMock,
        mock_session_manager: MagicMock,
    ) -> None:
        from ccgram.handlers import voice_handler

        mock_config.is_user_allowed.return_value = True
        mock_get_transcriber.return_value = MagicMock()
        mock_session_manager.resolve_window_for_thread.return_value = "@0"

        await voice_handler.handle_voice_message(
            _make_update(voice_file_size=26 * 1024 * 1024),
            MagicMock(user_data={}),
        )

        mock_reply.assert_called_once()
        assert "too large" in mock_reply.call_args.args[1]

    @patch(f"{_VH}._download_voice", new_callable=AsyncMock)
    @patch(f"{_VH}.session_manager")
    @patch(f"{_VH}.config")
    @patch(f"{_VH}.get_transcriber")
    @patch(f"{_VH}.safe_reply", new_callable=AsyncMock)
    async def test_transcription_success(
        self,
        mock_reply: AsyncMock,
        mock_get_transcriber: MagicMock,
        mock_config: MagicMock,
        mock_session_manager: MagicMock,
        mock_download: AsyncMock,
    ) -> None:
        from telegram.constants import ChatAction

        from ccgram.handlers import voice_handler

        mock_config.is_user_allowed.return_value = True
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="do the thing", language="en")
        )
        mock_get_transcriber.return_value = mock_transcriber
        mock_session_manager.resolve_window_for_thread.return_value = "@0"
        mock_download.return_value = b"fake audio bytes"

        mock_reply_msg = MagicMock()
        mock_reply_msg.chat = MagicMock()
        mock_reply_msg.chat.id = 999
        update = _make_update()
        mock_reply.return_value = mock_reply_msg

        context = MagicMock()
        context.user_data = {}

        await voice_handler.handle_voice_message(update, context)

        assert context.user_data[VOICE_PENDING][(999, 1)] == "do the thing"
        update.message.get_bot.return_value.send_chat_action.assert_awaited_once_with(
            chat_id=999, message_thread_id=42, action=ChatAction.TYPING
        )
        mock_transcriber.transcribe.assert_awaited_once_with(
            b"fake audio bytes", "voice.ogg"
        )

    @patch(f"{_VH}._download_voice", new_callable=AsyncMock)
    @patch(f"{_VH}.session_manager")
    @patch(f"{_VH}.config")
    @patch(f"{_VH}.get_transcriber")
    @patch(f"{_VH}.safe_reply", new_callable=AsyncMock)
    async def test_confirm_reply_gone_skips_pending(
        self,
        mock_reply: AsyncMock,
        mock_get_transcriber: MagicMock,
        mock_config: MagicMock,
        mock_session_manager: MagicMock,
        mock_download: AsyncMock,
    ) -> None:
        from ccgram.handlers import voice_handler

        mock_config.is_user_allowed.return_value = True
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="do the thing", language="en")
        )
        mock_get_transcriber.return_value = mock_transcriber
        mock_session_manager.resolve_window_for_thread.return_value = "@0"
        mock_download.return_value = b"fake audio bytes"

        mock_reply.return_value = None
        update = _make_update()

        context = MagicMock()
        context.user_data = {}

        await voice_handler.handle_voice_message(update, context)

        assert context.user_data == {}

    @patch(f"{_VH}._download_voice", new_callable=AsyncMock)
    @patch(f"{_VH}.session_manager")
    @patch(f"{_VH}.config")
    @patch(f"{_VH}.get_transcriber")
    @patch(f"{_VH}.safe_reply", new_callable=AsyncMock)
    async def test_factory_value_error(
        self,
        mock_reply: AsyncMock,
        mock_get_transcriber: MagicMock,
        mock_config: MagicMock,
        mock_session_manager: MagicMock,
        mock_download: AsyncMock,
    ) -> None:
        from ccgram.handlers import voice_handler

        mock_config.is_user_allowed.return_value = True
        mock_get_transcriber.side_effect = ValueError("missing OPENAI_API_KEY")
        mock_session_manager.resolve_window_for_thread.return_value = "@0"

        await voice_handler.handle_voice_message(
            _make_update(), MagicMock(user_data={})
        )

        mock_reply.assert_called_once()
        assert "missing openai_api_key" in mock_reply.call_args.args[1].lower()

    @patch(f"{_VH}._download_voice", new_callable=AsyncMock)
    @patch(f"{_VH}.session_manager")
    @patch(f"{_VH}.config")
    @patch(f"{_VH}.get_transcriber")
    @patch(f"{_VH}.safe_reply", new_callable=AsyncMock)
    async def test_empty_transcription(
        self,
        mock_reply: AsyncMock,
        mock_get_transcriber: MagicMock,
        mock_config: MagicMock,
        mock_session_manager: MagicMock,
        mock_download: AsyncMock,
    ) -> None:
        from ccgram.handlers import voice_handler

        mock_config.is_user_allowed.return_value = True
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="   ", language="en")
        )
        mock_get_transcriber.return_value = mock_transcriber
        mock_session_manager.resolve_window_for_thread.return_value = "@0"
        mock_download.return_value = b"fake audio bytes"

        context = MagicMock()
        context.user_data = {}

        await voice_handler.handle_voice_message(_make_update(), context)

        mock_reply.assert_called_once()
        assert "empty result" in mock_reply.call_args.args[1].lower()
        assert context.user_data == {}

    @patch(f"{_VH}._download_voice", new_callable=AsyncMock)
    @patch(f"{_VH}.session_manager")
    @patch(f"{_VH}.config")
    @patch(f"{_VH}.get_transcriber")
    @patch(f"{_VH}.safe_reply", new_callable=AsyncMock)
    async def test_transcription_runtime_error(
        self,
        mock_reply: AsyncMock,
        mock_get_transcriber: MagicMock,
        mock_config: MagicMock,
        mock_session_manager: MagicMock,
        mock_download: AsyncMock,
    ) -> None:
        from ccgram.handlers import voice_handler

        mock_config.is_user_allowed.return_value = True
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe = AsyncMock(
            side_effect=RuntimeError("Transcription failed: 401")
        )
        mock_get_transcriber.return_value = mock_transcriber
        mock_session_manager.resolve_window_for_thread.return_value = "@0"
        mock_download.return_value = b"fake audio bytes"

        await voice_handler.handle_voice_message(
            _make_update(), MagicMock(user_data={})
        )

        mock_reply.assert_called()
        assert "❌" in mock_reply.call_args.args[1]

    @patch(f"{_VH}.session_manager")
    @patch(f"{_VH}.config")
    @patch(f"{_VH}.safe_reply", new_callable=AsyncMock)
    async def test_download_failure(
        self,
        mock_reply: AsyncMock,
        mock_config: MagicMock,
        mock_session_manager: MagicMock,
    ) -> None:
        from telegram.error import TelegramError

        from ccgram.handlers import voice_handler

        mock_config.is_user_allowed.return_value = True
        mock_session_manager.resolve_window_for_thread.return_value = "@0"

        update = _make_update()
        update.message.get_bot.return_value.get_file = AsyncMock(
            side_effect=TelegramError("download failed")
        )

        with patch(f"{_VH}.get_transcriber", return_value=MagicMock()):
            await voice_handler.handle_voice_message(update, MagicMock(user_data={}))

        mock_reply.assert_called_once()
        assert "Failed to download" in mock_reply.call_args.args[1]


class TestHandleVoiceCallback:
    @patch(f"{_VC}.session_manager")
    @patch(f"{_VC}.get_thread_id")
    async def test_send_success(
        self, mock_get_thread_id: MagicMock, mock_session_manager: MagicMock
    ) -> None:
        from ccgram.handlers import voice_callbacks

        mock_session_manager.resolve_window_for_thread.return_value = "@0"
        mock_session_manager.send_to_window = AsyncMock(return_value=(True, None))
        mock_get_thread_id.return_value = 42

        update = MagicMock()
        update.callback_query = _make_callback_query("vc:send:42", message_id=42)
        update.effective_user = MagicMock()
        update.effective_user.id = 100

        context = MagicMock()
        context.user_data = {VOICE_PENDING: {(999, 42): "hello"}}

        await voice_callbacks.handle_voice_callback(update, context)

        mock_session_manager.send_to_window.assert_called_once_with("@0", "hello")
        update.callback_query.message.delete.assert_called_once()
        update.callback_query.answer.assert_called_once_with("✓ Sent")
        assert (999, 42) not in context.user_data.get(VOICE_PENDING, {})

    @patch(f"{_VC}.session_manager")
    @patch(f"{_VC}.get_thread_id")
    async def test_send_success_delete_fails(
        self, mock_get_thread_id: MagicMock, mock_session_manager: MagicMock
    ) -> None:
        from telegram.error import TelegramError

        from ccgram.handlers import voice_callbacks

        mock_session_manager.resolve_window_for_thread.return_value = "@0"
        mock_session_manager.send_to_window = AsyncMock(return_value=(True, None))
        mock_get_thread_id.return_value = 42

        update = MagicMock()
        update.callback_query = _make_callback_query("vc:send:42", message_id=42)
        update.callback_query.message.delete = AsyncMock(
            side_effect=TelegramError("gone")
        )
        update.effective_user = MagicMock()
        update.effective_user.id = 100

        context = MagicMock()
        context.user_data = {VOICE_PENDING: {(999, 42): "hello"}}

        await voice_callbacks.handle_voice_callback(update, context)

        update.callback_query.answer.assert_called_once_with("✓ Sent")

    @patch(f"{_VC}.session_manager")
    @patch(f"{_VC}.get_thread_id")
    async def test_drop(
        self, mock_get_thread_id: MagicMock, mock_session_manager: MagicMock
    ) -> None:
        from ccgram.handlers import voice_callbacks

        mock_get_thread_id.return_value = 42

        update = MagicMock()
        update.callback_query = _make_callback_query("vc:drop:42", message_id=42)
        update.effective_user = MagicMock()
        update.effective_user.id = 100

        context = MagicMock()
        context.user_data = {VOICE_PENDING: {(999, 42): "hello"}}

        await voice_callbacks.handle_voice_callback(update, context)

        update.callback_query.message.delete.assert_called_once()
        assert (999, 42) not in context.user_data.get(VOICE_PENDING, {})
        update.callback_query.answer.assert_called_once_with("Discarded")

    @patch(f"{_VC}.session_manager")
    @patch(f"{_VC}.get_thread_id")
    async def test_drop_delete_fails(
        self, mock_get_thread_id: MagicMock, mock_session_manager: MagicMock
    ) -> None:
        from telegram.error import TelegramError

        from ccgram.handlers import voice_callbacks

        mock_get_thread_id.return_value = 42

        update = MagicMock()
        update.callback_query = _make_callback_query("vc:drop:42", message_id=42)
        update.callback_query.message.delete = AsyncMock(
            side_effect=TelegramError("gone")
        )
        update.effective_user = MagicMock()
        update.effective_user.id = 100

        context = MagicMock()
        context.user_data = {VOICE_PENDING: {(999, 42): "hello"}}

        await voice_callbacks.handle_voice_callback(update, context)

        update.callback_query.answer.assert_called_once_with("Discarded")

    @patch(f"{_VC}.session_manager")
    @patch(f"{_VC}.get_thread_id")
    async def test_drop_no_pending_entry(
        self, mock_get_thread_id: MagicMock, mock_session_manager: MagicMock
    ) -> None:
        from ccgram.handlers import voice_callbacks

        mock_get_thread_id.return_value = 42

        update = MagicMock()
        update.callback_query = _make_callback_query("vc:drop:42", message_id=42)
        update.effective_user = MagicMock()
        update.effective_user.id = 100

        context = MagicMock()
        context.user_data = {}

        await voice_callbacks.handle_voice_callback(update, context)

        update.callback_query.answer.assert_called_once_with("Discarded")

    @patch(f"{_VC}.session_manager")
    @patch(f"{_VC}.get_thread_id")
    async def test_expired_entry(
        self, mock_get_thread_id: MagicMock, mock_session_manager: MagicMock
    ) -> None:
        from ccgram.handlers import voice_callbacks

        mock_get_thread_id.return_value = 42

        update = MagicMock()
        update.callback_query = _make_callback_query("vc:send:99", message_id=99)
        update.effective_user = MagicMock()
        update.effective_user.id = 100

        context = MagicMock()
        context.user_data = {VOICE_PENDING: {}}

        await voice_callbacks.handle_voice_callback(update, context)

        update.callback_query.answer.assert_called_once()
        assert "expired" in update.callback_query.answer.call_args.args[0].lower()

    @patch(f"{_VC}.session_manager")
    @patch(f"{_VC}.get_thread_id")
    async def test_send_without_bound_window(
        self, mock_get_thread_id: MagicMock, mock_session_manager: MagicMock
    ) -> None:
        from ccgram.handlers import voice_callbacks

        mock_get_thread_id.return_value = 42
        mock_session_manager.resolve_window_for_thread.return_value = None

        update = MagicMock()
        update.callback_query = _make_callback_query("vc:send:42", message_id=42)
        update.effective_user = MagicMock()
        update.effective_user.id = 100

        context = MagicMock()
        context.user_data = {VOICE_PENDING: {(999, 42): "hello"}}

        await voice_callbacks.handle_voice_callback(update, context)

        update.callback_query.answer.assert_called_once_with(
            "⚠️ No session bound.", show_alert=True
        )
        assert (999, 42) in context.user_data.get(VOICE_PENDING, {})

    @pytest.mark.parametrize(
        "error_msg",
        [
            pytest.param("tmux down", id="tmux_down"),
            pytest.param("window not found", id="window_not_found"),
        ],
    )
    @patch(f"{_VC}.session_manager")
    @patch(f"{_VC}.get_thread_id")
    async def test_send_failure_preserves_pending(
        self,
        mock_get_thread_id: MagicMock,
        mock_session_manager: MagicMock,
        error_msg: str,
    ) -> None:
        from ccgram.handlers import voice_callbacks

        mock_get_thread_id.return_value = 42
        mock_session_manager.resolve_window_for_thread.return_value = "@0"
        mock_session_manager.send_to_window = AsyncMock(return_value=(False, error_msg))

        update = MagicMock()
        update.callback_query = _make_callback_query("vc:send:42", message_id=42)
        update.effective_user = MagicMock()
        update.effective_user.id = 100

        context = MagicMock()
        context.user_data = {VOICE_PENDING: {(999, 42): "hello"}}

        await voice_callbacks.handle_voice_callback(update, context)

        update.callback_query.answer.assert_called_once_with(
            f"❌ {error_msg}", show_alert=True
        )
        assert (999, 42) in context.user_data.get(VOICE_PENDING, {})

    async def test_invalid_payload(self) -> None:
        from ccgram.handlers import voice_callbacks

        update = MagicMock()
        update.callback_query = _make_callback_query("vc:send:not-an-int")
        update.effective_user = MagicMock()
        update.effective_user.id = 100

        context = MagicMock()
        context.user_data = {}

        await voice_callbacks.handle_voice_callback(update, context)

        update.callback_query.answer.assert_called_once_with("Invalid callback data")

    @patch(f"{_VC}.ack_reaction", new_callable=AsyncMock)
    @patch(f"{_VC}.get_provider_for_window")
    @patch(f"{_VC}.session_manager")
    @patch(f"{_VC}.get_thread_id")
    async def test_send_shell_provider_routes_through_llm(
        self,
        mock_get_thread_id: MagicMock,
        mock_session_manager: MagicMock,
        mock_get_provider: MagicMock,
        mock_ack: AsyncMock,
    ) -> None:
        from ccgram.handlers import voice_callbacks

        mock_get_thread_id.return_value = 42
        mock_session_manager.resolve_window_for_thread.return_value = "@0"

        mock_provider = MagicMock()
        mock_provider.capabilities.name = "shell"
        mock_get_provider.return_value = mock_provider

        update = MagicMock()
        update.callback_query = _make_callback_query("vc:send:42", message_id=42)
        update.effective_user = MagicMock()
        update.effective_user.id = 100

        context = MagicMock()
        context.user_data = {VOICE_PENDING: {(999, 42): "list files"}}

        with patch(
            "ccgram.handlers.shell_commands.handle_shell_message",
            new_callable=AsyncMock,
        ) as mock_shell:
            await voice_callbacks.handle_voice_callback(update, context)

            mock_shell.assert_called_once_with(
                update.callback_query.message.get_bot(),
                100,
                42,
                "@0",
                "list files",
            )

        mock_session_manager.send_to_window.assert_not_called()
        update.callback_query.message.delete.assert_called_once()
        update.callback_query.answer.assert_called_once_with("✓ Sent")
        mock_ack.assert_called_once()

    @patch(f"{_VC}.get_provider_for_window")
    @patch(f"{_VC}.session_manager")
    @patch(f"{_VC}.get_thread_id")
    async def test_send_shell_provider_error_preserves_pending(
        self,
        mock_get_thread_id: MagicMock,
        mock_session_manager: MagicMock,
        mock_get_provider: MagicMock,
    ) -> None:
        from ccgram.handlers import voice_callbacks

        mock_get_thread_id.return_value = 42
        mock_session_manager.resolve_window_for_thread.return_value = "@0"

        mock_provider = MagicMock()
        mock_provider.capabilities.name = "shell"
        mock_get_provider.return_value = mock_provider

        update = MagicMock()
        update.callback_query = _make_callback_query("vc:send:42", message_id=42)
        update.effective_user = MagicMock()
        update.effective_user.id = 100

        context = MagicMock()
        context.user_data = {VOICE_PENDING: {(999, 42): "list files"}}

        with patch(
            "ccgram.handlers.shell_commands.handle_shell_message",
            new_callable=AsyncMock,
            side_effect=OSError("tmux died"),
        ):
            await voice_callbacks.handle_voice_callback(update, context)

        # Text restored to pending store
        assert context.user_data[VOICE_PENDING][(999, 42)] == "list files"
        update.callback_query.answer.assert_called_once_with(
            "❌ Failed to send", show_alert=True
        )
        mock_session_manager.send_to_window.assert_not_called()

    async def test_inaccessible_message(self) -> None:
        from ccgram.handlers import voice_callbacks

        update = MagicMock()
        query = MagicMock()
        query.data = "vc:send:42"
        query.message = MagicMock()
        query.answer = AsyncMock()
        update.callback_query = query
        update.effective_user = MagicMock()

        context = MagicMock()

        await voice_callbacks.handle_voice_callback(update, context)

        query.answer.assert_called_once_with("Message no longer available")
