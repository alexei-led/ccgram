"""Voice message handler — download OGG audio, transcribe via Whisper, and present confirm keyboard.

Handles Telegram voice messages by downloading the audio, transcribing it using
the configured Whisper provider, and showing the transcription with a confirm/discard
inline keyboard so the user can review before sending to the agent.

Key handler:
  - handle_voice_message: main entry point for filters.VOICE
"""

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..config import config
from ..session import session_manager
from ..whisper import get_transcriber
from ..whisper.base import TranscriptionResult
from .callback_helpers import get_thread_id
from .message_sender import safe_reply
from .user_state import VOICE_PENDING

logger = structlog.get_logger()

# Max voice file size: 25 MB (Telegram Bot API getFile limit)
_MAX_VOICE_SIZE = 25 * 1024 * 1024


def _pending_voice_key(chat_id: int, message_id: int) -> tuple[int, int]:
    """Build the user_data key for a pending voice transcription."""
    return (chat_id, message_id)


def _build_voice_keyboard(message_id: int) -> InlineKeyboardMarkup:
    """Build the confirm/discard inline keyboard for a transcribed voice message."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✓ Send to agent",
                    callback_data=f"vc:send:{message_id}",
                ),
                InlineKeyboardButton(
                    "✗ Discard",
                    callback_data=f"vc:drop:{message_id}",
                ),
            ]
        ]
    )


async def _download_voice(message: Message, file_id: str) -> bytes | None:
    """Download voice audio from Telegram. Returns bytes or None on error."""
    try:
        file = await message.get_bot().get_file(file_id)
        audio_bytearray = await file.download_as_bytearray()
        return bytes(audio_bytearray)
    except TelegramError as e:
        logger.warning("Failed to download voice message: %s", e)
        await safe_reply(message, "❌ Failed to download voice message.")
        return None


async def _get_transcriber_or_reply(message: Message):
    """Resolve the configured transcriber and surface user-facing errors."""
    try:
        transcriber = get_transcriber()
    except (ValueError, RuntimeError) as e:
        await safe_reply(message, f"❌ {e}")
        return None

    if transcriber is None:
        await safe_reply(
            message,
            "⚠️ Voice transcription is not configured. Set CCGRAM_WHISPER_PROVIDER to enable it.\n\nSupported providers: openai, groq",
        )
        return None

    return transcriber


async def _transcribe_audio(
    message: Message, transcriber, audio_bytes: bytes
) -> TranscriptionResult | None:
    """Transcribe audio bytes. Returns TranscriptionResult or None on error."""
    try:
        return await transcriber.transcribe(audio_bytes, "voice.ogg")
    except (ValueError, RuntimeError) as e:
        await safe_reply(message, f"❌ {e}")
        return None


async def _send_confirm_message(
    message: Message, text: str, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send transcription text, then attach the confirm/discard keyboard.

    We send without a keyboard first to learn the message_id, store the pending
    text under that id, then edit the keyboard in — so callback_data always carries
    a real message_id and the user_data entry exists before any tap is possible.
    """
    confirm_msg = await message.reply_text(
        f"🎤 Transcribed:\n\n{text}",
        parse_mode=None,
    )

    try:
        await confirm_msg.edit_reply_markup(
            reply_markup=_build_voice_keyboard(confirm_msg.message_id)
        )
    except TelegramError as e:
        logger.warning("Failed to attach voice confirm keyboard: %s", e)
        await safe_reply(
            message,
            "⚠️ Could not attach confirmation buttons. Please resend the voice message.",
        )
        return

    if context.user_data is not None:
        pending_key = _pending_voice_key(confirm_msg.chat.id, confirm_msg.message_id)
        context.user_data.setdefault(VOICE_PENDING, {})[pending_key] = text


async def handle_voice_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle incoming voice messages: transcribe and present confirm keyboard."""
    user = update.effective_user
    message = update.message
    if not user or not message or not message.voice:
        return

    if not config.is_user_allowed(user.id):
        await safe_reply(message, "You are not authorized to use this bot.")
        return

    thread_id = get_thread_id(update)
    window_id = session_manager.resolve_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(message, "⚠️ Bind this topic to an agent session first.")
        return

    voice = message.voice
    if voice.file_size is not None and voice.file_size > _MAX_VOICE_SIZE:
        size_mb = voice.file_size / (1024 * 1024)
        await safe_reply(
            message,
            f"❌ Voice message too large ({size_mb:.1f} MB). Maximum 25 MB.",
        )
        return

    transcriber = await _get_transcriber_or_reply(message)
    if transcriber is None:
        return

    audio_bytes = await _download_voice(message, voice.file_id)
    if audio_bytes is None:
        return

    await message.get_bot().send_chat_action(
        chat_id=message.chat.id,
        message_thread_id=message.message_thread_id,
        action=ChatAction.TYPING,
    )

    result = await _transcribe_audio(message, transcriber, audio_bytes)
    if result is None:
        return

    if not result.text.strip():
        await safe_reply(message, "⚠️ Could not transcribe audio (empty result).")
        return

    await _send_confirm_message(message, result.text, context)
