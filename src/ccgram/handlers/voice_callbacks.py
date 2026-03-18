"""Voice transcription callbacks — handle confirm (send to agent) and discard actions.

Handles the inline keyboard callbacks triggered after voice message transcription:
  - vc:send:<msg_id>: Send transcribed text to the bound agent window
  - vc:drop:<msg_id>: Discard the transcription and delete the confirmation message

Key function: handle_voice_callback
"""

import structlog
from telegram import CallbackQuery, Message, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..session import session_manager
from .callback_helpers import get_thread_id
from .user_state import VOICE_PENDING

logger = structlog.get_logger()


def _pending_voice_key(chat_id: int, message_id: int) -> tuple[int, int]:
    """Build the user_data key for a pending voice transcription."""
    return (chat_id, message_id)


async def handle_voice_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle voice transcription confirm/discard callbacks."""
    query = update.callback_query
    if not query or not query.data:
        return

    user = update.effective_user
    if not user:
        return

    # Ensure the message is accessible (not expired/deleted)
    if not isinstance(query.message, Message):
        await query.answer("Message no longer available")
        return

    try:
        parts = query.data.split(":", 2)  # ["vc", "send"/"drop", "<msg_id>"]
        action = parts[1]
        message_id = int(parts[2])
    except IndexError, ValueError:
        await query.answer("Invalid callback data")
        return

    if action == "send":
        await _handle_send(query.message, query, user.id, message_id, update, context)
    elif action == "drop":
        await _handle_drop(query.message, query, message_id, context)
    else:
        await query.answer("Invalid callback data")


async def _handle_send(
    msg: Message,
    query: CallbackQuery,
    user_id: int,
    message_id: int,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle vc:send — forward transcribed text to the agent window."""
    pending_key = _pending_voice_key(msg.chat.id, message_id)
    pending_store = (
        context.user_data.get(VOICE_PENDING, {}) if context.user_data else {}
    )
    pending_text = pending_store.get(pending_key)
    if pending_text is None:
        await query.answer("⚠️ Session expired, resend voice message", show_alert=True)
        return

    thread_id = get_thread_id(update)
    window_id = session_manager.resolve_window_for_thread(user_id, thread_id)
    if not window_id:
        await query.answer("⚠️ No session bound.", show_alert=True)
        return

    pending_text = pending_store.pop(pending_key)
    try:
        success, err = await session_manager.send_to_window(window_id, pending_text)
    except OSError, RuntimeError, ValueError:
        pending_store[pending_key] = pending_text
        logger.exception("Failed to send voice transcription to window")
        await query.answer("❌ Failed to send voice transcription", show_alert=True)
        return

    if success:
        try:
            await msg.delete()
        except TelegramError as e:
            logger.warning("Failed to delete voice confirm message: %s", e)
        await query.answer("✓ Sent")
    else:
        pending_store[pending_key] = pending_text
        await query.answer(f"❌ {err}", show_alert=True)


async def _handle_drop(
    msg: Message,
    query: CallbackQuery,
    message_id: int,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle vc:drop — discard the transcription and delete the confirm message."""
    if context.user_data is not None:
        pending_key = _pending_voice_key(msg.chat.id, message_id)
        context.user_data.get(VOICE_PENDING, {}).pop(pending_key, None)

    try:
        await msg.delete()
    except TelegramError as e:
        logger.warning("Failed to delete voice confirm message on discard: %s", e)

    await query.answer("Discarded")
