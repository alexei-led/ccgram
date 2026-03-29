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

from ..providers import get_provider_for_window
from ..session import session_manager
from ..thread_router import thread_router
from .callback_data import CB_VOICE
from .callback_helpers import get_thread_id
from .callback_registry import register
from .message_sender import ack_reaction
from .user_state import VOICE_PENDING

logger = structlog.get_logger()


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
    pending_store = (
        context.user_data.get(VOICE_PENDING, {}) if context.user_data else {}
    )
    pending_text = pending_store.pop((msg.chat.id, message_id), None)
    if pending_text is None:
        await query.answer("⚠️ Session expired, resend voice message", show_alert=True)
        return

    thread_id = get_thread_id(update)
    window_id = thread_router.resolve_window_for_thread(user_id, thread_id)
    if not window_id:
        pending_store[(msg.chat.id, message_id)] = pending_text
        await query.answer("⚠️ No session bound.", show_alert=True)
        return

    # Shell provider: route through LLM for NL→command generation
    provider = get_provider_for_window(window_id)
    if provider.capabilities.name == "shell" and thread_id is not None:
        from .shell_commands import handle_shell_message

        try:
            await handle_shell_message(
                msg.get_bot(), user_id, thread_id, window_id, pending_text
            )
        except (OSError, TelegramError) as exc:
            logger.warning("Shell message handling failed: %s", exc)
            pending_store[(msg.chat.id, message_id)] = pending_text
            await query.answer("❌ Failed to send", show_alert=True)
            return
        await ack_reaction(msg.get_bot(), msg.chat.id, message_id)
        try:
            await msg.delete()
        except TelegramError as e:
            logger.warning("Failed to delete voice confirm message: %s", e)
        await query.answer("✓ Sent")
        return

    success, err = await session_manager.send_to_window(window_id, pending_text)

    if success:
        await ack_reaction(msg.get_bot(), msg.chat.id, message_id)
        try:
            await msg.delete()
        except TelegramError as e:
            logger.warning("Failed to delete voice confirm message: %s", e)
        await query.answer("✓ Sent")
    else:
        pending_store[(msg.chat.id, message_id)] = pending_text
        await query.answer(f"❌ {err}", show_alert=True)


async def _handle_drop(
    msg: Message,
    query: CallbackQuery,
    message_id: int,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle vc:drop — discard the transcription and delete the confirm message."""
    if context.user_data is not None:
        context.user_data.get(VOICE_PENDING, {}).pop((msg.chat.id, message_id), None)

    try:
        await msg.delete()
    except TelegramError as e:
        logger.warning("Failed to delete voice confirm message on discard: %s", e)

    await query.answer("Discarded")


# --- Registry dispatch entry point ---


@register(CB_VOICE)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_voice_callback(update, context)
