"""Voice transcription callbacks — handle confirm (send to agent) and discard actions.

Handles the inline keyboard callbacks triggered after voice message transcription:
  - vc:send:<msg_id>: Send transcribed text to the bound agent window
  - vc:drop:<msg_id>: Discard the transcription and delete the confirmation message

Key function: handle_voice_callback
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import structlog
from telegram import CallbackQuery, Message, Update
from telegram.error import TelegramError
from ...providers import get_provider_for_window
from ...telegram_client import PTBTelegramClient
from ...window_query import get_window_provider
from ...tmux_manager import send_to_window
from ...thread_router import thread_router
from ..callback_data import CB_VOICE
from ..callback_helpers import get_thread_id
from ..callback_registry import register
from ..messaging_pipeline.message_sender import (
    REACT_DONE,
    REACT_SEEN,
    ack_reaction,
    react,
)
from ..user_state import VOICE_PENDING

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

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

    client = PTBTelegramClient(msg.get_bot())

    # 👀 ack: persistent "I see you" indicator on the original voice message.
    await react(client, msg.chat.id, message_id, REACT_SEEN)

    # Shell provider: route through LLM for NL→command generation
    provider = get_provider_for_window(
        window_id, provider_name=get_window_provider(window_id)
    )
    if provider.capabilities.chat_first_command_path and thread_id is not None:
        # Lazy: shell.shell_commands ↔ voice via approval callback wiring.
        from ..shell.shell_commands import handle_shell_message

        try:
            await handle_shell_message(
                client, user_id, thread_id, window_id, pending_text
            )
        except (OSError, TelegramError) as exc:
            logger.warning("Shell message handling failed: %s", exc)
            pending_store[(msg.chat.id, message_id)] = pending_text
            await query.answer("❌ Failed to send", show_alert=True)
            return
        await _ack_delivered(client, msg, query, message_id)
        return

    success, err = await send_to_window(window_id, pending_text)

    if success:
        await _ack_delivered(client, msg, query, message_id)
    else:
        pending_store[(msg.chat.id, message_id)] = pending_text
        await query.answer(f"❌ {err}", show_alert=True)


async def _ack_delivered(
    client: PTBTelegramClient, msg: Message, query: CallbackQuery, message_id: int
) -> None:
    """Replace the previous "✓ Sent" toast with a persistent reaction.

    Order matters: REACT_DONE replaces the prior 👀; ack_reaction (if user
    configured ``CCGRAM_ACK_REACTION``) overrides REACT_DONE in turn.
    """
    await react(client, msg.chat.id, message_id, REACT_DONE)
    await ack_reaction(client, msg.chat.id, message_id)
    try:
        await msg.delete()
    except TelegramError as e:
        logger.warning("Failed to delete voice confirm message: %s", e)
    await query.answer()


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
