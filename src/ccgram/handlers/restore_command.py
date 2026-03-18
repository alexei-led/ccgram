"""/restore command — auto-recover dead topics.

When a tmux window dies, the topic becomes stale. This command auto-recovers
the session: recreates the window in the same cwd/provider with --continue.

Key function: restore_command().
"""

from pathlib import Path

import contextlib

import structlog
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..config import config
from ..providers import get_provider_for_window, resolve_launch_command
from ..session import session_manager
from ..tmux_manager import tmux_manager
from .message_sender import safe_reply
from .status_polling import clear_dead_notification
from .topic_emoji import format_topic_name_for_mode

logger = structlog.get_logger()


async def restore_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /restore — auto-recover a dead topic with --continue."""
    user = update.effective_user
    if not user or not update.message:
        return

    if not config.is_user_allowed(user.id):
        await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    thread_id = update.message.message_thread_id
    if thread_id is None:
        await safe_reply(update.message, "Use this command inside a topic.")
        return

    user_id = user.id
    window_id = session_manager.resolve_window_for_thread(user_id, thread_id)
    if not window_id:
        await safe_reply(update.message, "No session bound to this topic.")
        return

    window = await tmux_manager.find_window_by_id(window_id)
    if window is not None:
        await safe_reply(
            update.message, "Window is still running — nothing to restore."
        )
        return

    ws = session_manager.get_window_state(window_id)
    cwd = ws.cwd or ""
    if not cwd or not Path(cwd).is_dir():
        await safe_reply(update.message, "Directory no longer exists.")
        return

    # Auto-recover: unbind old, create new window with --continue, rebind
    session_manager.unbind_thread(user_id, thread_id)
    clear_dead_notification(user_id, thread_id)

    provider = get_provider_for_window(window_id)
    approval_mode = session_manager.get_approval_mode(window_id)
    launch_command = resolve_launch_command(
        provider.capabilities.name, approval_mode=approval_mode
    )
    launch_args = provider.make_launch_args(use_continue=True)

    success, message, wname, wid = await tmux_manager.create_window(
        cwd, agent_args=launch_args, launch_command=launch_command
    )
    if not success:
        await safe_reply(update.message, f"\u274c {message}")
        return

    if provider.capabilities.supports_hook:
        await session_manager.wait_for_session_map_entry(wid)

    session_manager.set_window_provider(wid, provider.capabilities.name)
    session_manager.set_window_approval_mode(wid, approval_mode)
    session_manager.bind_thread(user_id, thread_id, wid, window_name=wname)

    with contextlib.suppress(TelegramError):
        await context.bot.edit_forum_topic(
            chat_id=session_manager.resolve_chat_id(user_id, thread_id),
            message_thread_id=thread_id,
            name=format_topic_name_for_mode(wname, approval_mode),
        )

    await safe_reply(
        update.message, f"\u2705 {message}\n\nContinuing previous session."
    )
