"""Slash-command forward orchestration for the active provider.

Routes unknown Telegram /commands to the topic's provider session via
tmux. This is the single entry point that ``handlers/registry.py`` wires
into the PTB COMMAND fallback handler.

Pipeline:
  1. resolve user/topic/window/provider
  2. translate the Telegram-friendly /-name back to the provider-native
     name (via menu_sync._build_provider_command_metadata)
  3. capture transcript + pane probe context
  4. send via tmux — any /<token> is forwarded as-is; unknown commands
     are surfaced reactively by the failure probe, not pre-rejected
  5. spawn the failure probe + status snapshot fallbacks
  6. handle /clear post-send cleanup (clear session, reset polling)
"""

from __future__ import annotations


from typing import TYPE_CHECKING

import structlog
from telegram import Message, Update
from telegram.constants import ChatAction

from ...config import config
from ...providers import (
    get_provider_for_window,
)
from ... import window_query
from ...telegram_client import PTBTelegramClient
from ...window_state_store import window_store
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager
from ...multiplexer.window_ops import send_followup_to_window, send_to_window
from ..callback_helpers import get_thread_id as _get_thread_id
from ..command_history import record_command
from ..messaging_pipeline.message_queue import enqueue_status_update
from ..messaging_pipeline.message_sender import safe_reply
from ..polling.polling_state import lifecycle_strategy, reset_window_polling_state
from .failure_probe import (
    _capture_command_probe_context,
    _spawn_command_failure_probe,
)
from .menu_sync import (
    _build_provider_command_metadata,
    sync_scoped_provider_menu,
)
from .status_snapshot import (
    _maybe_send_status_snapshot,
    _status_snapshot_probe_offset,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()


_NAV_KEYS = ("up", "down", "enter", "esc")
_PI_CLEAR_ALIAS_COMMAND = "clear"
_PI_FOLLOWUP_COMMAND = "followup"
_PI_NEW_COMMAND = "new"


def _picker_hint(provider_name: str) -> str:
    """Return the picker-hint suffix for ``provider_name``.

    Inspects the resolved toolbar layout — if the user customised
    ``toolbar.toml`` and dropped the navigation keys, emit the degraded
    hint that just points at ``/toolbar`` rather than promising buttons
    that don't exist.
    """
    # Lazy: handlers.toolbar pulls in PTB Application machinery; keep it
    # out of forward's module-load path.
    from ..toolbar.toolbar_keyboard import get_toolbar_config

    try:
        layout = get_toolbar_config().for_provider(provider_name)
        present = {name for row in layout.buttons for name in row}
    except Exception:  # noqa: BLE001 — toolbar lookup must never break forwarding
        present = set()
    if all(k in present for k in _NAV_KEYS):
        return "\n💡 Open /toolbar to drive the picker — 🔼 🔽 Enter Esc."
    return "\n💡 Open /toolbar to drive the picker."


def _default_command_args(cc_name: str, args: str, display: str) -> str:
    """Fill provider command args that default to the current display name."""
    if args or cc_name not in ("remote-control", "rc"):
        return args
    return display


def _provider_command_name(provider_name: str, cc_name: str) -> str:
    """Apply provider-specific compatibility aliases before forwarding."""
    if provider_name == "pi" and cc_name.lower() == _PI_CLEAR_ALIAS_COMMAND:
        return _PI_NEW_COMMAND
    return cc_name


def _is_session_reset_command(provider_name: str, cc_slash: str) -> bool:
    """Return whether the forwarded command starts a fresh provider session."""
    parts = cc_slash.split(None, 1)
    command = parts[0].lstrip("/").lower()
    if len(parts) > 1:
        return False
    return command == _PI_CLEAR_ALIAS_COMMAND or (
        provider_name == "pi" and command == _PI_NEW_COMMAND
    )


async def _handle_pi_followup_command(
    message: Message,
    user_id: int,
    window_id: str,
    display: str,
    args: str,
    cc_slash: str,
    thread_id: int | None,
) -> None:
    """Queue a Pi follow-up message via Alt+Enter."""
    if not args:
        await safe_reply(message, "Usage: `/followup <message>`")
        return
    logger.info("Forwarding Pi follow-up to window %s (user=%d)", display, user_id)
    await message.get_bot().send_chat_action(
        chat_id=message.chat.id,
        message_thread_id=thread_id,
        action=ChatAction.TYPING,
    )
    success, error_msg = await send_followup_to_window(window_id, args)
    if not success:
        await safe_reply(message, f"❌ {error_msg}")
        return
    if thread_id is not None:
        record_command(user_id, thread_id, cc_slash)
    await safe_reply(message, f"⏭️ [{display}] Follow-up queued.")


async def _handle_session_reset_command(
    update: Update,
    user_id: int,
    window_id: str,
    display: str,
    cc_slash: str,
    thread_id: int | None,
    provider_name: str,
) -> None:
    """Handle post-send cleanup when a provider starts a fresh session."""
    if not _is_session_reset_command(provider_name, cc_slash):
        return
    logger.info("Clearing session for window %s after %s", display, cc_slash)
    window_store.clear_window_session(window_id)

    await enqueue_status_update(
        PTBTelegramClient(update.get_bot()),
        user_id,
        window_id,
        None,
        thread_id=thread_id,
    )
    reset_window_polling_state(window_id)


def _arm_rc_probe_if_remote_control(
    update: Update, window_id: str, cc_name: str
) -> None:
    """Arm the RC outcome probe after a forwarded /remote-control.

    Claude-only: ``arm_rc_probe``'s capability gate no-ops for every
    other provider, so no per-provider branch is needed here.
    """
    if cc_name not in ("remote-control", "rc"):
        return
    # Lazy: rc_probe → providers/messaging_pipeline; keep it out of
    # forward's module-load path.
    from ..status.rc_probe import arm_rc_probe

    arm_rc_probe(window_id, PTBTelegramClient(update.get_bot()))


async def forward_command_handler(
    update: Update, _context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Forward any non-bot command as a slash command to the topic provider session."""
    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return
    if not update.message:
        return

    chat = update.message.chat
    thread_id = _get_thread_id(update)
    if chat.type in ("group", "supergroup") and thread_id is not None:
        thread_router.set_group_chat_id(user.id, thread_id, chat.id)

    cmd_text = update.message.text or ""
    parts = cmd_text.split(None, 1)
    raw_cmd = parts[0].split("@")[0] if parts else ""
    tg_cmd = raw_cmd.lstrip("/").lower()
    # args is forwarded verbatim to the provider via tmux send-keys -l (literal mode).
    # Gated by config.is_user_allowed — authorised users can type anything into their own agent.
    args = parts[1] if len(parts) > 1 else ""
    window_id = thread_router.resolve_window_for_thread(user.id, thread_id)
    if not window_id:
        await safe_reply(update.message, "❌ No session bound to this topic.")
        return

    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        display = thread_router.get_display_name(window_id)
        await safe_reply(update.message, f"❌ Window '{display}' no longer exists.")
        return

    display = thread_router.get_display_name(window_id)
    provider = get_provider_for_window(
        window_id, provider_name=window_query.get_window_provider(window_id)
    )
    provider_name = provider.capabilities.name
    await sync_scoped_provider_menu(update.message, user.id, provider)
    provider_map = _build_provider_command_metadata(provider)
    resolved_name = provider_map.get(tg_cmd, tg_cmd)
    cc_name = _provider_command_name(provider_name, resolved_name.lstrip("/"))
    args = _default_command_args(cc_name, args, display)
    cc_slash = f"/{cc_name} {args}".rstrip() if args else f"/{cc_name}"

    if provider_name == "pi" and cc_name.lower() == _PI_FOLLOWUP_COMMAND:
        await _handle_pi_followup_command(
            update.message, user.id, window_id, display, args, cc_slash, thread_id
        )
        return

    logger.info(
        "Forwarding command %s to window %s (user=%d)", cc_slash, display, user.id
    )
    await update.message.get_bot().send_chat_action(
        chat_id=update.message.chat.id,
        message_thread_id=thread_id,
        action=ChatAction.TYPING,
    )
    (
        probe_transcript_path,
        probe_transcript_offset,
        probe_pane_before,
    ) = await _capture_command_probe_context(window_id, provider)
    status_probe_offset = _status_snapshot_probe_offset(window_id, cc_slash)

    lifecycle_strategy.clear_probe_failures(window_id)
    success, error_msg = await send_to_window(window_id, cc_slash)
    if not success:
        await safe_reply(update.message, f"❌ {error_msg}")
        return

    if thread_id is not None:
        record_command(user.id, thread_id, cc_slash)
    confirmation = f"⚡ [{display}] Sent: {cc_slash}"
    # Picker hint only fires when no args were forwarded: picker commands that
    # accept a direct value (e.g. /model gpt-5) apply it without opening the
    # modal, so the "type /toolbar" guidance would mislead.
    if not args and cc_name.lower() in provider.capabilities.tui_picker_commands:
        confirmation += _picker_hint(provider_name)
    await safe_reply(update.message, confirmation)
    _arm_rc_probe_if_remote_control(update, window_id, cc_name)
    await _maybe_send_status_snapshot(
        update.message,
        window_id,
        display,
        cc_slash,
        since_offset=status_probe_offset,
    )
    _spawn_command_failure_probe(
        update.message,
        window_id,
        display,
        cc_slash,
        provider=provider,
        transcript_path=probe_transcript_path,
        since_offset=probe_transcript_offset,
        pane_before=probe_pane_before,
    )
    await _handle_session_reset_command(
        update, user.id, window_id, display, cc_slash, thread_id, provider_name
    )
