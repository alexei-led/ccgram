"""Shell command generation and approval flow.

Handles the NL description -> LLM -> suggested command -> approval keyboard
flow for the shell provider. Also handles raw command execution via ``!`` prefix.

Key components:
  - handle_shell_message: Route shell text (NL or raw ``!`` command)
  - handle_shell_callback: Dispatch approval keyboard callbacks
  - clear_shell_pending: Cleanup for topic deletion
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import structlog

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import TelegramError
from ...telegram_client import PTBTelegramClient, TelegramClient

from ...llm import get_completer
from ...llm import CommandResult
from ...thread_router import thread_router
from ...tmux_manager import send_to_window, tmux_manager
from ..callback_data import (
    CB_SHELL_CANCEL,
    CB_SHELL_CONFIRM_DANGER,
    CB_SHELL_EDIT,
    CB_SHELL_RUN,
)
from ..callback_helpers import get_thread_id
from ..callback_registry import register
from ..messaging_pipeline.message_sender import (
    REACT_RUNNING,
    react,
    safe_edit,
    safe_reply,
    safe_send,
)
from ..messaging_pipeline.message_queue import enqueue_status_update
from ..polling.polling_state import lifecycle_strategy
from ...topic_state_registry import topic_state

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

# pending entry = (command, user_id, source_message_id) — the message id
# powers ⚡/✅/❌ reaction feedback over the command lifecycle.
_shell_pending: dict[tuple[int, int], tuple[str, int, int]] = {}
_generation_counter: dict[tuple[int, int], int] = {}
_shell_hint_seen: set[tuple[int, int]] = set()

_BANG_HINT_TEXT = (
    "\U0001f4a1 Tip: prefix `!` to skip the LLM and run the command directly."
)

# Telegram chat-action expires after ~5s; refresh slightly faster.
_TYPING_REFRESH_INTERVAL = 4.0


async def _send_typing(client: TelegramClient, chat_id: int, thread_id: int) -> None:
    """Fire one ``typing`` chat action; swallow non-fatal transport errors."""
    try:
        await client.send_chat_action(
            chat_id=chat_id,
            message_thread_id=thread_id,
            action=ChatAction.TYPING,
        )
    except TelegramError, OSError:
        logger.debug("send_chat_action failed", exc_info=True)


@asynccontextmanager
async def _typing_pulse(
    client: TelegramClient, chat_id: int, thread_id: int
) -> AsyncIterator[None]:
    """Keep the ``typing`` indicator alive while a slow op runs.

    The indicator self-expires after ~5s server-side, so refresh every
    ``_TYPING_REFRESH_INTERVAL`` seconds. Cancellation on context exit is
    sufficient to stop the loop — Telegram has no explicit cancel.
    """

    async def pulse() -> None:
        while True:
            await _send_typing(client, chat_id, thread_id)
            await asyncio.sleep(_TYPING_REFRESH_INTERVAL)

    task = asyncio.create_task(pulse())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


# gather_llm_context, redact_for_llm, and _detect_shell_tools moved to
# shell_context.py — re-exported here for callers that haven't been migrated.
from .shell_context import (  # noqa: E402, F401
    gather_llm_context,
    redact_for_llm,
)


def has_shell_pending(chat_id: int, thread_id: int) -> bool:
    """Check if there is a pending shell command for this topic."""
    return (chat_id, thread_id) in _shell_pending


@topic_state.register("chat")
def clear_shell_pending(chat_id: int, thread_id: int) -> None:
    """Clear any pending shell command for this topic (used by cleanup)."""
    _shell_pending.pop((chat_id, thread_id), None)
    _generation_counter.pop((chat_id, thread_id), None)


@topic_state.register("chat")
def _clear_shell_hint_seen(chat_id: int, thread_id: int) -> None:
    """Forget the once-per-session bang hint when the topic is torn down."""
    _shell_hint_seen.discard((chat_id, thread_id))


async def _ensure_prompt_marker(window_id: str) -> None:
    """Lazily restore prompt marker if lost (exec bash, profile reload)."""
    # Lazy: sibling cycle — shell_prompt_orchestrator imports
    # shell_commands.handle_shell_message via callback dispatch.
    # Lazy: prompt orchestrator ↔ commands cycle
    from .shell_prompt_orchestrator import ensure_setup

    await ensure_setup(window_id, "lazy")


async def _cancel_stuck_input(window_id: str) -> None:
    """Send Ctrl+C if the shell is stuck in partial/continuation input.

    Uses two signals to distinguish stuck-at-continuation from running-command:

    1. ``pane_current_command`` — when a shell is idle at *any* prompt
       (including continuation), the foreground process is the shell itself
       (fish/bash/zsh).  When a command is running, it's that command
       (python/grep/etc).  Only proceed if the foreground is a known shell.
    2. Last pane line — if it's a clean bare ``ccgram:N❯`` prompt, the
       shell is ready.  Otherwise (continuation marker like ``...>``, or
       partial typed text), it's stuck.

    This avoids interrupting running commands while still recovering from
    LLM-generated malformed commands that leave the shell in multi-line
    input mode (e.g. unclosed ``begin`` block in fish).
    """
    # Lazy: providers.shell pulls in shell_infra (prompt-marker
    # machinery); load only when we actually need to inspect a prompt.
    # Lazy: providers.shell imports KNOWN_SHELLS late to avoid the providers package bootstrap
    from ...providers.shell import KNOWN_SHELLS, match_prompt

    # Step 1: check if the shell itself is the foreground process.
    # If a command is running (python, grep, etc.), don't interrupt it.
    window = await tmux_manager.find_window_by_id(window_id)
    if not window or not window.pane_current_command:
        return
    tokens = window.pane_current_command.split()
    if not tokens:
        return
    foreground = os.path.basename(tokens[0]).lstrip("-")
    if foreground not in KNOWN_SHELLS:
        return  # a command is running — don't interrupt

    # Step 2: check if the last pane line is a clean prompt.
    raw = await tmux_manager.capture_pane(window_id)
    if not raw:
        return
    lines = raw.rstrip().splitlines()
    if not lines:
        return

    last = lines[-1]
    m = match_prompt(last)
    if m and not m.trailing_text.strip():
        return  # clean bare prompt — all good

    # Shell is idle but NOT at a clean prompt → continuation or partial input.
    logger.debug("Cancelling stuck input in window %s", window_id)
    await tmux_manager.send_keys(window_id, "C-c", enter=False, literal=False)
    await asyncio.sleep(0.3)


async def handle_shell_message(
    client: TelegramClient,
    user_id: int,
    thread_id: int,
    window_id: str,
    text: str,
    message: Message | None = None,
) -> None:
    """Route shell provider messages: ``!`` prefix = raw, else = NL via LLM."""
    await enqueue_status_update(client, user_id, window_id, None, thread_id)
    lifecycle_strategy.clear_probe_failures(window_id)

    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    clear_shell_pending(chat_id, thread_id)
    await _ensure_prompt_marker(window_id)

    # Immediate ``typing`` action so the user sees the bot is processing
    # before any reply lands. Self-expires after ~5s; pulse() refreshes it
    # for the duration of the LLM call below.
    await _send_typing(client, chat_id, thread_id)

    msg_id = message.message_id if message is not None else 0

    # ⚡ Persistent ack so the user sees the bot is working before
    # tmux/LLM round-trip completes; replaced by ✅/❌ once exit is known.
    if message is not None:
        await react(client, message.chat.id, message.message_id, REACT_RUNNING)

    if text.startswith("!"):
        raw = text[1:].lstrip()
        if not raw:
            return
        await _execute_raw_command(
            client, user_id, thread_id, window_id, raw, message_id=msg_id
        )
        return

    try:
        completer = get_completer()
    except ValueError as exc:
        logger.warning("LLM misconfigured: %s", exc)
        await safe_send(
            client,
            chat_id,
            "⚠ LLM misconfigured — command not sent.\nUse `!` prefix for raw commands.",
            message_thread_id=thread_id,
        )
        return

    if not completer:
        # No LLM configured — raw mode is intentional
        await _execute_raw_command(
            client, user_id, thread_id, window_id, text, message_id=msg_id
        )
        return

    ctx = await gather_llm_context(window_id)
    recent_output = ""
    raw_pane = await tmux_manager.capture_pane(window_id)
    if raw_pane:
        lines = raw_pane.strip().splitlines()
        recent_output = redact_for_llm("\n".join(lines[-10:]))

    gen_key = (chat_id, thread_id)
    gen_id = _generation_counter.get(gen_key, 0) + 1
    _generation_counter[gen_key] = gen_id

    try:
        async with _typing_pulse(client, chat_id, thread_id):
            result = await completer.generate_command(
                text,
                cwd=ctx["cwd"],
                shell=ctx["shell"],
                shell_tools=ctx["shell_tools"],
                recent_output=recent_output,
            )
    except RuntimeError as exc:
        logger.warning("LLM command generation failed: %s", exc)
        await safe_send(
            client,
            chat_id,
            "⚠ LLM request failed — command not sent.\n"
            "Use `!` prefix for raw commands.",
            message_thread_id=thread_id,
        )
        return

    if _generation_counter.get(gen_key) != gen_id:
        return

    # Lazy: command_history → messaging_pipeline → status → shell cycle
    # (same shape as the status_bubble revert in F6.2).
    # Lazy: command_history ↔ shell cycle through status_bar_actions
    from ..command_history import record_command

    record_command(user_id, thread_id, text)

    await _show_bang_hint_once(client, chat_id, thread_id)

    await show_command_approval(
        client, chat_id, thread_id, window_id, result, user_id, message
    )


async def _show_bang_hint_once(
    client: TelegramClient, chat_id: int, thread_id: int
) -> None:
    """Send the once-per-session ``!`` prefix tip the first time it would help."""
    key = (chat_id, thread_id)
    if key in _shell_hint_seen:
        return
    _shell_hint_seen.add(key)
    await safe_send(client, chat_id, _BANG_HINT_TEXT, message_thread_id=thread_id)


async def _execute_raw_command(
    client: TelegramClient,
    user_id: int,
    thread_id: int,
    window_id: str,
    command: str,
    message_id: int = 0,
) -> None:
    """Send a raw command to the shell and start output capture.

    ``message_id`` (when non-zero) is the user's source Telegram message —
    forwarded to ``mark_telegram_command`` so the passive monitor can react
    on it once the exit code is known.
    """
    await _cancel_stuck_input(window_id)

    success, err_message = await send_to_window(window_id, command, raw=True)
    if not success:
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        await safe_send(
            client,
            chat_id,
            f"❌ {err_message}",
            message_thread_id=thread_id,
        )
        return

    # Lazy: sibling cycle — shell_capture imports show_command_approval
    # from this module via the approval callback wiring.
    # Lazy: shell_capture ↔ shell_commands cycle
    from .shell_capture import mark_telegram_command

    mark_telegram_command(window_id, command, user_id, thread_id, message_id)


async def show_command_approval(
    client: TelegramClient,
    chat_id: int,
    thread_id: int,
    window_id: str,
    result: CommandResult,
    user_id: int,
    message: Message | None = None,
) -> bool:
    """Show a suggested command with approval keyboard.

    Returns True if the command was stored, False if the slot was already
    occupied (avoids overwriting a user's pending command with an auto-fix).
    """
    key = (chat_id, thread_id)
    if key in _shell_pending:
        return False

    msg_id = message.message_id if message is not None else 0

    # Reserve the slot before awaiting to prevent concurrent callers
    # from emitting duplicate approval keyboards for the same topic.
    _shell_pending[key] = (result.command, user_id, msg_id)

    text = f"`{result.command}`"
    if result.explanation:
        text += f"\n{result.explanation}"
    if result.is_dangerous:
        text = f"⚠️ *Potentially dangerous*\n{text}"

    keyboard = _build_approval_keyboard(window_id, result.is_dangerous)
    try:
        if message:
            await safe_reply(message, text, reply_markup=keyboard)
        else:
            await safe_send(
                client,
                chat_id,
                text,
                message_thread_id=thread_id,
                reply_markup=keyboard,
            )
    except TelegramError, OSError:
        # If send fails, release the slot so future attempts aren't blocked
        _shell_pending.pop(key, None)
        raise
    return True


def _build_approval_keyboard(
    window_id: str, is_dangerous: bool
) -> InlineKeyboardMarkup:
    """Build the command approval inline keyboard."""
    if is_dangerous:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "⚠ Confirm Run",
                        callback_data=f"{CB_SHELL_CONFIRM_DANGER}{window_id}",
                    ),
                    InlineKeyboardButton(
                        "✕ Cancel",
                        callback_data=f"{CB_SHELL_CANCEL}{window_id}",
                    ),
                ],
            ]
        )
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "▶ Run",
                    callback_data=f"{CB_SHELL_RUN}{window_id}",
                ),
                InlineKeyboardButton(
                    "✏ Edit",
                    callback_data=f"{CB_SHELL_EDIT}{window_id}",
                ),
                InlineKeyboardButton(
                    "✕ Cancel",
                    callback_data=f"{CB_SHELL_CANCEL}{window_id}",
                ),
            ],
        ]
    )


async def handle_shell_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    client: TelegramClient,
    thread_id: int | None,
) -> None:
    """Handle shell command approval callbacks."""
    if thread_id is None:
        await query.answer("No topic context")
        return

    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    pending = _shell_pending.get((chat_id, thread_id))

    if data.startswith(CB_SHELL_RUN) or data.startswith(CB_SHELL_CONFIRM_DANGER):
        await _cb_run(query, client, user_id, thread_id, chat_id, pending)
    elif data.startswith(CB_SHELL_EDIT):
        await _cb_edit(query, user_id, chat_id, thread_id, pending)
    elif data.startswith(CB_SHELL_CANCEL):
        await _cb_cancel(query, user_id, chat_id, thread_id, pending)


async def _cb_run(
    query: CallbackQuery,
    client: TelegramClient,
    user_id: int,
    thread_id: int,
    chat_id: int,
    pending: tuple[str, int, int] | None,
) -> None:
    """Handle Run / Confirm Danger callbacks."""
    await query.answer()
    if not pending:
        await safe_edit(query, "❌ Command expired")
        return

    command, pending_user_id, msg_id = pending
    if pending_user_id != user_id:
        await safe_edit(query, "❌ Not your command")
        return

    # Use window from thread binding (authoritative), not callback data
    window_id = thread_router.get_window_for_thread(user_id, thread_id)
    if not window_id:
        clear_shell_pending(chat_id, thread_id)
        await safe_edit(query, "❌ No session bound")
        return

    clear_shell_pending(chat_id, thread_id)
    await safe_edit(query, f"▶ `{command}`")
    await _execute_raw_command(
        client, user_id, thread_id, window_id, command, message_id=msg_id
    )


async def _cb_edit(
    query: CallbackQuery,
    user_id: int,
    chat_id: int,
    thread_id: int,
    pending: tuple[str, int, int] | None,
) -> None:
    """Handle Edit callback."""
    await query.answer()
    if pending and pending[1] != user_id:
        await safe_edit(query, "❌ Not your command")
        return
    clear_shell_pending(chat_id, thread_id)
    if pending:
        await safe_edit(
            query,
            f"\U0001f4cb Copy, edit, and send back:\n`{pending[0]}`",
        )
    else:
        await safe_edit(query, "❌ Command expired")


async def _cb_cancel(
    query: CallbackQuery,
    user_id: int,
    chat_id: int,
    thread_id: int,
    pending: tuple[str, int, int] | None,
) -> None:
    """Handle Cancel callback."""
    if pending and pending[1] != user_id:
        await query.answer("Not your command", show_alert=True)
        return
    await query.answer("Cancelled")
    clear_shell_pending(chat_id, thread_id)
    await safe_edit(query, "Cancelled")


# --- Registry dispatch entry point ---


@register(CB_SHELL_RUN, CB_SHELL_EDIT, CB_SHELL_CANCEL, CB_SHELL_CONFIRM_DANGER)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    thread_id = get_thread_id(update)
    await handle_shell_callback(
        query, user.id, query.data, PTBTelegramClient(context.bot), thread_id
    )
