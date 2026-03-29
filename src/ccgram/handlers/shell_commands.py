"""Shell command generation and approval flow.

Handles the NL description -> LLM -> suggested command -> approval keyboard
flow for the shell provider. Also handles raw command execution via ``!`` prefix.

Key components:
  - handle_shell_message: Route shell text (NL or raw ``!`` command)
  - handle_shell_callback: Dispatch approval keyboard callbacks
  - clear_shell_pending: Cleanup for topic deletion
"""

import asyncio
import functools
import os
import re
import shutil

import structlog

from telegram import (
    Bot,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..llm import get_completer
from ..llm import CommandResult
from ..session import session_manager
from ..thread_router import thread_router
from ..tmux_manager import tmux_manager
from .callback_data import (
    CB_SHELL_CANCEL,
    CB_SHELL_CONFIRM_DANGER,
    CB_SHELL_EDIT,
    CB_SHELL_RUN,
)
from .callback_helpers import get_thread_id
from .callback_registry import register
from .message_sender import safe_edit, safe_reply, safe_send
from .message_queue import enqueue_status_update
from .polling_strategies import clear_probe_failures

logger = structlog.get_logger()

_shell_pending: dict[tuple[int, int], tuple[str, int]] = {}
_generation_counter: dict[tuple[int, int], int] = {}


_MODERN_TOOLS: dict[str, str] = {
    "fd": "find replacement (use fd syntax: fd PATTERN, fd --type file, NOT find syntax)",
    "rg": "grep replacement (use rg PATTERN, NOT grep syntax)",
    "bat": "cat replacement",
    "eza": "ls replacement (use eza, NOT ls)",
    "sd": "sed replacement (use sd 'from' 'to', NOT sed syntax)",
    "dust": "du replacement (use dust, NOT du)",
    "procs": "ps replacement",
}


@functools.cache
def _detect_shell_tools() -> str:
    """Detect available modern CLI tools on PATH (cached)."""
    available = []
    for tool, desc in _MODERN_TOOLS.items():
        if shutil.which(tool):
            available.append(f"{tool} ({desc})")
    return ", ".join(available)


# Patterns redacted from terminal output before sending to the LLM
_SENSITIVE_RE = re.compile(
    r"(?i)"
    r"(?:export\s+\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)\w*\s*=\s*\S+)"
    r"|(?:(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)\w*\s*[=:]\s*['\"]?\S{8,})"
    r"|(?:(?:sk|pk|ghp|gho|ghu|ghs|ghr|glpat|xoxb|xoxp|xoxs|AKIA)-[A-Za-z0-9_/+=]{10,})"
    r"|(?:Bearer\s+[A-Za-z0-9_.+/=-]{20,})",
)


def redact_for_llm(text: str) -> str:
    """Strip sensitive patterns from terminal text before sending to an LLM."""
    return _SENSITIVE_RE.sub("[REDACTED]", text)


async def gather_llm_context(window_id: str) -> dict[str, str]:
    """Gather cwd, shell type, and available tools for LLM calls."""
    from ..providers.shell import detect_pane_shell

    shell = await detect_pane_shell(window_id)
    tools = _detect_shell_tools()
    cwd = session_manager.get_window_state(window_id).cwd or ""
    return {"cwd": cwd, "shell": shell, "shell_tools": tools}


def has_shell_pending(chat_id: int, thread_id: int) -> bool:
    """Check if there is a pending shell command for this topic."""
    return (chat_id, thread_id) in _shell_pending


def clear_shell_pending(chat_id: int, thread_id: int) -> None:
    """Clear any pending shell command for this topic (used by cleanup)."""
    _shell_pending.pop((chat_id, thread_id), None)
    _generation_counter.pop((chat_id, thread_id), None)


async def _ensure_prompt_marker(window_id: str) -> None:
    """Lazily restore prompt marker if lost (exec bash, profile reload)."""
    from ..providers.shell import has_prompt_marker, setup_shell_prompt

    if not await has_prompt_marker(window_id):
        await setup_shell_prompt(window_id, clear=False)


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
    from ..providers.shell import KNOWN_SHELLS, match_prompt

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
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
    text: str,
    message: Message | None = None,
) -> None:
    """Route shell provider messages: ``!`` prefix = raw, else = NL via LLM."""
    await enqueue_status_update(bot, user_id, window_id, None, thread_id)
    clear_probe_failures(window_id)

    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    clear_shell_pending(chat_id, thread_id)
    await _ensure_prompt_marker(window_id)

    if text.startswith("!"):
        raw = text[1:].lstrip()
        if not raw:
            return
        await _execute_raw_command(bot, user_id, thread_id, window_id, raw)
        return

    try:
        completer = get_completer()
    except ValueError:
        logger.warning("LLM misconfigured")
        await safe_send(
            bot,
            chat_id,
            "\u26a0 LLM misconfigured \u2014 command not sent.\n"
            "Use `!` prefix for raw commands.",
            message_thread_id=thread_id,
        )
        return

    if not completer:
        # No LLM configured — raw mode is intentional
        await _execute_raw_command(bot, user_id, thread_id, window_id, text)
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
        result = await completer.generate_command(
            text,
            cwd=ctx["cwd"],
            shell=ctx["shell"],
            shell_tools=ctx["shell_tools"],
            recent_output=recent_output,
        )
    except RuntimeError:
        logger.warning("LLM command generation failed")
        await safe_send(
            bot,
            chat_id,
            "\u26a0 LLM request failed \u2014 command not sent.\n"
            "Use `!` prefix for raw commands.",
            message_thread_id=thread_id,
        )
        return

    if _generation_counter.get(gen_key) != gen_id:
        return

    from .command_history import record_command

    record_command(user_id, thread_id, text)

    await show_command_approval(
        bot, chat_id, thread_id, window_id, result, user_id, message
    )


async def _execute_raw_command(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
    command: str,
) -> None:
    """Send a raw command to the shell and start output capture."""
    await _cancel_stuck_input(window_id)

    success, err_message = await session_manager.send_to_window(
        window_id, command, raw=True
    )
    if not success:
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        await safe_send(
            bot, chat_id, f"\u274c {err_message}", message_thread_id=thread_id
        )
        return

    from .shell_capture import mark_telegram_command

    mark_telegram_command(window_id, command, user_id, thread_id)


async def show_command_approval(
    bot: Bot,
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

    # Reserve the slot before awaiting to prevent concurrent callers
    # from emitting duplicate approval keyboards for the same topic.
    _shell_pending[key] = (result.command, user_id)

    text = f"`{result.command}`"
    if result.explanation:
        text += f"\n{result.explanation}"
    if result.is_dangerous:
        text = f"\u26a0\ufe0f *Potentially dangerous*\n{text}"

    keyboard = _build_approval_keyboard(window_id, result.is_dangerous)
    try:
        if message:
            await safe_reply(message, text, reply_markup=keyboard)
        else:
            await safe_send(
                bot, chat_id, text, message_thread_id=thread_id, reply_markup=keyboard
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
                        "\u26a0 Confirm Run",
                        callback_data=f"{CB_SHELL_CONFIRM_DANGER}{window_id}",
                    ),
                    InlineKeyboardButton(
                        "\u2715 Cancel",
                        callback_data=f"{CB_SHELL_CANCEL}{window_id}",
                    ),
                ],
            ]
        )
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "\u25b6 Run",
                    callback_data=f"{CB_SHELL_RUN}{window_id}",
                ),
                InlineKeyboardButton(
                    "\u270f Edit",
                    callback_data=f"{CB_SHELL_EDIT}{window_id}",
                ),
                InlineKeyboardButton(
                    "\u2715 Cancel",
                    callback_data=f"{CB_SHELL_CANCEL}{window_id}",
                ),
            ],
        ]
    )


async def handle_shell_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    bot: Bot,
    thread_id: int | None,
) -> None:
    """Handle shell command approval callbacks."""
    if thread_id is None:
        await query.answer("No topic context")
        return

    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    pending = _shell_pending.get((chat_id, thread_id))

    if data.startswith(CB_SHELL_RUN) or data.startswith(CB_SHELL_CONFIRM_DANGER):
        await _cb_run(query, bot, user_id, thread_id, chat_id, pending)
    elif data.startswith(CB_SHELL_EDIT):
        await _cb_edit(query, user_id, chat_id, thread_id, pending)
    elif data.startswith(CB_SHELL_CANCEL):
        await _cb_cancel(query, user_id, chat_id, thread_id, pending)


async def _cb_run(
    query: CallbackQuery,
    bot: Bot,
    user_id: int,
    thread_id: int,
    chat_id: int,
    pending: tuple[str, int] | None,
) -> None:
    """Handle Run / Confirm Danger callbacks."""
    await query.answer()
    if not pending:
        await safe_edit(query, "\u274c Command expired")
        return

    command, pending_user_id = pending
    if pending_user_id != user_id:
        await safe_edit(query, "\u274c Not your command")
        return

    # Use window from thread binding (authoritative), not callback data
    window_id = thread_router.get_window_for_thread(user_id, thread_id)
    if not window_id:
        clear_shell_pending(chat_id, thread_id)
        await safe_edit(query, "\u274c No session bound")
        return

    clear_shell_pending(chat_id, thread_id)
    await safe_edit(query, f"\u25b6 `{command}`")
    await _execute_raw_command(bot, user_id, thread_id, window_id, command)


async def _cb_edit(
    query: CallbackQuery,
    user_id: int,
    chat_id: int,
    thread_id: int,
    pending: tuple[str, int] | None,
) -> None:
    """Handle Edit callback."""
    await query.answer()
    if pending and pending[1] != user_id:
        await safe_edit(query, "\u274c Not your command")
        return
    clear_shell_pending(chat_id, thread_id)
    if pending:
        await safe_edit(
            query,
            f"\U0001f4cb Copy, edit, and send back:\n`{pending[0]}`",
        )
    else:
        await safe_edit(query, "\u274c Command expired")


async def _cb_cancel(
    query: CallbackQuery,
    user_id: int,
    chat_id: int,
    thread_id: int,
    pending: tuple[str, int] | None,
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
    await handle_shell_callback(query, user.id, query.data, context.bot, thread_id)
