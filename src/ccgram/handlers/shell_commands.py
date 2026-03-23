"""Shell command generation and approval flow.

Handles the NL description -> LLM -> suggested command -> approval keyboard
flow for the shell provider. Also handles raw command execution via ``!`` prefix.
Prompt marker offer UI for shell provider setup on bind/switch.

Key components:
  - handle_shell_message: Route shell text (NL or raw ``!`` command)
  - handle_shell_callback: Dispatch approval keyboard callbacks
  - offer_prompt_setup: Offer inline keyboard to set up prompt marker
  - clear_shell_pending: Cleanup for topic deletion
"""

import asyncio
import os
import shutil

import structlog

from telegram import (
    Bot,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from ..llm import get_completer
from ..llm import CommandResult
from ..session import session_manager
from ..tmux_manager import tmux_manager
from .callback_data import (
    CB_SHELL_CANCEL,
    CB_SHELL_CONFIRM_DANGER,
    CB_SHELL_EDIT,
    CB_SHELL_RUN,
    CB_SHELL_SETUP,
    CB_SHELL_SETUP_SKIP,
)
from .message_sender import safe_edit, safe_reply, safe_send
from .message_queue import enqueue_status_update
from .shell_capture import start_shell_capture
from .status_polling import clear_probe_failures

logger = structlog.get_logger()

_shell_pending: dict[tuple[int, int], tuple[str, int]] = {}
_marker_setup_skipped: set[str] = set()
_generation_counter: dict[tuple[int, int], int] = {}


def clear_marker_skip(window_id: str) -> None:
    """Clear skip flag for a window (used by cleanup and provider switch)."""
    _marker_setup_skipped.discard(window_id)


_MODERN_TOOLS: dict[str, str] = {
    "fd": "find replacement (use fd syntax: fd PATTERN, fd --type file, NOT find syntax)",
    "rg": "grep replacement (use rg PATTERN, NOT grep syntax)",
    "bat": "cat replacement",
    "eza": "ls replacement (use eza, NOT ls)",
    "sd": "sed replacement (use sd 'from' 'to', NOT sed syntax)",
    "dust": "du replacement (use dust, NOT du)",
    "procs": "ps replacement",
}

_cached_shell_tools: str | None = None


def _detect_shell_tools() -> str:
    """Detect available modern CLI tools on PATH (cached)."""
    global _cached_shell_tools  # noqa: PLW0603
    if _cached_shell_tools is not None:
        return _cached_shell_tools

    available = []
    for tool, desc in _MODERN_TOOLS.items():
        if shutil.which(tool):
            available.append(f"{tool} ({desc})")

    _cached_shell_tools = ", ".join(available)
    return _cached_shell_tools


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


async def offer_prompt_setup(
    bot: Bot, user_id: int, thread_id: int, window_id: str
) -> None:
    """Offer to set up ccgram prompt marker via inline keyboard.

    If marker already present in pane capture or user already chose Skip,
    silently skips.  Otherwise sends [Set up] [Skip] keyboard to the topic.
    """
    if window_id in _marker_setup_skipped:
        return
    from ..providers.shell import has_prompt_marker

    if await has_prompt_marker(window_id):
        return
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Set up prompt",
                    callback_data=f"{CB_SHELL_SETUP}{window_id}",
                ),
                InlineKeyboardButton(
                    "Skip",
                    callback_data=f"{CB_SHELL_SETUP_SKIP}{window_id}",
                ),
            ],
        ]
    )
    await safe_send(
        bot,
        chat_id,
        "Shell detected without ccgram prompt marker.\n"
        "Set up for output capture and exit code detection?",
        message_thread_id=thread_id,
        reply_markup=keyboard,
    )


async def _ensure_prompt_marker(window_id: str) -> None:
    """Lazily restore prompt marker if lost (exec bash, profile reload)."""
    if window_id in _marker_setup_skipped:
        return
    from ..providers.shell import has_prompt_marker, setup_shell_prompt

    if not await has_prompt_marker(window_id):
        await setup_shell_prompt(window_id)


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
    from ..providers.shell import KNOWN_SHELLS, PROMPT_RE

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
    m = PROMPT_RE.match(last)
    if m and not m.group(2).strip():
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

    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
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
        recent_output = "\n".join(lines[-10:])

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

    # Capture baseline BEFORE sending so we can diff later
    baseline = ""
    raw_pane = await tmux_manager.capture_pane(window_id)
    if raw_pane:
        baseline = raw_pane.rstrip()

    success, err_message = await session_manager.send_to_window(
        window_id, command, raw=True
    )
    if not success:
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        await safe_send(
            bot, chat_id, f"\u274c {err_message}", message_thread_id=thread_id
        )
        return

    start_shell_capture(
        bot, user_id, thread_id, window_id, baseline=baseline, command=command
    )


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

    text = f"`{result.command}`"
    if result.explanation:
        text += f"\n{result.explanation}"
    if result.is_dangerous:
        text = f"\u26a0\ufe0f *Potentially dangerous*\n{text}"

    keyboard = _build_approval_keyboard(window_id, result.is_dangerous)
    if message:
        await safe_reply(message, text, reply_markup=keyboard)
    else:
        await safe_send(
            bot, chat_id, text, message_thread_id=thread_id, reply_markup=keyboard
        )
    _shell_pending[key] = (result.command, user_id)
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

    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    pending = _shell_pending.get((chat_id, thread_id))

    if data.startswith(CB_SHELL_SETUP_SKIP):
        await _cb_setup_skip(query, data, user_id)
    elif data.startswith(CB_SHELL_SETUP):
        await _cb_setup_prompt(query, data, user_id)
    elif data.startswith(CB_SHELL_RUN) or data.startswith(CB_SHELL_CONFIRM_DANGER):
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
    window_id = session_manager.get_window_for_thread(user_id, thread_id)
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


async def _cb_setup_prompt(query: CallbackQuery, data: str, user_id: int) -> None:
    """Handle CB_SHELL_SETUP: inject prompt marker into the shell."""
    from .callback_helpers import user_owns_window
    from ..providers.shell import setup_shell_prompt

    window_id = data[len(CB_SHELL_SETUP) :]
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your window", show_alert=True)
        return
    _marker_setup_skipped.discard(window_id)
    w = await tmux_manager.find_window_by_id(window_id)
    if not w:
        await safe_edit(query, "\u274c Window no longer exists")
        await query.answer("Window gone", show_alert=True)
        return
    await setup_shell_prompt(window_id)
    await safe_edit(query, "\u2705 Prompt marker configured")
    await query.answer("Done")


async def _cb_setup_skip(query: CallbackQuery, data: str, user_id: int) -> None:
    """Handle CB_SHELL_SETUP_SKIP: skip prompt marker setup."""
    from .callback_helpers import user_owns_window

    window_id = data[len(CB_SHELL_SETUP_SKIP) :]
    if not user_owns_window(user_id, window_id):
        await query.answer("Not your window", show_alert=True)
        return
    _marker_setup_skipped.add(window_id)
    await safe_edit(query, "Skipped \u2014 raw commands work, no exit code detection")
    await query.answer("Skipped")
