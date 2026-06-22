"""Interactive UI handling for Claude Code prompts.

Handles interactive terminal UIs displayed by Claude Code:
  - AskUserQuestion: Multi-choice question prompts
  - ExitPlanMode: Plan mode exit confirmation
  - Permission Prompt: Tool permission requests
  - RestoreCheckpoint: Checkpoint restoration selection

Provides:
  - Keyboard navigation (up/down/left/right/enter/esc)
  - Terminal capture and display
  - Interactive mode tracking per user and thread

State dicts are keyed by (user_id, thread_id_or_0) for Telegram topic support.
"""

import asyncio
import contextlib
import time

import structlog

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.error import BadRequest, NetworkError, RetryAfter, TelegramError, TimedOut

from ...providers import get_provider_for_window
from ...telegram_client import TelegramClient
from ...window_query import get_window_provider
from ...thread_router import thread_router
from ...multiplexer import multiplexer as tmux_manager
from ...topic_state_registry import topic_state
from ..callback_data import (
    CB_ASK_DOWN,
    CB_ASK_ENTER,
    CB_ASK_ESC,
    CB_ASK_LEFT,
    CB_ASK_REFRESH,
    CB_ASK_RIGHT,
    CB_ASK_SPACE,
    CB_ASK_TAB,
    CB_ASK_UP,
)
from ..messaging_pipeline.message_sender import (
    NO_LINK_PREVIEW,
    is_thread_gone,
    rate_limit_send,
)

logger = structlog.get_logger()

# Tool names that trigger interactive UI via JSONL (terminal capture + inline keyboard)
INTERACTIVE_TOOL_NAMES = frozenset(
    {
        "AskUserQuestion",
        "ExitPlanMode",
        # Codex native tool name before normalization/fallback.
        "request_user_input",
    }
)

# Track interactive UI message IDs: (user_id, thread_id_or_0) -> message_id
_interactive_msgs: dict[tuple[int, int], int] = {}

# Track interactive mode: (user_id, thread_id_or_0) -> window_id
_interactive_mode: dict[tuple[int, int], str] = {}

# Cooldown to prevent flood when interactive sends fail repeatedly
_send_cooldowns: dict[tuple[int, int], float] = {}
_SEND_RETRY_INTERVAL = 5.0  # seconds between retries for failed sends
_DEAD_TOPIC_RETRY_INTERVAL = 60.0  # longer backoff when topic is deleted

# Single in-call retry on transient transport errors when sending interactive UI.
_INTERACTIVE_SEND_RETRIES = 1
_INTERACTIVE_SEND_RETRY_BACKOFF_S = 1.0

# One-line cheatsheet prepended to every interactive UI message.
INTERACTIVE_INSTRUCTION_LINE = (
    "↑↓ select · Enter confirm · Esc cancel · type to enter text"
)

# Hard ceiling per Telegram message; leave headroom for entities.
_TELEGRAM_MAX_TEXT = 4096


def format_interactive_message(
    text: str,
    pane_id: str | None = None,
    pane_name: str | None = None,
) -> str:
    """Build the body of an interactive UI message.

    Prepends the navigation instruction line so users see the keyboard
    shortcuts without trial and error, and adds a pane prefix for
    non-active pane alerts. When ``pane_name`` is set, the prefix uses
    it instead of the generic word "Pane" so multi-pane teams surface
    a recognizable label (e.g. ``api-gateway (%5)`` instead of
    ``Pane (%5)``). Truncates the captured terminal text from the top
    (most recent lines win) when the combined message would exceed
    Telegram's 4096-char per-message limit.
    """
    header = INTERACTIVE_INSTRUCTION_LINE
    if pane_id:
        label = pane_name.strip() if pane_name and pane_name.strip() else "Pane"
        header = f"{header}\n\U0001f500 {label} ({pane_id}):"

    body = text
    overhead = len(header) + 1  # +1 for the newline between header and body
    if overhead + len(body) > _TELEGRAM_MAX_TEXT:
        # Drop oldest lines first; tail of the buffer is what the user needs.
        budget = _TELEGRAM_MAX_TEXT - overhead
        body = body[-budget:] if budget > 0 else ""
    return f"{header}\n{body}"


@topic_state.register("topic")
def clear_send_cooldowns(user_id: int, thread_id: int) -> None:
    """Clear send cooldown for this topic (called on topic cleanup)."""
    _send_cooldowns.pop((user_id, thread_id or 0), None)


def get_interactive_window(user_id: int, thread_id: int | None = None) -> str | None:
    """Get the window_id for user's interactive mode."""
    return _interactive_mode.get((user_id, thread_id or 0))


def set_interactive_mode(
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
) -> None:
    """Set interactive mode for a user."""
    logger.debug(
        "Set interactive mode: user=%d, window_id=%s, thread=%s",
        user_id,
        window_id,
        thread_id,
    )
    _interactive_mode[(user_id, thread_id or 0)] = window_id


def clear_interactive_mode(user_id: int, thread_id: int | None = None) -> None:
    """Clear interactive mode for a user (without deleting message)."""
    logger.debug("Clear interactive mode: user=%d, thread=%s", user_id, thread_id)
    _interactive_mode.pop((user_id, thread_id or 0), None)


def get_interactive_msg_id(user_id: int, thread_id: int | None = None) -> int | None:
    """Get the interactive message ID for a user."""
    return _interactive_msgs.get((user_id, thread_id or 0))


def _build_interactive_keyboard(
    window_id: str,
    ui_name: str = "",
    pane_id: str | None = None,
) -> InlineKeyboardMarkup:
    """Build keyboard for interactive UI navigation.

    ``ui_name`` controls the layout: ``RestoreCheckpoint`` omits ←/→ keys
    since only vertical selection is needed.

    When ``pane_id`` is set, it is appended to each callback data so
    responses route to a specific pane instead of the window's active pane.
    """
    # Lazy: pane delimiter constant
    from ..callback_data import CB_PANE_DELIMITER

    vertical_only = ui_name == "RestoreCheckpoint"
    # Target suffix: "@12" or "@12|%5" (tmux) / "w2:t1|w2:p1" (herdr) when pane-targeted
    target = f"{window_id}{CB_PANE_DELIMITER}{pane_id}" if pane_id else window_id

    rows: list[list[InlineKeyboardButton]] = []
    # Row 1: directional keys
    rows.append(
        [
            InlineKeyboardButton(
                "␣ Space", callback_data=f"{CB_ASK_SPACE}{target}"[:64]
            ),
            InlineKeyboardButton("↑", callback_data=f"{CB_ASK_UP}{target}"[:64]),
            InlineKeyboardButton("⇥ Tab", callback_data=f"{CB_ASK_TAB}{target}"[:64]),
        ]
    )
    if vertical_only:
        rows.append(
            [
                InlineKeyboardButton("↓", callback_data=f"{CB_ASK_DOWN}{target}"[:64]),
            ]
        )
    else:
        rows.append(
            [
                InlineKeyboardButton("←", callback_data=f"{CB_ASK_LEFT}{target}"[:64]),
                InlineKeyboardButton("↓", callback_data=f"{CB_ASK_DOWN}{target}"[:64]),
                InlineKeyboardButton("→", callback_data=f"{CB_ASK_RIGHT}{target}"[:64]),
            ]
        )
    # Row 2: action keys
    rows.append(
        [
            InlineKeyboardButton("⎋ Esc", callback_data=f"{CB_ASK_ESC}{target}"[:64]),
            InlineKeyboardButton("🔄", callback_data=f"{CB_ASK_REFRESH}{target}"[:64]),
            InlineKeyboardButton(
                "⏎ Enter", callback_data=f"{CB_ASK_ENTER}{target}"[:64]
            ),
        ]
    )
    return InlineKeyboardMarkup(rows)


async def _edit_interactive_msg(
    client: TelegramClient,
    chat_id: int,
    msg_id: int,
    text: str,
    keyboard: InlineKeyboardMarkup,
    ikey: tuple[int, int],
    window_id: str,
) -> bool | None:
    """Try to edit an existing interactive message.

    Returns True/False on success/failure, or None if no edit was attempted.
    """
    try:
        await client.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text=text,
            reply_markup=keyboard,
            link_preview_options=NO_LINK_PREVIEW,
        )
        _interactive_mode[ikey] = window_id
        return True
    except BadRequest as e:
        if "Message is not modified" in e.message:
            return True  # Content identical, no-op
        logger.warning("BadRequest editing interactive msg: %s", e.message)
        return False
    except RetryAfter:
        raise
    except TelegramError:
        logger.warning("Failed to edit interactive message", exc_info=True)
        return False


async def _capture_interactive_content(
    window_id: str,
    pane_id: str | None = None,
) -> tuple[str, str] | None:
    """Capture pane and extract interactive UI content.

    When *pane_id* is given, captures that specific pane (by stable ``%N`` ID)
    instead of the window's active pane.

    Returns (ui_name, text) if an interactive UI is detected, None otherwise.
    """
    if pane_id:
        pane_text = await tmux_manager.capture_pane_by_id(pane_id, window_id=window_id)
    else:
        w = await tmux_manager.find_window_by_id(window_id)
        if not w:
            return None
        pane_text = await tmux_manager.capture_pane(w.window_id)

    if not pane_text:
        logger.debug(
            "No pane text captured for window_id %s pane_id %s", window_id, pane_id
        )
        return None

    provider = get_provider_for_window(
        window_id, provider_name=get_window_provider(window_id)
    )
    pane_title = ""
    if provider.capabilities.uses_pane_title and not pane_id:
        pane_title = await tmux_manager.get_pane_title(window_id)
    status = provider.parse_terminal_status(pane_text, pane_title=pane_title)
    if status is None or not status.is_interactive:
        return None

    if not status.ui_type:
        logger.warning(
            "Interactive status with no ui_type in window_id %s pane %s",
            window_id,
            pane_id,
        )
        return None

    return status.ui_type, status.raw_text


def _lookup_pane_name(window_id: str, pane_id: str) -> str | None:
    """Return the user-supplied pane name if recorded, else None."""
    # Lazy: window_state_ports wiring is bootstrapped after this module
    # is registered as a callback target; keep at call site.
    from ...window_state_ports.pane_state import get_pane_projection

    pane = get_pane_projection(window_id, pane_id)
    return pane.name if pane else None


async def _send_interactive_with_retry(
    client: TelegramClient,
    *,
    chat_id: int,
    text: str,
    keyboard: InlineKeyboardMarkup,
    thread_kwargs: dict[str, int],
    ikey: tuple[int, int],
    thread_id: int | None,
    window_id: str,
    now: float,
) -> Message | None:
    """Send interactive UI with one retry on transient transport errors."""
    for attempt in range(_INTERACTIVE_SEND_RETRIES + 1):
        try:
            return await client.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=keyboard,
                **thread_kwargs,  # type: ignore[arg-type]
            )
        except BadRequest as e:
            if is_thread_gone(e):
                logger.warning(
                    "Topic gone for interactive UI (chat=%s thread=%s window=%s), "
                    "backing off %ss — use /sync to recreate",
                    chat_id,
                    thread_id,
                    window_id,
                    int(_DEAD_TOPIC_RETRY_INTERVAL),
                )
                _send_cooldowns[ikey] = (
                    now + _DEAD_TOPIC_RETRY_INTERVAL - _SEND_RETRY_INTERVAL
                )
            else:
                logger.error("Failed to send interactive UI to %s: %s", chat_id, e)
            return None
        except (TimedOut, NetworkError) as e:
            if attempt < _INTERACTIVE_SEND_RETRIES:
                logger.debug("Interactive UI send transient error, retrying: %s", e)
                await asyncio.sleep(_INTERACTIVE_SEND_RETRY_BACKOFF_S)
                continue
            logger.error("Failed to send interactive UI to %s: %s", chat_id, e)
            return None
        except TelegramError as e:
            logger.error("Failed to send interactive UI to %s: %s", chat_id, e)
            return None
    return None


async def handle_interactive_ui(
    client: TelegramClient,
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
    pane_id: str | None = None,
) -> bool:
    """Capture terminal and send interactive UI content to user.

    Handles AskUserQuestion, ExitPlanMode, Permission Prompt, and
    RestoreCheckpoint UIs. Returns True if UI was detected and sent,
    False otherwise.

    When *pane_id* is given, captures and targets a specific pane (for
    multi-pane windows such as agent teams).  The pane context is shown
    in the message and the keyboard routes responses to that pane.
    """
    captured = await _capture_interactive_content(window_id, pane_id=pane_id)
    if not captured:
        return False

    ui_name, text = captured
    pane_name = _lookup_pane_name(window_id, pane_id) if pane_id else None
    text = format_interactive_message(text, pane_id=pane_id, pane_name=pane_name)
    ikey = (user_id, thread_id or 0)
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    keyboard = _build_interactive_keyboard(window_id, ui_name=ui_name, pane_id=pane_id)

    # Try editing existing interactive message first
    existing_msg_id = _interactive_msgs.get(ikey)
    if existing_msg_id:
        return (
            await _edit_interactive_msg(
                client, chat_id, existing_msg_id, text, keyboard, ikey, window_id
            )
            or False
        )

    # Cooldown: prevent rapid retries when sends fail
    now = time.monotonic()
    last_attempt = _send_cooldowns.get(ikey, 0.0)
    if now - last_attempt < _SEND_RETRY_INTERVAL:
        return False

    # Send new message
    thread_kwargs: dict[str, int] = {}
    if thread_id is not None:
        thread_kwargs["message_thread_id"] = thread_id

    logger.info(
        "Sending interactive UI to user %d for window_id %s", user_id, window_id
    )
    _send_cooldowns[ikey] = now
    # Send as plain text — terminal content should not be formatted.
    await rate_limit_send(chat_id)
    sent = await _send_interactive_with_retry(
        client,
        chat_id=chat_id,
        text=text,
        keyboard=keyboard,
        thread_kwargs=thread_kwargs,
        ikey=ikey,
        thread_id=thread_id,
        window_id=window_id,
        now=now,
    )
    if sent:
        _interactive_msgs[ikey] = sent.message_id
        _interactive_mode[ikey] = window_id
        _send_cooldowns.pop(ikey, None)
    return sent is not None


async def clear_interactive_msg(
    user_id: int,
    client: TelegramClient | None = None,
    thread_id: int | None = None,
) -> None:
    """Clear tracked interactive message, delete from chat, and exit interactive mode."""
    ikey = (user_id, thread_id or 0)
    msg_id = _interactive_msgs.pop(ikey, None)
    _interactive_mode.pop(ikey, None)
    _send_cooldowns.pop(ikey, None)
    logger.debug(
        "Clear interactive msg: user=%d, thread=%s, msg_id=%s",
        user_id,
        thread_id,
        msg_id,
    )
    if client and msg_id:
        chat_id = thread_router.resolve_chat_id(user_id, thread_id)
        with contextlib.suppress(TelegramError):
            await client.delete_message(chat_id=chat_id, message_id=msg_id)
