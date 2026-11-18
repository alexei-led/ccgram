"""Interactive UI callback handlers.

Handles inline keyboard callbacks for AskUserQuestion/ExitPlanMode/Permission UIs:
  - CB_ASK_* direction/action keys: navigate interactive UI via tmux keys
  - CB_ASK_REFRESH: refresh the interactive UI display

Key function: handle_interactive_callback (uniform callback handler signature).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio
import structlog

from telegram import CallbackQuery, Message, Update
from ...telegram_client import PTBTelegramClient
from ...session_protocol.models import ProtocolError
from ..callback_data import (
    CB_ASK_CHOICE,
    CB_ASK_DOWN,
    CB_ASK_ENTER,
    CB_ASK_ESC,
    CB_ASK_LEFT,
    CB_ASK_REFRESH,
    CB_ASK_RIGHT,
    CB_ASK_SPACE,
    CB_ASK_TAB,
    CB_ASK_UP,
    CB_MENU_CONTROL,
)
from ..callback_registry import register
from .pi_controls import (
    MENU_EXPIRED,
    is_menu_reply,
    parse_menu_control,
)
from ..messaging_pipeline.message_sender import ack_reaction, safe_reply
from .interactive_ui import (
    PromptStateError,
    pane_has_interactive_prompt,
    get_menu_reply_target,
    get_interactive_window,
    get_interactive_menu,
    get_interactive_pane,
    advance_interactive_sequence,
    clear_interactive_msg,
    get_interactive_prompt_fingerprint,
    handle_interactive_ui,
    is_current_interactive_message,
    is_current_interactive_prompt,
)

if TYPE_CHECKING:
    from telegram.ext import ContextTypes

    from ...session_protocol.telegram import TelegramSessionAdapter
    from ...telegram_client import TelegramClient

logger = structlog.get_logger()

# cb_prefix -> (tmux_key, refresh_ui_after)
INTERACTIVE_KEY_MAP: dict[str, tuple[str, bool]] = {
    CB_ASK_UP: ("Up", True),
    CB_ASK_DOWN: ("Down", True),
    CB_ASK_LEFT: ("Left", True),
    CB_ASK_RIGHT: ("Right", True),
    CB_ASK_ESC: ("Escape", False),
    CB_ASK_ENTER: ("Enter", True),
    CB_ASK_SPACE: ("Space", True),
    CB_ASK_TAB: ("Tab", True),
}

# Answer-toast labels for interactive key callbacks
INTERACTIVE_KEY_LABELS: dict[str, str] = {
    CB_ASK_ESC: "\u238b Esc",
    CB_ASK_ENTER: "\u23ce Enter",
    CB_ASK_SPACE: "\u2423 Space",
    CB_ASK_TAB: "\u21e5 Tab",
}

# All interactive prefixes (key map + refresh)
INTERACTIVE_PREFIXES: tuple[str, ...] = (
    *INTERACTIVE_KEY_MAP,
    CB_ASK_REFRESH,
    CB_ASK_CHOICE,
    CB_MENU_CONTROL,
)


def parse_direct_choice_callback(
    data: str,
) -> tuple[str, int, str, str | None] | None:
    """Parse a direct choice callback without splitting opaque target IDs."""
    if not data.startswith(CB_ASK_CHOICE):
        return None
    remainder = data.removeprefix(CB_ASK_CHOICE)
    key, separator, remainder = remainder.partition(":")
    sequence_text, separator2, target = remainder.partition(":")
    if (
        not separator
        or not separator2
        or key not in {"y", "n", "1", "2", "3", "4", "5", "6", "7", "8"}
        or not sequence_text.isdecimal()
        or not target
    ):
        return None
    # Lazy: callback_data keeps the pane separator colocated with other formats.
    from ..callback_data import CB_PANE_DELIMITER

    if CB_PANE_DELIMITER in target:
        window_id, pane_id = target.split(CB_PANE_DELIMITER, 1)
        return key, int(sequence_text), window_id, pane_id
    return key, int(sequence_text), target, None


def match_interactive_prefix(data: str) -> tuple[str, str, str | None] | None:
    """Match callback data against interactive UI prefixes.

    Returns (cb_prefix, window_id, pane_id_or_None) or None.

    Callback data format:
      - ``"aq:enter:1:@12"``         → window @12, active pane
      - ``"aq:enter:1:@12|%5"``      → tmux: window @12, pane %5
      - callback target suffixes remain opaque and are resolved by the backend
    """
    if data.startswith(CB_MENU_CONTROL):
        parsed = parse_menu_control(data)
        return (CB_MENU_CONTROL, parsed[2], parsed[3]) if parsed else None

    direct_choice = parse_direct_choice_callback(data)
    if direct_choice is not None:
        _, _, window_id, pane_id = direct_choice
        return CB_ASK_CHOICE, window_id, pane_id

    # Lazy: avoid a module-level import that ruff flags as unused before the
    # function body is reached; CB_PANE_DELIMITER is a plain string constant.
    from ..callback_data import CB_PANE_DELIMITER

    for prefix in (*INTERACTIVE_KEY_MAP, CB_ASK_REFRESH):
        if data.startswith(prefix):
            sequence, separator, remainder = data.removeprefix(prefix).partition(":")
            if not separator or not sequence.isdecimal():
                return None
            if CB_PANE_DELIMITER in remainder:
                window_id, pane_id = remainder.split(CB_PANE_DELIMITER, 1)
                return prefix, window_id, pane_id
            return prefix, remainder, None
    return None


async def _handle_direct_choice(
    query: CallbackQuery,
    user_id: int,
    thread_id: int | None,
    chat_id: int | None,
    message_id: int | None,
    window_id: str,
    pane_id: str | None,
    choice: tuple[str, int, str, str | None] | None,
    adapter,
    action_id: str,
    expected_prompt: str | None,
) -> None:
    """Send one validated direct choice and invalidate its sequence."""
    if choice is None:
        await query.answer("This prompt has expired", show_alert=True)
        return
    key, sequence, _window_id, _pane_id = choice
    if (
        not is_current_interactive_prompt(
            user_id, thread_id, window_id, chat_id, message_id, sequence
        )
        or get_interactive_pane(user_id, thread_id, chat_id) != pane_id
    ):
        await query.answer("This prompt has expired", show_alert=True)
        return
    if expected_prompt is None:
        await query.answer(
            "Unverified terminal UI. Use /toolbar or the terminal.", show_alert=True
        )
        return
    enter = key in {"y", "n"}
    try:
        sent, message = await adapter.send_text(
            window_id,
            key,
            pane_id=pane_id,
            submit=enter,
            terminal_input=True,
            action_id=action_id,
            expected_prompt=expected_prompt,
        )
    except ProtocolError:
        raise
    except Exception:  # noqa: BLE001 — report any backend delivery failure to user
        logger.warning("Failed to send direct interactive choice", exc_info=True)
        sent = False
    if not sent:
        await query.answer("Unable to send choice. Try again.", show_alert=True)
        return
    advance_interactive_sequence(user_id, thread_id, chat_id=chat_id)
    await query.answer()


async def _handle_navigation_key(
    query: CallbackQuery,
    user_id: int,
    thread_id: int | None,
    chat_id: int | None,
    message_id: int | None,
    window_id: str,
    pane_id: str | None,
    cb_prefix: str,
    sequence: int,
    adapter: TelegramSessionAdapter,
    client: TelegramClient,
    action_id: str,
    expected_prompt: str | None,
) -> None:
    """Keep the existing provider navigation path separate from Pi controls."""
    if (
        not is_current_interactive_prompt(
            user_id, thread_id, window_id, chat_id, message_id, sequence
        )
        or get_interactive_pane(user_id, thread_id, chat_id) != pane_id
    ):
        await query.answer("This prompt has expired", show_alert=True)
        return
    if expected_prompt is None:
        await query.answer(
            "Unverified terminal UI. Use /toolbar or the terminal.", show_alert=True
        )
        return
    tmux_key, refresh_ui = INTERACTIVE_KEY_MAP[cb_prefix]
    sent, message_text = await adapter.send_keys(
        window_id,
        [tmux_key],
        pane_id=pane_id,
        action_id=action_id,
        expected_prompt=expected_prompt,
    )
    if not sent:
        await query.answer(
            message_text or "Unable to send key. Try again.", show_alert=True
        )
        return
    advance_interactive_sequence(user_id, thread_id, chat_id=chat_id)
    if refresh_ui:
        await asyncio.sleep(0.5)
        await handle_interactive_ui(
            client,
            user_id,
            window_id,
            thread_id,
            pane_id=pane_id,
            chat_id=chat_id,
        )
    else:
        await clear_interactive_msg(user_id, client, thread_id, chat_id=chat_id)
    await query.answer(INTERACTIVE_KEY_LABELS.get(cb_prefix, message_text))


async def handle_interactive_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle interactive UI callbacks (AskUserQuestion/ExitPlanMode navigation)."""
    matched = match_interactive_prefix(data)
    if not matched:
        await query.answer("This prompt has expired", show_alert=True)
        return

    cb_prefix, window_id, pane_id = matched
    # Lazy: callback_helpers ↔ callback_registry ↔ interactive_callbacks
    # cycle through registration side effects.
    from ..callback_helpers import get_thread_id, user_owns_window

    thread_id = get_thread_id(update)
    message = query.message
    chat_id = message.chat.id if message is not None else None
    private_topic = (
        message is not None
        and message.chat.type == "private"
        and getattr(message, "is_topic_message", False) is True
        and isinstance(getattr(message, "message_thread_id", None), int)
    )
    ownership_chat_id = (
        chat_id
        if private_topic or message is None or message.chat.type != "private"
        else None
    )
    if not user_owns_window(user_id, window_id, ownership_chat_id):
        await query.answer("Not your session", show_alert=True)
        return

    # One service instance is shared by all Telegram handlers and protocol clients.
    # Lazy: this callback module loads during registry setup; defer the protocol
    # backend and its provider imports until an interactive callback is accepted.
    from ...session_protocol.telegram import adapter_for

    adapter = adapter_for(context.bot_data)
    action_id = f"telegram-callback:{query.id}"

    message_id = message.message_id if message is not None else None
    expected_prompt = get_interactive_prompt_fingerprint(
        user_id, thread_id, chat_id, message_id
    )

    try:
        if cb_prefix == CB_MENU_CONTROL:
            await handle_menu_control(
                query, user_id, thread_id, PTBTelegramClient(context.bot), data, adapter
            )
            return
        if cb_prefix == CB_ASK_CHOICE:
            await _handle_direct_choice(
                query,
                user_id,
                thread_id,
                chat_id,
                message.message_id if message is not None else None,
                window_id,
                pane_id,
                parse_direct_choice_callback(data),
                adapter,
                action_id,
                expected_prompt,
            )
            return

        client = PTBTelegramClient(context.bot)
        if cb_prefix == CB_ASK_REFRESH:
            if not is_current_interactive_message(
                user_id, thread_id, chat_id, message_id
            ):
                await query.answer("This prompt has expired", show_alert=True)
                return
            await handle_interactive_ui(
                client,
                user_id,
                window_id,
                thread_id,
                pane_id=pane_id,
                chat_id=chat_id,
            )
            await query.answer("\U0001f504")
        else:
            await _handle_navigation_key(
                query,
                user_id,
                thread_id,
                chat_id,
                message_id,
                window_id,
                pane_id,
                cb_prefix,
                int(data.removeprefix(cb_prefix).partition(":")[0]),
                adapter,
                client,
                action_id,
                expected_prompt,
            )
    except ProtocolError as exc:
        if exc.code != "session_busy":
            raise
        await query.answer("Session is busy. Try again.", show_alert=True)


async def refresh_menu(
    client: TelegramClient,
    user_id: int,
    window_id: str,
    thread_id: int | None,
    chat_id: int,
    pane_id: str | None,
) -> None:
    if await handle_interactive_ui(
        client, user_id, window_id, thread_id, pane_id=pane_id, chat_id=chat_id
    ):
        return
    try:
        still_open = await pane_has_interactive_prompt(window_id, pane_id)
    except PromptStateError:
        return
    if not still_open:
        await clear_interactive_msg(user_id, client, thread_id, chat_id=chat_id)


async def handle_menu_control(
    query: CallbackQuery,
    user_id: int,
    thread_id: int | None,
    client: TelegramClient,
    data: str,
    adapter: TelegramSessionAdapter,
) -> None:
    parsed = parse_menu_control(data)
    message = query.message
    if parsed is None or message is None:
        await query.answer("This prompt has expired", show_alert=True)
        return
    action, sequence, window_id, pane_id = parsed
    chat_id = message.chat.id
    fingerprint = get_interactive_prompt_fingerprint(
        user_id, thread_id, chat_id, message.message_id
    )
    current = is_current_interactive_prompt(
        user_id, thread_id, window_id, chat_id, message.message_id, sequence
    )
    if action == "refresh":
        # Recapture needs message ownership, not input metadata that a failed
        # display update may already have retired.
        if (
            not is_current_interactive_message(
                user_id, thread_id, chat_id, message.message_id
            )
            or get_interactive_window(user_id, thread_id, chat_id=chat_id) != window_id
            or get_interactive_pane(user_id, thread_id, chat_id) != pane_id
        ):
            await query.answer("This prompt has expired", show_alert=True)
            return
        await query.answer()
        await refresh_menu(client, user_id, window_id, thread_id, chat_id, pane_id)
        return
    menu = get_interactive_menu(
        user_id, thread_id, chat_id, message.message_id, window_id
    )
    owner = get_interactive_pane(user_id, thread_id, chat_id)
    if fingerprint is None or not current or menu is None or owner != pane_id:
        await query.answer("This prompt has expired", show_alert=True)
        return
    index = int(action)
    if index >= len(menu.controls):
        await query.answer("This prompt has expired", show_alert=True)
        return
    keys = menu.controls[index].keys
    if len(keys) == 1 and len(keys[0]) == 1 and keys[0].isprintable():
        sent, detail = await adapter.send_text(
            window_id,
            keys[0],
            submit=False,
            pane_id=pane_id,
            action_id=f"telegram-callback:{query.id}",
            expected_prompt=fingerprint,
        )
    else:
        sent, detail = await adapter.send_keys(
            window_id,
            list(keys),
            pane_id=pane_id,
            action_id=f"telegram-callback:{query.id}",
            expected_prompt=fingerprint,
        )
    if not sent:
        await query.answer(detail or "Unable to send key", show_alert=True)
        return
    advance_interactive_sequence(user_id, thread_id, chat_id=chat_id)
    await query.answer()
    await asyncio.sleep(0.5)
    await refresh_menu(client, user_id, window_id, thread_id, chat_id, pane_id)


async def handle_menu_reply(
    client: TelegramClient,
    user_id: int,
    window_id: str,
    thread_id: int,
    text: str,
    message: Message,
    adapter: TelegramSessionAdapter,
) -> bool:
    """Only an explicit reply to a live, text-capable menu is literal input."""
    reply = message.reply_to_message
    if reply is None or reply.text is None or not is_menu_reply(message):
        return False
    target = get_menu_reply_target(
        user_id, thread_id, message.chat.id, reply.message_id, reply.text, window_id
    )
    if target is None:
        await safe_reply(message, MENU_EXPIRED)
        return True
    if not text.isprintable():
        await safe_reply(
            message,
            "Use a single line without control characters; tap Enter separately.",
        )
        return True
    pane_id, fingerprint, menu = target
    if not menu.text_input:
        await safe_reply(message, "This menu does not accept text. Use its buttons.")
        return True
    sent, detail = await adapter.send_text(
        window_id,
        text,
        submit=False,
        pane_id=pane_id,
        action_id=f"telegram-message:{message.chat.id}:{message.message_id}:menu-input",
        expected_prompt=fingerprint,
    )
    if not sent:
        await safe_reply(message, detail or "Unable to type into the menu.")
        return True
    await ack_reaction(client, message.chat.id, message.message_id)
    await asyncio.sleep(0.5)
    await refresh_menu(client, user_id, window_id, thread_id, message.chat.id, pane_id)
    return True


# --- Registry dispatch entry point ---


@register(*INTERACTIVE_PREFIXES)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    # Lazy: callback_helpers and callback_tokens participate in registry cycles.
    from ..callback_helpers import user_owns_window

    # Lazy: resolve tokens only when an interactive callback is dispatched.
    from ..callback_tokens import resolve_callback_data

    message = query.message
    chat_id = message.chat.id if message is not None else None
    private_topic = (
        message is not None
        and message.chat.type == "private"
        and getattr(message, "is_topic_message", False) is True
        and isinstance(getattr(message, "message_thread_id", None), int)
    )
    ownership_chat_id = (
        chat_id
        if private_topic or message is None or message.chat.type != "private"
        else None
    )
    data = resolve_callback_data(
        query.data,
        user.id,
        lambda uid, window_id: user_owns_window(uid, window_id, ownership_chat_id),
    )
    if data is None:
        await query.answer("This button has expired", show_alert=True)
        return
    await handle_interactive_callback(query, user.id, data, update, context)
