"""Telegram controls for the existing Pi TUI; no second agent or settings cache."""

import asyncio
from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ...providers.pi_tui import PI_MENU_INSTRUCTION
from ..callback_data import CB_PANE_DELIMITER, CB_PI_CONTROL
from ..callback_tokens import compact_callback_data
from ..messaging_pipeline.message_sender import ack_reaction, safe_reply

if TYPE_CHECKING:
    from telegram import CallbackQuery, Message

    from ...session_protocol.telegram import TelegramSessionAdapter
    from ...telegram_client import TelegramClient

# Jump with repeated arrows: Pi's SelectList/SettingsList do not handle PageDown.
_PI_KEYS: dict[str, tuple[str, ...]] = {
    "up": ("Up",),
    "down": ("Down",),
    "up5": ("Up",) * 5,
    "down5": ("Down",) * 5,
    "left": ("Left",),
    "right": ("Right",),
    "enter": ("Enter",),
    "esc": ("Escape",),
    "space": ("Space",),
    "tab": ("Tab",),
    "clear": ("C-u",),
    "backspace": ("BSpace",),
    "save": ("C-s",),
    "all": ("C-a",),
    "none": ("C-x",),
    "provider": ("C-p",),
    "reorder-up": ("M-Up",),
    "reorder-down": ("M-Down",),
    "newline": ("C-j",),
}


def build_pi_keyboard(
    window_id: str, ui_name: str, pane_id: str | None, sequence: int
) -> InlineKeyboardMarkup:
    target = f"{window_id}{CB_PANE_DELIMITER}{pane_id}" if pane_id else window_id
    rows = [
        [("↑5", "up5"), ("↑", "up"), ("↓", "down"), ("↓5", "down5")],
        [("←", "left"), ("␣ Space", "space"), ("→", "right"), ("⇥ Tab", "tab")],
        [("Clear search", "clear"), ("⌫", "backspace")],
    ]
    if ui_name in {"PiModel", "PiThinking", "PiScopedModels"}:
        rows.append(
            [("Save default" if ui_name != "PiScopedModels" else "Save scope", "save")]
        )
    if ui_name == "PiEditor":
        rows.append([("New line", "newline")])
    if ui_name == "PiScopedModels":
        rows.extend(
            [
                [
                    ("All matching", "all"),
                    ("None matching", "none"),
                    ("Provider", "provider"),
                ],
                [("Move ↑", "reorder-up"), ("Move ↓", "reorder-down")],
            ]
        )
    rows.append([("⎋ Esc", "esc"), ("🔄", "refresh"), ("⏎ Enter", "enter")])
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    label,
                    callback_data=compact_callback_data(
                        CB_PI_CONTROL,
                        f"{CB_PI_CONTROL}{action}:{sequence}:{target}",
                        window_id,
                    ),
                )
                for label, action in row
            ]
            for row in rows
        ]
    )


def parse_pi_control(data: str) -> tuple[str, int, str, str | None] | None:
    if not data.startswith(CB_PI_CONTROL):
        return None
    action, separator, remainder = data.removeprefix(CB_PI_CONTROL).partition(":")
    sequence, separator2, target = remainder.partition(":")
    if (
        not separator
        or not separator2
        or not sequence.isdecimal()
        or action not in {*_PI_KEYS, "refresh"}
        or not target
    ):
        return None
    window_id, delimiter, pane_id = target.partition(CB_PANE_DELIMITER)
    if not window_id or (delimiter and not pane_id):
        return None
    return action, int(sequence), window_id, pane_id if delimiter else None


async def refresh_pi_menu(
    client: TelegramClient,
    user_id: int,
    window_id: str,
    thread_id: int | None,
    chat_id: int,
    pane_id: str | None,
) -> None:
    # Lazy: interactive_ui builds this module's keyboard; callbacks import back
    # only after registration, avoiding an interactive_ui ↔ controls cycle.
    from .interactive_ui import (
        PromptStateError,
        clear_interactive_msg,
        handle_interactive_ui,
        pane_has_interactive_prompt,
    )

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


async def handle_pi_control(
    query: CallbackQuery,
    user_id: int,
    thread_id: int | None,
    client: TelegramClient,
    data: str,
    adapter: TelegramSessionAdapter,
) -> None:
    # Lazy: see refresh_pi_menu — rendering depends on this module's keyboard.
    from .interactive_ui import (
        advance_interactive_sequence,
        get_interactive_prompt_fingerprint,
        is_current_interactive_prompt,
        is_current_interactive_message,
        get_interactive_window,
    )

    parsed = parse_pi_control(data)
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
        # Input consumes its sequence even if the subsequent Telegram edit fails.
        # Recapture is safe from that older keyboard; sending another key is not.
        current = (
            is_current_interactive_message(
                user_id, thread_id, chat_id, message.message_id
            )
            and get_interactive_window(user_id, thread_id, chat_id=chat_id) == window_id
        )
    if fingerprint is None or not current:
        await query.answer("This prompt has expired", show_alert=True)
        return
    if action != "refresh":
        sent, detail = await adapter.send_keys(
            window_id,
            list(_PI_KEYS[action]),
            pane_id=pane_id,
            action_id=f"telegram-callback:{query.id}",
            expected_prompt=fingerprint,
        )
        if not sent:
            await query.answer(detail or "Unable to send key", show_alert=True)
            return
        advance_interactive_sequence(user_id, thread_id, chat_id=chat_id)
    await query.answer()
    if action != "refresh":
        await asyncio.sleep(0.5)
    await refresh_pi_menu(client, user_id, window_id, thread_id, chat_id, pane_id)


async def handle_pi_reply(
    client: TelegramClient,
    user_id: int,
    window_id: str,
    thread_id: int,
    text: str,
    message: Message,
    adapter: TelegramSessionAdapter,
) -> bool:
    """Only an explicit reply to the live Pi menu is literal, unsubmitted input."""
    reply = message.reply_to_message
    if (
        reply is None
        or not isinstance(reply.text, str)
        or not reply.text.startswith(PI_MENU_INSTRUCTION)
    ):
        return False
    # Lazy: the renderer imports the keyboard here; defer the reverse dependency.
    from .interactive_ui import get_pi_reply_target

    target = get_pi_reply_target(
        user_id, thread_id, message.chat.id, reply.message_id, reply.text, window_id
    )
    if target is None:
        await safe_reply(
            message, "This Pi menu has expired. Reply to the current menu."
        )
        return True
    if not text.isprintable():
        await safe_reply(
            message,
            "Use a single line without control characters; tap Enter separately.",
        )
        return True
    pane_id, fingerprint = target
    sent, detail = await adapter.send_text(
        window_id,
        text,
        submit=False,
        pane_id=pane_id,
        action_id=f"telegram-message:{message.chat.id}:{message.message_id}:pi-input",
        expected_prompt=fingerprint,
    )
    if not sent:
        await safe_reply(message, detail or "Unable to type into the Pi menu.")
        return True
    await ack_reaction(client, message.chat.id, message.message_id)
    await asyncio.sleep(0.5)
    await refresh_pi_menu(
        client, user_id, window_id, thread_id, message.chat.id, pane_id
    )
    return True
