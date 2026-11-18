"""Pure remote-menu presentation; native providers own supported controls."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ..callback_data import CB_PANE_DELIMITER, CB_MENU_CONTROL

if TYPE_CHECKING:
    from telegram import Message

    from ...providers.base import TerminalMenu

MENU_MARKER = "↑↓ navigate · Enter select/change · Esc back/cancel"
MENU_INSTRUCTION = (
    MENU_MARKER + "\n"
    "Reply to this message to type/filter without Enter. Never send credentials."
)
MENU_BUTTON_INSTRUCTION = (
    MENU_MARKER + "\nUse the buttons below; this menu does not accept text."
)
MENU_EXPIRED = "This menu has expired. Reply to the current menu."
_BUTTONS_PER_ROW = 2


def is_menu_reply(message: Message) -> bool:
    reply = message.reply_to_message
    return bool(
        reply and isinstance(reply.text, str) and reply.text.startswith(MENU_MARKER)
    )


def build_menu_keyboard(
    window_id: str,
    menu: TerminalMenu,
    pane_id: str | None,
    sequence: int,
    encode_callback: Callable[[str, str, str], str],
) -> InlineKeyboardMarkup:
    target = f"{window_id}{CB_PANE_DELIMITER}{pane_id}" if pane_id else window_id

    def button(label: str, action: str) -> InlineKeyboardButton:
        return InlineKeyboardButton(
            label,
            callback_data=encode_callback(
                CB_MENU_CONTROL,
                f"{CB_MENU_CONTROL}{action}:{sequence}:{target}",
                window_id,
            ),
        )

    controls = [
        button(control.label, str(index)) for index, control in enumerate(menu.controls)
    ]
    rows = [
        controls[start : start + _BUTTONS_PER_ROW]
        for start in range(0, len(controls), _BUTTONS_PER_ROW)
    ]
    rows.append([button("🔄", "refresh")])
    return InlineKeyboardMarkup(rows)


def parse_menu_control(data: str) -> tuple[str, int, str, str | None] | None:
    if not data.startswith(CB_MENU_CONTROL):
        return None
    action, separator, remainder = data.removeprefix(CB_MENU_CONTROL).partition(":")
    sequence, separator2, target = remainder.partition(":")
    if (
        not separator
        or not separator2
        or not sequence.isdecimal()
        or (action != "refresh" and not action.isdecimal())
        or not target
    ):
        return None
    window_id, delimiter, pane_id = target.partition(CB_PANE_DELIMITER)
    if not window_id or (delimiter and not pane_id):
        return None
    return action, int(sequence), window_id, pane_id if delimiter else None
