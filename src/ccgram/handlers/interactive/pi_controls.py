"""Telegram controls for the existing Pi TUI; no second agent or settings cache."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ..callback_data import CB_PANE_DELIMITER, CB_PI_CONTROL

if TYPE_CHECKING:
    from telegram import Message

PI_MENU_INSTRUCTION = (
    "↑↓ navigate · Enter select/change · Esc back/cancel\n"
    "Reply to this message to type/filter without Enter. Never send credentials."
)
PI_MENU_EXPIRED = "This Pi menu has expired. Reply to the current menu."


def is_pi_menu_reply(message: Message) -> bool:
    reply = message.reply_to_message
    return bool(
        reply
        and isinstance(reply.text, str)
        and reply.text.startswith(PI_MENU_INSTRUCTION)
    )


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


def pi_control_keys(action: str) -> tuple[str, ...]:
    """Return the fixed terminal keys for a validated control action."""
    return _PI_KEYS[action]


def build_pi_keyboard(
    window_id: str,
    ui_name: str,
    pane_id: str | None,
    sequence: int,
    encode_callback: Callable[[str, str, str], str],
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
                    callback_data=encode_callback(
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
