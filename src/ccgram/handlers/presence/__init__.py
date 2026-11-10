"""Presence subpackage: desk mode (quiet Telegram while at the computer).

``presence`` decides desk vs away per user, ``digest`` keeps the quiet
per-topic summary, ``desk_mode`` wires both into Telegram and polling.

Only the state module is re-exported: the messaging pipeline imports it, and
re-exporting ``digest`` here would import the pipeline back while it loads.
Import ``digest`` and ``desk_mode`` by module path.
"""

from .presence import (
    is_quiet,
    note_desk_prompt,
    note_telegram_activity,
    remember_telegram_text,
)

__all__ = [
    "is_quiet",
    "note_desk_prompt",
    "note_telegram_activity",
    "remember_telegram_text",
]
