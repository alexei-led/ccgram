"""Status subpackage — status bubble, status-bar callbacks, topic emoji.

Bundles the modules that own the per-topic status surface:
``status_bubble`` (status message lifecycle, keyboard layout, task-list
formatting, status-to-content conversion), ``status_bar_actions``
(inline-button callbacks for the status bubble — notify toggle, recall,
remote control, esc, quick keys), and ``topic_emoji`` (forum topic name
emoji updates with debounced state transitions).

Public surface re-exported here is the entry point for ``bot.py`` and
the rest of ``handlers/``; internals stay in the per-module files.
"""

from .status_bar_actions import build_dashboard_button
from .status_bubble import (
    build_status_keyboard,
    clear_status_message,
    clear_status_msg_info,
    convert_status_to_content,
    process_status_clear,
    process_status_update,
    send_status_text,
)
from .topic_emoji import (
    EMOJI_ACTIVE,
    EMOJI_DEAD,
    EMOJI_DONE,
    EMOJI_IDLE,
    EMOJI_RC,
    EMOJI_YOLO,
    clear_disabled_chat,
    clear_topic_emoji_state,
    format_topic_name_for_mode,
    reset_all_state,
    strip_emoji_prefix,
    sync_topic_name,
    update_stored_topic_name,
    update_topic_emoji,
)

__all__ = [
    "EMOJI_ACTIVE",
    "EMOJI_DEAD",
    "EMOJI_DONE",
    "EMOJI_IDLE",
    "EMOJI_RC",
    "EMOJI_YOLO",
    "build_dashboard_button",
    "build_status_keyboard",
    "clear_disabled_chat",
    "clear_status_message",
    "clear_status_msg_info",
    "clear_topic_emoji_state",
    "convert_status_to_content",
    "format_topic_name_for_mode",
    "process_status_clear",
    "process_status_update",
    "reset_all_state",
    "send_status_text",
    "strip_emoji_prefix",
    "sync_topic_name",
    "update_stored_topic_name",
    "update_topic_emoji",
]
