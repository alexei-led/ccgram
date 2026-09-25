"""Interactive subpackage — interactive UI rendering and callbacks.

Bundles the modules that own interactive Telegram UI driven by Claude
Code prompts: ``interactive_ui`` (AskUserQuestion / ExitPlanMode /
Permission Prompt rendering, terminal capture, mode tracking, message
lifecycle) and ``interactive_callbacks`` (inline-keyboard direction/
action callback dispatch).

Public surface re-exported here is the entry point for ``bot.py`` and
the rest of ``handlers/``; internals stay in the per-module files.
"""

from .interactive_callbacks import (
    INTERACTIVE_KEY_LABELS,
    INTERACTIVE_KEY_MAP,
    INTERACTIVE_PREFIXES,
    handle_interactive_callback,
    match_interactive_prefix,
)
from .interactive_ui import (
    INTERACTIVE_INSTRUCTION_LINE,
    INTERACTIVE_TOOL_NAMES,
    clear_interactive_mode,
    clear_interactive_msg,
    clear_send_cooldowns,
    format_interactive_message,
    get_interactive_msg_id,
    get_interactive_pane as get_interactive_pane,
    get_interactive_window,
    handle_interactive_ui,
    pane_has_interactive_prompt as pane_has_interactive_prompt,
    set_interactive_mode,
)

__all__ = [
    "INTERACTIVE_INSTRUCTION_LINE",
    "INTERACTIVE_KEY_LABELS",
    "INTERACTIVE_KEY_MAP",
    "INTERACTIVE_PREFIXES",
    "INTERACTIVE_TOOL_NAMES",
    "clear_interactive_mode",
    "clear_interactive_msg",
    "clear_send_cooldowns",
    "format_interactive_message",
    "get_interactive_msg_id",
    "get_interactive_window",
    "handle_interactive_callback",
    "handle_interactive_ui",
    "match_interactive_prefix",
    "set_interactive_mode",
]
