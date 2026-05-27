"""Topics subpackage — topic creation, lifecycle, directory browser, window picker.

Bundles the modules that drive Telegram-topic ↔ terminal-session binding:
``topic_orchestration`` (new-window/new-topic flow, retries),
``topic_lifecycle`` (autoclose timers, unbound TTL, topic close/edit
handlers), ``directory_browser`` (directory + window picker UI),
``directory_callbacks`` (browser callback dispatcher), and
``window_callbacks`` (window picker callback dispatcher).

Public surface re-exported here is the entry point for ``bot.py`` and
the rest of ``handlers/``; internals stay in the per-module files.
"""

from .cmux_callbacks import build_cmux_picker, cmux_command
from .directory_browser import (
    BROWSE_DIRS_KEY,
    BROWSE_PAGE_KEY,
    BROWSE_PATH_KEY,
    DIRS_PER_PAGE,
    STATE_BROWSING_DIRECTORY,
    STATE_KEY,
    STATE_SELECTING_WINDOW,
    UNBOUND_WINDOWS_KEY,
    build_directory_browser,
    build_mode_picker,
    build_provider_picker,
    build_window_picker,
    build_worktree_confirm,
    build_worktree_picker,
    clear_browse_state,
    clear_window_picker_state,
    clear_worktree_state,
    get_favorites,
)
from .directory_callbacks import handle_directory_callback
from .new_command import new_command
from .topic_lifecycle import (
    check_autoclose_timers,
    check_unbound_window_ttl,
    probe_topic_existence,
    prune_stale_state,
    topic_closed_handler,
    topic_edited_handler,
)
from .topic_orchestration import (
    adopt_unbound_windows,
    clear_topic_create_retry,
    collect_target_chats,
    create_topic_in_chat,
    handle_new_window,
)
from .window_callbacks import handle_window_callback

__all__ = [
    "BROWSE_DIRS_KEY",
    "build_cmux_picker",
    "cmux_command",
    "BROWSE_PAGE_KEY",
    "BROWSE_PATH_KEY",
    "DIRS_PER_PAGE",
    "STATE_BROWSING_DIRECTORY",
    "STATE_KEY",
    "STATE_SELECTING_WINDOW",
    "UNBOUND_WINDOWS_KEY",
    "adopt_unbound_windows",
    "build_directory_browser",
    "build_mode_picker",
    "build_provider_picker",
    "build_window_picker",
    "build_worktree_confirm",
    "build_worktree_picker",
    "check_autoclose_timers",
    "check_unbound_window_ttl",
    "clear_browse_state",
    "clear_worktree_state",
    "clear_topic_create_retry",
    "clear_window_picker_state",
    "collect_target_chats",
    "create_topic_in_chat",
    "get_favorites",
    "handle_directory_callback",
    "handle_new_window",
    "handle_window_callback",
    "new_command",
    "probe_topic_existence",
    "prune_stale_state",
    "topic_closed_handler",
    "topic_edited_handler",
]
