"""Centralized user-data key constants for context.user_data access.

All string keys used with PTB's context.user_data dict are defined here
to prevent typos and enable IDE navigation.
"""

PENDING_THREAD_ID = "_pending_thread_id"
PENDING_THREAD_TEXT = "_pending_thread_text"
RECOVERY_WINDOW_ID = "_recovery_window_id"
RECOVERY_SESSIONS = "_recovery_sessions"
RESUME_SESSIONS = "_resume_sessions"
VOICE_PENDING = (
    "_voice_pending"  # dict[tuple[int, int], str]: (chat_id, msg_id) → transcribed text
)

SEND_PATH_KEY = "send_path"
SEND_PAGE_KEY = "send_page"
SEND_ITEMS_KEY = "send_items"
SEND_WINDOW_ID_KEY = "send_window_id"
SEND_CWD_KEY = "send_cwd"

PANE_RENAME_WINDOW_ID = "_pane_rename_window_id"
PANE_RENAME_PANE_ID = "_pane_rename_pane_id"
PANE_RENAME_THREAD_ID = "_pane_rename_thread_id"

# Workspace picker flow (between worktree-resolve and provider-pick, herdr only)
PENDING_WORKSPACES = (
    "_pending_workspaces"  # list[tuple[str,str,str]] — cached workspace list
)
PENDING_WORKSPACE_ID = "_pending_workspace_id"  # str — chosen workspace id (or "")

# Worktree picker flow (between directory-confirm and provider-pick)
PENDING_WORKTREE_REPO = "_pending_worktree_repo"
PENDING_WORKTREE_BRANCH = "_pending_worktree_branch"
PENDING_WORKTREE_PATH = "_pending_worktree_path"
PENDING_WORKTREE_DIRTY = "_pending_worktree_dirty"
PENDING_WORKTREE_SUBDIR = "_pending_worktree_subdir"
PENDING_WORKTREE_CREATING = "_pending_worktree_creating"
AWAITING_WORKTREE_BRANCH_NAME = "_awaiting_worktree_branch_name"
