"""Feature-port package for window state — narrow read/write seams.

Each module exposes frozen projection dataclasses for reads and thin
write functions that delegate to ``WindowStateStore`` (the persistence
kernel). Ports do not own state; they are the only approved feature
seam between handler code and the raw ``WindowState`` shape.

Provider/session identity writes still go through ``SessionManager`` so
provider-capability resolution stays in one place.
"""

from __future__ import annotations

from .identity_state import (
    IdentityProjection,
    clear_transcript_path,
    get_approval_mode,
    get_cwd,
    get_identity,
    get_provider_name,
    get_session_id,
    get_transcript_path,
    get_window_name,
    is_provider_manually_overridden,
    set_provider_manual_override,
    set_window_approval_mode,
)
from .lifecycle_state import (
    LifecycleProjection,
    get_lifecycle,
    get_origin,
    set_window_origin,
)
from .pane_state import (
    PaneProjection,
    WindowPaneSnapshot,
    get_pane_lifecycle_notify,
    get_pane_projection,
    list_pane_projections,
    remove_pane,
    set_pane_lifecycle_notify,
    snapshot_window_panes,
    upsert_pane,
)
from .tool_state import (
    ToolModeProjection,
    cycle_batch_mode,
    cycle_tool_call_visibility,
    get_batch_mode,
    get_tool_call_visibility,
    get_tool_modes,
    is_ephemeral_tools,
    is_tool_calls_hidden,
    set_batch_mode,
    set_tool_call_visibility,
)
from .worktree_state import (
    WorktreeProjection,
    clear_worktree,
    get_worktree,
    set_worktree,
)

__all__ = [
    # Identity
    "IdentityProjection",
    "clear_transcript_path",
    "get_identity",
    "get_provider_name",
    "get_session_id",
    "get_cwd",
    "get_transcript_path",
    "get_window_name",
    "get_approval_mode",
    "is_provider_manually_overridden",
    "set_provider_manual_override",
    "set_window_approval_mode",
    # Lifecycle
    "LifecycleProjection",
    "get_lifecycle",
    "get_origin",
    "set_window_origin",
    # Pane
    "PaneProjection",
    "WindowPaneSnapshot",
    "get_pane_lifecycle_notify",
    "get_pane_projection",
    "list_pane_projections",
    "remove_pane",
    "set_pane_lifecycle_notify",
    "snapshot_window_panes",
    "upsert_pane",
    # Tool mode
    "ToolModeProjection",
    "cycle_batch_mode",
    "cycle_tool_call_visibility",
    "get_batch_mode",
    "get_tool_call_visibility",
    "get_tool_modes",
    "is_ephemeral_tools",
    "is_tool_calls_hidden",
    "set_batch_mode",
    "set_tool_call_visibility",
    # Worktree
    "WorktreeProjection",
    "clear_worktree",
    "get_worktree",
    "set_worktree",
]
