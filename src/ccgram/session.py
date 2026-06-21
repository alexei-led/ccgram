"""Claude Code session management — the core state hub.

Manages the key mappings:
  Window→Session (window_states): which Claude session_id a window holds (keyed by window_id).
  User→Thread→Window: delegated to ThreadRouter (see thread_router.py).

Responsibilities:
  - Persist/load state to ~/.ccgram/state.json.
  - Sync window↔session bindings from session_map.json (written by hook).
  - Resolve window IDs to ClaudeSession objects (JSONL file reading).
  - Delegate thread↔window routing to ThreadRouter.
  - Send keystrokes to tmux windows and retrieve message history.
  - Re-resolve stale window IDs on startup (tmux server restart recovery).

Key class: SessionManager (singleton instantiated as `session_manager`).
Thread routing: delegated to ThreadRouter (see thread_router.py) — no pass-throughs.
"""

import json
import structlog
from dataclasses import dataclass, field
from typing import Any

from .config import config
from .session_map import (
    SessionMapSync,
    install_session_map_sync,
    is_backend_window_id,
    live_window_session_ids,
    read_session_map_raw,
    session_map_prefix,
    session_map_sync,
)
from .state_persistence import StatePersistence
from .multiplexer import multiplexer as tmux_manager
from .thread_router import ThreadRouter, install_thread_router, thread_router
from .user_preferences import (
    UserPreferences,
    install_user_preferences,
    user_preferences,
)
from .window_view import WindowView
from .window_state_ports import identity_state as _identity_state
from .window_state_ports import lifecycle_state as _lifecycle_state
from .window_state_ports import tool_state as _tool_state
from .window_state_ports import worktree_state as _worktree_state
from .window_state_store import (
    WindowState,
    WindowStateStore,
    install_window_store,
    window_store,
)

logger = structlog.get_logger()


@dataclass
class AuditIssue:
    """A single issue found during state audit."""

    category: str  # ghost_binding | orphaned_display_name | orphaned_group_chat_id | stale_window_state | stale_offset | display_name_drift
    detail: str
    fixable: bool


@dataclass
class AuditResult:
    """Result of a state audit."""

    issues: list[AuditIssue]
    total_bindings: int
    live_binding_count: int

    @property
    def fixable_count(self) -> int:
        return sum(1 for i in self.issues if i.fixable)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0


@dataclass
class SessionManager:
    """Manages session state for Claude Code.

    All internal keys use window_id (e.g. '@0', '@12') for uniqueness.
    Display names (window_name) are stored separately for UI presentation.

    Thread routing (thread_bindings, display names, group_chat_ids) is
    delegated to ThreadRouter — see thread_router.py.

    window_states: window_id -> WindowState (session_id, cwd, window_name)

    User preferences (starred dirs, MRU, read offsets) are delegated to
    UserPreferences — see user_preferences.py.
    """

    # Delegated persistence (not serialized)
    _persistence: StatePersistence = field(default=None, repr=False, init=False)  # type: ignore[assignment]

    @property
    def window_states(self) -> dict[str, WindowState]:
        return window_store.window_states

    # Backward-compat properties for routing data (owned by thread_router)
    @property
    def thread_bindings(self) -> dict[int, dict[int, str]]:
        return thread_router.thread_bindings

    @property
    def group_chat_ids(self) -> dict[str, int]:
        return thread_router.group_chat_ids

    @property
    def window_display_names(self) -> dict[str, str]:
        return thread_router.window_display_names

    def __post_init__(self) -> None:
        self._persistence = StatePersistence(config.state_file, self._serialize_state)
        self._window_store = WindowStateStore(
            schedule_save=self._save_state,
            on_hookless_provider_switch=self._clear_session_map_entry,
        )
        install_window_store(self._window_store)
        self._thread_router = ThreadRouter(
            schedule_save=self._save_state,
            has_window_state=self._window_store.has_window,
        )
        install_thread_router(self._thread_router)
        self._user_preferences = UserPreferences(schedule_save=self._save_state)
        install_user_preferences(self._user_preferences)
        self._session_map_sync = SessionMapSync(schedule_save=self._save_state)
        install_session_map_sync(self._session_map_sync)
        self._load_state()

    def _serialize_state(self) -> dict[str, Any]:
        """Serialize all state to a dict for persistence."""
        result = {"window_states": window_store.to_dict()}
        result.update(user_preferences.to_dict())
        result.update(thread_router.to_dict())
        return result

    def _save_state(self) -> None:
        """Schedule debounced save (0.5s delay, resets on each call)."""
        self._persistence.schedule_save()

    def flush_state(self) -> None:
        """Force immediate save. Call on shutdown."""
        self._persistence.flush()

    def _is_window_id(self, key: str) -> bool:
        """Check if a key looks like a window ID for the active backend.

        Backend-aware: tmux ``@N`` ids, herdr ``wN:pM`` ids (see
        ``session_map.is_backend_window_id``). Old-format (window-name) keys
        return False on tmux so startup re-resolution migrates them.
        """
        return is_backend_window_id(key)

    def _load_state(self) -> None:
        """Load state during initialization.

        Detects old-format state (window_name keys without '@' prefix) and
        marks for migration on next startup re-resolution.
        """
        state = self._persistence.load()
        if not state:
            return

        window_store.from_dict(state.get("window_states", {}))

        # Load user preferences (starred dirs, MRU, read offsets)
        user_preferences.from_dict(state)

        # Load routing data into ThreadRouter (handles dedup + reverse index)
        thread_router.from_dict(state)

        # Detect old format: keys that don't look like window IDs
        needs_migration = False
        for k in window_store.window_states:
            if not self._is_window_id(k):
                needs_migration = True
                break
        if not needs_migration:
            for bindings in thread_router.thread_bindings.values():
                for wid in bindings.values():
                    if not self._is_window_id(wid):
                        needs_migration = True
                        break
                if needs_migration:
                    break

        if needs_migration:
            logger.info(
                "Detected old-format state (window_name keys), "
                "will re-resolve on startup"
            )

    async def resolve_stale_ids(self) -> None:
        """Re-resolve persisted window IDs against live tmux windows.

        Called on startup. Delegates to window_resolver for the heavy lifting.
        Dead window bindings and states are preserved for /restore recovery.
        """
        # Lazy: window_resolver imports session-state types; hoisting forms
        # session → window_resolver → session.WindowState cycle.
        # Lazy: window_resolver pulls back into session manager
        from .window_resolver import LiveWindow, resolve_stale_ids as _resolve

        windows = await tmux_manager.list_windows()
        live = [
            LiveWindow(window_id=w.window_id, window_name=w.window_name)
            for w in windows
        ]

        # Backends whose ids are not stable across a server restart (herdr)
        # re-resolve by durable agent session id instead of display name. The
        # live id -> session_id map comes from the hook-written session_map.
        caps = tmux_manager.capabilities
        live_session_ids: dict[str, str] | None = None
        if not caps.ids_stable_across_restart:
            raw = await read_session_map_raw() or {}
            live_session_ids = live_window_session_ids(
                raw, {w.window_id for w in windows}
            )

        changed = _resolve(
            live,
            self.window_states,
            thread_router.thread_bindings,
            user_preferences.user_window_offsets,
            thread_router.window_display_names,
            ids_stable=caps.ids_stable_across_restart,
            live_session_ids=live_session_ids,
        )

        if changed:
            thread_router._rebuild_reverse_index()
            self._save_state()
            logger.info("Startup re-resolution complete")

        # Prune session_map.json entries for dead windows
        live_ids = {w.window_id for w in live}
        session_map_sync.prune_session_map(live_ids)

        # Sync display names from live tmux windows (detect external renames)
        live_pairs = [(w.window_id, w.window_name) for w in live]
        self.sync_display_names(live_pairs)

        # Prune orphaned display names (preserve group_chat_ids for post-restart topic creation)
        self.prune_stale_state(live_ids, skip_chat_ids=True)

    # --- Display name management (delegated to thread_router) ---

    def set_display_name(self, window_id: str, window_name: str) -> None:
        """Update display name for a window_id."""
        thread_router.set_display_name(window_id, window_name)
        # Also update WindowState if it exists
        ws = self.window_states.get(window_id)
        if ws:
            ws.window_name = window_name

    def sync_display_names(self, live_windows: list[tuple[str, str]]) -> bool:
        """Sync display names from live tmux windows. Returns True if changed."""
        router_changed = thread_router.sync_display_names(live_windows)
        # Always reconcile WindowState.window_name — the router may already
        # have the correct name while WindowState is still stale from older
        # persisted state.
        ws_changed = False
        for window_id, window_name in live_windows:
            ws = self.window_states.get(window_id)
            if ws and ws.window_name != window_name:
                ws.window_name = window_name
                ws_changed = True
        # Router saves itself when router_changed; persist WindowState repairs
        # even when the router side was already correct.
        if ws_changed and not router_changed:
            self._save_state()
        return router_changed or ws_changed

    def prune_stale_state(
        self, live_window_ids: set[str], *, skip_chat_ids: bool = False
    ) -> bool:
        """Remove orphaned entries from window_display_names and group_chat_ids.

        Returns True if any changes were made.
        When skip_chat_ids=True, group_chat_ids are preserved (used during startup
        so they remain available for post-restart topic creation).
        """
        # Collect window_ids that are "in use" (bound or have window_states)
        in_use = set(self.window_states.keys())
        for bindings in thread_router.thread_bindings.values():
            in_use.update(bindings.values())

        # Prune window_display_names for dead windows not in use and not live
        stale_display = [
            wid
            for wid in thread_router.window_display_names
            if wid not in live_window_ids and wid not in in_use
        ]

        # Collect all bound thread keys "user_id:thread_id"
        bound_keys: set[str] = set()
        for user_id, bindings in thread_router.thread_bindings.items():
            for thread_id in bindings:
                bound_keys.add(f"{user_id}:{thread_id}")

        # Prune group_chat_ids for unbound threads (unless skipped)
        stale_chat = (
            []
            if skip_chat_ids
            else [k for k in thread_router.group_chat_ids if k not in bound_keys]
        )

        # Prune stale byte offsets (independent of display/chat pruning)
        all_known = live_window_ids | in_use
        offsets_changed = user_preferences.prune_stale_offsets(all_known)

        if not stale_display and not stale_chat:
            return offsets_changed

        for wid in stale_display:
            name = thread_router.pop_display_name(wid)
            logger.debug("Pruning stale display name: %s (%s)", wid, name)
        for key in stale_chat:
            logger.debug("Pruning stale group_chat_id: %s", key)
            del thread_router.group_chat_ids[key]
        logger.info(
            "Pruned stale state: %d display name(s), %d group chat id(s)",
            len(stale_display),
            len(stale_chat),
        )

        self._save_state()
        return True

    def _get_session_map_window_ids(self) -> set[str]:
        """Read session_map.json and return window IDs tracked by ccgram.

        Native windows are stripped to their @id form.
        """
        if not config.session_map_file.exists():
            return set()
        try:
            raw = json.loads(config.session_map_file.read_text())
        except (json.JSONDecodeError, OSError):  # fmt: skip
            return set()
        prefix = session_map_prefix()
        result: set[str] = set()
        for key in raw:
            if key.startswith(prefix):
                wid = key[len(prefix) :]
                if self._is_window_id(wid):
                    result.add(wid)
        return result

    def audit_state(
        self,
        live_window_ids: set[str],
        live_windows: list[tuple[str, str]],
    ) -> AuditResult:
        """Read-only audit of all state maps against live tmux windows.

        Args:
            live_window_ids: Set of currently alive tmux window IDs.
            live_windows: List of (window_id, window_name) for live windows.

        Returns:
            AuditResult with discovered issues.
        """
        issues: list[AuditIssue] = []

        # Collect all bound window IDs
        bound_window_ids: set[str] = set()
        total_bindings = 0
        live_binding_count = 0
        for _uid, bindings in thread_router.thread_bindings.items():
            for _tid, wid in bindings.items():
                total_bindings += 1
                bound_window_ids.add(wid)
                if wid in live_window_ids:
                    live_binding_count += 1

        session_map_wids = self._get_session_map_window_ids()

        # 1. Ghost bindings (thread → dead window) — fixable (close topic)
        for uid, bindings in thread_router.thread_bindings.items():
            for tid, wid in bindings.items():
                if wid not in live_window_ids:
                    display = thread_router.get_display_name(wid)
                    issues.append(
                        AuditIssue(
                            category="ghost_binding",
                            detail=f"user:{uid} thread:{tid} window:{wid} ({display})",
                            fixable=True,
                        )
                    )

        # 2. Orphaned display names
        in_use = set(self.window_states.keys()) | bound_window_ids
        for wid in thread_router.window_display_names:
            if wid not in live_window_ids and wid not in in_use:
                name = thread_router.get_display_name(wid)
                issues.append(
                    AuditIssue(
                        category="orphaned_display_name",
                        detail=f"{wid} ({name})",
                        fixable=True,
                    )
                )

        # 3. Orphaned group_chat_ids
        bound_keys: set[str] = set()
        for user_id, bindings in thread_router.thread_bindings.items():
            for thread_id in bindings:
                bound_keys.add(f"{user_id}:{thread_id}")
        for key in thread_router.group_chat_ids:
            if key not in bound_keys:
                issues.append(
                    AuditIssue(
                        category="orphaned_group_chat_id",
                        detail=f"key {key}",
                        fixable=True,
                    )
                )

        # 4. Stale window_states (not in session_map, not bound, not live)
        for wid in self.window_states:
            if (
                wid not in session_map_wids
                and wid not in bound_window_ids
                and wid not in live_window_ids
            ):
                display = self.window_states[wid].window_name or wid
                issues.append(
                    AuditIssue(
                        category="stale_window_state",
                        detail=f"{wid} ({display})",
                        fixable=True,
                    )
                )

        # 5. Stale user_window_offsets
        known_wids = live_window_ids | bound_window_ids | set(self.window_states.keys())
        for uid, offsets in user_preferences.user_window_offsets.items():
            for wid in offsets:
                if wid not in known_wids:
                    issues.append(
                        AuditIssue(
                            category="stale_offset",
                            detail=f"user {uid}, window {wid}",
                            fixable=True,
                        )
                    )

        # 6. Display name drift (stored != tmux)
        for wid, tmux_name in live_windows:
            stored_name = thread_router.window_display_names.get(wid)
            if stored_name and stored_name != tmux_name:
                issues.append(
                    AuditIssue(
                        category="display_name_drift",
                        detail=f"{wid}: stored={stored_name!r} tmux={tmux_name!r}",
                        fixable=True,
                    )
                )

        # 7. Orphaned tmux windows (live, known to ccgram, but not bound to any topic)
        known_wids = session_map_wids | set(self.window_states.keys())
        for wid in live_window_ids:
            if wid not in bound_window_ids and wid in known_wids:
                name = dict(live_windows).get(wid, wid)
                issues.append(
                    AuditIssue(
                        category="orphaned_window",
                        detail=f"{wid} ({name})",
                        fixable=True,
                    )
                )

        return AuditResult(
            issues=issues,
            total_bindings=total_bindings,
            live_binding_count=live_binding_count,
        )

    def prune_stale_window_states(self, live_window_ids: set[str]) -> bool:
        """Remove window_states not in session_map, not bound, and not live.

        Returns True if any changes were made.
        """
        session_map_wids = self._get_session_map_window_ids()
        bound_window_ids: set[str] = set()
        for bindings in thread_router.thread_bindings.values():
            bound_window_ids.update(bindings.values())

        stale = [
            wid
            for wid in self.window_states
            if (
                wid not in session_map_wids
                and wid not in bound_window_ids
                and wid not in live_window_ids
            )
        ]
        if not stale:
            return False
        for wid in stale:
            logger.debug("Pruning stale window_state: %s", wid)
            del self.window_states[wid]
        logger.info("Pruned %d stale window_state(s)", len(stale))
        self._save_state()
        return True

    # --- Window state management ---

    def view_window(self, window_id: str) -> WindowView | None:
        """Read-only snapshot of a window's state.

        Returns ``None`` when no state exists for the window. Prefer this
        over ``get_window_state`` for read-only callers — it documents the
        exact fields the caller depends on and insulates them from internal
        WindowState shape changes.
        """
        # Lazy: window_query imports SessionManager-adjacent modules; keep local
        from .window_query import view_window as _view_window

        return _view_window(window_id)

    @property
    def window_count(self) -> int:
        """Number of tracked windows — use instead of accessing window_states directly."""
        return len(window_store.window_states)

    def iter_window_ids(self) -> list[str]:
        """All tracked window IDs — use instead of accessing window_states.keys() directly."""
        return list(window_store.window_states.keys())

    # --- Provider management ---

    def set_window_provider(
        self,
        window_id: str,
        provider_name: str,
        *,
        cwd: str | None = None,
    ) -> None:
        """Set the provider for a window.

        Resolves whether the new provider supports hooks so that
        ``window_state_store`` remains free of provider imports.
        """
        supports_hook = True
        if provider_name:
            # Lazy: providers.registry imports concrete provider modules
            # which transitively touch session state; keep lookup local.
            from .providers.registry import UnknownProviderError, registry

            try:
                supports_hook = registry.get(provider_name).capabilities.supports_hook
            except UnknownProviderError:
                supports_hook = True
        window_store.set_window_provider(
            window_id,
            provider_name,
            cwd=cwd,
            new_provider_supports_hook=supports_hook,
        )

    def _clear_session_map_entry(self, window_id: str) -> None:
        """Delegate to session_map_sync — see session_map.py for implementation."""
        session_map_sync.clear_session_map_entry(window_id)

    def set_window_cwd(self, window_id: str, cwd: str) -> None:
        """Set the working directory for a window and persist state."""
        state = window_store.get_window_state(window_id)
        state.cwd = cwd
        self._save_state()

    def set_window_origin(self, window_id: str, origin: str) -> None:
        """Set the lifecycle origin for a window and persist state."""
        _lifecycle_state.set_window_origin(window_id, origin)

    def set_window_worktree(
        self, window_id: str, worktree_path: str, branch: str
    ) -> None:
        """Persist the git worktree path + branch for a window.

        Set when a new topic was created on a fresh worktree. No
        behaviour reads these yet — a forward investment for the
        eventual worktree cleanup UX.
        """
        _worktree_state.set_worktree(window_id, worktree_path, branch)

    def get_approval_mode(self, window_id: str) -> str:
        """Get approval mode for a window (default: 'normal')."""
        return _identity_state.get_approval_mode(window_id)

    def set_window_approval_mode(self, window_id: str, mode: str) -> None:
        """Set approval mode for a window."""
        _identity_state.set_window_approval_mode(window_id, mode.lower())

    # --- Batch mode ---

    def get_batch_mode(self, window_id: str) -> str:
        """Get batch mode for a window."""
        return _tool_state.get_batch_mode(window_id)

    def set_batch_mode(self, window_id: str, mode: str) -> None:
        """Set batch mode for a window."""
        _tool_state.set_batch_mode(window_id, mode)

    def cycle_batch_mode(self, window_id: str) -> str:
        """Cycle batch mode: batched → ephemeral → verbose → batched. Returns new mode."""
        return _tool_state.cycle_batch_mode(window_id)

    # --- Tool-call visibility ---

    def get_tool_call_visibility(self, window_id: str) -> str:
        """Get tool-call visibility for a window (default: 'default')."""
        return _tool_state.get_tool_call_visibility(window_id)

    def set_tool_call_visibility(self, window_id: str, mode: str) -> None:
        """Set tool-call visibility for a window."""
        _tool_state.set_tool_call_visibility(window_id, mode)

    def cycle_tool_call_visibility(self, window_id: str) -> str:
        """Cycle tool-call visibility: default → shown → hidden → default. Returns new mode."""
        return _tool_state.cycle_tool_call_visibility(window_id)


session_manager = SessionManager()
