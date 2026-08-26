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
from copy import deepcopy

import structlog
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import config
from .session_map import (
    SessionMapSync,
    install_session_map_sync,
    is_backend_window_id,
    session_map_prefix,
    session_map_sync,
)
from .state_persistence import StatePersistence
from .multiplexer import multiplexer as tmux_manager
from .multiplexer.reconciliation import list_windows_for_reconciliation
from .thread_router import ThreadRouter, install_thread_router, thread_router
from .user_preferences import (
    UserPreferences,
    install_user_preferences,
    user_preferences,
)
from .utils import atomic_write_json
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
            default_group_id=config.group_id,
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

        Backend-aware: tmux ``@N`` ids and guarded opaque Herdr session
        targets (see ``session_map.is_backend_window_id``). Raw Herdr tab/pane
        locators are legacy migration records, never valid identities.
        Old-format (window-name) keys return False on tmux so startup
        re-resolution migrates them.
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
        migrated = False

        # Herdr topics created before guarded session targets were tab/pane
        # bindings. Preserve them for an explicit archive/rollback migration,
        # but make them incapable of authorizing any action.
        if config.multiplexer_name == "herdr":
            legacy_ids = set(window_store.window_states)
            legacy_ids.update(
                window_id
                for bindings in state.get("thread_bindings", {}).values()
                if isinstance(bindings, dict)
                for window_id in bindings.values()
                if isinstance(window_id, str)
            )
            migrated = False
            for window_id in legacy_ids:
                # Do not use any(generator): it short-circuits after the first
                # mutation and leaves later legacy IDs accidentally actionable.
                migrated = (
                    window_store.mark_legacy_herdr(window_id, schedule_save=False)
                    or migrated
                )
            if migrated:
                logger.info("Marked legacy Herdr bindings for explicit rebind")

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
        if migrated:
            self._save_state()

    @staticmethod
    def _unique_window_aliases(
        windows: Sequence[Any], attribute: str
    ) -> dict[str, str]:
        """Return aliases with exactly one fresh canonical attestation."""
        candidates: dict[str, set[str]] = {}
        for window in windows:
            canonical_id = getattr(window, "window_id", "")
            if not canonical_id:
                continue
            for alias_id in getattr(window, attribute, ()):
                if alias_id and alias_id != canonical_id:
                    candidates.setdefault(alias_id, set()).add(canonical_id)
        aliases: dict[str, str] = {}
        for alias_id, canonical_ids in candidates.items():
            if len(canonical_ids) == 1:
                aliases[alias_id] = next(iter(canonical_ids))
            else:
                logger.warning(
                    "Identity migration deferred: alias has %d live targets; "
                    "explicitly rebind the affected topic",
                    len(canonical_ids),
                    alias=alias_id,
                )
        return aliases

    def _backup_identity_migration_state(self) -> bool:
        """Persist one pre-migration state snapshot for rollback/recovery."""
        backup = config.state_file.with_name(
            f"{config.state_file.name}.identity-migration.bak"
        )
        if backup.exists():
            return True
        try:
            atomic_write_json(backup, self._serialize_state())
        except OSError as exc:
            logger.warning("Identity migration deferred: state backup failed: %s", exc)
            return False
        return True

    def reconcile_window_aliases(self, windows: Sequence[Any]) -> None:
        """Converge uniquely-attested aliases through the single fold owner.

        Canonical aliases describe a target supersession.  Legacy aliases are
        adapter-provided migration evidence only; raw locators remain blocked
        unless one fresh listing attests exactly one target.  The file re-key
        happens under the hook lock before any in-memory state is changed, so
        a failed persistence operation leaves every coupled in-memory store
        intact for the next reconciliation.
        """
        # Lazy: same cycle as resolve_stale_ids — window_resolver imports
        # session-state types, so hoisting forms a session → window_resolver →
        # session.WindowState cycle.
        from .window_resolver import migrate_window_aliases

        aliases = self._unique_window_aliases(windows, "alias_window_ids")
        legacy_aliases = self._unique_window_aliases(windows, "legacy_alias_window_ids")
        aliases.update(legacy_aliases)
        if not aliases:
            return

        # Validate the complete in-memory fold before changing its coupled
        # hook file.  The actual maps remain unmodified until the locked write
        # and its recoverable backups have succeeded.
        cloned_states = deepcopy(self.window_states)
        cloned_bindings = deepcopy(thread_router.thread_bindings)
        cloned_chat_bindings = deepcopy(thread_router.chat_thread_bindings)
        cloned_offsets = deepcopy(user_preferences.user_window_offsets)
        cloned_display_names = deepcopy(thread_router.window_display_names)
        migrations = migrate_window_aliases(
            aliases,
            cloned_states,
            cloned_bindings,
            cloned_chat_bindings,
            cloned_offsets,
            cloned_display_names,
            record_redirects=False,
        )
        if migrations and not self._backup_identity_migration_state():
            return
        if not session_map_sync.rename_session_map_entries(list(aliases.items())):
            logger.warning(
                "Identity migration deferred: session_map re-key failed; "
                "state was preserved for retry"
            )
            return

        applied = migrate_window_aliases(
            aliases,
            self.window_states,
            thread_router.thread_bindings,
            thread_router.chat_thread_bindings,
            user_preferences.user_window_offsets,
            thread_router.window_display_names,
        )
        for migration in applied:
            if migration.alias_id not in legacy_aliases:
                continue
            state = self.window_states.get(migration.canonical_id)
            if state is not None:
                # The adapter just supplied the evidence that unblocks this
                # old record.  Archive metadata applies only to the raw key.
                state.legacy_herdr = False
                state.legacy_herdr_archived = False
                state.legacy_herdr_archive_user_id = None
                state.legacy_herdr_archive_thread_id = None
        if migrations or applied:
            thread_router._rebuild_reverse_index()
            self._save_state()

    async def resolve_stale_ids(self) -> None:
        """Re-resolve persisted window IDs against live tmux windows.

        Called on startup. Delegates to window_resolver for the heavy lifting.
        Dead window bindings and states are preserved for /restore recovery.
        """
        # Lazy: window_resolver imports session-state types; hoisting forms
        # session → window_resolver → session.WindowState cycle.
        # Lazy: window_resolver pulls back into session manager
        from .window_resolver import LiveWindow, resolve_stale_ids as _resolve

        windows = await list_windows_for_reconciliation(tmux_manager)
        if windows is None:
            logger.warning(
                "Startup window reconciliation skipped: multiplexer listing unavailable"
            )
            return

        # Alias convergence must happen before startup adoption and before
        # any session-map reader can discard an old key.  This also gives an
        # old tab/pane persistence record exactly one fresh, adapter-attested
        # chance to become its canonical durable target.
        self.reconcile_window_aliases(windows)

        live = [
            LiveWindow(window_id=w.window_id, window_name=w.window_name)
            for w in windows
        ]

        # Durable session targets are retained exactly as persisted. Their
        # current availability is decided by a fresh guarded action, never by
        # session-map, display-name, tab, pane, or focus recovery.
        caps = tmux_manager.capabilities
        changed = _resolve(
            live,
            self.window_states,
            thread_router.thread_bindings,
            user_preferences.user_window_offsets,
            thread_router.window_display_names,
            ids_stable=caps.ids_stable_across_restart,
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
        in_use.update(thread_router.all_bound_window_ids())

        # Prune window_display_names for dead windows not in use and not live
        stale_display = [
            wid
            for wid in thread_router.window_display_names
            if wid not in live_window_ids and wid not in in_use
        ]

        # Collect all bound thread keys "user_id:thread_id"
        bound_keys: set[str] = {
            f"{user_id}:{thread_id}"
            for user_id, thread_id, _ in thread_router.iter_thread_bindings()
        }

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
        if not isinstance(raw, dict):
            return set()
        # Lazy: state-file schema stays isolated from the SessionManager.
        from .hooks.state_files import StateFileValidationError, parse_session_map_entry

        prefix = session_map_prefix()
        result: set[str] = set()
        for key, info in raw.items():
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            wid = key[len(prefix) :]
            if not self._is_window_id(wid) or not isinstance(info, dict):
                continue
            try:
                parse_session_map_entry(info)
            except StateFileValidationError:
                continue
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
        for _uid, _tid, wid in thread_router.iter_thread_bindings():
            total_bindings += 1
            bound_window_ids.add(wid)
            if wid in live_window_ids:
                live_binding_count += 1

        session_map_wids = self._get_session_map_window_ids()

        # 1. Legacy Herdr bindings are retained for archive/rollback but never
        # actioned or implicitly remapped. Report their explicit rebind path
        # instead of classifying the old tab/pane locator as a ghost.
        legacy_bindings = {
            wid
            for _uid, _tid, wid in thread_router.iter_thread_bindings()
            if window_store.is_legacy_herdr(wid)
        }
        for wid in sorted(legacy_bindings):
            issues.append(
                AuditIssue(
                    category="legacy_herdr",
                    detail=f"{wid} is blocked; archive or explicitly rebind to a listed session target",
                    fixable=False,
                )
            )

        # 2. Ghost bindings (thread → dead window) — fixable (close topic)
        for uid, tid, wid in thread_router.iter_thread_bindings():
            if wid not in live_window_ids and wid not in legacy_bindings:
                display = thread_router.get_display_name(wid)
                issues.append(
                    AuditIssue(
                        category="ghost_binding",
                        detail=f"user:{uid} thread:{tid} window:{wid} ({display})",
                        fixable=True,
                    )
                )

        # 3. Orphaned display names
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
        bound_keys: set[str] = {
            f"{user_id}:{thread_id}"
            for user_id, thread_id, _ in thread_router.iter_thread_bindings()
        }
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
        bound_window_ids = thread_router.all_bound_window_ids()

        stale = [
            wid
            for wid in self.window_states
            if (
                wid not in session_map_wids
                and wid not in bound_window_ids
                and wid not in live_window_ids
                and not window_store.is_archived_legacy_herdr(wid)
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
