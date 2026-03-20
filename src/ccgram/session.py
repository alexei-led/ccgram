"""Claude Code session management — the core state hub.

Manages the key mappings:
  Window→Session (window_states): which Claude session_id a window holds (keyed by window_id).
  User→Thread→Window (thread_bindings): topic-to-window bindings (1 topic = 1 window_id).

Responsibilities:
  - Persist/load state to ~/.ccgram/state.json.
  - Sync window↔session bindings from session_map.json (written by hook).
  - Resolve window IDs to ClaudeSession objects (JSONL file reading).
  - Track per-user read offsets for unread-message detection.
  - Manage thread↔window bindings for Telegram topic routing.
  - Send keystrokes to tmux windows and retrieve message history.
  - Maintain window_id→display name mapping for UI display.
  - Re-resolve stale window IDs on startup (tmux server restart recovery).

Key class: SessionManager (singleton instantiated as `session_manager`).
Key methods for thread binding access:
  - resolve_window_for_thread: Get window_id for a user's thread
  - iter_thread_bindings: Generator for iterating all (user_id, thread_id, window_id)
  - find_users_for_session: Find all users bound to a session_id
"""

import asyncio
import json
import structlog
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Self

import aiofiles

from .config import config
from .handlers.callback_data import NOTIFICATION_MODES
from .providers import get_provider_for_window
from .state_persistence import StatePersistence
from .tmux_manager import tmux_manager
from .utils import atomic_write_json
from .window_resolver import EMDASH_SESSION_PREFIX, is_foreign_window, is_window_id

logger = structlog.get_logger()

APPROVAL_MODES: frozenset[str] = frozenset({"normal", "yolo"})
DEFAULT_APPROVAL_MODE = "normal"
YOLO_APPROVAL_MODE = "yolo"

BATCH_MODES: frozenset[str] = frozenset({"batched", "verbose"})
DEFAULT_BATCH_MODE = "batched"


_LEGACY_SESSION_PREFIX = "ccbot:"


def parse_session_map(raw: dict[str, Any], prefix: str) -> dict[str, dict[str, str]]:
    """Parse session_map.json entries matching a tmux session prefix.

    Also matches legacy "ccbot:" prefix keys when the current prefix is "ccgram:".
    Returns {window_name: {"session_id": ..., "cwd": ...}} for matching entries.
    """
    result: dict[str, dict[str, str]] = {}
    # Also accept legacy "ccbot:" prefix keys when session is "ccgram"
    legacy_prefix = _LEGACY_SESSION_PREFIX if prefix.startswith("ccgram:") else ""
    for key, info in raw.items():
        if key.startswith(prefix):
            window_name = key[len(prefix) :]
        elif legacy_prefix and key.startswith(legacy_prefix):
            window_name = key[len(legacy_prefix) :]
        else:
            continue
        if not isinstance(info, dict):
            continue
        session_id = info.get("session_id", "")
        if session_id:
            result[window_name] = {
                "session_id": session_id,
                "cwd": info.get("cwd", ""),
                "window_name": info.get("window_name", ""),
                "transcript_path": info.get("transcript_path", ""),
                "provider_name": info.get("provider_name", ""),
            }
    return result


def parse_emdash_provider(session_name: str) -> str:
    """Extract provider name from emdash session name.

    Format: emdash-{provider}-main-{id} or emdash-{provider}-chat-{id}
    """
    for sep in ("-main-", "-chat-"):
        if sep in session_name:
            prefix = session_name.split(sep)[0]
            return prefix.removeprefix(EMDASH_SESSION_PREFIX)
    return ""


@dataclass
class WindowState:
    """Persistent state for a tmux window.

    Attributes:
        session_id: Associated Claude session ID (empty if not yet detected)
        cwd: Working directory for direct file path construction
        window_name: Display name of the window
        transcript_path: Direct path to JSONL transcript file (from hook payload)
        notification_mode: "all" | "errors_only" | "muted"
        approval_mode: "normal" | "yolo"
        external: True for windows owned by external tools (emdash) — never killed by ccgram
    """

    session_id: str = ""
    cwd: str = ""
    window_name: str = ""
    transcript_path: str = ""
    notification_mode: str = "all"
    provider_name: str = ""
    approval_mode: str = DEFAULT_APPROVAL_MODE
    batch_mode: str = DEFAULT_BATCH_MODE
    external: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "cwd": self.cwd,
        }
        if self.window_name:
            d["window_name"] = self.window_name
        if self.transcript_path:
            d["transcript_path"] = self.transcript_path
        if self.notification_mode != "all":
            d["notification_mode"] = self.notification_mode
        if self.provider_name:
            d["provider_name"] = self.provider_name
        if self.approval_mode != DEFAULT_APPROVAL_MODE:
            d["approval_mode"] = self.approval_mode
        if self.batch_mode != DEFAULT_BATCH_MODE:
            d["batch_mode"] = self.batch_mode
        if self.external:
            d["external"] = True
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            session_id=data.get("session_id", ""),
            cwd=data.get("cwd", ""),
            window_name=data.get("window_name", ""),
            transcript_path=data.get("transcript_path", ""),
            notification_mode=data.get("notification_mode", "all"),
            provider_name=data.get("provider_name", ""),
            approval_mode=data.get("approval_mode", DEFAULT_APPROVAL_MODE),
            batch_mode=data.get("batch_mode", DEFAULT_BATCH_MODE),
            external=data.get("external", False),
        )


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
class ClaudeSession:
    """Information about a Claude Code session."""

    session_id: str
    summary: str
    message_count: int
    file_path: str


@dataclass
class SessionManager:
    """Manages session state for Claude Code.

    All internal keys use window_id (e.g. '@0', '@12') for uniqueness.
    Display names (window_name) are stored separately for UI presentation.

    window_states: window_id -> WindowState (session_id, cwd, window_name)
    user_window_offsets: user_id -> {window_id -> byte_offset}
    thread_bindings: user_id -> {thread_id -> window_id}
    window_display_names: window_id -> window_name (for display)
    """

    window_states: dict[str, WindowState] = field(default_factory=dict)
    user_window_offsets: dict[int, dict[str, int]] = field(default_factory=dict)
    thread_bindings: dict[int, dict[int, str]] = field(default_factory=dict)
    # group_chat_ids: "user_id:thread_id" -> chat_id (supports multiple groups per user)
    group_chat_ids: dict[str, int] = field(default_factory=dict)
    # window_id -> display name (window_name)
    window_display_names: dict[str, str] = field(default_factory=dict)
    # User directory favorites: user_id -> {"starred": [...], "mru": [...]}
    user_dir_favorites: dict[int, dict[str, list[str]]] = field(default_factory=dict)

    # Reverse index: (user_id, window_id) -> thread_id for O(1) inbound lookups
    _window_to_thread: dict[tuple[int, str], int] = field(
        default_factory=dict, repr=False
    )

    # Delegated persistence (not serialized)
    _persistence: StatePersistence = field(default=None, repr=False, init=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._persistence = StatePersistence(config.state_file, self._serialize_state)
        self._load_state()
        self._rebuild_reverse_index()

    def _rebuild_reverse_index(self) -> None:
        """Rebuild _window_to_thread from thread_bindings."""
        self._window_to_thread = {}
        for uid, bindings in self.thread_bindings.items():
            for tid, wid in bindings.items():
                self._window_to_thread[(uid, wid)] = tid

    def _serialize_state(self) -> dict[str, Any]:
        """Serialize all state to a dict for persistence."""
        return {
            "window_states": {k: v.to_dict() for k, v in self.window_states.items()},
            "user_window_offsets": {
                str(uid): offsets for uid, offsets in self.user_window_offsets.items()
            },
            "thread_bindings": {
                str(uid): {str(tid): wid for tid, wid in bindings.items()}
                for uid, bindings in self.thread_bindings.items()
            },
            "group_chat_ids": self.group_chat_ids,
            "window_display_names": self.window_display_names,
            "user_dir_favorites": {
                str(uid): favs for uid, favs in self.user_dir_favorites.items()
            },
        }

    def _save_state(self) -> None:
        """Schedule debounced save (0.5s delay, resets on each call)."""
        self._persistence.schedule_save()

    def flush_state(self) -> None:
        """Force immediate save. Call on shutdown."""
        self._persistence.flush()

    def _is_window_id(self, key: str) -> bool:
        """Check if a key looks like a tmux window ID (e.g. '@0', '@12')."""
        return is_window_id(key)

    def _load_state(self) -> None:
        """Load state during initialization.

        Detects old-format state (window_name keys without '@' prefix) and
        marks for migration on next startup re-resolution.
        """
        state = self._persistence.load()
        if not state:
            self._needs_migration = False
            return

        self.window_states = {
            k: WindowState.from_dict(v)
            for k, v in state.get("window_states", {}).items()
        }
        self.user_window_offsets = {
            int(uid): offsets
            for uid, offsets in state.get("user_window_offsets", {}).items()
        }
        self.thread_bindings = {
            int(uid): {int(tid): wid for tid, wid in bindings.items()}
            for uid, bindings in state.get("thread_bindings", {}).items()
        }
        self.group_chat_ids = state.get("group_chat_ids", {})
        self.window_display_names = state.get("window_display_names", {})
        self.user_dir_favorites = {
            int(uid): favs for uid, favs in state.get("user_dir_favorites", {}).items()
        }

        # Deduplicate thread bindings: enforce 1 window = 1 thread.
        # If multiple threads point to the same window, keep only the
        # highest thread_id (most recently created topic).
        for _uid, bindings in self.thread_bindings.items():
            window_threads: dict[str, list[int]] = {}
            for tid, wid in bindings.items():
                window_threads.setdefault(wid, []).append(tid)
            for wid, tids in window_threads.items():
                if len(tids) > 1:
                    keep = max(tids)
                    for tid in tids:
                        if tid != keep:
                            del bindings[tid]
                            logger.warning(
                                "Startup: removed duplicate binding "
                                "thread %d -> window %s (keeping %d)",
                                tid,
                                wid,
                                keep,
                            )

        # Detect old format: keys that don't look like window IDs
        # Foreign windows (emdash) use qualified IDs — not old format.
        needs_migration = False
        for k in self.window_states:
            if not self._is_window_id(k) and not is_foreign_window(k):
                needs_migration = True
                break
        if not needs_migration:
            for bindings in self.thread_bindings.values():
                for wid in bindings.values():
                    if not self._is_window_id(wid) and not is_foreign_window(wid):
                        needs_migration = True
                        break
                if needs_migration:
                    break

        if needs_migration:
            logger.info(
                "Detected old-format state (window_name keys), "
                "will re-resolve on startup"
            )
            self._needs_migration = True
        else:
            self._needs_migration = False

    async def resolve_stale_ids(self) -> None:
        """Re-resolve persisted window IDs against live tmux windows.

        Called on startup. Delegates to window_resolver for the heavy lifting.
        Dead window bindings and states are preserved for /restore recovery.
        """
        from .window_resolver import LiveWindow, resolve_stale_ids as _resolve

        windows = await tmux_manager.list_windows()
        live = [
            LiveWindow(window_id=w.window_id, window_name=w.window_name)
            for w in windows
        ]

        changed = _resolve(
            live,
            self.window_states,
            self.thread_bindings,
            self.user_window_offsets,
            self.window_display_names,
        )

        if changed:
            self._rebuild_reverse_index()
            self._save_state()
            logger.info("Startup re-resolution complete")

        self._needs_migration = False

        # Prune session_map.json entries for dead windows
        live_ids = {w.window_id for w in live}
        self.prune_session_map(live_ids)

        # Sync display names from live tmux windows (detect external renames)
        live_pairs = [(w.window_id, w.window_name) for w in live]
        self.sync_display_names(live_pairs)

        # Prune orphaned display names (preserve group_chat_ids for post-restart topic creation)
        self.prune_stale_state(live_ids, skip_chat_ids=True)

    # --- Display name management ---

    def get_display_name(self, window_id: str) -> str:
        """Get display name for a window_id, fallback to window_id itself."""
        return self.window_display_names.get(window_id, window_id)

    def set_display_name(self, window_id: str, window_name: str) -> None:
        """Update display name for a window_id."""
        if self.window_display_names.get(window_id) != window_name:
            self.window_display_names[window_id] = window_name
            # Also update WindowState if it exists
            ws = self.window_states.get(window_id)
            if ws:
                ws.window_name = window_name
            self._save_state()

    def sync_display_names(self, live_windows: list[tuple[str, str]]) -> bool:
        """Sync display names from live tmux windows. Returns True if changed.

        Saves state internally when changes are detected.
        """
        changed = False
        for window_id, window_name in live_windows:
            old = self.window_display_names.get(window_id)
            if old and old != window_name:
                self.window_display_names[window_id] = window_name
                ws = self.window_states.get(window_id)
                if ws:
                    ws.window_name = window_name
                changed = True
                logger.info(
                    "Synced display name: %s %s → %s", window_id, old, window_name
                )
        if changed:
            self._save_state()
        return changed

    async def wait_for_session_map_entry(
        self, window_id: str, timeout: float = 5.0, interval: float = 0.5
    ) -> bool:
        """Poll session_map.json until an entry for window_id appears.

        Returns True if the entry was found within timeout, False otherwise.
        """
        logger.debug(
            "Waiting for session_map entry: window_id=%s, timeout=%.1f",
            window_id,
            timeout,
        )
        key = f"{config.tmux_session_name}:{window_id}"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            try:
                if config.session_map_file.exists():
                    async with aiofiles.open(config.session_map_file, "r") as f:
                        content = await f.read()
                    session_map = json.loads(content)
                    info = session_map.get(key, {})
                    if info.get("session_id"):
                        # Found — load into window_states immediately
                        logger.debug(
                            "session_map entry found for window_id %s", window_id
                        )
                        await self.load_session_map()
                        return True
            except (json.JSONDecodeError, OSError):  # fmt: skip
                pass
            await asyncio.sleep(interval)
        logger.warning(
            "Timed out waiting for session_map entry: window_id=%s", window_id
        )
        return False

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
        for bindings in self.thread_bindings.values():
            in_use.update(bindings.values())

        # Prune window_display_names for dead windows not in use and not live
        stale_display = [
            wid
            for wid in self.window_display_names
            if wid not in live_window_ids and wid not in in_use
        ]

        # Collect all bound thread keys "user_id:thread_id"
        bound_keys: set[str] = set()
        for user_id, bindings in self.thread_bindings.items():
            for thread_id in bindings:
                bound_keys.add(f"{user_id}:{thread_id}")

        # Prune group_chat_ids for unbound threads (unless skipped)
        stale_chat = (
            []
            if skip_chat_ids
            else [k for k in self.group_chat_ids if k not in bound_keys]
        )

        if not stale_display and not stale_chat:
            return False

        for wid in stale_display:
            logger.info(
                "Pruning stale display name: %s (%s)",
                wid,
                self.window_display_names[wid],
            )
            del self.window_display_names[wid]
        for key in stale_chat:
            logger.info("Pruning stale group_chat_id: %s", key)
            del self.group_chat_ids[key]

        self._save_state()
        return True

    def prune_session_map(self, live_window_ids: set[str]) -> None:
        """Remove session_map.json entries for windows that no longer exist.

        Reads session_map.json, drops entries whose window_id is not in
        live_window_ids, and writes back only if changes were made.
        Also removes corresponding window_states.
        """
        if not config.session_map_file.exists():
            return
        try:
            raw = json.loads(config.session_map_file.read_text())
        except (json.JSONDecodeError, OSError):  # fmt: skip
            return

        prefix = f"{config.tmux_session_name}:"
        dead_entries: list[tuple[str, str]] = []  # (map_key, window_id)
        for key in raw:
            if not key.startswith(prefix):
                continue
            window_id = key[len(prefix) :]
            if self._is_window_id(window_id) and window_id not in live_window_ids:
                dead_entries.append((key, window_id))

        if not dead_entries:
            return

        changed_state = False
        for key, window_id in dead_entries:
            logger.info(
                "Pruning dead session_map entry: %s (window %s)", key, window_id
            )
            del raw[key]
            if window_id in self.window_states:
                del self.window_states[window_id]
                changed_state = True

        atomic_write_json(config.session_map_file, raw)
        if changed_state:
            self._save_state()

    def _get_session_map_window_ids(self) -> set[str]:
        """Read session_map.json and return window IDs tracked by ccgram.

        Includes native windows (stripped to @id) and emdash windows
        (full qualified key like "emdash-claude-main-xxx:@0").
        """
        if not config.session_map_file.exists():
            return set()
        try:
            raw = json.loads(config.session_map_file.read_text())
        except (json.JSONDecodeError, OSError):  # fmt: skip
            return set()
        prefix = f"{config.tmux_session_name}:"
        result: set[str] = set()
        for key in raw:
            if key.startswith(prefix):
                wid = key[len(prefix) :]
                if self._is_window_id(wid):
                    result.add(wid)
            elif key.startswith(EMDASH_SESSION_PREFIX):
                result.add(key)
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
        for _uid, bindings in self.thread_bindings.items():
            for _tid, wid in bindings.items():
                total_bindings += 1
                bound_window_ids.add(wid)
                if wid in live_window_ids:
                    live_binding_count += 1

        session_map_wids = self._get_session_map_window_ids()

        # 1. Ghost bindings (thread → dead window) — fixable (close topic)
        for uid, bindings in self.thread_bindings.items():
            for tid, wid in bindings.items():
                if wid not in live_window_ids:
                    display = self.get_display_name(wid)
                    issues.append(
                        AuditIssue(
                            category="ghost_binding",
                            detail=f"user:{uid} thread:{tid} window:{wid} ({display})",
                            fixable=True,
                        )
                    )

        # 2. Orphaned display names
        in_use = set(self.window_states.keys()) | bound_window_ids
        for wid in self.window_display_names:
            if wid not in live_window_ids and wid not in in_use:
                name = self.window_display_names[wid]
                issues.append(
                    AuditIssue(
                        category="orphaned_display_name",
                        detail=f"{wid} ({name})",
                        fixable=True,
                    )
                )

        # 3. Orphaned group_chat_ids
        bound_keys: set[str] = set()
        for user_id, bindings in self.thread_bindings.items():
            for thread_id in bindings:
                bound_keys.add(f"{user_id}:{thread_id}")
        for key in self.group_chat_ids:
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
        for uid, offsets in self.user_window_offsets.items():
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
            stored_name = self.window_display_names.get(wid)
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

    def prune_stale_offsets(self, known_window_ids: set[str]) -> bool:
        """Remove user_window_offsets entries for unknown windows.

        Returns True if any changes were made.
        """
        changed = False
        empty_users: list[int] = []
        for uid, offsets in self.user_window_offsets.items():
            stale = [wid for wid in offsets if wid not in known_window_ids]
            for wid in stale:
                logger.info("Pruning stale offset: user %d, window %s", uid, wid)
                del offsets[wid]
                changed = True
            if not offsets:
                empty_users.append(uid)
        for uid in empty_users:
            del self.user_window_offsets[uid]
            changed = True
        if changed:
            self._save_state()
        return changed

    def prune_stale_window_states(self, live_window_ids: set[str]) -> bool:
        """Remove window_states not in session_map, not bound, and not live.

        Returns True if any changes were made.
        """
        session_map_wids = self._get_session_map_window_ids()
        bound_window_ids: set[str] = set()
        for bindings in self.thread_bindings.values():
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
            logger.info("Pruning stale window_state: %s", wid)
            del self.window_states[wid]
        self._save_state()
        return True

    def _sync_window_from_session_map(
        self,
        window_id: str,
        info: dict[str, Any],
        *,
        mark_external: bool = False,
    ) -> bool:
        """Sync a single window's state from session_map entry.

        Returns True if any state was changed.
        """
        new_sid = info.get("session_id", "")
        if not new_sid:
            return False
        new_cwd = info.get("cwd", "")
        new_wname = info.get("window_name", "")
        new_transcript = info.get("transcript_path", "")
        changed = False

        state = self.get_window_state(window_id)
        if mark_external and not state.external:
            state.external = True
            changed = True
        if state.session_id != new_sid or state.cwd != new_cwd:
            logger.info(
                "Session map: window_id %s updated sid=%s, cwd=%s",
                window_id,
                new_sid,
                new_cwd,
            )
            state.session_id = new_sid
            state.cwd = new_cwd
            changed = True
        if new_transcript and state.transcript_path != new_transcript:
            state.transcript_path = new_transcript
            changed = True
        # Sync provider_name from session_map (hook data is authoritative).
        new_provider = info.get("provider_name", "")
        if new_provider and state.provider_name != new_provider:
            state.provider_name = new_provider
            changed = True
        # Initialize display name from session_map only when unknown.
        if (
            new_wname
            and not self.window_display_names.get(window_id)
            and not state.window_name
        ):
            state.window_name = new_wname
            self.window_display_names[window_id] = new_wname
            changed = True
        return changed

    async def load_session_map(self) -> None:
        """Read session_map.json and update window_states with new session associations.

        Keys in session_map are formatted as "tmux_session:window_id" (e.g. "ccgram:@12").
        Native entries (matching our tmux_session_name) and emdash entries (prefixed
        with "emdash-") are both processed. Emdash windows are marked as external.
        Also cleans up window_states entries not in current session_map.
        Updates window_display_names from the "window_name" field in values.
        """
        if not config.session_map_file.exists():
            return
        try:
            async with aiofiles.open(config.session_map_file, "r") as f:
                content = await f.read()
            session_map = json.loads(content)
        except (json.JSONDecodeError, OSError):  # fmt: skip
            return

        prefix = f"{config.tmux_session_name}:"
        valid_wids: set[str] = set()
        # Track session_ids from old-format entries so we don't nuke
        # migrated window_states before the new hook has fired.
        old_format_sids: set[str] = set()
        changed = False

        old_format_keys: list[str] = []
        for key, info in session_map.items():
            if not isinstance(info, dict):
                continue

            # Emdash entries: use the full key as window_id
            if key.startswith(EMDASH_SESSION_PREFIX):
                valid_wids.add(key)
                if self._sync_window_from_session_map(key, info, mark_external=True):
                    changed = True
                # Infer provider from session name — always attempt if missing,
                # regardless of whether _sync changed other fields.
                state = self.get_window_state(key)
                if not state.provider_name:
                    session_name = key.rsplit(":", 1)[0]
                    detected = parse_emdash_provider(session_name)
                    if detected:
                        state.provider_name = detected
                        changed = True
                continue

            # Native entries: strip prefix, process by window_id
            if not key.startswith(prefix):
                continue
            window_id = key[len(prefix) :]
            # Old-format key (window_name instead of window_id): remember the
            # session_id so migrated window_states survive stale cleanup,
            # then mark for removal from session_map.json.
            if not self._is_window_id(window_id):
                sid = info.get("session_id", "")
                if sid:
                    old_format_sids.add(sid)
                old_format_keys.append(key)
                continue
            valid_wids.add(window_id)
            if self._sync_window_from_session_map(window_id, info):
                changed = True

        # Clean up window_states entries not in current session_map.
        # Protect entries whose session_id is still referenced by old-format
        # keys — those sessions are valid but haven't re-triggered the hook yet.
        # Also protect entries bound to a topic (hookless providers like codex/gemini
        # never appear in session_map but still need their window state preserved).
        bound_wids = {
            wid
            for user_bindings in self.thread_bindings.values()
            for wid in user_bindings.values()
            if wid
        }
        stale_wids = [
            w
            for w in self.window_states
            if w
            and w not in valid_wids
            and w not in bound_wids
            and self.window_states[w].session_id not in old_format_sids
        ]
        for wid in stale_wids:
            logger.info("Removing stale window_state: %s", wid)
            del self.window_states[wid]
            changed = True

        # Purge old-format keys from session_map.json so they don't
        # get logged every poll cycle.
        if old_format_keys:
            for key in old_format_keys:
                logger.info("Removing old-format session_map key: %s", key)
                del session_map[key]
            atomic_write_json(config.session_map_file, session_map)

        if changed:
            self._save_state()

    def register_hookless_session(
        self,
        window_id: str,
        session_id: str,
        cwd: str,
        transcript_path: str,
        provider_name: str,
    ) -> None:
        """Register a session for a hookless provider (Codex, Gemini).

        Updates in-memory WindowState and schedules a debounced state save.
        Must be called from the event loop thread (not from asyncio.to_thread)
        because _save_state() touches asyncio timer handles.

        Pair with write_hookless_session_map() for the file-locked
        session_map.json write, which is safe to call from any thread.
        """
        state = self.get_window_state(window_id)
        state.session_id = session_id
        state.cwd = cwd
        state.transcript_path = transcript_path
        state.provider_name = provider_name
        self._save_state()

    def write_hookless_session_map(
        self,
        window_id: str,
        session_id: str,
        cwd: str,
        transcript_path: str,
        provider_name: str,
    ) -> None:
        """Write a synthetic entry to session_map.json for a hookless provider.

        Uses file locking consistent with hook.py. Safe to call from any
        thread (no asyncio handles touched).
        """
        import contextlib
        import fcntl

        map_file = config.session_map_file
        map_file.parent.mkdir(parents=True, exist_ok=True)
        # Foreign windows (emdash) are already fully qualified
        if is_foreign_window(window_id):
            window_key = window_id
        else:
            window_key = f"{config.tmux_session_name}:{window_id}"
        lock_path = map_file.with_suffix(".lock")
        try:
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                try:
                    session_map: dict[str, Any] = {}
                    if map_file.exists():
                        with contextlib.suppress(json.JSONDecodeError, OSError):
                            session_map = json.loads(map_file.read_text())
                    display_name = self.get_display_name(window_id)
                    session_map[window_key] = {
                        "session_id": session_id,
                        "cwd": cwd,
                        "window_name": display_name,
                        "transcript_path": transcript_path,
                        "provider_name": provider_name,
                    }
                    atomic_write_json(map_file, session_map)
                    logger.info(
                        "Registered hookless session: %s -> session_id=%s, cwd=%s",
                        window_key,
                        session_id,
                        cwd,
                    )
                finally:
                    fcntl.flock(lock_f, fcntl.LOCK_UN)
        except OSError:
            logger.exception("Failed to write session_map for hookless session")

    def get_session_id_for_window(self, window_id: str) -> str | None:
        """Look up session_id for a window from window_states."""
        state = self.window_states.get(window_id)
        return state.session_id if state and state.session_id else None

    # --- Window state management ---

    def get_window_state(self, window_id: str) -> WindowState:
        """Get or create window state."""
        if window_id not in self.window_states:
            self.window_states[window_id] = WindowState()
        return self.window_states[window_id]

    def clear_window_session(self, window_id: str) -> None:
        """Clear session association for a window (e.g., after /clear command)."""
        state = self.get_window_state(window_id)
        state.session_id = ""
        state.notification_mode = "all"
        self._save_state()
        logger.info("Cleared session for window_id %s", window_id)

    # --- Provider management ---

    def set_window_provider(
        self,
        window_id: str,
        provider_name: str,
        *,
        cwd: str | None = None,
    ) -> None:
        """Set the provider for a window. Empty string resets to config default.

        Always saves state unconditionally. When *cwd* is provided, persists it
        in the same write so provider/cwd updates stay atomic.
        """
        state = self.get_window_state(window_id)
        state.provider_name = provider_name
        if cwd:
            state.cwd = cwd
        self._save_state()

    def get_approval_mode(self, window_id: str) -> str:
        """Get approval mode for a window (default: 'normal')."""
        state = self.window_states.get(window_id)
        mode = state.approval_mode if state else DEFAULT_APPROVAL_MODE
        return mode if mode in APPROVAL_MODES else DEFAULT_APPROVAL_MODE

    def set_window_approval_mode(self, window_id: str, mode: str) -> None:
        """Set approval mode for a window."""
        normalized = mode.lower()
        if normalized not in APPROVAL_MODES:
            raise ValueError(f"Invalid approval mode: {mode!r}")
        state = self.get_window_state(window_id)
        state.approval_mode = normalized
        self._save_state()

    def get_window_for_chat_thread(self, chat_id: int, thread_id: int) -> str | None:
        """Resolve window_id for a specific Telegram chat/thread pair."""
        for user_id, bindings in self.thread_bindings.items():
            window_id = bindings.get(thread_id)
            if not window_id:
                continue
            key = f"{user_id}:{thread_id}"
            resolved_chat = self.group_chat_ids.get(key, user_id)
            if resolved_chat == chat_id:
                return window_id
        return None

    # --- Notification mode ---

    _NOTIFICATION_MODES = NOTIFICATION_MODES

    def get_notification_mode(self, window_id: str) -> str:
        """Get notification mode for a window (default: 'all')."""
        state = self.window_states.get(window_id)
        return state.notification_mode if state else "all"

    def set_notification_mode(self, window_id: str, mode: str) -> None:
        """Set notification mode for a window."""
        if mode not in self._NOTIFICATION_MODES:
            raise ValueError(f"Invalid notification mode: {mode!r}")
        state = self.get_window_state(window_id)
        if state.notification_mode != mode:
            state.notification_mode = mode
            self._save_state()

    def cycle_notification_mode(self, window_id: str) -> str:
        """Cycle notification mode: all → errors_only → muted → all. Returns new mode."""
        current = self.get_notification_mode(window_id)
        modes = self._NOTIFICATION_MODES
        idx = modes.index(current) if current in modes else 0
        new_mode = modes[(idx + 1) % len(modes)]
        self.set_notification_mode(window_id, new_mode)
        return new_mode

    # --- Batch mode ---

    def get_batch_mode(self, window_id: str) -> str:
        """Get batch mode for a window (default: 'batched')."""
        state = self.window_states.get(window_id)
        mode = state.batch_mode if state else DEFAULT_BATCH_MODE
        return mode if mode in BATCH_MODES else DEFAULT_BATCH_MODE

    def set_batch_mode(self, window_id: str, mode: str) -> None:
        """Set batch mode for a window."""
        if mode not in BATCH_MODES:
            raise ValueError(f"Invalid batch mode: {mode!r}")
        state = self.get_window_state(window_id)
        if state.batch_mode != mode:
            state.batch_mode = mode
            self._save_state()

    def cycle_batch_mode(self, window_id: str) -> str:
        """Toggle batch mode: batched ↔ verbose. Returns new mode."""
        current = self.get_batch_mode(window_id)
        new_mode = "verbose" if current == "batched" else "batched"
        self.set_batch_mode(window_id, new_mode)
        return new_mode

    # --- User directory favorites ---

    def get_user_starred(self, user_id: int) -> list[str]:
        """Get starred directories for a user."""
        return list(self.user_dir_favorites.get(user_id, {}).get("starred", []))

    def get_user_mru(self, user_id: int) -> list[str]:
        """Get MRU directories for a user."""
        return list(self.user_dir_favorites.get(user_id, {}).get("mru", []))

    def update_user_mru(self, user_id: int, path: str) -> None:
        """Insert path at front of MRU list, dedupe, cap at 5."""
        resolved = str(Path(path).resolve())
        favs = self.user_dir_favorites.setdefault(user_id, {})
        mru: list[str] = favs.get("mru", [])
        mru = [resolved] + [p for p in mru if p != resolved]
        favs["mru"] = mru[:5]
        self._save_state()

    def toggle_user_star(self, user_id: int, path: str) -> bool:
        """Toggle a directory in/out of starred list. Returns True if now starred."""
        resolved = str(Path(path).resolve())
        favs = self.user_dir_favorites.setdefault(user_id, {})
        starred: list[str] = favs.get("starred", [])
        if resolved in starred:
            starred.remove(resolved)
            now_starred = False
        else:
            starred.append(resolved)
            now_starred = True
        favs["starred"] = starred
        self._save_state()
        return now_starred

    def _build_session_file_path(self, session_id: str, cwd: str) -> Path | None:
        """Build the direct file path for a session from session_id and cwd."""
        if not session_id or not cwd:
            return None
        # Encode cwd: /data/code/ccbot -> -data-code-ccbot
        encoded_cwd = cwd.replace("/", "-")
        return config.claude_projects_path / encoded_cwd / f"{session_id}.jsonl"

    def _session_from_transcript_path(
        self,
        window_id: str,
        state: WindowState,
    ) -> ClaudeSession | None:
        """Build a lightweight session object from persisted transcript_path."""
        transcript = state.transcript_path
        if not transcript:
            return None
        file_path = Path(transcript)
        if not file_path.exists():
            return None
        summary = state.window_name or self.get_display_name(window_id) or "Untitled"
        return ClaudeSession(
            session_id=state.session_id,
            summary=summary,
            message_count=-1,  # unknown for non-JSONL transcript shortcuts
            file_path=str(file_path),
        )

    async def _get_session_direct(
        self, session_id: str, cwd: str, window_id: str = ""
    ) -> ClaudeSession | None:
        """Get a ClaudeSession directly from session_id and cwd (no scanning).

        Falls back to glob search when the direct path doesn't exist. If found
        via glob, attempts to recover the real cwd from the encoded directory
        name (only when ``window_id`` is provided and the decoded path is an
        existing absolute directory).
        """
        file_path = self._build_session_file_path(session_id, cwd)

        # Fallback: glob search if direct path doesn't exist
        if not file_path or not file_path.exists():
            pattern = f"*/{session_id}.jsonl"
            matches = list(config.claude_projects_path.glob(pattern))
            if matches:
                file_path = matches[0]
                logger.debug("Found session via glob: %s", file_path)
                # Try to recover real cwd so subsequent calls use direct path.
                # Encoding: /data/code/ccbot → -data-code-ccbot (replace "/" → "-")
                # Decoding is ambiguous: -home-user-my-app could be
                # /home/user/my-app or /home/user/my/app. We accept the
                # decoded path only if it's an existing absolute directory.
                encoded_dir = file_path.parent.name
                decoded_cwd = encoded_dir.replace("-", "/")
                if (
                    window_id
                    and decoded_cwd.startswith("/")
                    and Path(decoded_cwd).is_dir()
                ):
                    state = self.window_states.get(window_id)
                    if state and state.cwd != decoded_cwd:
                        logger.info(
                            "Glob fallback: updating cwd for window %s: %r -> %r",
                            window_id,
                            state.cwd,
                            decoded_cwd,
                        )
                        state.cwd = decoded_cwd
                        self._save_state()
            else:
                return None

        # Single pass: read file once, extract summary + count messages
        summary = ""
        last_user_msg = ""
        message_count = 0
        provider = get_provider_for_window(window_id)
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    message_count += 1
                    try:
                        data = json.loads(line)
                        # Check for summary
                        if data.get("type") == "summary":
                            s = data.get("summary", "")
                            if s:
                                summary = s
                        # Track last user message as fallback
                        elif provider.is_user_transcript_entry(data):
                            parsed = provider.parse_history_entry(data)
                            if parsed and parsed.text.strip():
                                last_user_msg = parsed.text.strip()
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return None

        if not summary:
            summary = last_user_msg[:50] if last_user_msg else "Untitled"

        return ClaudeSession(
            session_id=session_id,
            summary=summary,
            message_count=message_count,
            file_path=str(file_path),
        )

    # --- Window → Session resolution ---

    async def resolve_session_for_window(self, window_id: str) -> ClaudeSession | None:
        """Resolve a tmux window to the best matching Claude session.

        Uses persisted session_id + cwd to construct file path directly.
        Returns None if no session is associated with this window.
        """
        state = self.get_window_state(window_id)

        if not state.session_id or not state.cwd:
            return None

        # Hookless providers persist direct transcript paths outside Claude's
        # projects dir. Prefer that path first to avoid false "missing file"
        # clears when session_id/cwd don't map to Claude JSONL layout.
        direct = self._session_from_transcript_path(window_id, state)
        if direct:
            return direct

        session = await self._get_session_direct(state.session_id, state.cwd, window_id)
        if session:
            return session

        provider = get_provider_for_window(window_id)
        if not provider.capabilities.supports_hook:
            logger.debug(
                "Hookless session unresolved for window_id %s "
                "(sid=%s, transcript_path=%s); keeping state",
                window_id,
                state.session_id,
                state.transcript_path,
            )
            return None

        # File no longer exists, clear state
        logger.debug(
            "Session file no longer exists for window_id %s (sid=%s, cwd=%s)",
            window_id,
            state.session_id,
            state.cwd,
        )
        state.session_id = ""
        state.cwd = ""
        self._save_state()
        return None

    # --- User window offset management ---

    def get_user_window_offset(self, user_id: int, window_id: str) -> int | None:
        """Get the user's last read offset for a window.

        Returns None if no offset has been recorded (first time).
        """
        user_offsets = self.user_window_offsets.get(user_id)
        if user_offsets is None:
            return None
        return user_offsets.get(window_id)

    def update_user_window_offset(
        self, user_id: int, window_id: str, offset: int
    ) -> None:
        """Update the user's last read offset for a window."""
        if user_id not in self.user_window_offsets:
            self.user_window_offsets[user_id] = {}
        self.user_window_offsets[user_id][window_id] = offset
        self._save_state()

    # --- Thread binding management ---

    def bind_thread(
        self, user_id: int, thread_id: int, window_id: str, window_name: str = ""
    ) -> None:
        """Bind a Telegram topic thread to a tmux window.

        Enforces 1 topic = 1 window: if another thread is already bound to
        the same window_id, that stale binding is removed first.

        Args:
            user_id: Telegram user ID
            thread_id: Telegram topic thread ID
            window_id: Tmux window ID (e.g. '@0')
            window_name: Display name for the window (optional)
        """
        if user_id not in self.thread_bindings:
            self.thread_bindings[user_id] = {}

        # Enforce 1:1 — unbind any OTHER thread pointing to this window
        stale = [
            tid
            for tid, wid in self.thread_bindings[user_id].items()
            if wid == window_id and tid != thread_id
        ]
        for tid in stale:
            del self.thread_bindings[user_id][tid]
            logger.info(
                "Evicted stale binding: thread %d -> window_id %s (replaced by thread %d)",
                tid,
                window_id,
                thread_id,
            )

        self.thread_bindings[user_id][thread_id] = window_id
        self._window_to_thread[(user_id, window_id)] = thread_id
        if window_name:
            self.window_display_names[window_id] = window_name
        self._save_state()
        display = window_name or self.get_display_name(window_id)
        logger.info(
            "Bound thread %d -> window_id %s (%s) for user %d",
            thread_id,
            window_id,
            display,
            user_id,
        )

    def unbind_thread(self, user_id: int, thread_id: int) -> str | None:
        """Remove a thread binding. Returns the previously bound window_id, or None."""
        bindings = self.thread_bindings.get(user_id)
        if not bindings or thread_id not in bindings:
            return None
        window_id = bindings.pop(thread_id)
        self._window_to_thread.pop((user_id, window_id), None)
        if not bindings:
            del self.thread_bindings[user_id]
        logger.info(
            "Unbound thread %d (was %s) for user %d",
            thread_id,
            window_id,
            user_id,
        )

        # Clean up group_chat_id for the unbound thread
        chat_key = f"{user_id}:{thread_id}"
        self.group_chat_ids.pop(chat_key, None)

        # Remove display name if no other thread still references this window
        still_bound = any(
            wid == window_id
            for user_bindings in self.thread_bindings.values()
            for wid in user_bindings.values()
        )
        if not still_bound and window_id not in self.window_states:
            self.window_display_names.pop(window_id, None)

        self._save_state()
        return window_id

    def get_window_for_thread(self, user_id: int, thread_id: int) -> str | None:
        """Look up the window_id bound to a thread."""
        bindings = self.thread_bindings.get(user_id)
        if not bindings:
            return None
        return bindings.get(thread_id)

    def get_thread_for_window(self, user_id: int, window_id: str) -> int | None:
        """Reverse lookup: get thread_id for a window (O(1) via reverse index)."""
        return self._window_to_thread.get((user_id, window_id))

    def get_all_thread_windows(self, user_id: int) -> dict[int, str]:
        """Get all thread bindings for a user."""
        return dict(self.thread_bindings.get(user_id, {}))

    def resolve_window_for_thread(
        self,
        user_id: int,
        thread_id: int | None,
    ) -> str | None:
        """Resolve the tmux window_id for a user's thread.

        Returns None if thread_id is None or the thread is not bound.
        """
        if thread_id is None:
            return None
        return self.get_window_for_thread(user_id, thread_id)

    def iter_thread_bindings(self) -> Iterator[tuple[int, int, str]]:
        """Iterate all thread bindings as (user_id, thread_id, window_id).

        Provides encapsulated access to thread_bindings without exposing
        the internal data structure directly.
        """
        for user_id, bindings in self.thread_bindings.items():
            for thread_id, window_id in bindings.items():
                yield user_id, thread_id, window_id

    def find_users_for_session(
        self,
        session_id: str,
    ) -> list[tuple[int, str, int]]:
        """Find all users whose thread-bound window maps to the given session_id.

        Uses in-memory window_states for O(bindings) lookup with zero I/O.
        Returns list of (user_id, window_id, thread_id) tuples.
        """
        result: list[tuple[int, str, int]] = []
        for user_id, thread_id, window_id in self.iter_thread_bindings():
            state = self.window_states.get(window_id)
            if state and state.session_id == session_id:
                result.append((user_id, window_id, thread_id))
        return result

    # --- Group chat ID management ---

    def set_group_chat_id(self, user_id: int, thread_id: int, chat_id: int) -> None:
        """Store the group chat ID for a user's thread (for forum topic message routing).

        Uses composite key "user_id:thread_id" to support multiple groups per user.
        """
        key = f"{user_id}:{thread_id}"
        if self.group_chat_ids.get(key) != chat_id:
            self.group_chat_ids[key] = chat_id
            self._save_state()
            logger.info(
                "Stored group chat_id %d for user %d, thread %d",
                chat_id,
                user_id,
                thread_id,
            )

    def resolve_chat_id(self, user_id: int, thread_id: int | None = None) -> int:
        """Resolve the chat_id for sending messages.

        In forum topics (thread_id is set), returns the stored group chat_id
        for that specific thread (user_id:thread_id).
        Falls back to user_id for direct messages or if no group_id stored.
        """
        if thread_id is not None:
            key = f"{user_id}:{thread_id}"
            group_id = self.group_chat_ids.get(key)
            if group_id is not None:
                return group_id
        return user_id

    # --- Tmux helpers ---

    async def send_to_window(self, window_id: str, text: str) -> tuple[bool, str]:
        """Send text to a tmux window by ID."""
        display = self.get_display_name(window_id)
        logger.debug(
            "send_to_window: window_id=%s (%s), text_len=%d",
            window_id,
            display,
            len(text),
        )
        window = await tmux_manager.find_window_by_id(window_id)
        if not window:
            return False, "Window not found (may have been closed)"
        success = await tmux_manager.send_keys(window.window_id, text)
        if success:
            return True, f"Sent to {display}"
        return False, "Failed to send keys"

    # --- Message history ---

    async def get_recent_messages(
        self,
        window_id: str,
        *,
        start_byte: int = 0,
        end_byte: int | None = None,
    ) -> tuple[list[dict], int]:
        """Get user/assistant messages for a window's session.

        Resolves window → session, then reads the JSONL.
        Supports byte range filtering via start_byte/end_byte.
        Returns (messages, total_count).
        """
        session = await self.resolve_session_for_window(window_id)
        if not session or not session.file_path:
            return [], 0

        file_path = Path(session.file_path)
        if not file_path.exists():
            return [], 0

        # Read JSONL entries (optionally filtered by byte range)
        provider = get_provider_for_window(window_id)
        entries: list[dict[str, Any]] = []
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                if start_byte > 0:
                    await f.seek(start_byte)

                while True:
                    # Check byte limit before reading
                    if end_byte is not None:
                        current_pos = await f.tell()
                        if current_pos >= end_byte:
                            break

                    line = await f.readline()
                    if not line:
                        break

                    data = provider.parse_transcript_line(line)
                    if data:
                        entries.append(data)
        except OSError:
            logger.exception("Error reading session file %s", file_path)
            return [], 0

        agent_messages, _ = provider.parse_transcript_entries(entries, {})
        all_messages = [
            {
                "role": e.role,
                "text": e.text,
                "content_type": e.content_type,
                "timestamp": e.timestamp,
            }
            for e in agent_messages
        ]

        return all_messages, len(all_messages)


session_manager = SessionManager()
