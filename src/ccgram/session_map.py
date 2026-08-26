"""Session map I/O — reads and writes session_map.json.

Owns all logic for synchronising window states against the session_map.json
file written by the Claude Code hook. Extracted from SessionManager so that
session_map concerns live in one place without pulling in the full
SessionManager stack.

The ``schedule_save`` callback is injected via the constructor — the
sync cannot be built without an explicit callback.

Module-level access: ``get_session_map_sync()`` returns the
SessionManager-owned instance (raises RuntimeError until SessionManager
has constructed the sync). The legacy module attribute
``session_map_sync`` is a thin proxy that delegates to the same instance
for backward compat.

Key class: SessionMapSync.
Free functions: parse_session_map.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shutil
import time
import structlog
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import aiofiles

from .config import config
from .herdr_targets import is_herdr_session_target
from .hooks.state_files import StateFileValidationError, parse_session_map_entry
from .utils import atomic_write_json, log_throttle_reset, log_throttled
from .window_resolver import is_window_id, session_map_prefix_for

logger = structlog.get_logger()

_DEFAULT_PRIMARY_SESSION_GRACE_SEC = 60.0

# "A creation flow currently owns this window" — wired at startup to the
# topic-creation flow's pending set, which lives with the flow that owns it
# (a core → handlers import would invert the dependency).
_in_flight_window_predicate: Callable[[str], bool] | None = None


def register_in_flight_window_predicate(predicate: Callable[[str], bool]) -> None:
    """Wire the in-flight-creation check (called once at startup).

    Raises RuntimeError if called more than once — wiring happens exactly
    once at startup; double registration is a programming error.
    """
    global _in_flight_window_predicate
    if _in_flight_window_predicate is not None:
        raise RuntimeError("register_in_flight_window_predicate already registered")
    _in_flight_window_predicate = predicate


def _reset_in_flight_window_predicate_for_testing() -> None:
    """Restore the unwired default — only for tests."""
    global _in_flight_window_predicate
    _in_flight_window_predicate = None


def _creation_in_flight(window_id: str) -> bool:
    """Whether a creation flow currently owns this window id.

    Unwired (``doctor``, ``status``, unit tests) means nothing is being
    created, so nothing is protected.
    """
    if _in_flight_window_predicate is None:
        return False
    return _in_flight_window_predicate(window_id)


def _primary_session_grace_sec() -> float:
    raw = os.getenv("CCGRAM_NESTED_SESSION_GRACE_SEC")
    if raw is None:
        return _DEFAULT_PRIMARY_SESSION_GRACE_SEC
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "CCGRAM_NESTED_SESSION_GRACE_SEC must be a number, got %r; using default %.1f",
            raw,
            _DEFAULT_PRIMARY_SESSION_GRACE_SEC,
        )
        return _DEFAULT_PRIMARY_SESSION_GRACE_SEC


async def read_session_map_raw() -> dict[str, Any] | None:
    """Read and parse session_map.json once.

    Returns the parsed dict, ``{}`` if the file does not exist, or ``None``
    if read/parse failed.  Caller passes the result to both
    ``SessionMapSync.load_session_map`` and ``parse_session_map`` to avoid
    re-reading the file twice per poll cycle.
    """
    if not config.session_map_file.exists():
        return {}
    try:
        async with aiofiles.open(config.session_map_file, "r") as f:
            content = await f.read()
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError, OSError:
        return None


def session_map_prefix() -> str:
    """Return the session_map key prefix for the active multiplexer backend.

    The hook encodes the backend into each key's prefix: tmux keys are
    ``<tmux_session_name>:<@id>`` (the live tmux session name), herdr keys are
    ``herdr:<opaque-session-target>`` (the backend name plus a durable opaque
    target — see ``multiplexer.self_identify``). Raw Herdr pane/tab locators
    are never valid persisted identities.
    Readers mirror that here so they match the writer regardless of the active
    backend; the tmux branch is byte-identical to the previous hard-coded
    ``f"{config.tmux_session_name}:"``.
    """
    return session_map_prefix_for(config.multiplexer_name, config.tmux_session_name)


def is_backend_window_id(window_id: str) -> bool:
    """Validate a prefix-stripped session_map window id for the active backend.

    tmux requires the ``@N`` form so legacy window-name-keyed entries are still
    detected and purged as old format. Herdr persists only opaque durable
    ``herdr-session-v1-<digest>`` targets; a pane or tab locator is never a
    valid session-map identity.
    """
    if config.multiplexer_name == "tmux":
        return is_window_id(window_id)
    if config.multiplexer_name == "herdr":
        return is_herdr_session_target(window_id)
    return bool(window_id)


def _transcript_mtime(transcript_path: str) -> float | None:
    if not transcript_path:
        return None
    try:
        return Path(transcript_path).stat().st_mtime
    except OSError:
        return None


def _transcript_is_fresh(transcript_path: str, *, now: float | None = None) -> bool:
    mtime = _transcript_mtime(transcript_path)
    if mtime is None:
        return False
    reference = time.time() if now is None else now
    return reference - mtime < _primary_session_grace_sec()


def _prefer_existing_primary(
    window_id: str,
    incoming: dict[str, Any],
) -> dict[str, str] | None:
    # Lazy: session.py imports both session_map and window_state_store at
    # top; hoisting forms session → session_map → window_state_store →
    # session cycle.  Lazy import also guarantees the store has been
    # wired via install_window_store before access.
    # Lazy: window_state_store / thread_router proxies wired by SessionManager constructor
    from .window_state_store import is_window_store_wired, window_store

    if not is_window_store_wired():
        return None
    state = window_store.window_states.get(window_id)
    if not state or not state.session_id:
        return None

    incoming_sid = incoming.get("session_id", "")
    if not incoming_sid or incoming_sid == state.session_id:
        return None

    existing_mtime = _transcript_mtime(state.transcript_path)
    incoming_mtime = _transcript_mtime(incoming.get("transcript_path", ""))
    existing_is_fresh = _transcript_is_fresh(state.transcript_path)
    existing_is_newer = existing_mtime is not None and (
        incoming_mtime is None or existing_mtime >= incoming_mtime
    )
    if not existing_is_fresh and not existing_is_newer:
        return None

    log_throttled(
        logger,
        f"preserve-primary:{window_id}",
        "Preserving primary session for window_id %s: existing %s, incoming %s treated as nested",
        window_id,
        state.session_id,
        incoming_sid,
    )
    return {
        "session_id": state.session_id,
        "cwd": state.cwd,
        "window_name": incoming.get("window_name", "") or state.window_name,
        "transcript_path": state.transcript_path,
        "provider_name": incoming.get("provider_name", "") or state.provider_name,
    }


def effective_session_map_info(
    window_id: str,
    info: dict[str, Any],
) -> dict[str, str]:
    preferred = _prefer_existing_primary(window_id, info)
    if preferred is not None:
        return preferred
    return {
        "session_id": info.get("session_id", ""),
        "cwd": info.get("cwd", ""),
        "window_name": info.get("window_name", ""),
        "transcript_path": info.get("transcript_path", ""),
        "provider_name": info.get("provider_name", ""),
    }


def parse_session_map(raw: dict[str, Any], prefix: str) -> dict[str, dict[str, str]]:
    """Parse session_map.json entries matching a backend prefix.

    Returns {window_id: {"session_id": ..., "cwd": ...}} for matching entries,
    where window_id is the bare id after stripping the prefix — e.g. ``"@12"``
    for tmux (``"ccgram:@12"``) or a guarded opaque target for herdr.

    Safe to call from a clean interpreter (no SessionManager wired): the
    nested-session preference logic in ``_prefer_existing_primary`` short-
    circuits to ``None`` when the window store is unwired, so the result
    reflects the raw session_map rather than a wiring crash. When wired,
    the result also incorporates the in-memory primary-session preference.
    """
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, str]] = {}
    for key, info in raw.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        window_name = key[len(prefix) :]
        # A Herdr prefix alone is not authority: only an exact versioned
        # guarded-session target is accepted. Raw tab/pane IDs are legacy
        # migration records and must not reach monitor lifecycle processing.
        if (
            prefix == "herdr:" and not is_herdr_session_target(window_name)
        ) or not isinstance(info, dict):
            continue
        try:
            # parse_session_map_entry is used as a validation gate: it raises
            # StateFileValidationError for malformed entries (missing session_id,
            # unsupported schema_version) so they are skipped. The returned
            # SessionMapEntry is intentionally discarded; the raw dict is passed
            # downstream because effective_session_map_info/_prefer_existing_primary
            # merge it with in-memory window state and return a computed dict,
            # not a passthrough — rewiring to SessionMapEntry would add indirection
            # across two divergent call paths for no behavioral gain.
            parse_session_map_entry(info)
        except StateFileValidationError as exc:
            logger.debug("Skipping invalid session_map entry %s: %s", key, exc)
            continue
        effective = effective_session_map_info(window_name, info)
        if effective["session_id"]:
            result[window_name] = effective
    return result


def _read_session_map_for_pruning() -> dict[str, Any] | None:
    if not config.session_map_file.exists():
        return None
    try:
        raw = json.loads(config.session_map_file.read_text())
    except (json.JSONDecodeError, OSError):  # fmt: skip
        return None
    return raw if isinstance(raw, dict) else None


def _dead_session_map_entries(
    raw: dict[str, Any], live_window_ids: set[str]
) -> list[tuple[str, str]]:
    prefix = session_map_prefix()
    return [
        (key, window_id)
        for key in raw
        if key.startswith(prefix)
        and is_backend_window_id(window_id := key[len(prefix) :])
        and window_id not in live_window_ids
    ]


def _remove_dead_session_map_entries(
    raw: dict[str, Any], dead_entries: list[tuple[str, str]], window_store: Any
) -> bool:
    # Lazy: window_state_store / thread_router proxies wired by SessionManager constructor
    from .thread_router import thread_router

    # A window dying is exactly when the recovery banner goes up, and its
    # Fresh/Continue/Resume buttons read the directory back out of the window
    # state. Dropping that state here answered every button with "Directory no
    # longer exists" while the directory was sitting there (#176). The dead
    # entry still goes; the state a live topic still points at stays until the
    # topic unbinds and ``_remove_stale_window_states`` reclaims it.
    bound_wids = thread_router.all_bound_window_ids()
    changed_state = False
    for key, window_id in dead_entries:
        logger.info("Pruning dead session_map entry: %s (window %s)", key, window_id)
        del raw[key]
        log_throttle_reset(f"preserve-primary:{window_id}")
        if window_id not in bound_wids and window_store.has_window(window_id):
            window_store.remove_window(window_id)
            changed_state = True
    return changed_state


class SessionMapSync:
    """Session map I/O and window-state synchronisation.

    Reads and writes session_map.json, syncing window states from hook-written
    entries. Persistence of window_states is delegated: the ``schedule_save``
    callback (provided by SessionManager) triggers a debounced save after
    mutations.

    Depends on ``window_store`` and ``thread_router`` singletons for state access.
    """

    def __init__(self, *, schedule_save: Callable[[], None]) -> None:
        self._schedule_save: Callable[[], None] = schedule_save

    # ------------------------------------------------------------------
    # Public: async read/sync methods
    # ------------------------------------------------------------------

    async def load_session_map(self, raw: dict[str, Any] | None = None) -> None:
        """Read session_map.json and update window_states with new session associations.

        Keys in session_map are formatted as "tmux_session:window_id" for tmux
        (e.g. "ccgram:@12") or "herdr:<opaque-session-target>" for herdr.
        Only native entries (matching our tmux_session_name or the "herdr:" prefix) are processed.
        Also cleans up window_states entries not in current session_map.
        Updates window_display_names from the "window_name" field in values.

        If ``raw`` is provided (e.g., by ``read_session_map_raw``), use it
        directly to avoid a redundant file read.  Otherwise read the file.
        """
        if raw is None:
            raw = await read_session_map_raw()
        if raw is None or not isinstance(raw, dict):
            return
        if not raw:
            return
        session_map = raw

        prefix = session_map_prefix()
        valid_wids, old_format_sids, old_format_keys, changed = (
            self._process_session_map_entries(session_map, prefix)
        )
        changed |= self._remove_stale_window_states(valid_wids, old_format_sids)
        self._purge_old_format_keys(session_map, old_format_keys)

        if changed:
            self._schedule_save()

    def _process_session_map_entries(
        self,
        session_map: dict[str, Any],
        prefix: str,
    ) -> tuple[set[str], set[str], list[str], bool]:
        """Iterate session_map entries and sync window states.

        Returns (valid_wids, old_format_sids, old_format_keys, changed).
        """
        valid_wids: set[str] = set()
        old_format_sids: set[str] = set()
        old_format_keys: list[str] = []
        changed = False

        for key, info in session_map.items():
            if not isinstance(info, dict):
                continue
            if not key.startswith(prefix):
                continue
            window_id = key[len(prefix) :]
            if not is_backend_window_id(window_id):
                sid = info.get("session_id", "")
                if sid:
                    old_format_sids.add(sid)
                old_format_keys.append(key)
                continue
            # Protect this window_id from stale-state deletion regardless of
            # whether the entry passes schema validation.  Validation failures
            # are forward-compat / corruption guards; they must not delete
            # in-memory state for a window that is present in the file.
            valid_wids.add(window_id)
            try:
                # Validation gate — see the identical pattern in parse_session_map()
                # for the full rationale. Returned SessionMapEntry is discarded;
                # raw dict is consumed by _sync_window_from_session_map below.
                parse_session_map_entry(info)
            except StateFileValidationError as exc:
                logger.debug("Skipping invalid session_map entry %s: %s", key, exc)
                continue
            if self._sync_window_from_session_map(window_id, info):
                changed = True

        return valid_wids, old_format_sids, old_format_keys, changed

    def _remove_stale_window_states(
        self,
        valid_wids: set[str],
        old_format_sids: set[str],
    ) -> bool:
        """Remove window_states not in valid_wids, not bound, and not old-format.

        Returns True if any states were removed.
        """
        # Lazy: same session ↔ session_map ↔ stores cycle as
        # _prefer_existing_primary; both stores must be installed.
        # Lazy: window_state_store / thread_router proxies wired by SessionManager constructor
        from .thread_router import thread_router

        # Lazy: window_state_store / thread_router proxies wired by SessionManager constructor
        from .window_state_store import window_store

        # Must cover chat-scoped bindings too: ``set_group_chat_id`` moves a
        # binding out of ``thread_bindings`` into ``chat_thread_bindings``, so
        # in a forum deployment the legacy dict is empty and reading it alone
        # leaves this guard dead — sweeping the state of every bound window
        # whose provider has no hook to keep it in the session map.
        bound_wids = {wid for wid in thread_router.all_bound_window_ids() if wid}
        stale_wids = [
            w
            for w in window_store.iter_window_ids()
            if (
                w
                and w not in valid_wids
                and w not in bound_wids
                and window_store.get_session_id_for_window(w) not in old_format_sids
                and not window_store.is_archived_legacy_herdr(w)
                # A window being created is neither in the session map (its
                # hook has not fired) nor bound (the flow binds afterwards),
                # so it looks exactly like a stale one. Dropping it discards
                # the cwd, provider, approval mode and origin the flow just
                # wrote — the window then comes back re-derived and, having
                # lost its ccgram origin, outside ccgram's lifecycle.
                and not _creation_in_flight(w)
            )
        ]
        for wid in stale_wids:
            logger.info("Removing stale window_state: %s", wid)
            window_store.remove_window(wid)
        return bool(stale_wids)

    def _purge_old_format_keys(
        self,
        session_map: dict[str, Any],
        old_format_keys: list[str],
    ) -> None:
        """Remove old-format (window-name-keyed) entries from session_map.json."""
        if not old_format_keys:
            return
        for key in old_format_keys:
            logger.info("Removing old-format session_map key: %s", key)
            del session_map[key]
        atomic_write_json(config.session_map_file, session_map)

    async def wait_for_session_map_entry(
        self,
        window_id: str,
        timeout: float = 5.0,
        interval: float = 0.5,
        *,
        resolve_window_id: Callable[[str], str] | None = None,
    ) -> bool:
        """Poll session_map.json until an entry for window_id appears.

        ``resolve_window_id`` is re-applied on every poll. A backend whose
        window identity firms up over time (Herdr mints the durable one once
        the agent session is published) can supersede the id the caller was
        handed *while this wait runs*, and the hook then writes its entry
        under the new one. Re-resolving each pass means the wait watches the
        key the window actually answers to instead of timing out on an id
        nothing will ever write again. Callers on stable-id backends omit it.

        Returns True if the entry was found within timeout, False otherwise.
        """
        logger.debug(
            "Waiting for session_map entry: window_id=%s, timeout=%.1f",
            window_id,
            timeout,
        )
        current_id = window_id
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if resolve_window_id is not None:
                current_id = resolve_window_id(window_id)
            key = f"{session_map_prefix()}{current_id}"
            try:
                if config.session_map_file.exists():
                    async with aiofiles.open(config.session_map_file, "r") as f:
                        content = await f.read()
                    session_map = json.loads(content)
                    if not isinstance(session_map, dict):
                        raise ValueError("session_map root is not an object")
                    info = session_map.get(key)
                    if isinstance(info, dict):
                        parse_session_map_entry(info)
                        logger.debug(
                            "session_map entry found for window_id %s", current_id
                        )
                        await self.load_session_map(session_map)
                        return True
            except StateFileValidationError, json.JSONDecodeError, OSError, ValueError:
                pass
            await asyncio.sleep(interval)
        logger.warning(
            "Timed out waiting for session_map entry: window_id=%s", current_id
        )
        return False

    # ------------------------------------------------------------------
    # Public: sync read/write methods
    # ------------------------------------------------------------------

    def prune_session_map(self, live_window_ids: set[str]) -> None:
        """Remove stale tmux session-map entries, preserving Herdr targets."""
        # A Herdr target is a durable session digest, not a current locator.
        # Its absence from one ``agent.list`` snapshot can be a move, restart,
        # reconnect, or event-loss gap; only a guarded action may classify it
        # unresolved/ambiguous. Never prune it from hook persistence.
        if config.multiplexer_name == "herdr":
            return

        map_file = config.session_map_file
        if not map_file.exists():
            return
        lock_path = map_file.with_suffix(".lock")
        try:
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                try:
                    # Re-read only after the hook-compatible lock is held so a
                    # concurrent hook write cannot be lost between prune/read/write.
                    raw = _read_session_map_for_pruning()
                    if raw is None:
                        return
                    dead_entries = _dead_session_map_entries(raw, live_window_ids)
                    if not dead_entries:
                        return
                    # Lazy: same cycle + wiring contract as _prefer_existing_primary.
                    from .window_state_store import window_store

                    changed_state = _remove_dead_session_map_entries(
                        raw, dead_entries, window_store
                    )
                    atomic_write_json(map_file, raw)
                    if changed_state:
                        self._schedule_save()
                finally:
                    fcntl.flock(lock_f, fcntl.LOCK_UN)
        except OSError as exc:
            logger.warning("Failed to lock session_map for pruning: %s", exc)

    def rename_session_map_entry(self, alias_window_id: str, window_id: str) -> bool:
        """Re-key a hook-written entry from a superseded window id onto the live one.

        Pairs with ``window_resolver.migrate_window_aliases``: that moves the
        in-memory state, this moves the file the hook wrote it to. Without the
        file side, ``load_session_map`` recreates the alias window state on the
        next cycle and the reconciliation never sticks.

        Returns True when the file changed. When both keys exist the live entry
        wins and the alias is simply dropped — the hook has already written a
        fresher entry under the current identity.
        """
        map_file = config.session_map_file
        if alias_window_id == window_id or not map_file.exists():
            return False
        prefix = session_map_prefix()
        alias_key, live_key = f"{prefix}{alias_window_id}", f"{prefix}{window_id}"
        lock_path = map_file.with_suffix(".lock")
        try:
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                try:
                    # Re-read under the hook-compatible lock so a concurrent
                    # hook write cannot be lost between read and write.
                    raw = _read_session_map_for_pruning()
                    if raw is None or alias_key not in raw:
                        return False
                    entry = raw.pop(alias_key)
                    if live_key not in raw:
                        raw[live_key] = entry
                    atomic_write_json(map_file, raw)
                    logger.info(
                        "Re-keyed session_map entry %s -> %s", alias_key, live_key
                    )
                    return True
                finally:
                    fcntl.flock(lock_f, fcntl.LOCK_UN)
        except OSError as exc:
            logger.warning("Failed to lock session_map for re-keying: %s", exc)
            return False

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
        because _schedule_save() touches asyncio timer handles.

        Pair with write_hookless_session_map() for the file-locked
        session_map.json write, which is safe to call from any thread.
        """
        # Lazy: same cycle + wiring contract as _prefer_existing_primary.
        from .window_state_store import window_store

        state = window_store.get_window_state(window_id)
        state.session_id = session_id
        state.cwd = cwd
        state.transcript_path = transcript_path
        state.provider_name = provider_name
        self._schedule_save()

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
        # Lazy: same cycle + wiring contract as _prefer_existing_primary.
        from .thread_router import thread_router

        map_file = config.session_map_file
        map_file.parent.mkdir(parents=True, exist_ok=True)
        window_key = f"{session_map_prefix()}{window_id}"
        lock_path = map_file.with_suffix(".lock")
        try:
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                try:
                    session_map: dict[str, Any] = {}
                    if map_file.exists():
                        try:
                            parsed = json.loads(map_file.read_text())
                            if isinstance(parsed, dict):
                                session_map = parsed
                        except json.JSONDecodeError:
                            backup = map_file.with_suffix(".json.corrupt")
                            try:
                                shutil.copy2(map_file, backup)
                                logger.warning(
                                    "Corrupted session_map.json backed up to %s",
                                    backup,
                                )
                            except OSError:
                                logger.warning(
                                    "Corrupted session_map.json (backup failed)"
                                )
                        except OSError:
                            logger.warning(
                                "Failed to read session_map.json for hookless write"
                            )
                    display_name = thread_router.get_display_name(window_id)
                    # Lazy: same session_map ↔ stores cycle as _prefer_existing_primary
                    from .hooks.state_files import serialize_session_map_entry

                    session_map[window_key] = serialize_session_map_entry(
                        session_id, cwd, display_name, transcript_path, provider_name
                    )
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

    async def session_map_entry_may_exist(self, window_id: str) -> bool:
        """Return whether the hook may have an entry for ``window_id``.

        Deliberately answers True when the file cannot be read: callers use
        this to decide whether it is safe to write state that would clear a
        live entry, and an unreadable map is "unknown", not "absent". Guessing
        absent there destroys a running session's tracking; guessing present
        only defers a heal to the next tick.
        """
        raw = await read_session_map_raw()
        if raw is None:
            return True
        info = raw.get(f"{session_map_prefix()}{window_id}")
        if info is None:
            return False
        try:
            # The premise is that the monitor will rebuild state from this
            # entry, which only holds for entries load_session_map accepts. One
            # it rejects (no session_id, or a schema_version from a newer build
            # after a downgrade) never becomes state, so treating the bare key
            # as proof of tracking would wedge the window unhealed forever.
            parse_session_map_entry(info)
        except StateFileValidationError:
            return False
        return True

    def clear_session_map_entry(self, window_id: str) -> None:
        """Remove a window's entry from session_map.json if present."""
        if not config.session_map_file.exists():
            return
        lock_path = config.session_map_file.with_suffix(".lock")
        try:
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                try:
                    raw = json.loads(config.session_map_file.read_text())
                    key = f"{session_map_prefix()}{window_id}"
                    if key in raw:
                        del raw[key]
                        atomic_write_json(config.session_map_file, raw)
                        logger.debug("Cleared session_map entry for %s", window_id)
                except (json.JSONDecodeError, OSError):  # fmt: skip
                    return
                finally:
                    fcntl.flock(lock_f, fcntl.LOCK_UN)
        except OSError as exc:
            # Lock failure means the entry clear was lost — surface it.
            logger.warning(
                "Failed to lock session_map for clearing %s: %s", window_id, exc
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_window_from_session_map(
        self,
        window_id: str,
        info: dict[str, Any],
    ) -> bool:
        """Sync a single window's state from session_map entry.

        Returns True if any state was changed.
        """
        # Lazy: same cycle + wiring contract as _prefer_existing_primary.
        from .thread_router import thread_router

        # Lazy: window_state_store / thread_router proxies wired by SessionManager constructor
        from .window_state_store import window_store

        effective = effective_session_map_info(window_id, info)
        new_sid = effective["session_id"]
        if not new_sid:
            return False
        new_cwd = effective["cwd"]
        new_wname = effective["window_name"]
        new_transcript = effective["transcript_path"]
        changed = False

        state = window_store.get_window_state(window_id)
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
        new_provider = effective["provider_name"].lower()
        # Cross-check provider claim against the transcript path. session_map.json
        # may carry a stale `provider_name` from a previous run in the same tmux
        # window (e.g. codex once owned @9729, then claude took over). The
        # transcript path is observed reality and wins; without this guard the
        # transcript_reader spams "Provider mismatch" warnings every poll.
        path_for_inference = new_transcript or state.transcript_path
        if new_provider and path_for_inference:
            # Lazy: providers import pulls the agent provider registry which
            # imports the shell provider's prompt-marker machinery; defer.
            from .providers import detect_provider_from_transcript_path

            inferred = detect_provider_from_transcript_path(path_for_inference)
            if inferred and inferred != new_provider:
                new_provider = inferred
        if new_provider and state.provider_name != new_provider:
            # Log only on actual mutation so a persistent stale claim in
            # session_map.json doesn't spam the log every poll cycle once the
            # in-memory state has already been corrected.
            logger.warning(
                "Corrected provider for %s: state=%s -> %s "
                "(session_map claimed %s; transcript_path=%s)",
                window_id,
                state.provider_name,
                new_provider,
                effective["provider_name"].lower(),
                path_for_inference,
            )
            state.provider_name = new_provider
            changed = True
        if (
            new_wname
            and thread_router.get_display_name(window_id) == window_id
            and not state.window_name
        ):
            state.window_name = new_wname
            thread_router.set_display_name(window_id, new_wname)
            changed = True
        return changed


_active_sync: SessionMapSync | None = None


def get_session_map_sync() -> SessionMapSync:
    """Return the SessionManager-owned SessionMapSync.

    Raises:
        RuntimeError: when called before SessionManager has constructed
        and installed the sync.
    """
    if _active_sync is None:
        raise RuntimeError(
            "SessionMapSync not yet wired. "
            "Instantiate SessionManager() before accessing session_map_sync."
        )
    return _active_sync


def install_session_map_sync(sync: SessionMapSync) -> None:
    """Install the SessionManager-owned sync as the module-level singleton.

    Called once by ``SessionManager.__post_init__``. Replaces any
    previously installed instance (used by tests that build a fresh
    SessionManager).
    """
    global _active_sync
    _active_sync = sync


class _SessionMapSyncProxy:
    """Backward-compat module-level facade that resolves to the wired sync.

    All attribute access delegates to the SessionManager-owned
    ``SessionMapSync``. Raises ``RuntimeError`` if accessed before
    SessionManager has installed an instance.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(get_session_map_sync(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(get_session_map_sync(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(get_session_map_sync(), name)

    def __repr__(self) -> str:
        if _active_sync is None:
            return "<SessionMapSyncProxy unwired>"
        return f"<SessionMapSyncProxy → {_active_sync!r}>"


session_map_sync: SessionMapSync = cast("SessionMapSync", _SessionMapSyncProxy())
