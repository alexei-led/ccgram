"""Window ID resolution, format helpers, and startup migration.

Provides shared window ID helpers used across session, tmux_manager, and
handler modules (no intra-package imports — safe from circular dependencies):
  - is_window_id(): validate tmux window ID format (@0, @12).
  - resolve_stale_ids(): full startup recovery — remaps persisted window IDs
    against live tmux windows, handles old-format migration, prunes dead entries.
"""

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class LiveWindow:
    """Minimal representation of a live tmux window for resolution."""

    window_id: str
    window_name: str


def is_window_id(key: str) -> bool:
    """Check if a key looks like a tmux window ID (e.g. '@0', '@12')."""
    return key.startswith("@") and len(key) > 1 and key[1:].isdigit()


def session_map_prefix_for(mux_name: str, session_name: str) -> str:
    """Return the session_map key prefix for a given multiplexer backend.

    tmux keys are ``<tmux_session_name>:<window_id>`` (e.g. ``ccgram:@12``);
    non-tmux backends (herdr) key by backend name (e.g. ``herdr:w2:t1``).

    This is the pure, config-free version used by status_cmd and session_map.
    ``session_map.session_map_prefix()`` wraps this with ``config`` values.
    """
    if mux_name == "tmux":
        return f"{session_name}:"
    return f"{mux_name}:"


def _resolve_window_states(
    window_states: dict,
    window_display_names: dict,
    live_by_name: dict[str, str],
    live_ids: set[str],
) -> bool:
    """Re-resolve window_states dict in-place. Returns True if changed."""
    changed = False
    new_states: dict = {}
    for key, ws in window_states.items():
        if is_window_id(key):
            if key in live_ids:
                new_states[key] = ws
            else:
                display = window_display_names.get(
                    key, getattr(ws, "window_name", "") or key
                )
                new_id = live_by_name.get(display)
                if new_id:
                    logger.debug("Re-resolved stale window_id %s -> %s", key, new_id)
                    new_states[new_id] = ws
                    ws.window_name = display
                    window_display_names[new_id] = display
                    window_display_names.pop(key, None)
                    changed = True
                else:
                    # Keep dead window state — recovery needs cwd/provider
                    new_states[key] = ws
        else:
            new_id = live_by_name.get(key)
            if new_id:
                logger.debug("Migrating window_state key %s -> %s", key, new_id)
                ws.window_name = key
                new_states[new_id] = ws
                window_display_names[new_id] = key
                changed = True
            else:
                logger.debug("Dropping old-format window_state: %s", key)
                changed = True
    window_states.clear()
    window_states.update(new_states)
    return changed


def _resolve_thread_bindings(
    thread_bindings: dict,
    window_display_names: dict,
    live_by_name: dict[str, str],
    live_ids: set[str],
) -> bool:
    """Re-resolve thread_bindings dict in-place. Returns True if changed."""
    changed = False
    for uid, bindings in thread_bindings.items():
        new_bindings: dict[int, str] = {}
        for tid, val in bindings.items():
            if is_window_id(val):
                if val in live_ids:
                    new_bindings[tid] = val
                elif new_id := live_by_name.get(window_display_names.get(val, val)):
                    logger.debug("Re-resolved thread binding %s -> %s", val, new_id)
                    new_bindings[tid] = new_id
                    window_display_names[new_id] = window_display_names.get(val, val)
                    changed = True
                else:
                    # Keep dead window binding — /restore needs it
                    new_bindings[tid] = val
            elif new_id := live_by_name.get(val):
                logger.debug("Migrating thread binding %s -> %s", val, new_id)
                new_bindings[tid] = new_id
                window_display_names[new_id] = val
                changed = True
            else:
                logger.debug(
                    "Dropping old-format thread binding: user=%d, thread=%d, name=%s",
                    uid,
                    tid,
                    val,
                )
                changed = True
        bindings.clear()
        bindings.update(new_bindings)

    empty_users = [uid for uid, b in thread_bindings.items() if not b]
    for uid in empty_users:
        del thread_bindings[uid]
    return changed


def _resolve_offsets(
    user_window_offsets: dict,
    window_display_names: dict,
    live_by_name: dict[str, str],
    live_ids: set[str],
) -> bool:
    """Re-resolve user_window_offsets dict in-place. Returns True if changed."""
    changed = False
    for _uid, offsets in user_window_offsets.items():
        new_offsets: dict[str, int] = {}
        for key, offset in offsets.items():
            if is_window_id(key):
                if key in live_ids:
                    new_offsets[key] = offset
                elif new_id := live_by_name.get(window_display_names.get(key, key)):
                    new_offsets[new_id] = offset
                    changed = True
                else:
                    changed = True
            elif new_id := live_by_name.get(key):
                new_offsets[new_id] = offset
                changed = True
            else:
                changed = True
        offsets.clear()
        offsets.update(new_offsets)
    return changed


def resolve_stale_ids(
    live_windows: list[LiveWindow],
    window_states: dict,
    thread_bindings: dict,
    user_window_offsets: dict,
    window_display_names: dict,
    *,
    ids_stable: bool = True,
    live_session_ids: dict[str, str] | None = None,
) -> bool:
    """Re-resolve persisted window IDs against live multiplexer windows.

    Mutates all dicts in-place. Returns True if any changes were made.

    ``ids_stable`` gates the strategy on the backend capability
    ``ids_stable_across_restart`` (never the backend name):

    - True (tmux): window IDs survive a restart, so re-resolution matches a
      stale ID's display name against a live window. Handles two cases —
      old-format migration (window_name keys -> window_id keys) and stale IDs
      (window_id gone but display name matches a live window).
    - False (herdr): a server restart re-mints pane IDs, so display names are
      unreliable (tab labels collide across team splits). Re-resolution anchors
      on the durable agent ``session_id`` instead: a persisted state's stale ID
      is re-mapped to the live window currently running the same session
      (``live_session_ids``, from the hook-written session_map). No match keeps
      the entry as a dead window for /restore recovery.
    """
    if not ids_stable:
        return _resolve_by_session_id(
            live_windows,
            window_states,
            thread_bindings,
            user_window_offsets,
            window_display_names,
            live_session_ids or {},
        )

    live_by_name: dict[str, str] = {w.window_name: w.window_id for w in live_windows}
    live_ids: set[str] = {w.window_id for w in live_windows}

    changed = _resolve_window_states(
        window_states, window_display_names, live_by_name, live_ids
    )
    changed |= _resolve_thread_bindings(
        thread_bindings, window_display_names, live_by_name, live_ids
    )
    changed |= _resolve_offsets(
        user_window_offsets, window_display_names, live_by_name, live_ids
    )
    return changed


def _build_session_remap(
    live_windows: list[LiveWindow],
    window_states: dict,
    live_session_ids: dict[str, str],
) -> dict[str, str]:
    """Map each stale persisted window ID to its current live ID via session id.

    A persisted state with ``session_id`` S is re-mapped to the live window
    whose current session_map entry also carries S. Stale (pre-restart) ids and
    ids that are still live are skipped, so the result holds only genuine
    relocations (``old_id != new_id``).
    """
    live_ids: set[str] = {w.window_id for w in live_windows}
    by_session: dict[str, str] = {}
    for wid, sid in live_session_ids.items():
        if sid and wid in live_ids:
            by_session.setdefault(sid, wid)

    remap: dict[str, str] = {}
    for old_id, ws in window_states.items():
        if old_id in live_ids:
            continue
        sid = getattr(ws, "session_id", "") or ""
        new_id = by_session.get(sid) if sid else None
        if new_id and new_id != old_id:
            remap[old_id] = new_id
    return remap


def _remap_window_states(
    window_states: dict, window_display_names: dict, remap: dict[str, str]
) -> bool:
    """Re-key window_states (and their display names) per ``remap``."""
    changed = False
    new_states: dict = {}
    for key, ws in window_states.items():
        new_id = remap.get(key)
        if new_id:
            logger.debug(
                "Re-resolved herdr window_id %s -> %s (session match)", key, new_id
            )
            new_states[new_id] = ws
            display = window_display_names.pop(key, None)
            if display is not None:
                window_display_names[new_id] = display
            changed = True
        else:
            new_states[key] = ws
    window_states.clear()
    window_states.update(new_states)
    return changed


def _remap_thread_bindings(thread_bindings: dict, remap: dict[str, str]) -> bool:
    """Re-point thread bindings to the new window/tab id so bound topics re-attach."""
    changed = False
    for bindings in thread_bindings.values():
        for tid, val in list(bindings.items()):
            new_id = remap.get(val)
            if new_id:
                logger.debug("Re-resolved herdr thread binding %s -> %s", val, new_id)
                bindings[tid] = new_id
                changed = True
    return changed


def _remap_offsets(user_window_offsets: dict, remap: dict[str, str]) -> bool:
    """Re-key per-user read offsets per ``remap``."""
    changed = False
    for offsets in user_window_offsets.values():
        for key in list(offsets):
            new_id = remap.get(key)
            if new_id:
                offsets[new_id] = offsets.pop(key)
                changed = True
    return changed


def _resolve_by_session_id(
    live_windows: list[LiveWindow],
    window_states: dict,
    thread_bindings: dict,
    user_window_offsets: dict,
    window_display_names: dict,
    live_session_ids: dict[str, str],
) -> bool:
    """Re-resolve stale IDs by agent session id (non-stable-id backends).

    Used when ``ids_stable_across_restart`` is False. Unlike the tmux path this
    does not consult display names or ``is_window_id`` (herdr ids are ``wN:tM``,
    not ``@N``) and it re-points thread bindings to the new tab so a bound
    topic re-attaches to its agent after a herdr restart.
    """
    remap = _build_session_remap(live_windows, window_states, live_session_ids)
    if not remap:
        return False
    changed = _remap_window_states(window_states, window_display_names, remap)
    changed |= _remap_thread_bindings(thread_bindings, remap)
    changed |= _remap_offsets(user_window_offsets, remap)
    return changed
