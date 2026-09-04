"""Window ID resolution, format helpers, and startup migration.

Provides shared window ID helpers used across session, tmux_manager, and
handler modules (no intra-package imports — safe from circular dependencies):
  - is_window_id(): validate tmux window ID format (@0, @12).
  - resolve_stale_ids(): full startup recovery — remaps persisted window IDs
    against live tmux windows, handles old-format migration, prunes dead entries.
  - migrate_window_aliases(): per-cycle reconciliation — folds state persisted
    under a superseded window id (``WindowRef.alias_window_ids``) onto the id
    that identifies the same window now.
"""

from collections.abc import Callable
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
    non-tmux backends key by backend name plus their opaque durable target
    (e.g. ``herdr:herdr-session-v1-<digest>``), never a raw pane/tab locator.

    This is the pure, config-free version used by status_cmd and session_map.
    ``session_map.session_map_prefix()`` wraps this with ``config`` values.
    """
    if mux_name == "tmux":
        return f"{session_name}:"
    return f"{mux_name}:"


@dataclass(frozen=True)
class AliasMigration:
    """One superseded window identity folded onto its current one."""

    alias_id: str
    canonical_id: str


_MIGRATED_TEXT_FIELDS = (
    "session_id",
    "cwd",
    "transcript_path",
    "provider_name",
    "window_name",
)

# Persisted and transient scalar fields whose defaults are meaningful. During
# the brief alias/canonical collision the canonical row is commonly built from
# hook data with defaults, while the alias row carries user and lifecycle state.
_MIGRATED_DEFAULT_FIELDS = (
    "approval_mode",
    "batch_mode",
    "tool_call_visibility",
    "origin",
    "pane_lifecycle_notify",
    "rc_probe_state",
    "rc_armed_at",
    "worktree_path",
    "worktree_branch",
    "provider_manual_override",
    "legacy_herdr",
    "legacy_herdr_archived",
    "legacy_herdr_archive_user_id",
    "legacy_herdr_archive_thread_id",
)

# Superseded id -> the id that identifies the same window now. Written by
# ``migrate_window_aliases`` as it folds state over, read by anything still
# holding an id minted before the supersession — the topic-creation flow is
# the one that matters: it creates a window, then waits for the hook to
# register it, and on a backend whose identity firms up over time the hook
# writes under an id creation never saw.
#
# In-memory. Stale redirects are bounded, but aliases in the current live
# snapshot are never evicted: a creation flow may still hold any one of them.
_MAX_STALE_ALIAS_REDIRECTS = 256
_alias_redirects: dict[str, str] = {}


def resolve_window_alias(window_id: str) -> str:
    """Return the id ``window_id`` answers to now, following supersessions.

    Identity can be superseded more than once, so this walks the chain. A
    window whose identity was never superseded resolves to itself, which
    makes it safe to call unconditionally on any backend.
    """
    seen: set[str] = set()
    current = window_id
    while current in _alias_redirects and current not in seen:
        seen.add(current)
        current = _alias_redirects[current]
    return current


def _record_alias_redirect(alias_id: str, canonical_id: str) -> None:
    """Remember that ``alias_id`` now resolves to ``canonical_id``."""
    _alias_redirects[alias_id] = canonical_id


def _prune_stale_alias_redirects(active_aliases: set[str]) -> None:
    """Bound stale redirects without evicting any current alias or its chain."""
    protected: set[str] = set()
    for alias_id in active_aliases:
        current = alias_id
        seen: set[str] = set()
        while current in _alias_redirects and current not in seen:
            seen.add(current)
            protected.add(current)
            current = _alias_redirects[current]
    stale = [key for key in _alias_redirects if key not in protected]
    for key in stale[:-_MAX_STALE_ALIAS_REDIRECTS]:
        _alias_redirects.pop(key, None)


def reset_alias_redirects() -> None:
    """Drop every recorded redirect — only for tests."""
    _alias_redirects.clear()


_NO_DEFAULT = object()


def _adopt_non_default_fields(current: object, stale: object) -> None:
    """Fill canonical defaults from the superseded state without losing choices."""
    for name in _MIGRATED_DEFAULT_FIELDS:
        # Dataclass scalar defaults remain class attributes. Test stand-ins or
        # future state types without that contract are left unchanged.
        default = getattr(type(current), name, _NO_DEFAULT)
        if default is _NO_DEFAULT:
            continue
        stale_value = getattr(stale, name, default)
        if getattr(current, name, default) == default and stale_value != default:
            setattr(current, name, stale_value)


def _merge_panes(current: object, stale: object) -> None:
    """Merge per-pane state, preferring the canonical row on key collisions."""
    stale_panes = getattr(stale, "panes", None)
    if not isinstance(stale_panes, dict) or not stale_panes:
        return
    current_panes = getattr(current, "panes", None)
    if isinstance(current_panes, dict):
        setattr(current, "panes", {**stale_panes, **current_panes})


def _migrate_window_state(
    window_states: dict, alias_id: str, canonical_id: str
) -> None:
    """Fold the alias's complete window state onto the canonical id, in place."""
    stale = window_states.pop(alias_id, None)
    if stale is None:
        return
    current = window_states.get(canonical_id)
    if current is None:
        window_states[canonical_id] = stale
        return
    # Both rows represent the same live window. Keep values already resolved on
    # the canonical row, but fill every gap/default from the alias so identity
    # convergence cannot discard worktree, mode, lifecycle, or pane state.
    for field in _MIGRATED_TEXT_FIELDS:
        if not getattr(current, field, "") and getattr(stale, field, ""):
            setattr(current, field, getattr(stale, field))
    _adopt_non_default_fields(current, stale)
    _merge_panes(current, stale)


def _alias_is_referenced(
    alias_id: str,
    window_states: dict,
    thread_bindings: dict,
    chat_thread_bindings: dict,
    user_window_offsets: dict,
    window_display_names: dict,
) -> bool:
    return (
        alias_id in window_states
        or alias_id in window_display_names
        or alias_id in chat_thread_bindings.values()
        or any(alias_id in bindings.values() for bindings in thread_bindings.values())
        or any(alias_id in offsets for offsets in user_window_offsets.values())
    )


def _binding_scope(key: object) -> object:
    """Return the uniqueness scope of a binding key.

    Chat-scoped keys are ``(user_id, chat_id, thread_id)``; the per-user
    ``thread_bindings`` sub-dicts are already one scope.
    """
    return key[:2] if isinstance(key, tuple) else None


def _duplicate_bindings(
    bindings: dict,
    canonical_id: str,
    kept_keys: set,
) -> list:
    """Return bindings on ``canonical_id`` that duplicate a kept alias binding.

    Scoped exactly as ``ThreadRouter.bind_thread`` scopes its own eviction, so
    the same window legitimately bound in another chat is not a duplicate.
    """
    kept_scopes = {_binding_scope(key) for key in kept_keys}
    duplicates = [
        key
        for key, window_id in bindings.items()
        if window_id == canonical_id
        and key not in kept_keys
        and _binding_scope(key) in kept_scopes
    ]
    if duplicates:
        logger.info(
            "Unbinding %d duplicate topic(s) for window %s: "
            "the superseded id already owns a topic",
            len(duplicates),
            canonical_id,
        )
    return duplicates


def _repoint_bindings(bindings: dict, alias_id: str, canonical_id: str) -> None:
    """Move one binding map's alias entries onto the canonical id."""
    alias_keys = {key for key, window_id in bindings.items() if window_id == alias_id}
    if not alias_keys:
        return
    for key in _duplicate_bindings(bindings, canonical_id, alias_keys):
        del bindings[key]
    for key in alias_keys:
        bindings[key] = canonical_id


def _repoint_alias_references(
    alias_id: str,
    canonical_id: str,
    thread_bindings: dict,
    chat_thread_bindings: dict,
    user_window_offsets: dict,
    window_display_names: dict,
) -> None:
    """Point every binding, offset, and display name at the canonical id.

    One window is one topic (``ThreadRouter.bind_thread`` enforces that on the
    bind path). A repoint can violate it: if the canonical id was discovered as
    an unbound window before ccgram learned it supersedes the alias, a second
    topic is already bound to it. The alias's topic is the one that carries the
    user's history, so it wins and the duplicate is unbound — leaving both is
    what makes two topics answer for one agent.
    """
    for bindings in thread_bindings.values():
        _repoint_bindings(bindings, alias_id, canonical_id)
    _repoint_bindings(chat_thread_bindings, alias_id, canonical_id)
    for offsets in user_window_offsets.values():
        offset = offsets.pop(alias_id, None)
        if offset is not None:
            offsets.setdefault(canonical_id, offset)
    display_name = window_display_names.pop(alias_id, "")
    if display_name and not window_display_names.get(canonical_id):
        window_display_names[canonical_id] = display_name


def migrate_window_aliases(
    aliases: dict[str, str],
    window_states: dict,
    thread_bindings: dict,
    chat_thread_bindings: dict,
    user_window_offsets: dict,
    window_display_names: dict,
    *,
    record_redirects: bool = True,
) -> list[AliasMigration]:
    """Fold state persisted under superseded window ids onto the current ones.

    ``aliases`` maps a superseded id to the live id that now identifies the
    same window (``WindowRef.alias_window_ids`` inverted). A backend emits
    those when its identity is derived from facts that arrive over time: the
    SessionStart hook can resolve a window to a provisional identity moments
    before the durable one exists, so ``session_map.json`` and ``window_states``
    land under one id while the topic later binds the other. Inbound routing
    matches on the *bound* window's session id, so without this migration the
    two never meet and agent replies are dropped while status polling — which
    resolves the live window directly — keeps working.

    Mutates every dict in place. Returns the migrations performed so the caller
    can mirror them into ``session_map.json`` (whose hook-written entry would
    otherwise recreate the alias state on the next sync). ``record_redirects``
    may be disabled for a pure preflight over cloned maps; redirects then remain
    unchanged until the persisted migration commits.
    """
    migrations: list[AliasMigration] = []
    active_aliases = {
        alias_id
        for alias_id, canonical_id in aliases.items()
        if alias_id and canonical_id and alias_id != canonical_id
    }
    for alias_id, canonical_id in aliases.items():
        if alias_id == canonical_id or not alias_id or not canonical_id:
            continue
        # Record the redirect before the reference check: the identity is
        # superseded whether or not any state moved, and a flow still holding
        # the old id needs the answer either way. Preflight callers disable
        # this side effect until persistence succeeds.
        if record_redirects:
            _record_alias_redirect(alias_id, canonical_id)
        if not _alias_is_referenced(
            alias_id,
            window_states,
            thread_bindings,
            chat_thread_bindings,
            user_window_offsets,
            window_display_names,
        ):
            continue

        _migrate_window_state(window_states, alias_id, canonical_id)
        _repoint_alias_references(
            alias_id,
            canonical_id,
            thread_bindings,
            chat_thread_bindings,
            user_window_offsets,
            window_display_names,
        )
        migrations.append(AliasMigration(alias_id=alias_id, canonical_id=canonical_id))
        logger.info("Reconciled superseded window id %s -> %s", alias_id, canonical_id)
    if record_redirects:
        _prune_stale_alias_redirects(active_aliases)
    return migrations


def _resolve_window_states(
    window_states: dict,
    window_display_names: dict,
    live_by_name: dict[str, str],
    live_ids: set[str],
    window_id_predicate: Callable[[str], bool],
) -> bool:
    """Re-resolve window_states dict in-place. Returns True if changed."""
    changed = False
    new_states: dict = {}
    for key, ws in window_states.items():
        if window_id_predicate(key):
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
    display_lookup: dict,
    live_by_name: dict[str, str],
    live_ids: set[str],
    window_id_predicate: Callable[[str], bool],
) -> bool:
    """Re-resolve thread_bindings dict in-place. Returns True if changed."""
    changed = False
    for uid, bindings in thread_bindings.items():
        new_bindings: dict[int, str] = {}
        for tid, val in bindings.items():
            if window_id_predicate(val):
                if val in live_ids:
                    new_bindings[tid] = val
                elif new_id := live_by_name.get(display_lookup.get(val, val)):
                    logger.debug("Re-resolved thread binding %s -> %s", val, new_id)
                    new_bindings[tid] = new_id
                    window_display_names[new_id] = display_lookup.get(val, val)
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
    display_lookup: dict,
    live_by_name: dict[str, str],
    live_ids: set[str],
    window_id_predicate: Callable[[str], bool],
) -> bool:
    """Re-resolve user_window_offsets dict in-place. Returns True if changed."""
    changed = False
    for _uid, offsets in user_window_offsets.items():
        new_offsets: dict[str, int] = {}
        for key, offset in offsets.items():
            if window_id_predicate(key):
                if key in live_ids:
                    new_offsets[key] = offset
                elif new_id := live_by_name.get(display_lookup.get(key, key)):
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
    window_id_predicate: Callable[[str], bool] = is_window_id,
) -> bool:
    """Re-resolve persisted window IDs against live multiplexer windows.

    Mutates all dicts in-place. Returns True if any changes were made.

    ``ids_stable`` gates the strategy on the backend capability
    ``ids_stable_across_restart`` (never the backend name):

    - True (tmux and agterm): window IDs survive a restart, so re-resolution
      matches a stale ID's display name against a live window. Handles two cases —
      old-format migration (window_name keys -> window_id keys) and stale IDs
      (window_id gone but display name matches a live window).
    - False: bindings are durable opaque targets. A missing target can be
      temporarily unresolved and multiple live records can be ambiguous, so it
      is intentionally retained without any name, locator, or session-map
      remapping. Guarded backend actions decide its current availability.
    """
    if not ids_stable:
        return False

    live_by_name: dict[str, str] = {w.window_name: w.window_id for w in live_windows}
    live_ids: set[str] = {w.window_id for w in live_windows}
    # Resolve every persisted map against one pre-mutation name snapshot.
    # _resolve_window_states rewrites display names first; using that mutated
    # map for bindings/offsets used to leave them pointing at stale IDs.
    display_lookup = dict(window_display_names)

    changed = _resolve_window_states(
        window_states,
        window_display_names,
        live_by_name,
        live_ids,
        window_id_predicate,
    )
    changed |= _resolve_thread_bindings(
        thread_bindings,
        window_display_names,
        display_lookup,
        live_by_name,
        live_ids,
        window_id_predicate,
    )
    changed |= _resolve_offsets(
        user_window_offsets,
        display_lookup,
        live_by_name,
        live_ids,
        window_id_predicate,
    )
    return changed
