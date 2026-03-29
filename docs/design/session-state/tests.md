# Session State — Test Specification

## Unit Tests

### WindowState

- **test_window_state_defaults**: Create WindowState with no args. Expected: `session_id=""`, `notification_mode="all"`, `approval_mode="normal"`, `batch_mode="batched"`, `external=False`.
- **test_window_state_to_dict**: Create WindowState with all fields set. Expected: `to_dict()` returns dict with all fields, no extra keys.
- **test_window_state_from_dict**: Call `from_dict()` with a complete dict. Expected: returns WindowState with all fields populated.
- **test_window_state_from_dict_missing_fields**: Call `from_dict()` with partial dict (missing `batch_mode`). Expected: missing fields get defaults, no error.
- **test_window_state_from_dict_extra_fields**: Dict contains unknown key "future_field". Expected: ignored silently, no error.

### SessionManager — Window State

- **test_get_window_state_creates_default**: Get state for unknown window. Expected: returns default WindowState, persisted for future access.
- **test_get_window_state_returns_existing**: Set provider on window, then get state. Expected: returns WindowState with provider set.
- **test_set_window_provider**: Set provider to "codex" with cwd. Expected: `get_window_state()` shows `provider_name="codex"`, `cwd` set.
- **test_clear_window_session**: Set session_id, then clear. Expected: `session_id=""`, `transcript_path=""`.

### SessionManager — Preferences

- **test_notification_mode_cycle**: Cycle from "all". Expected: "all" → "errors_only" → "muted" → "all".
- **test_approval_mode_set**: Set approval mode to "yolo". Expected: `get_approval_mode()` returns "yolo".
- **test_batch_mode_cycle**: Cycle from "batched". Expected: "batched" → "verbose" → "batched".
- **test_preferences_persist**: Set notification mode, trigger save, reload. Expected: mode preserved after reload.

### SessionManager — Directory Favorites

- **test_update_mru**: Add path to MRU. Expected: path at front of MRU list.
- **test_mru_dedup**: Add same path twice. Expected: single entry, moved to front.
- **test_mru_cap**: Add 7 paths. Expected: MRU capped at 5, oldest dropped.
- **test_toggle_star**: Toggle path on. Expected: path in starred list. Toggle again: removed.
- **test_starred_independent_of_mru**: Star a path, add different path to MRU. Expected: starred list unchanged.

### SessionManager — User Offsets

- **test_update_user_window_offset**: Set offset to 1024. Expected: `get_user_window_offset()` returns 1024.
- **test_offset_default_zero**: Get offset for unknown window. Expected: returns 0.
- **test_prune_stale_offsets**: Set offsets for "@0" and "@5", prune with known_ids={"@0"}. Expected: "@5" offset removed.

### SessionManager — Session Map

- **test_load_session_map_new_entry**: Write session_map.json with new window entry. Expected: `window_states` updated with session_id, cwd, transcript_path.
- **test_load_session_map_session_change**: Window's session_id changes (post-/clear). Expected: old session cleaned up, new session_id stored.
- **test_register_hookless_session**: Register Codex session. Expected: session_map.json updated with synthetic entry; window_state synced.
- **test_prune_session_map**: Prune with live_ids excluding "@5". Expected: "@5" entry removed from session_map.json.

### SessionManager — Persistence

- **test_save_debounced**: Trigger 5 mutations rapidly. Expected: only 1 file write (debounced).
- **test_flush_state_immediate**: Call `flush_state()`. Expected: file written immediately regardless of debounce timer.
- **test_serialize_deserialize_roundtrip**: Populate all state, serialize, deserialize into fresh instance. Expected: all data preserved.

## Integration Contract Tests

- **test_window_state_store_protocol**: Verify SessionManager satisfies `WindowStateStore` protocol — has `get_window_state`, `get_display_name`, `get_session_id_for_window`, `clear_window_session`.
- **test_user_preferences_protocol**: Verify SessionManager satisfies `UserPreferences` protocol — has all notification/approval/batch mode methods.
- **test_session_resolver_protocol**: Verify SessionManager satisfies `SessionResolver` protocol — has `resolve_session_for_window`, `get_recent_messages`.
- **test_thread_router_persistence_contract**: Verify `_serialize_state()` calls `thread_router.to_dict()` and `_load_state()` calls `thread_router.from_dict()`.

## Boundary Tests

- **test_concurrent_saves**: Two mutations trigger saves simultaneously. Expected: no file corruption, last write wins.
- **test_load_corrupt_state_json**: state.json contains malformed JSON. Expected: graceful fallback to empty state, warning logged.
- **test_load_empty_state_json**: state.json is empty file. Expected: treated as fresh state.
- **test_session_map_file_missing**: session_map.json does not exist. Expected: `load_session_map()` returns cleanly, no error.
- **test_very_large_state**: 500 windows with full state. Expected: save/load completes within reasonable time.
- **test_window_state_isolation**: Get state for "@0", mutate it. Expected: `get_window_state("@0")` returns mutated state (same reference); "@1" unaffected.

## Behavior Tests

- **test_startup_state_recovery**: Persist state with bindings and preferences, create fresh SessionManager, load. Expected: all bindings, window states, preferences, and offsets restored.
- **test_stale_id_resolution**: Persist state with window "@0" named "my-project". Tmux restarts, window is now "@3" but named "my-project". Expected: `resolve_stale_ids()` re-maps "@0" → "@3" in all state maps.
- **test_hookless_session_full_cycle**: Register hookless Codex session, verify session_map entry, verify window_state sync, then clear session. Expected: clean lifecycle with no orphaned entries.
- **test_preference_change_reflected_in_polling**: Set notification mode to "muted" on window. Expected: polling subsystem (via `WindowStateStore`) sees "muted" when checking `get_window_state()`.
