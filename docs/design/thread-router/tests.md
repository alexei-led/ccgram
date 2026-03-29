# Thread Router — Test Specification

## Unit Tests

- **test_bind_thread**: Bind user_id=1, thread_id=42 to window_id="@0". Expected: `get_window_for_thread(1, 42)` returns "@0".
- **test_bind_thread_updates_reverse_index**: Bind thread. Expected: `get_thread_for_window(1, "@0")` returns 42.
- **test_bind_thread_overwrites**: Bind thread 42 to "@0", then rebind to "@5". Expected: `get_window_for_thread(1, 42)` returns "@5"; old reverse entry removed.
- **test_unbind_thread**: Bind then unbind thread. Expected: `get_window_for_thread` returns None; reverse index cleared.
- **test_unbind_nonexistent**: Unbind thread that was never bound. Expected: no error, no side effects.
- **test_get_all_thread_windows**: Bind three threads for user_id=1. Expected: returns dict with all three {thread_id: window_id} entries.
- **test_iter_thread_bindings**: Bind threads for two users. Expected: yields all (user_id, thread_id, window_id) tuples across both users.
- **test_iter_empty**: No bindings. Expected: yields nothing, no error.
- **test_resolve_window_for_thread_returns_none**: Unknown user/thread. Expected: returns None.
- **test_set_group_chat_id**: Set chat_id for user+thread. Expected: `resolve_chat_id(user_id, thread_id)` returns the chat_id.
- **test_resolve_chat_id_fallback**: No thread-specific chat_id, but user has other bindings with chat_ids. Expected: returns first known chat_id for that user.
- **test_resolve_chat_id_none**: No chat_ids at all for user. Expected: returns None.
- **test_set_display_name**: Set display name for window. Expected: `get_display_name("@0")` returns the name.
- **test_get_display_name_default**: No display name set. Expected: returns window_id as fallback.
- **test_get_window_for_chat_thread**: Set group chat_id, then query by chat_id + thread_id. Expected: returns correct window_id.

## Integration Contract Tests

- **test_to_dict_roundtrip**: Bind several threads, set chat IDs and display names. Call `to_dict()`, then `from_dict()` on a fresh instance. Expected: all bindings, chat IDs, and display names restored exactly.
- **test_to_dict_int_key_serialization**: Bind thread with integer user_id=123. Expected: `to_dict()` produces JSON-safe string keys; `from_dict()` restores integer keys.
- **test_schedule_save_called_on_mutation**: Bind a thread. Expected: `session_state.schedule_save()` is called once.
- **test_schedule_save_called_on_unbind**: Unbind a thread. Expected: `session_state.schedule_save()` is called.
- **test_iter_thread_bindings_type_contract**: Iterate bindings. Expected: each tuple is `(int, int, str)` — user_id, thread_id, window_id.

## Boundary Tests

- **test_bind_same_thread_twice**: Bind thread 42 to "@0", then bind thread 42 to "@0" again. Expected: idempotent, no duplicate entries, reverse index consistent.
- **test_bind_with_empty_window_name**: Bind with window_name="". Expected: display name set to empty, `get_display_name` returns window_id fallback.
- **test_large_user_count**: Bind 1000 threads across 100 users. Expected: iteration completes, no performance issues, reverse index consistent.
- **test_concurrent_bind_unbind**: Bind and unbind different threads rapidly. Expected: data structures remain consistent (forward and reverse maps agree).
- **test_composite_key_special_characters**: User_id=0, thread_id=0. Expected: composite key "0:0" resolves correctly, no collision.
- **test_unbind_preserves_other_users**: Two users bound to same window. Unbind one. Expected: other user's binding intact.

## Behavior Tests

- **test_outbound_message_routing**: User sends message in topic with thread_id=42. Expected: `get_window_for_thread(user_id, 42)` resolves to correct tmux window_id for message forwarding.
- **test_inbound_message_routing**: New message arrives for window "@0". Expected: `get_thread_for_window(user_id, "@0")` resolves to correct thread_id for Telegram delivery.
- **test_multi_group_routing**: Same user bound in two different Telegram groups. Expected: `resolve_chat_id` returns the correct group-specific chat_id based on thread context.
- **test_display_name_lifecycle**: Bind with name "my-project", then update via `set_display_name`. Expected: new name returned by `get_display_name`, save triggered.
