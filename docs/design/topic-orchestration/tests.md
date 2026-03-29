# Topic Orchestration — Test Specification

## Unit Tests

### Auto-Topic Creation

- **test_handle_new_window_creates_topic**: New window event with window_id="@5", name="my-project". Expected: Telegram `create_forum_topic` called with name "my-project".
- **test_handle_new_window_binds_topic**: Topic created successfully. Expected: `thread_router.bind_thread()` called with correct user_id, new thread_id, window_id.
- **test_handle_new_window_records_chat_id**: Topic created. Expected: `thread_router.set_group_chat_id()` called with chat_id from topic creation.
- **test_handle_new_window_skips_already_bound**: Window "@5" already has a thread binding. Expected: no topic creation, early return.
- **test_topic_name_from_window_name**: Window has explicit name. Expected: topic created with that name.
- **test_topic_name_from_cwd_basename**: Window has no name but has cwd="/home/user/my-project". Expected: topic name = "my-project".
- **test_topic_name_fallback_to_window_id**: No name, no cwd. Expected: topic name = "@5".

### Provider Auto-Detection

- **test_detect_from_pane_command**: Pane running "claude". Expected: provider detected as "claude".
- **test_detect_from_pane_command_node_wrapper**: Pane running "node" (JS runtime). Expected: falls through to runtime detection.
- **test_detect_from_runtime_fallback**: Pane command unrecognized, runtime probe finds "codex". Expected: provider set to "codex".
- **test_detect_sets_window_provider**: Provider detected as "gemini". Expected: `session_manager.set_window_provider("@5", "gemini")` called.
- **test_detect_default_when_unrecognized**: Both detection methods return None. Expected: config default provider used.

### Chat Enumeration

- **test_chat_from_active_bindings**: User has bindings in chat 12345. Expected: topic created in chat 12345.
- **test_chat_from_preserved_chat_ids**: No active bindings, but preserved group_chat_ids has chat 12345. Expected: topic created in chat 12345.
- **test_chat_from_config_fallback**: No bindings, no preserved IDs, config.group_id=12345. Expected: topic created in chat 12345.
- **test_dedup_across_sources**: Same chat_id from both active bindings and preserved IDs. Expected: topic created once.

### Rate Limiting

- **test_rate_limit_backoff_after_retry_after**: `create_forum_topic` raises `RetryAfter(10)`. Expected: chat added to `_topic_create_retry_until` with 10s backoff.
- **test_rate_limit_skips_backed_off_chat**: Chat is in backoff period. Expected: topic creation skipped for that chat.
- **test_rate_limit_expired**: Backoff period elapsed. Expected: topic creation proceeds normally.

### Orphan Adoption

- **test_adopt_unbound_windows**: Window "@3" in state but not in any binding. Expected: delegated to `_adopt_orphaned_windows`.
- **test_no_orphans**: All windows in state have bindings. Expected: adoption not triggered.

## Integration Contract Tests

- **test_bind_thread_contract**: Verify `thread_router.bind_thread()` called with `(int, int, str, str)` — user_id, thread_id, window_id, window_name.
- **test_set_window_provider_contract**: Verify `session_manager.set_window_provider()` called with `(str, str, str | None)` — window_id, provider_name, cwd.
- **test_detect_provider_from_pane_contract**: Verify `detect_provider_from_pane()` called with `(str, str, str)` — pane_current_command, pane_tty, window_id.
- **test_find_window_by_id_contract**: Verify `tmux_manager.find_window_by_id()` called with window_id string.
- **test_create_forum_topic_contract**: Verify `bot.create_forum_topic()` called with chat_id and name.

## Boundary Tests

- **test_new_window_event_with_no_chats**: No active bindings, no preserved IDs, no config.group_id. Expected: topic creation skipped with warning, no crash.
- **test_new_window_tmux_gone**: Window disappeared between event and handling. Expected: `find_window_by_id` returns None, handled gracefully.
- **test_create_forum_topic_bad_request**: Telegram rejects topic creation (e.g., forum disabled). Expected: `BadRequest` caught, warning logged.
- **test_multiple_new_windows_rapid**: Three new window events arrive in quick succession. Expected: all three processed, rate limiting applied per chat.
- **test_new_window_event_for_foreign_window**: Window is emdash-owned (external). Expected: topic creation proceeds, window marked external.

## Behavior Tests

- **test_full_auto_topic_lifecycle**: New tmux window created externally → SessionMonitor fires NewWindowEvent → provider auto-detected → topic created in Telegram → thread bound → user can send messages. Expected: end-to-end flow completes.
- **test_restart_recovery**: Bot restarts with windows in state but no topic bindings (topics still exist in Telegram). Expected: `_adopt_unbound_windows` re-binds orphaned windows to existing topics.
- **test_rate_limited_creation_retries**: First topic creation hits flood control. Expected: backs off, retries on next poll cycle, eventually succeeds.
