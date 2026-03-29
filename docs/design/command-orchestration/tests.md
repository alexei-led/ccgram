# Command Orchestration — Test Specification

## Unit Tests

### Command Forwarding

- **test_forward_known_command**: Send `/compact` in Claude topic. Expected: resolved to provider command, sent to tmux window via `send_to_window`.
- **test_forward_unknown_command_warns**: Send `/nonexistent` in Claude topic. Expected: user warned that command is not supported by this provider.
- **test_forward_command_known_in_other_provider**: Send Claude-only command in Codex topic. Expected: user informed the command exists but in a different provider.
- **test_forward_records_history**: Forward `/status` command. Expected: `record_command()` called with correct user_id, thread_id, command text.
- **test_forward_unbound_topic**: Send command in topic with no window binding. Expected: user informed topic is unbound, no tmux send.

### Command Probing

- **test_probe_detects_error**: After forwarding command, pane output contains error pattern. Expected: user notified of command failure.
- **test_probe_detects_success**: After forwarding, pane output shows normal activity. Expected: no failure notification.
- **test_probe_timeout**: Probe waits for configured delay. Expected: inspection happens after delay, not immediately.
- **test_status_snapshot_via_protocol**: Provider implements `build_status_snapshot()`. Expected: snapshot text returned and sent to user.
- **test_status_snapshot_unsupported**: Provider returns None from `build_status_snapshot()`. Expected: no snapshot sent, no error.
- **test_has_output_since_via_protocol**: Provider implements `has_output_since()`. Expected: returns True when new output exists after offset.

### Menu Caching

- **test_scoped_menu_set_on_first_command**: First command in a topic. Expected: `set_my_commands` called with provider-specific commands at member scope.
- **test_scoped_menu_cached**: Second command in same topic, same provider. Expected: no `set_my_commands` call (cached).
- **test_scoped_menu_invalidated_on_provider_change**: Switch topic to different provider. Expected: menu re-synced with new provider's commands.
- **test_menu_fallback_to_chat_scope**: Member scope `set_my_commands` raises `BadRequest`. Expected: falls back to chat scope.
- **test_menu_fallback_to_global_scope**: Chat scope also fails. Expected: falls back to global scope.
- **test_menu_refresh_scheduled**: After `post_init`, verify menu refresh job scheduled. Expected: job runs every 10 minutes.

## Integration Contract Tests

- **test_provider_protocol_capabilities_queried**: Forward command. Expected: `provider.capabilities` accessed to check supported commands.
- **test_provider_get_command_metadata_called**: Forward command. Expected: `provider.get_command_metadata()` called to resolve command names.
- **test_send_to_window_contract**: Forward command. Expected: `session_manager.send_to_window(window_id, text)` called with correct arguments.
- **test_resolve_chat_id_contract**: Forward command. Expected: `thread_router.resolve_chat_id()` called for error message delivery.
- **test_safe_reply_contract**: Command fails validation. Expected: `safe_reply(update, error_text)` called for user notification.

## Boundary Tests

- **test_forward_empty_command**: Send `/` with no command name. Expected: ignored or handled gracefully.
- **test_forward_command_with_args**: Send `/status --verbose`. Expected: full text forwarded including args.
- **test_provider_none_for_window**: Provider resolution returns None (stale window). Expected: user notified, no crash.
- **test_menu_cache_eviction**: LRU cache at capacity, new entry added. Expected: oldest entry evicted.
- **test_probe_with_empty_transcript**: Transcript path doesn't exist or is empty. Expected: probe completes without error.
- **test_concurrent_menu_syncs**: Two commands trigger menu sync simultaneously. Expected: no race condition, one sync wins.

## Behavior Tests

- **test_full_command_lifecycle**: User sends `/status` in Codex topic. Expected: command validated → forwarded to tmux → probed for errors → status snapshot built via provider → result sent to user.
- **test_provider_switch_updates_menu**: User switches from Claude topic to Codex topic, sends command. Expected: Telegram command menu updated to Codex commands before forwarding.
- **test_codex_status_via_protocol**: Send `/status` in Codex topic where CodexProvider implements `build_status_snapshot()`. Expected: Codex-specific status with token stats returned — no `capabilities.name == "codex"` check in this module.
