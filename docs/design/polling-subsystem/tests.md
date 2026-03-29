# Polling Subsystem — Test Specification

## Unit Tests

### TerminalStatusStrategy

- **test_pyte_parse_active_status**: Parse pyte screen output containing an active provider status line. Expected: returns `StatusUpdate` with correct status text and active=True.
- **test_pyte_parse_idle_prompt**: Parse pyte screen showing a shell prompt with no activity. Expected: returns idle status.
- **test_pyte_content_hash_cache**: Parse same pane content twice in succession. Expected: second call returns cached result without re-parsing.
- **test_pyte_cache_invalidated_on_change**: Parse different pane content after a cached result. Expected: returns fresh parse result.
- **test_rc_detection_on**: Parse screen with Remote Control status bar. Expected: sets `rc_active=True` on `WindowPollState`.
- **test_rc_debounce_off**: RC disappears from screen. Expected: `rc_active` remains True until debounce window (2s) elapses.
- **test_rc_debounce_completes**: RC absent for longer than debounce window. Expected: `rc_active` transitions to False.
- **test_spinner_detection**: Parse screen with active spinner character sequence. Expected: returns active status even without explicit status line.
- **test_startup_grace_period**: Window just created, no status seen yet. Expected: suppresses "idle" status during grace period.
- **test_startup_grace_expires**: Grace period elapsed without any status. Expected: starts reporting idle/done.

### InteractiveUIStrategy

- **test_detect_permission_prompt**: Pane contains AskUserQuestion-style prompt. Expected: triggers interactive mode, returns interactive UI data.
- **test_detect_plan_mode_prompt**: Pane contains ExitPlanMode prompt. Expected: triggers interactive mode with plan context.
- **test_no_interactive_when_already_active**: Interactive mode already active for user. Expected: skips duplicate detection, does not re-trigger.
- **test_multi_pane_alert_detection**: Non-active pane has a blocked permission prompt. Expected: creates pane alert entry in `_pane_alert_hashes`.
- **test_multi_pane_alert_dedup**: Same pane alert content detected twice. Expected: does not send duplicate alert.
- **test_multi_pane_alert_cleared_on_resolution**: Pane no longer shows prompt. Expected: clears pane alert hash.
- **test_pane_count_cache_ttl**: Pane count queried within TTL. Expected: returns cached count without tmux call. After TTL: re-queries.

### TopicLifecycleStrategy

- **test_autoclose_done_timer_start**: Window enters "done" state. Expected: starts autoclose timer with configured timeout.
- **test_autoclose_done_timer_fires**: Timer elapses while window still "done". Expected: triggers topic close + cleanup.
- **test_autoclose_timer_cancelled_on_activity**: Window becomes active again before timer fires. Expected: timer cancelled, no close.
- **test_autoclose_dead_timer_start**: Window detected as dead (process exited). Expected: starts dead autoclose timer with separate timeout.
- **test_dead_window_notification**: Dead window detected for first time. Expected: sends recovery keyboard to user.
- **test_dead_notification_dedup**: Same dead window detected again. Expected: no duplicate notification (tracked in `_dead_notified`).
- **test_topic_probe_success**: Topic exists in Telegram. Expected: resets probe failure counter.
- **test_topic_probe_failure_accumulation**: Topic probe fails 3 consecutive times (60s interval). Expected: counter reaches threshold, triggers cleanup.
- **test_unbound_window_ttl**: Window has no thread binding for longer than TTL. Expected: auto-kills tmux window.
- **test_unbound_window_ttl_reset**: Unbound window gets bound before TTL. Expected: TTL timer resets.

### ShellRelayStrategy

- **test_delegate_passive_check**: Shell provider window polled. Expected: delegates to `check_passive_shell_output()`.
- **test_skip_non_shell_window**: Non-shell provider window. Expected: skips passive output check.
- **test_clear_state_on_provider_change**: Window provider changes from shell to claude. Expected: clears shell monitor state.

### Polling Coordinator

- **test_poll_loop_iterates_bindings**: Two active bindings exist. Expected: both windows polled in one cycle.
- **test_poll_loop_error_recovery**: One window poll raises `OSError`. Expected: error logged, other windows still polled, exponential backoff applied.
- **test_poll_loop_skips_empty_bindings**: No active bindings. Expected: loop completes quickly without errors.

## Integration Contract Tests

- **test_iter_thread_bindings_contract**: Verify polling coordinator correctly consumes `thread_router.iter_thread_bindings()` output format — yields `(int, int, str)` tuples.
- **test_window_state_store_contract**: Verify strategies correctly read WindowState fields via `get_window_state()` — fields `provider_name`, `notification_mode`, `session_id` are accessible.
- **test_enqueue_status_update_contract**: Verify status enqueue call passes correct arguments — `user_id`, `thread_id`, `window_id`, `text` in expected positions.
- **test_update_topic_emoji_contract**: Verify emoji update receives valid state string (one of: "active", "idle", "done", "dead").
- **test_provider_parse_terminal_status_contract**: Verify provider's `parse_terminal_status()` is called with `list[str]` and returns `StatusUpdate | None`.
- **test_capture_pane_contract**: Verify `tmux_manager.capture_pane()` is called with window_id string and returns `str`.

## Boundary Tests

- **test_empty_pane_capture**: Pane capture returns empty string. Expected: no crash, treated as idle.
- **test_pane_capture_none**: Pane capture returns None (window disappeared). Expected: handled gracefully, triggers dead detection.
- **test_invalid_window_id**: Poll called with stale window_id not in tmux. Expected: window treated as dead, not as error.
- **test_very_long_pane_output**: Pane contains 10,000+ lines. Expected: only last N lines parsed (bounded), no memory spike.
- **test_concurrent_poll_state_access**: Two poll cycles overlap (slow tmux response). Expected: state access is safe, no corruption.
- **test_clear_functions_idempotent**: Call `clear_window_poll_state()` for non-existent window. Expected: no KeyError, no side effects.
- **test_typing_indicator_throttle**: Typing indicator sent, then immediately requested again. Expected: second send suppressed within throttle window.

## Behavior Tests

- **test_active_window_full_cycle**: Window transitions from startup → active (status detected) → idle (prompt) → done (process exits). Expected: correct emoji updates, status messages, and autoclose timer at each stage.
- **test_interactive_prompt_surfaced_to_user**: Agent asks permission question in terminal. Expected: interactive keyboard appears in Telegram within one poll cycle.
- **test_dead_window_recovery_offered**: Agent process crashes. Expected: user receives recovery keyboard with Fresh/Continue/Resume options.
- **test_shell_passive_output_relayed**: Shell command produces output between prompts. Expected: output captured and sent to user in Telegram.
- **test_multi_pane_blocked_alert**: Agent team member in pane %1 asks permission while pane %0 is active. Expected: user receives alert about blocked pane.
- **test_notification_mode_respected**: Window set to "muted". Expected: status updates suppressed, only errors delivered.
- **test_rc_mode_reported**: Remote Control active in terminal. Expected: status message includes RC indicator.
