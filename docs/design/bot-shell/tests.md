# Bot Shell — Test Specification

## Unit Tests

- **test_create_bot_returns_application**: Call `create_bot()`. Expected: returns a PTB `Application` instance with token configured.
- **test_command_handlers_registered**: Create bot, inspect `application.handlers`. Expected: all expected command handlers registered (`/new`, `/history`, `/sessions`, `/resume`, `/unbind`, `/upgrade`, `/recall`, `/screenshot`, `/panes`, `/sync`, `/toolbar`, `/verbose`, `/restore`).
- **test_callback_handler_registered**: Create bot. Expected: exactly one `CallbackQueryHandler` registered, pointing to `callback_registry.dispatch`.
- **test_message_handlers_registered**: Create bot. Expected: text, photo, document, voice, and unsupported content handlers registered in correct order.
- **test_topic_lifecycle_handlers_registered**: Create bot. Expected: topic_closed and topic_edited handlers registered with appropriate filters.
- **test_group_filter_applied**: Create bot with `config.group_id` set. Expected: all handlers have group filter applied.
- **test_group_filter_absent_when_no_group**: Create bot without `config.group_id`. Expected: no group filter applied.

## Integration Contract Tests

- **test_callback_registry_loaded**: After `create_bot()`, verify `callback_registry.load_handlers()` was called. Expected: registry contains registered handlers.
- **test_forward_command_handler_wired**: Create bot, find the command forwarding `MessageHandler`. Expected: points to `command_orchestration.forward_command_handler`.
- **test_post_init_starts_monitor**: Mock SessionMonitor. Call `post_init()`. Expected: monitor created with correct callbacks, `start()` called.
- **test_post_init_starts_polling**: Mock asyncio task creation. Call `post_init()`. Expected: `status_poll_loop` task created.
- **test_post_init_resolves_stale_ids**: Mock session_manager. Call `post_init()`. Expected: `resolve_stale_ids()` called before monitor starts.
- **test_post_shutdown_flushes_state**: Mock session_manager. Call `post_shutdown()`. Expected: `flush_state()` called.
- **test_post_shutdown_stops_monitor**: Mock session monitor. Call `post_shutdown()`. Expected: `monitor.stop()` called.
- **test_post_shutdown_cancels_poll_task**: Create poll task, call `post_shutdown()`. Expected: task cancelled.

## Boundary Tests

- **test_post_init_hook_check_non_blocking**: Hook validation fails (hooks not installed). Expected: warning logged, bot continues startup.
- **test_post_shutdown_handles_already_cancelled_task**: Poll task already cancelled before shutdown. Expected: no error, cleanup proceeds.
- **test_post_shutdown_handles_monitor_already_stopped**: Monitor already stopped. Expected: no error, state still flushed.
- **test_create_bot_with_missing_token**: No token configured. Expected: raises clear configuration error.

## Behavior Tests

- **test_startup_sequence_order**: Trace call order during `post_init()`. Expected: resolve_stale_ids → adopt_windows → validate_hooks → create_monitor → wire_callbacks → start_monitor → create_poll_task (strict order).
- **test_shutdown_sequence_order**: Trace call order during `post_shutdown()`. Expected: cancel_poll → drain_queues → stop_monitor → flush_state (strict order).
- **test_command_dispatched_to_handler**: Send a `/history` Update through the application. Expected: `history_command()` handler invoked.
- **test_unknown_command_forwarded**: Send an unknown `/foo` command through the application. Expected: `forward_command_handler()` invoked.
- **test_topic_close_triggers_unbind**: Send topic_closed Update. Expected: `unbind_thread()` called for the topic.
