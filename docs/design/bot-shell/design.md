# Bot Shell

## Functional Responsibilities

- Construct Telegram `Application` via python-telegram-bot builder pattern
- Register all command handlers (`/new`, `/history`, `/sessions`, `/resume`, etc.)
- Register message handlers (text, photos, documents, voice, topic lifecycle events)
- Wire callback dispatch to the Callback Registry (single `CallbackQueryHandler`)
- Manage application lifecycle:
  - `post_init`: startup sequence (resolve stale IDs, adopt windows, validate hooks, start monitor, start polling)
  - `post_stop`: send shutdown notification while HTTP transport is alive
  - `post_shutdown`: cancel polling, drain queues, stop monitor, flush state
- Configure HTTP transport (`ResilientPollingHTTPXRequest`), error handling, shutdown signals
- Apply group filter to all handlers (restrict to configured Telegram group)

## Encapsulated Knowledge

This module owns all knowledge that no other module should have:

- **PTB Application builder configuration** — token, request class, lifecycle hook wiring, builder options
- **Handler registration order** — filter evaluation order, group priorities, which handlers use which filters
- **Startup sequence** — the exact order of: resolve stale IDs → adopt unbound windows → validate hooks → create session monitor → wire callbacks → start monitor → create poll task
- **Shutdown sequence** — the exact order of: cancel poll task → drain message queues → stop session monitor → flush state
- **Group filter construction** — `filters.Chat(config.group_id)` applied to all handlers
- **Signal handling** — SIGINT/SIGTERM graceful shutdown coordination
- **Error handler registration** — global error handler for uncaught exceptions in handlers

## Subdomain Classification

**Supporting** — The bot shell is infrastructure plumbing. It changes when new commands are added (one line each) or when the PTB framework is upgraded, but the registration pattern itself is stable. The volatile business logic lives in the handler modules it registers, not in bot.py itself.

## Integration Contracts

### → Callback Dispatch (depends on)

- **Direction**: Bot Shell depends on Callback Dispatch
- **Contract type**: Contract (registration + dispatch)
- **What is shared**: The dispatch function as a handler, handler loading trigger
- **Contract definition**:
  - `callback_registry.load_handlers()` — called at import time to trigger `@register` decorators
  - `callback_registry.dispatch(update, context)` — registered as the sole `CallbackQueryHandler`

### → Command Orchestration (depends on)

- **Direction**: Bot Shell depends on Command Orchestration
- **Contract type**: Functional (handler registration)
- **What is shared**: Command forwarding handler function
- **Contract definition**: `forward_command_handler(update, context)` — registered as a `MessageHandler` with command filter for unknown commands

### → Topic Orchestration (depends on)

- **Direction**: Bot Shell depends on Topic Orchestration
- **Contract type**: Functional (callback wiring)
- **What is shared**: New window handler for SessionMonitor
- **Contract definition**: `handle_new_window(bot, event: NewWindowEvent)` — wired as `session_monitor.new_window_callback`

### → Session State (depends on)

- **Direction**: Bot Shell depends on Session State
- **Contract type**: Contract (lifecycle operations)
- **What is shared**: Startup resolution and shutdown persistence
- **Contract definition**: `resolve_stale_ids()`, `flush_state()`, `load_session_map()`, `sync_display_names(live_windows)`

### → Thread Router (depends on)

- **Direction**: Bot Shell depends on Thread Router
- **Contract type**: Contract (topic lifecycle)
- **What is shared**: Binding operations from topic close/delete handlers
- **Contract definition**: `unbind_thread(user_id, thread_id)`, `bind_thread(user_id, thread_id, window_id, window_name)`

### → Session Monitor (depends on)

- **Direction**: Bot Shell depends on Session Monitor
- **Contract type**: Functional (creation and lifecycle)
- **What is shared**: Monitor instance creation, callback wiring, start/stop
- **Contract definition**: `SessionMonitor(config, callbacks)`, `monitor.start()`, `monitor.stop()`

### → Polling Subsystem (depends on)

- **Direction**: Bot Shell depends on Polling Subsystem
- **Contract type**: Contract (task lifecycle)
- **What is shared**: Poll loop creation and cancellation
- **Contract definition**: `status_poll_loop(bot)` — created as an asyncio task, cancelled on shutdown

### → Handler modules (depends on)

- **Direction**: Bot Shell depends on individual handler modules
- **Contract type**: Contract (function references)
- **What is shared**: Handler function references for registration
- **Contract definition**: Each handler exposes a top-level `async def handler(update, context)` function; Bot Shell registers it with the appropriate filter

## Change Vectors

These are reasonable future changes that would require ONLY this module to change:

- **Adding a new bot command** — one `CommandHandler` registration line
- **Changing startup/shutdown sequence** — modify `post_init` / `post_shutdown`
- **Upgrading PTB version** — update builder pattern and handler registration API
- **Changing HTTP transport** — swap `ResilientPollingHTTPXRequest` configuration
- **Adding a new message content type handler** — one `MessageHandler` registration line
