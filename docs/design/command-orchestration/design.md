# Command Orchestration

## Functional Responsibilities

- Forward unknown `/command` messages to the provider CLI via tmux (the `forward_command_handler`)
- Manage provider-specific command menus with three-tier scoped caching (member → chat → global)
- Validate commands against provider capabilities (is this command supported? known in other providers?)
- Probe for command failure after forwarding (capture transcript offset before send, inspect pane output after delay)
- Build provider-specific status snapshots via the provider protocol's optional `build_status_snapshot()` method (replaces Codex-specific `_maybe_send_codex_status_snapshot`)
- Record forwarded commands to per-user command history for `/recall`
- Schedule periodic menu refresh (10-minute intervals via PTB job queue)

## Encapsulated Knowledge

This module owns all knowledge that no other module should have:

- **Three-tier menu caching strategy** — `_scoped_provider_menu` (LRU, per user+chat), `_chat_scoped_provider_menu` (LRU, per chat), `_global_provider_menu` (fallback); cache invalidation on provider change; Telegram `BotCommandScope` hierarchy
- **Command probing logic** — capture transcript byte offset + pane state before sending, wait for async delay, inspect pane output for error patterns (regex-based), report failures to user
- **Command validation** — resolve Telegram `/cmd` to provider's command metadata via `_get_provider_command_metadata()`; detect commands known in other providers but not in current; provide helpful error messages
- **Menu refresh scheduling** — 10-minute interval via PTB job queue, deduplication of concurrent refreshes
- **Status snapshot orchestration** — call `provider.build_status_snapshot()` (optional protocol method), format result for Telegram delivery; no provider-name checks

## Subdomain Classification

**Core** — Command forwarding is the primary user interaction path (user types `/cmd` → provider executes it). Changes frequently: every new provider capability, new command category, or probing improvement touches this module. The menu caching logic is particularly volatile as Telegram's command scope behavior has edge cases per chat type.

## Integration Contracts

### ← Bot Shell (depended on by)

- **Direction**: Bot Shell depends on Command Orchestration
- **Contract type**: Functional (handler registration)
- **What is shared**: Command forwarding handler function
- **Contract definition**: `forward_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE)` — registered as a `MessageHandler` in `create_bot()`

### → Provider Protocol (depends on)

- **Direction**: Command Orchestration depends on Provider Protocol
- **Contract type**: Contract (capability queries and optional methods)
- **What is shared**: Provider capabilities, command metadata, status snapshot
- **Contract definition**:
  - `get_provider_for_window(window_id) -> AgentProvider`
  - `provider.capabilities` — name, supported commands, feature flags
  - `provider.build_status_snapshot(window_id, transcript_path, offset) -> str | None` — optional, returns None if provider doesn't support it
  - `provider.get_command_metadata() -> dict` — command names and descriptions

### → Session State (depends on)

- **Direction**: Command Orchestration depends on Session State
- **Contract type**: Model (reads/writes window state)
- **What is shared**: Transcript path for probing, window context
- **Contract definition**: `get_window_state(window_id) -> WindowState` (reads `transcript_path`, `session_id`, `provider_name`), `send_to_window(window_id, text)` (via session_manager delegation to tmux)

### → Thread Router (depends on)

- **Direction**: Command Orchestration depends on Thread Router
- **Contract type**: Contract (resolution queries)
- **What is shared**: Window and chat identity
- **Contract definition**: `get_window_for_thread(user_id, thread_id) -> str | None`, `resolve_chat_id(user_id, thread_id) -> int | None`

### → Command History (depends on)

- **Direction**: Command Orchestration depends on Command History
- **Contract type**: Contract (recording interface)
- **What is shared**: Command text and context for recall
- **Contract definition**: `record_command(user_id, thread_id, command_text)`

### → Message Sender (depends on)

- **Direction**: Command Orchestration depends on Message Sender
- **Contract type**: Contract (delivery interface)
- **What is shared**: Error messages and status snapshots for user delivery
- **Contract definition**: `safe_reply(update, text, ...)`, `safe_send(bot, chat_id, thread_id, text, ...)`

## Change Vectors

These are reasonable future changes that would require ONLY this module to change:

- **Adding a new provider** — menu caching adapts automatically (capability-driven); no changes needed here
- **Changing command probe logic** (e.g., longer delay, different error patterns) — only probe functions change
- **Adding a new command category** (e.g., provider-specific settings commands) — only validation logic changes
- **Changing menu refresh interval** — only scheduling logic changes
- **Adding a status snapshot for a new provider** — that provider implements `build_status_snapshot()`; this module calls it uniformly
