# Topic Orchestration

## Functional Responsibilities

- Auto-create Telegram forum topics when new tmux windows are detected (via `SessionMonitor.new_window_callback`)
- Auto-detect provider for newly created windows using multi-step detection: process basename → pane title probe → runtime detection
- Adopt orphaned windows after bot restart (known in state but unbound to any topic)
- Rate-limit topic creation per chat to respect Telegram flood control
- Construct topic names from window metadata (window_name || cwd basename || window_id)
- Enumerate target chats for topic creation (active bindings → preserved chat IDs → config.group_id fallback)
- Bind newly created topics to the appropriate user

## Encapsulated Knowledge

This module owns all knowledge that no other module should have:

- **Topic name construction** — priority: explicit window_name → cwd directory basename → raw window_id; truncation rules for Telegram topic name limits
- **Chat enumeration strategy** — collect unique chat_ids from active thread bindings, fall back to preserved group_chat_ids (post-restart), fall back to `config.group_id` (cold start); deduplication across sources
- **Per-chat rate limiting** — `_topic_create_retry_until: dict[int, float]` maps chat_id → monotonic timestamp; backoff after Telegram `RetryAfter` errors; exponential retry with jitter
- **Orphan detection algorithm** — cross-reference persisted window_states against active thread_bindings to find known-but-unbound windows; delegate to sync module for adoption
- **Provider auto-detection for new windows** — multi-step: query `pane_current_command` → try `detect_provider_from_command()` (fast basename) → if unrecognized, try `detect_provider_from_runtime()` (ps-based TTY inspection) → persist result
- **User selection for binding** — find existing user in same chat with active bindings, fall back to first allowed user

## Subdomain Classification

**Supporting** — Topic orchestration changes less frequently than polling or command forwarding. It primarily responds to two events (new window detected, bot restart) with well-defined behavior. Changes occur when the topic creation UX evolves or when new provider detection methods are needed.

## Integration Contracts

### ← Bot Shell (depended on by)

- **Direction**: Bot Shell depends on Topic Orchestration
- **Contract type**: Functional (callback wiring)
- **What is shared**: New window event handler
- **Contract definition**: `handle_new_window(bot: Bot, event: NewWindowEvent) -> None` — called by SessionMonitor when a new tmux window is detected

### → Thread Router (depends on)

- **Direction**: Topic Orchestration depends on Thread Router
- **Contract type**: Contract (binding and enumeration)
- **What is shared**: Binding creation, chat enumeration
- **Contract definition**:
  - `bind_thread(user_id, thread_id, window_id, window_name)` — bind new topic to window
  - `iter_thread_bindings()` — enumerate active bindings for chat discovery
  - `set_group_chat_id(user_id, thread_id, chat_id)` — record chat ID for new topic

### → Session State (depends on)

- **Direction**: Topic Orchestration depends on Session State
- **Contract type**: Model (provider assignment, session map)
- **What is shared**: Provider persistence and session map polling
- **Contract definition**:
  - `set_window_provider(window_id, provider_name, cwd)` — persist detected provider
  - `wait_for_session_map_entry(window_id, timeout) -> bool` — wait for hook to register session
  - `get_window_state(window_id) -> WindowState` — read existing state

### → Provider Protocol (depends on)

- **Direction**: Topic Orchestration depends on Provider Protocol
- **Contract type**: Contract (detection functions)
- **What is shared**: Provider identity detection
- **Contract definition**:
  - `detect_provider_from_pane(pane_current_command, pane_tty, window_id) -> str | None`
  - `detect_provider_from_runtime(window_id) -> str | None`

### → Tmux Manager (depends on)

- **Direction**: Topic Orchestration depends on Tmux Manager
- **Contract type**: Contract (window metadata)
- **What is shared**: Window info for topic naming and detection
- **Contract definition**: `find_window_by_id(window_id) -> TmuxWindow | None`

### → Sync Command (depends on)

- **Direction**: Topic Orchestration depends on Sync Command (for orphan adoption)
- **Contract type**: Functional (delegation)
- **What is shared**: Orphaned window list
- **Contract definition**: `_adopt_orphaned_windows(bot, orphaned_windows)` — adopted from existing sync module

## Change Vectors

These are reasonable future changes that would require ONLY this module to change:

- **Changing topic naming convention** (e.g., adding provider prefix to topic name) — only name construction logic changes
- **Adding a new provider detection method** (e.g., environment variable inspection) — only detection sequence changes
- **Changing rate limiting strategy** (e.g., per-user instead of per-chat) — only rate limiting state changes
- **Changing orphan adoption behavior** (e.g., auto-delete instead of adopt) — only adoption logic changes
- **Supporting topic creation in multiple groups simultaneously** — only chat enumeration strategy changes
