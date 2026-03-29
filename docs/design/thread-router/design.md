# Thread Router

## Functional Responsibilities

- Bind Telegram topics (user_id + thread_id) to tmux windows (window_id) bidirectionally
- Resolve window_id from thread context (outbound: user message → tmux window)
- Resolve thread_id from window context (inbound: tmux output → Telegram topic)
- Manage group chat IDs for multi-group forum topic routing
- Maintain display names for windows (user-facing labels, synced from tmux)
- Iterate all active bindings for polling and message delivery
- Maintain O(1) reverse index (`_window_to_thread`) for inbound message routing

## Encapsulated Knowledge

This module owns all knowledge that no other module should have:

- **Binding data structures** — `thread_bindings: dict[int, dict[int, str]]` (user_id → {thread_id → window_id}), internal layout and access patterns
- **Reverse index maintenance** — `_window_to_thread: dict[tuple[int, str], int]` rebuild algorithm, consistency invariants between forward and reverse maps
- **Group chat ID composite key format** — `"user_id:thread_id"` string key encoding for `group_chat_ids` dict
- **Display name lifecycle** — set on bind, updated on tmux window rename via `sync_display_names()`, preserved across restarts
- **Chat ID resolution strategy** — fallback chain: thread-specific chat_id → user's first known chat_id → None
- **Serialization contract with Session State** — `to_dict()` / `from_dict()` for state.json persistence (int→str key conversion for JSON compatibility)

## Subdomain Classification

**Core** — Thread routing is the central nervous system of the bot. Every message flow (inbound and outbound) passes through this module. Changes to the binding model (e.g., per-pane bindings, multi-user topics) would affect the routing core. High volatility justified by frequent feature additions that touch binding logic.

## Integration Contracts

### ↔ Session State (bidirectional coordination)

- **Direction**: Bidirectional — Session State coordinates persistence
- **Contract type**: Functional (shared persistence lifecycle)
- **What is shared**: Serialized binding state for state.json
- **Contract definition**:
  - Thread Router exposes: `to_dict() -> dict`, `from_dict(data: dict) -> None`
  - Session State calls these during save/load cycles
  - Thread Router calls `session_state.schedule_save()` after mutations

### ← Polling Subsystem (depended on by)

- **Direction**: Polling Subsystem depends on Thread Router
- **Contract type**: Contract (iteration interface)
- **What is shared**: Active binding tuples for poll enumeration
- **Contract definition**: `iter_thread_bindings() -> Iterator[tuple[int, int, str]]`

### ← Bot Shell (depended on by)

- **Direction**: Bot Shell depends on Thread Router
- **Contract type**: Contract (lifecycle operations)
- **What is shared**: Binding creation and destruction
- **Contract definition**: `bind_thread(user_id, thread_id, window_id, window_name)`, `unbind_thread(user_id, thread_id)`

### ← Command Orchestration (depended on by)

- **Direction**: Command Orchestration depends on Thread Router
- **Contract type**: Contract (resolution queries)
- **What is shared**: Window and chat identity resolution
- **Contract definition**: `resolve_chat_id(user_id, thread_id) -> int | None`, `get_window_for_thread(user_id, thread_id) -> str | None`

### ← Message Queue (depended on by)

- **Direction**: Message Queue depends on Thread Router
- **Contract type**: Contract (delivery resolution)
- **What is shared**: Chat ID for message delivery
- **Contract definition**: `resolve_chat_id(user_id, thread_id) -> int | None`

### ← Topic Orchestration (depended on by)

- **Direction**: Topic Orchestration depends on Thread Router
- **Contract type**: Contract (binding and enumeration)
- **What is shared**: Binding creation, chat enumeration for topic creation
- **Contract definition**: `bind_thread(...)`, `iter_thread_bindings()`, `set_group_chat_id(...)`

### ← Multiple handlers (depended on by)

- **Direction**: Handler modules depend on Thread Router
- **Contract type**: Contract (query interface)
- **What is shared**: Window/thread resolution for handler context
- **Contract definition**: `get_window_for_thread(user_id, thread_id) -> str | None`, `get_thread_for_window(user_id, window_id) -> int | None`, `get_display_name(window_id) -> str`, `get_all_thread_windows(user_id) -> dict[int, str]`

## Change Vectors

These are reasonable future changes that would require ONLY this module to change:

- **Adding a new binding dimension** (e.g., per-pane bindings for agent teams) — restructure `thread_bindings` and `_window_to_thread`; consumers use the same `get_window_for_thread()` interface
- **Changing chat ID resolution strategy** (e.g., per-topic chat IDs instead of composite keys) — internal data structure change, `resolve_chat_id()` signature unchanged
- **Adding binding metadata** (e.g., bind timestamp, creator user_id) — enrich internal structures; existing query methods unchanged
- **Supporting multi-user topics** (multiple users bound to same thread) — internal binding model change; iteration interface may need extension
- **Changing display name sync strategy** — only `sync_display_names()` changes
