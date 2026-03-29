# Session State

## Functional Responsibilities

- Manage per-window state via `WindowState` dataclass (session_id, cwd, provider_name, transcript_path, preferences)
- Coordinate persistence of all state to state.json (debounced writes via `StatePersistence`)
- Sync window states from session_map.json (hook-generated data)
- Manage user preferences: notification mode (all/errors_only/muted), approval mode (normal/yolo), batch mode (batched/verbose)
- Manage user directory favorites (starred directories, MRU list)
- Track user read positions (byte offsets per window per user)
- Resolve sessions for transcript reading (session_id → JSONL file)
- Register hookless provider sessions (Codex, Gemini — no hook, synthetic session_map entries)
- Audit and prune stale state on startup (orphaned entries, dead windows)
- Expose narrow Protocol interfaces for consumers (`WindowStateStore`, `UserPreferences`, `SessionResolver`)

## Encapsulated Knowledge

This module owns all knowledge that no other module should have:

- **WindowState serialization** — `to_dict()` / `from_dict()` format, field defaults, migration of legacy formats
- **state.json schema** — root structure with nested dicts, int→str key conversion for JSON compatibility, version migration
- **Debounced persistence** — `StatePersistence` timing (0.5s delay), flush-on-shutdown guarantee, atomic writes
- **Session map sync algorithm** — merge hook-generated session_map.json entries into `window_states`, handle session_id changes (post-`/clear`), detect deletions
- **Stale ID resolution** — on tmux server restart, window IDs reset; `resolve_stale_ids()` matches persisted display names against live windows to re-map
- **Session resolution** — locate JSONL transcript from session_id + cwd, handle `transcript_path` shortcut for hookless providers
- **Hookless session registration** — write synthetic session_map entries with file locking for Codex/Gemini/Shell providers
- **User offset tracking** — per-user-per-window byte offset structure, pruning of stale offsets

## Subdomain Classification

**Core** — The session state module is the system's persistence backbone. Every new per-window or per-user setting gravitates here because it's the only persistent state abstraction. High volatility: new features regularly add WindowState fields, preference types, or audit rules.

## Integration Contracts

### ↔ Thread Router (bidirectional coordination)

- **Direction**: Bidirectional — Session State orchestrates persistence
- **Contract type**: Functional (shared persistence lifecycle)
- **What is shared**: Serialized state for state.json save/load
- **Contract definition**:
  - Session State calls: `thread_router.to_dict()` on save, `thread_router.from_dict(data)` on load
  - Thread Router calls: `session_state.schedule_save()` after binding mutations
  - Both participate in the same state.json file

### ← Polling Subsystem (depended on by, via WindowStateStore)

- **Direction**: Polling depends on Session State
- **Contract type**: Model (reads WindowState fields)
- **What is shared**: Window configuration and session identity
- **Contract definition** (`WindowStateStore` protocol):
  ```python
  class WindowStateStore(Protocol):
      def get_window_state(self, window_id: str) -> WindowState: ...
      def get_display_name(self, window_id: str) -> str: ...
      def get_session_id_for_window(self, window_id: str) -> str: ...
      def clear_window_session(self, window_id: str) -> None: ...
  ```

### ← Command Orchestration (depended on by)

- **Direction**: Commands depend on Session State
- **Contract type**: Model (reads/writes window state for probing)
- **What is shared**: Transcript path, session_id for command probing context
- **Contract definition**: `get_window_state(window_id) -> WindowState`, `set_window_provider(window_id, provider_name, cwd)`

### ← Topic Orchestration (depended on by)

- **Direction**: Topic creation depends on Session State
- **Contract type**: Model (writes provider, reads session map)
- **What is shared**: Provider assignment and session map polling
- **Contract definition**: `set_window_provider(window_id, provider_name, cwd)`, `wait_for_session_map_entry(window_id, timeout) -> bool`

### ← Multiple handlers (depended on by, via UserPreferences)

- **Direction**: Handlers depend on Session State
- **Contract type**: Contract (narrow preference interface)
- **What is shared**: User preference getters/setters
- **Contract definition** (`UserPreferences` protocol):
  ```python
  class UserPreferences(Protocol):
      def get_notification_mode(self, window_id: str) -> str: ...
      def set_notification_mode(self, window_id: str, mode: str) -> None: ...
      def cycle_notification_mode(self, window_id: str) -> str: ...
      def get_approval_mode(self, window_id: str) -> str: ...
      def set_window_approval_mode(self, window_id: str, mode: str) -> None: ...
      def get_batch_mode(self, window_id: str) -> str: ...
      def cycle_batch_mode(self, window_id: str) -> str: ...
  ```

### ← History / transcript consumers (depended on by, via SessionResolver)

- **Direction**: Consumers depend on Session State
- **Contract type**: Contract (transcript access)
- **What is shared**: Session resolution and message reading
- **Contract definition** (`SessionResolver` protocol):
  ```python
  class SessionResolver(Protocol):
      def resolve_session_for_window(self, window_id: str) -> ClaudeSession | None: ...
      def get_recent_messages(self, window_id: str, start_byte: int = 0, end_byte: int | None = None) -> tuple[list, int]: ...
  ```

### ← Bot Shell (depended on by)

- **Direction**: Bot depends on Session State for lifecycle
- **Contract type**: Contract (lifecycle operations)
- **What is shared**: Startup resolution and shutdown flush
- **Contract definition**: `resolve_stale_ids()`, `flush_state()`, `load_session_map()`, `prune_stale_state(live_window_ids)`

## Change Vectors

These are reasonable future changes that would require ONLY this module to change:

- **Adding a new per-window setting** (e.g., auto-screenshot preference) — add field to `WindowState`, update serialization; consumers access via new protocol method
- **Changing persistence backend** (JSON → SQLite) — rewrite `StatePersistence` internals; consumers use same protocol interfaces
- **Adding a new user preference type** — add getter/setter/cycle methods, extend `UserPreferences` protocol
- **Changing session_map.json format** — only `load_session_map()` and `_sync_window_from_session_map()` change
- **Adding state migration logic** — only `_load_state()` changes
