# Modularity Refactoring Round 2

## Overview

Implement architectural changes from the [third modularity review](../reviews/modularity-review.md) (2026-03-30) and the updated [design documents](../design/). Addresses 3 Significant and 1 Minor coupling imbalances:

- **E2-E5**: Fix 4 per-topic state cleanup gaps (leaked `_bash_capture_tasks`, `_send_cooldowns`, `_topic_create_retry_until`, `_disabled_chats`)
- **D1**: Promote mailbox private helpers to public API — `_sanitize_dir_name` → `sanitize_dir_name`, `_validate_no_traversal` → `validate_no_traversal`
- **D2**: Promote 3 private functions used across module boundaries — `_resolve_topic`, `_collect_target_chats`, `_create_topic_in_chat`
- **D3**: Replace shared mutable `_pending_requests` dict with public accessor API
- **C1**: Extract `msg_delivery.py` to break `msg_broker ↔ msg_telegram` circular dependency
- **C2**: Add public methods to `TerminalStatusStrategy`; move threshold logic inside strategies
- **B1**: Extract `UserPreferences` from `SessionManager` (starred/MRU/offsets → standalone class)
- **B2**: Narrow `SessionManager` — remove 9 dead pass-throughs, add Protocol interfaces, add `export_window_info()`
- **A1**: Create `TopicStateRegistry` with self-registration pattern, rewrite `cleanup.py`

Each step is independently mergeable. Steps ordered by increasing blast radius.

## Context (from design session)

- Design documents: `docs/design/*/design.md` (18 modules, 8 new this session)
- Test specifications: `docs/design/*/tests.md` (18 modules)
- Architecture overview: `docs/design/architecture.md` (rewritten this session)
- Modularity review: `docs/reviews/modularity-review.md` (third review, 2026-03-30)
- Previous refactoring: `docs/plans/completed/2026-03-29-modularity-refactoring.md`
- Codebase passes `make check` — 2795 tests pass, 0 lint errors
- Key files and line counts: `session.py` (1526), `polling_coordinator.py` (970), `msg_broker.py` (507), `cleanup.py` (136), `polling_strategies.py` (537), `topic_orchestration.py` (218), `msg_telegram.py` (367), `msg_spawn.py` (244), `spawn_request.py` (219), `mailbox.py` (577)

## Development Approach

- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes
- **CRITICAL: all tests must pass before starting next task** — `make check`
- **CRITICAL: update this plan file when scope changes during implementation**
- Run `make check` after each step (fmt + lint + typecheck + test)

## Testing Strategy

- **Unit tests**: Update existing test files when interfaces change; add new tests for new modules
- **Integration tests**: Verify state persistence roundtrip after `UserPreferences` extraction
- Run `make test` after every task, `make check` after every step

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix

## Implementation Steps

---

### Step E: Fix Cleanup Gaps (~6 files, lowest risk)

Design docs: `docs/design/topic-state-registry/design.md` (motivation section)

#### Task 1: Add missing cleanup functions and wire into cleanup.py

4 per-topic state dicts are not cleaned when topics close. Each fix is 2-4 lines.

- [x] verify `clear_shell_pending()` in `shell_commands.py` already clears `_generation_counter` (line 108) — if confirmed, no change needed; if not, extend it
- [x] `handlers/text_handler.py` (line 57): rename `_cancel_bash_capture` → `cancel_bash_capture`; update internal call at line 323
- [x] `handlers/interactive_ui.py`: add `clear_send_cooldowns(user_id: int, thread_id: int) -> None` — `_send_cooldowns.pop((user_id, thread_id), None)` (the `_send_cooldowns` dict is at line 60, keyed by `(user_id, thread_id)`)
- [x] `handlers/topic_orchestration.py`: add `clear_topic_create_retry(chat_id: int) -> None` — `_topic_create_retry_until.pop(chat_id, None)` (dict at line 36, keyed by `chat_id`)
- [x] `handlers/topic_emoji.py`: add `clear_disabled_chat(chat_id: int) -> None` — `_disabled_chats.discard(chat_id)` (set at line 68); add size guard: `if len(_disabled_chats) > 1000: _disabled_chats.clear()`
- [x] `handlers/cleanup.py`: add 4 lazy imports + calls inside `clear_topic_state()`:
  - `from .text_handler import cancel_bash_capture` → `cancel_bash_capture(user_id, thread_id)`
  - `from .interactive_ui import clear_send_cooldowns` → `clear_send_cooldowns(user_id, thread_id)`
  - `from .topic_orchestration import clear_topic_create_retry` → `clear_topic_create_retry(chat_id)` (in the `if chat_id:` block)
  - `from .topic_emoji import clear_disabled_chat` → `clear_disabled_chat(chat_id)` (in the `if chat_id:` block)
- [x] write tests: each new function called with valid key → state cleared; called with missing key → no error
- [x] `make check` — must pass

---

### Step D1: Mailbox Public API (~3 files)

Design doc: `docs/design/mailbox/design.md`

#### Task 2: Promote private mailbox helpers to public

`msg_broker.py` line 224 imports `_sanitize_dir_name` and `_validate_no_traversal` from `mailbox.py` — private functions encoding internal directory structure.

- [x] `mailbox.py` (line 95): rename `_sanitize_dir_name` → `sanitize_dir_name`; add backward compat alias `_sanitize_dir_name = sanitize_dir_name`
- [x] `mailbox.py` (line 89): rename `_validate_no_traversal` → `validate_no_traversal`; add alias `_validate_no_traversal = validate_no_traversal`
- [x] `handlers/msg_broker.py` (line 224): change import from `_sanitize_dir_name, _validate_no_traversal` → `sanitize_dir_name, validate_no_traversal`
- [x] `tests/ccgram/test_mailbox.py`: update import of `_sanitize_dir_name` → `sanitize_dir_name`
- [x] write/update tests: `test_sanitize_dir_name_handles_colons`, `test_validate_no_traversal_rejects_dotdot`
- [x] `make check` — must pass

---

### Step D2: Promote Private Functions to Public (~4 files)

Design doc: `docs/design/messaging-telegram/design.md`, `docs/design/topic-orchestration/design.md`

#### Task 3: Rename 3 underscore-prefixed functions used across module boundaries

These functions are imported by `msg_spawn.py` despite their private naming. The underscore is a lie — they are de facto public APIs.

- [x] `handlers/msg_telegram.py` (line 74): rename `_resolve_topic` → `resolve_topic`; update all 5 internal call sites within the same file (search for `_resolve_topic(` in the file)
- [x] `handlers/topic_orchestration.py` (line 82): rename `_collect_target_chats` → `collect_target_chats`; update 2 internal call sites
- [x] `handlers/topic_orchestration.py` (line 132): rename `_create_topic_in_chat` → `create_topic_in_chat`; update internal calls
- [x] `handlers/msg_spawn.py`: update 3 imports:
  - line 140/184: `from .msg_telegram import _resolve_topic` → `resolve_topic`
  - line 185: `from .topic_orchestration import _collect_target_chats, _create_topic_in_chat` → drop underscores
- [x] `tests/ccgram/handlers/test_topic_orchestration.py`: update ~9 occurrences of `_collect_target_chats` → `collect_target_chats` including patch strings like `"ccgram.handlers.topic_orchestration._collect_target_chats"`
- [x] `make check` — must pass

---

### Step D3: Spawn Request Accessor API (~3 files)

Design doc: `docs/design/spawn-request/design.md`

#### Task 4: Replace direct `_pending_requests` dict access with public functions

Three modules (`spawn_request.py`, `msg_broker.py`, `msg_spawn.py`) co-own `_pending_requests` dict with `.pop()` calls. Replace with accessor functions.

- [x] `spawn_request.py`: add 4 accessor functions after `_pending_requests` definition (line 69):

  ```python
  def get_pending(request_id: str) -> SpawnRequest | None:
      return _pending_requests.get(request_id)

  def pop_pending(request_id: str) -> SpawnRequest | None:
      return _pending_requests.pop(request_id, None)

  def iter_pending() -> Iterator[tuple[str, SpawnRequest]]:
      yield from _pending_requests.items()

  def register_pending(req: SpawnRequest) -> None:
      _pending_requests[req.id] = req
  ```

- [x] `spawn_request.py`: promote `_spawns_dir` → `spawns_dir`; keep `_spawns_dir = spawns_dir` alias
- [x] `handlers/msg_broker.py` (lines 477, 493, 503, 506): replace `from ..spawn_request import _pending_requests, scan_spawn_requests, _spawns_dir` → `from ..spawn_request import pop_pending, get_pending, scan_spawn_requests, spawns_dir`; update `.pop()` calls to `pop_pending()`
- [x] `handlers/msg_spawn.py` (lines 30-31): replace `from ..spawn_request import _pending_requests, _spawns_dir` → `from ..spawn_request import pop_pending, register_pending, spawns_dir`; update `.pop()` calls (lines 51, 123) to `pop_pending()`; update `_spawns_dir()` calls (lines 59, 68, 86, 127) to `spawns_dir()`
- [x] write tests: `test_get_pending_returns_request`, `test_pop_pending_removes`, `test_pop_pending_missing_returns_none`, `test_iter_pending_yields_all`, `test_register_pending_stores`
- [x] `make check` — must pass

---

### Step C1: Extract msg_delivery.py (~5 files)

Design doc: `docs/design/message-delivery/design.md`

#### Task 5: Break msg_broker ↔ msg_telegram circular dependency

`msg_broker.py` owns `MessageDeliveryStrategy` and `delivery_strategy` singleton. `msg_telegram.py` (line 338) does `from .msg_broker import delivery_strategy` — creating a circular dependency (both defer imports to function scope). Extracting the strategy to a standalone module breaks the cycle.

- [x] create `handlers/msg_delivery.py` — move from `msg_broker.py`:
  - `DeliveryState` dataclass (line ~30-50 of msg_broker)
  - `MessageDeliveryStrategy` class with all methods (`check_rate_limit`, `check_loop`, `pause_peer`, `allow_more`, `record_delivery`, `record_exchange`, `is_paused`, `get_state`, `_states` dict)
  - `delivery_strategy = MessageDeliveryStrategy()` singleton
  - `clear_delivery_state(qualified_id)` function
  - `reset_delivery_state()` function (for testing)
- [x] `handlers/msg_broker.py`: remove moved code; add re-exports for backward compatibility:
  ```python
  from .msg_delivery import delivery_strategy, DeliveryState, clear_delivery_state, reset_delivery_state
  ```
- [x] `handlers/msg_telegram.py` (line 338): change `from .msg_broker import delivery_strategy` → `from .msg_delivery import delivery_strategy`
- [x] `handlers/cleanup.py`: change `from .msg_broker import clear_delivery_state` → `from .msg_delivery import clear_delivery_state`
- [x] verify singleton identity in test: `assert msg_broker.delivery_strategy is msg_delivery.delivery_strategy`
- [x] write tests for `msg_delivery.py`: `test_delivery_state_lifecycle`, `test_clear_removes_entry`, `test_reset_clears_all`
- [x] `make check` — must pass

---

### Step C2: Polling Strategy Encapsulation (~3 files)

Design doc: `docs/design/polling-subsystem/design.md`

#### Task 6: Add public methods to TerminalStatusStrategy

`TopicLifecycleStrategy` directly accesses `self._terminal._states` (private dict) in 5 methods. `polling_coordinator.py` imports 5 private `_UPPERCASE` constants.

- [x] `handlers/polling_strategies.py` — add to `TerminalStatusStrategy`:

  ```python
  def reset_probe_failures(self, window_id: str) -> None:
      ws = self._states.get(window_id)
      if ws: ws.probe_failures = 0

  def clear_seen_status(self, window_id: str) -> None:
      ws = self._states.get(window_id)
      if ws: ws.has_seen_status = False; ws.startup_time = None

  def set_unbound_timer(self, window_id: str, ts: float) -> None:
      ws = self.get_state(window_id)
      ws.unbound_timer = ts

  def clear_unbound_timer(self, window_id: str) -> None:
      ws = self._states.get(window_id)
      if ws: ws.unbound_timer = None

  def reset_all_probe_failures(self) -> None:
      for ws in self._states.values(): ws.probe_failures = 0

  def reset_all_seen_status(self) -> None:
      for ws in self._states.values(): ws.has_seen_status = False; ws.startup_time = None

  def reset_all_unbound_timers(self) -> None:
      for ws in self._states.values(): ws.unbound_timer = None
  ```

- [x] `handlers/polling_strategies.py` — update `TopicLifecycleStrategy` to use public methods:
  - line 381 `reset_autoclose_state`: replace `for ws in self._terminal._states.values()` → `self._terminal.reset_all_unbound_timers()`
  - line 395 `clear_probe_failures`: replace `ws = self._terminal._states.get(window_id)` → `self._terminal.reset_probe_failures(window_id)`
  - line 400 `reset_probe_failures_state`: replace direct iteration → `self._terminal.reset_all_probe_failures()`
  - line 417 `clear_seen_status`: replace direct access → `self._terminal.clear_seen_status(window_id)`
  - line 423 `reset_seen_status_state`: replace iteration → `self._terminal.reset_all_seen_status()`

- [x] `handlers/polling_strategies.py`: promote 5 private constants to public (keep private aliases):
  - `_ACTIVITY_THRESHOLD` → `ACTIVITY_THRESHOLD`
  - `_MAX_PROBE_FAILURES` → `MAX_PROBE_FAILURES`
  - `_PANE_COUNT_TTL` → `PANE_COUNT_TTL`
  - `_STARTUP_TIMEOUT` → `STARTUP_TIMEOUT`
  - `_TYPING_INTERVAL` → `TYPING_INTERVAL`

- [x] `handlers/polling_coordinator.py` (lines 59-63): update imports to use public constant names
- [x] `handlers/polling_coordinator.py`: replace ~12 direct `WindowPollState` field assignments with strategy method calls — search for `.probe_failures`, `.has_seen_status`, `.startup_time`, `.unbound_timer` assignments
- [x] `tests/ccgram/handlers/test_polling_strategies.py`: update private constant import if present (line 15: `_MAX_PROBE_FAILURES` → `MAX_PROBE_FAILURES`)
- [x] write tests for new `TerminalStatusStrategy` methods: `test_reset_probe_failures`, `test_clear_seen_status`, `test_set_clear_unbound_timer`
- [x] `make check` — must pass

**Note**: `test_status_polling.py` and `test_polling_coordinator.py` use `terminal_strategy._states` directly in fixtures. This is acceptable — the `_states` dict remains accessible; the encapsulation adds public mutation methods but does not hide the dict.

---

### Step B1: User Preferences Extraction (~8 files)

Design doc: `docs/design/user-preferences/design.md`

#### Task 7: Create UserPreferences class and extract from SessionManager

`SessionManager` owns user directory favorites (starred/MRU) and per-user read offsets — 6 methods used by only 2 consumers (`directory_browser.py`, `directory_callbacks.py`). Extract to standalone class following the `ThreadRouter` pattern.

- [x] create `src/ccgram/user_preferences.py` with `UserPreferences` class:
  - Data: `_user_dir_favorites: dict`, `_user_window_offsets: dict`
  - Methods: `get_user_starred(user_id)`, `toggle_user_star(user_id, path)`, `get_user_mru(user_id)`, `update_user_mru(user_id, path)`, `get_user_window_offset(user_id, window_id)`, `update_user_window_offset(user_id, window_id, offset)`
  - Persistence: `to_dict() -> dict`, `from_dict(data: dict) -> None`
  - Constructor takes `schedule_save: Callable` callback (same pattern as `ThreadRouter.__init__`)
- [x] create module-level singleton: `user_preferences = UserPreferences()`
- [x] `session.py`: remove `user_dir_favorites` and `_user_window_offsets` fields from `SessionManager`
- [x] `session.py`: remove 6 methods: `get_user_starred`, `toggle_user_star`, `get_user_mru`, `update_user_mru`, `get_user_window_offset`, `update_user_window_offset`
- [x] `session.py` `__init__`: wire `user_preferences.set_schedule_save(self._schedule_save)` (same as `thread_router`)
- [x] `session.py` `_save_state()`: add `user_preferences.to_dict()` to saved data
- [x] `session.py` `_load_state()`: call `user_preferences.from_dict(data)` during load
- [x] `handlers/directory_browser.py`: replace `session_manager.get_user_starred(user_id)` → `user_preferences.get_user_starred(user_id)` (and MRU equivalents)
- [x] `handlers/directory_callbacks.py`: same import updates
- [x] `handlers/history.py`: replace `session_manager.get_user_window_offset` / `update_user_window_offset` → `user_preferences.*`
- [x] `bot.py`: update any direct preference calls (if any)
- [x] update `tests/ccgram/test_session_favorites.py`: test `UserPreferences` directly (import `user_preferences` instead of `session_manager`)
- [x] verify `tests/integration/test_state_roundtrip.py` passes — state.json format must remain unchanged (`user_dir_favorites` key preserved in the same location)
- [x] write tests: `test_starred_toggle`, `test_mru_eviction`, `test_offset_roundtrip`, `test_to_dict_from_dict`
- [x] `make check` — must pass

---

### Step B2: SessionManager Narrowing (~2 files)

Design doc: `docs/design/session-state/design.md`

#### Task 8: Remove dead pass-throughs, add Protocol interfaces, add export_window_info

With `UserPreferences` extracted (Task 7), `SessionManager` drops from ~50 to ~35 public methods. Now remove 9 more dead pass-throughs and add typed contracts.

- [x] `session.py`: delete 9 pass-through methods (lines ~1381-1440) — confirmed no callers:
      `bind_thread`, `unbind_thread`, `get_window_for_thread`, `get_thread_for_window`, `get_all_thread_windows`, `resolve_window_for_thread`, `iter_thread_bindings`, `set_group_chat_id`, `resolve_chat_id`
- [x] verify no callers: `grep -rn "session_manager\.\(bind_thread\|unbind_thread\|get_window_for_thread\|get_thread_for_window\|get_all_thread_windows\|resolve_window_for_thread\|iter_thread_bindings\|set_group_chat_id\|resolve_chat_id\)" src/`
- [x] `session.py`: add `export_window_info() -> dict[str, WindowInfo]` as a module-level function (not a method — CLI needs it without bot token):
  ```python
  def export_window_info() -> dict[str, WindowInfo]:
      """CLI-safe snapshot of window states. Reads state.json from disk."""
      from .msg_discovery import WindowInfo
      state = load_state_from_disk()  # reuse _load_state logic
      return {wid: WindowInfo(cwd=ws.get("cwd", ""), ...) for wid, ws in state.get("window_states", {}).items()}
  ```
- [x] `session.py`: define Protocol interfaces (additive — no consumer changes required):

  ```python
  class WindowStateStore(Protocol):
      def get_window_state(self, window_id: str) -> WindowState: ...
      def get_display_name(self, window_id: str) -> str: ...
      def get_session_id_for_window(self, window_id: str) -> str | None: ...
      def clear_window_session(self, window_id: str) -> None: ...

  class SessionIO(Protocol):
      async def send_to_window(self, window_id: str, text: str) -> None: ...
      def resolve_session_for_window(self, window_id: str) -> ClaudeSession | None: ...
      def get_recent_messages(self, window_id: str, start_byte: int = 0, end_byte: int | None = None) -> tuple[list, int]: ...

  class WindowModeConfig(Protocol):
      def get_approval_mode(self, window_id: str) -> str: ...
      def set_window_approval_mode(self, window_id: str, mode: str) -> None: ...
      def get_notification_mode(self, window_id: str) -> str: ...
      def set_notification_mode(self, window_id: str, mode: str) -> None: ...
      def cycle_notification_mode(self, window_id: str) -> str: ...
      def get_batch_mode(self, window_id: str) -> str: ...
      def set_batch_mode(self, window_id: str, mode: str) -> None: ...
      def cycle_batch_mode(self, window_id: str) -> str: ...
  ```

- [x] `msg_cmd.py`: replace `_load_window_states()` function (lines 122-142) with call to `from ..session import export_window_info`; remove hardcoded `state.json` field names
- [x] write tests: `test_export_window_info_returns_dict`, `test_export_window_info_empty_state`, `test_protocols_are_satisfied` (verify `SessionManager` satisfies all 3 protocols via `isinstance` or `typing.runtime_checkable`)
- [x] `make check` — must pass

---

### Step A1: Topic State Registry (~16 files, highest blast radius)

Design doc: `docs/design/topic-state-registry/design.md`

#### Task 9a: Create registry and dual-path cleanup (additive)

Create the registry module. Keep existing explicit cleanup calls in `cleanup.py` AND add registry dispatch — dual-path ensures nothing breaks during migration.

- [x] create `handlers/topic_state_registry.py`:

  ```python
  class TopicStateRegistry:
      _topic_cleanups: list[Callable[[int, int], None]]     # (user_id, thread_id)
      _window_cleanups: list[Callable[[str], None]]          # window_id
      _qualified_cleanups: list[Callable[[str], None]]       # qualified_id
      _chat_cleanups: list[Callable[[int, int], None]]       # (chat_id, thread_id) — legacy

      def register(self, scope: str) -> Callable: ...
      def clear_topic(self, user_id, thread_id) -> None: ...
      def clear_window(self, window_id) -> None: ...
      def clear_qualified(self, qualified_id) -> None: ...
      def clear_chat(self, chat_id, thread_id) -> None: ...
      def clear_all(self, user_id, thread_id, window_id=None, qualified_id=None, chat_id=None) -> None: ...
  ```

- [x] create module singleton: `topic_state = TopicStateRegistry()`
- [x] `cleanup.py`: import `topic_state`; call `topic_state.clear_all(...)` at the END of `clear_topic_state()` (after existing explicit calls — dual-path)
- [x] write tests: `test_register_and_clear_topic`, `test_register_and_clear_window`, `test_dedup_prevents_double_call`, `test_failing_cleanup_doesnt_block_others`, `test_clear_all_dispatches_all_scopes`
- [x] `make check` — must pass

#### Task 9b: Migrate handlers to self-register (batch 1 — topic-scoped)

Migrate modules that use `(user_id, thread_id)` key. After each migration, remove the explicit call from `cleanup.py`.

- [x] `handlers/message_queue.py`: register `clear_tool_msg_ids_for_topic`, `clear_status_msg_info`, `clear_batch_for_topic` with `@topic_state.register("topic")`; remove 3 explicit calls from `cleanup.py`
- [x] `handlers/interactive_ui.py`: register `clear_send_cooldowns` with `@topic_state.register("topic")`; `clear_interactive_msg` is async+bot so stays as explicit call in cleanup.py
- [x] `handlers/command_history.py`: register `clear_history` with `@topic_state.register("topic")`; remove from `cleanup.py`
- [x] `handlers/text_handler.py`: register `cancel_bash_capture` with `@topic_state.register("topic")`; remove from `cleanup.py`
- [x] `make check` — must pass

#### Task 9c: Migrate handlers to self-register (batch 2 — window-scoped)

Migrate modules that use `window_id` key.

- [x] `handlers/polling_strategies.py`: register `clear_window_poll_state`, `clear_pane_alerts` with `@topic_state.register("window")`; register `clear_dead_notification`, `clear_topic_poll_state` with `@topic_state.register("topic")`; remove 4 from `cleanup.py`
- [x] `handlers/shell_capture.py`: register `clear_shell_monitor_state` with `@topic_state.register("window")`; remove from `cleanup.py`
- [x] `handlers/hook_events.py`: register `clear_subagents` with `@topic_state.register("window")`; remove from `cleanup.py`
- [x] `providers/process_detection.py`: register `clear_detection_cache` with `@topic_state.register("window")`; remove from `cleanup.py`
- [x] `tmux_manager.py`: register `clear_vim_state` with `@topic_state.register("window")`; remove from `cleanup.py`
- [x] `make check` — must pass

#### Task 9d: Migrate handlers to self-register (batch 3 — chat + qualified scoped)

- [x] `handlers/topic_emoji.py`: register `clear_topic_emoji_state` with `@topic_state.register("chat")`; register `clear_disabled_chat` with `@topic_state.register("chat")`; remove from `cleanup.py`
- [x] `handlers/shell_commands.py`: register `clear_shell_pending` with `@topic_state.register("chat")`; remove from `cleanup.py`
- [x] `handlers/msg_delivery.py`: register `clear_delivery_state` with `@topic_state.register("qualified")`; remove from `cleanup.py`
- [x] `spawn_request.py`: register `clear_spawn_state` with `@topic_state.register("qualified")`; remove from `cleanup.py`
- [x] `msg_discovery.py`: register `clear_declared` with `@topic_state.register("qualified")`; remove from `cleanup.py`
- [x] `handlers/topic_orchestration.py`: register `clear_topic_create_retry` with `@topic_state.register("chat")`; remove from `cleanup.py`
- [x] `make check` — must pass

#### Task 9e: Finalize — remove old explicit calls from cleanup.py

- [x] `cleanup.py`: remove ALL old explicit cleanup calls and lazy imports; function becomes ~20 lines:

  ```python
  async def clear_topic_state(user_id, thread_id, bot=None, user_data=None, window_id=None):
      from .topic_state_registry import topic_state
      from ..thread_router import thread_router
      from ..utils import tmux_session_name

      chat_id = thread_router.resolve_chat_id(user_id, thread_id)
      qualified_id = f"{tmux_session_name()}:{window_id}" if window_id else None

      topic_state.clear_all(user_id, thread_id, window_id=window_id, qualified_id=qualified_id, chat_id=chat_id)
      # ... keep bot-specific async cleanup (status msg edit, user_data clear) that can't be registered
  ```

- [x] ensure `topic_state_registry` is imported by `callback_registry.load_handlers()` (or explicitly imported in `bot.py`) so all registrations trigger at startup
- [x] write integration test: register 5 mock cleanups across all scopes → `clear_all()` → verify all 5 called with correct arguments
- [x] `make check` — must pass

---

### Task 10: Verify acceptance criteria

- [x] verify all 4 review issues addressed:
  - No private imports across messaging modules (`_sanitize_dir_name`, `_pending_requests`, `_resolve_topic`, `_collect_target_chats`, `_create_topic_in_chat` all promoted or replaced)
  - `msg_broker ↔ msg_telegram` circular dependency broken (delivery_strategy in `msg_delivery.py`)
  - SessionManager narrowed: 9 pass-throughs removed, UserPreferences extracted, Protocols defined
  - cleanup.py uses TopicStateRegistry (14 lazy imports → 1 registry call)
  - Polling: TopicLifecycleStrategy uses public methods, coordinator uses public constants
- [x] verify no circular imports: `python -c "from ccgram.handlers.msg_broker import delivery_strategy; from ccgram.handlers.msg_telegram import resolve_topic"`
- [x] run full test suite: `make test`
- [x] run integration tests: `make test-integration`
- [x] run `make check` — all GREEN

### Task 11: [Final] Update documentation

- [x] update `CLAUDE.md` module inventory (new: `topic_state_registry.py`, `user_preferences.py`, `handlers/msg_delivery.py`)
- [x] update `.claude/rules/architecture.md` module inventory
- [x] update `docs/design/architecture.md` if any design decisions changed during implementation

## Technical Details

### Topic State Registry API

```python
class TopicStateRegistry:
    def register(self, scope: str) -> Callable:
        """Decorator. Scopes: 'topic', 'window', 'qualified', 'chat'."""
        def decorator(fn):
            self._cleanups[scope].append(fn)
            return fn
        return decorator

    def clear_all(self, user_id, thread_id, *, window_id=None, qualified_id=None, chat_id=None):
        for fn in self._cleanups["topic"]: fn(user_id, thread_id)
        if chat_id:
            for fn in self._cleanups["chat"]: fn(chat_id, thread_id)
        if window_id:
            for fn in self._cleanups["window"]: fn(window_id)
        if qualified_id:
            for fn in self._cleanups["qualified"]: fn(qualified_id)
```

### Spawn Request Accessor API

```python
def get_pending(request_id: str) -> SpawnRequest | None:
    return _pending_requests.get(request_id)

def pop_pending(request_id: str) -> SpawnRequest | None:
    return _pending_requests.pop(request_id, None)

def iter_pending() -> Iterator[tuple[str, SpawnRequest]]:
    yield from _pending_requests.items()

def register_pending(req: SpawnRequest) -> None:
    _pending_requests[req.id] = req
```

### UserPreferences class (follows ThreadRouter pattern)

```python
class UserPreferences:
    def __init__(self):
        self._user_dir_favorites: dict[int, dict] = {}
        self._user_window_offsets: dict[int, dict[str, int]] = {}
        self._schedule_save: Callable | None = None

    def set_schedule_save(self, callback: Callable) -> None:
        self._schedule_save = callback

    def _save(self) -> None:
        if self._schedule_save: self._schedule_save()
```

### Session State Protocol Interfaces

```python
@runtime_checkable
class WindowStateStore(Protocol):
    def get_window_state(self, window_id: str) -> WindowState: ...
    def get_display_name(self, window_id: str) -> str: ...
    def get_session_id_for_window(self, window_id: str) -> str | None: ...
    def clear_window_session(self, window_id: str) -> None: ...

@runtime_checkable
class SessionIO(Protocol):
    async def send_to_window(self, window_id: str, text: str) -> None: ...
    def resolve_session_for_window(self, window_id: str) -> ClaudeSession | None: ...

@runtime_checkable
class WindowModeConfig(Protocol):
    def get_approval_mode(self, window_id: str) -> str: ...
    def set_window_approval_mode(self, window_id: str, mode: str) -> None: ...
    # ... notification, batch mode getters/setters/cyclers
```

## Post-Completion

**Manual verification:**

- Start bot with `./scripts/restart.sh start`, create a topic, verify topic emoji updates
- Send inter-agent message (`ccgram msg send`), verify delivery and notifications
- Close a topic, verify all per-topic state cleaned via registry
- Test directory browser, verify starred/MRU still works after extraction

**Follow-up:**

- Run fourth modularity review to confirm all imbalances resolved
- Consider adopting Protocol type annotations across handler files (separate PR)
- Consider standardizing key types across state dicts (separate PR, requires TopicStateRegistry)
