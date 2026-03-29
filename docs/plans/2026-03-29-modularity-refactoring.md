# Modularity Refactoring

## Overview

Implement the 5-step modularity refactoring designed in `docs/design/architecture.md`, addressing coupling imbalances identified in `docs/reviews/modularity-review.md`:

- **C2**: Replace implicit shell prompt regex contract with typed `PromptMatch` dataclass
- **C1**: Move Codex-specific logic behind provider protocol, eliminate `capabilities.name == "codex"` checks
- **B1**: Physically extract Thread Router from SessionManager, add Protocol interfaces
- **A2**: Extract callback registry + command/topic orchestration from bot.py
- **A1**: Decompose status_polling.py into focused strategies + thin coordinator

Each step is independently mergeable. Steps are ordered by increasing blast radius.

## Context (from design session)

- Design documents: `docs/design/*/design.md` (10 modules)
- Test specifications: `docs/design/*/tests.md` (10 modules)
- Architecture overview: `docs/design/architecture.md`
- Modularity review: `docs/reviews/modularity-review.md`
- 26 files import `session_manager`; ~15 use thread routing methods
- Existing test files cover all affected modules

## Development Approach

- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes
- **CRITICAL: all tests must pass before starting next task**
- **CRITICAL: update this plan file when scope changes during implementation**
- Run `make check` after each step (fmt + lint + typecheck + test)
- Maintain backward compatibility within each step

## Testing Strategy

- **Unit tests**: Update existing test files when interfaces change; add new tests for new modules
- **Integration tests**: Verify handler dispatch still works after callback registry extraction
- Run `make test` after every task, `make check` after every step

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix

## Implementation Steps

---

### Step C2: Shell PromptMatch Contract (lowest risk, ~2 files)

Design doc: `docs/design/shell-provider/design.md`

#### Task 1: Add PromptMatch dataclass and update all consumers (Tasks 1+2 merged)

Tasks 1 and 2 were merged because changing match_prompt()'s return type immediately breaks shell_capture.py consumers — they cannot be done independently.

- [x] Add `PromptMatch` frozen dataclass to `src/ccgram/providers/shell.py` with fields: `sequence_number: int`, `trailing_text: str`, `exit_code: int`, `raw_line: str`
- [x] Update `match_prompt()` return type from `re.Match | None` to `PromptMatch | None`
- [x] Update internal parsing: construct `PromptMatch` from regex groups for both wrap and replace modes
- [x] Export `PromptMatch` from `providers/shell.py` public API
- [x] Update tests in `tests/ccgram/test_shell_provider.py`: verify `match_prompt()` returns `PromptMatch` with correct named fields
- [x] Add new tests: `test_prompt_match_frozen`, `test_wrap_mode_bare_prompt`, `test_wrap_mode_with_trailing`, `test_replace_mode_bare_prompt`
- [x] Replace all `m.group(1)` with `m.exit_code` and `m.group(2)` with `m.trailing_text` in `shell_capture.py` (`_extract_command_output`, `_find_command_echo`, `_find_in_progress`, `_command_from_echo`)
- [x] Replace `m.group(2)` with `m.trailing_text` in `shell_commands.py` (`_cancel_stuck_input`)
- [x] Run `make check` — all GREEN (fmt + lint + typecheck + test + integration)

---

### Step C1: Provider Abstraction Restoration (~5 files)

Design doc: `docs/design/codex-provider/design.md`, `docs/design/provider-protocol/design.md`

#### Task 3: Move Codex-specific modules into providers/

- [x] Move `src/ccgram/codex_status.py` → `src/ccgram/providers/codex_status.py`
- [x] Move `src/ccgram/interactive_prompt_formatter.py` → `src/ccgram/providers/codex_format.py`
- [x] Update import in `src/ccgram/providers/codex.py`: change `from ccgram.interactive_prompt_formatter import ...` to `from .codex_format import ...`
- [x] Update import in `src/ccgram/bot.py`: change `from .codex_status import ...` to `from .providers.codex_status import ...`
- [x] Update test imports in `tests/ccgram/test_codex_status.py` and `tests/ccgram/test_interactive_prompt_formatter.py`
- [x] Run `make test` — must pass

#### Task 4: Add optional protocol methods and eliminate name checks

- [x] Add `build_status_snapshot(self, transcript_path: str, *, display_name: str, session_id: str, cwd: str) -> str | None` method to `AgentProvider` protocol in `src/ccgram/providers/base.py` with default `return None`
- [x] Add `has_output_since(self, transcript_path: str, offset: int) -> bool` method with default `return False`
- [x] Add `supports_status_snapshot: bool = False` to `ProviderCapabilities` dataclass
- [x] Implement `build_status_snapshot()` in `CodexProvider` (`src/ccgram/providers/codex.py`) delegating to `codex_status.build_codex_status_snapshot()`
- [x] Implement `has_output_since()` in `CodexProvider` delegating to `codex_status.has_codex_assistant_output_since()`
- [x] Set `supports_status_snapshot=True` in CodexProvider capabilities
- [x] In `src/ccgram/bot.py`: replace `_maybe_send_codex_status_snapshot()` and `_codex_status_probe_offset()` with calls to `provider.build_status_snapshot()` and `provider.has_output_since()` — remove `capabilities.name == "codex"` checks
- [x] Remove direct import of `codex_status` from `bot.py`
- [x] Search codebase for remaining `capabilities.name == "codex"` or `provider_name == "codex"` — replace with capability queries
- [x] Add tests: verify `build_status_snapshot()` returns None for non-Codex providers, returns string for Codex
- [x] Update existing tests in `test_forward_command.py` for new call path
- [x] Run `make check` — must pass (completes Step C1)

---

### Step B1: Thread Router Extraction (~26 files)

Design doc: `docs/design/thread-router/design.md`, `docs/design/session-state/design.md`

#### Task 5: Create Protocol definitions

- [x] Create `src/ccgram/protocols.py` with Protocol classes:
  - `WindowStateStore`: `get_window_state()`, `get_display_name()`, `get_session_id_for_window()`, `clear_window_session()`
  - `UserPreferences`: notification/approval/batch mode getters, setters, cyclers
  - `SessionResolver`: `resolve_session_for_window()`, `get_recent_messages()`
- [x] Add `TYPE_CHECKING` guard for Protocol imports to avoid runtime overhead
- [x] Run `make typecheck` — must pass

#### Task 6: Extract ThreadRouter class

- [x] Create `src/ccgram/thread_router.py` with `ThreadRouter` class
- [x] Move from `session.py` → `thread_router.py`:
  - Data: `thread_bindings`, `_window_to_thread`, `group_chat_ids`, `window_display_names`
  - Methods: `bind_thread`, `unbind_thread`, `get_window_for_thread`, `resolve_window_for_thread`, `get_thread_for_window`, `get_all_thread_windows`, `iter_thread_bindings`, `set_group_chat_id`, `resolve_chat_id`, `get_window_for_chat_thread`, `get_display_name`, `set_display_name`, `sync_display_names`, `_rebuild_reverse_index`
- [x] Add `to_dict()` / `from_dict()` serialization methods to ThreadRouter
- [x] Add `schedule_save` callback parameter to ThreadRouter constructor (called after mutations)
- [x] Create module-level singleton `thread_router = ThreadRouter()`
- [x] Wire ThreadRouter into SessionManager: `session.py` holds reference, calls `thread_router.to_dict()` on save and `thread_router.from_dict()` on load
- [x] Remove moved methods and data from SessionManager (replace with delegation or remove entirely)
- [x] Add tests for ThreadRouter in `tests/ccgram/test_thread_router.py`: `test_bind_thread`, `test_unbind_thread`, `test_reverse_index`, `test_to_dict_roundtrip`, `test_resolve_chat_id_fallback`
- [x] Update existing tests in `tests/ccgram/test_session.py` to account for extraction
- [x] Run `make test` — must pass

#### Task 7: Update consumer imports (batch 1 — handlers)

- [x] Update `src/ccgram/handlers/text_handler.py`: import `thread_router` for routing methods
- [x] Update `src/ccgram/handlers/message_queue.py`: import `thread_router` for `resolve_chat_id`
- [x] Update `src/ccgram/handlers/interactive_ui.py`: import `thread_router` for routing
- [x] Update `src/ccgram/handlers/shell_commands.py`: import `thread_router` for routing
- [x] Update `src/ccgram/handlers/shell_capture.py`: import `thread_router` if needed
- [x] Update `src/ccgram/handlers/window_callbacks.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/directory_callbacks.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/directory_browser.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/recovery_callbacks.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/screenshot_callbacks.py`: import `thread_router`
- [x] Run `make test` — must pass

#### Task 8: Update consumer imports (batch 2 — core + remaining)

- [x] Update `src/ccgram/bot.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/status_polling.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/hook_events.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/cleanup.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/topic_emoji.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/restore_command.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/resume_command.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/sessions_dashboard.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/history.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/callback_helpers.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/voice_handler.py` and `voice_callbacks.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/file_handler.py`: import `thread_router`
- [x] Update `src/ccgram/handlers/sync_command.py`: import `thread_router`
- [x] Update `src/ccgram/session_monitor.py`: import `thread_router` if needed
- [x] Update `src/ccgram/providers/__init__.py`: not needed (no routing methods used)
- [x] Update test files that mock `session_manager` routing methods to mock `thread_router` instead
- [x] Run `make check` — must pass (completes Step B1)

---

### Step A2: Bot Dispatch Extraction (~20 files)

Design doc: `docs/design/bot-shell/design.md`, `docs/design/callback-dispatch/design.md`, `docs/design/command-orchestration/design.md`, `docs/design/topic-orchestration/design.md`

#### Task 9: Create callback registry

- [x] Create `src/ccgram/handlers/callback_registry.py` with:
  - `_registry: dict[str, Callable]` — prefix → handler mapping
  - `register(*prefixes)` decorator for handler self-registration
  - `dispatch(update, context)` function — longest-prefix match, authorization check, group chat ID recording
  - `load_handlers()` function — explicit imports of all callback-bearing handler modules
- [x] Add `@register(CB_...)` decorators to existing callback handler modules:
  - `directory_callbacks.py` — `@register(CB_DIR_*)`
  - `window_callbacks.py` — `@register(CB_WINDOW_*)`
  - `history_callbacks.py` — `@register(CB_HISTORY_*)`
  - `screenshot_callbacks.py` — `@register(CB_SCREENSHOT_*, CB_STATUS_*, CB_TOOLBAR_*)`
  - `interactive_callbacks.py` — `@register(CB_INTERACTIVE_*)`
  - `recovery_callbacks.py` — `@register(CB_RECOVERY_*)`
  - `resume_command.py` — `@register(CB_RESUME_*)`
  - `voice_callbacks.py` — `@register(CB_VOICE_*)`
  - `shell_commands.py` — `@register(CB_SHELL_*)`
- [x] Add tests in `tests/ccgram/handlers/test_callback_registry.py`: `test_register_single_prefix`, `test_dispatch_matches_prefix`, `test_dispatch_longest_prefix`, `test_dispatch_no_match`, `test_load_handlers_populates_registry`
- [x] Run `make test` — must pass

#### Task 10: Extract command orchestration from bot.py

- [x] Create `src/ccgram/handlers/command_orchestration.py`
- [x] Move from `bot.py`:
  - `forward_command_handler()` and its helpers
  - `_sync_scoped_provider_menu()`, `_sync_chat_scoped_provider_menu()`, `_get_provider_command_metadata()`
  - Menu cache state: `_scoped_provider_menu`, `_chat_scoped_provider_menu`, `_global_provider_menu`
  - `_maybe_send_codex_status_snapshot()` replacement (now `_maybe_send_status_snapshot()` using `provider.build_status_snapshot()`)
  - `_codex_status_probe_offset()` replacement
  - Menu refresh job setup function
- [x] Update `bot.py` to import `forward_command_handler` from `command_orchestration`
- [x] Update `create_bot()` handler registration to use imported function
- [x] Add tests in `tests/ccgram/handlers/test_command_orchestration.py`: `test_forward_known_command`, `test_forward_unknown_command_warns`, `test_menu_cache_invalidated_on_provider_change`
- [x] Run `make test` — must pass

#### Task 11: Extract topic orchestration from bot.py

- [x] Create `src/ccgram/handlers/topic_orchestration.py`
- [x] Move from `bot.py`:
  - `_handle_new_window()` and helpers
  - `_adopt_unbound_windows()`
  - Rate limiting state: `_topic_create_retry_until`
- [x] Update `bot.py` `post_init()` to import and use `handle_new_window` from `topic_orchestration`
- [x] Add tests in `tests/ccgram/handlers/test_topic_orchestration.py`: `test_handle_new_window_creates_topic`, `test_handle_new_window_skips_already_bound`, `test_rate_limit_backoff`
- [x] Run `make test` — must pass

#### Task 12: Wire callback registry into bot.py and remove old dispatch

- [x] In `bot.py` `create_bot()`: replace `CallbackQueryHandler(callback_handler)` with `CallbackQueryHandler(callback_registry.dispatch)`
- [x] Call `callback_registry.load_handlers()` before `create_bot()` returns
- [x] Remove the old `callback_handler()` function from `bot.py`
- [x] Remove all 47 `CB_*` imports from `bot.py` that were only used in `callback_handler()`
- [x] Remove handler function imports from `bot.py` that are now self-registered
- [x] Move sessions dashboard inline callback handlers (refresh/new/kill) to `sessions_dashboard.py` with `@register`
- [x] Move sync callback handlers to `sync_command.py` with `@register`
- [x] Update `tests/ccgram/test_bot_callbacks.py` to test via registry dispatch
- [x] Update `tests/ccgram/test_callback_auth.py` to test authorization in registry dispatch
- [x] Run `make check` — must pass (completes Step A2)

---

### Step A1: Polling Decomposition (~5 files)

Design doc: `docs/design/polling-subsystem/design.md`

#### Task 13: Create polling strategies module

- [x] Create `src/ccgram/handlers/polling_strategies.py` with 4 strategy classes:
  - `TerminalStatusStrategy` — pyte parsing, provider status, RC debounce, spinner detection; owns `WindowPollState`
  - `InteractiveUIStrategy` — permission prompt scanning, multi-pane alerts; owns `_pane_alert_hashes`
  - `TopicLifecycleStrategy` — autoclose timers, dead detection, topic probing, unbound TTL; owns `TopicPollState`, `_dead_notified`
  - `ShellRelayStrategy` — passive shell output delegation
- [x] Define shared `PollResult` dataclass returned by each strategy (status text, emoji state, actions to take)
- [x] Move `WindowPollState` and `TopicPollState` dataclasses from `status_polling.py`
- [x] Move module-level state dicts into their owning strategy classes
- [x] Move domain-specific functions into strategy methods:
  - `_parse_with_pyte()`, RC detection → `TerminalStatusStrategy`
  - State management, autoclose timers, dead notifications, probe failures → `TopicLifecycleStrategy`
  - Pane alert state management → `InteractiveUIStrategy`
  - ➕ Async orchestration functions remain in `status_polling.py` using strategy state (test patches require module-level patching compatibility)
- [x] Add tests in `tests/ccgram/handlers/test_polling_strategies.py`: test each strategy class independently
- [x] Run `make test` — must pass

#### Task 14: Create polling coordinator

- [x] Create `src/ccgram/handlers/polling_coordinator.py` with:
  - `status_poll_loop(bot)` — thin async loop (~200 lines)
  - Iterate `thread_router.iter_thread_bindings()`
  - For each binding: delegate to strategies in sequence
  - Error handling with exponential backoff (moved from `status_polling.py`)
  - Display name sync (moved from `status_polling.py`)
- [x] Instantiate strategies at module level (or in loop init)
- [x] Wire coordinator to use strategy instances
- [x] Run `make test` — must pass

#### Task 15: Remove old status_polling.py and finalize

- [x] Update `src/ccgram/bot.py`: change `from .handlers.status_polling import status_poll_loop` to `from .handlers.polling_coordinator import status_poll_loop`
- [x] Update all imports of `clear_*` functions from `status_polling` to point to new locations (strategies or coordinator)
- [x] Update `src/ccgram/handlers/cleanup.py` imports
- [x] Delete `src/ccgram/handlers/status_polling.py`
- [x] Move/update tests from `tests/ccgram/test_status_polling.py` to new test files
- [x] Run `make check` — must pass (completes Step A1)

---

### Task 16: Verify acceptance criteria

- [x] Verify all 5 review issues addressed:
  - C2: `match_prompt()` returns `PromptMatch`, no `re.Match` group access in shell_capture
  - C1: No `capabilities.name == "codex"` checks outside providers/, codex_status.py moved
  - B1: ThreadRouter is a separate class, consumers import it directly
  - A2: bot.py 1050 lines (core responsibilities remain), callback_handler() removed, orchestration extracted
  - A1: status_polling.py removed, replaced by coordinator + strategies
- [x] Verify no circular imports introduced
- [x] Run `make check` — all GREEN (fmt + lint + typecheck + test)
- [x] Run `make test-integration` — integration tests pass (71 passed)
- [x] Verify test coverage maintained (no regression)

### Task 17: [Final] Update documentation

- [x] Update `CLAUDE.md` module inventory table to reflect new modules
- [x] Update `.claude/rules/architecture.md` module inventory
- [x] Remove old file references, add new module descriptions

## Technical Details

### PromptMatch dataclass (Step C2)

```python
@dataclass(frozen=True)
class PromptMatch:
    sequence_number: int   # monotonic counter for output isolation
    trailing_text: str     # command text after marker (empty = bare prompt)
    exit_code: int         # exit code integer
    raw_line: str          # original terminal line
```

### Protocol definitions (Step B1)

```python
class WindowStateStore(Protocol):
    def get_window_state(self, window_id: str) -> WindowState: ...
    def get_display_name(self, window_id: str) -> str: ...

class UserPreferences(Protocol):
    def get_notification_mode(self, window_id: str) -> str: ...
    def cycle_notification_mode(self, window_id: str) -> str: ...
    # ... etc
```

### Callback registry pattern (Step A2)

```python
# In handler module:
@register(CB_DIR_SELECT, CB_DIR_BACK, CB_DIR_HOME)
async def handle_directory_callback(update, context): ...

# In callback_registry.py:
async def dispatch(update, context):
    data = update.callback_query.data
    for prefix in sorted(_registry, key=len, reverse=True):  # longest first
        if data.startswith(prefix):
            return await _registry[prefix](update, context)
```

## Post-Completion

**Manual verification:**

- Start bot locally (`./scripts/restart.sh start`), verify all features work
- Test callback dispatch (inline buttons), command forwarding, status polling
- Test shell provider prompt marker detection end-to-end

**Downstream updates:**

- None — this is an internal refactoring, no public API changes
