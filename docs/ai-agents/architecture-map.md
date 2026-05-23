# Architecture Map

Authoritative module inventory is `/.claude/rules/architecture.md`. This file covers request/response lifecycles and the design constraints that must be preserved across changes.

## Request/Response Lifecycles

Inbound user message (Telegram → tmux):

1. PTB dispatcher routes through handlers wired in `handlers/registry.py`.
2. `handlers/text/text_handler.py` validates context and resolves topic binding.
3. `session.py` maps `(user_id, thread_id)` → `window_id`.
4. `tmux_manager.py` sends keys to the mapped window/pane.

Shell provider (NL → command → shell):

1. `handlers/text/text_handler.py` detects shell-provider window, routes to `handlers/shell/shell_commands.py`.
2. `shell_commands.py` calls `llm/` to generate a suggested command.
3. Approval keyboard rendered; user confirms or cancels.
4. On approval, command sent via `tmux_manager.py`.
5. `handlers/shell/shell_capture.py` polls pane output and relays via in-place edits.

Voice message (voice → transcription → agent):

1. `handlers/voice/voice_handler.py` downloads audio, transcribes via `whisper/`.
2. Confirm/discard keyboard shown.
3. On confirm, `handlers/voice/voice_callbacks.py` checks provider:
   - Shell: routes transcribed text through `handlers/shell/shell_commands.py` (LLM → approval).
   - Other: sends directly to the tmux window.

Outbound agent output (provider transcript/event → Telegram):

1. `session_monitor.py` polls tracked sources incrementally.
2. Provider parser (`providers/*.py` + `transcript_parser.py` / `terminal_parser.py`) emits normalized updates.
3. `handlers/messaging_pipeline/message_queue.py` enforces ordering, merge rules, rate limits. Worker takes a `TelegramClient`.
4. `handlers/messaging_pipeline/message_sender.py` delivers via the Protocol.

Screenshots (`/screenshot`, 📷 status-bar button):

1. `handlers/live/screenshot_callbacks.py` calls `last_unit.capture_for_screenshot(window_id)`.
2. `last_unit.py` calls `tmux_manager.capture_pane_scrollback()` (default 500 lines, `CCGRAM_SCREENSHOT_HISTORY`).
3. For shell topics, `last_unit.extract_last_shell_block()` slices the last command+output using prompt markers; other providers get full scrollback.
4. `screenshot.py` renders ANSI text to PNG; result sent as photo.

Live view (terminal → auto-refresh screenshots):

1. User taps Live in `handlers/live/screenshot_callbacks.py`.
2. `handlers/live/live_view.py` registers active view for the topic.
3. `handlers/polling/periodic_tasks.py` calls `live_view.tick_live_views()` every `config.live_view_interval` seconds.
4. Each tick captures the pane via `tmux_manager.py` (viewport only — unchanged from pre-scrollback), hashes content, edits via `editMessageMedia` only when changed.
5. Auto-stops after `config.live_view_timeout` or when user taps Stop.

Recovery (dead/missing session):

1. `handlers/polling/polling_coordinator.py` detects stale/dead bindings via `handlers/polling/window_tick/`.
2. Recovery UI callbacks → `handlers/recovery/recovery_callbacks.py` (thin dispatcher) → `recovery_banner.py` (dead-window banner) or `resume_picker.py` (resume picker + transcript scan).
3. State updated in `session.py` and persisted to `state.json`.

Commands menu (`/commands`):

1. `handlers/registry.py` dispatches to `handlers/commands/__init__.py:commands_command`.
2. `command_catalog.py` discovers commands for the window's provider (60s TTL).
3. `cc_commands.py` renders the scoped menu as inline keyboard.
4. Selection sends command via `tmux_manager.py`. Failure path: `handlers/commands/failure_probe.py`. Status snapshot: `handlers/commands/status_snapshot.py`. Menu cache: `handlers/commands/menu_sync.py`.

## Transcript Sources (read-only)

- Claude: `~/.claude/projects/`
- Codex: `~/.codex/sessions/`
- Gemini: `~/.gemini/tmp/<project-hash>/chats/*.jsonl` (CLI v0.40+; append-only JSONL, byte-offset incremental reads). Discovery matches by `projectHash` (or configured alias dir); no full-scan of unrelated project dirs.
- Pi: `~/.pi/agent/sessions/--<encoded-cwd>--/<timestamp>_<uuid>.jsonl` (JSONL v3; discovery matches the header `cwd` against the window cwd).
- Shell: no transcript files; output captured directly from the tmux pane by `handlers/shell/shell_capture.py`.

## Design Constraints to Preserve

- 1 topic = 1 window mapping; internal identity keyed by tmux `window_id`.
- No parse-layer truncation; splitting only at the Telegram send layer.
- Per-window provider behavior + capability-gated UI.
- tmux operations centralized in `tmux_manager.py`; no raw tmux shell calls in handlers.
- State mutations route through `session.py` + persistence helpers; no ad-hoc JSON writes.
- Handlers depend on `TelegramClient` Protocol. Runtime `from telegram.ext` allowed only in `bot.py`, `bootstrap.py`, `handlers/registry.py`, `telegram_client.py`, `telegram_request.py`, `telegram_sender.py`. Everything else uses `if TYPE_CHECKING:` for types.
- `SessionManager` constructs `WindowStateStore`, `ThreadRouter`, `UserPreferences`, `SessionMapSync` via constructor DI. Do not reintroduce `_wire_singletons` or `unwired_save`.
- Handler reads go through `window_query` / `session_query` or `window_state_ports/*`. Direct `session_manager.<attr>` in `handlers/**` is restricted to the write/admin allow-list (`set_window_provider`, `set_window_origin`, `set_window_approval_mode`, `set_window_worktree`, `cycle_*`, `audit_state`, `prune_*`, `sync_display_names`). Enforced by `tests/ccgram/test_query_layer_only_for_handlers.py`.
- `WindowStateStore` is the only persisted window-state model. Feature-shaped reads and cohesive feature writes live in `src/ccgram/window_state_ports/{pane,identity,worktree,tool,lifecycle}_state.py`. Raw `WindowState`-field access in handlers, Mini App, or session_resolver/transcript_reader is rejected by `tests/ccgram/test_window_state_access_audit.py`. Provider identity writes still delegate to `SessionManager.set_window_provider`.
- `handlers/polling/polling_types.py` is pure (stdlib + `providers.base.StatusUpdate` only). `polling_state.py` owns strategies + module-level singletons. `decide.py` imports only from `polling_types`. Pinned by `tests/ccgram/handlers/polling/test_polling_types_purity.py`.
- In-function imports must carry `# Lazy: <reason>` (or live inside `if TYPE_CHECKING:` / `_reset_*_for_testing`). `make lint` runs `lint-lazy`.
- Ordering invariant: `bootstrap.wire_runtime_callbacks` must run before `bootstrap.start_session_monitor`. The monitor checks `_callbacks_wired` and raises if violated.
