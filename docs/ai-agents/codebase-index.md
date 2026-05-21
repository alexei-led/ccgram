# Codebase Index

## Where to Look First

Entry points:

- `src/ccgram/main.py`: process bootstrap.
- `src/ccgram/cli.py`: CLI command surface and config override rules.
- `src/ccgram/bot.py`: 172-line PTB Application factory + lifecycle delegate (post Round 4 F3).
- `src/ccgram/bootstrap.py`: `bootstrap_application` (post_init wiring) + `shutdown_runtime` (post_shutdown teardown). Named steps (`register_provider_commands`, `verify_hooks_installed`, `wire_runtime_callbacks`, `start_session_monitor`, `start_status_polling`, `start_miniapp_if_enabled`) are independently testable.
- `src/ccgram/handlers/registry.py`: central PTB command/message/callback/inline handler registration (`register_all`).
- `src/ccgram/telegram_client.py`: `TelegramClient` Protocol that handlers depend on; `PTBTelegramClient(bot)` adapter; `FakeTelegramClient` for tests; `unwrap_bot(client)` escape hatch.

State and routing:

- `src/ccgram/session.py`: thread bindings, window states, read offsets.
- `src/ccgram/window_resolver.py`: stale window ID migration/re-resolution.
- `src/ccgram/state_persistence.py`: atomic/debounced JSON persistence.

Monitoring and parsing:

- `src/ccgram/session_monitor.py`: polling engine for sessions and hook events.
- `src/ccgram/transcript_parser.py`: Claude transcript parsing and tool pairing.
- `src/ccgram/terminal_parser.py`: terminal status/UI detection.
- `src/ccgram/claude_task_state.py`: Claude task tracking from transcripts for live status bubble.

Telegram handler surface (post Round 4 F1 — feature subpackages):

- `src/ccgram/handlers/text/text_handler.py`: text-path orchestrator.
- `src/ccgram/handlers/messaging_pipeline/message_queue.py`: ordering, merge rules, status conversion. Worker takes `client: TelegramClient`.
- `src/ccgram/handlers/messaging_pipeline/message_sender.py`: `safe_reply`/`safe_edit`/`safe_send` + `rate_limit_send_message` + `edit_with_fallback` — depend on `TelegramClient`.
- `src/ccgram/handlers/messaging_pipeline/tool_batch.py`: Claude tool-use batching; uses `unwrap_bot(client)` for `DraftStream`.
- `src/ccgram/handlers/polling/polling_coordinator.py`: background status and stale-session cleanup.
- `src/ccgram/handlers/polling/window_tick/`: per-window poll cycle split into `decide.py` (pure), `observe.py` (pure inputs in, `TickContext` out), `apply.py` (DI-heavy side effects).
- `src/ccgram/handlers/polling/polling_types.py`: pure types module (Round 5 F1) — `TickContext`, `TickDecision`, `PaneTransition`, `WindowPollState`, `TopicPollState`, constants, pure `is_shell_prompt`. Stdlib + `providers.base.StatusUpdate` only.
- `src/ccgram/handlers/polling/polling_state.py`: stateful strategies + module-level singletons (Round 5 F1) — `TerminalPollState`, `TerminalScreenBuffer`, `InteractiveUIStrategy`, `TopicLifecycleStrategy`, `PaneStatusStrategy`, `terminal_poll_state`, `terminal_screen_buffer`, `interactive_strategy`, `lifecycle_strategy`, `pane_status_strategy`.
- `src/ccgram/handlers/polling/periodic_tasks.py`: periodic task orchestration (broker, sweep, lifecycle, live view).
- `src/ccgram/handlers/topics/directory_browser.py` + `directory_callbacks.py`: new-session UX.
- `src/ccgram/handlers/topics/topic_lifecycle.py`: autoclose timers for done/dead topics, unbound window TTL.
- `src/ccgram/handlers/topics/topic_orchestration.py`: new window/topic creation, unbound window adoption.
- `src/ccgram/handlers/topics/window_callbacks.py`: window picker callbacks (bind, new, cancel).
- `src/ccgram/handlers/topics/new_command.py`: `/new` (and `/start` alias) handler.
- `src/ccgram/handlers/topics/worktree.py`: pure git-worktree plumbing — `check_worktree_eligibility`, `suggest_branch_name`, `validate_branch_name`, `create_worktree` (raises `WorktreeError`); no Telegram/tmux/state deps.
- `src/ccgram/handlers/interactive/interactive_ui.py` + `interactive_callbacks.py`: interactive prompt UX.
- `src/ccgram/handlers/sessions_dashboard.py`: `/sessions` dashboard behavior.
- `src/ccgram/handlers/recovery/recovery_callbacks.py`: thin dispatcher (Round 5 F3) — `_dispatch`, `handle_recovery_callback`, recovery-state validate/clear.
- `src/ccgram/handlers/recovery/recovery_banner.py`: dead-window banner UX (Round 5 F3) — `RecoveryBanner`, `render_banner`, `build_recovery_keyboard`, fresh/continue/resume/back/browse/cancel handlers.
- `src/ccgram/handlers/recovery/resume_picker.py`: resume picker UX + transcript scan (Round 5 F3) — `_SessionEntry`, `scan_sessions_for_cwd`, picker keyboard builders, `_handle_resume_pick`.
- `src/ccgram/handlers/recovery/restore_command.py`: `/restore` command for dead topic recovery.
- `src/ccgram/handlers/recovery/resume_command.py`: `/resume` scan past sessions + inline picker.
- `src/ccgram/handlers/recovery/transcript_discovery.py`: hookless transcript discovery for Codex/Gemini.
- `src/ccgram/handlers/recovery/history.py` + `history_callbacks.py`: `/history` command + pagination callbacks.
- `src/ccgram/handlers/live/screenshot_callbacks.py`: screenshot refresh, Esc, quick-key, pane screenshots.
- `src/ccgram/handlers/live/live_view.py`: auto-refreshing terminal screenshots via editMessageMedia.
- `src/ccgram/handlers/live/pane_callbacks.py`: per-pane callbacks (rename, screenshot select).
- `src/ccgram/handlers/hook_events.py`: hook event dispatcher (Notification, Stop, Subagent*, Team*).
- `src/ccgram/handlers/cleanup.py`: centralized topic state teardown on close/delete.
- `src/ccgram/handlers/status/topic_emoji.py`: debounced topic name emoji updates (active/idle/done/dead). Color scheme is configurable via `CCGRAM_STATUS_MODE` (`system`: green=working; `user`: green=ready).
- `src/ccgram/handlers/status/status_bubble.py`: status-bubble keyboard + status message lifecycle.
- `src/ccgram/handlers/status/status_bar_actions.py`: status-bubble button callbacks.
- `src/ccgram/handlers/status/rc_probe.py`: Remote Control outcome probe — `arm_rc_probe`, pure `classify_rc_output`, `_classify_loop`; captures/classifies the pane after Claude `/remote-control`, posts one status reply, de-duped via in-memory `WindowState.rc_probe_state` (Claude-only).
- `src/ccgram/handlers/file_handler.py`: photo/document upload → `.ccgram-uploads/` → agent notification.
- `src/ccgram/handlers/upgrade.py`: `/upgrade` uv tool upgrade + `os.execv()` restart.
- `src/ccgram/handlers/sync_command.py`: `/sync` state audit + fix button.
- `src/ccgram/handlers/command_history.py`: per-user/per-topic command recall (in-memory, max 20).
- `src/ccgram/handlers/voice/voice_handler.py`: voice message download, Whisper transcription, confirm/discard keyboard.
- `src/ccgram/handlers/voice/voice_callbacks.py`: voice callback routing (vc:send/vc:drop); shell provider transcriptions route through LLM.
- `src/ccgram/handlers/messaging/msg_broker.py` + `msg_delivery.py` + `msg_telegram.py` + `msg_spawn.py`: inter-agent message broker, delivery state, Telegram notifications, spawn approval flow.
- `src/ccgram/handlers/inline.py`: top-level `inline_query_handler` and `unsupported_content_handler`.
- `src/ccgram/handlers/reactions.py`: Telegram message reactions helper (Bot API 7.0+).

Provider and command surface:

- `src/ccgram/providers/`: provider contract and implementations.
- `src/ccgram/command_catalog.py`: provider-agnostic command discovery + 60s TTL caching.
- `src/ccgram/cc_commands.py`: Telegram menu registration from discovered commands.
- `src/ccgram/handlers/commands/__init__.py`: `/commands` + `/toolbar` entry points (Round 5 F4) — re-exports `forward_command_handler`, `setup_menu_refresh_job`, `get_global_provider_menu`, `set_global_provider_menu`, `sync_scoped_*`.
- `src/ccgram/handlers/commands/forward.py`: forward command handler (Round 5 F4) — `forward_command_handler`, `_handle_clear_command`. Universal `/<token>` forward; unknown commands surfaced by `failure_probe`, never pre-rejected.
- `src/ccgram/handlers/commands/menu_sync.py`: provider menu cache + scoped sync (Round 5 F4) — `sync_scoped_provider_menu`, `setup_menu_refresh_job`, `_build_provider_command_metadata`, LRU helpers.
- `src/ccgram/handlers/commands/failure_probe.py`: command failure probing (Round 5 F4) — `_capture_command_probe_context`, `_probe_transcript_command_error`, `_spawn_command_failure_probe`.
- `src/ccgram/handlers/commands/status_snapshot.py`: status snapshot delegation (Round 5 F4).
- `src/ccgram/hook.py`: Claude hook install/status/uninstall and event writes.
- `src/ccgram/llm/`: LLM command generation (CommandGenerator protocol, httpx completers for OpenAI-compatible and Anthropic APIs, provider registry).
- `src/ccgram/handlers/shell/shell_commands.py`: shell NL→command approval flow; routes NL text through LLM, renders approval keyboard, handles raw `!` prefix execution.
- `src/ccgram/handlers/shell/shell_capture.py`: shell terminal output capture and relay; polls tmux pane output and streams updates to Telegram via in-place message editing.
- `src/ccgram/handlers/shell/shell_context.py`: shared shell helpers (`gather_llm_context`, `redact_for_llm`).
- `src/ccgram/handlers/shell/shell_prompt_orchestrator.py`: prompt marker setup orchestrator (`ensure_setup`).
- `src/ccgram/whisper/`: voice transcription (WhisperTranscriber protocol, httpx transcriber for OpenAI-compatible APIs, provider factory).

Supporting modules:

- `src/ccgram/screenshot.py`: terminal text → PNG rendering (PIL, ANSI color, font fallback).
- `src/ccgram/providers/codex_status.py`: Codex status snapshot from JSONL transcripts.
- `src/ccgram/session_map.py`: session map I/O for session_map.json.
- `src/ccgram/session_resolver.py`: JSONL session resolution and message history.
- `src/ccgram/window_state_store.py`: per-window state (WindowState dataclass, mode settings).
- `src/ccgram/state_persistence.py`: atomic/debounced JSON persistence.
- `src/ccgram/telegram_request.py`: resilient HTTPX transport for Telegram long polling.

## Decision Map (Where to Edit)

Change topic/window routing behavior:

- `src/ccgram/session.py` for bindings/state model.
- `src/ccgram/handlers/callback_helpers.py` for thread/window extraction helpers.
- `src/ccgram/window_resolver.py` for stale ID re-resolution.

Change monitor/event dispatch behavior:

- `src/ccgram/session_monitor.py` for polling and fan-out.
- `src/ccgram/monitor_state.py` for byte-offset persistence.
- `src/ccgram/handlers/hook_events.py` for hook event handling.

Change provider behavior (commands, parsing, capabilities):

- `src/ccgram/providers/base.py` for contract/capabilities.
- `src/ccgram/providers/__init__.py` for per-window provider resolution.
- `src/ccgram/providers/{claude,codex,gemini,pi,shell}.py` for provider-specific behavior.
- `src/ccgram/providers/pi_discovery.py` + `pi_format.py` for Pi command discovery and transcript parsing.
- `src/ccgram/providers/codex_format.py` for provider-facing interactive prompt text normalization (currently Codex edit approval readability).

Change shell command generation behavior:

- `src/ccgram/llm/` for LLM backend selection, prompt construction, and result parsing.
- `src/ccgram/handlers/shell/shell_commands.py` for approval keyboard flow and raw command execution.

Add new LLM provider:

- `src/ccgram/llm/__init__.py`: add entry to `_PROVIDERS` dict with `base_url`, `model`, and `api_key_env` keys. Temperature is passed through from config automatically.

Change Telegram interactive UX:

- `src/ccgram/handlers/interactive/interactive_ui.py` and `interactive_callbacks.py`.
- `src/ccgram/handlers/callback_data.py` for callback key contracts.
- `src/ccgram/handlers/messaging_pipeline/message_queue.py` for ordering/merge side effects.
- `src/ccgram/handlers/live/live_view.py` for terminal live view sessions.

Change command discovery:

- `src/ccgram/command_catalog.py` for filesystem scanning and caching.
- `src/ccgram/cc_commands.py` for Telegram menu registration.
- `src/ccgram/handlers/commands/menu_sync.py` for scoped per-window menu sync and provider menu cache (Round 5 F4).

Change `/commands` failure probe / status snapshot:

- `src/ccgram/handlers/commands/failure_probe.py` for transcript-based failure detection.
- `src/ccgram/handlers/commands/status_snapshot.py` for status snapshot delegation.

Change recovery UX (dead window banner / resume picker):

- `src/ccgram/handlers/recovery/recovery_banner.py` for the dead-window banner UX (Round 5 F3).
- `src/ccgram/handlers/recovery/resume_picker.py` for the resume picker UX + transcript scan (Round 5 F3).
- `src/ccgram/handlers/recovery/recovery_callbacks.py` is now a thin dispatcher only — do not add UX logic here.

Change polling pure types vs strategies:

- `src/ccgram/handlers/polling/polling_types.py` for contracts (`TickContext`, `TickDecision`, constants, `is_shell_prompt`) — keep imports stdlib + `providers.base.StatusUpdate` only (Round 5 F1).
- `src/ccgram/handlers/polling/polling_state.py` for strategies and module-level singletons.

Change tool-call visibility (hide/show `tool_use`/`tool_result`):

- `src/ccgram/window_state_store.py`: `tool_call_visibility` field on `WindowState` (`default`/`shown`/`hidden`).
- `src/ccgram/handlers/messaging_pipeline/message_queue.py` (`_handle_content_task`): visibility gate sits before batch eligibility; hidden entries are dropped before `_tool_msg_ids` registration. Hook events bypass the gate via `StatusUpdateTask`.
- `/toolcalls` command (in `src/ccgram/handlers/messaging_pipeline/topic_commands.py`): cycles per-window mode via `WindowStateStore` cycle method.

Change topic emoji color scheme:

- `src/ccgram/handlers/status/topic_emoji.py`: maps internal status (`active`/`idle`/`done`/`dead`) to Telegram emoji color via `CCGRAM_STATUS_MODE`. Add new modes by extending the mode→colorname dispatch.

Change PTB handler registration / lifecycle:

- `src/ccgram/handlers/registry.py` for command/message/callback/inline handler wiring (`register_all`, `command_specs`).
- `src/ccgram/bootstrap.py` for `post_init` (provider commands, hook verification, runtime callback wiring, session monitor + status polling start, mini-app start) and `post_shutdown` teardown. Respect the ordering invariant: `wire_runtime_callbacks` before `start_session_monitor`.
- `src/ccgram/bot.py` is the factory + lifecycle delegate only; do not push wiring back into it.

Change Telegram bot API surface used by handlers:

- `src/ccgram/telegram_client.py`: add new methods to the `TelegramClient` Protocol only when a handler genuinely needs them; mirror the addition in `PTBTelegramClient` (delegation) and `FakeTelegramClient` (recording + default return). Do not import `telegram.Bot` from inside `src/ccgram/handlers/**`; depend on the Protocol instead.

Change screenshot rendering:

- `src/ccgram/screenshot.py` only.

Change tmux behavior:

- `src/ccgram/tmux_manager.py` only.

## Change Mapping by Task Type

Add or change a Telegram command:

- Wire the command in `src/ccgram/handlers/registry.py` (`command_specs` list).
- Implement the handler in the appropriate `src/ccgram/handlers/<subpackage>/` module (or a top-level handler file for cross-cutting commands).
- Add callback constants in `handlers/callback_data.py` when needed.
- Take the `client: TelegramClient` argument (not `bot: Bot`).

Change session binding logic:

- `src/ccgram/session.py` and `src/ccgram/window_resolver.py`.
- Validate persistence compatibility in `tests/ccgram/test_state_migration.py`.

Adjust transcript/status parsing:

- provider-specific parsing in `src/ccgram/providers/*.py`.
- shared parse behavior in `transcript_parser.py` / `terminal_parser.py`.

Touch tmux behavior:

- `src/ccgram/tmux_manager.py` only; avoid shell calls spread across handlers.

Add or change live view behavior:

- `src/ccgram/handlers/live/live_view.py` for view sessions and ticking logic.
- `src/ccgram/handlers/polling/periodic_tasks.py` for tick scheduling.
- `src/ccgram/handlers/live/screenshot_callbacks.py` for Live button callback.

## Contracts You Must Not Break

- Keep topic-window identity 1:1 and window-id keyed.
- Preserve tool-use/tool-result pairing and in-order delivery.
- Keep provider logic behind provider interfaces/capabilities.
- Keep parsing full-fidelity; split only in Telegram send path.
- Use `handlers/user_state.py` keys for `context.user_data`; avoid new raw string keys.
- Handlers depend on the `TelegramClient` Protocol, not `telegram.Bot`. Only `bot.py`, `bootstrap.py`, `handlers/registry.py`, `telegram_client.py`, `telegram_request.py`, and `telegram_sender.py` import from `telegram.ext` at runtime; everything else uses `if TYPE_CHECKING:` for types.
- `SessionManager` constructs its stores via constructor DI (`schedule_save` callbacks). Do not reintroduce `_wire_singletons` or `unwired_save` defaults.
- `bot.py` stays a factory + lifecycle delegate. Wiring belongs in `bootstrap.py`; PTB handler registration belongs in `handlers/registry.py`.

## Debug Index

Symptom: messages routed to wrong topic/window

- inspect `thread_bindings` and window IDs in `session.py`.
- confirm callback/thread ID extraction in `handlers/callback_helpers.py`.

Symptom: no new assistant messages

- inspect `session_monitor.py` byte offsets and session map updates.
- verify provider parser compatibility for that window/provider.

Symptom: interactive keyboard not shown

- inspect `handlers/interactive/interactive_ui.py` and provider `parse_terminal_status` output.
- check hook events path (`hook.py` -> `handlers/hook_events.py`) for Claude.

Symptom: duplicated or out-of-order status/content messages

- inspect merge/send behavior in `handlers/messaging_pipeline/message_queue.py`.

Symptom: commands menu missing/wrong

- check `command_catalog.py` cache TTL and filesystem scan paths.
- check `cc_commands.py` menu registration and provider scoping.

Symptom: live view not refreshing

- inspect `handlers/live/live_view.py` active views dict and tick interval.
- check `handlers/polling/periodic_tasks.py` for live view tick scheduling.
- verify screenshot capture in `screenshot.py` and tmux pane availability.

Symptom: `RuntimeError("... not wired")` or `RuntimeError("... already registered")` at startup

- check `handlers/hook_events.register_stop_callback`, `handlers/status/status_bubble.register_rc_active_provider`, or `handlers/shell/shell_capture.register_approval_callback` — the wire-once / fail-loud contract from F2.6 raises if the callee is invoked before registration or if registration is attempted twice.
- verify `bootstrap.wire_runtime_callbacks` runs before `bootstrap.start_session_monitor` (ordering invariant; the monitor checks the `_callbacks_wired` flag and raises if violated).
- in tests, the autouse fixture `_reset_runtime_callbacks` (in `tests/ccgram/handlers/conftest.py` and `tests/e2e/conftest.py`) resets these between tests; missing fixture is a test-setup bug.

Symptom: `import` cycle / `partial-init` error from a clean interpreter

- run `make test-integration` and watch `tests/integration/test_import_no_cycles.py` — it parametrizes `python -c "import {module}"` over all 162 modules under `src/ccgram/` (Round 5 F5 expansion) and surfaces the offending path.
- the F6.3 audit + Round 5 F5 captured legitimate cycles with `# Lazy: <cycle path>` comments at the in-function import; do not blindly hoist those — `make lint` runs `lint-lazy` which fails on undocumented late imports.

Symptom: `lint-lazy` fails with "undocumented in-function import"

- the in-function import lacks a `# Lazy: <reason>` comment on the line immediately preceding it. Either hoist it (verify with `python -c "import {module}"`) or add the annotation citing the cycle path or wiring contract that requires lateness. See `scripts/lint_lazy_imports.py`.

Symptom: `test_query_layer_only_for_handlers` fails

- a handler file added a new `session_manager.<attr>` access that is not on the documented write/admin allow-list (Round 5 F2). Either route the read through `window_query` / `session_query`, or add the new method name to the allow-list constant in the test if it is genuinely a write/admin call.

Symptom: `test_polling_types_purity` fails (subprocess or AST check)

- `polling_types.py` imported a stateful module (likely `polling_state.py` or another non-stdlib module besides `ccgram.providers.base`). The pure-types invariant requires `polling_types.py` to load without pulling in any singletons. Move the offending import to `polling_state.py` or to the call site instead.
