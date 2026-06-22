# System Architecture

Component flow:

- Telegram Bot (`bot.py` + `handlers/`) drives outbound to tmux via `send_keys` and receives inbound via SessionMonitor callbacks. Handler registration in `handlers/registry.py`. Post_init wiring in `bootstrap.py`. Outbound formatting goes through `entity_formatting.py` (MD → plain + `MessageEntity`) and `telegram_sender.py` (`split_message`, 4096 limit). Per-user FIFO queue + worker + rate limiting in `messaging_pipeline/`. Terminal parsing via pyte (`screen_buffer.py`, `terminal_parser.py`).
- SessionMonitor (`session_monitor.py`) polls JSONL transcripts (2s, mtime cache, byte-offset incremental reads) and reads `events.jsonl` incrementally for instant hook dispatch.
- Multiplexer seam (`multiplexer/`) abstracts the terminal multiplexer behind the `Multiplexer` Protocol. tmux is the default backend (`multiplexer/tmux.py`, `TmuxManager`): list/find/create/kill windows, `send`, `capture`, `list_panes`, `send_to_pane`. Callers import the module-level `multiplexer` proxy, never a concrete backend.
- TranscriptParser (`transcript_parser.py`) parses JSONL, pairs tool_use ↔ tool_result, emits expandable quotes for thinking/history.
- Hook (`hook.py`) receives Claude Code hook stdin, writes `session_map.json` + `events.jsonl`.
- SessionManager + ThreadRouter resolve window ↔ session, own thread bindings and message history.
- State files in `~/.ccgram/`: `state.json` (thread bindings, window states, display names, read offsets), `session_map.json` (hook-generated window_id → session), `events.jsonl` (append-only hook event log), `monitor_state.json` (byte offsets per JSONL file).

Claude session transcripts live under `~/.claude/projects/` (`sessions-index` + `*.jsonl`).

## Module Inventory

### `multiplexer/`

Backend-neutral terminal-multiplexer seam (mirrors the `providers/` seam). Callers import the module-level `multiplexer` proxy and type against `multiplexer.base.Multiplexer`; they must not import a concrete backend (`multiplexer.tmux`/`multiplexer.herdr`). Enforced by the F1 boundary audit (`tests/ccgram/test_multiplexer_boundary.py`). Backend selected by `CCGRAM_MULTIPLEXER` (default `tmux`), wired in `bootstrap.py` from `config.multiplexer_name`.

- `base.py` — core, pure: `Multiplexer` Protocol, `MultiplexerCapabilities` dataclass (`name` + six capability flags: `ids_stable_across_restart`, `exposes_pane_tty`, `native_agent_status`, `read_max_lines`, `self_identify_env`, `supports_event_stream`), and neutral value types `WindowRef`, `PaneInfo`, `CaptureResult`, `ForegroundInfo`, `PaneDims`. No backend imports, no I/O library (F3 core-purity audit).
- `tmux.py` — adapter: `TmuxManager` (tmux backend) satisfying `Multiplexer`, returns the neutral value types, exposes tmux `capabilities` (`ids_stable_across_restart=True`, `exposes_pane_tty=True`, `native_agent_status=False`, `read_max_lines=None`, `self_identify_env="TMUX_PANE"`, `supports_event_stream=False`). Owns the single `tmux_manager` singleton. `foreground(window_id)` is the sole `pane_tty` + `ps -t <tty>` site in the codebase (the private detail behind the seam's foreground source); shell tool-detection and provider auto-detection read it through the `multiplexer.foreground()` proxy, never a tty.
- `herdr.py` — adapter: `HerdrManager` (herdr backend) satisfying `Multiplexer`, an anti-corruption layer over the `herdr` CLI/socket. All herdr JSON shapes and `wN:tN`/`wN:pN` ids stay private; methods return the neutral value types. **Tab identity:** `window_id = tab_id` (`"wN:tM"`) — one ccgram topic = one herdr tab; a split tab (agent team) is one topic with N panes. I/O-free constructor with an injectable command runner (unit-tested with JSON fixtures); pins `HERDR_PROTOCOL_VERSION` from `herdr status` and refuses on mismatch (`HerdrProtocolError`). `list_windows` builds one `WindowRef` per tab (not per pane): resolves workspace labels from `workspace list`, tab labels from `tab list`, representative agent from `pane list` (focused pane's `display_agent`/`agent`, else first non-empty); stamps `format_agent_topic_prefix(workspace_label, tab_label)` → `"<workspace> ▸ <tab>"` into `WindowRef.window_name`; skips workspace/tab labels matching `^__.*__$` so ccgram never auto-adopts itself. `create_window` returns the new **tab id** as `window_id`. All pane ops resolve the tab id to its active pane via `_active_pane(tab_id)` (focused pane, else first); `list_panes(tab_id)` returns all panes in the tab (multi-pane awareness). `kill_window` issues `tab close`; `rename_window` issues `tab rename`. herdr `capabilities` (`ids_stable_across_restart=False`, `exposes_pane_tty=False`, `native_agent_status=True`, `read_max_lines=1000`, `self_identify_env="HERDR_PANE_ID"`, `supports_event_stream=True`). `foreground(window_id)` maps `pane process-info.foreground_processes[]` → `ForegroundInfo` (pid/argv/cwd; `pgid` from `foreground_process_group_id`) — no tty, since `exposes_pane_tty=False`. Scrollback captures clamp to `read_max_lines` (1000) and set `CaptureResult.truncated` so a >1000-line command surfaces truncation instead of silently dropping output. Session-map key is `herdr:<tab_id>`; `session_map_prefix()` used in cleanup/lifecycle/transcript_discovery (backend-neutral key surface). CB_PANE_DELIMITER `"|"` used for pane callback encoding (avoids collision with the colon in herdr ids). Workspace picker step in `/new` flow gated on `native_agent_status`. `_resolve_by_session_id` handles `herdr:<tab_id>` keys for restart re-resolution (stale tab_id → new tab_id via shared session_id). Contract validated by `tests/ccgram/test_herdr_backend.py` (unit) + `tests/integration/test_herdr_contract.py` (`-m herdr`, live socket); shell-on-herdr by `tests/integration/test_shell_herdr.py` (`-m herdr`).
- `registry.py` — `get_multiplexer(name)` + singleton cache (mirrors `providers/registry.py`); backends (`tmux`, `herdr`) imported lazily inside their factory so the core stays I/O-free. `UnknownMultiplexerError` on unknown names.
- `__init__.py` — `multiplexer` proxy (forwards to the wired backend; raises a clear "not wired" error before bootstrap), `install_multiplexer`/`get_active_multiplexer` wiring, re-exports `get_multiplexer`.
- `vim_state.py` — backend-neutral vim-insert detection cache (`_vim_state`/`_vim_locks`, `notify_vim_insert_seen`). Lives outside the tmux backend so the polling layer can import the detection helpers without importing a concrete backend (F1 boundary).
- `window_ops.py` — backend-neutral `send_to_window`/`send_followup_to_window` convenience wrappers over the active `multiplexer` proxy + thread router.
- `self_identify.py` — backend-neutral hook identity resolver (`resolve_self_identity(env)` → `SelfIdentity`). Picks the backend by which `self_identify_env` var is present (`$TMUX_PANE` → tmux via injected `tmux_query`; `$HERDR_PANE_ID` → herdr). I/O-free: the tmux `display-message` probe is injected by `hook.py` (its `_resolve_window_id`), so the hook (separate process) keeps the subprocess and the resolver stays table-testable. The tmux branch is byte-identical to the previous direct path.
- `topic_mapping.py` — backend-neutral projection of multiplexer windows onto Telegram topics (consumer of the seam, not part of the `Multiplexer` contract). Two pure helpers. `is_agent_topic_window(window, caps)`: the capability-gated discovery filter for "topic = tab (herdr) / topic = window (tmux)". On `native_agent_status` backends (herdr) only tabs running an agent (non-empty `WindowRef.pane_current_command`) surface as topics — a bare shell tab does not; on tmux every window is eligible (unchanged). `format_agent_topic_prefix(workspace, tab)`: renders the herdr adaptive topic label `"<workspace> ▸ <tab>"` (tab name is primary so two tabs in the same workspace get distinct titles; missing parts degrade gracefully), which the herdr adapter stamps into `WindowRef.window_name`; the status emoji is prepended later by `topic_emoji`. Imported by `session_monitor._emit_unbound_window_events` (discovery) and `multiplexer.herdr` (labels). Lives outside the tmux backend so core + handlers can use it without crossing the F1 boundary.

### `providers/`

- `base.py` — `AgentProvider` protocol, `ProviderCapabilities`, event types.
- `registry.py` — `ProviderRegistry` (name→factory, singleton cache).
- `_jsonl.py` — shared JSONL parsing base for Codex + Gemini + Pi.
- `claude.py`, `codex.py`, `gemini.py`, `pi.py` — provider implementations.
- `pi_format.py` — Pi transcript parsers (user/assistant/toolResult/bashExecution, session header, pending-tool tracking).
- `pi_discovery.py` — Pi command discovery (builtins + skills + prompts + `pi.registerCommand` scans).
- `codex_status.py`, `codex_format.py` — Codex status snapshot + permission/tool prompt formatter.
- `shell.py` — slim ShellProvider (re-exports from `shell_infra`).
- `shell_infra.py` — prompt-marker detection, `KNOWN_SHELLS`, `PromptMatch`, `setup_shell_prompt`.
- `process_detection.py` — provider classification from a pane's foreground process. Consumes `ForegroundInfo.argv` resolved through the seam (`multiplexer.foreground(window_id)`); never touches a tty or forks `ps` (the backend owns that — tmux `foreground()` via `ps -t <tty>`, herdr via `pane process-info`). Skips wrapper tokens, matches provider patterns, caches the result. No-tty boundary enforced by `tests/ccgram/test_no_tty_outside_backend.py`.
- `__init__.py` — `get_provider_for_window`, `detect_provider_from_pane`, `detect_provider_from_command`, `get_provider`.

### `llm/`

- `base.py` — `CommandGenerator` + `TextCompleter` Protocols, `CommandResult`.
- `httpx_completer.py` — OpenAI-compatible + Anthropic completions via httpx.
- `summarizer.py` — completion summary (reads transcript, single-line summary for Ready).
- `__init__.py` — provider registry + `get_completer()` / `get_text_completer()` factories.

### `whisper/`

- `base.py` — `WhisperTranscriber` Protocol + `TranscriptionResult`.
- `httpx_transcriber.py` — OpenAI-compatible transcription (OpenAI, Groq, …).
- `__init__.py` — `get_transcriber()`.

### `src/ccgram/` (core)

- `bot.py` — PTB Application factory + lifecycle delegates (172 lines); compat re-exports for handlers patched in tests.
- `bootstrap.py` — `bootstrap_application()` (post_init) + `shutdown_runtime()` (post_shutdown). Named steps: `register_provider_commands`, `verify_hooks_installed`, `wire_runtime_callbacks`, `start_session_monitor`, `start_status_polling`, `start_miniapp_if_enabled`. Ordering invariant: `wire_runtime_callbacks` must run before `start_session_monitor` — the monitor dispatches approval prompts to `register_approval_callback`, which raises if unwired.
- `telegram_client.py` — `TelegramClient` Protocol covering 18 grep-verified bot API methods. `PTBTelegramClient(bot)` adapter; `FakeTelegramClient` for tests. `unwrap_bot(client)` is the escape hatch for PTB-only helpers (`do_api_request` for `DraftStream`).
- `cc_commands.py` — CC command discovery (skills, custom) + menu registration.
- `command_catalog.py` — provider-agnostic command discovery and caching.
- `claude_task_state.py` — Claude task tracking from transcripts; per-window snapshots for live status bubble.
- `cli.py` — Click CLI entry (run + bot-config flags).
- `config.py` — application config singleton (env, .env, defaults).
- `doctor_cmd.py` — `ccgram doctor [--fix]`.
- `monitor_state.py` — byte-offset persistence per session.
- `main.py` — Click dispatcher + run_bot bootstrap.
- `screen_buffer.py` — pyte VT100 buffer (ANSI → clean lines, separator detection).
- `screenshot.py` — terminal text → PNG (ANSI color, font fallback).
- `session.py` — `SessionManager` constructs and owns `WindowStateStore`, `ThreadRouter`, `UserPreferences`, `SessionMapSync` via constructor DI with explicit `schedule_save` and store-specific callbacks.
- `session_map.py` — reads/writes `session_map.json`, syncs window states against hook data.
- `session_query.py` — read-only session resolution free functions wrapping `session_resolver`.
- `session_resolver.py` — JSONL session resolution + message history extraction.
- `state_persistence.py` — atomic/debounced JSON persistence for `state.json`.
- `status_cmd.py` — `ccgram status`.
- `telegram_request.py` — resilient long-polling helpers (custom HTTPX transport).
- `thread_router.py` — thread bindings, display names, reverse index, chat ID resolution. Constructed by `SessionManager`; module-level `thread_router` is a proxy.
- `toolbar_config.py` — per-provider button grids from TOML.
- `topic_state_registry.py` — registry for per-topic/per-window cleanup functions with self-registration decorator and `register_bound()` for instance methods.
- `user_preferences.py` — directory favorites + per-user read offsets. Constructed by `SessionManager`; module-level `user_preferences` is a proxy.
- `utils.py` — `ccgram_dir`, `tmux_session_name`, `atomic_write_json`.
- `window_query.py` — read-only window state free functions for handlers; delegates feature-shaped reads to `window_state_ports/*`.
- `window_resolver.py` — window ID resolution, format helpers, startup migration.
- `window_state_store.py` — `WindowState` dataclass + persistence kernel. Remains the only persisted window-state model. Includes `provider_manual_override` (set by `/agent`, blocks `_detect_and_apply_provider`; serialized only when `True`). Constructed by `SessionManager`; module-level `window_store` is a proxy.
- `window_state_ports/` — feature-port package (`pane_state`, `identity_state`, `worktree_state`, `tool_state`, `lifecycle_state`). Thin adapters over `WindowStateStore` exposing frozen projection dataclasses and cohesive feature writes (pane upsert/remove/lifecycle, worktree metadata, batch mode, tool-call visibility, origin, provider-manual-override). Provider changes still route through `SessionManager.set_window_provider`. Sole approved raw `WindowState`-field access site outside `window_state_store.py`, `session.py`, and `window_query.py`; enforced by `tests/ccgram/test_window_state_access_audit.py`.
- `window_view.py` — read-only `WindowView` projection (frozen snapshot).
- `expandable_quote.py` — sentinel constants + `format_expandable_quote()` (markup contract between parsers and presentation).

### `handlers/`

Grouped into 14 feature subpackages. Each subpackage `__init__.py` re-exports the public surface; call sites use subpackage-qualified imports. Handlers depend on `TelegramClient` Protocol, not `telegram.Bot`.

Top-level (constants, leaves, top-level commands):

- `agent_command.py` — `/agent` (alias `/provider`) command for manual provider override. Picker UI with `(manual override)` badge + `🔄 Auto`. Sets `WindowState.provider_manual_override` so `_detect_and_apply_provider` skips the window; clears stale `transcript_path` and session_map entry so SessionMonitor stops polling the wrong transcript.
- `callback_data.py` — `CB_*` callback data constants.
- `callback_helpers.py` — `user_owns_window`, `get_thread_id`.
- `callback_registry.py` — prefix-based callback dispatch with self-registration decorator.
- `cleanup.py` — topic teardown via TopicStateRegistry + async bot cleanup.
- `command_history.py` — per-user/per-topic in-memory command recall (max 20).
- `file_handler.py` — photo/document handler (save to `.ccgram-uploads/`, notify agent).
- `hook_events.py` — dispatcher for `Stop`, `StopFailure`, `SessionEnd`, `Notification`, `Subagent*`, `Team*`.
- `inline.py` — `inline_query_handler`, `unsupported_content_handler` (documented exception: no feature subpackage).
- `last_reply.py` — `/last` command + `send_last_reply` backend; AI path walks the transcript for the last assistant turn, shell path extracts last command+output via prompt markers; overflows >4096 chars to a `.txt` document upload.
- `reactions.py` — Telegram message reactions helper (Bot API 7.0+).
- `registry.py` — central PTB handler registration (`register_all`): `CommandSpec` table + Message/Callback/Inline handler wiring. Documented exception: only handler module with runtime `from telegram.ext` import — the PTB wiring spine.
- `response_builder.py` — response pagination and formatting.
- `sessions_dashboard.py` — `/sessions` overview + kill.
- `sync_command.py` — `/sync`.
- `upgrade.py` — `/upgrade` (`uv tool upgrade` + restart).
- `user_state.py` — `context.user_data` string key constants.

`handlers/commands/` — `/commands` + `/toolbar` orchestration:

- `__init__.py` — `commands_command`, `toolbar_command`; re-exports `forward_command_handler`, `setup_menu_refresh_job`, `get_global_provider_menu`, `set_global_provider_menu`, `sync_scoped_*`.
- `forward.py` — `forward_command_handler`, `_handle_clear_command`. Forwards every `/<token>` to the active provider; unknown commands caught reactively by `failure_probe`.
- `menu_sync.py` — provider menu cache + scoped sync (`sync_scoped_provider_menu`, `sync_scoped_menu_for_text_context`, `setup_menu_refresh_job`, LRU helpers, `_build_provider_command_metadata`).
- `failure_probe.py` — `_capture_command_probe_context`, `_probe_transcript_command_error`, `_spawn_command_failure_probe`.
- `status_snapshot.py` — `_status_snapshot_probe_offset`, `_maybe_send_status_snapshot`.

`handlers/interactive/` — interactive UI prompts:

- `interactive_ui.py` — AskUserQuestion / ExitPlanMode / Permission UI rendering.
- `interactive_callbacks.py` — callbacks (arrow keys, enter, esc).

`handlers/live/` — live view + screenshots:

- `live_view.py` — auto-refreshing terminal via `editMessageMedia`, content-hash gating, auto-stop.
- `screenshot_callbacks.py` — capture, quick-key, live view toggle.
- `pane_callbacks.py` — per-pane rename, screenshot select.

`handlers/messaging_pipeline/` — outbound message queue:

- `message_queue.py` — per-user FIFO + worker; merge, status dedup, tool-use batching. Worker takes `client: TelegramClient`.
- `message_routing.py` — routes new assistant messages from SessionMonitor to Telegram topics.
- `message_sender.py` — `safe_reply`/`safe_edit`/`safe_send`, `rate_limit_send_message`, `edit_with_fallback`. All take `client: TelegramClient`.
- `message_task.py` — dependency-free sum type (`ContentTask`, `StatusTask`, `ToolResultTask`) shared by queue, tool_batch, status_bubble.
- `tool_batch.py` — Claude tool-use batching: state machine, formatting, edit-in-place. Uses `unwrap_bot(client)` for `DraftStream`.
- `topic_commands.py` — `/verbose` and `/toolcalls` per-topic toggles.

`handlers/polling/` — status polling + per-window tick:

- `polling_coordinator.py` — iterates thread bindings, delegates per-window work to `window_tick`, runs periodic/lifecycle tasks.
- `polling_types.py` — pure types module: `TickContext`, `TickDecision`, `PaneTransition`, `WindowPollState`, `TopicPollState`, constants (`STARTUP_TIMEOUT`, `RC_DEBOUNCE_SECONDS`, `MAX_PROBE_FAILURES`, `TYPING_INTERVAL`, `PANE_COUNT_TTL`, `ACTIVITY_THRESHOLD`, `SHELL_COMMANDS`), pure `is_shell_prompt`. Imports stdlib + `ccgram.providers.base.StatusUpdate` only.
- `polling_state.py` — stateful: `TerminalPollState`, `TerminalScreenBuffer`, `InteractiveUIStrategy`, `TopicLifecycleStrategy`, `PaneStatusStrategy`, the five module-level singletons, `reset_window_polling_state`.
- `periodic_tasks.py` — topic lifecycle management, live view ticking, state pruning.
- `window_tick/__init__.py` — `tick_window` (thin orchestrator).
- `window_tick/decide.py` — pure decision kernel (`decide_tick`, `build_status_line`, `is_shell_prompt`). Zero deps on tmux/PTB/singletons.
- `window_tick/observe.py` — pure inputs → `TickContext` (pane-text capture, last-activity lookup, screen-buffer parsing, status resolve, vim-insert detection).
- `window_tick/apply.py` — DI-heavy side effects: `_apply_*_transition`, `_update_status`, `_send_typing_throttled`, `_handle_dead_window_notification`, `_scan_window_panes`, pane forwarding.

`handlers/recovery/` — dead window recovery + history:

- `recovery_callbacks.py` — thin dispatcher (~170 LOC): `_dispatch`, `handle_recovery_callback`, shared `_validate_recovery_state`/`_clear_recovery_state` validators.
- `recovery_banner.py` — dead-window banner UX: `RecoveryBanner`, `render_banner`, `build_recovery_keyboard`, `_create_and_bind_window`, fresh/continue/resume/back/browse/cancel handlers.
- `resume_picker.py` — resume picker UX + transcript scan: `_SessionEntry`, `scan_sessions_for_cwd`, `_scan_index_for_cwd`, `_scan_bare_jsonl_for_cwd`, picker keyboard builders, `_handle_resume_pick`.
- `restore_command.py` — `/restore`.
- `resume_command.py` — `/resume` (scan past sessions, paginated picker).
- `transcript_discovery.py` — hookless transcript discovery for Codex/Gemini, provider auto-detection, shell↔agent transitions.
- `history.py` + `history_callbacks.py` — `/history` + pagination.

`handlers/send/` — `/send` file delivery:

- `send_command.py` — search, list, upload utilities.
- `send_callbacks.py` — browser navigation.
- `send_security.py` — multi-layer access control.

`handlers/shell/` — shell provider command flow:

- `shell_commands.py` — NL→command approval, dangerous command detection via LLM.
- `shell_capture.py` — prompt-marker output isolation, exit code detection, baseline-diff fallback, glyph stripping.
- `shell_context.py` — shared helpers (`gather_llm_context`, `redact_for_llm`, `_detect_shell_tools`).
- `shell_prompt_orchestrator.py` — single `ensure_setup` entry point centralizing five trigger sites.

`handlers/status/` — status bubble + topic emoji:

- `status_bubble.py` — keyboard + status message lifecycle (`_status_msg_info`, `send_status_text`, `clear_status_message`, `build_status_keyboard`).
- `status_bar_actions.py` — button callbacks (last reply, get file, recall, esc, keys).
- `topic_emoji.py` — topic name emoji updates (active/idle/done/dead + RC/YOLO badges), debounced. Color scheme via `CCGRAM_STATUS_MODE`.
- `rc_probe.py` — Claude `/remote-control` outcome probe: `arm_rc_probe`, pure `classify_rc_output`, `_classify_loop`. De-duped via `WindowState.rc_probe_state` (in-memory).

`handlers/text/` — `text_handler.py` (UI guards → unbound → dead → forward).

`handlers/toolbar/` — `/toolbar` inline keyboard:

- `toolbar_keyboard.py` — builder from TOML config with per-window label overrides.
- `toolbar_callbacks.py` — dispatch for inline button clicks.

`handlers/topics/` — topic lifecycle + window picker:

- `topic_orchestration.py` — new window/topic creation, unbound window adoption, rate limiting.
- `topic_lifecycle.py` — autoclose timers for done/dead topics, unbound window TTL.
- `directory_browser.py` — directory selection UI + worktree picker/confirm keyboard builders.
- `directory_callbacks.py` — navigate, confirm, provider pick, worktree flow.
- `worktree.py` — pure git-worktree plumbing: `check_worktree_eligibility`, `suggest_branch_name`, `slug_for_path`, `worktree_path_for`, `validate_branch_name`, `create_worktree` (raises `WorktreeError`). No Telegram/tmux/state deps.
- `window_callbacks.py` — bind, new, cancel.
- `new_command.py` — `/new` and `/start`.

`handlers/voice/` — voice transcription:

- `voice_handler.py` — download, transcription, confirm keyboard.
- `voice_callbacks.py` — `vc:send`/`vc:drop` routing; shell-provider transcriptions route through LLM.

## Key Design Decisions

- Topic-centric. Each Telegram topic binds to one tmux window. Topics _are_ the session list; no centralized session list.
- Window-ID-centric. All internal state keyed by tmux window ID (e.g. `@0`, `@12`), unique within a tmux server session. Names are display labels in `window_display_names`. Same directory may have multiple windows.
- Hook-based events. Claude Code hooks write `session_map.json` + `events.jsonl`. SessionMonitor reads both: session_map for tracking, `events.jsonl` for instant dispatch (interactive UI, done, API error alert, session lifecycle, subagent, team). Terminal scraping is fallback. Missing hooks logged at startup with fix command.
- Multi-pane awareness. Windows with multiple panes (e.g. agent teams) are scanned for interactive prompts in non-active panes. Blocked panes surfaced as inline keyboard alerts. `/panes` lists all panes with status + per-pane screenshot. Callback data includes pane_id: `"aq:enter:@12:%5"`.
- Tool use ↔ tool result pairing. `tool_use_id` tracked across poll cycles; result edits the original tool_use Telegram message in place.
- Entity-based formatting. All messages go through `safe_reply`/`safe_edit`/`safe_send` (markdown → plain + `MessageEntity` via `telegramify-markdown`, fallback to plain). No parse errors possible.
- No truncation at parse layer. Splitting only at send layer; respects 4096 char limit with expandable quote atomicity.
- Only sessions in `session_map.json` (via hook) are monitored.
- Notifications routed via thread bindings (topic → window_id → session).
- Startup re-resolution. Window IDs reset on tmux server restart. `resolve_stale_ids()` matches persisted display names against live windows to re-map. Old name-keyed `state.json` auto-migrated.
- Per-window provider. CLI-specific behavior (launch args, transcript parsing, status, command discovery) delegated to `AgentProvider`. `ProviderCapabilities` gate UX per-window: hook checks, resume/continue buttons, command registration. `WindowState.provider_name` is source of truth; `get_provider_for_window(window_id)` resolves with config-default fallback. External windows auto-detected via `detect_provider_from_command()`. `get_provider()` is the no-window-context fallback (`doctor`, `status`).
- Multiplexer seam. Terminal-multiplexer access is abstracted behind the `Multiplexer` Protocol (`multiplexer/base.py`, core, I/O-free). tmux is the default backend (`multiplexer/tmux.py`); a `CCGRAM_MULTIPLEXER` switch (default `tmux`) selects it, wired in `bootstrap.py` from `config.multiplexer_name`. Callers depend only on the module-level `multiplexer` proxy and `MultiplexerCapabilities` flags — never a concrete backend or `name == "<backend>"` conditional. Boundary enforced by `tests/ccgram/test_multiplexer_boundary.py` (F1, concrete-backend imports forbidden outside `multiplexer/**`, `bootstrap.py`, `main.py`), contract by `tests/ccgram/test_multiplexer_contract.py` (F2, parametrized per backend), core purity by `tests/integration/test_import_no_cycles.py` (F3, `base` imports no backend/I/O). Mirrors the `AgentProvider` seam and the `window_store`/`thread_router` proxy pattern.
- Foreground process via the seam (no-tty boundary). `Multiplexer.foreground(window_id) -> ForegroundInfo` is the single source of foreground-process truth for shell foreground detection (`shell_infra.detect_pane_shell`/`_is_interactive_shell`) and provider auto-detection (`process_detection`). The tty/`ps -t` mechanism is a private detail of `multiplexer/tmux.py`'s `foreground()`; herdr's `foreground()` reads `pane process-info` (no tty, since `exposes_pane_tty=False`). No module outside `multiplexer/tmux.py` references `pane_tty`, `ps -t`, or `get_foreground_args`. Enforced by the no-tty drift gate `tests/ccgram/test_no_tty_outside_backend.py` (AST/source walk, allow-listing only the tmux backend), modeled on the window-state access audit.
- Live terminal view. Auto-refreshing screenshots via `editMessageMedia` (default 5s). Content-hash gating skips API calls when unchanged. One active view per topic, auto-stop after timeout (default 300s). Managed by `handlers/live/live_view.py`, ticked from `handlers/polling/periodic_tasks.py`.
- Completion summaries. On agent Stop, `llm/summarizer.py` reads transcript, produces one line, edits Ready message in place. Non-blocking: static enriched Ready appears immediately, LLM enhancement ~1-2s later.
- Constructor DI for stores. `SessionManager` constructs `WindowStateStore`, `ThreadRouter`, `UserPreferences`, `SessionMapSync` with explicit `schedule_save` (and store-specific) callbacks. Module-level singletons are proxy objects forwarding to the wired instance. `register_*_callback` helpers raise on double-registration; unwired callees raise `RuntimeError("not wired")`.
- `TelegramClient` Protocol. Handlers depend on the Protocol (`src/ccgram/telegram_client.py`), not `telegram.Bot`. Allowed runtime `from telegram.ext` importers: `bot.py`, `bootstrap.py`, `handlers/registry.py`, `telegram_client.py`, `telegram_request.py`, `telegram_sender.py`. Everything else uses `if TYPE_CHECKING:`. `unwrap_bot(client)` is the escape hatch for PTB-only helpers.
- Pure decision kernel for window tick. `handlers/polling/window_tick/decide.py` is pure (zero deps on tmux/PTB/singletons), `observe.py` produces `TickContext`, `apply.py` is the only side-effect file. `decide_tick` and helpers unit-tested without mocks.
- Pure types vs stateful split for polling. `polling_types.py` holds contracts (stdlib + `StatusUpdate` only); `polling_state.py` holds strategies + singletons. `decide.py` imports only from `polling_types`. Codified by `tests/ccgram/handlers/polling/test_polling_types_purity.py`.
- Single read path through query layer. Handler reads of window/session state go through `window_query` / `session_query` free functions or `window_state_ports/*` feature projections. Direct `session_manager.<attr>` access in `handlers/**` is restricted to a documented write/admin allow-list (`set_window_provider`, `set_window_origin`, `set_window_approval_mode`, `set_window_worktree`, `cycle_*`, `audit_state`, `prune_*`, `sync_display_names`). Codified by `tests/ccgram/test_query_layer_only_for_handlers.py` (AST walk over 86 handler files).
- Window-state feature ports. `WindowStateStore` remains the single persistence kernel. `window_state_ports/{pane,identity,worktree,tool,lifecycle}_state.py` expose frozen projection dataclasses and cohesive feature writes. Reads return projections, not raw `WindowState`; writes only touch fields owned by the port. Provider/session identity writes still delegate to `SessionManager.set_window_provider` to preserve capability coordination. Boundary enforced by `tests/ccgram/test_window_state_access_audit.py`: raw feature-field access outside `window_state_store.py`, `window_state_ports/*`, `session.py`, `window_query.py`, and serialization tests fails the audit. A second import-boundary check (`tests/ccgram/test_window_store_import_boundary.py`) forbids handler/Mini App modules from importing `window_state_store.window_store` directly; the only allowed exceptions are `handlers/status/rc_probe.py` (transient in-memory RC-probe state never persisted) and `handlers/commands/forward.py` (`clear_window_session` coordination).
- Lazy-import contract. `scripts/lint_lazy_imports.py` flags every in-function `Import`/`ImportFrom` not preceded by `# Lazy:`, not inside `if TYPE_CHECKING:`, and not inside `_reset_*_for_testing`. Walker recurses through compound statements (try/except/finally/if/else/with/for/while) and nested def/class bodies. Multi-line `# Lazy:` blocks supported. Wired into `make lint` as `lint-lazy`. All in-function imports annotated. Cycle test (`tests/integration/test_import_no_cycles.py`) enumerates all modules under `src/ccgram/`.
