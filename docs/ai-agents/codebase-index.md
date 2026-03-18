# Codebase Index

## Where to Look First

Entry points:

- `src/ccgram/main.py`: process bootstrap.
- `src/ccgram/cli.py`: CLI command surface and config override rules.
- `src/ccgram/bot.py`: handler registration and main user flows.

State and routing:

- `src/ccgram/session.py`: thread bindings, window states, read offsets.
- `src/ccgram/window_resolver.py`: stale window ID migration/re-resolution.
- `src/ccgram/state_persistence.py`: atomic/debounced JSON persistence.

Monitoring and parsing:

- `src/ccgram/session_monitor.py`: polling engine for sessions and hook events.
- `src/ccgram/transcript_parser.py`: Claude transcript parsing and tool pairing.
- `src/ccgram/terminal_parser.py`: terminal status/UI detection.

Telegram handler surface:

- `src/ccgram/handlers/text_handler.py`: text-path orchestrator.
- `src/ccgram/handlers/message_queue.py`: ordering, merge rules, status conversion.
- `src/ccgram/handlers/status_polling.py`: background status and stale-session cleanup.
- `src/ccgram/handlers/directory_browser.py` + `directory_callbacks.py`: new-session UX.
- `src/ccgram/handlers/interactive_ui.py` + `interactive_callbacks.py`: interactive prompt UX.
- `src/ccgram/handlers/sessions_dashboard.py`: `/sessions` dashboard behavior.
- `src/ccgram/handlers/recovery_callbacks.py`: dead window recovery flow (fresh/continue/resume).
- `src/ccgram/handlers/screenshot_callbacks.py`: screenshot refresh, Esc, quick-key, pane screenshots.
- `src/ccgram/handlers/history_callbacks.py`: history pagination callbacks (prev/next).
- `src/ccgram/handlers/hook_events.py`: hook event dispatcher (Notification, Stop, Subagent*, Team*).
- `src/ccgram/handlers/cleanup.py`: centralized topic state teardown on close/delete.
- `src/ccgram/handlers/topic_emoji.py`: debounced topic name emoji updates (active/idle/done/dead).
- `src/ccgram/handlers/file_handler.py`: photo/document upload → `.ccgram-uploads/` → agent notification.
- `src/ccgram/handlers/resume_command.py`: `/resume` scan past sessions + inline picker.
- `src/ccgram/handlers/upgrade.py`: `/upgrade` uv tool upgrade + `os.execv()` restart.
- `src/ccgram/handlers/sync_command.py`: `/sync` state audit + fix button.
- `src/ccgram/handlers/command_history.py`: per-user/per-topic command recall (in-memory, max 20).

Provider and command surface:

- `src/ccgram/providers/`: provider contract and implementations.
- `src/ccgram/command_catalog.py`: provider-agnostic command discovery + 60s TTL caching.
- `src/ccgram/cc_commands.py`: Telegram menu registration from discovered commands.
- `src/ccgram/hook.py`: Claude hook install/status/uninstall and event writes.

Supporting modules:

- `src/ccgram/screenshot.py`: terminal text → PNG rendering (PIL, ANSI color, font fallback).
- `src/ccgram/codex_status.py`: Codex status snapshot from JSONL transcripts.

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
- `src/ccgram/providers/{claude,codex,gemini}.py` for provider-specific behavior.
- `src/ccgram/interactive_prompt_formatter.py` for provider-facing interactive prompt text normalization (currently Codex edit approval readability).

Change Telegram interactive UX:

- `src/ccgram/handlers/interactive_ui.py` and `interactive_callbacks.py`.
- `src/ccgram/handlers/callback_data.py` for callback key contracts.
- `src/ccgram/handlers/message_queue.py` for ordering/merge side effects.

Change command discovery:

- `src/ccgram/command_catalog.py` for filesystem scanning and caching.
- `src/ccgram/cc_commands.py` for Telegram menu registration.

Change screenshot rendering:

- `src/ccgram/screenshot.py` only.

Change tmux behavior:

- `src/ccgram/tmux_manager.py` only.

## Change Mapping by Task Type

Add or change a Telegram command:

- Start in `src/ccgram/bot.py` command wiring.
- Implement behavior in `src/ccgram/handlers/` module.
- Add callback constants in `handlers/callback_data.py` when needed.

Change session binding logic:

- `src/ccgram/session.py` and `src/ccgram/window_resolver.py`.
- Validate persistence compatibility in `tests/ccgram/test_state_migration.py`.

Adjust transcript/status parsing:

- provider-specific parsing in `src/ccgram/providers/*.py`.
- shared parse behavior in `transcript_parser.py` / `terminal_parser.py`.

Touch tmux behavior:

- `src/ccgram/tmux_manager.py` only; avoid shell calls spread across handlers.

## Contracts You Must Not Break

- Keep topic-window identity 1:1 and window-id keyed.
- Preserve tool-use/tool-result pairing and in-order delivery.
- Keep provider logic behind provider interfaces/capabilities.
- Keep parsing full-fidelity; split only in Telegram send path.
- Use `handlers/user_state.py` keys for `context.user_data`; avoid new raw string keys.

## Debug Index

Symptom: messages routed to wrong topic/window

- inspect `thread_bindings` and window IDs in `session.py`.
- confirm callback/thread ID extraction in `handlers/callback_helpers.py`.

Symptom: no new assistant messages

- inspect `session_monitor.py` byte offsets and session map updates.
- verify provider parser compatibility for that window/provider.

Symptom: interactive keyboard not shown

- inspect `handlers/interactive_ui.py` and provider `parse_terminal_status` output.
- check hook events path (`hook.py` -> `handlers/hook_events.py`) for Claude.

Symptom: duplicated or out-of-order status/content messages

- inspect merge/send behavior in `handlers/message_queue.py`.

Symptom: commands menu missing/wrong

- check `command_catalog.py` cache TTL and filesystem scan paths.
- check `cc_commands.py` menu registration and provider scoping.
