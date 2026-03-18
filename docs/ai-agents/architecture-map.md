# Architecture Map

## Runtime Layers

1. CLI and bootstrap

- `src/ccgram/main.py` starts logging and launches the PTB application.
- `src/ccgram/cli.py` maps CLI flags to env vars before config loads.

2. Bot orchestration

- `src/ccgram/bot.py` wires handlers and owns callback dispatch.
- Topic routing and authorization checks live here.

3. Session and monitor core

- `src/ccgram/session.py` is the state hub (thread bindings, window states, offsets).
- `src/ccgram/session_monitor.py` tails transcripts/events and emits parsed messages.
- `src/ccgram/monitor_state.py` persists byte offsets for incremental reads.

4. Provider abstraction

- `src/ccgram/providers/base.py` defines the provider contract.
  - `discover_transcript(cwd, window_key, *, max_age=None)` is the hookless discovery contract (used by Codex/Gemini; `max_age=0` disables staleness checks for alive panes).
- `src/ccgram/providers/__init__.py` resolves per-window provider selection.
- `src/ccgram/providers/{claude,codex,gemini}.py` implement provider-specific behavior.
- `src/ccgram/command_catalog.py` discovers provider commands from filesystem (skills, custom commands) with 60s TTL caching.
- `src/ccgram/cc_commands.py` registers discovered commands as Telegram bot menu entries.
- `src/ccgram/interactive_prompt_formatter.py` normalizes provider interactive prompt text for Telegram readability (currently Codex edit approvals).
- `src/ccgram/codex_status.py` extracts Codex status snapshots from JSONL transcripts.
- `src/ccgram/screenshot.py` renders terminal text to PNG (PIL, ANSI color, font fallback).

5. Integrations

- `src/ccgram/tmux_manager.py` is the tmux IO boundary.
- `src/ccgram/hook.py` writes Claude hook events to both `session_map.json` and `events.jsonl`.

## Request/Response Lifecycles

Inbound user message (Telegram -> tmux):

1. PTB handler entry in `bot.py`.
2. `handlers/text_handler.py` validates context and resolves topic binding.
3. `session.py` maps `(user_id, thread_id)` -> `window_id`.
4. `tmux_manager.py` sends keys to the mapped window/pane.

Outbound agent output (provider transcript/event -> Telegram):

1. `session_monitor.py` polls tracked transcript/event sources incrementally.
2. Provider parser (`providers/*.py` + `transcript_parser.py`/`terminal_parser.py`) emits normalized updates.
3. `handlers/message_queue.py` enforces ordering, merge rules, and rate limits.
4. Telegram send helpers deliver messages and status updates.

Recovery flow (dead/missing session):

1. `handlers/status_polling.py` detects stale/dead bindings.
2. Recovery UI callbacks route through `handlers/recovery_callbacks.py`.
3. Session/window state is updated in `session.py` and persisted to `state.json`.

Commands menu flow (`/commands`):

1. User invokes `/commands` in a topic.
2. `handlers/` routes to command handler in `bot.py`.
3. `command_catalog.py` discovers available commands for the window's provider (filesystem scan with 60s TTL cache).
4. `cc_commands.py` renders the scoped command menu as inline keyboard.
5. User selection sends the command text to the agent via `tmux_manager.py`.

## Data Model and State Files

Config/state directory is `~/.ccgram` unless overridden by `CCGRAM_DIR`.

- `state.json`: topic<->window bindings and window metadata.
- `session_map.json`: hook-generated tmux window -> session map.
- `events.jsonl`: append-only hook events stream.
- `monitor_state.json`: monitor byte offsets (session/event files).

Provider transcript sources (read-only):

- Claude: `~/.claude/projects/`
- Codex: `~/.codex/sessions/`
- Gemini: `~/.gemini/tmp/`
  - Gemini discovery matches by `projectHash` (or configured project alias dir) and does not full-scan unrelated project dirs.

## Core Flow

Inbound (Telegram -> agent):

- message enters `bot.py` -> `handlers/text_handler.py` -> resolve bound window in `session.py` -> send keys via `tmux_manager.py`.

Outbound (agent -> Telegram):

- `session_monitor.py` reads transcript/event deltas -> provider parser transforms entries -> `handlers/message_queue.py` orders/rate-limits sends -> Telegram API.

## Design Constraints to Preserve

- one topic = one window mapping.
- internal identity keyed by tmux `window_id` (not window names).
- no parse-layer truncation; splitting only at Telegram send layer.
- per-window provider behavior and capability-gated UI.
- tmux operations stay centralized in `tmux_manager.py`; do not spread raw shell tmux calls across handlers.
- state mutations route through `session.py` + persistence helpers, not ad-hoc JSON writes.
