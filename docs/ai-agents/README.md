# AI Agent Orientation

This folder is the codebase index for AI agents working on `ccgram`.

Use this reading order:

1. [`Architecture Map`](./architecture-map.md)
2. [`Codebase Index`](./codebase-index.md)
3. [`Tooling and Tests`](./tooling-and-tests.md)
4. [`Extension and Fix Playbook`](./extension-and-fix-playbook.md)

## Fast Onboarding Paths

For bug fixes:

1. [`Codebase Index`](./codebase-index.md) (locate the owning module quickly)
2. [`Architecture Map`](./architecture-map.md) (confirm invariants and data flow)
3. [`Tooling and Tests`](./tooling-and-tests.md) (run targeted + full checks)

For new features:

1. [`Architecture Map`](./architecture-map.md) (boundary and flow constraints)
2. [`Codebase Index`](./codebase-index.md) (integration points)
3. [`Extension and Fix Playbook`](./extension-and-fix-playbook.md) (safe extension path)
4. [`Tooling and Tests`](./tooling-and-tests.md) (validation gate)

## Scope

These docs are intentionally selective. They do not list every file or method.
They focus on:

- where core behavior lives,
- where to make changes safely,
- how to validate work before marking tasks done.

## Project Summary

`ccgram` bridges Telegram topics to tmux windows running AI coding agents.
Core mapping model:

- one Telegram topic -> one tmux window (`@id`)
- one tmux window -> one active provider session

The key implementation areas are:

- runtime and startup: `src/ccgram/main.py`, `src/ccgram/cli.py`, `src/ccgram/bot.py`
- session/state core: `src/ccgram/session.py`, `src/ccgram/session_monitor.py`
- providers: `src/ccgram/providers/`
- command discovery: `src/ccgram/command_catalog.py`, `src/ccgram/cc_commands.py`
- Telegram UI handlers: `src/ccgram/handlers/`
- tmux integration: `src/ccgram/tmux_manager.py`

## Non-Negotiable Architecture Rules

- Topic-centric routing only: one Telegram topic maps to one tmux window.
- Internal identity must remain tmux `window_id` based (for example `@3`), never window-name keyed.
- Message parsing must preserve full content; splitting/truncation belongs only in Telegram send logic.
- Provider behavior is per-window and capability-driven; avoid global provider assumptions in window-specific paths.
