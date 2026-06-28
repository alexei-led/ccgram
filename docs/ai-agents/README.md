# AI Agent Orientation

Codebase index for AI agents working on `ccgram`. Read in order:

1. `architecture-map.md` — lifecycles and invariants.
2. `codebase-index.md` — where to edit + debug index.
3. `tooling-and-tests.md` — fast test targeting.
4. `extension-and-fix-playbook.md` — recipes.

Authoritative architecture lives in `docs/architecture.md` and `/.claude/rules/architecture.md`. These docs cover request/response lifecycles, decision maps for common tasks, and debug entrypoints by symptom.

## Project Summary

`ccgram` bridges Telegram topics to terminal multiplexer windows running AI coding agents.

- 1 Telegram topic = 1 multiplexer window/tab = 1 provider session.
- Internal identity is `window_id`, not a display name.
- Message parsing preserves full content; splitting only at Telegram send.
- Provider behavior is per-window and capability-driven.

## Non-Negotiable Rules

- Topic-centric routing; one topic ↔ one window/tab.
- Use `window_id` for identity; never key by display name.
- No parse-layer truncation.
- Per-window provider via `WindowState.provider_name` + `ProviderCapabilities`.
- Handlers depend on `TelegramClient` Protocol, never `telegram.Bot`.
- Reads go through `window_query` / `session_query`; direct `session_manager.<attr>` only on the documented allow-list.
- In-function imports must carry `# Lazy: <reason>` (enforced by `lint-lazy`).
