# Extension and Fix Playbook

## Common Extension Paths

Add a new provider:

1. Implement provider in `src/ccgram/providers/<name>.py` following `AgentProvider` contract.
2. Register provider in `src/ccgram/providers/__init__.py`.
3. Define capabilities accurately (resume/continue/hook/status behavior).
4. Add provider tests in `tests/ccgram/test_provider_contracts.py` and provider-specific tests.
5. If provider launch requires runtime hardening (for example Gemini shell mode), keep it in `resolve_launch_command()` and cover it with launch-command tests.
6. If you change provider contract signatures (for example `discover_transcript(..., max_age=...)`), update:
   - `src/ccgram/providers/base.py` protocol
   - shared base implementations (`_jsonl.py`, Claude/Codex/Gemini as needed)
   - call sites (status polling/session monitor)
   - contract + behavior tests

Add a new Telegram command or callback:

1. Register command/callback in `src/ccgram/bot.py`.
2. Implement handler in `src/ccgram/handlers/`.
3. Add callback prefix/constant in `handlers/callback_data.py` if needed.
4. Add/adjust tests for routing + handler behavior.

Add session state fields:

1. Extend dataclasses/serialization in `src/ccgram/session.py`.
2. Ensure load path is backward compatible with missing keys.
3. Update migration logic if key semantics change (`window_resolver.py` / migration tests).

Add a new slash command (agent-side):

1. Add command definition to the agent's command surface (e.g. `.claude/commands/` for Claude).
2. `command_catalog.py` discovers it on next scan (60s TTL cache; restart or wait).
3. `cc_commands.py` registers it in the Telegram `/commands` menu automatically.
4. No bot-side code changes needed unless the command requires special Telegram UI.

Add file upload handling:

1. `handlers/file_handler.py` handles photo/document messages.
2. Files are saved to `.ccgram-uploads/` under the config directory.
3. Agent is notified via tmux keys with the file path.
4. Extend `file_handler.py` for new media types or post-processing.

Adjust status or transcript parsing:

1. Keep parsing provider-specific where possible.
2. Preserve message queue ordering and tool-use/tool-result pairing semantics.
3. Validate with parser unit tests and monitor integration tests.

## Bug-Fix Triage

1. Localize the layer first:

- routing/state (`session.py`)
- monitor/parsing (`session_monitor.py`, providers, parsers)
- delivery/UI (`handlers/*`, `message_queue.py`)
- integration boundary (`tmux_manager.py`, `hook.py`)

2. Reproduce with narrow tests:

- start with module-local tests, then run broader suites.

3. Fix with architecture-safe changes:

- avoid bypassing SessionManager state model.
- avoid handler-to-handler tight coupling when shared helper/module fits better.

4. Re-run checks:

- `make fmt && make test && make lint`
- then `make typecheck` (or `make check` for full gate)

## Safe Change Checklist

- uses existing abstractions (`session_manager`, provider protocol, tmux manager).
- no regressions in topic<->window identity behavior.
- no direct raw-string `context.user_data` keys; use `handlers/user_state.py` constants.
- tests updated for changed behavior.
- formatting, lint, and tests pass.
