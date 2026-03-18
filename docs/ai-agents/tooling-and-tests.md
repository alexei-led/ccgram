# Tooling and Tests

## Build and Validation Commands

Primary command set from `Makefile`:

- `make fmt`
- `make test`
- `make lint`
- `make typecheck`
- `make check` (runs fmt + lint + typecheck + test + integration)
- `make build`

Default local workflow for code changes:

1. `make fmt`
2. `make test`
3. `make lint`
4. `make typecheck`

Before considering work complete, run at least:

- `make check` (full gate: fmt + lint + typecheck + test)

## Toolchain and Libraries

- Python: `>=3.14`
- Package/dependency manager: `uv`
- Telegram framework: `python-telegram-bot`
- tmux integration: `libtmux`
- async/file IO: `aiofiles`
- logging: `structlog`
- terminal parsing: `pyte`
- screenshot rendering: `Pillow`

## Test Layout

- `tests/ccgram/`: unit tests mirroring source modules.
- `tests/integration/`: integration tests for monitor flow, dispatch, tmux manager, state roundtrips.
- `tests/conftest.py`: required test env setup before imports.
- Hypothesis property-based tests: `tests/ccgram/test_message_queue_properties.py`.

## Fast Test Targeting

Use focused test files that match changed modules first, then full test run.

Examples:

- session/state changes -> `tests/ccgram/test_session.py`, `tests/ccgram/test_state_migration.py`
- monitor/parsing changes -> `tests/ccgram/test_session_monitor.py`, `tests/ccgram/test_transcript_parser.py`
- handlers/UI changes -> `tests/ccgram/test_text_handler.py`, `tests/ccgram/test_status_polling.py`, `tests/ccgram/test_bot_callbacks.py`
- command changes -> `tests/ccgram/test_command_catalog.py`, `tests/ccgram/test_commands_command.py`, `tests/ccgram/test_cc_commands.py`
- hook/event changes -> `tests/ccgram/test_hook.py`, `tests/ccgram/test_hook_events.py`, `tests/ccgram/test_session_monitor_events.py`
- cleanup/lifecycle changes -> `tests/ccgram/test_cleanup.py`, `tests/ccgram/test_topic_emoji.py`
- provider changes -> `tests/ccgram/test_provider_contracts.py`, `tests/ccgram/test_jsonl_providers.py`

## Quality Constraints

- all hook/check issues are blocking.
- fix failing checks before proceeding to unrelated work.
- preserve existing architecture constraints (topic-window identity, provider boundaries, send-layer split).
