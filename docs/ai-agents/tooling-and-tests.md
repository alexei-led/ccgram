# Tooling and Tests

Commands and toolchain are in `docs/guides.md`. This file is for fast test targeting.

## Workflow

For code changes: `make fmt && make test && make lint && make typecheck`.
Before declaring complete: `make check` (fmt + lint + typecheck + test + integration).

All hook/check issues are blocking. Fix failing checks before unrelated work.

## Fast Test Targeting

Match changed modules to tests, then run the full suite.

Session/state: `tests/ccgram/test_session.py`, `tests/ccgram/test_window_state_store.py`.

Monitor/parsing: `tests/ccgram/test_session_monitor.py`, `tests/ccgram/test_transcript_parser.py`.

Handlers/UI: `tests/ccgram/handlers/text/test_text_handler.py`, `tests/ccgram/handlers/polling/test_polling_coordinator.py`, `tests/ccgram/handlers/polling/test_polling_strategies.py`, `tests/ccgram/handlers/test_bot_callbacks.py`.

Commands: `tests/ccgram/test_command_catalog.py`, `tests/ccgram/test_commands_command.py`, `tests/ccgram/test_cc_commands.py`, `tests/ccgram/handlers/commands/test_forward.py`, `tests/ccgram/handlers/commands/test_menu_sync.py`, `tests/ccgram/handlers/commands/test_failure_probe.py`, `tests/ccgram/handlers/commands/test_status_snapshot.py`.

Hook/events: `tests/ccgram/test_hook.py`, `tests/ccgram/handlers/test_hook_events.py`, `tests/ccgram/test_session_monitor_events.py`.

Cleanup/lifecycle: `tests/ccgram/handlers/test_cleanup.py`, `tests/ccgram/handlers/status/test_topic_emoji.py`, `tests/ccgram/handlers/topics/test_topic_lifecycle.py`.

Providers: `tests/ccgram/providers/test_contracts.py`, `tests/ccgram/providers/test_jsonl_providers.py`, `tests/ccgram/providers/test_autodetect.py`, `tests/ccgram/providers/test_picker_capability_drift.py` (picker commands subset + bare-name format).

Shell/LLM: `tests/ccgram/providers/test_shell.py`, `tests/ccgram/handlers/shell/test_shell_commands.py`, `tests/ccgram/handlers/shell/test_shell_capture.py`, `tests/ccgram/handlers/shell/test_shell_prompt_orchestrator.py`.

Voice: `tests/ccgram/handlers/voice/test_voice_handler.py`.

Live view: `tests/ccgram/handlers/live/test_live_view.py`.

Screenshot / scrollback capture: `tests/ccgram/test_last_unit.py`.

Polling/periodic: `tests/ccgram/handlers/polling/test_polling_coordinator.py`, `tests/ccgram/handlers/polling/test_polling_strategies.py`, `tests/ccgram/handlers/polling/test_polling_types_purity.py`, `tests/ccgram/handlers/polling/test_status_polling.py`.

Recovery UX: `tests/ccgram/handlers/recovery/test_recovery_banner.py`, `tests/ccgram/handlers/recovery/test_recovery_ui.py`, `tests/ccgram/handlers/recovery/test_recovery_subpackage_surface.py`.

Structural invariants: `tests/ccgram/test_query_layer_only_for_handlers.py`, `tests/ccgram/test_lint_lazy_imports.py`, `tests/integration/test_import_no_cycles.py`.

Topic lifecycle: `tests/ccgram/handlers/topics/test_topic_orchestration.py`, `tests/ccgram/handlers/topics/test_topic_lifecycle.py`.

Tool-call visibility / `/toolcalls`: `tests/ccgram/handlers/messaging_pipeline/test_message_queue.py` (visibility gate), `tests/ccgram/test_window_state_store.py` (state field + cycle).

Provider switching (claude↔shell↔gemini): `tests/ccgram/handlers/polling/test_status_polling.py::TestProviderSwitchPromptSetup`, `TestProviderSwitchChain`.

## Test Layout

- `tests/ccgram/` — unit, mirrors source modules.
- `tests/integration/` — monitor flow, dispatch, tmux manager, state roundtrips.
- `tests/conftest.py` — required test env before imports.
- Hypothesis property tests: `tests/ccgram/test_message_queue_properties.py`.
