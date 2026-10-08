# Extension and Fix Playbook

Common change recipes. Architectural constraints they must respect are in `architecture-map.md`.

## Common Extensions

Add a new provider:

1. Implement in `src/ccgram/providers/<name>.py` per `AgentProvider` contract.
2. Register in `src/ccgram/providers/__init__.py`.
3. Define capabilities accurately (resume/continue/hook/status).
4. Add tests in `tests/ccgram/providers/test_contracts.py` + provider-specific tests.
5. Launch hardening (e.g. Gemini shell mode) stays in `resolve_launch_command()` with launch-command tests.
6. If you change the provider contract (e.g. `discover_transcript(..., max_age=...)`): update `providers/base.py`, shared base (`_jsonl.py`, Claude/Codex/Gemini as needed), call sites (status polling + session monitor), contract + behavior tests.

Add a new Telegram command or callback:

1. Wire in `handlers/registry.py` (`command_specs`).
2. Implement in `src/ccgram/handlers/<subpackage>/`.
3. Add callback prefix/constant in `handlers/callback_data.py` if needed.
4. Take `client: TelegramClient` (never `bot: Bot`).
5. Add routing + handler tests.

Add session state fields:

1. Extend dataclasses/serialization in `src/ccgram/session.py` (or `window_state_store.py` for per-window fields).
2. Ensure load path is backward-compat with missing keys (`.get()`-style).
3. Update migration if key semantics change (`window_resolver.py` + migration tests).

Add a new slash command (agent-side, e.g. for Claude):

1. Add to the agent's command surface (e.g. `.claude/commands/` for Claude).
2. `command_catalog.py` discovers it on next scan (60s TTL).
3. The pinned topic `/commands` panel displays the provider-native name and forwards that exact name.
4. The native Telegram `/` menu remains limited to the four shared CCGram controls. Add bot-side code only for new panel behavior.

Add file upload handling:

- `handlers/file_handler.py` handles photos/documents. Saved to `.ccgram-uploads/` in the session's cwd. Agent notified via tmux keys with the absolute path; the Telegram reply keeps the relative form. Extend `file_handler.py` for new media types or post-processing.

Add a new LLM provider (shell command generation):

1. Entry in `src/ccgram/llm/__init__.py` `_PROVIDERS` (`base_url`, `model`, `api_key_env`).
2. OpenAI-compatible: no new completer code.
3. Different format: add a class to `src/ccgram/llm/httpx_completer.py` extending `_BaseCompleter`.
4. Temperature passes through from `config.llm_temperature` automatically.
5. Voice in shell topics auto-routes through LLM via `voice_callbacks.py` → `shell_commands.py`.

Add a new Whisper provider:

1. Entry in `src/ccgram/whisper/__init__.py`.
2. OpenAI-compatible Whisper: no new transcriber code.
3. Different format: add a class to `src/ccgram/whisper/httpx_transcriber.py`.

Adjust status or transcript parsing:

- Keep parsing provider-specific where possible.
- Preserve message-queue ordering and tool-use/tool-result pairing.
- Validate with parser unit tests + monitor integration tests.

## Bug-Fix Triage

1. Localize the layer:
   - Routing/state → `session.py`.
   - Monitor/parsing → `session_monitor.py`, providers, parsers.
   - Delivery/UI → `handlers/*`, `message_queue.py`, `live_view.py`, `periodic_tasks.py`.
   - Integration boundary → `tmux_manager.py`, `hook.py`.
2. Reproduce with narrow module-local tests, then broader suites.
3. Fix with architecture-safe changes:
   - No bypassing `SessionManager` state model.
   - No handler-to-handler tight coupling when a shared helper fits.
4. Re-run `make fmt && make test && make lint && make typecheck` (or `make check` for the full gate).

## Safe-Change Checklist

- Uses existing abstractions (`session_manager`, provider Protocol, tmux manager, `TelegramClient`).
- No regression in topic↔window identity behavior.
- No raw-string `context.user_data` keys; use `handlers/user_state.py` constants.
- Tests updated for changed behavior.
- `make check` is green.
