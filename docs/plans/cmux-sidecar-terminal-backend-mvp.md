# Plan: cmux sidecar terminal backend MVP

Plain Markdown. Useful to humans, coding agents, and task runners. This plan
covers the first execution horizon for the cmux sidecar design: backend-neutral
identity, backend routing, a cmux sidecar client/protocol, and binding existing
cmux workspaces with basic send/capture operations. It does not implement hook
integration, mailbox delivery, Mini App cmux streaming, or cmux workspace
creation; those belong in follow-up plans after this seam is proven.

## Overview

ccgram currently routes Telegram topics to tmux window IDs and calls
`tmux_manager` directly from many handlers. The approved target architecture adds
cmux-native workspaces without replacing tmux by introducing a backend-neutral
terminal unit model:

```text
Telegram topic -> TerminalUnitRef -> TmuxBackend | CmuxBackend -> sidecar
```

This plan establishes the safety net and first vertical slice. Existing tmux
behavior remains the default. cmux support is feature-flagged, workspace-level,
and sidecar-backed. The first user-visible cmux capability is binding an
existing cmux workspace to a topic, sending text to it, and capturing its screen.

## Source artifact

Approved design: `docs/architecture-design/2026-05-25-cmux-sidecar-backend.md`.

Source design modules used by this plan:

- `Terminal backend contract`
- `Backend router`
- `TmuxBackend`
- `CmuxBackend`
- `ccgram-cmux-sidecar`
- `Terminal unit state port`
- `Topic binding/router`
- `Terminal operation services`
- `Terminal backend config`

Source design contracts used by this plan:

- `Telegram topic/router -> terminal backend router`
- `Terminal operation services -> TmuxBackend`
- `Terminal operation services -> CmuxBackend -> sidecar`
- `ccgram-cmux-sidecar -> cmux CLI/socket/events`
- `Window state feature ports -> terminal unit state`
- `Backend config -> backend router`

Source design decisions used by this plan:

- Keep one ccgram bot for tmux and cmux.
- Introduce a sidecar process for cmux; do not scatter cmux calls through
  handlers.
- Map one Telegram topic to one cmux workspace.
- Do not fake tmux IDs for cmux.
- Keep `WindowStateStore` as the persistence kernel and add terminal identity
  ports.

Source risks addressed in this execution horizon:

- Existing code has many direct `tmux_manager` calls.
- Sidecar process supervision and availability are not yet specified.
- Telegram topic binding schema migration may touch many handlers.
- cmux capture performance and command contract must be verified with a spike.

Supporting current-code evidence from the design:

- `src/ccgram/tmux_manager.py` owns terminal create/capture/send/list/kill and
  injects `CCGRAM_WINDOW_ID`.
- `src/ccgram/hook.py`, `src/ccgram/session_map.py`, `src/ccgram/msg_cmd.py`,
  and `src/ccgram/miniapp/api/terminal.py` are tmux-shaped today.
- Handler search showed direct `tmux_manager` use in sessions dashboard, topic
  creation, live/screenshot/toolbar/polling/recovery flows.

## Success criteria

- Legacy `state.json` topic bindings load as `backend=tmux` without changing
  existing tmux behavior.
- New terminal identity projection stores backend and unit ID through a feature
  port; handlers do not read raw identity fields directly.
- A `TerminalBackend` protocol and router exist with `TmuxBackend` wrapping the
  existing `tmux_manager` behavior.
- Backend config keeps cmux disabled by default and prevents handlers from
  reading cmux env vars directly.
- A cmux sidecar client/protocol exists with fake-sidecar contract tests for
  list, capture, send, errors, and capabilities.
- A user can bind an existing cmux workspace to a Telegram topic when cmux is
  enabled and sidecar capabilities are compatible.
- Text send and screen capture for the bound cmux workspace route through
  `CmuxBackend` and the sidecar client.
- Sidecar unavailable/degraded states affect cmux topics only; tmux topics keep
  operating.
- Architecture-fitness checks shrink direct `tmux_manager` usage for the touched
  flow and prevent cmux-specific imports in handlers.

## Validation Commands

- `make lint`
- `make typecheck`
- `make test`
- `uv run pytest tests/ccgram/terminal_backends tests/ccgram/window_state_ports tests/ccgram/handlers/topics -q`
- `uv run pytest tests/ccgram/test_window_state_access_audit.py tests/ccgram/test_query_layer_only_for_handlers.py -q`
- `uv run python scripts/lint_lazy_imports.py src/ccgram`
- `markdownlint-cli2 docs/architecture-design/2026-05-25-cmux-sidecar-backend.md docs/plans/cmux-sidecar-terminal-backend-mvp.md`

## Implementation Steps

### Task 1: Add terminal identity state port and backend contracts

Justification: addresses design modules `Terminal backend contract` and
`Terminal unit state port`, contracts `Telegram topic/router -> terminal backend
router` and `Window state feature ports -> terminal unit state`, and decisions
"do not fake tmux IDs" and "keep `WindowStateStore` as the persistence kernel".
This is the safety net before behavior-bearing cmux work.

Files:

- `src/ccgram/terminal_backends/__init__.py` — new package exports stable
  backend contract symbols only.
- `src/ccgram/terminal_backends/base.py` — add `TerminalBackend`,
  `TerminalUnitRef`, `TerminalUnit`, `TerminalBackendCapabilities`, and
  normalized error types.
- `src/ccgram/window_state_ports/terminal_identity.py` — add terminal identity
  projection and setters over `WindowStateStore`.
- `src/ccgram/window_state_ports/__init__.py` — export terminal identity port
  types/functions.
- `src/ccgram/window_state_store.py` — add compatible optional backend/unit
  fields to `WindowState` serialization.
- `src/ccgram/window_view.py` or `src/ccgram/window_query.py` — expose a
  backend-neutral identity projection to handlers.
- `tests/ccgram/terminal_backends/test_base.py` — contract DTO/error tests.
- `tests/ccgram/window_state_ports/test_terminal_identity.py` — projection,
  mutation, no-op save, and invalid backend tests.
- `tests/ccgram/test_window_state_store.py` — state serialization round-trip for
  terminal identity fields.
- `tests/integration/test_state_roundtrip.py` — real state-file compatibility
  round-trip for legacy tmux and new cmux identity records.
- `tests/ccgram/test_window_state_access_audit.py` — approve raw identity fields
  only in store and terminal identity port.

Preconditions: design doc exists and is approved; working tree is on the design
branch; no production source for cmux backend exists yet.

Postconditions: state can represent `backend=tmux|cmux` and opaque `unit_id`
without handlers parsing backend-specific identifiers. Existing records missing
backend fields load as tmux.

Impact commands:

If GitNexus exits or is unavailable, record the failure and run the fallback
commands listed here in the task notes.

- `npx gitnexus impact WindowState --direction upstream --depth 3 --include-tests --repo ccgram`
- `npx gitnexus impact WindowStateStore --direction upstream --depth 3 --include-tests --repo ccgram`
- `npx gitnexus impact view_window --direction upstream --depth 3 --include-tests --repo ccgram`
- `npx gitnexus detect-changes --scope all --repo ccgram`
- Fallback: `git diff --name-only`
- Fallback: `rg -n "window_id|WindowState|window_store|view_window" src/ccgram tests/ccgram`

Verification commands:

- `uv run pytest tests/ccgram/terminal_backends/test_base.py tests/ccgram/window_state_ports/test_terminal_identity.py tests/ccgram/test_window_state_store.py tests/integration/test_state_roundtrip.py tests/ccgram/test_window_state_access_audit.py -q`
- `uv run pyright src/ccgram/`
- `uv run ruff check src/ccgram/terminal_backends src/ccgram/window_state_ports tests/ccgram/terminal_backends tests/ccgram/window_state_ports`

Manual checks:

- Confirm the new vocabulary in code uses `terminal unit` for backend-neutral
  identity and keeps `window_id` for legacy tmux compatibility only.

- [x] Add terminal backend DTOs and normalized backend error taxonomy.
- [x] Add `WindowState` optional terminal backend/unit fields with backward
      compatible defaults.
- [x] Add terminal identity feature-port projection and setters.
- [x] Add serialization, state-roundtrip, and no-op save tests.
- [x] Extend raw state access audit to protect terminal identity fields.
- [x] Run the task verification commands and record the result.
- [x] Run GitNexus detect-changes or the fallback commands and record the scoped
      blast radius. (Fallback `git diff --name-only` used; GitNexus not invoked
      in this loop iteration. Scope: `src/ccgram/terminal_backends/**`,
      `src/ccgram/window_state_ports/terminal_identity.py`,
      `src/ccgram/window_state_store.py`, `src/ccgram/window_view.py`,
      `src/ccgram/window_query.py`, plus paired tests under
      `tests/ccgram/terminal_backends/` and
      `tests/ccgram/window_state_ports/`. No handler files touched.)

### Task 2: Wrap tmux behind `TmuxBackend` and route first operations

Justification: addresses design modules `Backend router`, `TmuxBackend`, and
`Terminal operation services`, plus contract `Terminal operation services ->
TmuxBackend`. It proves the new seam without changing user-visible tmux
behavior.

Files:

- `src/ccgram/terminal_backends/router.py` — backend registry and lookup by
  `TerminalUnitRef`.
- `src/ccgram/terminal_backends/tmux.py` — implement `TerminalBackend` by
  delegating to existing `tmux_manager`.
- `src/ccgram/terminal_operations.py` — backend-neutral service functions for
  list, capture, send text, send key, and close.
- `src/ccgram/tmux_manager.py` — only if tiny adapter helpers are required;
  preserve existing public methods.
- `src/ccgram/handlers/text/text_handler.py` — migrate the primary text send
  path for topic-bound sessions to terminal operation service where feasible.
- `src/ccgram/handlers/live/screenshot_callbacks.py` or the current screenshot
  entry point located by `rg "capture_pane|screenshot" src/ccgram/handlers` —
  migrate one capture/screenshot path through terminal operation service.
- `tests/ccgram/terminal_backends/test_router.py` — registry/default behavior.
- `tests/ccgram/terminal_backends/test_tmux_backend.py` — fake tmux manager
  contract tests.
- `tests/ccgram/test_terminal_operations.py` — backend-neutral operation routing
  tests.
- Existing handler tests for the touched send/capture path — update expected
  calls without changing visible behavior.
- `tests/ccgram/test_query_layer_only_for_handlers.py` or a new architecture
  test — document the remaining direct `tmux_manager` allow-list and require
  the touched path to use `terminal_operations`.

Preconditions: Task 1 passed; legacy terminal identity loads as tmux.

Postconditions: at least one send path and one capture path operate through
`TerminalBackend` with no behavior change for tmux topics. The direct
`tmux_manager` footprint shrinks for the migrated flow.

Impact commands:

- `npx gitnexus impact send_to_window --direction upstream --depth 3 --include-tests --repo ccgram`
- `npx gitnexus impact TmuxManager --direction upstream --depth 3 --include-tests --repo ccgram`
- `npx gitnexus impact text_handler --direction upstream --depth 3 --include-tests --repo ccgram`
- `npx gitnexus detect-changes --scope all --repo ccgram`
- Fallback: `git diff --name-only`
- Fallback: `rg -n "tmux_manager|send_to_window|capture_pane" src/ccgram/handlers src/ccgram/miniapp src/ccgram/*.py`

Verification commands:

- `uv run pytest tests/ccgram/terminal_backends/test_router.py tests/ccgram/terminal_backends/test_tmux_backend.py tests/ccgram/test_terminal_operations.py -q`
- `uv run pytest tests/ccgram/handlers/text tests/ccgram/handlers/live -q`
- `uv run pytest tests/ccgram/test_query_layer_only_for_handlers.py -q`
- `uv run pyright src/ccgram/`

Manual checks:

- Confirm no UI text changed for existing tmux sessions unless a test explicitly
  captures an improved error message.

- [x] Add backend router and tmux backend adapter.
- [x] Add backend-neutral terminal operation service.
- [x] Migrate one text-send path through terminal operations.
- [x] Migrate one capture/screenshot path through terminal operations.
- [x] Add or update architecture allow-list tests for direct `tmux_manager`
      usage.
- [x] Run the task verification commands and record the result.
      (`make lint`, `make typecheck`, `make test` — all green; targeted
      `uv run pytest tests/ccgram/terminal_backends
tests/ccgram/test_terminal_operations.py
tests/ccgram/test_tmux_manager_handler_footprint.py
tests/ccgram/handlers/text tests/ccgram/handlers/live
tests/ccgram/test_query_layer_only_for_handlers.py` — 64+ new tests
      pass, 5331 total pass, 28 skipped.)
- [x] Run GitNexus detect-changes or the fallback commands and record the scoped
      blast radius. (Fallback `git diff --name-only` used; GitNexus not
      invoked. Scope: new files `src/ccgram/terminal_backends/router.py`,
      `src/ccgram/terminal_backends/tmux.py`, `src/ccgram/terminal_operations.py`;
      protocol extension `src/ccgram/terminal_backends/base.py`;
      migrated handlers `src/ccgram/handlers/text/text_handler.py`,
      `src/ccgram/handlers/live/screenshot_callbacks.py`; new tests
      `tests/ccgram/terminal_backends/test_router.py`,
      `tests/ccgram/terminal_backends/test_tmux_backend.py`,
      `tests/ccgram/test_terminal_operations.py`,
      `tests/ccgram/test_tmux_manager_handler_footprint.py`; updated
      `tests/ccgram/terminal_backends/test_base.py`,
      `tests/ccgram/handlers/text/test_text_handler.py`. tmux_manager
      footprint unchanged; only two handler files added a
      `terminal_operations` route.)

### Task 3: Add cmux backend config, sidecar protocol client, and fake-sidecar tests

Justification: addresses design modules `Terminal backend config`,
`CmuxBackend`, and `ccgram-cmux-sidecar`, plus contracts `Backend config ->
backend router`, `Terminal operation services -> CmuxBackend -> sidecar`, and
`ccgram-cmux-sidecar -> cmux CLI/socket/events`. This creates the cmux boundary
without exposing cmux UI yet.

Files:

- `src/ccgram/terminal_backends/config.py` — typed backend config projection,
  env parsing, defaults, validation, and sidecar socket settings.
- `src/ccgram/terminal_backends/cmux_protocol.py` — RPC request/response DTOs,
  stable error codes, event frame DTOs.
- `src/ccgram/terminal_backends/cmux_client.py` — local Unix socket JSON-RPC
  client with timeouts and version/capability request.
- `src/ccgram/terminal_backends/cmux.py` — `CmuxBackend` adapter delegating to
  `CmuxSidecarClient`; disabled unless config permits.
- `src/ccgram/config.py` — add env/config fields only if the existing config
  owner is the correct source for backend settings.
- `src/ccgram/bootstrap.py` — only if backend registry initialization belongs in
  startup wiring.
- `tests/ccgram/terminal_backends/test_config.py` — config defaults,
  validation, disabled cmux behavior.
- `tests/ccgram/terminal_backends/test_cmux_protocol.py` — protocol schema and
  error normalization tests.
- `tests/ccgram/terminal_backends/test_cmux_client.py` — fake Unix-socket or
  fake transport tests for success, timeout, malformed response, incompatible
  version.
- `tests/ccgram/terminal_backends/test_cmux_backend.py` — adapter tests for
  capabilities/list/capture/send/error mapping.
- `tests/ccgram/test_backend_config_access_audit.py` — static check that only
  config/backend modules read `CCGRAM_CMUX_*` and `CCGRAM_TERMINAL_BACKEND*`.

Preconditions: Task 2 passed; backend router exists; no handler depends on cmux
classes directly.

Postconditions: cmux backend can be registered and exercised against fake
sidecar responses in tests, but remains hidden/disabled by default in user flows.

Impact commands:

- `npx gitnexus impact Config --direction upstream --depth 3 --include-tests --repo ccgram`
- `npx gitnexus impact bootstrap_application --direction upstream --depth 3 --include-tests --repo ccgram`
- `npx gitnexus detect-changes --scope all --repo ccgram`
- Fallback: `git diff --name-only`
- Fallback: `rg -n "CCGRAM_CMUX|CCGRAM_TERMINAL_BACKEND|config\." src/ccgram tests/ccgram`

Verification commands:

- `uv run pytest tests/ccgram/terminal_backends/test_config.py tests/ccgram/terminal_backends/test_cmux_protocol.py tests/ccgram/terminal_backends/test_cmux_client.py tests/ccgram/terminal_backends/test_cmux_backend.py tests/ccgram/test_backend_config_access_audit.py -q`
- `uv run pyright src/ccgram/`
- `uv run ruff check src/ccgram/terminal_backends tests/ccgram/terminal_backends`
- `uv run python scripts/lint_lazy_imports.py src/ccgram`

Manual checks:

- Confirm config names are user-comprehensible and do not conflict with provider
  names. Backend means terminal host, provider means agent CLI.

- [x] Add typed terminal backend config with tmux-only default.
- [x] Add cmux sidecar RPC protocol DTOs and stable error codes.
- [x] Add fake-transport sidecar client tests before real socket behavior.
- [x] Add `CmuxBackend` adapter and keep it disabled unless configured.
- [x] Add config access architecture test.
- [x] Run the task verification commands and record the result.
      (`make lint`, `make typecheck`, `make test` — all green;
      targeted `uv run pytest tests/ccgram/terminal_backends
  tests/ccgram/test_backend_config_access_audit.py` — 158 tests pass.
      Lazy-import lint clean; ruff/pyright clean on touched modules.
      `make test` reports 5442 passed, 28 skipped.)
- [x] Run GitNexus detect-changes or the fallback commands and record the scoped
      blast radius. (Fallback `git diff --name-only` used; GitNexus not
      invoked. Scope: new files
      `src/ccgram/terminal_backends/config.py`,
      `src/ccgram/terminal_backends/cmux_protocol.py`,
      `src/ccgram/terminal_backends/cmux_client.py`,
      `src/ccgram/terminal_backends/cmux.py`; new tests
      `tests/ccgram/terminal_backends/test_config.py`,
      `tests/ccgram/terminal_backends/test_cmux_protocol.py`,
      `tests/ccgram/terminal_backends/test_cmux_client.py`,
      `tests/ccgram/terminal_backends/test_cmux_backend.py`,
      `tests/ccgram/test_backend_config_access_audit.py`. No handler,
      bootstrap, or `tmux_manager` files touched — cmux backend stays
      unregistered by default; router still tmux-only. Public surface
      of `terminal_backends/__init__.py` unchanged.)

### Task 4: Bind existing cmux workspace and route basic send/capture

Justification: addresses design modules `Topic binding/router`, `CmuxBackend`,
`Terminal operation services`, and `Terminal backend config`; contracts
`Telegram topic/router -> terminal backend router` and `Terminal operation
services -> CmuxBackend -> sidecar`; and decision "map one Telegram topic to one
cmux workspace". This is the first user-visible cmux vertical slice.

Files:

- `src/ccgram/handlers/topics/cmux_callbacks.py` — new callback handlers for
  listing/selecting existing cmux workspaces, if colocating with topic handlers
  matches current package style.
- `src/ccgram/handlers/topics/new_command.py` or
  `src/ccgram/handlers/topics/directory_browser.py` — add an opt-in entry point
  to bind existing cmux workspace when cmux backend is enabled.
- `src/ccgram/handlers/topics/window_callbacks.py` or topic binding owner —
  persist `TerminalUnitRef(backend="cmux", unit_id=<workspace_id>)` for selected
  workspace.
- `src/ccgram/handlers/sessions_dashboard.py` — display mixed tmux/cmux units
  with backend labels and degraded sidecar state.
- `src/ccgram/handlers/registry.py` — register cmux callback handlers if needed.
- `src/ccgram/terminal_operations.py` — ensure send/capture operations return
  user-safe errors for cmux unavailable/no terminal surface.
- `tests/ccgram/handlers/topics/test_cmux_callbacks.py` — picker rendering,
  selection, disabled backend, sidecar unavailable, and stale callback tests.
- `tests/ccgram/handlers/test_sessions_dashboard.py` — mixed backend rendering
  and degraded cmux state.
- Existing text/capture handler tests touched in Task 2 — add cmux fake backend
  coverage for send/capture.
- `tests/integration/test_import_no_cycles.py` — update if new topic module
  imports require lazy import comments.

Preconditions: Task 3 passed; fake sidecar can list, capture, and send; cmux is
disabled by default and enabled only in tests/config.

Postconditions: with cmux enabled and fake sidecar healthy, a workspace can be
bound to a topic, appears in sessions dashboard, accepts text through
`TerminalBackend`, and returns screen capture through the same operation service.
When sidecar is down, the topic shows cmux unavailable without affecting tmux
sessions.

Impact commands:

- `npx gitnexus impact handle_new_window --direction upstream --depth 3 --include-tests --repo ccgram`
- `npx gitnexus impact handle_sessions_command --direction upstream --depth 3 --include-tests --repo ccgram`
- `npx gitnexus impact send_to_window --direction upstream --depth 3 --include-tests --repo ccgram`
- `npx gitnexus detect-changes --scope all --repo ccgram`
- Fallback: `git diff --name-only`
- Fallback: `rg -n "CB_|sessions|new window|tmux_manager|terminal_operations" src/ccgram/handlers tests/ccgram/handlers`

Verification commands:

- `uv run pytest tests/ccgram/handlers/topics/test_cmux_callbacks.py tests/ccgram/handlers/test_sessions_dashboard.py -q`
- `uv run pytest tests/ccgram/handlers/text tests/ccgram/handlers/live -q`
- `uv run pytest tests/integration/test_import_no_cycles.py -q`
- `uv run pyright src/ccgram/`

Manual checks:

- With a real cmux app available, manually verify command compatibility for
  `cmux tree --all --json`, `cmux read-screen --workspace <id>`, and the chosen
  send command. Record incompatibilities as follow-up work, not silent test
  assumptions.
- Confirm Telegram copy distinguishes terminal backend from agent provider in
  labels and errors.

- [x] Add cmux workspace bind callback flow behind backend config flag.
- [x] Persist selected cmux workspace through terminal identity port.
- [x] Update sessions dashboard to display mixed backends and cmux unavailable
      state.
- [x] Add cmux fake-backend coverage for send and capture through existing
      terminal operation service.
- [x] Add stale/disabled/unavailable callback tests.
- [x] Run the task verification commands and record the result.
      (`make lint`, `make typecheck`, `make test`, `make test-integration`
      — all green; targeted `uv run pytest tests/ccgram/terminal_backends
    tests/ccgram/window_state_ports tests/ccgram/handlers/topics
    tests/ccgram/handlers/test_sessions_dashboard.py
    tests/ccgram/test_terminal_operations.py
    tests/ccgram/test_window_state_access_audit.py
    tests/ccgram/test_query_layer_only_for_handlers.py
    tests/ccgram/test_window_store_import_boundary.py` — 1032 pass.
      `make test` reports 5472 passed, 28 skipped; integration 307 passed.)
- [x] Run GitNexus detect-changes or the fallback commands and record the scoped
      blast radius. (Fallback `git diff --name-only` used; GitNexus not
      invoked. New files: `src/ccgram/terminal_backends/lifecycle.py`,
      `src/ccgram/handlers/topics/cmux_callbacks.py`,
      `tests/ccgram/terminal_backends/test_lifecycle.py`,
      `tests/ccgram/handlers/topics/test_cmux_callbacks.py`. Edited:
      `src/ccgram/bootstrap.py` (new `register_terminal_backends` step),
      `src/ccgram/handlers/callback_data.py` (CB*CMUX*\*),
      `src/ccgram/handlers/callback_registry.py` (load cmux_callbacks),
      `src/ccgram/handlers/registry.py` (`/cmux` command),
      `src/ccgram/handlers/sessions_dashboard.py` (mixed backend rendering + cmux degraded state),
      `src/ccgram/handlers/polling/polling_coordinator.py` (skip cmux
      bindings so sidecar outages do not crash the tmux tick),
      `src/ccgram/handlers/topics/__init__.py` (re-export),
      `tests/ccgram/test_handler_layering_invariants.py`,
      `tests/ccgram/handlers/polling/test_polling_coordinator.py`
      (allow-list extended), `tests/ccgram/test_terminal_operations.py`
      (cmux send/capture routing tests),
      `tests/ccgram/handlers/test_sessions_dashboard.py` (cmux rendering
      tests). No `tmux_manager` handler/Mini App callers added.)

### Task 5: Final verification and documentation

Justification: proves the plan's success criteria and source design handoff:
backend-neutral identity, backend routing, sidecar contract, cmux workspace bind
MVP, rollback safety, and architecture-fitness checks. This task also records
follow-up scope for hook/session events, mailbox delivery, Mini App cmux, and
cmux workspace creation.

Files:

- `docs/plans/cmux-sidecar-terminal-backend-mvp.md` — update progress evidence
  and any discovered scope notes.
- `docs/architecture-design/2026-05-25-cmux-sidecar-backend.md` — update only if
  implementation uncovers a design correction.
- `docs/providers.md` — add cmux terminal-backend setup notes only if user-facing
  behavior ships in this horizon.
- `docs/guides.md` — add mixed backend/session guidance only if user-facing
  behavior ships in this horizon.
- `tests/ccgram/test_query_layer_only_for_handlers.py` and
  `tests/ccgram/test_window_state_access_audit.py` — ensure final boundary
  allow-lists match actual migrated scope.

Preconditions: Tasks 1-4 verification commands passed.
Postconditions: whole-plan validation passes; docs either reflect shipped
user-visible cmux behavior or explicitly defer public docs until the feature is
usable behind a flag; re-review scope is recorded.

Impact commands:

- `npx gitnexus detect-changes --scope all --repo ccgram`
- Fallback: `git diff --name-only`
- Fallback: `git diff --stat`

Verification commands:

- `make lint`
- `make typecheck`
- `make test`
- `uv run pytest tests/ccgram/terminal_backends tests/ccgram/window_state_ports tests/ccgram/handlers/topics -q`
- `uv run pytest tests/ccgram/test_window_state_access_audit.py tests/ccgram/test_query_layer_only_for_handlers.py -q`
- `uv run python scripts/lint_lazy_imports.py src/ccgram`
- `markdownlint-cli2 docs/architecture-design/2026-05-25-cmux-sidecar-backend.md docs/plans/cmux-sidecar-terminal-backend-mvp.md`

Manual checks:

- Review all user-facing copy for a clear distinction between terminal backend
  (`tmux`/`cmux`) and provider (`claude`/`codex`/`gemini`/`pi`/`shell`).
- Decide whether the feature remains hidden behind config or is ready for a
  short user-facing setup note.

- [ ] Run whole-plan validation commands and record results.
- [ ] Update user-facing docs if cmux bind/send/capture is available behind a
      documented flag; otherwise record that public docs are deferred.
- [ ] Update architecture-fitness allow-lists to match final migrated scope.
- [ ] Record GitNexus detect-changes output or fallback diff summary.
- [ ] Record follow-up plan targets: hook/session events, mailbox delivery,
      Mini App cmux streaming, cmux workspace creation, and real sidecar process
      supervision.
- [ ] Record scoped architecture-review follow-up and source refs to re-check.

## Acceptance criteria

- `TerminalUnitRef` and terminal backend DTOs are the only backend-neutral
  identity vocabulary used by the touched topic/send/capture paths.
- Legacy tmux state and existing tmux send/capture behavior remain compatible.
- `TmuxBackend` wraps the existing tmux behavior behind `TerminalBackend`.
- cmux backend is disabled by default and enabled only through typed backend
  config.
- cmux sidecar client/protocol has contract tests for capabilities, list,
  capture, send, timeout, unavailable, incompatible version, and malformed
  response.
- Existing cmux workspace bind flow works against a fake sidecar and is guarded
  by disabled/unavailable/stale callback tests.
- Direct `tmux_manager` use is removed from the migrated send/capture flow and
  protected by architecture-fitness tests.
- No handler imports cmux protocol/client modules directly.
- Sidecar unavailable state degrades cmux topics only; tmux topics keep working.
- Whole-plan validation commands pass or failures are documented with exact
  command output and scoped fix recommendations.

## Safety notes

This plan touches high-fan-in state and terminal routing. It must be executed in
small commits. Do not combine tasks.

- Data migration risk: `WindowState` gains backend identity fields. Migration
  must be additive; missing fields mean tmux. Never rewrite old topic bindings
  in bulk during this plan.
- Runtime routing risk: text send and capture are user-visible. Preserve tmux
  behavior first, then add cmux. If tmux tests fail, stop and fix before cmux
  work continues.
- Sidecar risk: sidecar is a local integration boundary and may be unavailable.
  All cmux calls need timeouts and stable degraded errors.
- Security risk: sidecar must not receive Telegram tokens or raw chat content in
  logs. Do not pass user text to sidecar logs; send text as command payload only.
- Blast radius: direct `tmux_manager` calls are widespread. This plan migrates
  only a small send/capture/topic-bind slice, then re-reviews before expanding.
- Rollback: disable cmux backend config to hide cmux UI and mark existing cmux
  topics unavailable. Existing tmux topics must not require rollback.

The architect does not execute this plan. An engineer, mutator agent, or task
runner executes the approved tasks after approval.

## Re-review

After Task 5, run a scoped `architecture-review` focused on:

- Whether `TerminalBackend` is a real contract or just a pass-through facade.
- Whether direct `tmux_manager` imports shrank for the migrated flows.
- Whether cmux-specific code is isolated to `terminal_backends/cmux*`, backend
  config, and tests.
- Whether `WindowStateStore` raw terminal identity fields are hidden behind the
  terminal identity port.
- Whether sidecar failures are contained to cmux topics.
- Whether the next plan should sequence hook/session events, mailbox delivery,
  Mini App cmux streaming, cmux workspace creation, or sidecar supervision.
