# Multiplexer seam: add herdr alongside tmux

## Overview

Introduce a `Multiplexer` contract that both tmux and herdr satisfy, so ccgram stops depending on the concrete `tmux_manager` singleton, then implement a herdr backend behind that contract. tmux stays the default and its behavior is unchanged; herdr is selected by a `CCGRAM_MULTIPLEXER` switch.

The work is ordered safety-first. Phase 1 (Tasks 1–5) builds the seam with **zero behavior change** and locks it with fitness audits; Task 5 is a verification checkpoint and the recommended re-review point. Phase 2 (Tasks 6–9) adds the herdr backend and Phase 3 (Tasks 10–11) binds herdr agent panes to Telegram topics (group = herdr session, topic = pane = agent); both are behavior-bearing on the herdr path only. Task 12 is final verification.

## Context

- Impacted components: `src/ccgram/tmux_manager.py` (becomes `multiplexer/tmux.py`), ~48 call sites importing `tmux_manager`, `hook.py`, `providers/process_detection.py`, `providers/shell_infra.py`, `bootstrap.py`, `config.py`, `window_resolver.py` (restart re-resolution), `doctor_cmd.py`, `.archfit.yaml`, the boundary-audit test suite.
- Constraints: tmux path must stay byte-for-byte behavior-compatible through Phase 1; herdr backend must quarantine all herdr socket/JSON internals; checkboxes only inside Task sections (ralphex parses these as work items).
- Adopted from the approved design `docs/architecture-design/2026-06-21-herdr-multiplexer-support.md` and the project memory `project_herdr_multiplexer`.

## Source artifact

- Design: `docs/architecture-design/2026-06-21-herdr-multiplexer-support.md`.
- Confirmed decisions used: thin identity (reuse `window_id`, restart-only re-resolution via `session_id`); reuse polling now (defer event stream); keep ccgram's own hook (identity via `$HERDR_PANE_ID`); additive (`CCGRAM_MULTIPLEXER`, tmux default).
- Contracts used: the `Multiplexer` Protocol and `MultiplexerCapabilities` (design "The Multiplexer contract"); neutral value types `WindowRef`/`PaneInfo`/`CaptureResult`/`ForegroundInfo`/`PaneDims`.
- Modules used: `multiplexer/base.py` (core), `multiplexer/{tmux,herdr,registry,__init__}.py` (adapter), identity resolver (design "Module map").
- Risks used: herdr maturity v0.7.0, herdr server restart id reassignment, hook coexistence, deferred event stream (design "Open risks").
- Fitness used: F1 boundary audit, F2 contract test, F3 core-purity, F4 lazy-import, F5 archfit re-shape, F6 archfit CI wiring (design "Architecture-fitness checks summary").

## Success criteria

- No module under `handlers/**`, `polling/**`, `session*`, or other callers imports a concrete backend (`multiplexer.tmux`, `multiplexer.herdr`, `libtmux`, raw `tmux`/`herdr` shell-out); they depend only on the `Multiplexer` Protocol/proxy. (F1)
- The same parametrized contract test passes against the tmux backend and (when a herdr socket is present) the herdr backend. (F2)
- `multiplexer/base.py` imports no backend and no I/O library. (F3)
- `CCGRAM_MULTIPLEXER=tmux` reproduces today's behavior; the full existing test suite stays green through Phase 1.
- `CCGRAM_MULTIPLEXER=herdr` drives a real agent: create window, send, capture, status, kill, and survive a herdr server restart via session-id re-resolution.
- On herdr, each agent pane maps to one Telegram topic (group = herdr session), bound by agent session id, titled with an adaptive `<workspace> ▸ <agent>` prefix (+ `/<tab>` on splits); new-topic creation reuses the herdr workspace for the cwd.
- `.archfit.yaml` models the seam (core `multiplexer` vs adapter backends) with a forbidden-dependency rule.

## Development Approach

- Testing approach: regular (characterization-first for the tmux refactor; new behavior on herdr gets unit + contract tests).
- Complete each task fully — green verification — before starting the next.
- Phase 1 (Tasks 1–5) must not change any observable tmux behavior. Update this plan if scope shifts during implementation.

## Testing Strategy

- Unit tests for every code-changing task; the tmux refactor is guarded by the existing suite as characterization.
- The `Multiplexer` contract is enforced by one parametrized test run against each backend.
- Run project tests after each task before proceeding.

## Validation Commands

Whole-plan commands (repo Makefile targets):

- `make check` — fmt, lint, typecheck, deptry, unit + integration tests (the full gate).
- `make test` — `uv run pytest tests/ -m "not integration and not e2e" -n auto --dist=loadscope`.
- `make typecheck` — `uv run pyright src/ccgram/ tests/`.
- `make lint` — `scripts/lint_lazy_imports.py` + `uv run ruff check src/ tests/`.
- `uv run deptry src`.
- New seam audits: `uv run pytest tests/ccgram/test_multiplexer_boundary.py tests/ccgram/test_multiplexer_contract.py -v`.
- herdr leg (integration, needs a running herdr): `uv run pytest tests/integration/ -m "herdr" -v`.
- Deterministic architecture gate: `archfit check --config .archfit.yaml --full`. Note: `archfit` is **not installed locally and not wired into CI** today, so the enforced gate is the pytest boundary/contract audit; the `.archfit.yaml` rule change (Task 9) is the recommended promotion.
- Impact/blast-radius: GitNexus is not available in this repo. Fallback per task: `git diff --name-only` plus the F1 boundary audit as the dependency-direction proxy.

## Technical Details

- Value types must be field-compatible with today's `TmuxWindow`/`PaneInfo` so the tmux refactor is mechanical: `WindowRef` carries `window_id, window_name, cwd, pane_current_command, pane_tty, pane_width, pane_height`; `PaneInfo` carries `pane_id, index, active, command, path, width, height`. `ForegroundInfo` carries `pid, pgid, argv, cwd` (the neutral shape `process_detection` already needs).
- Package layering: `multiplexer/base.py` is core (pure Protocol + types); `multiplexer/{tmux,herdr,registry,__init__}.py` are adapter (I/O + wiring). Callers import the `multiplexer` proxy and type against `multiplexer.base.Multiplexer` (adapter→core allowed). The module-level `multiplexer` proxy mirrors the existing `window_store`/`thread_router` proxy pattern; `bootstrap.py` wires the backend from `config.multiplexer_name`.
- Capabilities gate control flow (never `name == "herdr"`): `ids_stable_across_restart`, `exposes_pane_tty`, `native_agent_status`, `read_max_lines`, `self_identify_env`, `supports_event_stream`.
- Re-review checkpoint: Task 5 is the safe stopping point. An `architecture-review` of the seam is recommended there before the Phase-2 herdr tasks change behavior.

## Implementation Steps

### Task 1: Define the Multiplexer Protocol and value types (core, pure)

- Justification: design "Module map" (`multiplexer/base.py`, core) and "The Multiplexer contract"; fitness F3 (core imports no I/O).
- Files: new `src/ccgram/multiplexer/__init__.py` (empty for now), new `src/ccgram/multiplexer/base.py` (Protocol + `MultiplexerCapabilities` + value types).
- Preconditions: none; purely additive, nothing imports it yet.
- Postconditions: `Multiplexer` Protocol, `MultiplexerCapabilities`, and value types exist and type-check; no runtime wiring.
- Impact: `git diff --name-only` (GitNexus unavailable).
- Fitness gate: add F3 assertion that `multiplexer.base` imports neither a backend nor `libtmux`/`subprocess`/`asyncio.subprocess`; extend later in Task 4.
- Verification: `make typecheck`; `uv run pytest tests/integration/test_import_no_cycles.py -v`.
- Manual checks: None.
- [x] add `multiplexer/base.py` with the `Multiplexer` Protocol covering today's `tmux_manager` public surface, normalized to value types (`ensure_session`, `list_windows`, `find_window`, `capture`, `capture_scrollback`, `pane_dims`, `send`, `send_to_pane`, `kill_window`, `rename_window`, `list_panes`, `create_window`, `set_title`, `foreground`)
- [x] add `MultiplexerCapabilities` dataclass with the six capability fields plus `name`
- [x] add field-compatible value types `WindowRef`, `PaneInfo`, `CaptureResult`, `ForegroundInfo`, `PaneDims`
- [x] write tests asserting value-type construction and that `multiplexer.base` imports no I/O module
- [x] run project tests (`make test`) - must pass before next task

### Task 2: Make tmux the first backend behind the Protocol (zero behavior change)

- Justification: design decision "Mirror the AgentProvider seam"; "Module map" (`multiplexer/tmux.py`, adapter). Characterization-guarded refactor.
- Files: move `src/ccgram/tmux_manager.py` → `src/ccgram/multiplexer/tmux.py`; add a thin compat shim at `src/ccgram/tmux_manager.py` re-exporting `tmux_manager` so existing imports keep working until Task 4.
- Preconditions: Task 1 merged.
- Postconditions: `TmuxManager` implements `Multiplexer`, returns the neutral value types, and exposes `capabilities` (`name="tmux"`, `ids_stable_across_restart=True`, `exposes_pane_tty=True`, `native_agent_status=False`, `read_max_lines=None`, `self_identify_env="TMUX_PANE"`, `supports_event_stream=False`). All existing tests pass unchanged.
- Impact: `git diff --name-only`; expect `multiplexer/tmux.py`, `tmux_manager.py` shim, value-type call-site reads unchanged because field names match.
- Fitness gate: none new; rely on the existing suite as characterization.
- Verification: `make check` (full gate must stay green — proves zero behavior change).
- Manual checks: confirm value-type field names match the old `TmuxWindow`/`PaneInfo` so no call site changed semantics.
- [x] move the tmux implementation into `multiplexer/tmux.py` and make it satisfy `Multiplexer`
- [x] return `WindowRef`/`PaneInfo`/`CaptureResult`/`ForegroundInfo`/`PaneDims` from the relevant methods, preserving field names
- [x] add the tmux `MultiplexerCapabilities`
- [x] leave a `tmux_manager.py` compat shim re-exporting the singleton (removed in Task 4)
- [x] write tests pinning tmux capabilities and one round-trip per refactored method
- [x] run project tests (`make test`) - must pass before next task

### Task 3: Registry, proxy, and CCGRAM_MULTIPLEXER switch (tmux-only)

- Justification: design "Module map" (`registry.py`, `__init__.py` proxy) and decision 4 (additive switch); mirrors `providers/registry.py` + the `window_store` proxy.
- Files: new `src/ccgram/multiplexer/registry.py`, `src/ccgram/multiplexer/__init__.py` (proxy + `get_multiplexer()`); `src/ccgram/config.py` (`multiplexer_name` from `CCGRAM_MULTIPLEXER`, default `tmux`); `src/ccgram/bootstrap.py` (wire the proxy in `bootstrap_application`).
- Preconditions: Task 2 merged.
- Postconditions: `get_multiplexer("tmux")` returns the tmux backend; the module-level `multiplexer` proxy forwards to the wired instance; bootstrap selects the backend from config. No call site migrated yet.
- Impact: `git diff --name-only`; blast radius limited to the new package + bootstrap/config.
- Fitness gate: none new yet (F1 lands in Task 4 after migration).
- Verification: `make test`; a unit test that the proxy raises a clear "not wired" error before bootstrap and forwards after wiring (mirrors existing proxy tests).
- Manual checks: confirm bootstrap ordering — the multiplexer proxy is wired before `start_session_monitor` / status polling use it.
- [x] add `registry.py` with `get_multiplexer(name)` + singleton cache
- [x] add the `multiplexer` proxy and `get_multiplexer()` in `__init__.py`
- [x] add `config.multiplexer_name` (`CCGRAM_MULTIPLEXER`, default `tmux`)
- [x] wire the proxy in `bootstrap.py` before monitor/polling start
- [x] write tests for registry resolution, proxy not-wired/forwarding, and config default
- [x] run project tests (`make test`) - must pass before next task

### Task 4: Migrate call sites to the proxy and land the F1–F3 fitness audits

- Justification: design "Integration contracts" (callers → `Multiplexer` Protocol; lower strength to contract); fitness F1 (boundary), F2 (contract), F3 (core purity).
- Files: every `handlers/**`, `polling/**`, `session*`, `window_*`, `live`/`screenshot` module importing `tmux_manager` → import the `multiplexer` proxy and type against `multiplexer.base.Multiplexer`; delete the `tmux_manager.py` compat shim; new `tests/ccgram/test_multiplexer_boundary.py`, new `tests/ccgram/test_multiplexer_contract.py`; extend `tests/integration/test_import_no_cycles.py`.
- Preconditions: Task 3 merged.
- Postconditions: no caller imports a concrete backend; F1/F2/F3 pass; full suite green.
- Impact: largest task — `git diff --name-only` will list the migrated call sites; the F1 audit is the dependency-direction proxy for blast radius.
- Fitness gate: F1 boundary audit (AST walk modeled on `tests/ccgram/test_window_state_access_audit.py`). Before-fail/after-pass: planting a direct `from ccgram.multiplexer.tmux import ...` in a handler must fail the audit; removing it must pass.
- Verification: `uv run pytest tests/ccgram/test_multiplexer_boundary.py tests/ccgram/test_multiplexer_contract.py -v`; then `make check`.
- Manual checks: spot-check that no `if multiplexer.name == "tmux"` conditional leaked into handlers (gate on capabilities, not names).
- [x] migrate all call sites from `tmux_manager` to the `multiplexer` proxy + `Multiplexer` type
- [x] delete the `tmux_manager.py` compat shim
- [x] add `test_multiplexer_boundary.py` (F1) forbidding backend/`libtmux`/raw-shell imports outside `multiplexer/**`, `bootstrap.py`, `main.py`
- [x] add `test_multiplexer_contract.py` (F2) parametrized over backends, tmux leg active
- [x] extend `test_import_no_cycles.py` (F3) to include `multiplexer/**` and assert `base` imports no backend
- [x] confirm F1 fails on a planted direct-backend import, then passes after removal
- [x] write tests covering the above audits
- [x] run project tests (`make test`) - must pass before next task

### Task 5: Phase-1 verification checkpoint (seam locked, zero behavior change)

- Justification: design "Handoff" (safety-gate first; re-review before behavior-bearing changes); architecture-plan safety ordering.
- Files: none (verification only); optionally update `.claude/rules/architecture.md` module inventory to add the `multiplexer/` package.
- Preconditions: Tasks 1–4 merged.
- Postconditions: full gate green with `CCGRAM_MULTIPLEXER` unset (tmux); the seam is enforced; this is the recommended `architecture-review` point before Phase 2.
- Impact: `git diff --name-only` since the plan start should show only the new package, audits, and migrated imports — no behavior files.
- Fitness gate: F1–F3 all green.
- Verification: `make check`; `uv run pytest tests/ccgram/test_multiplexer_boundary.py tests/ccgram/test_multiplexer_contract.py -v`.
- Manual checks: confirm tmux behavior is unchanged by exercising one real flow (create window, send, capture) under `CCGRAM_MULTIPLEXER=tmux`. Recommended: run `architecture-review` on the seam before continuing.
- [x] run the full gate and the seam audits; all green
- [x] update `.claude/rules/architecture.md` module inventory for the `multiplexer/` package
- [x] write/refresh any characterization test that proves tmux behavior is unchanged
- [x] run project tests (`make test`) - must pass before next task

### Task 6: Neutral hook identity resolver (tmux behavior unchanged)

- Justification: design "Integration contracts" (hook → identity resolver) and decision 3 (keep ccgram's own hook, identity via `$HERDR_PANE_ID`).
- Files: new `src/ccgram/multiplexer/self_identify.py` (or a function in `hook.py`); `src/ccgram/hook.py` (call the resolver instead of the hard `$TMUX_PANE`+`display-message` path).
- Preconditions: Task 5 checkpoint green.
- Postconditions: the hook resolves identity by which `self_identify_env` var is present — tmux via `$TMUX_PANE`+`display-message` (unchanged), herdr via `$HERDR_PANE_ID`; neither present → today's warning path.
- Impact: `git diff --name-only`; behavior change is additive (new herdr branch), tmux branch identical.
- Fitness gate: none archfit; the resolver is dependency-light so the hook (separate process) can import it without the backend wiring.
- Verification: table-driven unit test for the resolver; `make test`.
- Manual checks: confirm the tmux branch output is byte-identical to the previous `_resolve_window_id` result.
- [x] add `resolve_self_identity(env)` returning the neutral `SelfIdentity` (mux, session_window_key, window_id, window_name)
- [x] route `hook.py` through the resolver; keep the tmux branch behavior identical
- [x] add the herdr branch reading `$HERDR_PANE_ID` (+ `$HERDR_SOCKET_PATH` for cwd)
- [x] write table-driven tests (tmux env / herdr env / neither / nested-session)
- [x] run project tests (`make test`) - must pass before next task

### Task 7: Implement the herdr backend and wire the contract-test leg

- Justification: design "Module map" (`multiplexer/herdr.py`, adapter, anti-corruption) and "The Multiplexer contract"; risk "herdr maturity" (quarantine internals, check protocol version).
- Files: new `src/ccgram/multiplexer/herdr.py`; register it in `registry.py`; new `tests/ccgram/test_herdr_backend.py` (unit, JSON fixtures); enable the herdr leg in `test_multiplexer_contract.py` (skip without a socket).
- Preconditions: Tasks 5–6 merged.
- Postconditions: `get_multiplexer("herdr")` returns a working backend over `$HERDR_SOCKET_PATH`/CLI; all herdr JSON shapes stay private; capabilities set (`ids_stable_across_restart=False`, `exposes_pane_tty=False`, `native_agent_status=True`, `read_max_lines=1000`, `self_identify_env="HERDR_PANE_ID"`, `supports_event_stream=True`).
- Impact: `git diff --name-only`; the F1 audit guarantees no caller reaches into `herdr.py`.
- Fitness gate: F2 contract test now runs the herdr leg when a socket is present; F1 keeps `polling/**` and handlers off `multiplexer.herdr`.
- Verification: `uv run pytest tests/ccgram/test_herdr_backend.py -v`; with a running herdr: `uv run pytest tests/integration/ -m "herdr" -v`.
- Manual checks: drive a real agent under `CCGRAM_MULTIPLEXER=herdr` — create window, send, capture, kill; confirm `foreground()` comes from `pane process-info` (no `ps -t`), scrollback clamps at 1000 lines, protocol-version mismatch refuses cleanly.
- [x] implement the `Multiplexer` methods over the herdr socket/CLI (`pane get/list/read/run/send-text/send-keys/close`, `tab`/`workspace` create/rename, `pane layout`, `pane process-info`)
- [x] map `wN:pN` ↔ `window_id`, parse JSON fixtures into neutral value types, clamp scrollback to `read_max_lines`
- [x] check and pin the herdr `protocol` version from `herdr status`; refuse on mismatch
- [x] register herdr in `registry.py` and enable the herdr contract-test leg
- [x] write unit tests (fixtures) + boundary tests (socket down, bad id, truncation)
- [x] run project tests (`make test`) - must pass before next task

### Task 8: Extend restart re-resolution for non-stable ids (herdr)

- Justification: design "Integration contracts" (identity/`session_map` → restart re-resolution) and decision 1 (thin identity, session-id anchor); risk "herdr server restart".
- Files: `src/ccgram/window_resolver.py` (extend `resolve_stale_ids` / startup migration); `src/ccgram/session_map.py` / `session_resolver.py` as needed for the session-id lookup.
- Preconditions: Task 7 merged.
- Postconditions: when `caps.ids_stable_across_restart` is false, startup re-maps persisted `session_id` → current herdr pane (via `agent_session`), instead of display-name matching; tmux path unchanged.
- Impact: `git diff --name-only`; behavior change is gated on the capability flag, so tmux is untouched.
- Fitness gate: none archfit; covered by unit tests on the re-resolution branch.
- Verification: unit test simulating a herdr restart (changed pane ids, stable session id) → correct re-map; `make test`.
- Manual checks: restart a real herdr server and confirm a bound topic re-attaches to its agent.
- [x] branch `resolve_stale_ids` on `caps.ids_stable_across_restart`
- [x] add the herdr session-id → live-pane re-resolution using `agent_session` (joined via the hook-written session_map: live pane id → durable agent session id)
- [x] keep the tmux display-name path unchanged
- [x] write tests for the herdr restart re-map and the tmux no-op
- [x] run project tests (`make test`) - must pass before next task

### Task 9: herdr-aware doctor and archfit re-shape (F5)

- Justification: design "Open risks" (hook coexistence) and fitness F5 (archfit module/rule re-shape).
- Files: `src/ccgram/doctor_cmd.py` (verify herdr socket + hook coexistence with herdr's own Claude hook); `.archfit.yaml` (split `tmux_adapter` → core `multiplexer` (`ccgram.multiplexer.base`) + adapter `multiplexer_backends` (`ccgram.multiplexer.tmux/herdr/registry`); add the forbidden-dependency rule and `subdomain`/`volatility`/boundary labels).
- Preconditions: Tasks 7–8 merged.
- Postconditions: `ccgram doctor` reports multiplexer + (when herdr) socket/hook health; `.archfit.yaml` models the seam with a rule that core `multiplexer.base` cannot import backends and that `handlers`/`polling`/`session_state` cannot import `multiplexer_backends`.
- Impact: `git diff --name-only`.
- Fitness gate: F5. `archfit check --config .archfit.yaml --full` should report the new rule satisfied. Before-fail/after-pass: a temporary handler import of `multiplexer_backends` should make the archfit rule fail; removing it passes. Note: archfit is not installed/CI-wired, so the enforced equivalent remains the F1 pytest audit; F5 is the recommended promotion and F6 (CI wiring) is in Post-Completion.
- Verification: `uv run pytest` doctor tests; `archfit check --config .archfit.yaml --full` if `archfit` is available, else record the missing-tool note and rely on F1.
- Manual checks: confirm ccgram's hook and `herdr integration install claude` coexist without clobbering `~/.claude/settings.json`.
- [x] extend `doctor_cmd.py` with multiplexer + herdr socket/hook checks
- [x] re-shape `.archfit.yaml` modules and add the forbidden-dependency rule + labels
- [x] write doctor tests for tmux and herdr modes
- [x] run project tests (`make test`) - must pass before next task

### Task 10: Bind herdr agent panes to Telegram topics (one topic per agent)

- Justification: design "Telegram topic mapping (herdr)" (topic = pane = agent, bind on session id); decision A.
- Files: `src/ccgram/handlers/topics/topic_orchestration.py` (surface a herdr agent pane as a topic), `src/ccgram/handlers/messaging_pipeline/message_routing.py` (route inbound per pane/session), `src/ccgram/thread_router.py` (topic ↔ session-id binding), `src/ccgram/session_monitor.py` / `src/ccgram/handlers/recovery/transcript_discovery.py` (discover new herdr agent panes).
- Preconditions: Tasks 7–8 merged.
- Postconditions: on herdr, each agent pane has one topic bound by agent session id; new agent panes (including tab splits) surface as new topics; inbound/outbound route per pane; tmux topic behavior unchanged.
- Impact: `git diff --name-only`; topic handlers must use the `multiplexer` proxy (F1), never `multiplexer.herdr`.
- Fitness gate: F1 keeps topic handlers off the concrete backend; routing covered by unit tests.
- Verification: unit tests for pane→topic routing and session-id binding; with herdr running, integration test: two panes → two topics with independent streams.
- Manual checks: split a herdr tab into two agents → confirm two topics, no stream cross-talk.
- [x] route each herdr agent pane to its own topic, bound by agent session id
- [x] surface newly-detected agent panes (including tab splits) as new topics
- [x] keep the tmux topic flow unchanged (gate on capabilities, not backend name)
- [x] write tests for pane→topic routing and session-id binding
- [x] run project tests (`make test`) - must pass before next task

### Task 11: Adaptive topic prefix and cwd→workspace creation

- Justification: design "Telegram topic mapping (herdr)" (adaptive `<workspace> ▸ <agent>` prefix; cwd→workspace); risk "herdr server restart" (labels are derived state, not keys).
- Files: `src/ccgram/handlers/status/topic_emoji.py` (title/prefix rendering from `agent_status` + workspace/tab labels), `src/ccgram/handlers/topics/directory_callbacks.py` + `topic_orchestration.py` (resolve cwd→workspace, then tab+pane), event consumption for `workspace.renamed`/`tab.renamed`.
- Preconditions: Task 10 merged.
- Postconditions: topic title = `"[emoji] <workspace> ▸ <agent>"`, adding `"/<tab>"` only when the tab has more than one pane; new-topic creation reuses the herdr workspace matching the chosen cwd (creates one if absent); rename events re-label topics without rebinding.
- Impact: `git diff --name-only`.
- Fitness gate: none archfit; table-driven label-rendering tests.
- Verification: unit tests for prefix rendering (single-pane vs split tab) and cwd→workspace resolution; `make test`.
- Manual checks: rename a workspace in herdr → topic re-labels; create a second agent in the same repo → same workspace prefix, distinct topic.
- [x] render the adaptive topic title from `agent_status` + workspace/tab labels (add `/tab` only on splits) — `format_agent_topic_prefix` (pure, in `multiplexer/topic_mapping.py`); the herdr adapter stamps it into `WindowRef.window_name` in `list_windows`, the existing `topic_emoji` machinery prepends the status emoji
- [x] resolve new-topic cwd to an existing herdr workspace, creating one only if absent, then add tab+pane — `HerdrManager._resolve_workspace_id` (`workspace list` match by cwd, `workspace create` if absent) + `tab create --workspace`; lives in the adapter so `directory_callbacks`/`topic_orchestration` delegate through the seam unchanged
- [x] re-label topics on `workspace.renamed`/`tab.renamed` without rebinding — poll-driven (event stream deferred per design): each `list_windows` recomputes the derived label, `sync_display_names` propagates it, the binding key (durable session id, Task 8) is untouched
- [x] write table-driven tests for prefix rendering and cwd→workspace resolution — `TestFormatAgentTopicPrefix` (10-case table) + herdr adaptive-label/split/rename + workspace-reuse/create/fallback tests
- [x] run project tests (`make test`) - must pass before next task — 5158 passed, 28 skipped; lint + typecheck + deptry clean

### Task 12: Verify acceptance criteria

- Justification: architecture-plan final verification/documentation/handoff; design "Handoff" (re-review).
- Files: docs only — `.claude/rules/architecture.md`, `docs/architecture.md` / `docs/providers.md` if multiplexer config needs documenting; `README` env-var note for `CCGRAM_MULTIPLEXER`.
- Preconditions: Tasks 1–11 merged.
- Postconditions: whole-plan validation green for both backends; herdr topic mapping verified; docs updated; re-review recorded.
- Impact: `git diff --name-only` for the whole branch; record it in the PR.
- Fitness gate: F1–F5 green; F6 noted as follow-up.
- Verification: `make check`; the seam audits; with herdr running, `uv run pytest tests/integration/ -m "herdr" -v`.
- Manual checks: run `architecture-review` scoped to the multiplexer seam and topic mapping and confirm code matches the design.
- [x] verify all requirements from Overview are implemented for both `CCGRAM_MULTIPLEXER=tmux` and `=herdr` — tmux path: full suite green (5158 passed, 28 skipped) + F1/F2 seam audits (181 passed); herdr backend covered by `test_herdr_backend.py` unit + contract legs. The live herdr run (`-m herdr`, needs a running socket) is integration-only and not automatable here.
- [x] verify herdr topic mapping: one topic per agent pane, adaptive prefix, cwd→workspace — covered by `test_topic_mapping.py` (`TestHerdrPaneRouting`, `TestFormatAgentTopicPrefix`) + herdr backend adaptive-label/workspace-resolution tests, all passing
- [x] run the full project test suite and the seam audits — `make test` 5158 passed; `test_multiplexer_boundary.py`+`test_multiplexer_contract.py` 181 passed
- [x] run the project linter (`make lint`, including `lint-lazy`) - all issues fixed — lint-lazy clean, ruff all checks passed; pyright 0 errors; deptry no issues
- [x] update architecture docs and the `CCGRAM_MULTIPLEXER` env-var note — README + `docs/guides.md` config tables add the `CCGRAM_MULTIPLEXER` row; `docs/architecture.md` overview/diagrams updated to the `multiplexer/` seam (stale `tmux_manager.py` references removed); `.claude/rules/architecture.md` inventory already covers the seam
- [x] run project tests (`make test`) - must pass — 5158 passed, 28 skipped

## Acceptance criteria

- Full `make check` passes with the default (tmux) backend and with `CCGRAM_MULTIPLEXER=herdr` (herdr leg where a socket exists).
- F1 boundary audit and F2 contract test are part of the suite and green; F1 demonstrably fails on a planted direct-backend import.
- No caller imports a concrete backend; no `name == "<backend>"` conditional in handlers.
- A herdr server restart re-attaches bound topics via session-id re-resolution.
- On herdr, each agent pane has its own topic under the session's group with the adaptive prefix; a second agent in the same repo shares the workspace prefix.
- `.archfit.yaml` models the seam with a forbidden-dependency rule.

## Safety notes

- Phase 1 (Tasks 1–5) is behavior-preserving and reversible; Task 2 (file move + return-type change) is the widest blast radius — gated by `make check` as characterization. Roll back by restoring `tmux_manager.py` and reverting the value-type returns.
- Task 4 deletes the compat shim and touches ~48 call sites; commit it atomically and keep F1 green.
- Phase 2 (Tasks 6–9) changes behavior on the herdr path only; tmux branches are gated by capability flags. herdr v0.7.0 is young — pin the protocol version and keep internals quarantined.
- Phase 3 (Tasks 10–11) changes the herdr topic UX (one topic per agent pane); it does not touch tmux topic behavior. New topics on herdr create herdr `tab`/`pane`, reversible by closing the topic.
- Hook coexistence (Task 9) edits `~/.claude/settings.json`-adjacent behavior; verify it does not clobber herdr's own hook.
- Execution: an engineer, mutator agent, or `ralphex` runs this approved plan task by task. Stop at the Task 5 checkpoint for the recommended architecture-review before Phase 2.

## Post-Completion

Items requiring manual intervention. No checkboxes — informational only.

- Recommended re-review: run `architecture-review` scoped to the multiplexer seam after Task 5 and again after Task 10 to confirm code matches the design.
- F6 (defer): wire `archfit check --config .archfit.yaml` into `.github/workflows/ci.yml` to promote the seam rules from advisory to an enforced gate. Requires installing `archfit` in the CI image; do it as a separate, reviewed change so it cannot break CI mid-run. Also consider adding `make lint` (lint-lazy) to CI.
- herdr maturity watch: track herdr `protocol` version bumps and id-scheme changes across releases; the capability flags and the anti-corruption layer absorb most drift, but re-run the contract test after a herdr upgrade.
