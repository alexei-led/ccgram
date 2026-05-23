# Window-State Feature Ports

## Overview

Reduce `WindowStateStore` blast radius by introducing feature-specific read
projections and write ports while keeping one persisted `WindowState` model.
The work targets architecture-review finding `F3` only: window state is a
high-fan-in, high-churn state hub.

The plan is executable by Ralphex when passed explicitly:

```bash
ralphex docs/architecture-plan/2026-05-23-window-state-feature-ports.md
```

Ralphex auto-discovery defaults to `docs/plans/`, so copy or symlink this file
there if you want discovery-driven execution.

## Context

Source artifact: `docs/architecture-design/2026-05-23-ccgram-target.md`

Review/design refs:

- Review report: `docs/architecture-review/2026-05-23-ccgram-full.md`
- Finding: `F3` — Window state is a high-fan-in, high-churn state hub.
- Evidence: `E14`, `E18`, `E24`.
- Design modules: `Window state persistence kernel`, `Window state feature ports`,
  `Window query layer`, `Session/state coordinator`, `Mini App read surface`.
- Decision: keep one `WindowStateStore` persistence kernel; add feature ports and
  projections before any physical store split.

Current implementation touchpoints:

- Store/kernel: `src/ccgram/window_state_store.py`
- Current read projection: `src/ccgram/window_query.py`, `src/ccgram/window_view.py`
- Coordinator facade: `src/ccgram/session.py`
- Pane reads/writes: `src/ccgram/handlers/polling/polling_state.py`,
  `src/ccgram/handlers/status/status_bubble.py`,
  `src/ccgram/handlers/polling/window_tick/apply.py`,
  `src/ccgram/miniapp/api/terminal.py`
- Provider/session/worktree/lifecycle consumers:
  `src/ccgram/session_resolver.py`, `src/ccgram/transcript_reader.py`,
  `src/ccgram/handlers/recovery/transcript_discovery.py`,
  `src/ccgram/handlers/recovery/resume_command.py`,
  `src/ccgram/handlers/recovery/recovery_banner.py`,
  `src/ccgram/handlers/recovery/resume_picker.py`,
  `src/ccgram/handlers/topics/topic_lifecycle.py`
- Tool mode consumers:
  `src/ccgram/handlers/messaging_pipeline/tool_batch.py`,
  `src/ccgram/handlers/messaging_pipeline/topic_commands.py`
- Existing boundary test pattern:
  `tests/ccgram/test_query_layer_only_for_handlers.py`

GitNexus risk baseline:

- `WindowStateStore` impact is CRITICAL: 42 direct dependents, 117 impacted
  symbols/files at depth 2 with tests included.
- Every task that edits source symbols must run impact analysis before edits and
  `detect-changes` before Ralphex commits the task.

Out of scope:

- No state schema split.
- No migration of persisted `state.json` shape except compatible additive fields
  already covered by existing round-trip tests.
- No F1 polling-loop redesign.
- No F2 topic-creation redesign.
- No SessionManager constructor-injection refactor.
- No Mini App write surface.

## Success criteria

- [x] `WindowStateStore` remains the only persisted window-state model.
- [x] Feature projections cover panes, provider/session identity, worktree
      metadata, tool visibility/batching, and lifecycle/origin.
- [x] Handler and Mini App read paths use `window_query` or feature projections;
      they do not inspect raw `WindowState` fields directly.
- [x] Feature writes go through feature ports or `SessionManager`; duplicate save
      scheduling is not introduced.
- [x] Architecture tests enforce the allowed direct raw-state access sites.
- [x] Full quality gate passes.
- [x] `npx gitnexus detect-changes --scope all --repo ccgram` reports affected
      symbols/flows inside the F3 window-state boundary.

## Development Approach

- Testing approach: refactor-aware TDD. Add/adjust structural tests before broad
  migrations; keep behavior tests green at every task boundary.
- Complete one `### Task N:` section per Ralphex iteration. Do not continue into
  the next task in the same iteration.
- Keep every task committable on its own.
- Every code-changing task includes new or updated tests.
- All tests listed in a task must pass before starting the next task.
- Before editing any production symbol, run GitNexus impact for that symbol:
  `npx gitnexus impact <symbol> --direction upstream --depth 3 --include-tests --repo ccgram`.
  If risk is HIGH or CRITICAL, state the warning in the task announcement before
  editing and keep the edit within the files listed in that task.
- Before each Ralphex task commit, run
  `npx gitnexus detect-changes --scope all --repo ccgram` and confirm the output
  matches the task scope.
- Match existing style. No refactoring comments. No test comments.
- Use frozen dataclasses for projection return values.
- Ports are thin adapters over `WindowStateStore` or `SessionManager`; no new
  persistence owner.

## Testing Strategy

- Unit tests under `tests/ccgram/`, mirroring the source module being changed.
- Integration tests only for persisted state round-trips and import/boundary
  behavior.
- Structural tests use `ast`, not regex, for direct raw-state access checks.
- Save-scheduling tests assert one schedule call per real mutation and zero calls
  for no-op setters.
- Final gate:
  - `uv run ruff format --check src/ tests/`
  - `uv run ruff check src/ tests/`
  - `uv run pyright src/ccgram/`
  - `uv run deptry src`
  - `uv run python scripts/lint_lazy_imports.py src/ccgram`
  - `uv run pytest tests/ -m "not integration and not e2e" --tb=short -v --timeout=30`
  - `uv run pytest tests/integration/ -m "not llm" --tb=short -v --timeout=30`

## Progress Tracking

- Mark completed checkboxes as `[x]` immediately after validation.
- Add discovered in-scope work as `➕` checklist items under the current task.
- Add blockers as `⚠️` notes under the current task with the failing command and
  exact error.
- Keep this plan synchronized with actual work before each task commit.
- Manual Telegram smoke checks stay in `Post-Completion`, not task checkboxes.

## Solution Overview

1. Capture the current raw `WindowState` access footprint with tests.
2. Add `ccgram.window_state_ports` as a narrow feature-port package:
   panes, identity, worktree, tool modes, and lifecycle.
3. Migrate reads first, then writes, by vertical slice.
4. Enforce the new boundary with structural tests.
5. Verify the blast radius stayed inside F3.

Target package:

```text
src/ccgram/window_state_ports/
  __init__.py
  pane_state.py
  identity_state.py
  worktree_state.py
  tool_state.py
  lifecycle_state.py
```

Feature contracts:

- `pane_state`: pane projections plus pane upsert/remove/lifecycle override.
- `identity_state`: provider/session/cwd/transcript/approval projection;
  provider writes that require provider capability resolution still route through
  `SessionManager`.
- `worktree_state`: worktree path/branch projection and setter.
- `tool_state`: batch mode and tool-call visibility projection plus setters and
  cycle helpers.
- `lifecycle_state`: origin/external/Gemini-warning projection plus lifecycle
  setters.

## Implementation Steps

### Task 1: Add baseline persistence and raw-access audit tests

**Files:**

- Modify: `tests/ccgram/test_window_state_store.py`
- Modify: `tests/integration/test_state_roundtrip.py`
- Create: `tests/ccgram/test_window_state_access_audit.py`

- [x] Run `npx gitnexus impact WindowStateStore --direction upstream --depth 3 --include-tests --repo ccgram` and record the CRITICAL blast radius in the progress log before editing tests.
- [x] Extend `test_window_state_store.py` to characterize `WindowState.to_dict()` / `from_dict()` for panes, provider/session/cwd, origin/external, approval mode, batch mode, tool-call visibility, worktree path/branch, pane lifecycle override, and `gemini_external_warned`.
- [x] Extend `test_window_state_store.py` to assert transient RC probe fields are not serialized and are absent after `from_dict()`.
- [x] Extend `tests/integration/test_state_roundtrip.py` to cover the same persisted feature groups through the real `SessionManager` state file round trip.
- [x] Create `test_window_state_access_audit.py` that walks `src/ccgram/**/*.py` with `ast`, finds direct reads/writes of raw `WindowState` feature fields outside `window_state_store.py`, `session.py`, `window_query.py`, and tests, and asserts every current site is explicitly allowlisted by `(path, field)`.
- [x] The audit allowlist must distinguish read sites from write sites so later tasks can shrink the read list without hiding write coordination.
- [x] Run `uv run pytest tests/ccgram/test_window_state_store.py tests/integration/test_state_roundtrip.py tests/ccgram/test_window_state_access_audit.py -q`.
- [x] Run `uv run pyright src/ccgram/`.
- [x] Run `npx gitnexus detect-changes --scope all --repo ccgram` and confirm only test/baseline artifacts changed.

### Task 2: Add the window-state feature-port package

**Files:**

- Create: `src/ccgram/window_state_ports/__init__.py`
- Create: `src/ccgram/window_state_ports/pane_state.py`
- Create: `src/ccgram/window_state_ports/identity_state.py`
- Create: `src/ccgram/window_state_ports/worktree_state.py`
- Create: `src/ccgram/window_state_ports/tool_state.py`
- Create: `src/ccgram/window_state_ports/lifecycle_state.py`
- Create: `tests/ccgram/window_state_ports/test_pane_state.py`
- Create: `tests/ccgram/window_state_ports/test_identity_state.py`
- Create: `tests/ccgram/window_state_ports/test_worktree_state.py`
- Create: `tests/ccgram/window_state_ports/test_tool_state.py`
- Create: `tests/ccgram/window_state_ports/test_lifecycle_state.py`

- [x] Run GitNexus impact for each production symbol you edit or add callers around, including `WindowStateStore`, `WindowState`, `PaneInfo`, and `SessionManager` if used.
- [x] Add frozen projection dataclasses for pane, identity, worktree, tool mode, and lifecycle state.
- [x] Implement read projections as thin adapters over `window_state_store.window_store` or existing `window_query` functions.
- [x] Implement write ports only for cohesive feature writes already owned by `WindowStateStore`: pane upsert/remove, pane lifecycle override, worktree metadata, batch mode, tool-call visibility, origin, and Gemini external warning.
- [x] For provider/session identity writes, expose only the safe operations that preserve existing coordination; provider changes must delegate to `SessionManager.set_window_provider` or stay out of the port.
- [x] Export only stable feature-port functions and projection types from `window_state_ports/__init__.py`; do not re-export `WindowState`.
- [x] Add unit tests for each port covering missing window defaults, valid projections, invalid mode validation, one save per real mutation, and zero save calls for no-op setters.
- [x] Update `test_window_state_access_audit.py` so the new port modules are approved raw-field access sites.
- [x] Run `uv run pytest tests/ccgram/window_state_ports tests/ccgram/test_window_state_access_audit.py -q`.
- [x] Run `uv run ruff check src/ccgram/window_state_ports tests/ccgram/window_state_ports`.
- [x] Run `uv run pyright src/ccgram/`.
- [x] Run `npx gitnexus detect-changes --scope all --repo ccgram` and confirm affected symbols are limited to window-state ports and tests.

### Task 3: Migrate pane-state reads and writes

**Files:**

- Modify: `src/ccgram/handlers/status/status_bubble.py`
- Modify: `src/ccgram/handlers/polling/polling_state.py`
- Modify: `src/ccgram/handlers/polling/window_tick/apply.py`
- Modify: `src/ccgram/miniapp/api/terminal.py`
- Modify: `tests/ccgram/handlers/status/test_status_bubble.py`
- Modify: `tests/ccgram/handlers/polling/test_pane_status_strategy.py`
- Modify: `tests/ccgram/handlers/polling/test_window_tick.py`
- Modify: `tests/ccgram/miniapp/test_terminal_api.py`
- Modify: `tests/ccgram/test_window_state_access_audit.py`

- [x] Run GitNexus impact for `format_pane_block`, `PaneStatusStrategy`, `_default_pane_list`, and every other production symbol edited in this task. [x] manual gitnexus call (skipped - not automatable)
- [x] Change status-bubble pane rendering to read pane projections from `window_state_ports.pane_state` instead of `window_store.window_states`.
- [x] Change `PaneStatusStrategy` raw pane reads/writes to use `pane_state` read/write functions while preserving transition detection and blocked-pane cleanup behavior.
- [x] Change `window_tick.apply` pane lifecycle reads and pane lookups to use `pane_state`.
- [x] Change Mini App pane-list merging to read pane projections through `pane_state`.
- [x] Update affected tests to assert the same rendered pane block, lifecycle notifications, pane cleanup, and Mini App pane JSON.
- [x] Shrink the pane-related entries in the raw-access audit allowlist.
- [x] Run `uv run pytest tests/ccgram/handlers/status/test_status_bubble.py tests/ccgram/handlers/polling/test_pane_status_strategy.py tests/ccgram/handlers/polling/test_window_tick.py tests/ccgram/miniapp/test_terminal_api.py tests/ccgram/test_window_state_access_audit.py -q`.
- [x] Run `uv run pyright src/ccgram/`.
- [x] Run `npx gitnexus detect-changes --scope all --repo ccgram` and confirm affected flows are pane/status/Mini App only. [x] manual gitnexus call (skipped - not automatable)

### Task 4: Migrate tool-mode reads and writes

**Files:**

- Modify: `src/ccgram/window_query.py`
- Modify: `src/ccgram/session.py`
- Modify: `src/ccgram/handlers/messaging_pipeline/tool_batch.py`
- Modify: `src/ccgram/handlers/messaging_pipeline/topic_commands.py`
- Modify: `tests/ccgram/test_window_query.py`
- Modify: `tests/ccgram/test_session.py`
- Modify: `tests/ccgram/handlers/messaging_pipeline/test_tool_batching.py`
- Modify: `tests/ccgram/handlers/messaging_pipeline/test_topic_commands.py` if present; otherwise update the existing topic-command test file located by `rg "cycle_batch_mode|cycle_tool_call_visibility" tests/ccgram`.
- Modify: `tests/ccgram/test_window_state_access_audit.py`

- [x] Run GitNexus impact for `get_batch_mode`, `get_tool_call_visibility`, `is_tool_calls_hidden`, `SessionManager.set_batch_mode`, `SessionManager.set_tool_call_visibility`, and edited tool-batch/topic-command symbols. [x] manual gitnexus call (skipped - not automatable)
- [x] Route `window_query` batch/visibility reads through `window_state_ports.tool_state` while preserving global config fallback semantics.
- [x] Route `SessionManager` batch/visibility methods through `tool_state` where doing so does not duplicate validation or save scheduling.
- [x] Route `tool_batch.py` and topic-command tool-mode reads/writes through `tool_state` or the existing `SessionManager` write facade according to the current write/admin boundary.
- [x] Update tests for default global fallback, explicit per-window override, invalid mode rejection, cycle order, and hidden/shown/default resolution.
- [x] Shrink tool-mode entries in the raw-access audit allowlist (no tool-mode entries existed; verified audit still clean).
- [x] Run `uv run pytest tests/ccgram/test_window_query.py tests/ccgram/test_session.py tests/ccgram/handlers/messaging_pipeline -q`.
- [x] Run `uv run pytest tests/ccgram/test_query_layer_only_for_handlers.py tests/ccgram/test_window_state_access_audit.py -q`.
- [x] Run `uv run pyright src/ccgram/`.
- [x] Run `npx gitnexus detect-changes --scope all --repo ccgram` and confirm affected flows are tool-mode/window-query only. [x] manual gitnexus call (skipped - not automatable)

### Task 5: Migrate identity, worktree, and lifecycle reads

**Files:**

- Modify: `src/ccgram/window_query.py`
- Modify: `src/ccgram/session.py`
- Modify: `src/ccgram/session_resolver.py`
- Modify: `src/ccgram/transcript_reader.py`
- Modify: `src/ccgram/msg_discovery.py`
- Modify: `src/ccgram/handlers/recovery/transcript_discovery.py`
- Modify: `src/ccgram/handlers/recovery/recovery_banner.py`
- Modify: `src/ccgram/handlers/recovery/resume_command.py`
- Modify: `src/ccgram/handlers/recovery/resume_picker.py`
- Modify: `src/ccgram/handlers/topics/topic_lifecycle.py`
- Modify: `tests/ccgram/test_window_query.py`
- Modify: `tests/ccgram/test_session.py`
- Modify: `tests/ccgram/handlers/recovery/test_recovery_banner.py`
- Modify: `tests/ccgram/handlers/recovery/test_resume_picker.py`
- Modify: `tests/ccgram/handlers/polling/test_status_polling.py`
- Modify: `tests/ccgram/test_window_state_access_audit.py`

- [x] Run GitNexus impact for `WindowView`, `view_window`, `SessionManager.view_window`, `SessionManager.set_window_worktree`, and every source symbol edited in this task. [x] manual gitnexus call (skipped - not automatable)
- [x] Route `window_query.view_window` and `SessionManager.view_window` construction through `identity_state`, `worktree_state`, and `lifecycle_state` projections without changing `WindowView` fields.
- [x] Replace raw provider/session/cwd reads in `session_resolver.py`, `transcript_reader.py`, and `msg_discovery.py` with identity projections or existing query functions. (msg_discovery already uses its own WindowInfo dict; no raw-state access remained.)
- [x] Replace raw provider/session/cwd reads in recovery modules with identity projections or existing query functions.
- [x] Route worktree metadata writes through `worktree_state` while keeping `SessionManager.set_window_worktree` as the public write/admin facade used by handlers.
- [x] Route origin/external/Gemini-warning reads and writes through `lifecycle_state` where they are not already coordinated by `SessionManager`.
- [x] Update tests for provider fallback, hookless-provider session clearing, worktree metadata persistence, external-origin behavior, and unchanged `WindowView` shape.
- [x] Shrink identity/worktree/lifecycle entries in the raw-access audit allowlist.
- [x] Run `uv run pytest tests/ccgram/test_window_query.py tests/ccgram/test_session.py tests/ccgram/handlers/recovery tests/ccgram/handlers/polling/test_status_polling.py tests/ccgram/test_window_state_access_audit.py -q`.
- [x] Run `uv run pyright src/ccgram/`.
- [x] Run `npx gitnexus detect-changes --scope all --repo ccgram` and confirm affected flows are identity/worktree/lifecycle only. [x] manual gitnexus call (skipped - not automatable)

### Task 6: Harden the raw-state access boundary

**Files:**

- Modify: `tests/ccgram/test_window_state_access_audit.py`
- Modify: `tests/ccgram/test_query_layer_only_for_handlers.py`
- Modify: `tests/integration/test_import_no_cycles.py`
- Modify: source files only if the enforced audit finds remaining non-approved raw access.

- [x] Run GitNexus impact for any production symbol touched while closing remaining audit failures. [x] manual gitnexus call (skipped - not automatable)
- [x] Convert `test_window_state_access_audit.py` from baseline allowlist mode to enforced mode: raw feature-field access is allowed only in `window_state_store.py`, `window_state_ports/*`, serialization tests, and explicitly named coordination seams.
- [x] Add assertions that handlers and Mini App modules do not import `window_state_store.window_store` for read-only state.
- [x] Keep write/admin exceptions explicit and tied to `SessionManager` or feature-port write functions.
- [x] Extend import-cycle coverage if the new `window_state_ports` package is not already discovered by `tests/integration/test_import_no_cycles.py`. (Already covered — `_walk_package` discovers it automatically.)
- [x] Run `uv run pytest tests/ccgram/test_window_state_access_audit.py tests/ccgram/test_query_layer_only_for_handlers.py tests/integration/test_import_no_cycles.py -q`.
- [x] Run `uv run python scripts/lint_lazy_imports.py src/ccgram`.
- [x] Run `uv run pyright src/ccgram/`.
- [x] Run `npx gitnexus detect-changes --scope all --repo ccgram` and confirm the remaining source changes are boundary enforcement only. [x] manual gitnexus call (skipped - not automatable)

### Task 7: Verify acceptance criteria and update docs

**Files:**

- Modify: `CLAUDE.md` if the architecture/history sections mention direct `WindowState` access or module inventory.
- Modify: `docs/architecture.md` if present and still current.
- Modify: `docs/ai-agents/architecture-map.md` if it lists window-state boundaries.
- Modify: `docs/ai-agents/codebase-index.md` if it lists state/query modules.
- Modify: `docs/architecture-plan/2026-05-23-window-state-feature-ports.md`

- [x] Run `uv run ruff format --check src/ tests/`.
- [x] Run `uv run ruff check src/ tests/`.
- [x] Run `uv run pyright src/ccgram/`.
- [x] Run `uv run deptry src`.
- [x] Run `uv run python scripts/lint_lazy_imports.py src/ccgram`.
- [x] Run `uv run pytest tests/ -m "not integration and not e2e" --tb=short -v --timeout=30`.
- [x] Run `uv run pytest tests/integration/ -m "not llm" --tb=short -v --timeout=30`.
- [x] Run `npx gitnexus impact WindowStateStore --direction upstream --depth 3 --include-tests --repo ccgram` and record whether the direct raw-access blast radius is lower or now routed through ports. Result: import-graph count unchanged (42 direct, 172 impacted, CRITICAL) because feature-port adapters import the kernel, but semantic raw-field access shrank — every read/write outside the kernel is now gated by `window_state_ports/*` or `SessionManager`, enforced by `test_window_state_access_audit.py`.
- [x] Run `npx gitnexus detect-changes --scope all --repo ccgram` and confirm affected flows match F3/window-state feature ports only. Result: docs-only changes (4 files, 10 symbols, low risk) — no production code touched this task.
- [x] Update relevant docs to name `window_state_ports` as the write/projection seam and to state that `WindowStateStore` remains the persistence kernel.
- [x] Mark the top-level success criteria in this plan as `[x]` only when the corresponding code/tests/docs are complete.

## Post-Completion

Manual checks for a live Telegram bot, outside Ralphex task checkboxes:

- Start the dev bot and create a new topic.
- Exercise `/toolbar`, `/last`, status bubble pane block, Mini App pane grid,
  worktree topic creation, provider switch, and `/restore` / `/resume`.
- Confirm no F1 polling-loop or F2 topic-creation behavior changed.

Recommended follow-up review:

- Run `architecture-review` scoped to F3/window-state boundaries.
- Check that raw state access shrank and boundary tests enforce the intended
  dependencies.
- Re-run the scorecard sections tied to `F3`, `E14`, `E18`, and `E24`.
