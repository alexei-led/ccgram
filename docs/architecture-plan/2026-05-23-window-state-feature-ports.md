# Architecture plan: window-state feature ports

Plain Markdown. Useful to humans and coding agents. Scope is one hotspot:
window-state feature ports/projections for `WindowStateStore` blast-radius
reduction. This plan is not execution.

## Overview

Reduce `WindowStateStore` blast radius by adding feature-specific
projections/accessors while keeping one persisted storage model. Scope is F3
only. F2 topic creation and F1 polling need separate plans.

## Source artifact

`source_artifact`: `docs/architecture-design/2026-05-23-ccgram-target.md`

Refs used:

- Review report: `docs/architecture-review/2026-05-23-ccgram-full.md`
- Finding: `F3` — Window state is a high-fan-in, high-churn state hub.
- Evidence: `E14`, `E18`, `E24`.
- Design modules: `Window state persistence kernel`, `Window state feature
ports`, `Window query layer`, `Session/state coordinator`, `Mini App read
surface`.
- Design decisions: keep one `WindowStateStore` persistence kernel; add feature
  ports/projections before any physical store split.
- Contracts: handler reads use query/projections; handler writes use
  `SessionManager` or feature ports; feature ports map to `WindowStateStore`.

## Success criteria

- [ ] Direct raw `WindowState` feature-field access outside store/port/
      serialization tests is removed or explicitly migration-allowlisted. Ties
      to `F3`, `E14`, design state-access boundary check.
- [ ] Feature projections exist for panes, provider/session identity, worktree
      metadata, tool visibility/batching, and lifecycle/origin. Ties to
      `Window state feature ports`.
- [ ] `WindowStateStore` remains the single persisted model. No schema split or
      migration unless separately approved. Ties to design decision.
- [ ] Handler and Mini App reads go through `window_query` or projections, not
      raw store imports. Ties to handler/Mini App contracts.
- [ ] CI and architecture checks pass.

## Phases

### Phase 1: Baseline safety net and access audit

Justification: `F3`, `E14`, `E18`, `E24`; design test specs for persistence
kernel and feature ports.

Preconditions: current mainline tests pass or known failures are documented.
Postconditions: behavior and boundary baseline exists before touching state code.

- [ ] Add characterization tests for current `WindowStateStore`
      persistence/reload behavior: panes, provider/session/cwd, origin,
      worktree, tool visibility, batching.
- [ ] Add tests confirming transient RC/probe fields are not serialized.
- [ ] Add an AST/access audit test in permissive mode that reports raw
      `WindowState` feature-field access outside store/serialization tests.
- [ ] Record current allow-list for direct state access. Keep it small and
      named.

Verification:

- [ ] `uv run pytest tests/ccgram/test_query_layer_only_for_handlers.py -q`
- [ ] New persistence and audit tests pass.
- [ ] `uv run pyright src/ccgram/`

### Phase 2: Add feature-port package as pass-through layer

Justification: design module `Window state feature ports`; contract
`feature ports -> WindowStateStore persistence kernel`; decision to keep one
store.

Preconditions: Phase 1 tests exist.
Postconditions: ports exist but behavior and persistence shape are unchanged.

- [ ] Add feature-port modules, e.g. `pane_state`, `provider_state`,
      `worktree_state`, `tool_visibility_state`, `window_lifecycle_state`.
- [ ] Add frozen/read-only projection types for each feature.
- [ ] Implement ports as thin adapters over public `WindowStateStore`
      methods/fields.
- [ ] Keep persistence centralized in `WindowStateStore`; no physical split.
- [ ] Add unit tests that each port validates inputs and schedules save exactly
      once per mutation.

Verification:

- [ ] Port tests pass.
- [ ] Existing `WindowStateStore` tests pass unchanged.
- [ ] Audit test shows port modules as approved raw-field access sites.

### Phase 3: Migrate read consumers by vertical slice

Justification: design contracts `Handler reads -> window/session query layer` and
`Mini App -> window/session/terminal read projections`; `F3` leakage risk.

Preconditions: feature projections are available.
Postconditions: read paths no longer require consumers to know raw store shape.

- [ ] Migrate pane read consumers to pane projection first.
- [ ] Migrate tool visibility/batching reads to tool visibility projection.
- [ ] Migrate worktree reads to worktree projection.
- [ ] Migrate provider/session metadata reads to provider projection.
- [ ] Migrate Mini App read routes to projections where they currently inspect
      state internals.

Verification:

- [ ] `uv run pytest tests/ccgram/test_query_layer_only_for_handlers.py -q`
- [ ] New state-access audit has no non-allowlisted handler or Mini App read
      violations.
- [ ] Relevant handler and Mini App route tests pass.

### Phase 4: Migrate write consumers without duplicating coordination

Justification: design contract
`Handler writes -> SessionManager/window state feature ports`; failure modes
include duplicate save scheduling and bypassed audit/prune rules.

Preconditions: read migration is green.
Postconditions: feature-specific writes use cohesive ports; cross-store
coordination still stays in `SessionManager`.

- [ ] Route pane mutations through pane port.
- [ ] Route tool visibility/batching mutations through tool visibility port.
- [ ] Route worktree metadata mutations through worktree port.
- [ ] Route provider/session identity mutations through provider port only where
      no cross-store invariant is owned by `SessionManager`.
- [ ] Keep audit/prune/startup/load/save orchestration in `SessionManager`.

Verification:

- [ ] Save-scheduling tests prove one save per mutation.
- [ ] Existing session/window mutation tests pass.
- [ ] No new handler imports of raw `window_state_store.window_store`.

### Phase 5: Harden boundaries and remove migration escape hatches

Justification: design architecture-fitness checks summary items 1 and 5;
acceptance signals; `F3`, `E18`, `E24`.

Preconditions: read and write migrations are complete.
Postconditions: architecture intent is enforced by tests, not vibes. Vibes are
how mud wins.

- [ ] Turn state-access audit from permissive to enforced.
- [ ] Remove obsolete direct-access allow-list entries.
- [ ] Add import-boundary checks: handlers/Mini App -> query/projections/ports;
      ports -> store; store must not import handlers.
- [ ] Add optional change-locality monitor for future commits touching
      `window_state_store.py` plus unrelated handler packages.
- [ ] Run full quality gate.

Verification:

- [ ] `uv run ruff format --check src/ tests/`
- [ ] `uv run ruff check src/ tests/`
- [ ] `uv run pyright src/ccgram/`
- [ ] `uv run deptry src`
- [ ] `uv run python scripts/lint_lazy_imports.py src/ccgram`
- [ ] `uv run pytest tests/ -m "not integration and not e2e" --tb=short -v --timeout=30`
- [ ] `uv run pytest tests/integration/ -m "not llm" --tb=short -v --timeout=30`

## Acceptance criteria

- [ ] `WindowStateStore` remains the persistence kernel.
- [ ] Feature ports/projections cover panes, provider/session, worktree, tool
      visibility/batching, and lifecycle/origin.
- [ ] Handler and Mini App reads no longer import raw store internals.
- [ ] Feature writes go through ports or `SessionManager`, with no duplicate save
      scheduling.
- [ ] Enforced architecture tests prevent regression.
- [ ] Full CI gate passes.
- [ ] Before commit, engineer runs `gitnexus_detect_changes()` and verifies
      affected flows match F3 scope only.

## Safety notes

Risk is elevated. Source design cites `WindowStateStore` as CRITICAL impact: 42
direct dependents, 172 total impacted symbols/files.

No data migration in this plan. No physical store split. Rollback is ordinary
code revert if phases stay incremental.

Before editing any symbol, engineer must run GitNexus impact for that symbol. If
impact is HIGH or CRITICAL, warn before proceeding.

Architect does not apply this plan. Engineer/mutator agent executes after
approval.

## Re-review

Run `architecture-review` after Phase 5, scoped to F3/window-state boundaries.

Check:

- Raw state access shrinkage.
- Boundary tests enforce intended dependencies.
- `WindowStateStore` blast radius for feature changes is lower or at least routed
  through ports.
- No accidental F1 polling or F2 topic-creation scope creep.
