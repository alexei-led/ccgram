# Architecture design: ccgram target-state repair

Plain Markdown. Target architecture for modularity repair after the full architecture review. This is design only; production source changes belong in a follow-up implementation plan.

## Overview

ccgram is a Telegram control plane for tmux-hosted AI coding agents. One Telegram Forum topic controls one tmux window and its agent session; tmux remains the source of truth.

This is a target-state redesign based on `docs/architecture-review/2026-05-23-ccgram-full.md`. It preserves the healthy seams already in the codebase and repairs three reviewed risks:

1. `WindowStateStore` is still the high-fan-in, high-churn state hub.
2. `directory_callbacks.py` still carries too many topic-creation subflows.
3. `polling_state.py` still concentrates mutable polling strategy state.

The design lowers coupling strength at volatile/high-distance boundaries by adding explicit feature contracts and read/write ports. It does not propose a physical persistence split until projections prove useful. Persistence migrations, if any, should be earned. The universe has enough accidental schemas.

## Source inputs and drift notes

- Requirements: architecture-review recommendations F1, F2, F3 in `docs/architecture-review/2026-05-23-ccgram-full.md`.
- Existing docs/ADRs/reports:
  - `README.md`: product goal, topic-per-agent model, provider list, live/status/pane/worktree features.
  - `docs/architecture.md`: generated module map and intended handler/query/state/provider/polling layers.
  - `docs/ai-agents/architecture-map.md`: constraints to preserve, including topic-window mapping, centralized tmux operations, query-layer reads, provider capabilities, polling purity, lazy imports, and bootstrap ordering.
  - `.claude/rules/architecture.md`: current module inventory and key design decisions.
  - `.github/workflows/ci.yml`: CI gate for format, lint, pyright, deptry, unit tests, and integration tests.
- Existing implementation checked: yes; sampled state, topic creation, polling, provider, Telegram, and architecture-test files.
  - `src/ccgram/window_state_store.py`, `src/ccgram/window_query.py`, `src/ccgram/window_view.py`, `src/ccgram/session.py`.
  - `src/ccgram/handlers/topics/directory_callbacks.py`, `directory_browser.py`, `worktree.py`.
  - `src/ccgram/handlers/polling/polling_state.py`, `window_tick/observe.py`, `window_tick/decide.py`, `window_tick/apply.py`.
  - `src/ccgram/providers/base.py`, `src/ccgram/telegram_client.py`.
  - `tests/ccgram/test_query_layer_only_for_handlers.py`, `tests/ccgram/handlers/polling/test_polling_types_purity.py`, `tests/integration/test_import_no_cycles.py`, `scripts/lint_lazy_imports.py`.
- GitNexus impact evidence refreshed during design:
  - `WindowStateStore`: CRITICAL upstream impact, 42 direct dependents, 172 total impacted symbols/files.
  - `PaneStatusStrategy`: HIGH upstream impact, 22 direct dependents, 93 total impacted symbols/files.
  - `_create_window_and_bind`: LOW upstream impact, 4 direct callers, 14 total impacted symbols/files, local to Topics.
- Drift risks:
  - Review and GitNexus index match commit `3d9f04d`, but the working tree already contains untracked architecture/docs/config files.
  - No CODEOWNERS or ownership file; owner/deploy expectations below are architectural expectations, not proven team boundaries.
  - `WindowStateStore` should remain one persisted model until accessor/projection extractions show lower co-change; premature storage splitting would move pain into migrations.
  - Historical co-change evidence remains git-log fallback; use it as a signal, not scripture.

## Domain and volatility map

Core = differentiating behavior and likely to change. Supporting = necessary but not differentiating. Generic = solved infrastructure, though providers may still have implementation churn.

| Area                                               | Classification | Volatility | Rationale                                                                                                                                 | Open questions                                                                    |
| -------------------------------------------------- | -------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Telegram topic UX and callback flows               | Core           | High       | The product is controlled through topic interactions, inline keyboards, status bubbles, file/voice/shell/tool flows, and topic lifecycle. | None blocking.                                                                    |
| Topic/window/session routing                       | Core           | High       | Correct binding of topic -> tmux window -> provider transcript/session is the control-plane spine.                                        | Ownership unknown.                                                                |
| Provider abstraction and status/transcript parsing | Core           | High       | Extensibility across Claude/Codex/Gemini/Pi/Shell is a differentiator; provider CLIs change.                                              | Provider capability growth rate unknown.                                          |
| Polling/status detection                           | Core           | High       | Live feedback, interactive prompt detection, pane status, RC detection, and done/idle transitions are user-visible and volatile.          | Exact desired polling-port granularity should be validated during implementation. |
| Window state and projections                       | Core           | High       | Many features read/write per-window state: provider, panes, worktrees, tool visibility, lifecycle flags, session metadata.                | Physical store split deferred.                                                    |
| Inter-agent messaging                              | Core           | Medium     | Swarm messaging is differentiating but less central than topic/session routing.                                                           | Cross-agent delivery guarantees may evolve.                                       |
| Mini App                                           | Core           | High       | Optional but product-visible; reads terminal/transcript/window state and will likely grow.                                                | Whether Mini App gets separate ownership is unknown.                              |
| Session monitoring, hooks, state persistence       | Supporting     | Medium     | Required for reliable control plane; mostly change when providers or persistence needs change.                                            | Persistence migration policy should be explicit before physical store split.      |
| Shell, voice, send-file UX                         | Supporting     | Medium     | Important user flows but not the core routing model.                                                                                      | Shell safety policy may become higher volatility.                                 |
| tmux integration                                   | Generic        | Low        | Centralized wrapper over a stable local tool; change comes from platform edge cases.                                                      | None.                                                                             |
| Telegram Bot API adapter                           | Generic        | Low        | External API seam already isolated through `TelegramClient`; Bot API changes are usually additive.                                        | None.                                                                             |
| LLM/Whisper HTTP integrations                      | Generic        | Low/Medium | Solved integration; provider churn mainly around API compatibility and credentials.                                                       | None.                                                                             |

## Module map

| Module                          | Responsibility                                                                                     | Owned knowledge                                                                                                                                  | Public interface                                                                                                                                                | Private internals                                                                 | Owner/deploy expectation                              | Change vectors                                                 |
| ------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------- |
| Telegram application/bootstrap  | Build PTB app, wire runtime callbacks, start/stop monitor, status polling, Mini App.               | Startup order, lifecycle sequencing, allowed PTB runtime import sites.                                                                           | `create_application`, bootstrap/shutdown functions, handler registration.                                                                                       | PTB-specific app wiring.                                                          | Same local bot process.                               | Runtime lifecycle, CI/import constraints.                      |
| Telegram outbound seam          | Hide PTB `Bot` from handlers and tests.                                                            | Methods ccgram actually uses, entity-safe send/edit/upload semantics.                                                                            | `TelegramClient`, `PTBTelegramClient`, `FakeTelegramClient`, `unwrap_bot`.                                                                                      | PTB delegation, fake call recording.                                              | Same process; provider-independent.                   | New Telegram API methods used by handlers.                     |
| Provider contract               | Represent agent CLI behavior through explicit capability and parsing contracts.                    | Launch args, transcript formats, status parsing, command discovery, task tracking, hook support.                                                 | `AgentProvider`, `ProviderCapabilities`, provider registry.                                                                                                     | Provider-specific JSONL/status parsing and discovery.                             | Same process; per-provider modules.                   | CLI changes, new provider, new capabilities.                   |
| Topic creation orchestrator     | Keep the end-to-end new-topic state machine visible while delegating subflows.                     | Ordered creation lifecycle: stale guard -> directory/worktree -> provider/mode -> tmux create -> state write -> bind -> pending text forwarding. | `handle_directory_callback` router plus `TopicCreationDraft`/service operations.                                                                                | Callback prefix dispatch, operation ordering, race-guard release.                 | Same handler package; no separate deploy.             | New launch options, extra provider setup, worktree cleanup UX. |
| Directory navigation subflow    | Browse paths, favorites, paging, star/unstar, stale browser guard.                                 | Directory UI state, favorite/MRU semantics, path existence handling.                                                                             | `DirectoryNavigationPort` functions and callback handlers for `CB_DIR_*`.                                                                                       | `context.user_data` key details, keyboard layout.                                 | Same handler package.                                 | Browser UX, favorite policy.                                   |
| Worktree selection subflow      | Offer current branch vs new worktree and create validated worktree.                                | Git eligibility, branch naming, dirty-worktree warning, subdir preservation.                                                                     | `WorktreeSelectionPort`, pure `worktree.py` helpers, callback handlers for `CB_WT_*`.                                                                           | Blocking git subprocesses, pending worktree keys, re-entrancy guard.              | Same handler package.                                 | Branch naming policy, cleanup UX, git edge cases.              |
| Provider/mode selection subflow | Validate provider, choose launch mode, decide whether mode picker is needed.                       | Provider capability gates, YOLO availability, callback-data parsing.                                                                             | `ProviderModeSelectionPort`, provider/mode callback handlers.                                                                                                   | Keyboard rendering, provider validation details.                                  | Same handler package.                                 | Provider capabilities, launch modes.                           |
| Window binding/launch service   | Create tmux window, persist initial state, bind topic, apply provider setup, forward pending text. | Launch command resolution, race guard, initial state writes, group chat ID, hook wait, shell setup, messaging skill install, topic rename.       | `WindowLaunchRequest`, `WindowLaunchResult`, `create_and_bind_window()`.                                                                                        | tmux calls, SessionManager mutation sequence, pending text forwarding details.    | Same handler package, uses state/tmux/provider seams. | Launch ordering, provider setup, pending-message behavior.     |
| Window state persistence kernel | Own one persisted per-window storage model and serialization.                                      | Canonical `WindowState` schema, persistence callbacks, migration tolerance.                                                                      | `WindowStateStore`, `WindowState`, `PaneInfo`, install/get store.                                                                                               | Module-level proxy, schedule-save callbacks, dict storage.                        | Same process; one state file.                         | New persisted fields, migration, cleanup/pruning.              |
| Window state feature ports      | Reduce direct feature dependence on the whole storage shape.                                       | Feature-specific rules for panes, provider/session identity, worktree metadata, tool visibility/batching, origin/lifecycle.                      | Typed accessors/projections: `pane_state`, `provider_state`, `worktree_state`, `tool_visibility_state`, `window_lifecycle_state`; `WindowView`-style snapshots. | Mapping to `WindowStateStore` fields.                                             | Same source package; no separate persistence yet.     | New feature state, state read/write locality.                  |
| Window query layer              | Read-only handler contract over window/session state.                                              | Which window fields handlers may observe.                                                                                                        | `window_query.*`, `session_query.*`, `WindowView`.                                                                                                              | Direct store/session resolver reads.                                              | Same process.                                         | New read projections, deprecating direct reads.                |
| Session/state coordinator       | Coordinate writes and persistence across window store, thread router, user prefs, session map.     | Save scheduling, startup load, audit/prune, allowed write/admin mutations.                                                                       | `SessionManager` write/admin methods.                                                                                                                           | Store construction, proxies, state file shape.                                    | Same process.                                         | New write operations, pruning/audit rules.                     |
| Polling pure contract           | Keep status decision inputs/outputs deterministic and cheap to test.                               | `TickContext`, `TickDecision`, constants, pure shell-prompt detection.                                                                           | `polling_types.py`, `window_tick/decide.py`.                                                                                                                    | None that touch tmux/PTB/singletons.                                              | Same package.                                         | New status transitions; must preserve purity.                  |
| Polling runtime ports           | Own mutable poll-cycle state behind narrow contracts instead of one singleton hub.                 | Terminal buffer/RC cache, lifecycle timers, interactive state, pane status, startup/seen-status state.                                           | `TerminalBufferPort`, `WindowPollStatePort`, `LifecycleTimerPort`, `PaneStatusPort`, `InteractiveStatusPort`; production implementations wired once.            | Pyte buffer cache, module-level singleton compatibility, callback registry hooks. | Same process.                                         | Pane lifecycle, RC debounce, terminal parsing, status cache.   |
| Window tick observe/apply       | Observe pane/provider state, decide pure transition, apply Telegram/tmux/state side effects.       | Tick orchestration, status update delivery, topic emoji transitions, passive shell relay, pane scan notifications.                               | `tick_window`, `build_context`, `_update_status`-equivalent service accepting runtime ports.                                                                    | Telegram sends, tmux captures, singleton adapter glue.                            | Same process.                                         | Status semantics, pane scanning, provider status parsing.      |
| Messaging pipeline              | Queue, format, batch, and deliver provider output to Telegram topics.                              | Ordering, rate limiting, tool-use/result pairing, message splitting.                                                                             | Queue/enqueue APIs, safe send/edit APIs.                                                                                                                        | DraftStream/unwrap escape hatch, batching state.                                  | Same process.                                         | Tool-call presentation, Telegram send behavior.                |
| Mini App read surface           | Serve terminal/transcript/window views without gaining write ownership of core state.              | HTTP/WebSocket routes, auth, read-only terminal/transcript views.                                                                                | `miniapp` routes using query/projection contracts.                                                                                                              | aiohttp server and static assets.                                                 | Optional same-process server.                         | UI growth, terminal streaming needs.                           |
| Tmux infrastructure             | Centralize terminal operations.                                                                    | tmux/libtmux/subprocess behavior, foreign window targeting.                                                                                      | `tmux_manager` methods only.                                                                                                                                    | tmux command details and platform quirks.                                         | Same local machine.                                   | tmux edge cases, emdash support.                               |

## Integration contracts

For each relationship, strength means how much knowledge is shared. Distance covers package/runtime/ownership/deploy separation. Volatility comes from domain change likelihood first, history second.

### Telegram handlers -> Telegram outbound seam

- Strength: contract. Handlers know `TelegramClient`, not PTB internals.
- Distance: medium package distance; same process/runtime; third-party API hidden behind adapter.
- Volatility: medium in UX, low in Telegram API shape.
- Balanced: yes; low strength offsets external API distance.
- Contract: `TelegramClient` protocol plus `safe_send`/`safe_edit` message-sender helpers.
- Knowledge shared: only Telegram operations and message IDs cross; PTB implementation details stay private.
- Balancing move: keep as-is; add protocol methods only when handler usage exists.
- Failure modes: PTB-specific calls leaking into feature handlers, parse-mode regressions, send/edit fallback drift.

### Handler reads -> window/session query layer

- Strength: contract/model. Handlers share approved read projections, not the full state model.
- Distance: medium package distance inside one deployable.
- Volatility: high; handler UX and window fields change often.
- Balanced: mostly yes, but projections must grow feature-specific.
- Contract: `window_query`, `session_query`, `WindowView`, and future feature projections.
- Knowledge shared: stable view fields cross; store layout and persistence callbacks stay private.
- Balancing move: lower strength by adding specific projections instead of widening `WindowView` into a god DTO.
- Failure modes: handlers importing `window_store` for reads, `WindowView` accumulating unrelated fields, test allow-list drifting.

### Handler writes -> SessionManager/window state feature ports

- Strength: functional/model today; target is contract for feature-specific writes.
- Distance: medium package distance; same process.
- Volatility: high; writes reflect UX features.
- Balanced: not yet; direct write/admin allow-list keeps strength high across feature distance.
- Contract: keep `SessionManager` for cross-store coordination; add feature ports for cohesive write groups: panes, provider/session identity, worktree, tool visibility/batching, lifecycle/origin.
- Knowledge shared: feature intent crosses; raw state shape should not.
- Balancing move: lower strength. Keep write sequencing centralized where atomicity matters; do not split persistence first.
- Failure modes: duplicate save scheduling, bypassed audit/prune rules, two write APIs for one invariant.

### Window state feature ports -> WindowStateStore persistence kernel

- Strength: contract. Ports map feature operations to one storage schema.
- Distance: low package distance; same process.
- Volatility: high at feature ports, medium at persistence schema.
- Balanced: yes if ports stay close to store and schema remains private.
- Contract: typed accessors, small frozen projections, and mutator methods with validation.
- Knowledge shared: field names may be known only inside port implementations; features know domain operations.
- Balancing move: lower strength for callers while keeping distance low for implementation.
- Failure modes: port modules becoming pass-through clutter, inconsistent validation between port and store, migration complexity if later split.

### Topic creation orchestrator -> directory navigation/worktree/provider-mode/window-launch subflows

- Strength: functional. These steps form one user-visible transaction.
- Distance: low package distance; same owner/deploy.
- Volatility: high.
- Balanced: yes if orchestrator remains small and subflows expose explicit draft/operation contracts.
- Contract: `TopicCreationDraft` stored in user-data through a single adapter; subflow functions return next screen/action rather than mutating unrelated state blindly.
- Knowledge shared: callback-data prefixes, stale-guard invariant, pending thread/text/worktree keys.
- Balancing move: lower distance for sequence-critical logic by keeping orchestration in one module; lower strength for subflows by hiding their private state keys behind a draft adapter.
- Failure modes: lost cleanup order, duplicate tmux window creation on double tap, stale callback accepting bot cwd, pending text not cleared.

### Window-launch service -> tmux manager/provider registry/session state/thread router

- Strength: functional/intrusive risk today because launch touches many implementation APIs in order.
- Distance: high within the package: topics, providers, tmux, state, shell, messaging.
- Volatility: high.
- Balanced: no unless wrapped as an explicit service with a narrow request/result contract.
- Contract: `WindowLaunchRequest(user_id, thread_id, cwd, provider_name, approval_mode, pending_text, worktree)` -> `WindowLaunchResult(window_id, display_name, hook_wait_status, forwarded_pending_text)`.
- Knowledge shared: launch result and state-init facts; tmux/session/thread internals stay behind ports.
- Balancing move: lower strength through a service boundary; keep the ordered state mutation sequence together in one place.
- Failure modes: race with SessionMonitor auto-topic creation, partial state after tmux success but bind failure, hookless provider stale session map, shell setup failure, pending message double-send.

### Polling observe/apply -> polling runtime ports

- Strength: model/shared-state today; target is contract.
- Distance: medium package distance; same process.
- Volatility: high.
- Balanced: not yet; `PaneStatusStrategy` and sibling singletons have high impact.
- Contract: runtime port bundle passed to observe/apply. Ports expose only operations each phase needs: terminal parse/cache, seen-status/startup, lifecycle timers, pane scanning, interactive status.
- Knowledge shared: operation semantics, not singleton layout.
- Balancing move: lower strength by dependency injection. Keep compatibility singletons as adapter defaults during migration.
- Failure modes: subtle poll-cycle state ordering regressions, duplicate timers, tests accidentally using production singletons, RC debounce behavior drift.

### Polling decide -> polling types/provider status value

- Strength: contract.
- Distance: low package distance.
- Volatility: medium/high; status transitions evolve, but pure contract limits blast radius.
- Balanced: yes.
- Contract: `TickContext` -> `TickDecision`, pure functions.
- Knowledge shared: only status/transition values.
- Balancing move: keep as-is and enforce purity.
- Failure modes: importing stateful polling modules, reading time/source data inside decision functions beyond accepted inputs.

### Provider implementations -> provider protocol

- Strength: contract/model.
- Distance: medium package distance; same process; provider CLIs are external.
- Volatility: high for provider behavior.
- Balanced: yes due to explicit `AgentProvider` and immutable capabilities.
- Contract: provider protocol methods and `ProviderCapabilities`.
- Knowledge shared: transcript/status/launch language; provider-specific DTOs stay private.
- Balancing move: keep as-is; add capabilities before special-casing provider names in handlers.
- Failure modes: provider-specific conditionals leaking across handlers, capability drift between picker UI and provider behavior.

### Mini App -> window/session/terminal read projections

- Strength: contract target; some model sharing is acceptable for read-only dashboards.
- Distance: medium package/runtime distance inside same process with HTTP/WebSocket surface.
- Volatility: high if Mini App grows.
- Balanced: yes only if it reads through projections and does not write state.
- Contract: read-only query/projection functions, terminal/transcript APIs.
- Knowledge shared: view models for panes/transcripts/windows.
- Balancing move: lower strength before adding write actions to Mini App.
- Failure modes: HTTP routes importing `window_store`, live terminal API depending on private pane schema, auth bypass around topic/window ownership.

### Messaging pipeline -> Telegram outbound seam/provider messages

- Strength: contract/model.
- Distance: medium package distance; same process.
- Volatility: medium.
- Balanced: yes; queue contracts and safe send helpers hide Telegram details.
- Contract: message task types, queue APIs, `TelegramClient` delivery.
- Knowledge shared: normalized provider messages and Telegram delivery results.
- Balancing move: keep; avoid pushing state-store decisions into message formatting.
- Failure modes: tool-call visibility reading raw store fields, message splitting bypassing entity formatting.

## Key flows

1. Topic creation and binding.
   - Participants: topic creation orchestrator, directory navigation, worktree selection, provider/mode selection, window-launch service, tmux manager, SessionManager/feature ports, thread router, provider registry, messaging skill/shell setup.
   - Data/control path: callback prefix -> stale guard -> selected cwd/worktree decision -> provider/mode -> launch request -> tmux window -> state initialization -> topic binding -> hook wait -> pending text forwarding.
   - Boundary contracts: `TopicCreationDraft`, `WindowLaunchRequest`, `WindowLaunchResult`, provider capabilities, tmux manager create-window result.
   - Local-change expectation: adding a provider launch option should touch provider/mode selection and launch service, not navigation/favorites. Worktree changes should stay in worktree subflow plus launch request.

2. Window state read/write.
   - Participants: handlers, query layer, feature ports, `WindowStateStore`, `SessionManager`, persistence.
   - Data/control path: handlers read projections; handlers write through `SessionManager` or feature ports; ports map to one persisted store; persistence remains debounced and centralized.
   - Boundary contracts: `WindowView` and feature projections for panes/provider/session/worktree/tool visibility/lifecycle.
   - Local-change expectation: adding pane metadata should touch pane-state port/store serialization/tests, not provider metadata, topic creation, and generic query tests unless the UI consumes it.

3. Polling tick.
   - Participants: polling coordinator, runtime port bundle, observe, decide, apply, provider status parser, Telegram sender, tmux manager, pane-status port.
   - Data/control path: observe captures pane/provider/status through ports -> pure `TickContext` -> pure `TickDecision` -> apply emits Telegram/tmux/state side effects.
   - Boundary contracts: runtime ports, `TickContext`, `TickDecision`, provider `StatusUpdate`.
   - Local-change expectation: RC debounce changes stay in terminal-buffer/RC port; pane lifecycle changes stay in pane-status port and pane tests; status decision changes stay in pure decide tests.

4. Provider output to Telegram.
   - Participants: session monitor, provider parser, session/window query, message queue, message sender, Telegram client.
   - Data/control path: transcript/event read -> normalized provider messages -> topic/session resolution -> queued Telegram delivery.
   - Boundary contracts: `AgentProvider.parse_*`, message task types, `TelegramClient`.
   - Local-change expectation: a new transcript field should touch provider parser/tests and message formatting only if displayed.

5. Mini App terminal/transcript view.
   - Participants: Mini App API routes, auth, query/projection layer, tmux manager, transcript reader.
   - Data/control path: authenticated HTTP/WebSocket request -> read-only projection -> terminal/transcript source -> response.
   - Boundary contracts: read-only APIs and view models.
   - Local-change expectation: adding dashboard fields should extend projections, not import or mutate storage internals.

## Module test specifications

### Window state persistence kernel

Behavior tests:

- Persist and reload every canonical `WindowState` field, including panes, provider/session/cwd, origin, worktree, and visibility fields.
- Preserve transient fields as transient: RC probe fields are not serialized.

Unit tests:

- Invalid approval/batch/tool visibility/pane states are rejected.
- Hookless provider switch clears stale session fields and invokes the injected cleanup callback.
- `remove_window`, `prune_stale_window_states`, and `set_window_origin` preserve existing semantics.

Contract tests:

- Feature ports use `WindowStateStore` through public methods only; no external module mutates `window_states[...]` directly except approved store/serialization tests.
- `WindowView` and feature projections are frozen/read-only.

Boundary tests:

- Handler read access to `session_manager` remains forbidden outside write/admin allow-list.
- New handler reads must import query/projection modules, not `window_store`.

Architecture-fitness checks:

- Extend `tests/ccgram/test_query_layer_only_for_handlers.py` to forbid handler imports of `ccgram.window_state_store.window_store` except in explicitly approved write modules during migration.
- Add a focused AST test that feature-port modules are the only modules allowed to access raw `WindowState.panes`, worktree fields, and tool visibility fields outside serialization tests.

### Window state feature ports/projections

Behavior tests:

- Pane rename/subscribe/state transitions update only pane projection behavior and persistence.
- Tool-call visibility and batch mode resolve global defaults vs per-window overrides correctly.
- Worktree metadata persists only when the created window cwd is inside the pending worktree.

Unit tests:

- Each port validates its own inputs and delegates save scheduling exactly once per mutation.
- Projection constructors handle missing/corrupt stored fields with defaults.

Contract tests:

- Feature consumers see only their projection fields; adding a field to `WindowState` does not require unrelated projection changes.

Boundary tests:

- Direct store access in Mini App/routes and handlers fails the architecture test unless it is a registered port implementation.

Architecture-fitness checks:

- Add dependency/import tests for allowed state access: `handlers/**` -> `window_query`/feature ports; feature ports -> `window_state_store`; no reverse import from store to handlers.

### Topic creation orchestrator

Behavior tests:

- Stale callbacks fail closed when `PENDING_THREAD_ID` is missing or topic mismatches.
- Double-click confirm/provider/mode/worktree actions do not create duplicate tmux windows.
- Pending text is forwarded once after successful bind and cleared afterward.
- Existing bound topic short-circuits duplicate creation.

Unit tests:

- `TopicCreationDraft` conversion from/to user-data preserves pending thread/text/worktree state.
- Provider/mode parser rejects malformed callback data.
- Window-launch request construction uses provider capabilities instead of provider-name special cases where possible.

Contract tests:

- Directory navigation subflow cannot call tmux or SessionManager.
- Worktree subflow cannot bind threads or mutate window state.
- Window-launch service is the only subflow that creates tmux windows and performs initial state writes.

Boundary tests:

- Callback-data prefixes map to exactly one subflow handler.
- Cleanup on cancel/abort clears browsing, worktree, pending thread, and pending text keys.

Architecture-fitness checks:

- Add an AST import test for `handlers/topics`: navigation/worktree/provider-mode modules must not import `tmux_manager` or `session_manager`; launch service may.
- Add an integration test that topic creation through a git-worktree path still binds exactly one topic/window and persists worktree metadata.

### Polling runtime ports

Behavior tests:

- Polling status transitions stay identical before/after port extraction for active/idle/done/starting/dead cases.
- Pane scanning reports blocked panes and lifecycle notifications once per transition.
- RC active/off debounce and content-hash cache behavior remain stable.

Unit tests:

- Each runtime port can be tested with an in-memory fake; observe/apply consume interfaces rather than module-level singletons.
- `TerminalBufferPort` strips hook-runner noise and does not shadow non-Claude provider status parsing.
- `LifecycleTimerPort` typing/autoclose/dead-notified state has deterministic tests.

Contract tests:

- `window_tick/decide.py` imports only pure types and provider status value objects.
- `window_tick/observe.py` depends only on the observe-facing subset of polling ports.
- `window_tick/apply.py` depends only on the apply-facing subset of polling ports.

Boundary tests:

- No new code outside polling-runtime adapter imports `pane_status_strategy`, `terminal_poll_state`, or `lifecycle_strategy` directly.
- Test fakes prove side effects are injected, not globally discovered.

Architecture-fitness checks:

- Extend `test_polling_types_purity.py` with a broader import rule: `window_tick/decide.py` must not import `polling_state`, `tmux_manager`, `telegram`, `window_state_store`, or `thread_router`.
- Add an AST test limiting direct imports from `handlers.polling.polling_state` to adapter modules during migration.

### Provider contract

Behavior tests:

- Provider picker and launch mode UI reflect `ProviderCapabilities` for all registered providers.
- Provider status parsing remains provider-specific when `uses_pyte_status_parsing` is false.

Unit tests:

- Each provider returns immutable capabilities and implements required protocol methods.
- New capabilities have default-safe behavior.

Contract tests:

- Provider contract tests fail when picker/menu/status behavior drifts from capabilities.

Boundary tests:

- Handlers do not branch on provider name when a capability can express the behavior.

Architecture-fitness checks:

- Keep/extend `tests/ccgram/providers/test_contracts.py` and picker capability drift tests.

### Telegram outbound seam and messaging pipeline

Behavior tests:

- Safe send/edit handles Telegram failures, splitting, and entity fallback.
- Tool-call visibility uses feature projection/port, not raw state.

Unit tests:

- New Telegram operations require a `TelegramClient` protocol method and fake implementation.

Contract tests:

- Message pipeline works with `FakeTelegramClient` without importing PTB Bot.

Boundary tests:

- Runtime `telegram.ext` imports remain limited to approved modules.

Architecture-fitness checks:

- Add or keep import-boundary tests for PTB runtime import allow-list.

### Mini App read surface

Behavior tests:

- Terminal/transcript/window routes return read-only projections and enforce auth.
- Pane/grid APIs consume pane projection, not raw store fields.

Unit tests:

- Projection serialization handles missing windows and deleted panes.

Contract tests:

- Mini App route tests use projection fakes; no route mutates core state.

Boundary tests:

- Mini App API modules cannot import `window_store` directly unless the module is a projection adapter.

Architecture-fitness checks:

- Extend state-access AST rule to include `miniapp/api/**`.

## Architecture-fitness checks summary

Existing enforced checks to preserve:

- CI runs format, ruff, pyright, deptry, unit tests, and integration tests.
- Handler read/write split: `tests/ccgram/test_query_layer_only_for_handlers.py`.
- Polling pure-types import rule: `tests/ccgram/handlers/polling/test_polling_types_purity.py`.
- Clean interpreter import-cycle guard: `tests/integration/test_import_no_cycles.py`.
- Lazy import contract: `scripts/lint_lazy_imports.py` and related tests.

Recommended new checks with this design:

1. State access boundary check:
   - Handlers and Mini App read through `window_query` or feature projections.
   - Only store/port implementation modules access raw `WindowState` feature fields.
2. Topic subflow import boundary check:
   - Directory/worktree/provider-mode subflows cannot import tmux/session state.
   - Window-launch service is the only topic-creation module allowed to create windows and perform initial state writes.
3. Polling singleton access check:
   - Only polling runtime adapter modules may import production polling singletons.
   - `window_tick/decide.py` remains pure; `observe/apply` accept ports.
4. Provider capability drift check:
   - Picker/menu/status UI cannot hard-code provider behavior that is representable as a capability.
5. Change-locality monitor:
   - Optional script flags commits that touch `window_state_store.py` plus unrelated handler packages after port extraction, to detect whether projections actually reduce churn.

## Design decisions and trade-offs

- Decision: keep one `WindowStateStore` persistence kernel for now.
  - Chosen because: review identified state fan-in and churn, not a proven need for multiple persisted stores.
  - Alternatives considered: split persistence by feature immediately.
  - Trade-offs: projections reduce caller coupling without migration pain; they do not by themselves reduce schema size.
  - Revisit when: feature-port extractions still show repeated cross-feature co-change or migrations become manageable.

- Decision: make topic creation a small orchestrator plus subflow modules, not a fully generic workflow engine.
  - Chosen because: the flow is sequential and domain-specific; a framework would hide the invariants and add ceremony.
  - Alternatives considered: keep `directory_callbacks.py` as-is; introduce a generic state machine.
  - Trade-offs: subflow modules improve review locality while the orchestrator keeps order/race handling visible.
  - Revisit when: additional creation branches make the orchestrator larger than the subflows again.

- Decision: split polling mutable state by runtime concern behind ports.
  - Chosen because: `PaneStatusStrategy` has HIGH impact and `polling_state.py` mixes unrelated caches/timers/UI/pane knowledge.
  - Alternatives considered: physical file split only; direct imports remain.
  - Trade-offs: ports add plumbing but make observe/apply testable with fakes and reduce singleton blast radius.
  - Revisit when: port boundaries become pass-through clutter or poll-cycle order becomes harder to reason about.

- Decision: preserve existing provider and Telegram seams.
  - Chosen because: review found these balanced and already tested.
  - Alternatives considered: provider-specific handler modules or direct PTB use.
  - Trade-offs: protocol updates add boilerplate, but the boilerplate is cheaper than API leakage.
  - Revisit when: provider capabilities become too broad and need sub-capability groups.

- Decision: add fitness checks next to each boundary repair.
  - Chosen because: architecture intent in prose decays. Code does not read docs. Tragic, but observable.
  - Alternatives considered: document-only guidance.
  - Trade-offs: tests may need migration allow-lists during staged refactors.
  - Revisit when: checks become brittle or block legitimate feature work.

## Self-review

| Issue                                                                         | Severity | Evidence/rationale                                                                                | Resolution                                                                                                               |
| ----------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Physical state split deferred may leave `window_state_store.py` large.        | Medium   | Review F3 says the store has CRITICAL upstream impact and high churn.                             | Deliberate: first lower caller strength with feature ports/projections; use change-locality monitor before schema split. |
| Topic creation split can hide sequence and cleanup invariants.                | Medium   | Current file keeps ordering visible, and stale/double-click cleanup is subtle.                    | Keep a small orchestrator and central `TopicCreationDraft`; add behavior and import-boundary tests.                      |
| Polling ports may add abstraction without reducing complexity.                | Medium   | Poll cycle depends on state ordering, timers, pyte cache, provider status, Telegram side effects. | Ports are concern-based and phase-specific; preserve pure decide tests and add fakes for observe/apply.                  |
| Ownership expectations are vague.                                             | Low      | No CODEOWNERS/team file found.                                                                    | Record same-process/module ownership only; do not claim team boundaries.                                                 |
| Mini App future writes could pierce state boundaries.                         | Medium   | Mini App is volatile and near state/terminal data.                                                | Design Mini App as read-only projection consumer; add state-access boundary check before write actions are added.        |
| Provider capability growth could turn `ProviderCapabilities` into a grab bag. | Low      | Capabilities already have many flags.                                                             | Keep behavior capability-gated, but consider sub-capability structs if new flags cluster around separate concerns.       |

No critical unresolved design issue remains. The largest risk is implementation discipline: extracting projections/ports without adding fitness checks would turn this into a nicer diagram over the same old mud. The mud would win. It usually does.

## Open risks

- State projection extraction may not reduce real co-change. Owner: implementer. Revisit after 2-3 feature changes or a change-locality check.
- Polling runtime port boundaries may need adjustment after the first extraction. Owner: implementer. Revisit if tests require broad fake objects or order-dependent fixtures.
- Topic-creation draft may duplicate existing `context.user_data` semantics during migration. Owner: implementer. Revisit once subflows are extracted; remove transitional helpers quickly.
- No ownership file means future module ownership is social, not enforced. Owner: project maintainer. Revisit if contributors grow.
- Existing GitNexus history lacks rename-aware co-change through exposed tools. Owner: reviewer/maintainer. Revisit when history tooling improves.

## Handoff

- Recommended next skill: `architecture-plan`.
- Implementation notes:
  1. Sequence F3 first: add window-state feature ports/projections and fitness checks before any physical persistence split.
  2. Sequence F2 second: extract topic subflows behind `TopicCreationDraft`, keeping `_create_window_and_bind` behavior covered before moving it into a launch service.
  3. Sequence F1 third or parallel only after F3/F2 scaffolding: introduce polling runtime ports with compatibility adapters, then restrict singleton imports.
  4. Keep provider and Telegram seams stable; changes there should be contract extensions, not rewrites.
  5. Each implementation phase should add the boundary fitness check in the same phase as the boundary repair.
- Acceptance signals:
  - Handler/Mini App state reads use projections or feature ports; direct raw store access shrinks.
  - `WindowStateStore` impact decreases for feature changes even if the file still owns serialization.
  - `directory_callbacks.py` becomes a router/orchestrator; navigation/worktree/provider-mode/window-launch logic is separately testable.
  - `polling_state.py` no longer acts as the only import path for all mutable polling strategy state.
  - Existing architecture tests still pass, and new fitness checks prevent regression.
