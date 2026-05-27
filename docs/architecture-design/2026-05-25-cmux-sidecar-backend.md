# Architecture design: cmux sidecar terminal backend

Plain Markdown. Target architecture for adding cmux-native terminal sessions to
ccgram without replacing the existing tmux backend. This is design only;
production source changes belong in a follow-up implementation plan.

Design status: target-state design. The working model below is ready for an
implementation plan, but the cmux command contract and capture performance must
be verified during the first spike. No production symbols are changed by this
artifact.

## Overview

ccgram currently treats tmux as the source of terminal truth: one Telegram Forum
topic binds to one tmux window, and that window hosts one agent session. The new
requirement is to let the same Telegram bot control native cmux workspaces with
comparable integration, without forcing cmux to run tmux internally.

The chosen target keeps one user-facing ccgram bot and introduces terminal
backend routing behind it:

```text
Telegram topic
  -> TerminalUnitRef(backend, unit_id)
      -> TmuxBackend -> tmux window
      -> CmuxBackend -> ccgram-cmux-sidecar -> cmux terminal session
```

The cmux integration is not a separate Telegram bot. It is a sidecar process
owned by the local ccgram runtime and accessed through a narrow local contract.
The sidecar translates stable ccgram terminal operations into cmux socket/CLI
operations and cmux event-stream updates. This follows a strangler-style design:
existing tmux sessions keep using the current implementation while cmux sessions
are added one vertical slice at a time behind the same product surface.

The design intentionally maps one Telegram topic to one cmux terminal session:
the terminal tab/panel, not the workspace. cmux workspaces are mutable groupings
of tabs and stay display metadata. This preserves the ccgram invariant that each
topic has one primary agent session, even when cmux tabs move between
workspaces.

The design uses Balanced Coupling: high-strength relationships stay close
inside a backend adapter or sidecar; higher-distance relationships use explicit
contracts. It does not try to make cmux look like tmux. A cmux terminal-session
ID is not a tmux `@N`. Pretending otherwise would make the diagram prettier and
the code worse. Diagrams are already too agreeable.

## Source inputs and drift notes

- Requirements:
  - Add no-tmux cmux integration where every relevant cmux terminal session
    can be bound to a Telegram topic in ccgram.
  - Prefer one mixed ccgram bot over separate tmux/cmux bots unless clear
    trade-offs show otherwise.
  - Preserve modular architecture and apply Balanced Coupling explicitly.
  - Design the sidecar path before implementation.
- Existing docs/reports checked:
  - `docs/architecture.md`: current module map, topic -> window -> session
    flow, state/query/provider/polling seams, and key design decisions.
  - `docs/architecture-design/2026-05-23-ccgram-target.md`: target-state
    modularity repair, domain map, integration contracts, and fitness style.
  - `docs/architecture-plan/2026-05-23-window-state-feature-ports.md`: existing
    feature-port approach and testing discipline for high-fan-in state.
  - `docs/providers.md`: provider capabilities, hook/transcript/status behavior
    for Claude, Codex, Gemini, Pi, and Shell.
  - `docs/guides.md`: external tmux session discovery and inter-agent messaging
    behavior.
- Existing implementation checked:
  - `src/ccgram/tmux_manager.py`: terminal operations, real tmux window
    creation, capture, send, external session discovery, and `CCGRAM_WINDOW_ID`
    injection.
  - `src/ccgram/hook.py`: hook resolution currently depends on tmux pane
    context and session/window mapping.
  - `src/ccgram/msg_cmd.py` and `src/ccgram/msg_discovery.py`: mailbox peer IDs
    and qualified window identifiers.
  - `src/ccgram/session_map.py`: hook-generated session map and foreign-window
    key handling.
  - `src/ccgram/miniapp/api/terminal.py`: live terminal capture depends on
    tmux pane capture APIs.
  - Handler search results for `tmux_manager` usage in sessions dashboard,
    topic creation, live/screenshot/toolbar/polling/recovery flows.
- External cmux repo/docs checked from a shallow clone of
  `manaflow-ai/cmux`:
  - `README.md`: cmux purpose, CLI/socket API, notifications, session restore,
    and native workspace/surface vocabulary.
  - `docs/cli-contract.md`: commands including `events`, `tree`, `read-screen`,
    `send`, `send-key`, `notify`, `set-status`, and `surface resume`.
  - `docs/events.md`: reconnectable event stream, categories, replay contract,
    and snapshot commands.
  - `docs/agent-hooks.md`: agent hook support and Pi extension behavior.
  - `docs/dock.md`: Dock controls as an optional local UI integration.
  - `Packages/CmuxExtensionKit/README.md` and sources: prototype sidebar API.
    Current cmux source has `CmuxExtensionSidebarSelection.providers` returning
    `[]`, so external sidebar providers are not a viable first integration
    target.
- External research checked through Perplexity:
  - For local control-plane tools, one product surface with pluggable backends
    usually improves UX and operability; separate backend processes are useful
    for failure isolation and implementation volatility.
  - Strangler Fig, feature toggles, plugin architecture, and bounded-context
    guidance favor one front door when adding a new implementation of the same
    product capability.
- Drift risks:
  - cmux is moving quickly. Treat its CLI/socket contracts as integration
    contracts, but validate exact commands during implementation.
  - ccgram architecture docs still say all routing is keyed by tmux window ID.
    This design supersedes that statement for target state only.
  - Existing code has direct `tmux_manager` calls in many handlers. The design
    requires phased migration through backend-neutral ports; no big-bang rewrite.
  - Existing hook and session-map file formats are tmux-shaped. cmux support
    should add sidecar event/session mapping rather than overloading every tmux
    field.

## Working model and scope

Functional areas and business capabilities:

- Telegram topic UX remains the product surface. Users should not need to know
  whether a topic is backed by tmux or cmux for ordinary send/capture/status
  actions.
- Terminal-unit routing becomes the core model below topics. A terminal unit is
  the runtime controlled by a topic. tmux windows and cmux terminal sessions are
  two implementations of that model.
- Provider lifecycle/status remains provider-owned. The terminal backend may
  supply host context and capture, but it should not parse Claude/Codex/Gemini/Pi
  transcripts or own provider business rules.
- The cmux sidecar owns cmux process integration: terminal-session discovery,
  workspace metadata, event replay, command execution, and cmux version
  compatibility.
- ccgram remains the owner of Telegram auth, topic bindings, message queues,
  user preferences, mailbox state, and persistent topic-to-terminal mappings.

Known runtime constraints:

- This is a local macOS integration. cmux is a native macOS app with local
  sockets/CLI; a Docker container is not the default because it would complicate
  access to local app sockets, user PATH, Keychain-adjacent state, and GUI app
  lifecycle.
- The sidecar is a local process. It may later be supervised by ccgram, launched
  on demand, or run manually for development. That launch policy is intentionally
  left to the implementation plan.
- The cmux backend must degrade independently. Sidecar failure may degrade cmux
  topics, but it must not stop tmux topics, Telegram polling, or mailbox access.
- Sidecar communication must be local-only by default: Unix socket, current user
  permissions, no network listener unless explicitly configured later.
- cmux workspaces may contain browser/file/markdown surfaces. MVP operations
  expose terminal tabs/panels as bindable units and keep workspace identity as
  metadata only.

Non-goals for this design:

- No separate Telegram bot for cmux.
- No fake tmux server or `tmux` binary shim.
- No production source edits in this artifact.
- No native cmux sidebar extension until cmux exposes an external provider
  loading mechanism.
- No Mini App write/control surface for cmux in the MVP.
- No physical split of ccgram `state.json` before backend identity ports prove
  the need.

Assumptions accepted for this design:

- The first cmux unit granularity is terminal session: surface/panel terminal,
  not workspace.
- Small ccgram source changes are allowed to add the backend seam.
- Existing tmux behavior must remain default and backward-compatible.
- The first implementation should prioritize binding existing cmux terminal
  sessions before creating new cmux terminals/workspaces from Telegram.

## Domain and volatility map

Core = differentiating behavior and likely to change. Supporting = necessary but
not differentiating. Generic = solved infrastructure, with possible
implementation churn.

| Area                          | Classification     | Volatility  | Rationale                                                                                                                 | Open questions                                                                    |
| ----------------------------- | ------------------ | ----------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Telegram topic control plane  | Core               | High        | The product value is controlling agents from Telegram topics with status, replies, toolbar controls, files, and recovery. | None.                                                                             |
| Terminal unit routing         | Core               | High        | Topic-to-terminal identity becomes the new spine: tmux windows and cmux terminal sessions must coexist.                   | Exact persisted field migration sequence.                                         |
| cmux backend integration      | Core               | High        | New capability and likely to change as cmux APIs and user expectations evolve.                                            | Which cmux commands need socket-level calls for performance.                      |
| Provider lifecycle/status     | Core               | High        | Provider hooks/transcripts/status determine when Telegram topics update and mailbox messages inject.                      | How much cmux agent-hook state can replace current terminal scraping.             |
| Terminal live capture/control | Core               | Medium/High | Capture, send, key controls, screenshots, and live view are user-visible and backend-specific.                            | Whether cmux read-screen is fast enough for current live-view cadence.            |
| ccgram-cmux sidecar runtime   | Supporting         | Medium      | Needed to isolate cmux event/socket behavior and cache workspace state.                                                   | Supervisor ownership and launch policy.                                           |
| Window state persistence      | Core               | High        | State must store backend identity without spreading raw storage knowledge.                                                | Compatibility with existing `state.json` and topic bindings.                      |
| Hook/session-map ingestion    | Supporting         | Medium      | Needed for instant session/status updates; implementation differs by backend.                                             | Whether to extend existing hook files or introduce terminal-event files.          |
| Inter-agent messaging         | Core               | Medium      | Mailbox peers must include cmux units; delivery should remain topic/user-visible.                                         | Whether cmux sessions can self-identify with a stable `CCGRAM_UNIT_ID`.           |
| Mini App terminal view        | Core               | High        | Optional but product-visible; terminal streaming currently assumes tmux panes.                                            | Whether cmux multi-surface controls should be exposed after terminal-session MVP. |
| Telegram Bot API adapter      | Generic            | Low         | Existing protocol seam remains valid regardless of terminal backend.                                                      | None.                                                                             |
| tmux backend                  | Generic/Supporting | Low         | Existing stable backend, must keep working.                                                                               | None.                                                                             |
| cmux CLI/socket API           | Generic/External   | Medium      | External local tool; command contracts exist but are still new.                                                           | Version/capability detection policy.                                              |

## Module map

| Module                      | Responsibility                                                                        | Owned knowledge                                                                                  | Public interface                                                                                                      | Private internals                                                                                 | Owner/deploy expectation                                    | Change vectors                                                |
| --------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- |
| Terminal backend contract   | Define backend-neutral terminal unit operations and identity.                         | Terminal unit vocabulary, operation semantics, capability flags, backend error taxonomy.         | `TerminalBackend` protocol, `TerminalUnit`, `TerminalUnitRef`, `TerminalBackendCapabilities`, `TerminalBackendError`. | None beyond simple DTO validation.                                                                | Same ccgram process.                                        | New operations, capability flags, backend selection.          |
| Backend router              | Resolve topic/window state to the right backend and delegate operations.              | Mapping from persisted backend/unit ID to backend implementation; default tmux behavior.         | `get_backend(ref)`, `terminal_backends.create/list/capture/send/close`.                                               | Backend registry, config defaults, feature flag checks.                                           | Same ccgram process.                                        | Mixed mode rollout, future backends.                          |
| TmuxBackend                 | Preserve current behavior behind the new contract.                                    | tmux window IDs, pane capture, send keys, window creation/kill, external session scan.           | Implements `TerminalBackend`; delegates to existing `tmux_manager`.                                                   | libtmux/subprocess quirks, `@N` IDs, tmux session names.                                          | Same process; no new deploy.                                | Gradual handler migration, existing tmux bug fixes.           |
| CmuxBackend                 | Translate ccgram terminal operations to sidecar calls.                                | cmux terminal-session semantics as seen by ccgram; workspace metadata; timeout/retry policy.     | Implements `TerminalBackend`; local RPC client to sidecar.                                                            | Sidecar socket path, protocol version, response normalization.                                    | Same process client; sidecar process server.                | cmux command coverage, event-driven status/capture cache.     |
| ccgram-cmux-sidecar         | Own live cmux socket/CLI/event integration and terminal-session cache.                | cmux workspace/surface/panel IDs, event cursor/replay, terminal filtering, cmux command details. | Local Unix socket JSON-RPC: list/create/close/capture/send/send_key/stream_events/capabilities.                       | `cmux events`, `cmux tree`, `cmux read-screen`, `cmux send`, socket auth, reconnect loops, cache. | Separate local process supervised by ccgram or user launch. | cmux API changes, performance tuning, event enrichment.       |
| Terminal unit state port    | Store and read backend identity without leaking `WindowState` internals.              | `backend`, `unit_id`, display name, cwd, provider, origin, lifecycle flags.                      | Feature projection and setters in `window_state_ports` or successor package.                                          | Mapping to existing `WindowStateStore` schema; compatibility defaults.                            | Same process; same state file.                              | Additive fields, migration of `window_id` assumptions.        |
| Topic binding/router        | Bind Telegram topics to terminal units instead of only tmux windows.                  | User/thread ownership, display names, topic lifecycle, binding cleanup.                          | Existing thread router plus backend-aware `TerminalUnitRef` accessors.                                                | Legacy `thread_id -> window_id` compatibility.                                                    | Same process.                                               | Binding migration, sessions dashboard, topic close semantics. |
| Terminal operation services | Replace handler-level `tmux_manager` calls with backend-neutral use cases.            | User-visible terminal actions: send, key, capture, screenshot, live stream, close, recover.      | `send_to_unit`, `capture_unit`, `close_unit`, `create_unit`, `list_units`.                                            | Backend selection, permission checks, error-to-Telegram messages.                                 | Same process.                                               | Handler migration, new controls.                              |
| Hook resolution adapter     | Map provider hook events to terminal units for tmux and cmux.                         | Hook source environment, provider session IDs, terminal unit lookup priority.                    | `resolve_hook_terminal_unit(event, env)` returning `TerminalUnitRef` and display metadata.                            | `TMUX_PANE`, `CCGRAM_WINDOW_ID`, `CCGRAM_UNIT_ID`, `CMUX_TERMINAL_ID`, sidecar lookup.            | Hook subprocess plus ccgram process.                        | cmux hook support, provider changes.                          |
| Session/event coordinator   | Merge tmux hook events and cmux sidecar events into ccgram session monitoring.        | Event cursoring, session map updates, done/idle/notification translation.                        | Existing session monitor event interface plus cmux event reader adapter.                                              | File offsets, sidecar event stream, dedupe, provider-specific mapping.                            | Same process plus sidecar.                                  | Lifecycle semantics, mailbox idle gating.                     |
| Messaging peer discovery    | Include cmux terminal units in peer listings and mailbox delivery.                    | Peer ID format, declared task/team metadata, idle delivery eligibility.                          | `ccgram msg list-peers --json`, mailbox send/read/reply, peer projections.                                            | State-file lookup, sidecar terminal cache, delivery injection.                                    | Same process; CLI reads state files.                        | Cross-backend messaging, spawn requests.                      |
| Mini App terminal adapter   | Stream terminal output for tmux and cmux units through one read-only API.             | Terminal frame shape, auth scope, pane/surface listing.                                          | Existing Mini App routes using backend-neutral capture/list operations.                                               | tmux pane capture, cmux terminal-session capture, throttling.                                     | Same optional aiohttp process.                              | cmux multi-surface support, write actions later.              |
| cmux Dock/config helpers    | Optional user convenience for exposing ccgram controls inside cmux.                   | Local cmux config snippets and bridge commands.                                                  | Generated/example `.cmux/dock.json` and `cmux.json` snippets.                                                         | Project trust prompts, local path resolution.                                                     | User/project config, not core runtime.                      | UX polish only.                                               |
| Terminal backend config     | Own backend enablement, default backend selection, sidecar socket path, and timeouts. | Runtime policy for which backends are available and how users opt into cmux.                     | `CCGRAM_TERMINAL_BACKENDS`, `CCGRAM_DEFAULT_TERMINAL_BACKEND`, `CCGRAM_CMUX_SIDECAR_SOCKET`, config projection.       | Env/config parsing, validation, warning text.                                                     | Same ccgram process.                                        | Rollout flags, local setup, sidecar supervision.              |

## Integration contracts

### Telegram topic/router -> terminal backend router

- Strength: contract/model. Topics must share terminal identity fields, but not
  backend implementation details.
- Distance: medium package distance inside one ccgram process; high product
  importance.
- Volatility: high because topic lifecycle and terminal backend support will
  evolve together during migration.
- Balanced: yes if the shared model is `TerminalUnitRef`, not raw tmux or cmux
  identifiers.
- Contract: every topic-bound terminal operation accepts a `TerminalUnitRef` with
  `backend`, `unit_id`, optional `display_name`, and optional `provider_name`.
- Knowledge shared: topic code knows which backend owns a unit; it does not know
  whether a send action uses `tmux send-keys`, `cmux send`, or a socket method.
- Balancing move: lower strength by using a narrow published language for
  terminal identity; keep topic routing close enough to preserve lifecycle
  invariants.
- Failure modes: a handler reconstructs a backend-specific ID from a string,
  topic close kills the wrong runtime, or a legacy path assumes `@N`.

### Terminal operation services -> TmuxBackend

- Strength: contract. The backend hides libtmux and current `tmux_manager`.
- Distance: low/medium package distance; same process and owner.
- Volatility: low for tmux tool behavior, medium for ccgram operation needs.
- Balanced: yes; low implementation volatility makes direct adapter wrapping
  acceptable.
- Contract: `TerminalBackend` methods map to existing tmux operations with
  preserved semantics.
- Knowledge shared: operation request/response and error categories. tmux pane
  IDs, session names, and libtmux quirks stay private.
- Balancing move: leave as close adapter. Do not split tmux infrastructure just
  because cmux exists.
- Failure modes: adapter changes behavior of existing tmux sessions, live view
  loses pane-level capture, or external-session discovery regresses.

### Terminal operation services -> CmuxBackend -> sidecar

- Strength: contract. ccgram and sidecar share a local RPC schema.
- Distance: high runtime distance relative to in-process modules: separate
  process, local socket, independent reconnect/error handling.
- Volatility: high during early cmux integration.
- Balanced: yes only if the contract is narrow and versioned. High distance
  needs low strength.
- Contract: local JSON-RPC over Unix socket with versioned capabilities and
  request IDs. Required methods for MVP: `hello`, `list_terminal_sessions`,
  `capture_screen`, `send_text`, `send_key`, `close_terminal_session`,
  `stream_events`.
- Knowledge shared: terminal-session IDs, workspace metadata, cwd/title/provider
  hints, status, event sequence. cmux command retries, event replay, and
  socket-path details stay private.
- Balancing move: lower strength through anti-corruption. The sidecar translates
  cmux vocabulary to ccgram terminal-unit vocabulary.
- Failure modes: sidecar unavailable, protocol version mismatch, stale terminal
  cache after cmux restart, slow capture, partial command send, or command routed
  to a moved/closed terminal session.

### ccgram-cmux-sidecar -> cmux CLI/socket/events

- Strength: functional/contract mix. The sidecar depends on documented cmux
  commands but must also understand workspace/surface semantics.
- Distance: high system boundary: separate app and process, possibly separate
  release cadence.
- Volatility: medium/high because cmux APIs are young.
- Balanced: borderline but acceptable because the coupling is isolated inside
  the sidecar. It would be unbalanced if spread across handlers.
- Contract: documented cmux CLI/socket behavior from `docs/cli-contract.md` and
  `docs/events.md`; sidecar owns compatibility checks.
- Knowledge shared: cmux workspace/surface/panel IDs, event sequence/cursor,
  command arguments, and capability responses. ccgram core does not share this
  knowledge.
- Balancing move: lower distance locally by keeping cmux-specific code in one
  sidecar package and lower strength upstream by normalizing to ccgram DTOs.
- Failure modes: cmux command output changes, event replay gap, socket auth
  changes, cmux app not running, or multiple cmux app variants/sockets.

### Hook subprocess -> hook resolution adapter

- Strength: model today for tmux, target contract for mixed backends.
- Distance: medium runtime distance: short-lived hook subprocess writes files or
  events consumed by ccgram.
- Volatility: high because provider hook formats and terminal hosts vary.
- Balanced: not yet; current tmux-only hook resolution is too strong for cmux.
- Contract: resolve hook terminal unit in priority order:
  1. `CCGRAM_UNIT_ID` or `CCGRAM_WINDOW_ID` when explicitly set.
  2. `CMUX_TERMINAL_ID`/`CMUX_SURFACE_ID`/`CMUX_PANEL_ID` via sidecar lookup.
  3. `TMUX_PANE` via existing tmux resolution.
- Knowledge shared: hook env vars and resolved terminal unit reference. Provider
  hook payloads remain provider parser concern.
- Balancing move: lower strength by making hook resolution backend-neutral;
  keep provider-specific hook parsing separate.
- Failure modes: duplicate events if both cmux and ccgram hooks are installed,
  missing env vars in cmux terminals, stale cmux terminal mapping, or old hooks
  writing only tmux-shaped session-map keys.

### Session monitor -> sidecar event stream

- Strength: contract. Session monitor should consume normalized terminal events.
- Distance: medium/high runtime distance due to sidecar stream.
- Volatility: medium/high for event shapes during rollout.
- Balanced: yes if sidecar emits ccgram-shaped events and monitor does not parse
  raw cmux frames.
- Contract: sidecar event frames include `seq`, `backend`, `unit_id`, `event`,
  `provider_name`, `session_id`, `status`, timestamps, and optional redacted
  metadata. Frames are persisted or cursored only after successful handling.
- Knowledge shared: lifecycle/status event language, not raw cmux event payloads.
- Balancing move: lower strength through event normalization and replay/cursor
  guarantees.
- Failure modes: event loss on sidecar restart, duplicate ready notifications,
  mailbox injection before agent idle, or cmux event replay gap without snapshot
  refresh.

### Window state feature ports -> terminal unit state

- Strength: contract/model. Existing state code needs new fields but should hide
  schema details.
- Distance: low package distance; same persistence kernel.
- Volatility: high during migration.
- Balanced: yes if terminal identity gets a feature port rather than raw field
  reads everywhere.
- Contract: add a projection such as `TerminalIdentity(backend, unit_id,
legacy_window_id, cwd, provider_name, origin, external)` and setters that
  schedule saves once per real mutation.
- Knowledge shared: feature-level identity fields; raw serialization format
  remains private.
- Balancing move: keep low distance to persistence, lower strength for callers.
- Failure modes: duplicate identity fields drift, topic bindings point to one ID
  while state points to another, or old `window_id` indexes become ambiguous.

### Messaging peer discovery -> terminal unit state and sidecar

- Strength: model. Peers share identity, provider, cwd, task/team, and idle
  eligibility.
- Distance: medium package/process distance: CLI reads state files and may query
  sidecar.
- Volatility: medium.
- Balanced: acceptable if peer ID format is explicit and stable.
- Contract: peer IDs become backend-qualified. Examples: `tmux:ccgram:@3` and
  `cmux:term:9B6920C1`. CLI output includes backend separately so humans are
  not forced to parse it.
- Knowledge shared: peer identity and user-declared metadata. Injection details
  stay in backend delivery service.
- Balancing move: lower strength by making peer IDs opaque and using metadata
  fields for display/filtering.
- Failure modes: mailbox directory sanitization collisions, missing
  `CCGRAM_UNIT_ID` in cmux sessions, cross-backend replies delivered to wrong
  session, or spawn requests targeting unsupported backend.

### Backend config -> backend router

- Strength: contract. Runtime config selects available backends and sidecar
  connection details.
- Distance: low package distance inside one process.
- Volatility: medium during rollout, then low.
- Balanced: yes because the config shape is small and close to the router.
- Contract: backend config exposes enabled backend names, default backend,
  sidecar socket path, operation timeouts, and feature-flag status. It does not
  expose handler-specific choices.
- Knowledge shared: deployment policy and local socket settings. Backend
  internals and cmux command details stay private.
- Balancing move: keep config close to backend routing; do not duplicate backend
  enablement checks in handlers.
- Failure modes: cmux appears in UI when disabled, sidecar socket path differs
  between sidecar and backend client, or tmux default changes unexpectedly.

### Mini App -> terminal backend router

- Strength: contract. Mini App needs read-only terminal frames and unit metadata.
- Distance: medium runtime distance: HTTP/WebSocket server in same process, plus
  optional sidecar for cmux.
- Volatility: high if Mini App grows write actions.
- Balanced: yes for read-only operations; write actions should be a separate
  design.
- Contract: route auth token to `TerminalUnitRef`, call backend-neutral capture
  and optional list-subtargets operation.
- Knowledge shared: terminal frame text/ANSI and subtarget metadata. cmux
  surface internals stay behind backend adapters.
- Balancing move: lower strength by keeping Mini App read-only in MVP.
- Failure modes: token authorizes tmux-style `window_id` but not cmux unit,
  capture floods sidecar, or cmux terminal session moves/closes mid-stream.

## Target data model

### TerminalUnitRef

A terminal unit is the smallest ccgram-controlled runtime bound to a Telegram
topic. It is backend-owned and must be treated as opaque outside the backend
router.

```text
TerminalUnitRef
  backend: "tmux" | "cmux"
  unit_id: string
  display_id: string        # stable human/debug form, e.g. tmux:@3 or cmux:<terminal-id>
```

Rules:

- `backend` is required for all new bindings.
- `unit_id` is backend-local and not parsed by handlers.
- Existing tmux bindings with only `window_id` load as `backend="tmux"` and
  `unit_id=<window_id>`.
- cmux unit IDs use cmux terminal-session IDs, never a fake tmux window ID and
  never workspace IDs.

### TerminalUnit

A terminal unit projection is a read model used by dashboards and handlers.

```text
TerminalUnit
  ref: TerminalUnitRef
  title: string
  cwd: string | null
  provider_name: string | null
  state: "starting" | "working" | "waiting" | "ready" | "dead" | "unknown"
  supports_capture: bool
  supports_send_text: bool
  supports_send_key: bool
  supports_close: bool
  supports_resume: bool
  backend_metadata: redacted dict
```

`backend_metadata` is for diagnostics only. User-visible code should prefer
first-class fields so backend-specific metadata does not become a covert API.

### Sidecar terminal-session cache

The sidecar may cache cmux state for performance:

```text
CmuxTerminalSessionRecord
  terminal_id: string
  workspace_id: string | null
  workspace_title: string
  pane_id: string | null
  surface_id: string | null
  panel_id: string | null
  title: string
  cwd: string | null
  provider_name: string | null
  status: string
  latest_notification_text: string | null
  unread_count: int
  cmux_boot_id: string
  cmux_event_seq: int
```

The cache is sidecar-private. ccgram requests snapshots when it needs durable
truth and treats cache-backed responses as operational read models, not
persistence. Workspace fields are metadata, not identity.

### Sidecar RPC envelope

The sidecar protocol is intentionally boring JSON-RPC over a local Unix socket.
Boring wins here. A clever protocol would add choreography where the system needs
contracts.

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "method": "capture_screen",
  "params": { "terminal_id": "...", "with_ansi": true }
}
```

Responses use one of two shapes:

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "result": { "text": "...", "truncated": false }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "error": { "code": "not_found", "message": "terminal session not found" }
}
```

Stable error codes:

- `unavailable`: cmux app/socket/CLI is unavailable.
- `not_found`: terminal session no longer exists.
- `unsupported`: operation is not supported for the target or sidecar version.
- `no_terminal_surface`: target is not a terminal-capable panel.
- `timeout`: operation exceeded configured deadline.
- `rejected`: sidecar rejected invalid input or unsafe target.
- `internal_error`: unexpected sidecar failure; logs hold details.

### Sidecar event frame

The sidecar event stream normalizes cmux events before ccgram sees them:

```json
{
  "seq": 42,
  "backend": "cmux",
  "unit_id": "terminal-session-uuid",
  "event": "unit.status.changed",
  "status": "ready",
  "provider_name": "claude",
  "session_id": "optional-provider-session",
  "occurred_at": "2026-05-25T12:00:00Z",
  "redacted": true,
  "metadata": { "cmux_event": "workspace.selected" }
}
```

Rules:

- `seq` is sidecar-local and monotonic.
- Raw cmux event payloads stay sidecar-private unless explicitly redacted and
  promoted to first-class fields.
- ccgram persists the last successfully handled `seq`, not the last received
  one.
- Replay gaps force a sidecar snapshot refresh before status decisions.

## Key flows

1. Bind existing cmux terminal session to a Telegram topic.
   - Participants: user, Telegram topic handler, backend router, CmuxBackend,
     sidecar, cmux app, thread router, terminal unit state port.
   - Data/control path: user selects "Bind cmux terminal session" -> ccgram
     calls `CmuxBackend.list_units()` -> sidecar refreshes `cmux tree --all --json`
     or socket snapshot -> handler renders terminal-session picker -> user selects
     terminal -> ccgram stores topic binding with `backend=cmux` and
     `unit_id=<terminal_id>` -> topic title/status initializes from
     `TerminalUnit` projection. Workspace id/title are metadata.
   - Boundary contracts: `TerminalUnit`, backend-neutral picker callbacks,
     sidecar `list_terminal_sessions`.
   - Local-change expectation: cmux discovery changes stay in sidecar; Telegram
     picker UX changes stay in handlers; state field changes stay in terminal
     identity port.

2. Create new cmux terminal session from Telegram.
   - Participants: topic creation flow, provider/mode picker, backend router,
     CmuxBackend, sidecar, cmux app, provider registry.
   - Data/control path: directory/provider/mode selected -> launch request has
     `backend=cmux` -> sidecar calls cmux terminal/workspace creation with cwd and
     command -> cmux starts terminal with `CCGRAM_UNIT_ID` and provider command
     when supported -> sidecar returns terminal ID -> ccgram persists binding
     and waits for provider hook/session discovery.
   - Boundary contracts: backend-neutral `create_unit` request and `TerminalUnit`
     result, provider launch command contract.
   - Local-change expectation: provider command logic remains provider module;
     cmux launch mechanics stay in sidecar; topic flow only knows selected
     backend.

3. Send a Telegram message to a cmux terminal session.
   - Participants: text handler, terminal operation service, CmuxBackend,
     sidecar, cmux app.
   - Data/control path: text handler resolves topic -> `TerminalUnitRef` ->
     `send_text(ref, text, enter=True)` -> sidecar validates the terminal
     session -> calls cmux send/socket command -> returns success/failure ->
     handler replies with sent/failure status as today.
   - Boundary contracts: `send_text` idempotence/timeout semantics and sidecar
     error codes.
   - Local-change expectation: text routing does not change when cmux command
     changes from CLI to socket.

4. Capture screen / live view.
   - Participants: screenshot/live handlers or Mini App, terminal operation
     service, backend, sidecar, cmux app.
   - Data/control path: request resolves ref -> `capture(ref, ansi=True)` ->
     sidecar calls `cmux read-screen` for the terminal session or socket method ->
     returns text/ANSI plus truncation metadata -> existing rendering/splitting
     code handles response.
   - Boundary contracts: capture response includes text, `truncated`,
     `source_changed_at`, and optional dimensions.
   - Local-change expectation: rendering and Telegram upload remain shared;
     backend capture mechanics stay in backend/sidecar.

5. Provider hook event from cmux terminal.
   - Participants: provider CLI, ccgram hook command, hook resolution adapter,
     sidecar, session map/event writer, session monitor.
   - Data/control path: provider hook runs -> env contains `CCGRAM_UNIT_ID` or
     `CMUX_TERMINAL_ID` -> hook resolution returns `TerminalUnitRef` -> event
     is written with backend-qualified unit reference -> session monitor routes
     status/message updates to the bound topic.
   - Boundary contracts: backend-aware hook event shape and resolution priority.
   - Local-change expectation: provider hook parsing stays provider-specific;
     terminal resolution stays in hook adapter.

6. cmux event stream updates topic status.
   - Participants: sidecar, cmux events stream, ccgram session/event
     coordinator, Telegram status updater.
   - Data/control path: sidecar subscribes to `cmux events` with cursor ->
     normalizes workspace/surface/notification/agent events -> ccgram consumes
     sidecar event stream -> updates unit state/status and maybe topic emoji ->
     snapshots are fetched when event replay gap occurs.
   - Boundary contracts: sidecar normalized event stream with replay/cursor
     semantics.
   - Local-change expectation: cmux event catalog drift is isolated in sidecar;
     ccgram status semantics stay in existing polling/session flow.

7. Inter-agent messaging to a cmux unit.
   - Participants: `ccgram msg`, mailbox, message broker, backend router,
     CmuxBackend, sidecar.
   - Data/control path: agent sends to peer ID -> mailbox stores
     backend-qualified recipient -> broker waits for idle/ready status ->
     `send_text` injects message into recipient unit -> Telegram silent notice
     is sent as today.
   - Boundary contracts: peer ID format, idle status projection, backend send.
   - Local-change expectation: mailbox persistence remains file-based; delivery
     mechanics move behind terminal backend.

8. Close/recover cmux topic.
   - Participants: topic lifecycle/recovery handlers, backend router,
     CmuxBackend, sidecar, cmux app, provider resume logic.
   - Data/control path: user closes topic or chooses recovery -> backend close or
     create/resume operation runs -> state binding is removed or updated ->
     cmux surface resume command may be used where supported.
   - Boundary contracts: backend `close_unit`, provider resume capabilities,
     sidecar create/resume result.
   - Local-change expectation: recovery UI remains shared; backend-specific
     resume mechanics stay in backend/sidecar/provider modules.

## Module test specifications

### Terminal backend contract

Behavior tests:

- Existing tmux topic can still send, capture, close, and recover through
  `TmuxBackend` with unchanged visible behavior.
- cmux topic resolves to `CmuxBackend` based on persisted backend field.

Unit tests:

- Missing backend defaults to tmux for legacy state.
- Unknown backend returns a user-safe unsupported-backend error.
- `TerminalUnitRef` rejects blank backend or unit ID.

Contract tests:

- Both backends satisfy the same protocol in a shared backend test suite.
- Operation errors normalize to stable categories: unavailable, not_found,
  unsupported, timeout, rejected, internal_error.

Boundary tests:

- Handlers cannot import `tmux_manager` directly after their feature is migrated;
  allow-list only adapter modules and legacy unmigrated files.
- Backend-neutral services never parse cmux terminal IDs or tmux `@N` IDs.

Architecture-fitness checks:

- AST test that `src/ccgram/handlers/**` direct `tmux_manager` imports shrink per
  migration task and eventually fail outside approved terminal-operation
  services.
- AST test that only backend adapters import sidecar client or cmux-specific
  modules.

### Backend router

Behavior tests:

- Mixed `/sessions` view lists tmux and cmux units with clear backend labels.
- Topic operations route to the backend stored on the topic binding.

Unit tests:

- Registry refuses duplicate backend names.
- Disabled cmux backend surfaces a setup message instead of crashing.
- Backend capabilities gate UI buttons such as live/capture/close/resume.

Contract tests:

- Router passes through normalized backend errors without leaking sidecar stack
  traces or raw command output.

Boundary tests:

- No code path falls back to tmux when a cmux unit is present but sidecar is down;
  it must report cmux unavailable.

Architecture-fitness checks:

- Static check that topic binding DTOs include backend for new writes.

### Terminal backend config

Behavior tests:

- Default config enables only tmux and preserves existing startup behavior.
- Enabling cmux with a sidecar socket shows cmux bind/create actions when the
  sidecar reports compatible capabilities.
- Disabled or unhealthy cmux backend renders setup/degraded messages, not stack
  traces.

Unit tests:

- Unknown backend names are rejected with clear config errors.
- Sidecar timeout values are clamped to safe minimum/maximum ranges.
- Default backend cannot be set to a disabled backend.

Contract tests:

- Backend router receives a typed config projection, not raw env/config strings.

Boundary tests:

- Handler modules do not read cmux backend env vars directly.

Architecture-fitness checks:

- Static test that only config/backend modules read `CCGRAM_CMUX_*` and
  `CCGRAM_TERMINAL_BACKEND*` settings.

### ccgram-cmux-sidecar

Behavior tests:

- Lists terminal sessions from a fake cmux tree snapshot.
- Sends text to the selected terminal session.
- Captures terminal text and returns truncation metadata.
- Reconnects event stream after EOF and resumes from the last processed seq.
- Refreshes snapshot after a cmux event replay gap.

Unit tests:

- Normalizes cmux workspace/surface/panel events into terminal-session event
  frames.
- Rejects commands for non-terminal panels.
- Applies command timeouts and returns stable error categories.
- Handles multiple cmux socket paths/app variants by explicit configuration.

Contract tests:

- JSON-RPC schema version negotiation rejects incompatible clients.
- `capabilities` reports supported methods and cmux version metadata.
- Event stream frames are newline-delimited JSON with monotonic sidecar sequence.

Boundary tests:

- Sidecar never reads ccgram Telegram state or bot token.
- Sidecar never writes ccgram `state.json`; only ccgram owns persistent bindings.

Architecture-fitness checks:

- Sidecar package has no imports from `telegram`, PTB, or ccgram handler modules.

### Terminal unit state port

Behavior tests:

- Legacy `state.json` with tmux `window_id` loads as `backend=tmux`.
- New cmux binding persists backend/unit fields and round-trips without losing
  provider/cwd/display metadata.

Unit tests:

- Setting the same terminal identity is a no-op and schedules no save.
- Changing backend or unit ID schedules exactly one save.
- Invalid backend names are rejected at the port boundary.

Contract tests:

- Query layer exposes terminal identity through frozen projections.
- Handlers do not read raw terminal identity fields directly.

Boundary tests:

- Persisted schema remains backward compatible: missing backend fields do not
  break old state files.

Architecture-fitness checks:

- Extend `test_window_state_access_audit.py` to approve raw terminal identity
  fields only in the persistence kernel and feature port.

### Hook resolution adapter

Behavior tests:

- Hook with `CCGRAM_UNIT_ID=cmux:...` resolves without tmux.
- Hook with `CMUX_TERMINAL_ID` resolves through sidecar lookup.
- Hook with only `TMUX_PANE` keeps existing tmux behavior.

Unit tests:

- Resolution priority is deterministic when multiple env vars are present.
- Missing/unknown terminal context logs a debug-safe message and drops or stores
  event according to current hook policy.
- Provider session ID maps to backend-qualified event key.

Contract tests:

- Hook event file schema includes backend-qualified terminal identity.
- Existing tmux hook schema remains readable during migration.

Boundary tests:

- Hook resolution does not call Telegram APIs.
- Hook resolution does not import handler modules.

Architecture-fitness checks:

- Lazy-import lint covers any in-function imports added to avoid hook cycles.

### Session/event coordinator

Behavior tests:

- cmux `workspace.selected`, `notification.created`, and `agent.hook.*` events
  update the bound topic once and do not duplicate tmux events.
- Sidecar replay gap triggers a snapshot refresh before processing live events.

Unit tests:

- Event dedupe by `(backend, unit_id, event_id)`.
- Status transitions preserve current ready/working/waiting topic emoji rules.
- Mailbox delivery gates on normalized idle/ready state.

Contract tests:

- Sidecar event frames are accepted only at compatible schema versions.
- Raw cmux event payloads do not leak into session monitor handlers.

Boundary tests:

- Session monitor can run without cmux sidecar when no cmux units exist.
- Sidecar outage degrades cmux topics only, not tmux topics.

Architecture-fitness checks:

- Event coordinator imports sidecar client, but provider modules and Telegram
  handlers do not.

### Messaging peer discovery

Behavior tests:

- `ccgram msg list-peers --json` includes tmux and cmux peers with backend,
  provider, cwd, branch, task, and team fields.
- Sending to a cmux peer writes to the expected mailbox directory and broker
  injects via `CmuxBackend`.

Unit tests:

- Backend-qualified peer IDs sanitize without collisions.
- Bare legacy IDs qualify as tmux peers only in the local tmux backend.
- cmux self-identification works through `CCGRAM_UNIT_ID` or sidecar lookup.

Contract tests:

- Existing mailbox JSON remains readable.
- New peer ID fields are optional for old messages and required for new cmux
  messages.

Boundary tests:

- `ccgram msg` does not require a running Telegram bot to list peers.
- Sidecar-unavailable peers are marked unavailable, not silently dropped.

Architecture-fitness checks:

- Mailbox code depends on terminal identity projections, not raw window state.

### Mini App terminal adapter

Behavior tests:

- Existing tmux Mini App terminal stream remains unchanged.
- cmux terminal-session stream returns frames through the same WebSocket
  protocol.

Unit tests:

- Token payload authorizes exactly one terminal unit regardless of backend.
- Capture throttling prevents sidecar flooding.
- Large cmux captures are truncated with metadata.

Contract tests:

- HTTP pane/surface listing uses backend-neutral subtarget projections.

Boundary tests:

- No write operations are introduced in the cmux MVP.
- cmux unit tokens cannot capture a tmux unit and vice versa.

Architecture-fitness checks:

- Mini App routes import terminal operation service, not `tmux_manager` or
  sidecar client directly.

## Security and privacy

- The sidecar must not receive Telegram bot tokens, Telegram user IDs, raw chat
  text, or allowed-user lists. It only needs terminal-unit commands and cmux
  workspace data.
- ccgram remains the permission boundary. Topic ownership checks happen before a
  backend operation is invoked.
- Sidecar logs must not include command text, prompt text, raw terminal capture,
  notification bodies, or secrets. Use lengths, hashes, unit IDs, and error
  categories.
- Local socket permissions should restrict access to the current user. Network
  listening is out of scope for the MVP.
- Project-level cmux Dock/config helpers are optional and must rely on cmux's
  trust prompt. Do not auto-write project cmux config as part of backend setup.
- The sidecar should treat cmux event streams as local-sensitive data. It should
  normalize and redact before forwarding to ccgram.
- A failed or compromised sidecar must not mutate ccgram persistent state
  directly. Only ccgram writes `state.json`, `session_map.json`, and mailbox
  state.

## Rollout and compatibility

1. Add backend-neutral identity while all existing topics continue to load as
   tmux units.
2. Wrap existing tmux operations in `TmuxBackend` and keep tmux as the default.
3. Add cmux backend behind an explicit feature flag. Hidden unless sidecar is
   configured and healthy.
4. Ship bind-existing-cmux-terminal-session before create-cmux-terminal-session.
   Discovery is easier to validate than launch orchestration.
5. Add cmux send/capture/status slices one at a time. Each slice shrinks direct
   `tmux_manager` access for the touched feature.
6. Add cmux hook/session event support after basic terminal control works.
7. Expose mixed `/sessions` and mailbox peer discovery after identity and basic
   operations are stable.

Rollback rules:

- Disabling cmux backend hides cmux creation/bind UI but keeps existing cmux
  topic bindings visible as degraded/unavailable instead of deleting them.
- Existing tmux topics must continue operating when cmux sidecar is stopped.
- State migrations must be additive. Old `window_id`-only records load as tmux.

## Design decisions and trade-offs

- Decision: keep one ccgram bot for tmux and cmux.
  - Chosen because: tmux and cmux are two implementations of the same product
    capability: Telegram control over local agent terminal sessions. One front
    door keeps topics, permissions, message queues, and user commands coherent.
  - Alternatives considered: separate cmux bot/instance; cmux-only fork.
  - Trade-offs: one bot has a larger state model and test matrix, but avoids
    duplicate Telegram UX and eventual merge pain.
  - Revisit when: cmux support becomes a separate product with different users,
    auth, deployment, or lifecycle.

- Decision: introduce a sidecar process for cmux, not direct cmux calls from all
  handlers.
  - Chosen because: cmux socket/events/CLI details are a high-distance,
    medium-volatility external integration. Isolating them lowers coupling
    strength across the ccgram core.
  - Alternatives considered: direct cmux CLI calls inside ccgram;
    tmux-compatible shim.
  - Trade-offs: sidecar adds process supervision and local RPC failure modes,
    but protects the core bot from cmux API churn and event-loop complexity.
  - Revisit when: the cmux CLI/socket API is stable enough and direct calls prove
    simpler for all required operations.

- Decision: map one Telegram topic to one cmux terminal session.
  - Chosen because: a cmux workspace is a mutable grouping of tabs; the terminal
    tab/panel is the controlled session. Topic routing must follow the terminal
    session when it moves between workspaces.
  - Alternatives considered: topic per cmux workspace; topic per cmux app window.
  - Trade-offs: terminal-session mapping creates more rows when users open many
    tabs, but avoids command routing ambiguity and workspace history pollution.
  - Revisit when: cmux changes workspace semantics so workspaces become durable
    terminal-session identity, not mutable grouping.

- Decision: do not fake tmux IDs for cmux.
  - Chosen because: backend identity is core domain language. A fake `@N` would
    spread accidental compatibility logic and make errors hard to diagnose.
  - Alternatives considered: tmux compatibility shim and synthetic window IDs.
  - Trade-offs: backend-neutral migration touches more code now, but avoids a
    long-lived identity lie.
  - Revisit when: never, unless cmux itself exposes a tmux server with real
    tmux-compatible IDs and semantics.

- Decision: keep `WindowStateStore` as the persistence kernel and add terminal
  identity ports.
  - Chosen because: current architecture already moved toward feature ports and
    warns against premature physical store splits.
  - Alternatives considered: new `TerminalUnitStore` file.
  - Trade-offs: one state file remains high-value and must be guarded by tests,
    but migration is compatible and local.
  - Revisit when: terminal identity/state changes independently enough to earn a
    physical split.

- Decision: use sidecar normalized events instead of raw cmux event frames in
  ccgram core.
  - Chosen because: raw cmux events are cmux-owned vocabulary and will change at
    a different cadence than ccgram topic semantics.
  - Alternatives considered: session monitor directly subscribes to `cmux
events`.
  - Trade-offs: normalization adds code and may hide useful cmux details, but it
    gives ccgram a stable contract.
  - Revisit when: sidecar proves unnecessary after implementation and event
    vocabulary is stable.

## Open risks

- cmux capture performance may not match tmux capture-pane live-view cadence.
  Owner: implementation plan. Revisit after measuring `cmux read-screen` and
  socket alternatives.
- cmux hooks and ccgram hooks may duplicate provider events. Owner: hook
  implementation. Revisit when installing cmux and ccgram hooks together for
  Claude/Codex/Gemini/Pi.
- cmux terminal-session IDs must remain stable across workspace moves. Owner:
  sidecar design. Revisit if cmux exposes only workspace-scoped ephemeral IDs.
- Sidecar process supervision is not specified by current ccgram runtime.
  Owner: implementation plan. Revisit before coding sidecar startup/shutdown.
- Telegram topic binding schema migration may touch many handlers. Owner:
  architecture-plan. Revisit after impact analysis of thread router and
  window-state identity symbols.
- Current Mini App token payload likely assumes tmux window IDs. Owner: Mini App
  implementation slice. Revisit when designing backend-neutral auth payloads.
- cmux external sidebar plugins are not currently loadable, so cmux UI
  integration beyond Dock/config helpers depends on upstream cmux changes.
  Owner: future integration. Revisit only after sidecar MVP works.

## Self-review

| Issue                                                        | Severity | Evidence/rationale                                                                      | Resolution                                                                                                         |
| ------------------------------------------------------------ | -------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Mixed backend identity could become a new god model          | High     | Topic routing, state, hooks, messaging, Mini App, and sessions all need identity.       | Use `TerminalUnitRef` as the published language and terminal identity feature port as the only raw-state boundary. |
| Sidecar RPC can become a distributed monolith                | Medium   | If ccgram depends on broad sidecar internals, every cmux change cascades.               | Keep sidecar contract terminal-session-level and versioned; sidecar owns raw cmux vocabulary.                      |
| Initial MVP may underdeliver versus tmux multi-pane features | Medium   | tmux pane-level capture/listing is richer than cmux terminal-session MVP.               | Declare terminal-session-first scope and expose subtargets only through optional backend capabilities.             |
| Existing code has many direct `tmux_manager` calls           | High     | Search shows handlers, Mini App, sessions dashboard, and hook paths call tmux directly. | Migration must be incremental through terminal operation services and structural allow-list tests.                 |
| Hook/session-map compatibility is underspecified             | Medium   | Current hooks resolve via `TMUX_PANE`; cmux uses terminal/surface/panel IDs.            | Add hook resolution adapter and backend-qualified event schema before relying on cmux hooks.                       |
| Sidecar launch policy could leak into product design         | Low      | A Docker-style sidecar would fight macOS local app sockets.                             | Treat sidecar as a local process by default; containerization is out of scope unless later required.               |

## Handoff

- Recommended next step: `architecture-plan` to sequence implementation into
  narrow vertical slices.
- Implementation notes:
  - Start with backend-neutral identity and `TmuxBackend` wrapper before adding
    cmux behavior. Prove the existing tmux path still works behind the seam.
  - Add cmux discovery/bind existing terminal session before cmux terminal
    creation. Binding is easier to test than launch orchestration.
  - Keep sidecar protocol small and versioned from day one.
  - Add structural tests with allow-lists so direct `tmux_manager` usage shrinks
    instead of silently growing.
  - Do not attempt native cmux sidebar extension until sidecar MVP is stable;
    current cmux extension provider list is not externally pluggable.
- Acceptance signals:
  - Existing tmux session flows pass unchanged through `TmuxBackend`.
  - A cmux terminal session can be bound to a Telegram topic, receive text,
    return screen capture, and show status without tmux running.
  - `/sessions` shows both tmux and cmux units in one bot.
  - Hook events from a cmux-hosted provider resolve to a backend-qualified topic.
  - Sidecar outage marks only cmux topics degraded; tmux topics keep working.
  - Architecture tests prevent handlers from importing cmux or tmux
    infrastructure directly after their migration slice.
