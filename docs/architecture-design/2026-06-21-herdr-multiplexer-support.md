# Architecture design: herdr as an alternative multiplexer

Plain Markdown. Target architecture for adding [herdr](https://github.com/ogulcancelik/herdr) as a second terminal multiplexer alongside tmux, behind one contract seam. Design only; production source changes belong in a follow-up implementation plan. tmux stays the default; herdr is additive.

## Design Revision (2026-06-22)

The original scoping decision 1 ("Identity keying — thin: treat herdr's `pane_id` as the opaque `window_id`") and the Telegram topic mapping section ("topic = pane = agent") were **superseded** during implementation (plan `docs/plans/20260622-herdr-tab-topics.md`).

**Actual implemented identity:** `window_id = tab_id` (`"wN:tM"`), not pane id. One ccgram topic = one herdr tab, mirroring the tmux-window analog. A split tab (agent team) is one topic with N panes — handled by ccgram's existing multi-pane awareness (`/panes`, `list_panes`), not N separate topics.

**Topic label:** `"<workspace> ▸ <tab>"` (tab name primary, not agent name). This fixes the `"ccgram ▸ claude"` collision when a workspace runs two sessions of the same agent.

**Session-map key:** `herdr:<tab_id>` (hook resolves pane→tab via injected probe before writing).

**Restart re-resolution:** `_resolve_by_session_id` in `window_resolver.py` maps a stale `herdr:<tab_id>` key to the new tab id via the shared Claude session uuid — same anchor as original, different key shape.

The rest of the design (Multiplexer Protocol, capability flags, anti-corruption layer, polling-first, hook identity from `$HERDR_PANE_ID`, additive scope) remains as written.

## Overview

ccgram is a Telegram control plane for terminal-hosted AI coding agents. Today every terminal operation goes through the concrete `tmux_manager` singleton; there is no multiplexer abstraction. This design introduces a `Multiplexer` contract that both tmux and herdr satisfy, mirroring the existing `AgentProvider` seam, so the rest of the system stops knowing which multiplexer it is talking to.

The design follows the Balanced Coupling principle of the cheapest balancing move: the multiplexer is a **Generic** subdomain whose **implementation volatility just rose** (one backend becoming two, herdr being young at v0.7.0). The correct response to rising implementation volatility on a generic subdomain is an explicit contract that hides backend internals — not a rewrite, not generic decoupling.

Four scoping decisions were confirmed with the maintainer:

1. **Identity keying — thin.** Treat herdr's `pane_id` (`w2:p1`) as the opaque `window_id` string. Reuse the existing restart re-resolution path (`resolve_stale_ids`), anchored on `session_id` for herdr. No re-keying of `state.json` or `callback_data`.
2. **Status source — reuse polling now.** On herdr, the existing polling loop reads `agent_status`/captures via the backend; the herdr event stream is deferred to a later phase.
3. **Session identity — keep ccgram's own hook.** The Claude hook resolves identity from `$HERDR_PANE_ID` instead of `tmux display-message`. It coexists with herdr's own Claude hook. `session_map.json` stays the single source of truth.
4. **Scope — additive.** tmux remains default and unchanged in behavior; a `CCGRAM_MULTIPLEXER` switch selects the backend.

## Source inputs and drift notes

- Inputs read:
  - Code reality: a fresh survey of tmux coupling — `tmux_manager.py` public surface, leak sites (`hook.py`, `providers/process_detection.py`, `providers/shell_infra.py`), the `session_window_key` format, and confirmation that **no multiplexer Protocol exists today** (`tmux_manager` is a singleton imported across ~48 files).
  - herdr verification: live `herdr` 0.7.0 (running on this machine) plus its source (`/tmp/herdr-src`). Confirmed: Unix-socket JSON-RPC CLI; `pane get/current/list/layout/process-info/read/run/send-text/send-keys/close`; per-pane env `HERDR_PANE_ID`/`HERDR_SOCKET_PATH`/`HERDR_ENV`; `agent_status` native; `events.subscribe` stream; durable identity is the **agent session id**, not the pane id.
  - Empirical ID-churn test (live 0.7.0): created p1/p2/p3, closed the middle pane — survivors kept their ids (no renumber/compaction), a new split got `p4` (closed number never reused), `terminal_id`s strictly monotonic and never reused. **Within a server run there is zero churn.** Only a herdr server restart reassigns ids (workspace counter resets; `terminal_id` re-minted, not persisted).
  - `.archfit.yaml` (layers `core`/`adapter`; module `tmux_adapter` = `tmux_manager`/`thread_router`/`screenshot`; rules `no-forbidden-deps`, `no-import-cycles`).
  - `.claude/rules/architecture.md`, `.claude/rules/topic-architecture.md`, `docs/architecture-design/2026-05-23-ccgram-target.md`.

- Drift notes (intent vs this design):
  - The May 2026 target design classifies "tmux integration" as **Generic / Low volatility / single `tmux_manager`** (its module map: "tmux/libtmux/subprocess behavior … `tmux_manager` methods only"). This design **deliberately diverges**: adding a second backend raises the subdomain's _implementation_ volatility from low to medium and converts a single-vendor wrapper into a provider-swap seam. Reconciliation: functional volatility stays low (multiplexing is still a solved problem); the rise is purely implementation-side, so the balancing move is a contract, consistent with how the May doc already treats the `AgentProvider` seam.
  - `.archfit.yaml` models `tmux_adapter` as one adapter module with no `subdomain`/`volatility`/public-private labels. It must be re-shaped (see Architecture-fitness checks). **archfit is configured but not wired into CI** (no `.github` reference), so its rules are advisory today, not enforced gates.
  - deepwiki/herdr docs disagree with the live binary on ID format (`1-1` vs `w2:p1`) and a `--json` flag (only `status` has it). Source/live binary is ground truth; docs are not.

## Domain and volatility map

Core = differentiating behavior, likely to change. Supporting = necessary, not differentiating. Generic = solved problem, off-the-shelf.

| Area                                 | Subdomain  | Functional volatility | Implementation volatility | Rationale                                                                                                                                                                                             |
| ------------------------------------ | ---------- | --------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Terminal multiplexing (the new seam) | Generic    | Low                   | **Medium**                | Multiplexing is solved; the change driver is swapping/adding a backend. herdr v0.7.0 is young — its socket protocol (currently `protocol: 14`), id scheme, and CLI surface may shift. tmux is stable. |
| tmux backend                         | Generic    | Low                   | Low                       | Mature; change comes from platform edge cases.                                                                                                                                                        |
| herdr backend                        | Generic    | Low                   | Medium                    | Young tool; protocol/version drift, id-on-restart reassignment, macOS feature gaps (no tty in `process-info`).                                                                                        |
| Hook identity resolution             | Supporting | Low                   | Medium                    | Each multiplexer exposes identity differently (`$TMUX_PANE` + `display-message` vs `$HERDR_PANE_ID`). Adding backends adds branches.                                                                  |
| Topic/window/session routing         | Core       | High                  | —                         | Unchanged; stays keyed by opaque `window_id`.                                                                                                                                                         |

Labels to confirm before any archfit gate consumes them: `subdomain: generic`, `volatility: low` (functional) with an implementation-volatility note, and the public/private boundary of the `multiplexer` package. These are architect-inferred and want human approval before deterministic gating.

## Module map

New package `ccgram.multiplexer`, split across both layers so the layering rule has teeth (core contract, adapter backends/wiring). It mirrors the `providers/` and `llm/` precedents.

| Module                                                   | Layer   | Responsibility                                                                                                                            | Owned knowledge                                                                                                          | Public interface                                                                                                                       | Private internals                                                                     | Change vectors                                                        |
| -------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `multiplexer/base.py`                                    | core    | Define the multiplexer contract and neutral value types. Pure — no I/O, no libtmux, no subprocess.                                        | The published multiplexer language: window/pane refs, capture results, foreground info, capabilities.                    | `Multiplexer` Protocol, `MultiplexerCapabilities`, value types `WindowRef`, `PaneInfo`, `CaptureResult`, `ForegroundInfo`, `PaneDims`. | None (types only).                                                                    | New primitive needed by a caller; a new capability flag.              |
| `multiplexer/tmux.py`                                    | adapter | tmux backend satisfying `Multiplexer`. The current `tmux_manager` body, refactored to return neutral value types.                         | libtmux/`tmux` subcommand behavior, `@id`/`%id` formats, vim-insert send quirk, pane tty.                                | The `Multiplexer` methods.                                                                                                             | libtmux objects, subprocess calls, vim-insert delay loop.                             | tmux/libtmux quirks, platform edge cases.                             |
| `multiplexer/herdr.py`                                   | adapter | herdr backend satisfying `Multiplexer`. Anti-corruption layer over the herdr socket/CLI.                                                  | herdr socket JSON-RPC, `wN:pN`/`wN:tN` ids, `pane get/run/read/layout/process-info`, `$HERDR_*` env, `protocol` version. | The `Multiplexer` methods.                                                                                                             | Socket framing, JSON-RPC request ids, `herdr` CLI shell-out, JSON→value-type mapping. | herdr protocol/version drift, id-on-restart reassignment, macOS gaps. |
| `multiplexer/registry.py`                                | adapter | name→backend factory + singleton cache (mirrors `providers/registry.py`).                                                                 | Backend construction, config-default resolution.                                                                         | `get_multiplexer(name)`.                                                                                                               | Backend instance cache.                                                               | New backend.                                                          |
| `multiplexer/__init__.py`                                | adapter | Module-level `multiplexer` proxy forwarding to the wired instance (mirrors `window_store`/`thread_router` proxies) + `get_multiplexer()`. | Proxy wiring.                                                                                                            | `multiplexer` proxy, `get_multiplexer`.                                                                                                | Proxy forwarding mechanics.                                                           | Wiring policy.                                                        |
| `identity/self_identify.py` (or a function in `hook.py`) | adapter | Resolve "which window am I?" from environment, multiplexer-neutral. Used by the hook (separate process; cannot import bot config).        | `$TMUX_PANE`+`display-message` for tmux; `$HERDR_PANE_ID`(+`$HERDR_SOCKET_PATH`) for herdr; nested-session rejection.    | `resolve_self_identity(env) -> SelfIdentity \| None`.                                                                                  | Per-backend env/CLI probing, nested-session detection.                                | New backend's identity mechanism.                                     |

Wiring: callers import the `multiplexer` proxy and **type against `multiplexer.base.Multiplexer`** (adapter→core is allowed). `bootstrap.py` selects the backend from `config.multiplexer_name` (`CCGRAM_MULTIPLEXER`, default `tmux`) and wires the proxy — the same constructor-DI pattern already used for stores. The old module-level `tmux_manager` singleton is replaced by the `multiplexer` proxy; `tmux_manager.py` becomes `multiplexer/tmux.py`.

### The Multiplexer contract

Method surface derived from the current `tmux_manager` public methods, normalized to neutral value types. Each row maps the contract method to its tmux and herdr realization.

| `Multiplexer` method                                    | tmux backend                                | herdr backend                                                                        |
| ------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------ |
| `ensure_session()`                                      | `get_or_create_session()` (libtmux)         | ensure server reachable + a working workspace                                        |
| `list_windows() -> list[WindowRef]`                     | libtmux `session.windows`                   | `herdr pane list` (+ `tab`/`workspace list`), one window ≈ one agent pane            |
| `find_window(id\|name)`                                 | filter `list_windows`                       | filter `pane list`                                                                   |
| `capture(window_id, *, ansi) -> CaptureResult`          | `capture-pane [-e] -p`                      | `pane read --source visible --format text\|ansi`                                     |
| `capture_scrollback(window_id, lines) -> CaptureResult` | `capture-pane -p -J -S -N`                  | `pane read --source recent --lines N` (**cap 1000** → capability)                    |
| `pane_dims(window_id) -> PaneDims`                      | `display-message #{pane_width/height}`      | `pane layout` → `rect{width,height}`                                                 |
| `send(window_id, text, *, enter, literal, raw)`         | `pane.send_keys` (+ vim-insert/Enter-delay) | `pane run` (atomic text+Enter) or `send-text`+`send-keys Enter`                      |
| `send_to_pane(pane_id, …)`                              | `pane.send_keys`                            | `pane run/send-text/send-keys`                                                       |
| `kill_window(window_id)`                                | `window.kill()`                             | `tab close` / `pane close`                                                           |
| `rename_window(window_id, name)`                        | `rename-window`                             | `workspace`/`tab rename`                                                             |
| `list_panes(window_id) -> list[PaneInfo]`               | `window.panes`                              | `pane list --workspace`                                                              |
| `create_window(spec) -> window_id`                      | `new_window` + `send_keys` launch           | `tab create` (or `workspace create`) + `pane run <launch>`                           |
| `set_title(window_id, provider)`                        | `select-pane -T`                            | `pane report-metadata --title`                                                       |
| `foreground(window_id) -> ForegroundInfo`               | `pane_tty` + `ps -t <tty>`                  | `pane process-info` → `foreground_processes[]` (no `ps` needed; **no tty** on macOS) |

`MultiplexerCapabilities` (immutable, gates UX and control flow — mirrors `ProviderCapabilities`):

| Capability                  | tmux          | herdr             | Consumed by                                                           |
| --------------------------- | ------------- | ----------------- | --------------------------------------------------------------------- |
| `name`                      | `"tmux"`      | `"herdr"`         | logging, doctor                                                       |
| `ids_stable_across_restart` | `True`        | `False`           | restart re-resolution (triggers session-id anchored re-map for herdr) |
| `exposes_pane_tty`          | `True`        | `False`           | `foreground()` strategy; `shell_infra`                                |
| `native_agent_status`       | `False`       | `True`            | polling may read status directly instead of pyte-scraping             |
| `read_max_lines`            | `None`        | `1000`            | scrollback callers clamp request                                      |
| `self_identify_env`         | `"TMUX_PANE"` | `"HERDR_PANE_ID"` | hook identity resolver                                                |
| `supports_event_stream`     | `False`       | `True`            | reserved for the deferred event-stream phase                          |

## Integration contracts

Balanced Coupling assessment per important edge. Strength is the goal-state after this design.

### Callers (handlers, polling, session, window-launch, live view) -> `Multiplexer` Protocol

- Strength: **contract** (target). Today it is module strength — callers import the concrete `tmux_manager`.
- Distance: **high** — a backend can be a separate process over a Unix socket; external tool; independent lifecycle.
- Volatility: implementation medium (two backends), functional low.
- Balanced: yes once the Protocol is in place. High distance + contract strength = loose coupling. The current module-strength + high-distance + rising-implementation-volatility is the one unbalanced edge this design fixes.
- Contract: `Multiplexer` Protocol + `MultiplexerCapabilities`; neutral value types only across the boundary.
- Balancing move: **lower strength** (introduce the Protocol). Do not co-locate; do not rewrite callers.
- Failure modes: a caller reaching past the Protocol into a backend (`libtmux`, raw `herdr` shell-out) — caught by the boundary audit test.

### `Multiplexer` Protocol -> herdr backend (`multiplexer/herdr.py`)

- Strength: **contract** outward; the backend privately holds herdr's socket model (anti-corruption).
- Distance: **high** — Unix-socket IPC to a separate process; young external tool.
- Volatility: implementation **medium** — herdr `protocol: 14` and id scheme may change.
- Severity (BC mapping): low-strength + high-distance + medium-volatility = loose coupling, acceptable, but the medium implementation volatility demands guards.
- Contract: herdr JSON-RPC over `$HERDR_SOCKET_PATH` (or the `herdr` CLI), pinned by capability flags and a checked `protocol` version from `herdr status`.
- Balancing move: keep all herdr JSON shapes private to `herdr.py`; surface only neutral value types; gate behavior on capabilities, not on `name == "herdr"`.
- Failure modes: socket unavailable, protocol-version mismatch, pane-id reassignment after herdr restart, missing tty on macOS, `read` truncation past 1000 lines.

### `Multiplexer` Protocol -> tmux backend (`multiplexer/tmux.py`)

- Strength: contract outward; libtmux model stays private. Distance: medium (in-process, same machine). Volatility: low. Balanced — leave as-is; this is the current healthy behavior relocated behind the Protocol.

### Hook (`hook.py`) -> neutral identity resolver -> environment/CLI

- Strength: **contract** — env var names + one query call (`display-message` / `pane get`). Distance: **high** — the hook is a separate process spawned by Claude Code; it cannot import bot config. Volatility: low (medium when adding backends).
- Balanced: yes. The resolver picks the backend by which `self_identify_env` var is present, removing the hard `$TMUX_PANE`-only assumption. The herdr branch is _simpler_ than tmux — `$HERDR_PANE_ID` is the identity directly, no `display-message` subprocess.
- Contract: `resolve_self_identity(env) -> SelfIdentity(mux, session_window_key, window_id, window_name)`.
- Failure modes: neither env var present (today's "TMUX_PANE not set" warning generalizes), nested-agent false fire, herdr socket query failure for cwd.

### Status polling -> herdr `pane get` (reuse-polling decision)

- Strength: **contract** — polling consumes the backend's neutral `PaneInfo`/`ForegroundInfo`, not raw herdr JSON. Distance: high (socket). Volatility: medium.
- Balanced: yes, with a noted accepted trade-off: this re-polls state herdr already pushes over its event stream. Accepted because functional volatility is low and a single status path is cheaper now. The `supports_event_stream` capability reserves the upgrade.
- Balancing move: route herdr status through `foreground()`/`capture()` on the Protocol; never let `polling/**` import `multiplexer.herdr`.

### Identity / `session_map` -> restart re-resolution

- Strength: **functional** — shares the session-id↔window mapping rule. Distance: medium (same process). Volatility: low.
- Balanced: yes. `resolve_stale_ids()` is extended so that when `caps.ids_stable_across_restart` is false, it re-maps persisted `session_id` (which herdr persists for native restore) to the current herdr pane, instead of matching display names. tmux path unchanged.
- Failure modes: herdr restart reassigns workspace ids; mitigated by the session-id anchor. Stale `session_map` entry if herdr did not resume the agent.

## Key flows

- **Outbound (user → agent):** topic → `window_id` (unchanged) → `multiplexer.send(window_id, text, enter=True)`. tmux: `pane.send_keys`; herdr: `pane run`.
- **Capture / live view:** `multiplexer.capture(window_id, ansi=True)` and `pane_dims(window_id)`. herdr clamps scrollback to `read_max_lines`.
- **New window:** topic creation → `multiplexer.create_window(spec)` → `window_id`. herdr: `tab create` + `pane run <launch>`; the rest of the topic-creation state machine is unchanged.
- **Hook → identity:** Claude hook → `resolve_self_identity(env)` → writes `session_map.json` + `events.jsonl` keyed by `session_window_key`. herdr branch reads `$HERDR_PANE_ID`.
- **Restart re-resolution:** startup → for each persisted window, if `caps.ids_stable_across_restart` is false, match `session_id` → live herdr pane; else current display-name match.
- **Status tick:** polling reads `multiplexer.foreground(window_id)` (+ `capture`); herdr's `native_agent_status` lets the decision kernel skip pyte scraping when present.

## Telegram topic mapping (herdr)

This consumes the seam; it is not part of the `Multiplexer` contract (which stops at opaque `window_id`). It defines how herdr's `session → workspace → tab → pane` tree projects onto Telegram's flat `group → topic` structure.

- **group = herdr session.** Forced, not chosen: bots cannot create Telegram groups via the Bot API, so the group must be a stable, pre-existing container; a herdr session is exactly that. (Named herdr sessions → separate groups when needed.)
- **topic = pane = agent.** Each herdr agent pane is one Telegram topic. Preserves ccgram's `1 topic = 1 session` invariant with no "primary pane" fudge; a tab with splits (an agent team) becomes N topics — one independent chat thread per agent, which suits an agent-native multiplexer. Rejected alternative: topic = tab (the tmux-window analog) keeps a team as one topic via the existing multi-pane code, but multiplexes two agents' message streams into one thread; choose it only if teams are predominantly used as a single unit.
- **Binding key = agent session id** (durable across herdr restart); `pane_id` is the live handle; `workspace_id`/`tab_id` are label sources only. Renaming a workspace re-labels the topic, never rebinds.
- **Adaptive topic title.** `"[status-emoji] <workspace> ▸ <agent label>"`; add `"/<tab>"` only when the tab holds more than one pane. Sources: status-emoji from herdr `agent_status` (existing topic-emoji machinery); `<workspace>` from `workspace list` label; `<agent label>` from herdr `display_agent`/`title`. The title is derived state, recomputed from `pane get` + `workspace.renamed`/`tab.renamed`/`pane.agent_status_changed` events — never a binding key.
- **cwd → workspace.** New-topic creation reuses the herdr workspace whose cwd matches the chosen directory (creating one only if absent), then adds a tab+pane inside it. This makes the workspace prefix the repo automatically and keeps herdr's per-workspace state rollup meaningful. So `create_window(spec)` on herdr resolves cwd→workspace, then `tab create` + `pane run <launch>`; `window_id` is the resulting pane.

## Module test specifications

### Multiplexer contract (shared, runs against every backend)

- **Contract tests (parametrized over backends):** one behavior suite asserts the `Multiplexer` Protocol — `create_window` → `send` → `capture` round-trips text; `list_windows`/`list_panes` shapes; `kill_window` removes the window; `pane_dims` returns positive cols/rows; `foreground` returns a pid+argv. tmux runs always; herdr marked `integration`, auto-skipped when no `$HERDR_SOCKET_PATH`/socket. This is the contract test that keeps both implementations honest.
- **Capability honesty:** assert each backend's declared capabilities match observed behavior (e.g. herdr `read_max_lines == 1000` truncates; tmux returns a tty from `foreground`, herdr does not on macOS).

### herdr backend (unit, boundary)

- Unit: parse fixed `pane get`/`process-info`/`layout` JSON fixtures into neutral value types; map `wN:pN` ↔ `window_id`; clamp scrollback request to 1000.
- Boundary: socket-down → typed error, not crash; protocol-version mismatch → refuse with a clear message; invalid/closed `window_id` → `None`/typed error; `read` past 1000 lines → `truncated=True`.

### Identity resolver (unit)

- Table-driven: env with `$TMUX_PANE` → tmux branch; env with `$HERDR_PANE_ID` → herdr branch; neither → `None` (today's warning path); nested-session env → rejected.

## Architecture-fitness checks summary

Separating **existing/enforced** from **recommended** (per the fitness methodology — documented intent is not enforcement).

Enforced today (CI runs `pytest`): the boundary-audit tests (`test_window_state_access_audit.py`, `test_query_layer_only_for_handlers.py`, `test_window_store_import_boundary.py`, `test_import_no_cycles.py`, `test_polling_types_purity.py`), `ruff`, `pyright`, `deptry`. Not in CI: `lint-lazy`, `archfit`.

New checks this design adds, cheapest enforced form first:

| #   | Check                                                                                                                                                                                                                                                                                                                                                                                                                                  | Form                                                                                                        | Enforced?                                                        |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| F1  | No module outside `multiplexer/**`, `bootstrap.py`, `main.py` imports `multiplexer.tmux`, `multiplexer.herdr`, `libtmux`, or shells out to `herdr`/`tmux` directly; callers import only `multiplexer`/`multiplexer.base`.                                                                                                                                                                                                              | New pytest audit `test_multiplexer_boundary.py` (AST walk, modeled on `test_window_state_access_audit.py`). | **Yes** — CI runs pytest.                                        |
| F2  | The `Multiplexer` contract holds for every backend.                                                                                                                                                                                                                                                                                                                                                                                    | Parametrized contract test (above).                                                                         | **Yes** (herdr leg skips without a socket).                      |
| F3  | `multiplexer.base` (core) imports no backend and no I/O lib.                                                                                                                                                                                                                                                                                                                                                                           | Extend `test_import_no_cycles.py` / a layer assertion over `multiplexer/**`.                                | **Yes**.                                                         |
| F4  | Any in-function import in `herdr.py`/`tmux.py` (lazy socket/libtmux) is annotated `# Lazy:`.                                                                                                                                                                                                                                                                                                                                           | Existing `scripts/lint_lazy_imports.py`.                                                                    | Local `make lint` only — **recommend adding `make lint` to CI**. |
| F5  | archfit layer rule: re-shape `tmux_adapter` → `multiplexer` (core: `multiplexer.base`) + `multiplexer_backends` (adapter: `multiplexer.tmux`, `multiplexer.herdr`, `registry`, `__init__`); add a `forbidden_dependency` so core `multiplexer.base` cannot import backends and `handlers`/`polling`/`session_state` cannot import `multiplexer_backends`. Add `subdomain: generic`, `volatility` note, public/private boundary labels. | `.archfit.yaml` rule/module edits.                                                                          | **Recommended/advisory** until F6.                               |
| F6  | Promote archfit from advisory to gate.                                                                                                                                                                                                                                                                                                                                                                                                 | Add `archfit check --config .archfit.yaml` step to `.github/workflows/ci.yml`.                              | Recommended — makes F5 enforced.                                 |

F1–F3 give the seam real teeth in CI on day one without depending on archfit. F5/F6 raise the floor and are the "using archfit to keep it from drifting" piece the maintainer asked for; they are recommendations, not evidence of fitness, until F6 lands.

## Design decisions and trade-offs

- **Mirror the `AgentProvider` seam.** Lowest novelty, proven in this codebase; reviewers already understand the pattern. Rejected: a bespoke abstraction or a generic "TerminalManager" framework.
- **Thin identity (reuse `window_id`).** Justified by the churn test — herdr ids are stable within a server run, so the opaque-string key still holds intra-session. Only restart needs re-resolution, which already exists for tmux. Trade-off: a herdr server restart requires session-id re-mapping; accepted, far cheaper than re-keying every state file and `callback_data`. The deep split stays available if a future need (e.g. cross-multiplexer windows) appears.
- **Reuse polling, defer the event stream.** One status path now; `supports_event_stream` reserves the upgrade. Trade-off: re-polls state herdr already pushes; accepted given low functional volatility and the cost of maintaining two status paths.
- **Keep ccgram's own hook.** Consistency with the tmux path and independence from herdr's integration being installed. Trade-off: two Claude hooks fire (ccgram + herdr); they coexist and `doctor` must account for both.
- **Backends own their model privately; gate on capabilities, not names.** Prevents herdr JSON or `name == "herdr"` conditionals leaking into handlers — the same failure mode the provider seam guards against.

## Self-review

- **Critical:** none. The single high-strength/high-distance/rising-volatility edge (callers→concrete tmux) is resolved by the Protocol.
- **Significant:**
  - _herdr protocol drift_ (medium implementation volatility). Resolved in design: capability flags + a checked `protocol` version; herdr internals quarantined in `herdr.py`; F1 keeps the quarantine enforced.
  - _Hook is a separate process and cannot import the Protocol package wiring._ Resolved: the identity resolver is a small, dependency-light function the hook can import directly; it does not need the backend instances.
  - _macOS `process-info` exposes no tty._ Resolved: `foreground()` returns `ForegroundInfo` from herdr's `foreground_processes[]` directly; `ps -t <tty>` is a tmux-only strategy gated by `exposes_pane_tty`.
- **Minor:** herdr 1000-line scrollback cap (callers clamp via `read_max_lines`); `session_window_key` prefix should encode the multiplexer/herdr session to avoid cross-backend key collisions; live view refresh latency on herdr is bounded by polling, not the event stream, until the deferred phase.
- No vague ownership, no untestable boundary (F2 covers the contract), no documented-but-unenforced intent counted as fitness.

## Open risks

- **herdr maturity (v0.7.0).** Protocol/CLI/id scheme may change between releases. Mitigation: pin/observe `protocol` version, gate on capabilities, keep the anti-corruption layer thin and well-tested.
- **herdr server restart.** Reassigns workspace ids and re-mints `terminal_id`. Mitigation: session-id-anchored re-resolution; relies on herdr having reported the agent session (its native restore path).
- **Hook coexistence.** ccgram's and herdr's Claude hooks both fire; ordering and `settings.json` patching must not clobber each other. `doctor` must verify both.
- **Deferred event stream.** Accepted latency/duplication; revisit if status responsiveness or herdr load becomes a concern.
- **archfit not yet a gate.** F5 labels are advisory until F6 wires archfit into CI; until then, F1–F3 pytest audits are the real enforcement.

## Handoff

Recommended next skill: **`architecture-plan`** to sequence implementation. Suggested ordering for the plan (each step independently verifiable):

1. **Safety gate first — extract `multiplexer/base.py` Protocol + value types and make `tmux_manager` satisfy it with zero behavior change.** Land F1–F3 audits against the tmux-only world. This is the reversible, low-risk move that makes the seam real.
2. Add `multiplexer/registry.py` + proxy + `CCGRAM_MULTIPLEXER` switch (still tmux-only).
3. Neutral identity resolver in the hook; tmux behavior unchanged.
4. Implement `multiplexer/herdr.py` + capabilities; wire the herdr contract-test leg.
5. Extend restart re-resolution for `ids_stable_across_restart == False`.
6. herdr-aware `doctor` (hook coexistence) and the `.archfit.yaml` re-shape (F5) + CI wiring (F6).

Do not begin source changes from this document; turn it into a sequenced plan first, then hand the plan to a mutator/engineer.
