# herdr tab-identity topics (full integration)

## Overview

ccgram's multiplexer seam already speaks herdr (status, foreground, send, capture, shell).
What is missing is the **topic model**: today the herdr backend uses thin identity
(`window_id == pane_id`, "topic = pane = agent"), so only hookless agents that ccgram
happens to discover ever surface, topic titles use the agent name (colliding when a
workspace runs two of the same agent), and there is no create/rename/delete story that
matches how a user actually drives herdr.

This plan re-bases the herdr backend on **tab identity**: one ccgram window/topic = one
herdr **tab** (`window_id == tab_id`), exactly like a tmux window. Panes inside a tab
become ccgram's existing multi-pane awareness (a split tab = an agent team = one topic
with N panes). Topics are titled `"<workspace> ▸ <tab>"`. Discovery rescans every live
tab on start and binds/creates topics for all agent tabs except those whose workspace or
tab label matches `__*__` (the self-host escape hatch, so ccgram can run itself inside a
`__main__` workspace). Create, rename, and delete sync both directions, and a new-window
flow gains a workspace step (folder → workspace → agent → tab).

The backend already carries the tab/workspace label machinery (`_adaptive_label`,
`_tab_labels`, `_workspace_labels`, `_resolve_workspace_id`, `create_window` → `tab create`);
the flip is about routing **identity** (`window_id`) and the pane ops through the tab,
fixing the consumer surfaces that still assume a tmux key namespace or `%`-style pane ids,
and adding the lifecycle/flow UX.

## Context

- Impacted components:
  - `src/ccgram/multiplexer/herdr.py` — identity flip (`_to_window_ref`/`list_windows` → tab id, dedup to one `WindowRef` per tab, `__*__` filter), tab→pane resolution for send/capture/foreground/dims/`list_panes`, `create_window` returns `tab_id`, `kill_window`→`tab close`, `rename_window`→`tab rename`.
  - `src/ccgram/multiplexer/topic_mapping.py` — `format_agent_topic_prefix` → tab name primary (`"<workspace> ▸ <tab>"`).
  - `src/ccgram/multiplexer/self_identify.py` + `src/ccgram/hook.py` — herdr hook identity resolves pane→tab via an injected probe; `session_map` key `herdr:<tab_id>`.
  - `src/ccgram/session_map.py` — already has the backend-neutral `session_map_prefix()` (line 104); consumers below must use it.
  - `src/ccgram/handlers/cleanup.py:61`, `src/ccgram/handlers/topics/topic_lifecycle.py:169`, `src/ccgram/handlers/recovery/transcript_discovery.py:166` — qualified-id strings hardcode `config.tmux_session_name`; must use `session_map_prefix()`.
  - `src/ccgram/handlers/live/pane_callbacks.py:73`, `src/ccgram/handlers/live/screenshot_callbacks.py:222`, `src/ccgram/handlers/interactive/interactive_callbacks.py:78` — pane callback-data parsing splits on `:%` / first `:`, which collides with herdr `wN:tM`+`wN:pM` ids.
  - `src/ccgram/session_monitor.py` — `_emit_unbound_window_events` + the session_map delta path (`session_lifecycle.initialize` baselines the full map at startup).
  - `src/ccgram/handlers/topics/*` — new-window workspace step (`new_command.py`, `directory_browser.py`, `directory_callbacks.py`, `topic_orchestration.py`); lifecycle (`topic_lifecycle.py`).
  - `src/ccgram/handlers/registry.py:111` — `FORUM_TOPIC_EDITED` (already wired) for the Telegram→herdr rename direction.
  - `src/ccgram/window_resolver.py` — restart re-resolution (`_resolve_by_session_id` for `ids_stable_across_restart=False`).
  - Tests: `tests/ccgram/test_herdr_backend.py`, `test_multiplexer_contract.py`, `test_topic_mapping.py`, `test_session_monitor.py`, `tests/integration/test_herdr_contract.py`.
- Constraints:
  - **tmux path unchanged** — gate every herdr-specific behavior on capabilities (`native_agent_status`, never `name == "herdr"`); F1/F2/F3 multiplexer audits stay green.
  - Checkboxes only inside Task sections (ralphex parses `- [ ]` as work items).
- herdr facts verified live (protocol 14, session "default"):
  - Hierarchy `session` (named) ▸ `workspace` (cwd+label) ▸ `tab` (cwd+label) ▸ `pane` (agent).
  - `tab list`/`workspace list` carry aggregated `agent_status`; `tab get` has **no** active-pane field → resolve tab→pane via `pane list` filtered by `tab_id`, prefer `focused`, else first.
  - `tab create` returns the new tab under `result["tab"]["tab_id"]` plus a `root_pane`.
  - herdr does **not** surface agent session-id/transcript path → transcripts stay: claude via hook, others via transcript discovery; only the `session_map` key granularity changes (pane→tab).
  - No streaming events command → status stays poll-based off `tab.agent_status`.

## Source artifact

- Design: `docs/architecture-design/2026-06-21-herdr-multiplexer-support.md`.
- **Design revision (this plan):** that doc's "topic = pane = agent / thin identity (pane_id = window_id)" is **superseded** by "topic = tab (tab_id = window_id); a split tab = one topic, multiple panes." Rationale: the user drives herdr at tab granularity (create/rename/delete a tab like a tmux window), tab titles `"<workspace> ▸ <tab>"` are unique where agent names collide, and ccgram already has multi-pane awareness for the team case. Fold back into the design doc on completion (Task 11).
- Contracts used: `Multiplexer` Protocol + neutral value types; `MultiplexerCapabilities` (`native_agent_status`, `ids_stable_across_restart`, `read_max_lines`).

## Success criteria

- On `CCGRAM_MULTIPLEXER=herdr`, every live agent **tab** (except `__*__` workspace/tab labels) surfaces and binds to a topic on startup; a tab created later surfaces within one poll; inbound/outbound delivery works end to end.
- A topic is titled `"<workspace> ▸ <tab>"`; two same-agent tabs in one workspace get distinct titles; a workspace/tab rename in herdr re-labels the bound topic on the next poll without rebinding.
- `window_id` is the **tab id** everywhere on herdr; send/capture/foreground/dims/`list_panes` resolve the tab to its active pane; a split tab shows N panes via `/panes`, and pane-targeted callbacks (screenshot/interactive prompts) work with herdr ids.
- Create, rename, delete sync both directions: `/new` creates a herdr tab; deleting/closing a topic closes the tab; closing a tab in herdr cleans up the topic; renaming the topic renames the tab and vice-versa.
- ccgram can run inside a `__main__` workspace (or a `__*__` tab) and never adopts itself.
- After a herdr **server** restart (ids re-minted) ccgram re-resolves bound topics by **agent session-id** (`_resolve_by_session_id`), not by parsing ids.
- `CCGRAM_MULTIPLEXER=tmux` suite stays green (zero behavior change); F1/F2/F3 audits pass.

## Development Approach

- Capability-gated throughout; no `name == "herdr"` conditionals outside the backend.
- **Tasks 1–3 are a coupled foundation and must land together before herdr is usable**: the identity flip (1) makes `window_id` a tab id, so the hook key (2) and the consumer key surfaces (3) must move in lockstep or herdr delivery is broken in between. Each is still independently testable, but do not pause for a manual herdr soak until 3 is done. Pane ops (capture/foreground/send/`list_panes`) are intentionally deferred to Task 4 — between Tasks 1–3 and Task 4 they still call `_pane_get(window_id)` with a tab id and will fail; rely on the fixture unit tests, not a live socket, until Task 4 lands.
- Complete each task with green verification before the next.
- The herdr backend is unit-tested with JSON fixtures (no live socket); live legs run under `-m herdr`.

## Testing Strategy

- Unit-test every code-changing task. herdr backend: feed `pane list`/`tab list`/`workspace list`/`pane get`/`tab get`/`tab create` JSON fixtures to the injectable runner; assert tab-id `window_id`, `"<workspace> ▸ <tab>"`, one `WindowRef` per tab, tab→pane resolution, `__*__` filtered.
- `test_topic_mapping.py`: table-test the label (lone/team/numeric tab) and shell-tab handling.
- `test_session_monitor.py`: known-but-unbound tab surfaces on startup; bound tab does not re-fire; `__*__` never surfaces.
- New: callback-data round-trip tests for herdr `wN:tM`+`wN:pM` ids.
- Live legs under `-m herdr` (skip without a socket): contract parity, create/rename/delete, restart re-resolution.
- Run `make test` after each task.

## Validation Commands

- `make check` — fmt, lint, typecheck, deptry, unit + integration.
- `make test` — `uv run pytest tests/ -m "not integration and not e2e" -n auto --dist=loadscope`.
- `make typecheck` — `uv run pyright src/ccgram/ tests/`.
- `make lint` — `scripts/lint_lazy_imports.py` + `uv run ruff check`.
- Multiplexer boundary/contract/purity: `uv run pytest tests/ccgram/test_multiplexer_boundary.py tests/ccgram/test_multiplexer_contract.py tests/integration/test_import_no_cycles.py -v`.
- herdr legs (need a running herdr): `uv run pytest tests/integration/ -m herdr -v`.
- Impact/blast-radius: `git diff --name-only` (GitNexus unavailable).

## Technical Details

- **Identity flip.** `WindowRef.window_id = tab_id`. `list_windows` builds one `WindowRef` **per tab** (dedup the `pane list` by `tab_id`; today it emits one per pane) from `tab list` + `workspace list` (labels) + `pane list` (per-tab agent + pane count): `window_name = format_agent_topic_prefix(workspace_label, tab_label)`, `pane_current_command =` the tab's representative agent (the focused pane's `agent`, else first non-empty), `cwd =` the tab's cwd, `agent_status` from `tab.agent_status`. `create_window` returns `result["tab"]["tab_id"]` (not `root_pane.pane_id`).
- **`__*__` skip in the adapter.** The neutral `WindowRef` does not separately carry the workspace label once `window_name` is composed, so do the `^__.*__$` filter inside `list_windows` (it holds the raw workspace + tab labels) — a `__*__` workspace (all its tabs) or `__*__` tab is simply not emitted **from `list_windows`** (the discovery/relabel source). `find_window(tab_id)` deliberately bypasses the filter (`tab get` direct), so a tab that was somehow explicitly bound still resolves for send/capture/status — the filter suppresses auto-adoption, not lookup. `is_agent_topic_window` stays the agent-presence gate (`native_agent_status` → non-empty `pane_current_command`).
- **Tab→pane resolution.** A private `_active_pane(tab_id)` (`pane list` filtered by `tab_id`, prefer `focused`, else first) backs `send`, `send_to_pane`, `capture`, `capture_scrollback`, `pane_dims`, `foreground`, `set_title`, `get_pane_title`. `list_panes(tab_id)` returns **all** panes in the tab (team awareness).
- **Hook identity.** `self_identify` herdr branch maps `$HERDR_PANE_ID` → tab id via an injected `herdr_query` (hook.py implements it: `herdr pane get <pane>` → `tab_id`; `tab get`/`workspace get` → labels). `session_window_key = f"herdr:{tab_id}"`. Resolver stays I/O-free (mirrors injected `tmux_query`).
- **Backend-neutral key surface.** `session_map.py` already centralizes the prefix in `session_map_prefix()` (tmux→session name, else backend name). The leaks are the three `qualified_id`/`window_key` strings that hardcode `config.tmux_session_name` (`cleanup.py:61`, `topic_lifecycle.py:169`, `transcript_discovery.py:166`); switch them to `session_map_prefix()`-derived keys so topic-state cleanup, lifecycle, and hookless transcript discovery match herdr's `herdr:<tab_id>` namespace.
- **Pane callback ids.** Callback data encodes `<window_id>:<pane_id>` and is parsed at **four** sites: the shared `callback_helpers.parse_target` (line 18 — used by `screenshot_callbacks.py` 4× and `status_bar_actions.py:192` for `CB_KEYS_PREFIX`), the inline first-`:` split in `screenshot_callbacks.py:222`, `pane_callbacks._parse_target` (line 66, `rfind(":%")`), and `interactive_callbacks.match_interactive_prefix` (line 65, `split(":%")`). herdr ids are colon-heavy (`w2:t1` + `w2:p1`) and have no `%`. Add a shared `CB_PANE_DELIMITER` (non-colon, e.g. `|`) in `callback_data.py`; update all four parsers + their builders to take the backend's pane id verbatim, so both tmux (`@12`+`%5`) and herdr (`w2:t1`+`w2:p1`) round-trip.
- **Discovery.** The per-poll gap is real: `session_lifecycle.initialize` baselines the entire `session_map` at startup, so a pre-existing known-but-unbound tab is never a delta (delta path never fires) and is also skipped by `_emit_unbound_window_events` (known set) — only the one-shot startup `adopt_unbound_windows` catches it. Fix order matters: with correct tab-id keys (Tasks 1–3), `audit_state` → `orphaned_window` → `adopt_unbound_windows` matches live tabs and binds them on startup. Add a bounded steady-state self-heal so a tab that fails one-shot adoption retries on later polls, without per-poll `NewWindowEvent` spam for already-bound tabs (the `bound_window_ids` skip already covers bound; `handle_new_window` is idempotent via `_is_window_already_bound`).
- **Restart re-resolution.** herdr is `ids_stable_across_restart=False`, so `resolve_stale_ids` re-resolves by **agent session-id** (`_resolve_by_session_id`), not display name. After a server restart, the hook re-registers `herdr:<new_tab_id>` → session-id; `_resolve_by_session_id` suffix-matches the persisted session id to the new tab id. Verify this path with tab-id keys (depends on Task 2); do not switch to display-name matching.
- **Status/transcripts.** Status poll-based off `tab.agent_status` (native). Transcripts unchanged: claude via hook, others via transcript discovery keyed on the tab's cwd. Shell tabs (no agent) do not auto-surface (correct under `native_agent_status`) but `/new` can still create them — the discovery filter must not block explicit creation/binding.

## Implementation Steps

### Task 1: herdr backend — tab identity + CRUD

- Justification: design revision "topic = tab". Evidence: `herdr.py:297` `_to_window_ref` sets `window_id=pane_id`; `herdr.py:342` `list_windows` emits one ref per pane; `kill_window:479`/`rename_window:486` act on the pane; `create_window:593` returns `pane_id`. Tab-label machinery (`_adaptive_label:266`, `_tab_labels:230`, `_workspace_labels:250`) already exists.
- Files: `src/ccgram/multiplexer/herdr.py`.
- Preconditions: none.
- Postconditions: `list_windows` returns one `WindowRef` per tab (window_id = tab_id, `"<workspace> ▸ <tab>"`, `tab.agent_status`, representative agent, tab cwd), `__*__` workspace/tab labels filtered; `find_window(tab_id)` works; `create_window` returns `result["tab"]["tab_id"]`; `kill_window`→`tab close`; `rename_window`→`tab rename`.
- Impact: `git diff --name-only`; F2 contract test.
- Fitness gate: `test_multiplexer_contract.py` (herdr params) green; F1 boundary unaffected.
- Verification: `uv run pytest tests/ccgram/test_herdr_backend.py tests/ccgram/test_multiplexer_contract.py -v`.
- Manual checks: against the live socket, `list_windows` yields `"archfit ▸ core"`, `"archfit ▸ cli-tune"`, `"ccgram ▸ herdr-support"`, `"ccgram ▸ ralphex"` with tab-id window ids; `__main__` absent.
- [x] flip `_to_window_ref`/`list_windows` to tab id; dedup `pane list` to one `WindowRef` per tab (representative agent, tab cwd, `tab.agent_status`)
- [x] filter `^__.*__$` workspace and tab labels inside `list_windows`
- [x] `create_window` returns `result["tab"]["tab_id"]`; `kill_window`→`tab close`; `rename_window`→`tab rename`; `find_window` via `tab get`
- [x] update fixtures + unit tests (per-tab refs, CRUD, `__*__` filter, tab-id create return)
- [x] run `make test` — must pass before next task

### Task 2: herdr hook identity → tab id; `session_map` key `herdr:<tab_id>`

- Justification: with tab identity the hook (which sees `$HERDR_PANE_ID`, a pane id) must register under the tab id or `session_map` never matches live windows — claude monitoring breaks. Evidence: `self_identify.py:76-80` builds `herdr:<pane_id>`.
- Files: `src/ccgram/multiplexer/self_identify.py` (inject `herdr_query`; key `herdr:<tab_id>`), `src/ccgram/hook.py` (implement the probe via `herdr pane get`/`tab get`).
- Preconditions: Task 1 (live windows are tab ids).
- Postconditions: a claude hook firing in a herdr pane registers `herdr:<tab_id>` matching `list_windows`; stale `herdr:<pane_id>` entries are pruned as non-live; resolver stays I/O-free (probe injected).
- Impact: `git diff --name-only`.
- Fitness gate: F3 core purity (`self_identify` imports no backend/I/O).
- Verification: `uv run pytest tests/ccgram/ -k "self_identify or hook" -v`.
- Manual checks: trigger a claude event in a herdr tab; `session_map.json` shows `herdr:<tab_id>` matching `list_windows`.
- [x] add injected `herdr_query` to `resolve_self_identity`; build `herdr:<tab_id>`
- [x] implement the pane→tab probe in `hook.py`
- [x] table-test herdr identity resolution with a fake `herdr_query`
- [x] run `make test` — must pass before next task

### Task 3: backend-neutral qualified-id surface (delivery/cleanup/discovery)

- Justification: blast radius — three consumer sites hardcode the tmux namespace, so on herdr they build keys that never match `herdr:<tab_id>`, breaking topic-state cleanup, lifecycle, and hookless transcript discovery. Evidence: `cleanup.py:61`, `topic_lifecycle.py:169`, `transcript_discovery.py:166` use `f"{config.tmux_session_name}:{window_id}"`; `session_map.py:104` already exposes `session_map_prefix()`.
- Files: `src/ccgram/handlers/cleanup.py`, `src/ccgram/handlers/topics/topic_lifecycle.py`, `src/ccgram/handlers/recovery/transcript_discovery.py` (+ audit for any other `tmux_session_name}:` qualified-id construction).
- Preconditions: Tasks 1–2.
- Postconditions: every qualified-id/window-key is built from `session_map_prefix()`; topic-state cleanup, autoclose, and transcript discovery match herdr's namespace; tmux keys byte-identical.
- Impact: `git diff --name-only`.
- Fitness gate: F1 boundary; no `name == "herdr"`.
- Verification: `grep -rn "config\.tmux_session_name}:" src/ccgram/handlers/` returns nothing (scoped to handlers — `session_map.py:116` is the canonical helper and `hook.py:685/896` are the legitimate tmux-path local-var key constructions, both intentional); `uv run pytest tests/ccgram/ -k "cleanup or lifecycle or transcript_discovery" -v`.
- Manual checks: on herdr, closing a topic clears its topic-state callbacks; a hookless agent tab is discovered by transcript discovery.
- [x] replace the three hardcoded `tmux_session_name}:` keys with `session_map_prefix()`-derived keys
- [x] grep-audit for any remaining tmux-namespace key construction
- [x] tests for the herdr key path (cleanup + transcript discovery)
- [x] run `make test` — must pass before next task

### Task 4: tab→pane resolution for pane ops + multi-pane awareness

- Justification: callers pass a tab id but herdr pane ops need a pane id; team tabs need all panes. Success criteria "send/capture/foreground/dims resolve the tab to its active pane; a split tab shows N panes".
- Files: `src/ccgram/multiplexer/herdr.py` (private `_active_pane(tab_id)`; route `send`/`send_to_pane`/`capture`/`capture_scrollback`/`pane_dims`/`foreground`/`set_title`/`get_pane_title`; `list_panes` over the tab).
- Preconditions: Tasks 1–3.
- Postconditions: pane ops resolve tab→active pane (prefer `focused`); `list_panes(tab_id)` returns every pane in the tab; the completed shell-on-herdr + foreground seam still works (it calls through `window_id` → now tab id → active pane).
- Impact: `git diff --name-only`; no-tty drift gate stays green.
- Fitness gate: `test_no_tty_outside_backend.py`, F2 contract green.
- Verification: `uv run pytest tests/ccgram/test_herdr_backend.py -k "pane or capture or foreground or send or list_panes" -v`; `uv run pytest tests/integration -m herdr -k shell -v` if socket present.
- Manual checks: `capture`/`foreground` of a tab id return the active pane; `list_panes` of a split tab returns >1 pane; a shell tab still runs end to end.
- [x] add `_active_pane(tab_id)` and route all pane ops through it
- [x] `list_panes(tab_id)` returns all panes in the tab
- [x] unit tests: single-pane and split-tab fixtures (resolution + multi-pane); shell-on-herdr regression
- [x] run `make test` — must pass before next task

### Task 5: pane callback-data encoding for herdr ids

- Justification: pane-targeted callbacks parse `<window_id>:<pane_id>` by `:%` or first-`:`, which collide with herdr `wN:tM`+`wN:pM`. Evidence: `interactive_callbacks.py:78` (`split(":%",1)`), `pane_callbacks.py:73` (`rfind(":%")`), `screenshot_callbacks.py:222` (`find(":")`).
- Files: `src/ccgram/handlers/callback_data.py` (shared `CB_PANE_DELIMITER` constant), `src/ccgram/handlers/callback_helpers.py` (shared `parse_target`, line 18), `src/ccgram/handlers/status/status_bar_actions.py` (`CB_KEYS_PREFIX` consumer at :192), `src/ccgram/handlers/live/screenshot_callbacks.py` (shared `parse_target` 4× + inline parser :222), `src/ccgram/handlers/live/pane_callbacks.py` (`_parse_target` :66), `src/ccgram/handlers/interactive/interactive_callbacks.py` (`match_interactive_prefix` :65).
- Preconditions: Task 4 (multi-pane tabs are real).
- Postconditions: callback data uses a non-colon window↔pane delimiter; all four parsers round-trip both tmux (`@12`+`%5`) and herdr (`w2:t1`+`w2:p1`) ids; the `CB_KEYS_PREFIX` path through the shared `parse_target` works; Telegram 64-byte callback-data limit respected for the longest action prefix + `wN:tM|wN:pM`.
- Impact: `git diff --name-only`.
- Fitness gate: F1 boundary.
- Verification: `uv run pytest tests/ccgram/ -k "callback or pane_callback or interactive or screenshot or status_bar" -v` with herdr-id cases.
- Manual checks: on a split herdr tab, `/panes` per-pane screenshot, a non-active-pane interactive prompt callback, and the status-bar keys (`kb:`) action all target the right pane.
- [x] add `CB_PANE_DELIMITER` (non-colon) and update all four parsers + their builders
- [x] round-trip tests for tmux and herdr ids across the four parse sites incl. the `CB_KEYS_PREFIX` path; assert the 64-byte callback-data bound
- [x] run `make test` — must pass before next task

### Task 6: topic label `"<workspace> ▸ <tab>"`

- Justification: success criteria "titled `<workspace> ▸ <tab>`; same-agent tabs distinct".
- Files: `src/ccgram/multiplexer/topic_mapping.py` (`format_agent_topic_prefix` → tab label primary), `src/ccgram/multiplexer/herdr.py` (`_adaptive_label` passes the tab label as primary).
- Preconditions: Task 1.
- Postconditions: lone and team tabs render `"<workspace> ▸ <tab>"`; numeric/auto tab labels still render usefully; `"ccgram ▸ herdr-support"` and `"ccgram ▸ ralphex"` are distinct (no `"ccgram ▸ claude"` collision). Shell tabs created via `/new` are not blocked by the agent-presence gate.
- Impact: `git diff --name-only`.
- Fitness gate: F1 boundary (topic_mapping stays pure, backend-neutral).
- Verification: `uv run pytest tests/ccgram/test_topic_mapping.py -v`.
- Manual checks: the four agent tabs get distinct titles; a `/new` shell tab binds despite no agent.
- [x] `format_agent_topic_prefix` renders workspace ▸ tab (tab name primary); update `_adaptive_label` call site
- [x] table-test labels (lone, team, numeric tab) + shell-tab note
- [x] run `make test` — must pass before next task

### Task 7: startup rescan + steady-state self-heal of unbound tabs

- Justification: observed — agent tabs already known at startup and not bound never surface via the per-poll paths (`session_lifecycle.initialize` baselines the full map; `_emit_unbound_window_events` skips known); only one-shot `adopt_unbound_windows` catches them and it missed the live claude tabs. With Tasks 1–3 the keys/ids now match; this task makes adoption reliable + self-healing.
- Files: `src/ccgram/handlers/topics/topic_orchestration.py` (`adopt_unbound_windows`, `collect_target_chats`, `handle_new_window`), `src/ccgram/session_monitor.py` (bounded steady-state re-adoption of known-but-unbound agent tabs without re-firing for bound tabs).
- Preconditions: Tasks 1–3.
- Postconditions: every live agent tab not bound gets a topic on startup; a tab that fails one-shot adoption retries on a later poll; bound tabs never re-fire `NewWindowEvent`; `__*__` tabs never adopted.
- Impact: `git diff --name-only`.
- Fitness gate: F1 boundary.
- Verification: `uv run pytest tests/ccgram/test_session_monitor.py -k "unbound or discovery or adopt" -v`.
- Manual checks: with three unbound claude tabs, all three bind on startup; restart the bot and confirm they rebind; `__main__` never binds.
- [x] confirm `audit_state`→`orphaned_window`→`adopt_unbound_windows` binds all agent tabs on startup with tab-id keys
- [x] add a bounded steady-state re-adoption for known-but-unbound agent tabs (idempotent; no re-fire for bound)
- [x] tests: known-but-unbound surfaces; bound skipped; `__*__` skipped; no per-poll spam
- [x] run `make test` — must pass before next task

### Task 8: lifecycle sync — create/delete/rename both directions

- Justification: user requirement — tabs behave like tmux windows; rename both directions. Evidence: `topic_lifecycle.py:161,214` kill via `tmux_manager.kill_window` (proxy → `tab close`); `topic_closed_handler:244` only unbinds; `registry.py:111` `FORUM_TOPIC_EDITED`.
- Files: `src/ccgram/handlers/topics/topic_lifecycle.py` (confirm the kill paths route through the proxy for herdr; decide/align `topic_closed_handler` vs delete semantics), a `FORUM_TOPIC_EDITED` → `multiplexer.rename_window` handler (Telegram→herdr), herdr→Telegram rename via the adaptive-label re-sync (Task 6).
- Preconditions: Tasks 1–6.
- Postconditions: deleting/autoclosing a topic closes the herdr tab (via `kill_window`→`tab close`); closing a tab in herdr prunes the topic (existing stale-binding cleanup); renaming the topic renames the tab; renaming the tab/workspace re-labels the topic next poll. Document whether a Telegram topic _close_ (vs delete) kills the tab, matching tmux behavior.
- Impact: `git diff --name-only`.
- Fitness gate: F1 boundary (handlers use the proxy).
- Verification: `uv run pytest tests/ccgram/ -k "lifecycle or rename or topic_edit" -v`; herdr leg under `-m herdr`.
- Manual checks: rename a topic in Telegram → `tab list` shows the new label; rename a tab in herdr → topic re-labels; delete a topic → tab gone.
- [x] confirm kill paths route through the `multiplexer` proxy for herdr; align close/delete semantics with tmux
- [x] add `FORUM_TOPIC_EDITED` → `rename_window` (works for tmux `rename-window` too)
- [x] verify herdr→Telegram rename via adaptive-label re-sync
- [x] tests for both rename directions + delete sync
- [x] run `make test` — must pass before next task

### Task 9: new-window flow — folder → workspace → agent → tab

- Justification: user requirement — a Telegram command to create a new herdr workspace/tab: select folder, then workspace (existing or new), then agent. Worktree via ccgram's `worktree.py`.
- Files: `src/ccgram/handlers/topics/new_command.py`, `directory_browser.py`, `directory_callbacks.py`, `topic_orchestration.py` (insert a workspace pick/create step gated on `native_agent_status`; tmux flow unchanged), `multiplexer/herdr.py` (`create_window` accepts an explicit workspace id from the flow; `_resolve_workspace_id` already reuses the cwd-matching workspace).
- Preconditions: Tasks 1–8.
- Postconditions: on herdr, `/new` walks folder → workspace (reuse the cwd-match or create) → agent (claude/codex/pi/gemini/shell) → `tab create` in that workspace, rooted at the optional worktree path; binds the topic. tmux `/new` unchanged.
- Impact: `git diff --name-only`.
- Fitness gate: F1 boundary; query-layer audit.
- Verification: `uv run pytest tests/ccgram/ -k "new_command or directory or topic_orchestration" -v`.
- Manual checks: `/new` on herdr creates a tab in the chosen workspace with the chosen agent; worktree path honored; shell agent creates a shell tab.
- [x] insert capability-gated workspace pick/create step into the directory flow
- [x] thread the chosen workspace id into `create_window`; keep worktree via `worktree.py`
- [x] tests for the herdr branch (workspace reuse vs create, shell agent); tmux flow regression
- [x] run `make test` — must pass before next task

### Task 10: server-restart re-resolution by agent session-id

- Justification: herdr re-mints ids on server restart (`ids_stable_across_restart=False`); bindings must survive via session-id, not id parsing. Evidence: `window_resolver._resolve_by_session_id` is the `ids_stable=False` path.
- Files: `src/ccgram/window_resolver.py` (confirm `_resolve_by_session_id` suffix-matches `herdr:<tab_id>` keys to live tab ids), startup wiring if needed.
- Preconditions: Tasks 1–9 (esp. Task 2 keys).
- Postconditions: after a herdr server restart, each bound topic re-maps to the new tab id by persisted session id; no rebinding/duplicate topics; no display-name fallback used for herdr.
- Impact: `git diff --name-only`.
- Fitness gate: existing re-resolution tests.
- Verification: `uv run pytest tests/ccgram/ -k "stale or resolve or migrat or session_id" -v`; herdr restart leg under `-m herdr`.
- Manual checks: restart herdr server; topics re-bind to new tab ids and messages still route.
- [x] verify/extend `_resolve_by_session_id` for `herdr:<tab_id>` keys
- [x] integration leg: restart herdr, assert session-id re-resolution (deferred to live `-m herdr` contract suite — Task 11)
- [x] run `make test` — must pass before next task

### Task 11: contract/integration tests + docs/architecture + memory

- Justification: lock the tab-identity contract and record the design revision.
- Files: `tests/integration/test_herdr_contract.py` (tab-identity parity, create/rename/delete, restart), `.claude/rules/architecture.md` (herdr = tab identity; topic = workspace ▸ tab; `__*__` skip; backend-neutral key surface), `docs/architecture-design/2026-06-21-herdr-multiplexer-support.md` (fold in the revision), memory `project_herdr_multiplexer.md` (already updated to topic = tab; sync details).
- Preconditions: Tasks 1–10.
- Postconditions: live-socket contract test covers the tab model end to end; docs + memory reflect tab identity.
- Impact: `git diff --name-only`.
- Fitness gate: full `make check`; F1/F2/F3 audits green.
- Verification: `make check`; `uv run pytest tests/integration/ -m herdr -v`.
- Manual checks: docs read true against behavior.
- [x] extend `test_herdr_contract.py` for tab identity + lifecycle + restart
- [x] update `.claude/rules/architecture.md` and the design doc
- [x] sync memory `project_herdr_multiplexer.md`
- [x] run `make check` — must pass to close the plan
