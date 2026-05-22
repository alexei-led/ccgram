# Screenshot Viewport Redesign + Last-Reply + Toolbar Revamp

## Overview

PR #95 made `/screenshot` (and the status-bar 📸 button, refresh, toolbar Screen) capture 500 lines of tmux scrollback and render it as one PNG. Regression:

- Claude windows (real tmux history) → ~21000px-tall, ~3.5 MB PNGs, unreadable in Telegram.
- Pi and other alternate-screen TUIs → zero scrollback history; scrollback capture returns only the viewport anyway, so the feature does nothing for them.
- The renderer (`screenshot.py`) only parses SGR color escapes; with `capture-pane -e` it leaks non-SGR escapes (cursor moves, OSC, bracketed-paste) into the rendered text.

Fix: a screenshot shows **one readable terminal viewport**. The legitimate "see the reply that scrolled off" need is _semantic_ and moves to a transcript-backed text feature (`/last`), never a giant image. Alongside, revamp the two button surfaces (status bar + `/toolbar`) to drop dead/broken actions and add the new ones.

Key benefits: readable screenshots for every provider, a proper "last reply" view, leaner codebase (notify-mode feature removed), clearer toolbar naming.

## Context (from discovery)

Files/components involved:

- **Capture**: `src/ccgram/tmux_manager.py` (`capture_pane`, `capture_pane_scrollback`), `src/ccgram/last_unit.py` (`capture_for_screenshot` to delete, `extract_last_shell_block` to keep), `src/ccgram/config.py` (`screenshot_history`).
- **Screenshot call sites**: `src/ccgram/handlers/live/screenshot_callbacks.py` (`_handle_refresh`, `_handle_status_screenshot`, `screenshot_command`), `src/ccgram/handlers/status/status_bar_actions.py` (`_do_refresh`).
- **Renderer**: `src/ccgram/screenshot.py` (`_RE_ANSI_SGR`, `text_to_image`, `_parse_ansi_line`).
- **Status bar**: `src/ccgram/handlers/status/status_bubble.py` (`build_status_keyboard`), `status_bar_actions.py` (`_handle_notify_toggle`, `_handle_remote_control`, dispatch + `@register`), `src/ccgram/handlers/callback_data.py` (`CB_STATUS_*`).
- **Toolbar**: `src/ccgram/toolbar_config.py` (`_b(...)` builtins + per-provider default grids), `src/ccgram/handlers/toolbar/toolbar_callbacks.py` (`_BUILTIN_DISPATCH`, `_builtin_send`), `docs/examples/toolbar.toml`.
- **Last-reply backend**: `src/ccgram/session_query.py` (`get_recent_messages`), `src/ccgram/window_query.py` (`get_window_provider`, `view_window`), `src/ccgram/handlers/messaging_pipeline/message_sender.py` (`safe_send`), `src/ccgram/handlers/send/send_command.py` (`upload_file` pattern for `.txt`).
- **Notify-mode (to remove) — entangled across**: `window_state_store.py` (`NOTIFICATION_MODES`, `notification_mode` field, `to_dict`/`from_dict`, `get/set/cycle_notification_mode`), `session.py` (same accessors + `WindowView` build), `window_query.py` (`get_notification_mode`), `window_view.py` (`notification_mode` field), `handlers/messaging_pipeline/message_routing.py:62-71` (errors_only content suppression), `handlers/hook_events.py:206-210` (muted/errors_only status clear), `handlers/polling/window_tick/apply.py:112,425,564-573` (typing/status gating), `handlers/polling/window_tick/observe.py:92,119` (`build_context` param), `handlers/polling/polling_types.py:106` (`TickContext.notification_mode`), `handlers/polling/window_tick/decide.py` (consumes `ctx.notification_mode`).

Related patterns:

- Handlers depend on `TelegramClient` Protocol; reads go through `window_query`/`session_query` (query-layer-only rule, AST-enforced by `tests/ccgram/test_query_layer_only_for_handlers.py`).
- Lazy-import contract enforced by `scripts/lint_lazy_imports.py` (annotate in-function imports with `# Lazy:`).
- Pure polling kernel: `decide.py`/`observe.py`/`polling_types.py` purity enforced by `tests/ccgram/handlers/polling/test_polling_types_purity.py` — changing `TickContext` ripples to `test_decide.py`, `test_tick_decision.py`, `test_window_tick.py`, `test_status_polling.py`.
- Callback registration: prefix match via `@register(...)` in `callback_registry.py`; new status button = new `CB_*` constant + handler + add to existing `@register` + `with_update` routing dict.
- Toolbar builtins: `_b(...)` def in `toolbar_config.py` + key in `_BUILTIN_DISPATCH`; builtin names are referenced by users in `toolbar.toml` grids.

Dependencies identified: `/last` backend is shared by three entry points (`/last` command, status-bar Last button, toolbar `last` builtin) — implement once, wire thrice. Status-bar final layout depends on both notify-mode removal (drops bell) and the new buttons (Last, Get File) — sequence Task 4 before Task 5.

## Development Approach

- **Testing approach**: Regular (code first, then tests) — matches the existing suite style; tests mirror source in `tests/ccgram/`, no docstrings/comments in tests.
- Complete each task fully before the next; small focused changes.
- **Every task includes new/updated tests** as separate checklist items (success + error/edge).
- **All tests pass before starting the next task.** Verify gate per task: `make check`.
- Respect: query-layer-only rule, lazy-import contract, polling-kernel purity, `TelegramClient` Protocol, catch specific exceptions, module docstring on every `.py`.

## Testing Strategy

- **Unit tests**: required every task (`tests/ccgram/`, mirrors source). `asyncio_mode = "auto"` — no `@pytest.mark.asyncio`.
- **Integration**: `tests/integration/` (real PTB + `_do_post` patch; real tmux for capture). Add a `capture_pane_scrollback` plain-capture integration check (currently untested).
- **E2E**: `tests/e2e/` touches notify-mode (`test_claude_lifecycle.py`) — update for removal. No new e2e needed.
- No reliable Telegram mock server: use `FakeTelegramClient` (`fake.calls`, `fake.last_call`, `fake.returns`).

## Progress Tracking

- Mark `[x]` immediately when done. New tasks `➕`, blockers `⚠️`. Keep this file in sync if scope shifts.

## Solution Overview

1. Unify static screenshots onto the existing viewport path (`capture_pane`) that `/live` already uses. Delete the scrollback-PNG machinery and its config knob.
2. Make the renderer strip every non-SGR escape before drawing (independent correctness fix).
3. Add a single provider-aware `send_last_reply()` backend; expose via `/last`, a 📄 status button, and a `last` toolbar builtin. AI providers read the transcript; shell reuses `extract_last_shell_block`. Overflow (>4096 chars) → `.txt` document.
4. Remove the notify-mode feature entirely (only entry point was the bell button; user finds modes valueless). All windows always forward all events after removal — simplifies routing, hooks, and the polling kernel.
5. Toolbar: rename the misleadingly-named `send` action (it downloads a file from the agent CWD to Telegram) to `getfile` (📥 "Get File"); add `last` + `getfile` to default grids.

Final status-bar row: `[⎋ Esc] [📸 Screenshot] [📄 Last] [📥 Get File]`. Remote (📡) dropped from the bar but still reachable via `/remote-control` and `/rc` commands.

## Technical Details

- `capture_pane(window_id, with_ansi=True)` → viewport text with SGR. `screenshot.text_to_image(text, with_ansi=True)` → PNG.
- `capture_pane_scrollback(window_id, history=200)` → plain text (drop `with_ansi`/`-e`). Used by `shell_capture.py` and the shell `/last` path.
- `/last` backend signature (proposed): `async def send_last_reply(client: TelegramClient, chat_id: int, thread_id: int | None, window_id: str) -> None` in new `src/ccgram/handlers/last_reply.py`.
  - AI providers (`provider.capabilities.supports_structured_transcript`): `messages, _ = await session_query.get_recent_messages(window_id)`; walk `reversed(messages)`, gather contiguous `role=="assistant" and content_type=="text"` from the last user turn onward (stop at the preceding `role=="user"`); join with `\n\n`. Fallback: most recent assistant text. Else `"No reply yet."`.
  - shell (`provider.capabilities.name == "shell"`): `text = await tmux_manager.capture_pane_scrollback(window_id, history=200)` (plain) → `extract_last_shell_block(text)`; if `None`, `"No command output found."`.
  - Output: `len(text) <= 4096` → `safe_send(...)`; else write to a temp `.txt` and `client.send_document(...)` (filename e.g. `last-reply-<window>.txt`). Reuse the upload helper/pattern from `handlers/send`.
- Notify-mode removal: drop `NOTIFICATION_MODES`, `WindowState.notification_mode`, all `*_notification_mode` accessors, `WindowView.notification_mode`, `TickContext.notification_mode`, `build_context(notification_mode=...)` param. In `message_routing.py` remove the errors_only suppression branch (always forward). In `hook_events.py` remove the muted/errors_only status-clear branch (always set status). In `apply.py` remove the `notif_mode not in (...)` gates (always send typing/status). In `decide.py` drop `ctx.notification_mode` consumption. Keep `pane_lifecycle_notify` — unrelated.

## What Goes Where

- **Implementation Steps** (checkboxes): all code, tests, and `docs/examples/toolbar.toml` updates below.
- **Post-Completion** (no checkboxes): manual Telegram smoke test on a live bot; CHANGELOG/release handled separately.

## Implementation Steps

### Task 1: Screenshot → viewport everywhere; delete scrollback-PNG path

**Files:**

- Modify: `src/ccgram/handlers/live/screenshot_callbacks.py`
- Modify: `src/ccgram/handlers/status/status_bar_actions.py`
- Modify: `src/ccgram/last_unit.py`
- Modify: `src/ccgram/config.py`
- Modify: `src/ccgram/tmux_manager.py`
- Modify: `tests/ccgram/test_last_unit.py`
- Modify: `tests/ccgram/handlers/live/test_live_view.py`

- [x] In `screenshot_callbacks.py`, switch non-pane paths of `_handle_refresh`, `_handle_status_screenshot`, `screenshot_command` from `capture_for_screenshot(...)` to `tmux_manager.capture_pane(window_id, with_ansi=True)`; remove the now-unused `capture_for_screenshot` import.
- [x] In `status_bar_actions.py` `_do_refresh`, switch the non-pane path to `capture_pane(window_id, with_ansi=True)`; remove the import.
- [x] Delete `capture_for_screenshot` from `last_unit.py` (KEEP `extract_last_shell_block` and `_strip_ansi`/`_ANSI_RE`); update module docstring.
- [x] Delete `screenshot_history` from `config.py` (`_init_live_view`) and the `CCGRAM_SCREENSHOT_HISTORY` env read; remove any doc reference in code comments.
- [x] In `tmux_manager.py` `capture_pane_scrollback`, drop the `with_ansi` param and the `-e` branch (default plain); confirm `shell_capture.py` still compiles (it calls plain).
- [x] In `test_last_unit.py`, delete the 4 `capture_for_screenshot` cases (`test_capture_*`); KEEP the 8 `extract_last_shell_block` cases.
- [x] In `test_live_view.py` (~line 801), retarget the `status_bar_actions.capture_for_screenshot` mock to `capture_pane`.
- [x] Grep repo for `capture_for_screenshot`, `screenshot_history`, `CCGRAM_SCREENSHOT_HISTORY`, `capture_pane_scrollback(.*with_ansi` → zero hits remain.
- [x] Update CLAUDE.md "Live View" section: screenshots capture viewport; remove `CCGRAM_SCREENSHOT_HISTORY` line.
- [x] Run `make check` — must pass before Task 2.

### Task 2: Renderer strips non-SGR escapes

**Files:**

- Modify: `src/ccgram/screenshot.py`
- Modify: `tests/ccgram/test_screenshot.py`

- [x] Add a pre-pass in `text_to_image` (before line parsing) that removes all non-SGR escapes: CSI sequences not ending in `m`, OSC strings (`ESC ] ... BEL`/`ST`), and two-byte ESC designators; preserve `ESC[...m` for `_RE_ANSI_SGR`. Implement as a module-level regex (e.g. `_RE_NON_SGR`) applied per input.
- [x] Ensure stripping runs for both `with_ansi=True` and `False` paths (plain text with stray escapes still cleaned).
- [x] Add test: input with cursor-move (`ESC[2A`), OSC title (`ESC]0;t BEL`), bracketed-paste (`ESC[?2004h`) mixed with SGR color → renders valid PNG and the escapes do not appear as glyphs (assert via a stripping-helper unit check if extracted, or assert PNG validity + no exception).
- [x] Add test: SGR color still applied after stripping (a colored run survives the pre-pass).
- [x] Run `make check` — must pass before Task 3.

### Task 3: `/last` backend + command

**Files:**

- Create: `src/ccgram/handlers/last_reply.py`
- Modify: `src/ccgram/handlers/registry.py`
- Create: `tests/ccgram/handlers/test_last_reply.py`

- [x] Create `handlers/last_reply.py` with `send_last_reply(client, chat_id, thread_id, window_id)` implementing the AI/shell/overflow logic in Technical Details; reads provider via `window_query.get_window_provider` + `get_provider_for_window`; AI path via `session_query.get_recent_messages`; shell path via `tmux_manager.capture_pane_scrollback(window_id, history=200)` + `extract_last_shell_block`. Module docstring required.
- [x] Add a `last_command` handler in `last_reply.py` (resolve window_id from thread binding via `window_query`, guard unbound/dead, call `send_last_reply`).
- [x] Register `/last` in `handlers/registry.py` `CommandSpec` table.
- [x] Overflow path: >4096 chars → write temp `.txt`, `client.send_document`; clean up temp file in `finally`.
- [x] Tests: AI last-turn extraction (assistant text after last user turn, joined); fallback to most-recent assistant text when last turn has none; "No reply yet." when no assistant text.
- [x] Tests: shell path returns `extract_last_shell_block` output; "No command output found." when `None`.
- [x] Tests: ≤4096 → `safe_send`/text call recorded in `fake.calls`; >4096 → `send_document` recorded; unbound/dead window guarded.
- [x] Run `make check` — must pass before Task 4.

### Task 4: Rip notify-mode feature

**Files:**

- Modify: `src/ccgram/window_state_store.py`
- Modify: `src/ccgram/session.py`
- Modify: `src/ccgram/window_query.py`
- Modify: `src/ccgram/window_view.py`
- Modify: `src/ccgram/handlers/messaging_pipeline/message_routing.py`
- Modify: `src/ccgram/handlers/hook_events.py`
- Modify: `src/ccgram/handlers/polling/polling_types.py`
- Modify: `src/ccgram/handlers/polling/window_tick/observe.py`
- Modify: `src/ccgram/handlers/polling/window_tick/decide.py`
- Modify: `src/ccgram/handlers/polling/window_tick/apply.py`
- Modify: `src/ccgram/handlers/status/status_bubble.py`
- Modify: `src/ccgram/handlers/status/status_bar_actions.py`
- Modify: `src/ccgram/handlers/callback_data.py`
- Delete: `tests/ccgram/test_session_notification_mode.py`
- Modify: notify-mode assertions in `tests/ccgram/test_window_view.py`, `test_window_query.py`, `test_window_state_store.py`, `test_state_migration.py`, `test_state_roundtrip.py` (integration), `test_schedule_save_wiring.py`, `handlers/polling/**` tests (`test_decide.py`, `test_tick_decision.py`, `test_window_tick.py`, `test_status_polling.py`), `handlers/messaging_pipeline/test_message_routing.py`, `handlers/test_hook_events.py`, `handlers/status/test_status_buttons.py`, `handlers/status/test_status_bar_actions.py`, `handlers/topics/test_topic_lifecycle.py`, `tests/e2e/test_claude_lifecycle.py`

- [x] `window_state_store.py`: remove `NOTIFICATION_MODES`, `notification_mode` field + docstring line, `to_dict`/`from_dict` handling, reset in clear (line ~323), and `get/set/cycle_notification_mode`.
- [x] `session.py`: remove `get/set/cycle_notification_mode` and the `notification_mode=` arg in `WindowView` construction (line ~567).
- [x] `window_query.py`: remove `get_notification_mode` and the `notification_mode=` field in the view projection (line ~41).
- [x] `window_view.py`: remove `notification_mode` field.
- [x] `polling_types.py`: remove `TickContext.notification_mode`.
- [x] `observe.py`: drop `notification_mode` param of `build_context` and its forwarding (lines ~92,119).
- [x] `decide.py`: remove all consumption of `ctx.notification_mode` (always-on typing/status).
- [x] `apply.py`: remove `notif_mode` lookups + the `not in ("muted","errors_only")` gates (lines ~112,425,564-573) → always send.
- [x] `message_routing.py`: remove the errors_only suppression branch (lines ~62-71) → always forward content.
- [x] `hook_events.py`: remove the muted/errors_only status-clear branch (lines ~206-210) → always set status; fix the docstring at line ~172.
- [x] `status_bubble.py`: remove the 🔔 bell button + `NOTIFY_MODE_ICONS` usage from `build_status_keyboard`.
- [x] `status_bar_actions.py`: remove `_handle_notify_toggle`, `NOTIFY_MODE_ICONS`/`NOTIFY_MODE_REACT`, `CB_STATUS_NOTIFY` from `@register` and routing dict.
- [x] `callback_data.py`: remove `CB_STATUS_NOTIFY`.
- [x] Delete `tests/ccgram/test_session_notification_mode.py`; remove notify-mode assertions/params from the test files listed above (incl. e2e).
- [x] Grep `notification_mode|NOTIFY_MODE|cycle_notification_mode|CB_STATUS_NOTIFY|errors_only` across `src/` and `tests/` → zero hits (except `pane_lifecycle_notify`, which stays).
- [x] Verify polling-kernel purity test still passes (`test_polling_types_purity.py`) and query-layer AST test passes.
- [x] Run `make check` — must pass before Task 5.

### Task 5: Status-bar final layout (drop Remote, add Last + Get File)

**Files:**

- Modify: `src/ccgram/handlers/callback_data.py`
- Modify: `src/ccgram/handlers/status/status_bubble.py`
- Modify: `src/ccgram/handlers/status/status_bar_actions.py`
- Modify: `tests/ccgram/handlers/status/test_status_buttons.py`
- Modify: `tests/ccgram/handlers/status/test_status_bar_actions.py`

- [x] Add `CB_STATUS_LAST_REPLY = "st:lr:"` and `CB_STATUS_GET_FILE = "st:gf:"` to `callback_data.py`.
- [x] In `build_status_keyboard`, set Row 1 to `[⎋ Esc] [📸] [📄 Last] [📥 Get File]`; remove the 📡 Remote button (`CB_STATUS_REMOTE`).
- [x] In `status_bar_actions.py`: add `_handle_last_reply` (calls `last_reply.send_last_reply`) and `_handle_get_file` (calls `send_command.open_file_browser` with the window CWD via `window_query.view_window`); add both constants to `@register` and `with_update` routing; remove `_handle_remote_control` registration/routing and the `CB_STATUS_REMOTE` button wiring (KEEP `rc_probe.py` and `/remote-control`/`/rc` command paths untouched).
- [x] Confirm `arm_rc_probe`/`rc_probe.py` and `/remote-control`+`/rc` commands remain registered and functional (grep + read registry).
- [x] Tests: keyboard now contains Esc/Screenshot/Last/Get-File and NOT Notify/Remote; Last button → `send_last_reply` invoked; Get-File button → file browser opened; RC commands still registered.
- [x] Run `make check` — must pass before Task 6.

### Task 6: Toolbar revamp (rename send→getfile, add last)

**Files:**

- Modify: `src/ccgram/toolbar_config.py`
- Modify: `src/ccgram/handlers/toolbar/toolbar_callbacks.py`
- Modify: `docs/examples/toolbar.toml`
- Modify: `tests/ccgram/` toolbar tests (e.g. `test_toolbar_config.py` / toolbar callback + drift tests — locate exact files)
- Modify: CLAUDE.md (Toolbar section default grids)

- [x] In `toolbar_config.py`: rename `_b("send","📤","Send","builtin","send")` → `_b("getfile","📥","Get File","builtin","getfile")`; add `_b("last","📄","Last","builtin","lastreply")`.
- [x] Update EVERY provider default grid: replace `"send"` with `"getfile"` and add `"last"` to each final row (claude/codex/gemini/pi/shell).
- [x] In `toolbar_callbacks.py`: rename `_builtin_send`→`_builtin_getfile`; change `_BUILTIN_DISPATCH` key `"send"`→`"getfile"`; add `"lastreply": _builtin_last` invoking `last_reply.send_last_reply` (resolve chat/thread from query/update).
- [x] Update `docs/examples/toolbar.toml`: rename `[actions.send]`/references → `getfile`; show `last`; refresh sample grids.
- [x] Update CLAUDE.md Toolbar "Default rows per provider" to include Last/Get File and the `builtin` reserved list (`screen`, `ctrlc`, `live`, `getfile`, `last`, `close`).
- [x] Tests: default grids resolve, reference `getfile`+`last`, no `send` builtin remains; `getfile` builtin opens file browser; `last` builtin calls `send_last_reply`; update/extend the picker/toolbar drift test.
- [x] Grep `"send"` builtin references in `src/` and `docs/examples/toolbar.toml` → only legitimate non-toolbar uses remain.
- [x] Run `make check` — must pass before Task 7.

### Task 7: Verify acceptance criteria

- [x] Screenshots (status button, `/screenshot`, refresh, toolbar Screen) produce a single viewport PNG for Claude, Pi, shell — no multi-thousand-px image.
- [x] Renderer: a viewport containing cursor-move/OSC/bracketed-paste escapes renders clean (no escape glyphs).
- [x] `/last`, 📄 status button, and toolbar `last` all return the last assistant reply (AI) or last command+output (shell); >4096 chars → `.txt`.
- [x] Status bar = `[Esc] [Screenshot] [Last] [Get File]`; no bell, no remote; `/remote-control` + `/rc` still work.
- [x] Toolbar grids show Get File + Last; no `send` builtin; `toolbar.toml` example valid.
- [x] No notify-mode code or tests remain (grep clean).
- [x] Run full suite: `make check` (fmt + lint + lint-lazy + typecheck 0 errors + test + integration).
- [x] Run `make test-e2e` (skipped — no live agent in this run; notify-mode e2e path removed).

### Task 8: Documentation + plan close-out

- [x] CLAUDE.md: confirm Live View, Toolbar, and any notify/status-bubble references reflect the new state; remove notify-mode mentions if present.
- [x] `.claude/rules/architecture.md`: update status-bar/handlers descriptions if they named notify or remote button behavior.
- [x] Update CHANGELOG via the release process when releasing (deferred to release — not here).
- [x] Move this plan to `docs/plans/completed/`.

## Post-Completion

**Manual verification:**

- Live bot smoke test in a Telegram forum: screenshot a long Claude reply (readable viewport), screenshot a Pi window, `/last` on Claude + on shell, send an oversized reply to confirm `.txt`, tap Get File and download a project file, confirm `/remote-control` still works without the status button.

**External system updates:**

- Any user `~/.ccgram/toolbar.toml` referencing the `send` builtin must rename it to `getfile` (config-breaking by decision). Note in release notes.
