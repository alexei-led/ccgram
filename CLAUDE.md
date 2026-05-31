<!-- markdownlint-disable MD025 -->

# CLAUDE.md

ccgram (Command & Control Bot) manages AI coding agents from Telegram via tmux. Each Telegram Forum topic binds to one tmux window running one agent CLI (Claude Code, Codex, Gemini, Pi, or shell).

Tech stack: Python, python-telegram-bot, tmux, uv.

## Commands

```bash
make check                    # fmt + lint + typecheck + test + integration
make fmt                      # format
make lint                     # MUST pass before commit
make typecheck                # MUST be 0 errors before commit
make test                     # unit only
make test-integration         # real tmux, fs
make test-e2e                 # real agent CLIs (~3-4 min)
make test-all                 # all except e2e
./scripts/restart.sh {start|restart|stop|status}   # local dev instance
ccgram status                 # running state (no token needed)
ccgram doctor [--fix]         # validate / auto-fix setup
ccgram hook {--install|--uninstall|--status}
ccgram --version | --help | -v
ccgram --tmux-session <name>
ccgram --autoclose-{done,dead} 0
```

Bot commands in topics: `/send`, `/toolbar`, `/history`, `/sessions`, `/restore`, `/resume`, `/panes`, `/live`, `/last`, `/sync`, `/agent` (alias `/provider`), `/upgrade`.

## Core Constraints

- 1 Topic = 1 Window = 1 Session. Routing keyed by tmux window ID (`@0`, `@12`), not window name. Window names are display labels. Same directory may have multiple windows.
- Topic-only. No `active_sessions`, no `/list`, no General topic routing, no backward-compat for non-topic mode.
- No truncation at parse layer. Splitting only at send layer (`split_message`, 4096 char limit).
- Entity-based formatting. Use `safe_reply`/`safe_edit`/`safe_send` — convert markdown to plain text + `MessageEntity` offsets via `telegramify-markdown`, auto-fallback to plain. Internal queue/UI code calls bot API directly with its own fallback.
- Hook-based session tracking. Claude Code hooks write `session_map.json` and `events.jsonl`; monitor polls both. Missing hooks logged at startup with fix command; terminal scraping is fallback.
- Per-user FIFO message queue, merging up to 3800 chars, tool_use/tool_result pairing.
- Rate limit: 0.5s min between messages per user via `rate_limit_send()`. PTB `AIORateLimiter` adds flood protection.

## Code Conventions

- Every `.py` starts with a module-level docstring (one-sentence summary first line, then responsibilities; clear within 10 lines).
- Telegram: inline keyboards > reply keyboards. Use `edit_message_text` for in-place updates. Callback data <64 bytes. `answer_callback_query` for instant feedback.
- Full variable names: `window_id` not `wid`, `thread_id` not `tid`, `session_id` not `sid`.
- All `context.user_data` keys are in `handlers/user_state.py` — import constants, never raw strings.
- Catch specific exceptions (`OSError`, `ValueError`); never bare `except Exception`.
- Tests: `tests/ccgram/` (unit, mirrors source), `tests/integration/`, `tests/e2e/`. `asyncio_mode = "auto"` — no `@pytest.mark.asyncio`. No comments or docstrings in test files.

## Logging

`structlog` everywhere (`structlog.get_logger()`); the hook subprocess uses stdlib `logging` (separate config in `hook.py`, not colored — fine, it's short-lived). Daemon level policy:

- **DEBUG**: decisions / "why" / per-iteration detail. Off in normal runs.
- **INFO**: durable lifecycle or state-change events only — never per poll/tick/callback. A genuine once-per-event transition (session changed, window deleted) is INFO; steady-state observations are not.
- **WARNING**: recoverable anomaly, kept rare. Transient races (window/pane gone mid-capture, token rejected on refresh, missing optional config) are DEBUG — flooding WARNING with races trains operators to ignore it and hides the real ones.
- **ERROR** / `logger.exception`: a unit of work failed and needs action. Recoverable fallbacks (previous menu kept, glob-scan fallback) are WARNING, not ERROR.

**Steady-state rule**: in poll/tick loops, log on _transition_, not every iteration. Reuse existing patterns — do not invent:

- `log_throttled(logger, key, msg, *args)` (`utils.py`) — first occurrence + 300s cooldown per key, debug-level. Used in `session_map.py` (preserve-primary), `transcript_reader.py` (provider mismatch / partial jsonl), `miniapp/api/terminal.py` stream loop, `polling_coordinator.py`.
- Mutation-gating — log only when state actually changes (`session_map.py` `_sync_window_from_session_map`, the provider-correction log ~line 625).
- Warn-once-then-debug counter — `ResilientPollingHTTPXRequest._should_warn_for_reset` (`telegram_request.py`): warn once per interval, debug after, reset on success. Use for a WARNING-level condition that must stay visible without flooding.
- Per-entry prune loops: DEBUG per item + one INFO summary after (`session.py`, `window_state_store.py`, `user_preferences.py`).

**Color**: `setup_logging` (`main.py`) sets `level_styles` (debug grey, info green, warning bold yellow, error bold red, critical bold bright red) and gates **both** `colors=` and `level_styles` on `_use_colors()` (`isatty()`, honoring `NO_COLOR`/`FORCE_COLOR`). `level_styles` colors the level even when `colors=False`, so both must be gated or raw ANSI leaks into redirected/piped log files. Keep the `key=value` layout — only the level token is colored.

**Never log secrets or PII**: no bot-token bytes (not even a prefix), no full allowed-user-id lists (log a count), no raw message text / chat content.

## Tmux Auto-Detection

When started inside tmux (`$TMUX` set) with no `--tmux-session` flag, attaches to current session — no creation, no `__main__` placeholder. Excludes own window. Refuses startup if another ccgram is in the same session. `--tmux-session` overrides. Outside tmux, creates `ccgram` session + `__main__` window.

## Configuration

Precedence: CLI flag > env var > `.env` (local > config dir) > default.

- Config dir: `~/.ccgram/` or `--config-dir` / `CCGRAM_DIR`.
- `TELEGRAM_BOT_TOKEN`: env-only (flags visible in `ps`).
- Multi-instance: `--group-id` / `CCGRAM_GROUP_ID` restricts to one Telegram group. `--instance-name` / `CCGRAM_INSTANCE_NAME` is a display label.
- Claude config: `--claude-config-dir` / `CLAUDE_CONFIG_DIR` overrides `~/.claude` (for Claude wrappers: `ce`, `cc-mirror`, `zai`). Used by hook install, command discovery, session monitoring.
- Directory browser: `--show-hidden-dirs` / `CCGRAM_SHOW_HIDDEN_DIRS`.
- Pane lifecycle notifications: `CCGRAM_PANE_LIFECYCLE_NOTIFY` (default `false`). Toggle per-window via `/panes`.
- Topic emoji scheme: `CCGRAM_STATUS_MODE` = `system` (default; green=working, yellow=idle) or `user` (green=ready, yellow=working). Invalid falls back to `system`.
- Tool-call visibility: `CCGRAM_HIDE_TOOL_CALLS` (default `true`) suppresses `tool_use`/`tool_result` globally. Per-window via `WindowState.tool_call_visibility` (`default`/`shown`/`hidden`) takes precedence; cycle via status-bar toggle.
- Mini App (optional): `CCGRAM_MINIAPP_BASE_URL` (HTTPS, externally reachable — Mini App disabled until set). `CCGRAM_MINIAPP_HOST` (default `127.0.0.1`), `CCGRAM_MINIAPP_PORT` (default `8765`). Binds locally; expects external TLS + reverse proxy.

State files (in config dir): `state.json` (thread bindings, window states, display names, read offsets), `session_map.json` (hook-generated), `events.jsonl` (hook events), `monitor_state.json` (byte offsets).

Project layout: handlers in `src/ccgram/handlers/`, core modules in `src/ccgram/`, optional Mini App in `src/ccgram/miniapp/`, tests mirror source in `tests/ccgram/`.

## Providers

Providers in `src/ccgram/providers/`. Per-window resolution: window's `provider_name` first, then config default (`CCGRAM_PROVIDER`, defaults to `claude`).

Launch override: `CCGRAM_<NAME>_COMMAND` (e.g. `CCGRAM_CLAUDE_COMMAND=ce --current`). Shell has no override — tmux opens `$SHELL`. Resolved by `resolve_launch_command()` in `providers/__init__.py`.

Key functions:

- `get_provider_for_window(window_id)` — resolves instance per window.
- `detect_provider_from_pane(pane_current_command, pane_tty, window_id)` — basename first, then `ps -t` (PGID cached) for JS-wrapped CLIs (node/bun).
- `detect_provider_from_command(pane_current_command)` — fast basename match (claude/codex/gemini/pi/shell).
- `set_window_provider(window_id, provider_name)` — persist choice.
- `get_provider()` — fallback for CLI commands without window context (e.g., `doctor`, `status`).

Topic creation lets users pick provider (Claude default, Codex, Gemini, Pi, Shell). External tmux windows auto-detected. Runtime re-detection (1s poll) triggers prompt-marker check on each transition to shell.

### Capabilities (gates UX per-window)

- Hook events: Claude (all event types), Codex (`SessionStart`, `Stop`), Gemini (`SessionStart`, `AfterAgent`, `SessionEnd`, `Notification`), Pi (via cc-thingz hook-runner), Shell (none).
- Resume: Claude `--resume`, Codex `resume`, Gemini `--resume idx/latest`, Pi `--session <path>`, Shell none.
- Continue: all except Shell.
- Transcript: Claude/Codex/Pi JSONL, Gemini JSONL (incremental), Shell none.
- Commands: all except Shell. Pi adds builtins + skills.
- Status detection: Claude (hook + pyte + spinner), Codex (Stop hook + activity), Gemini (AfterAgent + pane title), Pi (Stop hook + transcript), Shell (prompt idle).
- YOLO auto-accept, mode scraping, RC feedback: Claude only.
- Picker hints (modal-opening slash commands): Claude 12, Codex 5, Gemini 12, Pi 6, Shell none.

`ccgram doctor` checks managed hook installs for Claude, Codex, Gemini. Pi hooks are owned by cc-thingz hook-runner — ccgram does not modify Pi config. Codex hooks go in `~/.codex/hooks.json` + `~/.codex/config.toml`; Gemini in `~/.gemini/settings.json`. When hooks absent or fail, JSONL providers fall back to transcript-scan discovery — degraded latency, not functionality.

### Picker Hints

`ProviderCapabilities.tui_picker_commands` lists slash commands that open in-TUI modal selectors. When forwarded, the `Sent: …` reply gets a one-line hint pointing at `/toolbar`. `forward._picker_hint()` introspects the resolved `ToolbarLayout` — if `up`/`down`/`enter`/`esc` are all present, hint reads `💡 Open /toolbar to drive the picker — 🔼 🔽 Enter Esc.`; else degrades to `💡 Open /toolbar to drive the picker.` Forward lowercases the cc_name before lookup so `/MODEL` matches `model`. Drift guard: `tests/ccgram/providers/test_picker_capability_drift.py`.

### Remote Control Feedback (Claude only)

`/remote-control` is silent on outcome. Forwarding `/remote-control` or `/rc` to the agent calls `arm_rc_probe(window_id, client)` in `handlers/status/rc_probe.py` (via `commands/forward.py`). Probe captures pane ~1.5s after, re-scans every 1.5s up to 10s, classifies via pure `classify_rc_output()` regex (success-with-URL, success-without-URL, unavailable, failed, timeout) with `terminal_screen_buffer.is_rc_active(window_id)` as tiebreaker, posts one status reply. De-duped per-window via in-memory `WindowState.rc_probe_state` (never serialized).

### Shell Provider

Chat-first: text goes through LLM for NL→command by default; prefix `!` to send raw. When no LLM configured, all text forwards raw.

Prompt modes (output isolation + exit code detection):

- `wrap` (default): appends dimmed `⌘N⌘` marker after user's existing prompt, preserving Tide/Starship/Powerlevel10k.
- `replace` (legacy, `CCGRAM_PROMPT_MODE=replace`): replaces entire prompt with `{prefix}:N❯`. Marker prefix `CCGRAM_PROMPT_MARKER` (default `ccgram`) only used here.

Setup paths:

- Auto-setup: explicit shell-topic creation via directory browser — configures marker immediately, no ask.
- Ask flow: external window bind or runtime switch to shell — shows `[Set up] / [Skip]` keyboard. Skip respected for session (lazy recovery won't override). Switch away and back triggers fresh offer. Lost markers (`exec bash`, profile reload) lazily restored on next command unless Skip chosen.

Marker setup is session-scoped (PS1/PROMPT override) — never modifies shell config files.

### Pi Provider

Pi (pi.dev) is a Node.js coding agent with JSONL v3 transcripts. Hooks via cc-thingz hook-runner send `SessionStart`, `Stop`, `SessionEnd`, subagent events to `ccgram hook`. Transcripts: `~/.pi/agent/sessions/--<encoded-cwd>--/<timestamp>_<uuid>.jsonl`; canonical session id in header line `{"type":"session","id":"<uuid>","cwd":"...","version":3}`. Resume uses `--session <path>` (plain `--resume` opens interactive picker we can't drive).

Command discovery (`pi_discovery.py`): Telegram-safe builtins (`/changelog`, `/clear`, `/clone`, `/colors`, `/compact`, `/copy`, `/debug`, `/export`, `/fork`, `/hotkeys`, `/import`, `/login`, `/logout`, `/model`, `/name`, `/quit`, `/reload`, `/session`, `/settings`, `/share`, `/tree`) + skills (`.pi/skills`, `.agents/skills`, `~/.pi/agent/skills`, `~/.agents/skills`) + prompts (`.pi/prompts`, `~/.pi/agent/prompts`) + extension `pi.registerCommand(...)` scans (`.pi/extensions`, `~/.pi/agent/extensions`). `/new` and `/resume` reserved by ccgram, excluded from Pi builtins (same as Codex). Builtin list is advisory — forward path lets any `/<token>` through regardless.

## LLM Configuration

LLM is shared by two features: shell command generation (NL→command in shell topics) and completion summaries (single-line summary at agent Stop).

- `CCGRAM_LLM_PROVIDER` (empty): one of `openai`, `xai`, `deepseek`, `anthropic`, `groq`, `ollama`.
- `CCGRAM_LLM_API_KEY` (empty), `CCGRAM_LLM_BASE_URL`, `CCGRAM_LLM_MODEL` (provider-default), `CCGRAM_LLM_TEMPERATURE` (`0.1`).
- API key resolution: `CCGRAM_LLM_API_KEY` > provider env (`XAI_API_KEY` etc.) > `OPENAI_API_KEY` (universal fallback).

When `CCGRAM_LLM_PROVIDER` is unset: shell forwards raw, no completion summaries (static enriched Ready instead). Set temperature `0` for cheap/fast models.

Completion summary: on Stop hook, waits up to 3s for LLM to produce one line, then edits Ready message in-place with `Done — {summary}`. On timeout/missing config, static Ready with task checklist and last status is used.

## Live View

- `CCGRAM_LIVE_VIEW_INTERVAL` (5s), `CCGRAM_LIVE_VIEW_TIMEOUT` (300s).
- `MONITOR_POLL_INTERVAL` (1.0s), `CCGRAM_STATUS_POLL_INTERVAL` (1.0s).

Live view + poll intervals clamped to 0.5s min (live view: 1s). Auto-refreshes via `editMessageMedia`, auto-stops after timeout.

Screenshots (`/screenshot`, 📷 status-bar button) capture the current terminal viewport with ANSI colors. Live view also captures viewport only.

## /send Command

Three modes in one command:

```
/send docs/arch.png   # exact path → upload
/send *.png           # glob → pick if multiple
/send arch            # substring search → pick if multiple
/send                 # no args → interactive browser at CWD
```

Security (project-scoped, deny-by-default):

- Path containment: resolved path must stay within window CWD (blocks `../`, symlink escape).
- Hidden files/dirs (`.`-prefixed): denied.
- Secret patterns: `*.pem`, `*.key`, `*.p12`, `*credential*`, `*secret*`, `.env`, etc.
- Gitleaks: if `.gitleaks.toml` exists, path regexes from `[[rules]]` enforced.
- Gitignored: `git check-ignore -q` primary, `pathspec` fallback for non-git.
- Size limit: 50 MB (Telegram bot API cap).
- Excluded dirs: `node_modules`, `__pycache__`, `.venv`, `dist`, `build`, etc.

Tunables: `CCGRAM_SEND_SEARCH_DEPTH` (5), `CCGRAM_SEND_MAX_RESULTS` (50).

## Toolbar

`/toolbar` shows an inline keyboard from `~/.ccgram/toolbar.toml` (or `CCGRAM_TOOLBAR_CONFIG`), else built-in per-provider defaults (3×3 grid, `emoji_text` style, ≤8 cells per row). See `docs/examples/toolbar.toml`.

Default rows per provider:

- Claude: `[Screen, Ctrl-C, Live] [Mode, Think, Esc] [Up, Enter, Down] [Last, Get File, Close]`
- Codex: `[Screen, Ctrl-C, Live] [Esc, Tab, Mode] [Up, Enter, Down] [Last, Get File, Close]`
- Gemini: `[Screen, Ctrl-C, Live] [Mode, YOLO, Esc] [Up, Enter, Down] [Last, Get File, Close]`
- Pi: `[Screen, Ctrl-C, Live] [Esc, Tab, π Model] [Up, Enter, Down] [Last, Get File, Close]`
- Shell: `[Screen, Ctrl-C, Live] [Enter, ^D EOF, ^Z Susp] [Last, Get File, Esc, Close]`

Toggle actions with state readback (Mode = Shift+Tab, Think = Tab, YOLO = Ctrl+Y): capture pane ~250ms after press, scrape agent mode-line, surface in answer toast (e.g., `auto-accept edits on`). Falls back to static toast when no recognized mode-line.

Action types in TOML:

- `key`: tmux key sequence (`"Tab"`, `"C-c"`, `'\x1b[Z'`). `literal=true` for raw bytes (single-quoted TOML literal strings).
- `text`: literal text + Enter (`"/clear"`, prompt template).
- `builtin`: reserved (`screen`, `ctrlc`, `live`, `getfile`, `last`, `close`). Users cannot define new ones.

Action names ≤24 chars (callback_data budget). Providers absent from TOML keep defaults. Malformed entries logged and skipped; loader never raises. Provider resolved from `WindowState.provider_name`.

Schema:

```toml
[actions.clear]
emoji = "🧹"
text  = "Clear"
type  = "text"
payload = "/clear"

[providers.claude]
style = "emoji_text"
buttons = [
  ["screen", "ctrlc", "live"],
  ["mode",   "think", "clear"],
  ["last",   "getfile", "close"],
]
```

## Git Worktree Integration

New-topic flow inserts an opt-in worktree step between directory-confirm and provider-pick. `check_worktree_eligibility(path)` (in `handlers/topics/worktree.py`) runs four `git -C <path>` probes plus a merge/rebase fs check. Step shown only for eligible repo (in-work-tree, not bare, on named branch, no in-progress merge/rebase); else flow unchanged, no warning.

Picker: `Use current branch` or `New worktree`. New worktree suggests `ccg/<kebab(topic-title)>` or `ccg/agent-<n>` with branch+worktree collision avoidance, one-tap confirm or edit via text reply. Created at `<repo>.worktrees/<slug>` (slug = branch with `/`→`-`) via `git -C <repo> worktree add`. `WorktreeError` surfaces as one-line error with Cancel button. Dirty source repo allowed with warning.

Chosen branch + worktree path persisted on `WindowState` (`worktree_path`, `worktree_branch`) atomically with rest of topic metadata — omitted from `to_dict` when unset, `.get()`-loaded for back-compat. Reads go through `window_state_ports.worktree_state` (`get_worktree`); writes through `WindowStateStore.set_worktree` / `clear_worktree`. `SessionManager.set_window_worktree` is on the query-layer write/admin allow-list.

Edit-name is the only free-text input: `AWAITING_WORKTREE_BRANCH_NAME` in `user_data` routes the next text message to branch-name validation (`git check-ref-format --branch`) before `text_handler` forwards it. Cancel is the inline button (`/cancel` is a command and never reaches `text_handler`).

## Testing

No reliable Telegram Bot API mock server exists. Three tiers:

- Unit: `FakeTelegramClient` (or `AsyncMock`) injected via `TelegramClient` Protocol. Handlers depend on the Protocol (`src/ccgram/telegram_client.py`), not `telegram.Bot`. Tests construct `FakeTelegramClient()`, pass as `client=`, assert against `fake.calls` (list of `(method, kwargs)`), or use `fake.last_call` / `fake.call_count`. Configure per-method returns: `fake.returns[method] = value` (or `lambda **kw:`). `fake.set_side_effect(method, [v1, v2, ...])` mirrors `unittest.mock.Mock.side_effect`. Production wraps the real bot with `PTBTelegramClient(bot)` at the call site (typically `bootstrap.py` or top of a callback that has `context.bot`).
- Integration: Real PTB Application + `_do_post` patch. Register real handlers, patch `type(application.bot)._do_post` to intercept HTTP. Dispatch real `Update`/`Message` via `application.process_update()`. Exercises PTB filter eval, handler matching, Forum routing (`message_thread_id`). See `tests/integration/test_message_dispatch.py`.
- E2E: real agent CLIs + real tmux, no Telegram.

Shell provider has dedicated tests:

- `tests/ccgram/providers/test_shell.py` — capabilities, shell detection, prompt setup.
- `tests/ccgram/handlers/shell/test_shell_commands.py` — routing, LLM flow, approval keyboard, callbacks.
- `tests/ccgram/handlers/shell/test_shell_capture.py` — output extraction, passive monitoring, relay formatting, error suggestions.
- `tests/integration/test_shell_{flow,dispatch,llm_integration}.py` — full round-trips.

## Hooks

`ccgram hook --install` installs hooks for Claude Code event types:

- SessionStart (sync) — session tracking, writes `session_map.json`.
- Notification (sync) — instant interactive UI detection.
- Stop (sync) — instant done/idle detection.
- StopFailure (async) — alert on API error terminations.
- SessionEnd (async) — session lifecycle cleanup.
- SubagentStart/SubagentStop (async) — subagent status.
- TeammateIdle/TaskCompleted (async) — team notifications.

All hooks append structured events to `events.jsonl`; SessionStart also writes `session_map.json`. Session monitor reads `events.jsonl` incrementally (byte-offset). Terminal scraping remains fallback. Install/status/uninstall respects `CLAUDE_CONFIG_DIR`.

At startup, Claude hook presence is checked; warnings logged with fix command if any missing. Non-blocking.

## Spec-Driven Development

Task management via `.spec/` directory. One task per session — complete fully before starting another.

```
.spec/
├── reqs/     REQ-*.md (WHAT — requirements, success criteria)
├── epics/    EPIC-*.md (grouping)
├── tasks/    TASK-*.md (HOW — implementation steps)
├── memory/   conventions.md, decisions.md
└── SESSION.yaml
```

Commands: `/spec:work` (select+plan+implement+verify), `/spec:status`, `/spec:new`, `/spec:done`.

Quick CLI (`~/.claude/scripts/specctl`): `status`, `ready` (priority-ordered), `session show`, `validate`.

Never mark done until `make check` passes.

## Release Process

Tag format: `v` prefix (e.g. `v2.1.2`) — hatch-vcs strips it to `2.1.2`.

```bash
git cliff --tag vX.Y.Z --output CHANGELOG.md
git add CHANGELOG.md && git commit -m "docs: update CHANGELOG.md for vX.Y.Z"
git push origin main
git tag vX.Y.Z && git push origin vX.Y.Z
```

Triggers `.github/workflows/release.yml` (3 jobs):

1. publish — `uv build` + PyPI via OIDC trusted publishing.
2. update-homebrew — `scripts/generate_homebrew_formula.py` + push to `alexei-led/homebrew-tap`.
3. github-release — git-cliff release notes + GitHub Release.

CHANGELOG.md maintained locally only (CI cannot push to protected `main`).

Gotchas:

- `[skip ci]` kills tag-triggered workflows. Never tag a `[skip ci]` commit. If needed, create an empty commit (`git commit --allow-empty -m "chore: release vX.Y.Z"`) as the tag target.
- Action refs: use exact format from docs (`release/v1` vs `v1` — branch refs differ from tags).
- Scope `id-token: write` at job level for OIDC, not workflow level.
- PyPI trusted publishing: match owner/repo/workflow/environment exactly in PyPI settings.

Auto-generated `src/ccgram/_version.py` is gitignored (regenerated by hatch-vcs); excluded from linting via `pyproject.toml` `[tool.ruff] exclude`.

## Mini App Dashboard (Optional)

aiohttp web surface running alongside the bot when `CCGRAM_MINIAPP_BASE_URL` is set. Opens from `🪟 Dashboard` inline button on status bubble inside Telegram's WebApp container. Three surfaces: live xterm.js terminal (read-only), paginated transcript with full-text search, multi-pane grid view.

Subpackage `src/ccgram/miniapp/`:

- `__init__.py` — public API (`start_server`, `stop_server`, `build_app`, `sign_token`, `verify_token`, `validate_init_data`).
- `auth.py` — HMAC-signed window tokens (window_id + user_id + expiry); Telegram WebApp `initData` validation.
- `server.py` — aiohttp factory + lifecycle; routes `/healthz`, `/app/{token}`, `/static/`, sub-route registration.
- `api/terminal.py` — websocket `/ws/terminal/{token}`, pane-list `/api/panes/{token}`, per-pane multiplex `?pane=`.
- `api/transcript.py` — paginated `/api/transcript/{token}`, search `/api/transcript/{token}/search?q=...`.
- `static/` — `index.html` (SPA shell, Telegram WebApp SDK), `terminal.js` (xterm.js + delta), `transcript.js`, `panes.js`.

Lifecycle: `start_miniapp_if_enabled` / `stop_miniapp_if_enabled` (`src/ccgram/main.py`) wired into `bot.py` `post_init` / `post_shutdown`. Start failures logged and swallowed; bot keeps running. Gated entirely on `CCGRAM_MINIAPP_BASE_URL`.

Auth: tokens HMAC-signed with bot token, scoped to single (window_id, user_id), short-lived. Every API request validates token; no cross-window access.

Deployment: production needs external TLS + reverse proxy (cloudflared, caddy, nginx). Server binds locally; expects proxy to forward HTTPS. Register the Mini App via BotFather `/setdomain` and `/newapp`.

## Architecture

See @.claude/rules/architecture.md for the module inventory and module-level design decisions.
See @.claude/rules/topic-architecture.md for topic→window→session mapping.
See @.claude/rules/message-handling.md for the message queue and rate limiting.

`bot.py` is a 172-line factory + lifecycle delegate. Handler registration in `handlers/registry.py` (`register_all`). Post_init wiring in `bootstrap.py` (`bootstrap_application` + `shutdown_runtime`). Handlers depend on `TelegramClient` Protocol (`src/ccgram/telegram_client.py`); `PTBTelegramClient` adapts real PTB `Bot` in production, `FakeTelegramClient` in unit tests.

<!-- gitnexus:start -->

# GitNexus — Code Intelligence

This project is indexed by GitNexus as **ccgram** (16681 symbols, 38134 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource                                | Use for                                  |
| --------------------------------------- | ---------------------------------------- |
| `gitnexus://repo/ccgram/context`        | Codebase overview, check index freshness |
| `gitnexus://repo/ccgram/clusters`       | All functional areas                     |
| `gitnexus://repo/ccgram/processes`      | All execution flows                      |
| `gitnexus://repo/ccgram/process/{name}` | Step-by-step execution trace             |

## CLI

| Task                                         | Read this skill file                                        |
| -------------------------------------------- | ----------------------------------------------------------- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md`       |
| Blast radius / "What breaks if I change X?"  | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?"             | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md`       |
| Rename / extract / split / refactor          | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md`     |
| Tools, resources, schema reference           | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md`           |
| Index, status, clean, wiki CLI commands      | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md`             |

<!-- gitnexus:end -->
