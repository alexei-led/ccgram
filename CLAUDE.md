# CLAUDE.md

ccgram (Command & Control Bot) — manage AI coding agents from Telegram via tmux. Each Telegram Forum topic is bound to one tmux window running one agent CLI instance (Claude Code, Codex, Gemini, Pi, or a plain shell).

Tech stack: Python, python-telegram-bot, tmux, uv.

## Common Commands

```bash
make check                            # Run all: fmt, lint, typecheck, test, integration
make fmt                              # Format code
make lint                             # Lint — MUST pass before committing
make typecheck                        # Type check — MUST be 0 errors before committing
make test                             # Unit tests (excludes integration and e2e)
make test-integration                 # Integration tests (real tmux, filesystem)
make test-e2e                         # E2E tests (real agent CLIs, ~3-4 min)
make test-all                         # All tests except e2e
./scripts/restart.sh start            # Start local dev instance in tmux ccgram:__main__ (auto-installs Claude hooks)
./scripts/restart.sh restart          # Restart local dev instance (Ctrl-C in control pane)
./scripts/restart.sh stop             # Stop local dev instance (Ctrl-\ in control pane)
./scripts/restart.sh status           # Show control pane status and logs
ccgram status                          # Show running state (no token needed)
ccgram doctor                          # Validate setup and diagnose issues
ccgram doctor --fix                    # Auto-fix issues (install hook, kill orphans)
ccgram hook --install                  # Auto-install Claude Code hooks (all supported event types)
ccgram hook --uninstall                # Remove hook from ~/.claude/settings.json
ccgram hook --status                   # Check if hook is installed
ccgram --version                       # Show version
ccgram --help                          # Show all available flags
ccgram -v                              # Run bot with verbose (DEBUG) logging
ccgram --tmux-session my-session       # Run with flag overrides
ccgram --autoclose-done 0              # Disable auto-close for done topics
ccgram --autoclose-dead 0              # Disable auto-close for dead sessions
```

Bot commands (in Telegram topics):

```
/send [pattern]   Send workspace file to Telegram (exact path, glob, or browse)
/toolbar          Show provider-specific inline action toolbar
/history          Browse paginated message history
/sessions         Active sessions dashboard
/restore          Recover a dead topic
/resume           Scan past sessions and pick one to resume
/panes            List panes with per-pane screenshot buttons
/live             Start auto-refreshing terminal screenshot view
/sync             Sync window state with tmux
/upgrade          Upgrade ccgram via uv and restart
```

## Core Design Constraints

- **1 Topic = 1 Window = 1 Session** — all internal routing keyed by tmux window ID (`@0`, `@12`), not window name. Window names kept as display names. Same directory can have multiple windows.
- **Topic-only** — no backward-compat for non-topic mode. No `active_sessions`, no `/list`, no General topic routing.
- **No message truncation** at parse layer — splitting only at send layer (`split_message`, 4096 char limit).
- **Entity-based formatting** — use `safe_reply`/`safe_edit`/`safe_send` helpers which convert markdown to plain text + MessageEntity offsets (no parse errors possible, auto fallback to plain text). Internal queue/UI code calls bot API directly with its own fallback.
- **Hook-based session tracking** — Claude Code hooks (SessionStart, Notification, Stop, StopFailure, SessionEnd, SubagentStart, SubagentStop, TeammateIdle, TaskCompleted) write to `session_map.json` and `events.jsonl`; monitor polls both to detect session changes, refresh Claude task lists in Telegram, and deliver instant event notifications. Missing hooks are detected at startup with an actionable warning.
- **Shell provider chat-first design** — text sent to a shell topic goes through the LLM for NL→command generation by default; prefix with `!` to send a raw command directly. When no LLM is configured, all text is forwarded as raw commands. Two prompt modes for output isolation and exit code detection: **wrap** (default) appends a small `⌘N⌘` marker after the user's existing prompt, preserving Tide/Starship/Powerlevel10k/etc.; **replace** replaces the entire prompt with `{prefix}:N❯` (legacy, opt-in via `CCGRAM_PROMPT_MODE=replace`). Two setup paths: **Auto-setup** (explicit shell topic creation via directory browser) configures the marker immediately without asking. **Ask flow** (external window bind or runtime provider switch to shell) shows an inline keyboard [Set up] / [Skip]; Skip is respected for the session (lazy recovery won't override). On provider switch away from shell and back, a fresh offer is shown. If marker is lost mid-session (`exec bash`, profile reload), it is lazily restored on the next command send (unless user chose Skip). Marker setup is session-scoped (PS1/PROMPT override) — never modifies shell config files.
- **Message queue per user** — FIFO ordering, message merging (3800 char limit), tool_use/tool_result pairing.
- **Rate limiting** — 0.5s minimum interval between messages per user via `rate_limit_send()`. PTB's AIORateLimiter provides additional flood protection.

## Code Conventions

- Every `.py` file starts with a module-level docstring: purpose clear within 10 lines, one-sentence summary first line, then core responsibilities and key components.
- Telegram interaction: prefer inline keyboards over reply keyboards; use `edit_message_text` for in-place updates; keep callback data under 64 bytes; use `answer_callback_query` for instant feedback.
- Full variable names: `window_id` not `wid`, `thread_id` not `tid`, `session_id` not `sid`.
- User-data keys: all `context.user_data` string keys are defined in `handlers/user_state.py` — import from there, never use raw strings.
- Specific exceptions: catch specific exception types (`OSError`, `ValueError`, etc.), never bare `except Exception`.

## Tmux Session Auto-Detection

When ccgram starts inside an existing tmux session (i.e. `$TMUX` is set) and no explicit `--tmux-session` flag is given, it auto-detects the current session and attaches to it — no session creation, no `__main__` placeholder window. The bot also detects and excludes its own tmux window from the window list. If another ccgram instance is already running in the same session, startup is refused with an error.

- `--tmux-session` flag overrides auto-detection (backward compatible).
- Outside tmux, behavior is unchanged (creates `ccgram` session + `__main__` window).

## Configuration

- **Precedence**: CLI flag > env var > `.env` file > default.
- Config directory: `~/.ccgram/` by default, override with `--config-dir` flag or `CCGRAM_DIR` env var.
- `.env` loading priority: local `.env` > config dir `.env`.
- All config values accept both CLI flags and env vars (see `ccgram --help`). `TELEGRAM_BOT_TOKEN` is env-only (security: flags visible in `ps`).
- Multi-instance: `--group-id` / `CCGRAM_GROUP_ID` restricts to one Telegram group. `--instance-name` / `CCGRAM_INSTANCE_NAME` is a display label.
- Claude config: `--claude-config-dir` / `CLAUDE_CONFIG_DIR` overrides `~/.claude` (for Claude wrappers like `ce`, `cc-mirror`, `zai`). Used by hook install, command discovery, and session monitoring.
- Directory browser: `--show-hidden-dirs` / `CCGRAM_SHOW_HIDDEN_DIRS` shows dot-directories in the browser.
- State files: `state.json` (thread bindings), `session_map.json` (hook-generated), `events.jsonl` (hook events), `monitor_state.json` (byte offsets).
- Project structure: handlers in `src/ccgram/handlers/`, core modules in `src/ccgram/`, optional Mini App backend in `src/ccgram/miniapp/`, tests mirror source under `tests/ccgram/`.
- Pane lifecycle: `CCGRAM_PANE_LIFECYCLE_NOTIFY` (default `false`) sets the per-window default for pane create/close notifications; toggle per-window via the `/panes` keyboard.
- Topic emoji color scheme: `CCGRAM_STATUS_MODE` (`system` default or `user`) controls which color maps to active vs idle. `system`: green=working, yellow=idle (default, system POV). `user`: green=idle/ready, yellow=working (user POV: green=ready for me). Invalid values fall back to `system`.
- Tool-call visibility: `CCGRAM_HIDE_TOOL_CALLS` (default `true`) globally suppresses `tool_use`/`tool_result` messages in Telegram. Per-window override via `WindowState.tool_call_visibility` (`default`/`shown`/`hidden`) takes precedence; cycle via the status bar toggle.
- Mini App (optional, v3.0+): `CCGRAM_MINIAPP_BASE_URL` (externally reachable HTTPS URL — Mini App is fully disabled until set), `CCGRAM_MINIAPP_HOST` (default `127.0.0.1`), `CCGRAM_MINIAPP_PORT` (default `8765`). Server binds locally; expects external TLS termination + reverse proxy.

## Provider Configuration

ccgram supports multiple agent CLI backends via the provider abstraction (`src/ccgram/providers/`). Providers are resolved per-window — different topics can use different providers simultaneously.

| Setting              | Env Var                 | Default         |
| -------------------- | ----------------------- | --------------- |
| Default provider     | `CCGRAM_PROVIDER`       | `claude`        |
| Per-provider command | `CCGRAM_<NAME>_COMMAND` | (from provider) |

Launch command override: `CCGRAM_<NAME>_COMMAND` (e.g. `CCGRAM_CLAUDE_COMMAND=ce --current`, `CCGRAM_PI_COMMAND=pi --model sonnet`), falls back to provider default. The shell provider has no override — tmux opens `$SHELL` by default. Resolved by `resolve_launch_command()` in `providers/__init__.py`.

### Per-Window Provider Model

Each tmux window tracks its own provider in `WindowState.provider_name`. Resolution order:

1. Window's stored `provider_name` (set during topic creation or auto-detected)
2. Config default (`CCGRAM_PROVIDER` env var, defaults to `claude`)

Key functions:

- `get_provider_for_window(window_id)` — resolves provider instance for a specific window
- `detect_provider_from_pane(pane_current_command, pane_tty, window_id)` — auto-detects provider from process name with ps-based TTY fallback for JS-runtime-wrapped CLIs
- `detect_provider_from_command(pane_current_command)` — fast-path detection from process basename (claude/codex/gemini/pi/shell)
- `set_window_provider(window_id, provider_name)` — persists provider choice on SessionManager

When creating a topic via the directory browser, users can choose the provider (Claude default, Codex, Gemini, Pi, Shell). Externally created tmux windows are auto-detected via `detect_provider_from_pane()` which tries process basename first, then falls back to `ps -t` foreground process inspection (with PGID caching) when the pane command is a JS runtime wrapper (node/bun). The global `get_provider()` remains as fallback for CLI commands without window context (e.g., `doctor`, `status`). Runtime re-detection (every 1s poll cycle) triggers prompt marker check on each transition to shell. Explicit shell topic creation (directory browser) auto-configures the marker.

### Provider Capability Matrix

| Capability       | Claude                              | Codex                          | Gemini                                                           | Pi                              | Shell                       |
| ---------------- | ----------------------------------- | ------------------------------ | ---------------------------------------------------------------- | ------------------------------- | --------------------------- |
| Hook events      | Yes (all supported event types)     | Yes (`SessionStart`, `Stop`)   | Yes (`SessionStart`, `AfterAgent`, `SessionEnd`, `Notification`) | Yes via hook-runner             | No                          |
| Resume           | Yes (`--resume`)                    | Yes (`resume`)                 | Yes (`--resume idx/latest`)                                      | Yes (`--session <path>`)        | No                          |
| Continue         | Yes                                 | Yes                            | Yes                                                              | Yes                             | No                          |
| Transcript       | JSONL                               | JSONL                          | JSONL (incremental)                                              | JSONL (v3)                      | None                        |
| Incremental read | Yes                                 | Yes                            | Yes                                                              | Yes                             | No                          |
| Commands         | Yes                                 | Yes                            | Yes                                                              | Yes (builtins + skills)         | No                          |
| Status detection | Hook events + pyte + spinner        | Stop hook + activity heuristic | AfterAgent hook + pane title                                     | Stop hook + transcript activity | Shell prompt idle detection |
| YOLO auto-accept | Yes                                 | No                             | No                                                               | No                              | No                          |
| Mode scraping    | Yes (mode-line parse)               | No                             | No                                                               | No                              | No                          |
| RC feedback      | Yes (probe after `/remote-control`) | No                             | No                                                               | No                              | No                          |
| Picker hints     | Yes (12 commands)                   | Yes (5 commands)               | Yes (12 commands)                                                | Yes (6 commands)                | No                          |

Capabilities gate UX per-window: recovery keyboard only shows Continue/Resume buttons when supported; `ccgram doctor` checks managed hook installs for Claude, Codex, and Gemini. Pi hook support is supplied by cc-thingz hook-runner; transcript/process detection remains fallback for all non-shell agents.

**Picker hints** (`ProviderCapabilities.tui_picker_commands`): each provider declares the slash commands that open a modal in-TUI selector (e.g. `/model`, `/login`, `/theme`). When such a command is forwarded, the topic's `Sent: …` reply gets a one-line hint pointing at `/toolbar`. `forward._picker_hint()` introspects the resolved `ToolbarLayout` for the window's provider — if `up`/`down`/`enter`/`esc` are all present, the hint reads `💡 Open /toolbar to drive the picker — 🔼 🔽 Enter Esc.`; if a user's custom `~/.ccgram/toolbar.toml` drops any of those nav keys, it degrades to `💡 Open /toolbar to drive the picker.` (no promise of buttons that don't exist). The picker set is per-provider — Claude declares 12 (sourced from `code.claude.com/docs/en/commands`), Codex 5 (from `tui/tooltips.txt`), Gemini 12 (from `geminicli.com/docs/reference/commands`), Pi 6 (verified against the live binary). Shell has none. Forward lower-cases the resolved cc_name before the picker lookup so `/MODEL` matches `model`. The drift guard (`tests/ccgram/providers/test_picker_capability_drift.py`) enumerates providers via the registry — a new provider can't silently skip the subset invariant.

### Remote Control Feedback

Claude's `/remote-control` is silent on outcome — no signal on success, "feature unavailable", or failure. Both trigger paths (status-bubble RC button, forwarded `/remote-control` or `/rc` slash) call `arm_rc_probe(window_id, client)` in `handlers/status/rc_probe.py`. The probe captures the pane ~1.5s after RC fires and re-scans every 1.5s up to 10s, classifying via the pure `classify_rc_output()` regex (success-with-URL, success-without-URL, unavailable, failed, timeout) with `terminal_screen_buffer.is_rc_active(window_id)` as a tiebreaker, then posts one status reply in the topic. De-duped per-window via the in-memory `WindowState.rc_probe_state` field (double-tap is a no-op; never serialized — safe to drop on restart). Capability-gated to Claude; other providers keep their existing "not supported by &lt;provider&gt;" reply.

### Shell Prompt Configuration

| Setting       | Env Var                | Default  |
| ------------- | ---------------------- | -------- |
| Prompt mode   | `CCGRAM_PROMPT_MODE`   | `wrap`   |
| Marker prefix | `CCGRAM_PROMPT_MARKER` | `ccgram` |

Prompt mode controls how the shell prompt marker is injected: **wrap** (default) appends a dimmed `⌘N⌘` marker after the user's existing prompt, preserving custom prompts (Tide, Starship, Powerlevel10k); **replace** replaces the entire prompt with `{prefix}:N❯` (legacy). Marker prefix is only used in `replace` mode.

### LLM Configuration

The LLM is used for two features: (1) **shell command generation** — translates natural language to shell commands in shell topics, and (2) **completion summaries** — produces a single-line summary when an agent finishes. LLM settings are shared across both features.

| Setting         | Env Var                  | Default         |
| --------------- | ------------------------ | --------------- |
| LLM provider    | `CCGRAM_LLM_PROVIDER`    | (empty)         |
| LLM API key     | `CCGRAM_LLM_API_KEY`     | (empty)         |
| LLM base URL    | `CCGRAM_LLM_BASE_URL`    | (from provider) |
| LLM model       | `CCGRAM_LLM_MODEL`       | (from provider) |
| LLM temperature | `CCGRAM_LLM_TEMPERATURE` | `0.1`           |

Supported LLM providers: `openai`, `xai`, `deepseek`, `anthropic`, `groq`, `ollama`. API key resolution: `CCGRAM_LLM_API_KEY` > provider-specific env var (e.g. `XAI_API_KEY`) > `OPENAI_API_KEY` (universal fallback). When `CCGRAM_LLM_PROVIDER` is unset, the shell provider skips NL→command generation and forwards all input as raw commands. Set temperature to `0` for deterministic output with cheap/fast models.

The LLM is also used for **completion summaries**: when an agent finishes (Stop hook), ccgram waits up to 3s for the LLM to produce a single-line summary, then sends a single "Done — {summary}" status message. When no LLM is configured or the LLM times out, the static enriched Ready (with task checklist and last status) is shown instead.

### Live View Configuration

| Setting           | Env Var                       | Default   |
| ----------------- | ----------------------------- | --------- |
| Refresh interval  | `CCGRAM_LIVE_VIEW_INTERVAL`   | `5` (s)   |
| Auto-stop timeout | `CCGRAM_LIVE_VIEW_TIMEOUT`    | `300` (s) |
| Monitor poll      | `MONITOR_POLL_INTERVAL`       | `1.0` (s) |
| Status poll       | `CCGRAM_STATUS_POLL_INTERVAL` | `1.0` (s) |
| Screenshot lines  | `CCGRAM_SCREENSHOT_HISTORY`   | `500`     |

Live view and poll intervals are clamped to a minimum of 0.5s (live view: 1s). Live view auto-refreshes terminal screenshots via `editMessageMedia` at the configured interval, and auto-stops after the timeout.

Screenshots (`/screenshot`, 📷 status-bar button) capture scrollback (default 500 lines) with ANSI color rather than just the visible viewport. For shell topics, the last command and its output are extracted from between prompt markers when available, falling back to full scrollback. Live view keeps viewport capture unchanged.

### /send Command — File Delivery

Send workspace files to Telegram. Three modes in one command:

```
/send docs/arch.png   # Exact path → immediate upload
/send *.png           # Glob → find matches, pick if multiple
/send arch            # Substring → search, pick if multiple
/send                 # No args → interactive file browser at CWD
```

**Security model** — project-scoped, deny-by-default:

- Path containment: resolved path must stay within window CWD (blocks `../` traversal, symlink escape)
- Hidden files/dirs: anything starting with `.` is denied
- Secret patterns: `*.pem`, `*.key`, `*.p12`, `*credential*`, `*secret*`, `.env` etc.
- Gitleaks: if `.gitleaks.toml` exists, path regexes from `[[rules]]` are enforced
- Gitignored: `git check-ignore -q` primary, `pathspec` library fallback for non-git repos
- Size limit: 50 MB (Telegram bot API cap)
- Excluded dirs: `node_modules`, `__pycache__`, `.venv`, `dist`, `build`, etc. — never shown in browser or search

| Setting      | Env Var                    | Default |
| ------------ | -------------------------- | ------- |
| Search depth | `CCGRAM_SEND_SEARCH_DEPTH` | `5`     |
| Max results  | `CCGRAM_SEND_MAX_RESULTS`  | `50`    |

### Toolbar — Configurable Per-Provider

`/toolbar` shows an inline keyboard whose layout is loaded from a TOML file (or built-in defaults). Each provider has a grid of buttons (any rows × cols, ≤8 cells per row) and a rendering style (`emoji`, `text`, or `emoji_text`).

**Default**: 3×3 grid per provider, `emoji_text` style:

| Provider | Row 1                        | Row 2                    | Row 3                                     |
| -------- | ---------------------------- | ------------------------ | ----------------------------------------- |
| Claude   | 📷 Screen, ⏹ Ctrl-C, 📺 Live | 🔀 Mode, 💭 Think, ⎋ Esc | 📤 Send, ⏎ Enter, ✖ Close                 |
| Codex    | 📷 Screen, ⏹ Ctrl-C, 📺 Live | ⎋ Esc, ⏎ Enter, ⇥ Tab    | 📤 Send, 🔀 Mode, ✖ Close                 |
| Gemini   | 📷 Screen, ⏹ Ctrl-C, 📺 Live | 🔀 Mode, 🅨 YOLO, ⎋ Esc   | 📤 Send, ⏎ Enter, ✖ Close                 |
| Pi       | 📷 Screen, ⏹ Ctrl-C, 📺 Live | ⎋ Esc, ⇥ Tab, π Model    | 🔼 Up, ⏎ Enter, 🔽 Down, 📤 Send, ✖ Close |
| Shell    | 📷 Screen, ⏹ Ctrl-C, 📺 Live | ⏎ Enter, ^D EOF, ^Z Susp | 📤 Send, ⎋ Esc, ✖ Close                   |

**Toggle actions with state readback**: Mode (Shift+Tab), Think (Tab), YOLO (Ctrl+Y) capture the pane ~250ms after the key press, scrape the agent CLI's mode-line, and surface it in the answer toast (e.g., "auto-accept edits on"). Falls back to the static toast when no recognized mode-line is found.

**Action types** users can define in TOML:

- **`key`** — send a tmux key sequence (e.g. `"Tab"`, `"C-c"`, `'\x1b[Z'`). Set `literal=true` for raw byte sequences (TOML literal strings — single-quoted).
- **`text`** — send literal text + Enter (e.g. `"/clear"`, prompt template). Useful for slash commands the agent itself interprets.
- **`builtin`** — reserved; users cannot define new builtins. Existing builtins: `screen`, `ctrlc`, `live`, `send`, `close`.

**Configuration**: place a TOML file at `~/.ccgram/toolbar.toml` (auto-detected) or set `CCGRAM_TOOLBAR_CONFIG=/path/to/toolbar.toml`. See `docs/examples/toolbar.toml` for a fully-annotated example. Schema:

```toml
[actions.clear]                    # define a custom action
emoji = "🧹"
text  = "Clear"
type  = "text"
payload = "/clear"

[providers.claude]                 # override claude's default grid
style = "emoji_text"
buttons = [
  ["screen", "ctrlc", "live"],
  ["mode",   "think", "clear"],
  ["send",   "enter", "close"],
]
```

Providers absent from the TOML keep their built-in defaults. Malformed entries are logged and skipped — the loader never raises. Action names must be ≤24 chars (callback_data budget). Provider is resolved from `WindowState.provider_name`.

### Migration Notes

Existing Claude deployments need no changes — `claude` is the default provider. Windows without an explicit `provider_name` fall back to the config default. The hook subsystem (`ccgram hook --install`) defaults to Claude; pass `--provider {codex,gemini,pi}` to manage other providers. Codex and Gemini hook installs are ccgram-managed (settings written into `~/.codex/hooks.json` + `~/.codex/config.toml` and `~/.gemini/settings.json` respectively). Pi hooks are owned by the cc-thingz hook-runner extension — ccgram receives Pi events but does not modify Pi configuration. Shell windows have no hooks. When hooks are absent or fail, providers with JSONL transcripts (Codex/Gemini/Pi) fall back to transcript-scan discovery, so missing hooks degrade latency rather than functionality.

### Pi Provider

[Pi](https://pi.dev) is a Node.js-based coding agent CLI with JSONL v3 transcripts and hook support via cc-thingz hook-runner. The extension sends `SessionStart`, `Stop`, `SessionEnd`, and subagent events to `ccgram hook`; transcript discovery remains fallback and message source of truth. Transcripts live under `~/.pi/agent/sessions/--<encoded-cwd>--/<timestamp>_<uuid>.jsonl`; the canonical session id sits in the header line (`{"type":"session","id":"<uuid>","cwd":"...","version":3}`). Resume always uses `--session <path>` — `--resume` would open an interactive picker ccgram can't drive. Command discovery (`pi_discovery.py`) surfaces Telegram-friendly builtins (`/changelog`, `/clear`, `/clone`, `/colors`, `/compact`, `/copy`, `/debug`, `/export`, `/fork`, `/hotkeys`, `/import`, `/login`, `/logout`, `/model`, `/name`, `/quit`, `/reload`, `/session`, `/settings`, `/share`, `/tree`) plus on-disk sources: skills under `.pi/skills`, `.agents/skills`, `~/.pi/agent/skills`, and `~/.agents/skills`; prompt templates under `.pi/prompts` and `~/.pi/agent/prompts`; extension commands via `pi.registerCommand(...)` scans in `.pi/extensions` and `~/.pi/agent/extensions`. `/new` and `/resume` are reserved by ccgram (bot-native handlers) and excluded from the Pi builtins list, same as for Codex. The builtin list is advisory — it drives `/commands` listings and menu autocomplete, but the forward path lets any `/<token>` through to the provider regardless of whether it appears in the discovery set.

## Git Worktree Integration

The new-topic flow inserts an opt-in worktree step between directory-confirm and provider-pick. After the directory is confirmed, `check_worktree_eligibility(path)` (in `handlers/topics/worktree.py`) runs four `git -C <path>` probes plus a merge/rebase filesystem check. The step is shown only for an eligible git repo (in-work-tree, not bare, on a named branch, no in-progress merge/rebase); for any other directory the flow is unchanged — straight to the provider picker, no warning UI.

When shown, the user picks **Use current branch** (today's behaviour, no worktree) or **New worktree**. New worktree suggests a branch name (`ccg/<kebab(topic-title)>` or `ccg/agent-<n>` with branch+worktree collision avoidance), one-tap confirm or edit-via-text-reply. Worktrees are created at `<repo>.worktrees/<slug>` (slug = branch with `/`→`-`) via `git -C <repo> worktree add`; `WorktreeError` is raised and surfaced as a one-line error with a Cancel button on failure. A dirty source repo is allowed with a warning line.

The chosen branch and worktree path are persisted on the existing `WindowState` (`worktree_path`, `worktree_branch`) atomically with the rest of the topic metadata — omitted from `to_dict` when unset, `.get()`-loaded for backward-compat with old `state.json`. No behaviour reads them yet; they are a forward investment for eventual cleanup UX. `SessionManager.set_window_worktree` is on the query-layer write/admin allow-list. Edit-name uses the only free-text input in the new flow: `AWAITING_WORKTREE_BRANCH_NAME` in user_data routes the next text message to branch-name validation (`git check-ref-format --branch`) before `text_handler` forwards it to the window. Cancel is the inline button (`/cancel` is a command and never reaches `text_handler`).

## Testing

### Test Structure

Tests mirror the source layout: `tests/ccgram/` for unit tests (with `handlers/` and `providers/` subdirectories matching source), `tests/integration/` for integration tests, `tests/e2e/` for end-to-end tests. Uses `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` decorators needed. No comments or docstrings in test files.

### Telegram Bot Testing Strategy

No reliable Telegram Bot API mock server exists. The project uses a tiered approach:

| Tier            | Pattern                                                                          | When to use                                       |
| --------------- | -------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Unit**        | `FakeTelegramClient` (or `AsyncMock`) injected via the `TelegramClient` Protocol | Testing handler logic in isolation                |
| **Integration** | Real PTB `Application` + `_do_post` patch                                        | Testing handler registration and dispatch routing |
| **E2E**         | Real agent CLIs + real tmux (no Telegram)                                        | Testing full agent lifecycle                      |

**Unit test pattern** (`FakeTelegramClient`): Handlers depend on the `TelegramClient` Protocol (`src/ccgram/telegram_client.py`), not `telegram.Bot`. Tests construct a `FakeTelegramClient()` from `ccgram.telegram_client`, pass it as the `client=` argument, and assert against `fake.calls` (a list of `(method, kwargs)` tuples) or use `fake.last_call` / `fake.call_count` helpers. Per-method return values can be configured via `fake.returns[method] = value` (or a `lambda **kw:` for dynamic responses); `fake.set_side_effect(method, [v1, v2, ...])` mirrors `unittest.mock.Mock.side_effect`. Production wraps the real bot with `PTBTelegramClient(bot)` at the call site (typically inside `bootstrap.py` or at the top of a callback handler that has `context.bot` in scope).

**Integration test pattern** (`_do_post` patch): Instantiate a real PTB Application, register real handlers, patch `type(application.bot)._do_post` to intercept all outbound HTTP calls. Dispatch real `Update`/`Message` objects via `application.process_update()`. This exercises PTB's filter evaluation, handler matching, and Forum topic routing (`message_thread_id`) without any network calls. The `PTBTelegramClient` adapter is real PTB internally, so integration tests still cover the adapter end to end. See `tests/integration/test_message_dispatch.py` for the base pattern.

### Shell Provider Tests

The shell provider has dedicated tests for each layer:

| Test File                                            | Coverage                                                                   |
| ---------------------------------------------------- | -------------------------------------------------------------------------- |
| `tests/ccgram/providers/test_shell.py`               | Provider capabilities, shell detection, prompt setup                       |
| `tests/ccgram/handlers/shell/test_shell_commands.py` | Command routing, LLM flow, approval keyboard, callbacks                    |
| `tests/ccgram/handlers/shell/test_shell_capture.py`  | Output extraction, passive monitoring, relay formatting, error suggestions |
| `tests/integration/test_shell_flow.py`               | Complete Telegram → Shell → Telegram round-trip                            |
| `tests/integration/test_shell_dispatch.py`           | PTB dispatch routing to shell handler                                      |
| `tests/integration/test_shell_llm_integration.py`    | Real LLM API round-trip with command execution                             |

## Emdash Integration

ccgram auto-discovers [emdash](https://github.com/generalaction/emdash) tmux sessions and lets users control emdash-managed agents from Telegram. Zero configuration — works automatically when both tools run on the same machine.

### Prerequisites

1. Enable persistent tmux sessions in emdash: add `"tmux": true` to `.emdash.json`
2. Install ccgram's hooks: `ccgram hook --install` (global hooks coexist with emdash's per-project hooks)

### How It Works

When emdash creates a tmux session (e.g. `emdash-claude-main-abc123`), ccgram's global hook fires and writes the session to `session_map.json`. The session monitor picks it up, and emdash sessions appear in the window picker when creating a new Telegram topic.

- **Discovery**: `tmux list-sessions` filtered by `emdash-` prefix
- **Window IDs**: Foreign windows use qualified IDs like `emdash-claude-main-abc123:@0` — these are valid tmux target strings
- **Lifecycle**: ccgram never kills emdash windows. They are marked `external=True` in `WindowState`
- **Provider detection**: Parsed from session name (`emdash-{provider}-main-{id}`)
- **Hook coexistence**: ccgram hooks are in `~/.claude/settings.json` (global), emdash hooks are in `.claude/settings.local.json` (per-project). Claude Code merges both

### Architecture

```
emdash (tmux: true)                  ccgram
─────────────────                    ──────
Creates tmux session ──────────────► Hook fires → session_map.json
emdash-claude-main-abc123            SessionMonitor reads entry
                                     Window picker shows session
User binds topic ──────────────────► send_keys/capture_pane to foreign session
                                     Status polling, emoji, interactive UI
User closes topic ─────────────────► Unbind only (no kill)
emdash kills session ──────────────► Dead window detection → cleanup
```

## Hook Configuration

Auto-install: `ccgram hook --install` — installs hooks for these Claude Code event types:

| Event         | Purpose                               | Async |
| ------------- | ------------------------------------- | ----- |
| SessionStart  | Session tracking (`session_map.json`) | No    |
| Notification  | Instant interactive UI detection      | No    |
| Stop          | Instant done/idle detection           | No    |
| StopFailure   | Alert on API error terminations       | Yes   |
| SessionEnd    | Session lifecycle cleanup             | Yes   |
| SubagentStart | Track subagent activity in status     | Yes   |
| SubagentStop  | Clear subagent status                 | Yes   |
| TeammateIdle  | Notify when a teammate goes idle      | Yes   |
| TaskCompleted | Notify when a team task completes     | Yes   |

All hooks write structured events to `events.jsonl`; SessionStart also writes `session_map.json`. The session monitor reads `events.jsonl` incrementally (byte-offset) and dispatches events to handlers. Terminal scraping remains as fallback when hook events are unavailable. Hook install/status/uninstall respects `CLAUDE_CONFIG_DIR` for non-default Claude config locations.

At startup, ccgram checks whether hooks are installed (Claude provider only) and logs a warning with the fix command if any are missing. This is non-blocking — terminal scraping works as fallback.

## Spec-Driven Development

Task management via `.spec/` directory. One task per session — complete fully before starting another.

```
.spec/
├── reqs/     # REQ-*.md (WHAT — requirements, success criteria)
├── epics/    # EPIC-*.md (grouping)
├── tasks/    # TASK-*.md (HOW — implementation steps)
├── memory/   # conventions.md, decisions.md
└── SESSION.yaml
```

| Command        | Purpose                         |
| -------------- | ------------------------------- |
| `/spec:work`   | Select, plan, implement, verify |
| `/spec:status` | Progress overview               |
| `/spec:new`    | Create new task or requirement  |
| `/spec:done`   | Mark complete with evidence     |

**Quick queries** (`~/.claude/scripts/specctl`):

```bash
specctl status                # Progress overview
specctl ready                 # Next tasks (priority-ordered)
specctl session show          # Current session state
specctl validate              # Check for issues
```

Never mark done until: `make check` passes (fmt + lint + typecheck + test).

## Publishing & Release

### PyPI + Homebrew Release Process

Tag format: use `v` prefix (e.g., `v2.1.2`) — hatch-vcs strips it to generate version `2.1.2`.

Release process:

```bash
# 1. Generate CHANGELOG locally
git cliff --tag vX.Y.Z --output CHANGELOG.md
# 2. Commit (do NOT use [skip ci] — see gotcha below)
git add CHANGELOG.md && git commit -m "docs: update CHANGELOG.md for vX.Y.Z"
git push origin main
# 3. Tag and push
git tag vX.Y.Z && git push origin vX.Y.Z
```

This triggers `.github/workflows/release.yml` (3 jobs):

1. **publish**: Build (`uv build`) + publish to PyPI via OIDC trusted publishing
2. **update-homebrew**: Generate formula via `scripts/generate_homebrew_formula.py` + push to `alexei-led/homebrew-tap`
3. **github-release**: Generate release notes (git-cliff inline) + create GitHub Release

CHANGELOG.md is maintained locally only — CI cannot push to protected `main`.

### Release Gotchas

- **`[skip ci]` kills tag-triggered workflows** — GitHub Actions skips workflows when the tag points to a commit with `[skip ci]` in its message. Never tag a `[skip ci]` commit. If needed, create an empty commit (`git commit --allow-empty -m "chore: release vX.Y.Z"`) as the tag target.

### GitHub Actions Best Practices

- Action refs: use exact format from docs (`release/v1` vs `v1` — branch refs differ from tags)
- Workflow permissions: scope `id-token: write` at job level for OIDC, not workflow level
- PyPI trusted publishing: match owner/repo/workflow/environment exactly in PyPI settings

### Auto-Generated Files

- Gitignore: `src/ccgram/_version.py` (regenerated by hatch-vcs from git tags)
- Exclude from linting: add to `pyproject.toml` `[tool.ruff] exclude` (not CLI flags)

## Inter-Agent Messaging

Agents in tmux windows can discover each other, exchange messages, broadcast notifications, and spawn new agents — with human oversight via Telegram.

### CLI: `ccgram msg`

```bash
ccgram msg list-peers [--json]                    # Show all active agent windows
ccgram msg find --provider claude --team backend  # Filter peers by attributes
ccgram msg send <to> <body> [--wait] [--notify]   # Send message (async default)
ccgram msg inbox [--json]                         # Check incoming messages
ccgram msg read <msg-id>                          # Read and mark message
ccgram msg reply <msg-id> <body>                  # Reply to a message
ccgram msg broadcast <body> [--team X]            # Send to all matching peers
ccgram msg register --task "..." --team "..."     # Declare task/team for discovery
ccgram msg spawn --provider claude --cwd ~/proj   # Request new agent (needs approval)
ccgram msg sweep                                  # Clean expired messages
```

### Key Design

- **File-based mailbox** (`~/.ccgram/mailbox/`) — per-window inbox directories with timestamp-prefixed JSON messages, atomic writes
- **Qualified IDs** — `session:@N` format (e.g. `ccgram:@0`) matching session_map convention
- **Broker delivery** — poll loop injects pending messages into idle agent windows via send_keys; shell windows are inbox-only
- **Telegram visibility** — silent notifications in both sender/recipient topics, grouped, edit-in-place for replies
- **Spawn approval** — agents request new instances, user approves via Telegram inline keyboard, auto-creates topic
- **Self-identification** — `CCGRAM_WINDOW_ID` env var set automatically on window creation; tmux fallback
- **Safety** — rate limiting (messages + spawns), loop detection with pause/allow, deadlock prevention for `--wait`

### Messaging Configuration

| Setting       | Env Var                    | Default              |
| ------------- | -------------------------- | -------------------- |
| Auto-spawn    | `CCGRAM_MSG_AUTO_SPAWN`    | `false`              |
| Max windows   | `CCGRAM_MSG_MAX_WINDOWS`   | `10`                 |
| Wait timeout  | `CCGRAM_MSG_WAIT_TIMEOUT`  | `60` (seconds)       |
| Spawn timeout | `CCGRAM_MSG_SPAWN_TIMEOUT` | `300` (seconds)      |
| Spawn rate    | `CCGRAM_MSG_SPAWN_RATE`    | `3` (per window/hr)  |
| Message rate  | `CCGRAM_MSG_RATE_LIMIT`    | `10` (per window/5m) |

## Mini App Dashboard (Optional)

Optional aiohttp web surface that runs alongside the bot when `CCGRAM_MINIAPP_BASE_URL` is set. Opens from a "🪟 Dashboard" inline button on the status bubble inside Telegram's WebApp container. Three surfaces ship in v3.0: live xterm.js terminal (read-only), paginated transcript with full-text search, and a multi-pane grid view.

**Subpackage layout** (`src/ccgram/miniapp/`):

| Module                 | Description                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`          | Public API: `start_server`, `stop_server`, `build_app`, `sign_token`, `verify_token`, `validate_init_data`                             |
| `auth.py`              | HMAC-signed window tokens (window_id + user_id + expiry) and Telegram WebApp `initData` validation                                     |
| `server.py`            | aiohttp app factory + lifecycle (`start_server`/`stop_server`); routes: `/healthz`, `/app/{token}`, `/static/`, sub-route registration |
| `api/__init__.py`      | Router registration helpers (`register_terminal_routes`, `register_transcript_routes`)                                                 |
| `api/terminal.py`      | Live-terminal websocket (`/ws/terminal/{token}`) and pane-list HTTP (`/api/panes/{token}`); per-pane multiplex via `?pane=`            |
| `api/transcript.py`    | Transcript HTTP — paginated history (`/api/transcript/{token}`) + search (`/api/transcript/{token}/search?q=...`)                      |
| `static/index.html`    | SPA shell — Telegram WebApp SDK, payload meta tag, mounts terminal/transcript/panes surfaces                                           |
| `static/terminal.js`   | xterm.js client with delta streaming                                                                                                   |
| `static/transcript.js` | Paginated transcript viewer + search UI                                                                                                |
| `static/panes.js`      | Multi-pane grid view with focused-tile transitions                                                                                     |

**Lifecycle**: `start_miniapp_if_enabled` / `stop_miniapp_if_enabled` (`src/ccgram/main.py`) are wired into `bot.py` `post_init` / `post_shutdown`. Server start failures are logged and swallowed — the bot keeps running even if the optional server can't bind. Server is gated entirely on `CCGRAM_MINIAPP_BASE_URL`; when unset, neither the HTTP listener nor the dashboard button are exposed.

**Auth**: Tokens are HMAC-signed with the bot token, scoped to a single (window_id, user_id) pair, and short-lived. Every API surface validates the token on every request — there is no cross-window access.

**Deployment**: Production requires TLS termination + reverse proxy (cloudflared, caddy, nginx). The aiohttp server binds to a local host/port and expects an external proxy to forward HTTPS. BotFather configuration: register the Mini App via `/setdomain` and `/newapp`.

## Architecture Details

See @.claude/rules/architecture.md for full system diagram and module inventory.
See @.claude/rules/topic-architecture.md for topic→window→session mapping details.
See @.claude/rules/message-handling.md for message queue, merging, and rate limiting.

`bot.py` is a 172-line factory + lifecycle delegate. Command/message/callback registration lives in `src/ccgram/handlers/registry.py` (`register_all`); post_init wiring lives in `src/ccgram/bootstrap.py` (`bootstrap_application` + `shutdown_runtime`). Handlers depend on the `TelegramClient` Protocol (`src/ccgram/telegram_client.py`), with `PTBTelegramClient` adapting a real PTB `Bot` in production and `FakeTelegramClient` injected by unit tests.

## Round 4 Outcomes (modularity decouple)

Round 4 (May 2026, branch `modularity-decouple-round-4`) reshaped the codebase to lower per-task context size and raise testability without changing user-visible behavior:

- **F1 — handler subpackages.** `src/ccgram/handlers/` was 50+ flat peer modules; now grouped into 14 feature subpackages (`interactive/`, `live/`, `messaging/`, `messaging_pipeline/`, `polling/`, `recovery/`, `send/`, `shell/`, `status/`, `text/`, `toolbar/`, `topics/`, `voice/`) plus the documented top-level handlers (`callback_*`, `cleanup`, `command_*`, `file_handler`, `hook_events`, `inline`, `reactions`, `registry`, `response_builder`, `sessions_dashboard`, `sync_command`, `upgrade`, `user_state`). Hard cut — no compat shims; subpackage `__init__.py` re-exports the public surface.
- **F2 — constructor DI for stores.** `SessionManager` constructs `WindowStateStore`, `ThreadRouter`, `UserPreferences`, and `SessionMapSync` with explicit `schedule_save` (and store-specific) callbacks. The `_wire_singletons` monkey-patch and the silent `unwired_save` default are gone. `register_*_callback` helpers fail loud on double-registration; unwired callee defaults raise `RuntimeError("not wired")`. Module-level singletons survive as proxy objects forwarding to the wired instance.
- **F3 — bootstrap split.** `bot.py` shrank from ~720 to 172 lines. `handlers/registry.py` owns PTB handler registration; `bootstrap.py` owns `post_init` (named functions: `register_provider_commands`, `verify_hooks_installed`, `wire_runtime_callbacks`, `start_session_monitor`, `start_status_polling`, `start_miniapp_if_enabled`) and `post_shutdown` teardown. Ordering invariant: `wire_runtime_callbacks` must run before `start_session_monitor`.
- **F4 — `window_tick` decide/observe/apply.** The 694-line god module became `handlers/polling/window_tick/` with `decide.py` (pure, zero deps on tmux/PTB/singletons), `observe.py` (pure inputs in, `TickContext` out), and `apply.py` (the only side-effect file). `decide_tick` is unit-tested without mocks.
- **F5 — `TelegramClient` Protocol.** A grep-verified Protocol covering exactly the 18 bot API methods used by handlers. `PTBTelegramClient(bot)` in production, `FakeTelegramClient` in tests, `unwrap_bot(client)` as the escape hatch for PTB-only helpers (`do_api_request`/`DraftStream`). All `from telegram.ext` imports inside `handlers/` are now `if TYPE_CHECKING:` (only `handlers/registry.py` keeps a runtime import, as the documented PTB wiring spine).
- **F6 — lazy-import audit.** 251 in-function imports inventoried, 25 hoisted to module level, ~25 redundant ones removed during the sweep, 160 remaining sites documented with `# Lazy: <reason>` comments citing the cycle path or wiring contract that requires lateness. Net: 251 → 201 with the rest justified inline.
- **Cycle detection.** New integration test `tests/integration/test_import_no_cycles.py` parametrizes `python -c "import {module}"` over 29 modules from a clean interpreter, catching circular-import regressions before they break runtime.

State files (`state.json`, `session_map.json`, `events.jsonl`, `monitor_state.json`, `mailbox/`), CLI flags, env vars, bot commands, and hook configuration are unchanged. `make check` (fmt + lint + typecheck + 4401 unit + 126 integration) is green; `make test-e2e` has pre-existing unrelated failures (group-chat-id pruning) tracked separately.

## Round 5 Outcomes (modularity decouple)

Round 5 (May 2026, branch `modularity-decouple-round-5`) closed the residual gaps from the post-Round-4 modularity review (`docs/modularity-review/2026-05-01/modularity-review.md`). All five fixes are structural and behaviour-preserving:

- **F1 — `polling_strategies.py` split.** The 1 073-LOC mixed module was split into `handlers/polling/polling_types.py` (~150 LOC: `TickContext`, `TickDecision`, `PaneTransition`, `WindowPollState`, `TopicPollState`, all module-level constants, the pure `is_shell_prompt`; imports stdlib + `ccgram.providers.base.StatusUpdate` only) and `handlers/polling/polling_state.py` (~900 LOC: `TerminalPollState`, `TerminalScreenBuffer`, `InteractiveUIStrategy`, `TopicLifecycleStrategy`, `PaneStatusStrategy`, the five module-level singletons, `reset_window_polling_state`). `polling_strategies.py` deleted; the `from . import polling as _polling` workaround in `handlers/registry.py` is gone. `window_tick/decide.py` now imports only the pure contract.
- **F2 — read-path migration.** ~14 read-ish handler call sites that bypassed the query layer now go through `window_query` / `session_query`. `session_manager.*` direct access is restricted to the documented write/admin allow-list (~30 sites: `set_window_provider`, `set_window_origin`, `set_window_approval_mode`, `cycle_*`, `audit_state`, `prune_*`, `sync_display_names`). Codified by `tests/ccgram/test_query_layer_only_for_handlers.py` — an AST walk over 86 handler files asserts every `session_manager.<attr>` access is on the allow-list. Read access slipping back in fails the build.
- **F3 — `recovery_callbacks.py` split.** The 890-LOC two-flow module became three siblings: `handlers/recovery/recovery_banner.py` (~450 LOC, dead-window banner UX), `handlers/recovery/resume_picker.py` (~400 LOC, resume picker UX + transcript scan), and a thin `recovery_callbacks.py` dispatcher (~170 LOC: `_dispatch`, `handle_recovery_callback`, plus the shared `_validate_recovery_state`/`_clear_recovery_state` validators that both siblings need). Subpackage `__init__.py` re-exports the same public surface; pinned by `tests/ccgram/handlers/recovery/test_recovery_subpackage_surface.py`.
- **F4 — `command_orchestration.py` split.** The 775-LOC four-concern module became `handlers/commands/` subpackage following the `shell/` pattern: `forward.py` (forward command handler + `_normalize_slash_token` + `_handle_clear_command`), `menu_sync.py` (provider menu cache, `sync_scoped_*`, `setup_menu_refresh_job`), `failure_probe.py` (`_extract_*`, `_capture_command_probe_context`, `_probe_transcript_command_error`, `_spawn_command_failure_probe`), `status_snapshot.py` (`_status_snapshot_probe_offset`, `_maybe_send_status_snapshot`). `commands/__init__.py` hosts `commands_command` + `toolbar_command` and re-exports the public surface; pinned by `tests/ccgram/handlers/commands/test_commands_subpackage_surface.py`.
- **F5 — lazy-import lint + cycle test expansion.** `scripts/lint_lazy_imports.py` walks `src/ccgram/**/*.py` via AST, flags every in-function `Import`/`ImportFrom` not preceded by a `# Lazy:` comment, not inside `if TYPE_CHECKING:`, and not inside a `_reset_*_for_testing` function. The walker recurses through compound statements (`try`/`except`/`except*`/`finally`/`if`/`else`/`with`/`for`/`while`) and into nested `def`/`class` bodies, and accepts multi-line `# Lazy:` comment blocks (contiguous `#`-prefixed lines above the import are scanned for the marker). Wired into `make lint` as `lint-lazy`. All 250 in-function imports across the codebase are now annotated. `tests/integration/test_import_no_cycles.py` expanded from 29 hand-listed modules to programmatic enumeration of all 162 modules under `src/ccgram/`.

New structural tests codify the invariants: `test_polling_types_purity.py` (subprocess load-time + AST static check), `test_query_layer_only_for_handlers.py` (86 parametrized cases), `test_recovery_subpackage_surface.py` (6 cases), `test_commands_subpackage_surface.py` (5 cases), `test_lint_lazy_imports.py` (27 cases), `test_import_no_cycles.py` (162 cases). State files, CLI flags, env vars, bot commands, and hook configuration are unchanged. `make check` (fmt + lint + typecheck + 4540 unit + 259 integration) is green; the one observed flaky test (`test_uses_pyte_result_when_available` under xdist worker pollution) is pre-existing from Round 4.
