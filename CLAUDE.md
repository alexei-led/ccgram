# CLAUDE.md

ccgram (Command & Control Bot) — manage AI coding agents from Telegram via tmux. Each Telegram Forum topic is bound to one tmux window running one agent CLI instance (Claude Code, Codex, Gemini, or a plain shell).

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
./scripts/restart.sh start            # Start local dev instance in tmux ccgram:__main__
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

## Core Design Constraints

- **1 Topic = 1 Window = 1 Session** — all internal routing keyed by tmux window ID (`@0`, `@12`), not window name. Window names kept as display names. Same directory can have multiple windows.
- **Topic-only** — no backward-compat for non-topic mode. No `active_sessions`, no `/list`, no General topic routing.
- **No message truncation** at parse layer — splitting only at send layer (`split_message`, 4096 char limit).
- **Entity-based formatting** — use `safe_reply`/`safe_edit`/`safe_send` helpers which convert markdown to plain text + MessageEntity offsets (no parse errors possible, auto fallback to plain text). Internal queue/UI code calls bot API directly with its own fallback.
- **Hook-based session tracking** — Claude Code hooks (SessionStart, Notification, Stop, StopFailure, SessionEnd, SubagentStart, SubagentStop, TeammateIdle, TaskCompleted) write to `session_map.json` and `events.jsonl`; monitor polls both to detect session changes and deliver instant event notifications. Missing hooks are detected at startup with an actionable warning.
- **Shell provider chat-first design** — text sent to a shell topic goes through the LLM for NL→command generation by default; prefix with `!` to send a raw command directly. When no LLM is configured, all text is forwarded as raw commands. Prompt marker `ccgram:N❯` enables output isolation and exit code detection. Two setup paths: **Auto-setup** (explicit shell topic creation via directory browser) configures the marker immediately without asking. **Ask flow** (external window bind or runtime provider switch to shell) shows an inline keyboard [Set up] / [Skip]; Skip is respected for the session (lazy recovery won't override). On provider switch away from shell and back, a fresh offer is shown. If marker is lost mid-session (`exec bash`, profile reload), it is lazily restored on the next command send (unless user chose Skip). Marker setup is session-scoped (PS1/PROMPT override) — never modifies shell config files.
- **Message queue per user** — FIFO ordering, message merging (3800 char limit), tool_use/tool_result pairing.
- **Rate limiting** — 1.1s minimum interval between messages per user via `rate_limit_send()`.

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
- Project structure: handlers in `src/ccgram/handlers/`, core modules in `src/ccgram/`, tests mirror source under `tests/ccgram/`.

## Provider Configuration

ccgram supports multiple agent CLI backends via the provider abstraction (`src/ccgram/providers/`). Providers are resolved per-window — different topics can use different providers simultaneously.

| Setting              | Env Var                 | Default         |
| -------------------- | ----------------------- | --------------- |
| Default provider     | `CCGRAM_PROVIDER`       | `claude`        |
| Per-provider command | `CCGRAM_<NAME>_COMMAND` | (from provider) |

Launch command override: `CCGRAM_<NAME>_COMMAND` (e.g. `CCGRAM_CLAUDE_COMMAND=ce --current`), falls back to provider default. The shell provider has no override — tmux opens `$SHELL` by default. Resolved by `resolve_launch_command()` in `providers/__init__.py`.

### Per-Window Provider Model

Each tmux window tracks its own provider in `WindowState.provider_name`. Resolution order:

1. Window's stored `provider_name` (set during topic creation or auto-detected)
2. Config default (`CCGRAM_PROVIDER` env var, defaults to `claude`)

Key functions:

- `get_provider_for_window(window_id)` — resolves provider instance for a specific window
- `detect_provider_from_pane(pane_current_command, pane_tty, window_id)` — auto-detects provider from process name with ps-based TTY fallback for JS-runtime-wrapped CLIs
- `detect_provider_from_command(pane_current_command)` — fast-path detection from process basename (claude/codex/gemini/shell)
- `set_window_provider(window_id, provider_name)` — persists provider choice on SessionManager

When creating a topic via the directory browser, users can choose the provider (Claude default, Codex, Gemini, Shell). Externally created tmux windows are auto-detected via `detect_provider_from_pane()` which tries process basename first, then falls back to `ps -t` foreground process inspection (with PGID caching) when the pane command is a JS runtime wrapper (node/bun). The global `get_provider()` remains as fallback for CLI commands without window context (e.g., `doctor`, `status`). Runtime re-detection (every 1s poll cycle) triggers prompt marker check on each transition to shell. Explicit shell topic creation (directory browser) auto-configures the marker.

### Provider Capability Matrix

| Capability       | Claude                          | Codex              | Gemini                      | Shell                       |
| ---------------- | ------------------------------- | ------------------ | --------------------------- | --------------------------- |
| Hook events      | Yes (all supported event types) | No                 | No                          | No                          |
| Resume           | Yes (`--resume`)                | Yes (`resume`)     | Yes (`--resume idx/latest`) | No                          |
| Continue         | Yes                             | Yes                | Yes                         | No                          |
| Transcript       | JSONL                           | JSONL              | JSON (whole-file read)      | None                        |
| Incremental read | Yes                             | Yes                | No (whole-file JSON)        | No                          |
| Commands         | Yes                             | Yes                | Yes                         | No                          |
| Status detection | Hook events + pyte + spinner    | Activity heuristic | Pane title + interactive UI | Shell prompt idle detection |

Capabilities gate UX per-window: recovery keyboard only shows Continue/Resume buttons when supported; `ccgram doctor` checks all hook event types for Claude. Codex, Gemini, and Shell have no hooks — session tracking for these providers relies on auto-detection from running processes.

### LLM Configuration

The shell provider uses an LLM to translate natural language input into shell commands. LLM settings are independent of the agent provider and apply only when a shell topic is active.

| Setting         | Env Var                  | Default         |
| --------------- | ------------------------ | --------------- |
| LLM provider    | `CCGRAM_LLM_PROVIDER`    | (empty)         |
| LLM API key     | `CCGRAM_LLM_API_KEY`     | (empty)         |
| LLM base URL    | `CCGRAM_LLM_BASE_URL`    | (from provider) |
| LLM model       | `CCGRAM_LLM_MODEL`       | (from provider) |
| LLM temperature | `CCGRAM_LLM_TEMPERATURE` | `0.1`           |

Supported LLM providers: `openai`, `xai`, `deepseek`, `anthropic`, `groq`, `ollama`. API key resolution: `CCGRAM_LLM_API_KEY` > provider-specific env var (e.g. `XAI_API_KEY`) > `OPENAI_API_KEY` (universal fallback). When `CCGRAM_LLM_PROVIDER` is unset, the shell provider skips NL→command generation and forwards all input as raw commands. Set temperature to `0` for deterministic output with cheap/fast models.

### Migration Notes

Existing Claude deployments need no changes — `claude` is the default provider. Windows without an explicit `provider_name` fall back to the config default. The hook subsystem (`ccgram hook --install`) is Claude-specific and skipped for other providers.

## Testing

### Test Structure

Tests mirror the source layout: `tests/ccgram/` for unit tests, `tests/integration/` for integration tests, `tests/e2e/` for end-to-end tests.

### Telegram Bot Testing Strategy

No reliable Telegram Bot API mock server exists. The project uses a tiered approach:

| Tier            | Pattern                                   | When to use                                       |
| --------------- | ----------------------------------------- | ------------------------------------------------- |
| **Unit**        | `MagicMock`/`AsyncMock` for PTB objects   | Testing handler logic in isolation                |
| **Integration** | Real PTB `Application` + `_do_post` patch | Testing handler registration and dispatch routing |
| **E2E**         | Real agent CLIs + real tmux (no Telegram) | Testing full agent lifecycle                      |

**Integration test pattern** (`_do_post` patch): Instantiate a real PTB Application, register real handlers, patch `type(application.bot)._do_post` to intercept all outbound HTTP calls. Dispatch real `Update`/`Message` objects via `application.process_update()`. This exercises PTB's filter evaluation, handler matching, and Forum topic routing (`message_thread_id`) without any network calls. See `tests/integration/test_message_dispatch.py` for the base pattern.

### Shell Provider Tests

The shell provider has dedicated tests for each layer:

| Test File                                         | Coverage                                                                   |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| `tests/ccgram/test_shell_provider.py`             | Provider capabilities, shell detection, prompt setup                       |
| `tests/ccgram/test_shell_commands.py`             | Command routing, LLM flow, approval keyboard, callbacks                    |
| `tests/ccgram/test_shell_capture.py`              | Output extraction, passive monitoring, relay formatting, error suggestions |
| `tests/integration/test_shell_flow.py`            | Complete Telegram → Shell → Telegram round-trip                            |
| `tests/integration/test_shell_dispatch.py`        | PTB dispatch routing to shell handler                                      |
| `tests/integration/test_shell_llm_integration.py` | Real LLM API round-trip with command execution                             |

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

## Architecture Details

See @.claude/rules/architecture.md for full system diagram and module inventory.
See @.claude/rules/topic-architecture.md for topic→window→session mapping details.
See @.claude/rules/message-handling.md for message queue, merging, and rate limiting.
