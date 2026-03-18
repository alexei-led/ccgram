# Guides

## Upgrading

```bash
uv tool upgrade ccgram                # uv (recommended)
pipx upgrade ccgram                   # pipx
brew upgrade ccgram                   # Homebrew
```

## CLI Reference

```
ccgram                        # Start the bot
ccgram status                 # Show running state (no token needed)
ccgram doctor                 # Validate setup and diagnose issues
ccgram doctor --fix           # Auto-fix issues (install hook, kill orphans)
ccgram hook --install         # Install Claude Code hooks (7 event types)
ccgram hook --uninstall       # Remove all hooks
ccgram hook --status          # Check per-event hook installation status
ccgram --version              # Show version
ccgram -v                     # Run with debug logging
```

## Local Dev in tmux

Recommended local development model:

- Run ccgram in a dedicated control window `ccgram:__main__`.
- Keep agent windows in the same `ccgram` tmux session.
- Restart by sending Ctrl-C to the control pane.

Use the helper script:

```bash
./scripts/restart.sh start      # fresh start; creates ccgram:__main__ if missing
./scripts/restart.sh status     # show current command + last logs
./scripts/restart.sh restart    # sends Ctrl-C to control pane (supervisor restarts)
./scripts/restart.sh stop       # sends Ctrl-\ to control pane (supervisor exits)
```

Direct key behavior in the control pane (`ccgram:__main__`):

- `Ctrl-C`: restart ccgram.
- `Ctrl-\`: stop the local dev supervisor loop.

### Fresh Start Guide

If you are starting from scratch:

1. `cd /path/to/ccgram`
2. `./scripts/restart.sh start`
3. `tmux attach -t ccgram`
4. In another terminal (or another pane), open your agent windows in the same tmux session.

The `start` command creates the tmux session/window if they do not exist, so no manual tmux bootstrap is required.

## Testing

CCGram has three test tiers:

| Tier        | Command                 | Time     | Requirements      |
| ----------- | ----------------------- | -------- | ----------------- |
| Unit        | `make test`             | ~10s     | None (all mocked) |
| Integration | `make test-integration` | ~7s      | tmux              |
| E2E         | `make test-e2e`         | ~3-4 min | tmux + agent CLIs |

`make check` runs unit + integration tests together with formatting, linting, and type checking.

### E2E Tests

End-to-end tests exercise the full lifecycle: inject fake Telegram updates → real PTB application → real tmux windows → real agent CLI processes → intercept Bot API responses. Each provider's tests are skipped automatically if its CLI is not installed.

**Prerequisites:**

- tmux installed and in PATH
- One or more agent CLIs installed and authenticated: `claude`, `codex`, `gemini`

**Test coverage per provider:**

| Provider | Tests | Scenarios                                                                                                                                                    |
| -------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Claude   | 9     | Lifecycle, `/sessions`, `/screenshot`, `/help` forwarding, recovery (fresh + continue), status transitions, multi-topic isolation, notification mode cycling |
| Codex    | 3     | Lifecycle, command forwarding, recovery                                                                                                                      |
| Gemini   | 3     | Lifecycle, command forwarding, recovery                                                                                                                      |

**How it works:** The Bot API HTTP layer is mocked — fake `Update` objects are injected via `app.process_update()` and all outgoing API calls are intercepted and recorded for assertions. The tests drive through the full topic binding flow (directory browser → provider picker → mode select → window creation) and verify agent processes launch, messages are forwarded, and responses are delivered.

**Running:**

```bash
make test-e2e                                         # All providers
uv run pytest tests/e2e/test_claude_lifecycle.py -v   # Claude only
uv run pytest tests/e2e/test_codex_lifecycle.py -v    # Codex only
uv run pytest tests/e2e/test_gemini_lifecycle.py -v   # Gemini only
```

The tests create an isolated `ccgram-e2e` tmux session that does not interfere with a running `ccgram` instance. Safe to run from a tmux window.

## Configuration

All settings accept both CLI flags and environment variables. CLI flags take precedence. `TELEGRAM_BOT_TOKEN` is env-only for security (flags are visible in `ps`).

| Variable / Flag                                | Default           | Description                                          |
| ---------------------------------------------- | ----------------- | ---------------------------------------------------- |
| `TELEGRAM_BOT_TOKEN`                           | _(required)_      | Bot token from @BotFather (env only)                 |
| `ALLOWED_USERS` / `--allowed-users`            | _(required)_      | Comma-separated Telegram user IDs                    |
| `CCGRAM_DIR` / `--config-dir`                  | `~/.ccgram`       | Config and state directory                           |
| `TMUX_SESSION_NAME` / `--tmux-session`         | `ccgram`          | tmux session name                                    |
| `CCGRAM_PROVIDER` / `--provider`               | `claude`          | Default agent provider (`claude`, `codex`, `gemini`) |
| `CCGRAM_<NAME>_COMMAND`                        | _(from provider)_ | Per-provider launch command (env only, see below)    |
| `CCGRAM_GROUP_ID` / `--group-id`               | _(all groups)_    | Restrict to one Telegram group                       |
| `CCGRAM_INSTANCE_NAME` / `--instance-name`     | hostname          | Display label for this instance                      |
| `CCGRAM_LOG_LEVEL` / `--log-level`             | `INFO`            | Logging level (DEBUG, INFO, WARNING, ERROR)          |
| `MONITOR_POLL_INTERVAL` / `--monitor-interval` | `2.0`             | Seconds between transcript polls                     |
| `AUTOCLOSE_DONE_MINUTES` / `--autoclose-done`  | `30`              | Auto-close done topics after N minutes (0=off)       |
| `AUTOCLOSE_DEAD_MINUTES` / `--autoclose-dead`  | `10`              | Auto-close dead sessions after N minutes (0=off)     |

## Tmux Session Auto-Detection

When ccgram starts inside an existing tmux session, it auto-detects the session name and attaches to it instead of creating a new `ccgram` session. This is useful when you already have a tmux session with agent windows.

**How it works:**

1. If `$TMUX` is set and no `--tmux-session` flag is given, ccgram detects the current session name
2. The bot's own tmux window is automatically excluded from the window list
3. If another ccgram instance is already running in the same session, startup is refused

**Override:** `--tmux-session=NAME` or `TMUX_SESSION_NAME=NAME` always takes precedence over auto-detection.

**Outside tmux:** Behavior is unchanged — ccgram creates a `ccgram` session with a `__main__` placeholder window.

| Scenario                         | Behavior                                            |
| -------------------------------- | --------------------------------------------------- |
| Outside tmux, no flags           | Creates `ccgram` session + `__main__` window        |
| Outside tmux, `--tmux-session=X` | Creates/attaches `X` + `__main__` window            |
| Inside tmux, no flags            | Auto-detects session, skips own window, no creation |
| Inside tmux, `--tmux-session=X`  | Overrides auto-detect, uses `X`                     |

## Auto-Close Behavior

CCGram automatically closes Telegram topics when sessions end, reducing clutter:

- **Done topics** (`--autoclose-done`, default: 30 min) — When Claude finishes a task and the session completes normally, the topic auto-closes after 30 minutes.
- **Dead sessions** (`--autoclose-dead`, default: 10 min) — When a Claude process crashes or the tmux window is killed externally, the topic auto-closes after 10 minutes.

Set to `0` to disable:

```bash
ccgram --autoclose-done 0 --autoclose-dead 0
```

## Multi-Instance Setup

Run multiple ccgram instances on the same machine, each owning a different Telegram group. All instances can share a single bot token.

**Example: work + personal instances**

Instance 1 (`~/.ccgram-work/.env`):

```ini
TELEGRAM_BOT_TOKEN=same_token_for_both
ALLOWED_USERS=123456789
CCGRAM_GROUP_ID=-1001111111111
CCGRAM_INSTANCE_NAME=work
CCGRAM_DIR=~/.ccgram-work
TMUX_SESSION_NAME=ccgram-work
```

Instance 2 (`~/.ccgram-personal/.env`):

```ini
TELEGRAM_BOT_TOKEN=same_token_for_both
ALLOWED_USERS=123456789
CCGRAM_GROUP_ID=-1002222222222
CCGRAM_INSTANCE_NAME=personal
CCGRAM_DIR=~/.ccgram-personal
TMUX_SESSION_NAME=ccgram-personal
```

Run both:

```bash
CCGRAM_DIR=~/.ccgram-work ccgram &
CCGRAM_DIR=~/.ccgram-personal ccgram &
```

Each instance uses a separate tmux session, config directory, and state. When `CCGRAM_GROUP_ID` is set, an instance silently ignores updates from other groups.

Without `CCGRAM_GROUP_ID`, a single instance processes all groups (the default).

> To find your group's chat ID, add [@RawDataBot](https://t.me/RawDataBot) to the group — it replies with the chat ID (a negative number like `-1001234567890`).

## Creating Sessions from the Terminal

Besides creating sessions through Telegram topics, you can create tmux windows directly:

```bash
# Attach to the ccgram tmux session
tmux attach -t ccgram

# Create a new window for your project
tmux new-window -n myproject -c ~/Code/myproject

# Start any supported agent CLI
claude     # or: codex, gemini
```

The window must be in the ccgram tmux session (configurable via `TMUX_SESSION_NAME`). For Claude, the SessionStart hook registers it automatically. For Codex and Gemini, CCGram auto-detects the provider from the running process name. In both cases, the bot creates a matching Telegram topic.

This works even on a fresh instance with no existing topic bindings (cold-start).

## Session Recovery

When an agent session exits or crashes, the bot detects the dead window and offers recovery options via inline buttons:

- **Fresh** — Kill the old window, create a new one in the same directory
- **Continue** — Resume the last conversation (all providers support this)
- **Resume** — Browse and select a past session to resume from

The buttons shown adapt to each provider's capabilities. All three providers (Claude, Codex, Gemini) support Fresh, Continue, and Resume.

## Provider Support

CCGram supports multiple agent CLI backends. Each Telegram topic can use a different provider — you choose when creating a session via the directory browser.

### Supported Providers

| Provider    | CLI Command | Hook Events         | Status Detection                                          |
| ----------- | ----------- | ------------------- | --------------------------------------------------------- |
| Claude Code | `claude`    | Yes (7 event types) | Hook events + pyte VT100 + spinner                        |
| Codex CLI   | `codex`     | No                  | pyte VT100 interactive UI + transcript activity heuristic |
| Gemini CLI  | `gemini`    | No                  | Pane title + interactive UI                               |

### Choosing a Provider

**From Telegram**: When you create a new topic and select a directory, a provider picker appears with Claude (default), Codex, and Gemini options. After provider selection, CCGram asks for session mode:

- `✅ Standard` (normal approvals)
- `🚀 YOLO` (provider-specific permissive mode)

**From the terminal**: If you create a tmux window manually and start an agent CLI, CCGram auto-detects the provider from the running process name. For Gemini sessions launched via bun/node wrappers, it also checks Gemini pane-title symbols (`✦`, `✋`, `◇`).

**Default provider**: Set `CCGRAM_PROVIDER=codex` (or `gemini`) to change the default. Claude is the default if unset.

### Session Mode (Standard vs YOLO)

CCGram stores mode per window and reuses it for recover/continue/resume flows.

- `normal` mode launches the provider command as-is.
- `yolo` mode appends the provider-native permissive flag:
  - Claude: `--dangerously-skip-permissions`
  - Codex: `--dangerously-bypass-approvals-and-sandbox`
  - Gemini: `--yolo`

YOLO sessions are indicated in Telegram topic titles with a positive `🚀` badge and in `/sessions` with a `[YOLO]` tag.

### Provider Differences

**Claude Code** has the richest integration — 7 hook event types (SessionStart, Notification, Stop, SubagentStart, SubagentStop, TeammateIdle, TaskCompleted) provide instant session tracking, interactive UI detection, done/idle detection, subagent activity monitoring, and agent team notifications. The bot also uses a pyte VT100 screen buffer as fallback for terminal status parsing. Multi-pane windows (e.g. from agent teams) are automatically scanned for blocked panes and surfaced as inline keyboard alerts.

**Codex CLI** and **Gemini CLI** lack a session hook, so session tracking relies on hookless transcript discovery plus provider detection. Codex interactive prompts (question lists, permission prompts, and other selection UIs) are detected from terminal screen content via pyte and shown with inline keyboard controls. For edit-approval prompts, CCGram reformats dense terminal diffs into a compact summary with a short preview while keeping the Yes/No confirmation choices and bottom action hints intact. Gemini sets pane titles (`Working: ✦`, `Action Required: ✋`, `Ready: ◇`) that CCGram reads for status, and its `@inquirer/select` permission prompts are detected as interactive UI. Gemini transcript discovery matches project hash/alias only (no cross-project full scan) to avoid wrong-session attachment.

### Codex Edit Approval Formatting

When Codex asks for approval on file edits, terminal output can include dense side-by-side diff lines that are hard to read in Telegram. CCGram reformats that content before sending the interactive prompt:

- Keeps the approval controls and action hints intact (`Yes/No`, `Press enter`, `Esc`).
- Adds a compact summary (`File`, `Changes: +N -M`).
- Adds a short preview of parsed changed lines when available.
- Omits unreadable wrapped diff blobs instead of forwarding noisy raw text.

Typical output shape:

```text
Do you want to make this edit to src/ccgram/example.py?
File: src/ccgram/example.py
Changes: +1 -1
Preview:
  - return old_value
  + return new_value

› 1. Yes, proceed (y)
  2. Yes, and don't ask again for these files (a)
  3. No, and tell Codex what to do differently (esc)
Press enter to confirm or esc to cancel
```

### Custom Launch Commands

Override the CLI command used to launch each provider via `CCGRAM_<NAME>_COMMAND` env vars:

```ini
CCGRAM_CLAUDE_COMMAND=ce --current
CCGRAM_CODEX_COMMAND=my-codex-wrapper
CCGRAM_GEMINI_COMMAND=/opt/gemini/run
```

`<NAME>` is uppercase: `CLAUDE`, `CODEX`, `GEMINI`. Defaults to the provider's built-in command (`claude`, `codex`, `gemini`) when unset. New providers automatically support `CCGRAM_<NAME>_COMMAND` without code changes.

You can use this for a global "today" setup (all new sessions), for example:

```ini
CCGRAM_CLAUDE_COMMAND=claude --dangerously-skip-permissions
CCGRAM_CODEX_COMMAND=codex --dangerously-bypass-approvals-and-sandbox
CCGRAM_GEMINI_COMMAND=gemini --yolo
```

For ccgram-managed Gemini launches, CCGram also injects
`GEMINI_CLI_SYSTEM_SETTINGS_PATH=~/.ccgram/gemini-system-settings.json` with
`tools.shell.enableInteractiveShell=false` to avoid node-pty `EBADF` crashes in
tmux. If you set `CCGRAM_GEMINI_COMMAND`, your override is used as-is.

### Provider-Specific Commands

Each provider exposes its own slash commands to the Telegram menu. Examples:

- **Claude**: `/clear`, `/compact`, `/cost`, `/doctor`, `/permissions`...
- **Codex**: `/model`, `/mode`, `/status`, `/diff`, `/compact`, `/mcp`...
- **Gemini**: `/chat`, `/clear`, `/compress`, `/model`, `/memory`, `/vim`...

For Codex, `/status` now sends a transcript-based fallback snapshot in Telegram
(session/cwd/token/rate-limit summary) because some Codex builds render status
in the terminal UI without emitting a transcript assistant message.

## Data Storage

All state files live in `$CCGRAM_DIR` (`~/.ccgram/` by default):

| File                 | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `state.json`         | Thread bindings, window states, display names, read offsets |
| `session_map.json`   | Hook-generated window → session mappings                    |
| `events.jsonl`       | Append-only hook event log (read incrementally by monitor)  |
| `monitor_state.json` | Byte offsets per session (prevents duplicate notifications) |

Session transcripts are read from provider-specific locations (read-only): `~/.claude/projects/` (Claude), `~/.codex/sessions/` (Codex), `~/.gemini/tmp/` (Gemini). The bot never writes to agent data directories.

## Running as a Service

For persistent operation, run ccgram as a systemd service or under a process manager:

```bash
# systemd user service (~/.config/systemd/user/ccgram.service)
[Unit]
Description=CCGram - Command & Control Bot for AI coding agents
After=network.target

[Service]
ExecStart=%h/.local/bin/ccgram
Restart=on-failure
RestartSec=5
Environment=CCGRAM_DIR=%h/.ccgram

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable ccgram
systemctl --user start ccgram
```

On macOS, you can use a launchd plist or simply run in a detached tmux session:

```bash
tmux new-session -d -s ccgram-daemon 'ccgram'
```
