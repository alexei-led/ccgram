# Providers

CCGram supports multiple agent CLI backends. Each Telegram topic can use a different provider — you choose when creating a session via the directory browser.

## Overview

| Provider    | CLI Command | Hook Events | Resume | Continue | Transcript | Status Detection                                                      |
| ----------- | ----------- | ----------- | ------ | -------- | ---------- | --------------------------------------------------------------------- |
| Claude Code | `claude`    | Yes         | Yes    | Yes      | JSONL      | Hook events + pyte VT100 + spinner                                    |
| Codex CLI   | `codex`     | Yes         | Yes    | Yes      | JSONL      | Hook Stop + pyte VT100 interactive UI + transcript activity heuristic |
| Gemini CLI  | `gemini`    | Yes         | Yes    | Yes      | JSONL      | Hook AfterAgent + pane title + interactive UI + `/status` snapshot    |
| Pi          | `pi`        | Yes         | Yes    | Yes      | JSONL (v3) | Hook-runner Stop + transcript activity heuristic                      |
| Antigravity | `agy`       | No          | Yes    | Yes      | JSONL      | Transcript activity heuristic + `/status` snapshot                    |
| Shell       | `bash`      | No          | No     | No       | None       | Shell prompt idle detection                                           |

`Resume` in this table means the CLI accepts a known session ID. CCGram's Telegram Resume picker can enumerate sessions for Claude and Antigravity. Codex, Gemini, and Pi currently expose Fresh and Continue recovery actions only.

## Native command menus

Pi, Codex, and Claude Code declare controls for recognized native screens.
CCGram mirrors the screen in an editable Telegram message. Buttons follow that
screen's keys and scope; numbered rows alone do not create selection shortcuts.

| Provider | Recognized menu families |
| --- | --- |
| Pi | Model, thinking, settings, scoped models, tree/fork, standard extension selectors, inputs and editors |
| Codex | Model/reasoning, permissions, skills/review selectors, statusline, title, theme, keymap, configuration |
| Claude Code | Model, effort, permissions, skills, settings/config, theme, help, empty rewind |

These captures cover Pi 1.4.0, Codex 0.164.0, and Claude Code 2.1.312, not every
command or nested screen. Codex review/skills child activation and keymap editing,
Claude populated rewind and nested editors, and authenticated/private resource
screens remain unverified. Unsupported screens require the terminal.

- **Reply to the menu message** only when it invites text. A reply appends one
  line without Enter; slash-prefixed replies stay literal, including bot command
  names. Root shortcut menus may require **Search** first.
- **Esc follows native behavior:** it can clear a filter, leave search focus,
  return to a parent, or close. Observe the updated screen before the next action.
- **Persistence is native:** Save, Set default, Apply policy, and Change setting
  can write preferences immediately. Esc does not undo saved changes. Session-only
  actions are labelled separately where the native footer offers them.
- Buttons and replies revalidate the displayed prompt and owning pane. Old
  controls expire after a transition or bot restart. Tap **🔄** after a failed
  display update. Advisory detections cannot send unverified input.
- Ordinary messages reach the agent only after a blocking prompt is confirmed
  closed; otherwise the message is not forwarded. Permission/question prompts
  retain their separate reply behavior and do not become command menus.

Captured footers establish available controls, not proof that every save/apply
action was executed. Use default CLI keybindings. Finish passwords, API keys,
OAuth and other authentication in the terminal.

## Transcript Delivery Guarantees

Structured transcripts mix conversation with provider instructions, API-request snapshots, and other internal records. CCGram accepts only supported conversation envelopes for each provider before building history or live output. Internal and unknown records are not promoted to chat based on a nested role or text field. The same classification applies to history, live updates, resume and completion summaries, and task status. Visible tool results, configured thinking, local-command notices, and provider errors retain their existing presentation. Shell command output uses its separate explicit terminal path.

For transcript-backed providers, CCGram delivers parsed output through an ordered queue. Consecutive eligible text items may be losslessly combined into one Telegram message only within the same chat, topic/thread, window, role, and transcript source session. Tool-use batching (`/verbose`), attachments/TTS, status updates, and any other queue boundary remain separate. Each original text item is rendered before batching, so its Telegram formatting is preserved and cannot leak into the next item.

Delivery is at-least-once. CCGram persists a transcript's delivered watermark only after its queued delivery receipts succeed (or are intentionally dropped by a confirmed Jump to live). After a process restart or an unconfirmed Telegram result, it can replay output and therefore show a duplicate rather than risk losing output. Provider transcript files stay read-only; a confirmed Jump to live records a durable barrier and a visible skipped-range notice before advancing past that range. See [Delivery, Backlog, and Jump to Live](guides.md#delivery-backlog-and-jump-to-live) for queue metrics, severe thresholds, and the action's exact boundaries.

## Choosing a Provider

**From Telegram**: Type the session line in a fresh topic:

```text
<agent>[:<mode>] <project|path> [@branch] <instruction...>
```

`codex ccgram fix the hook timeout` starts Codex on the `ccgram` checkout with
that instruction, in one message. Omit the agent word to use
`CCGRAM_PROVIDER`. See
[Starting a Session](guides.md#starting-a-session-one-typed-line) for the
grammar, project resolution, and every default.

The typed line never opens a provider picker: an omitted agent word uses
`CCGRAM_PROVIDER`, and a provider that is disabled or whose CLI is not
installed fails the launch with a one-line reason. A provider that supports no
permissive flag (Shell, Pi) always launches in normal mode, whatever
`CCGRAM_DEFAULT_MODE` says. The default provider list — also used by the
directory browser's picker — is Antigravity, Claude, Codex, Pi, and Shell;
Gemini is opt-in via `CCGRAM_PROVIDERS_ENABLED=claude,codex,gemini,pi,shell,antigravity`.

**From the terminal**: If you create a window manually and start an agent CLI, CCGram auto-detects the provider from the running process name. When the pane command is a JS runtime wrapper (node, bun), it inspects the pane's foreground process to reliably identify the actual CLI. Evidence is limited to the executable name and, for wrapper commands, the script path: helper data paths, inline `-e` code, and flag values never name a provider, so a helper that mentions another CLI's directory cannot override the running one. How the foreground process is read is owned by the multiplexer backend — tmux uses `ps -t <tty>`, herdr reads `pane process-info` (no tty needed), and agterm reports foreground argv (no tty or process-group ID) — so detection works through the same seam on all three. The shell provider uses the same seam to classify a bare shell pane. Agterm reports `foregroundShell` (and `splitForegroundShell` for a split peer) for a known shell, so the Terminal provider is available there. This does not prove the shell is at a prompt: a builtin such as `read` is invisible in argv, so agterm disables shell prompt-marker setup rather than risk sending setup commands while foreground state is unknown. As a last resort, Gemini pane-title symbols (`✦`, `✋`, `◇`) are checked.

**Default provider**: Set `CCGRAM_PROVIDER=codex` (or `gemini`, `pi`, `antigravity`, `shell`) to change the default. Claude is the default if unset.

## Session Mode (Standard vs YOLO)

Sessions launch in YOLO mode by default (`CCGRAM_DEFAULT_MODE=yolo`); there is
no mode picker in the normal flow. Override per launch with the `:safe` suffix
(`claude:safe ccgram …`), or globally with `CCGRAM_DEFAULT_MODE=normal`.
CCGram stores mode per window and reuses it for recover/continue/resume flows.

- `normal` mode launches the provider command as-is.
- `yolo` mode appends the provider-native permissive flag:
  - Claude: `--dangerously-skip-permissions`
  - Codex: `--dangerously-bypass-approvals-and-sandbox`
  - Gemini: `--yolo`
  - Antigravity: `--dangerously-skip-permissions`

YOLO sessions are indicated in Telegram topic titles with a `🚀` badge and in `/sessions` with a `[YOLO]` tag. When Remote Control is active, a `📡` badge also appears in the topic title.

## First-Run Approval Screens

A fresh agent can block before its prompt. CCGram clears both known cases
during launch when `CCGRAM_AUTOSTART_TRUST` is on (the default):

| Provider | Screen | How CCGram clears it |
| --- | --- | --- |
| Claude | "Do you trust the files in this folder?" | Seeds `hasTrustDialogAccepted` for the project in the profile the launch reads — `$CLAUDE_CONFIG_DIR/.claude.json` when that is set, otherwise `~/.claude.json`; falls back to one `Enter` on the exact dialog text. |
| Codex | Hooks review: "trust all and continue" | Seeds `[projects."<path>"]` with `trust_level = "trusted"` in `$CODEX_HOME/config.toml`; falls back to `Down`+`Enter` on the exact hooks prompt. |
| Pi, Antigravity, Shell | none known | No recipe declared: a blocked screen is reported, never guessed at. |

Seeding is preferred: it removes the screen rather than answering it, so it
costs no keystrokes and no timing window. Turn the layer off globally with
`CCGRAM_AUTOSTART_TRUST=false`, or per agent with
`CCGRAM_<NAME>_AUTOSTART=off`.

## Custom Launch Commands

Override the CLI command used to launch each provider via `CCGRAM_<NAME>_COMMAND` env vars:

```ini
CCGRAM_CLAUDE_COMMAND=ce --current
CCGRAM_CODEX_COMMAND=my-codex-wrapper
CCGRAM_GEMINI_COMMAND=/opt/gemini/run
CCGRAM_PI_COMMAND=pi --model sonnet
CCGRAM_ANTIGRAVITY_COMMAND=agy --effort high
```

`<NAME>` is uppercase: `CLAUDE`, `CODEX`, `GEMINI`, `PI`, `ANTIGRAVITY`. Defaults to the provider's built-in command (`claude`, `codex`, `gemini`, `pi`, `agy`) when unset. New providers automatically support `CCGRAM_<NAME>_COMMAND` without code changes.

A custom launch script must replace itself with the recognizable provider CLI. For example, `my-codex-wrapper` can contain:

```sh
#!/bin/sh
exec codex "$@"
```

A long-running intermediary shell or an unrecognized wrapper is not foreground evidence for its child agent. Such launches remain quarantined; configure the CLI directly or use an `exec` wrapper. CCGram does not send input based only on a wrapper name or a stored hook identity.

You can use this for a global "today" setup (all new sessions), for example:

```ini
CCGRAM_CLAUDE_COMMAND=claude --dangerously-skip-permissions
CCGRAM_CODEX_COMMAND=codex --dangerously-bypass-approvals-and-sandbox
CCGRAM_GEMINI_COMMAND=gemini --yolo
CCGRAM_ANTIGRAVITY_COMMAND=agy --dangerously-skip-permissions
```

## Provider Commands in Telegram

Each bound topic's pinned `/commands` panel includes the commands discovered for that topic's provider. Agent command buttons preserve provider-native names. For example, a custom Claude command named `spec:work` stays `/spec:work` in the panel and is sent to Claude unchanged. It is not rewritten to a Telegram-friendly spelling.

The same panel includes CCGram actions that fit the attached session. Shell topics omit agent commands and transcript-only actions. Provider command discovery remains available through the existing provider adapters and custom-command sources.

Telegram's native slash suggestion list cannot vary by forum topic. CCGram therefore keeps `/commands`, `/sessions`, `/sync`, and `/upgrade` in the shared `/` menu. Use the topic panel for provider commands. See [Topic Command Panels](guides.md#topic-command-panels-commands) for screenshots and behavior.

---

## Claude Code

Claude Code has the richest integration — hook events (SessionStart, Notification, Stop, StopFailure, SessionEnd, SubagentStart, SubagentStop, TeammateIdle, TaskCompleted) provide instant session tracking, interactive UI detection, done/idle detection, API error alerting, session lifecycle cleanup, subagent activity monitoring, and agent team notifications.

The bot also detects Remote Control mode (📡 topic badge + one-tap activation button). Claude's `/remote-control` reports no outcome, so ccgram captures the pane ~1.5 s after RC is triggered (re-scanning every 1.5 s up to 10 s), classifies the result — success with sharing URL, success without URL, unavailable, or failed — and posts a single status reply in the topic. This is Claude-only; other providers keep their existing "not supported" reply. The bot uses a pyte VT100 screen buffer as fallback for terminal status parsing. Multi-pane windows (e.g. from agent teams) are automatically scanned for blocked panes and surfaced as inline keyboard alerts.

### Hooks

Install hooks with `ccgram hook --install`.

If hooks are missing, ccgram warns at startup with the fix command. Hooks are optional — terminal scraping works as fallback.

### Claude Transcript

Claude transcripts are JSONL files under `~/.claude/projects/`. They are read incrementally (byte offsets) for efficient polling.

A newly started Claude session can create its transcript only after the first input. Until that first file appears, history is empty, but input remains available when the foreground process and prompt can be verified. Once a transcript has been observed, its disappearance or a read failure still makes the source unavailable; this startup exception never supplies foreground evidence.

When SessionStart provides the exact path for a not-yet-created Claude transcript, the monitor reserves offset zero so a reply arriving before the next poll is not skipped. Existing Claude transcripts are not rewound.

### Task Lists

Claude task state is derived from the transcript, not from terminal footer scraping. CCGram recognizes the structured `TaskCreate`, `TaskUpdate`, and `TaskList` tool flow, plus legacy `TodoWrite`, and renders the current tasks inside the topic's single editable status bubble. Hook notifications are used to refresh wait-state headers such as waiting for input or approval prompts faster, but they do not replace the transcript as the source of truth.

## Codex CLI

Codex CLI supports feature-flagged hooks. Install ccgram's lifecycle hooks with `ccgram hook --provider codex --install`; ccgram writes user-level `~/.codex/hooks.json` entries for `SessionStart` and `Stop` and enables `[features].hooks = true` in `~/.codex/config.toml`. Transcript discovery remains as fallback and as the source of message truth.

### Interactive Prompts

Codex interactive prompts (question lists, permission prompts, and other selection UIs) are detected from terminal screen content via pyte and shown with inline keyboard controls.

### Edit Approval Formatting

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

### Status Fallback

For Codex, `/status` sends a transcript-based fallback snapshot in Telegram (session/cwd/token/rate-limit summary) because some Codex builds render status in the terminal UI without emitting a transcript assistant message.

### Codex Transcript

Codex transcripts are JSONL files under `~/.codex/sessions/`. They are read incrementally (byte offsets).

## Gemini CLI

Gemini CLI supports command hooks in `settings.json`. Install ccgram's lifecycle hooks with `ccgram hook --provider gemini --install`; ccgram writes user-level `~/.gemini/settings.json` entries for `SessionStart`, `AfterAgent`, `SessionEnd`, and `Notification`. Transcript discovery remains as fallback and as the source of message truth.

Gemini sets pane titles (`Working: ✦`, `Action Required: ✋`, `Ready: ◇`) that CCGram reads for status, and its `@inquirer/select` permission prompts are detected as interactive UI. Gemini transcript discovery matches project hash/alias only (no cross-project full scan) to avoid wrong-session attachment.

### Launch Hardening

For ccgram-managed Gemini launches, CCGram injects `GEMINI_CLI_SYSTEM_SETTINGS_PATH=~/.ccgram/gemini-system-settings.json` with `tools.shell.enableInteractiveShell=false` to avoid node-pty `EBADF` crashes in tmux. If you set `CCGRAM_GEMINI_COMMAND`, your override is used as-is.

### Gemini Transcript

As of Gemini CLI v0.40+, transcripts are append-only JSONL files under `~/.gemini/tmp/<project-hash>/chats/`. Each line is a JSON record (header with `sessionId`/`projectHash`/`startTime`, then message records and `{"$set": {...}}` metadata updates). CCGram reads them incrementally via the shared `JsonlProvider` byte-offset reader and dedupes repeated message ids and pending tool_use ids — a single tool announcement, then one tool_result on the update that carries the result.

Older `session-*.json` whole-file transcripts are no longer monitored; only `.jsonl` files are picked up.

### Status Snapshot

Gemini supports `/status` snapshots: CCGram parses recent transcript activity to render an inline summary of the current session (last activity, pending tools, tool_use counts) without waiting for the next pane refresh.

## Pi

[Pi](https://pi.dev) is a Node.js-based CLI with JSONL v3 transcripts. Install or check CCGram's Pi command hooks with:

```bash
ccgram hook --provider pi --install
ccgram hook --provider pi --status
```

CCGram writes `SessionStart`, `Stop`, `SessionEnd`, and `Notification` entries to `~/.pi/agent/hooks.json`. Set `PI_CODING_AGENT_DIR` to use another Pi agent directory for both hook configuration and transcript discovery. Existing top-level event maps are preserved; if an object-valued `hooks` map exists, it takes precedence, matching hook-runner behavior.

Pi must load the existing `cc-thingz` `hook-runner` extension. CCGram installs hook commands but does not install or replace the extension. The runner caches hook configuration and does not watch the file. Restart Pi or reload the extension after changes. Without the runner, session tracking falls back to scanning transcripts under `~/.pi/agent/sessions/`.

### Launch

The default command is `pi`. Override via `CCGRAM_PI_COMMAND` to change models, flags, or wrappers.

### Resume

Resume always uses `--session <path-or-uuid>`. Pi's `--resume` flag opens an interactive picker ccgram can't drive over `send_keys`, so ccgram always passes the resolved transcript path (or UUID prefix) directly. Pi's own `--continue` is used for the Continue recovery button.

### Pi Transcript

Pi transcripts are JSONL files (v3 format) under `~/.pi/agent/sessions/--<encoded-cwd>--/<timestamp>_<uuid>.jsonl`. The canonical session id lives in the header line (`{"type":"session","id":"<uuid>","cwd":"...","version":3}`) — the filename prefix is just a timestamp. Transcripts are read incrementally via byte offsets.

### Commands

Pi exposes a Telegram-safe command list based on the [Pi usage docs](https://pi.dev/docs/latest/usage): `/changelog`, `/clone`, `/compact`, `/copy`, `/export`, `/fork`, `/hotkeys`, `/login`, `/logout`, `/model`, `/name`, `/new`, `/quit`, `/reload`, `/scoped_models` (native `/scoped-models`), `/session`, `/settings`, `/share`, `/thinking`, and `/tree`. `/followup <message>` is ccgram's Pi-only bridge to Pi's Alt+Enter behavior: it queues the message after current work finishes instead of steering the active turn. `/clear` is accepted as a hidden compatibility alias for Pi `/new`, but it is not advertised because Pi does not define `/clear`. Pi `/resume` is not advertised because it collides with ccgram's bot-native session picker. Dynamic discovery surfaces three more sources:

- **Skills** — `SKILL.md` under `~/.pi/agent/skills/<name>/`, `~/.agents/skills/<name>/`, `<project>/.pi/skills/<name>/`, or `<project>/.agents/skills/<name>/`. Loose `.md` files at the root of `~/.pi/agent/skills/` or `<project>/.pi/skills/` are also picked up.
- **Prompt templates** — `.md` files under `~/.pi/agent/prompts/` or `<project>/.pi/prompts/` (per-project walk stops at the first `.git` ancestor).
- **Extension commands** — TypeScript/JavaScript files (`.ts`, `.js`, `.mjs`, `.cjs`) under `~/.pi/agent/extensions/` or `<project>/.pi/extensions/`, scanned for `pi.registerCommand("name", ...)` calls. The walker prunes `node_modules`, `dist`, `build`, and `.git` before descent.

Names collide-dedupe with first-source wins (skills > prompts > extensions).

### Status Detection

With hook-runner installed, Pi `Stop` hooks mark the topic ready immediately. Without hooks, status is inferred from transcript activity — idle when the latest assistant message has no pending tool calls, working when there are unreturned tool uses.

### Interactive commands

CCGram mirrors Pi's live terminal menu into one editable Telegram message.
This covers `/model`, `/thinking`, `/settings` and its submenus,
`/scoped_models`, `/fork`, `/tree`,
authentication selectors, and extension commands that use Pi's standard
selector, settings, input, or editor components. Detection follows the displayed UI, not a
command-name allowlist; an extension's `/modes` menu works the same way.

| Task | Telegram control |
| --- | --- |
| Navigate a long list | ↑ / ↓; ↑5 / ↓5 sends five arrows |
| Select or change a value | Enter; Space where Pi supports it |
| Search or type | **Reply to the menu message** with one line; no Enter is sent |
| Clear or correct search | Clear search / ⌫ |
| Add a line in an extension editor | New line, then reply with the next line |
| Change model scope | Enter toggles; All matching / None matching / Provider |
| Reorder scoped models | Move ↑ / Move ↓ |
| Persist a model/thinking default or scope | Save default / Save scope (Pi's Ctrl+S) |
| Go back or close | Esc; a parent settings menu stays visible |
| Capture again | 🔄 |

Replies append to the current input; clear search before replacing it.
Slash-prefixed replies, including bot command names, are literal menu input too.
Ordinary messages still dismiss a blocking prompt before reaching the agent.
Replies to an expired menu are rejected, not forwarded as agent messages.
Buttons and replies target the owning pane and revalidate the captured prompt
before sending input. If Telegram cannot update the display, tap 🔄 to recapture
it; old action buttons stay rejected. Restarting the bot expires existing menu controls.

Only visible terminal rows are mirrored; navigation scrolls Pi itself. Pi's
SelectList and SettingsList do not use PageUp/PageDown, so the five-arrow
buttons work across both. Selection and persistence follow Pi's semantics:
changing settings can take effect immediately; Esc is not an undo operation.

Use default Pi keybindings for these controls. Nonstandard custom extension
screens or rebound keys may require `/screenshot` plus `/toolbar`, or the
terminal itself. Do not send API keys, passwords, or OAuth codes through
Telegram; finish authentication in the terminal.

### Toolbar

Pi's default toolbar omits Mode/Think/YOLO and adds a dedicated navigation row.
The automatic menu above is the primary control; the toolbar remains a fallback:

- Row 1: `📷 Screen, ⏹ Ctrl-C, 📺 Live`
- Row 2: `⎋ Esc, ⇥ Tab, π Model`
- Row 3: `🔼 Up, ⏎ Enter, 🔽 Down, 📤 Send, ✖ Close`

Override with a `[providers.pi]` block in `~/.ccgram/toolbar.toml`.

## omo

[omo](https://github.com/) is an omo-ai distribution built on
`@code-yeongyu/senpi`, a Pi fork. Its sessions, TUI parser and Telegram command
list are shared with Pi, so CCGram serves both through the same code path.

Unlike Pi, omo ships no `cc-thingz` hook runner: CCGram marks the provider as
hook-less (`ccgram doctor` reports "hook check skipped") and relies on transcript
scanning only. State lives under `~/.omo/agent`; the agent directory is resolved
from `OMO_CODING_AGENT_DIR`, then `SENPI_CODING_AGENT_DIR`, then
`PI_CODING_AGENT_DIR`, matching omo's own resolution order.

### Launch

The default command is `omo`; override it with `CCGRAM_OMO_COMMAND`. YOLO mode
appends `--permission-preset full-access`, which omo needs to run unattended -
without it the agent waits on approval prompts Telegram cannot answer.

### Detection

The omo launcher is a JS entry point, so tmux reports `bun` as the pane command.
CCGram therefore asks for the pane title on `bun`/`node` panes and matches omo's
window title (`OmO - <project>`); a process literally named `omo`, and the
`@code-yeongyu/senpi` path markers used by the agterm backend, are recognised as
well.

### Transcript

`~/.omo/agent/sessions/--<encoded-cwd>--/<timestamp>_<uuid>.jsonl`, with the same
JSONL v3 header as Pi. Resume uses `--session <path-or-uuid>`, and `--continue`
serves the Continue button.

### Model selection

`/model` in omo prints the model list into the transcript instead of opening a
modal selector, so it is not advertised as a TUI picker; pin the model in the
launch command when a specific one is required.

## Shell

The shell provider opens a plain shell session in tmux. It has no hooks, no transcript, and no resume/continue support — shell sessions are ephemeral.

Text messages are sent through an LLM to generate shell commands; prefix with `!` for raw commands. When no LLM is configured, all text is forwarded as raw commands.

### LLM Configuration

Configure an LLM provider to enable natural language to shell command generation.

| Setting         | Env Var                  | Default           |
| --------------- | ------------------------ | ----------------- |
| LLM provider    | `CCGRAM_LLM_PROVIDER`    | _(empty)_         |
| LLM API key     | `CCGRAM_LLM_API_KEY`     | _(empty)_         |
| LLM base URL    | `CCGRAM_LLM_BASE_URL`    | _(from provider)_ |
| LLM model       | `CCGRAM_LLM_MODEL`       | _(from provider)_ |
| LLM temperature | `CCGRAM_LLM_TEMPERATURE` | `0.1`             |

API key resolution: `CCGRAM_LLM_API_KEY` > provider-specific env var (e.g. `XAI_API_KEY`) > `OPENAI_API_KEY` (universal fallback).

Set temperature to `0` for deterministic output with cheap/fast models.

#### Supported LLM Providers

**OpenAI** (default model: `gpt-5.4-nano`):

```bash
CCGRAM_LLM_PROVIDER=openai
# Uses OPENAI_API_KEY by default — no extra key needed
```

**x.ai / Grok** (default model: `grok-3-fast`):

```bash
CCGRAM_LLM_PROVIDER=xai
XAI_API_KEY=xai-...              # or set OPENAI_API_KEY as fallback
```

**DeepSeek** (default model: `deepseek-chat`):

```bash
CCGRAM_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...          # or set OPENAI_API_KEY as fallback
```

**Anthropic** (default model: `claude-sonnet-4-20250514`):

```bash
CCGRAM_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...     # or set OPENAI_API_KEY as fallback
```

**Groq** (default model: `llama-3.3-70b-versatile`):

```bash
CCGRAM_LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...             # or set OPENAI_API_KEY as fallback
```

**Ollama** (default model: `llama3.1`, no API key needed):

```bash
CCGRAM_LLM_PROVIDER=ollama
CCGRAM_LLM_BASE_URL=http://localhost:11434/v1
```

### Command Generation Flow

1. Send a text message describing what you want (e.g., "list all Python files")
2. The LLM generates a shell command (e.g., `find . -name "*.py"`)
3. An approval keyboard appears: **▶ Run** | **✏ Edit** | **✕ Cancel**
4. Tap **Run** to execute, **Edit** to copy and modify, or **Cancel** to discard
5. Dangerous commands (`rm -rf`, `dd`, etc.) show an extra confirmation step

### Raw Commands

Prefix with `!` to bypass LLM and send directly to the shell:

- `!ls -la` → sends `ls -la` directly
- `! git status` → sends `git status` (leading space stripped)

### Voice Messages

Voice messages in shell topics flow through Whisper transcription → LLM command generation → approval keyboard automatically.

### Shell Status

- Idle at prompt: "🐚 Shell ready" (or "✓ Ready" with standard status)
- `/history` is not available (no transcript)
- Resume and Continue are not supported (shell sessions are ephemeral)

## Google Antigravity CLI (agy)

Google Antigravity CLI is a command-line AI coding agent surface with directory-scoped sessions, JSONL transcript logs (`transcript.jsonl`), and built-in slash commands.

### Supported Platforms & Executable Resolution

CCGram supports Antigravity CLI across macOS (Apple Silicon arm64 & Intel x86_64) and Linux via a deterministic resolution order:

1. **Environment Override**: `CCGRAM_ANTIGRAVITY_COMMAND` (e.g. `CCGRAM_ANTIGRAVITY_COMMAND="agy --effort high"`)
2. **PATH Lookup**: Executables named `agy` or `antigravity` resolved via system `PATH`
3. **Platform Candidates**:
   - `~/.local/bin/agy`
   - `~/.gemini/antigravity-cli/bin/agy`
   - `~/.antigravity/bin/agy`
   - `/usr/local/bin/agy`
   - `/opt/homebrew/bin/agy`
   - `/usr/bin/agy`

### Transcript & Data Directory Resolution

Transcript brain directories are discovered in order:

1. `CCGRAM_ANTIGRAVITY_DATA_DIR` environment override
2. `~/.gemini/antigravity-cli/brain`
3. `~/.antigravity/brain`
4. `~/.config/antigravity/brain`
5. `~/.local/share/antigravity/brain`

CCGram parses structured workspace fields and local `file://` URIs, decodes percent escapes, and compares normalized paths (`Path.expanduser().resolve()`) for exact equality. Sibling and descendant paths never match the topic's target working directory (`cwd`).

### Herdr integration

With `CCGRAM_MULTIPLEXER=herdr`, install Herdr's native Antigravity integration before launching `agy`:

```bash
herdr integration install antigravity-cli
```

Herdr reports the real conversation ID after the first prompt creates it. CCGram binds the topic only after that report; if no report arrives, it closes the new tab rather than storing a transient pane ID. Restart already-running `agy` sessions after installing the integration.

### Resume and Continue

- **Resume**: Uses `agy --conversation <session_id>` (plus `--dangerously-skip-permissions` in YOLO mode).
- **Continue**: Uses `agy --continue` (plus `--dangerously-skip-permissions` in YOLO mode).

### Troubleshooting Guidance

- **Binary Not Found**: Ensure `agy` is in your `$PATH` or set `CCGRAM_ANTIGRAVITY_COMMAND=/path/to/agy`.
- **Session Discovery Failure**: Check that transcripts are created under `~/.gemini/antigravity-cli/brain/` or set `CCGRAM_ANTIGRAVITY_DATA_DIR=/path/to/brain`.
