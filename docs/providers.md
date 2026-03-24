# Providers

CCGram supports multiple agent CLI backends. Each Telegram topic can use a different provider — you choose when creating a session via the directory browser.

## Overview

| Provider    | CLI Command | Hook Events | Resume | Continue | Transcript | Status Detection                                          |
| ----------- | ----------- | ----------- | ------ | -------- | ---------- | --------------------------------------------------------- |
| Claude Code | `claude`    | Yes         | Yes    | Yes      | JSONL      | Hook events + pyte VT100 + spinner                        |
| Codex CLI   | `codex`     | No          | Yes    | Yes      | JSONL      | pyte VT100 interactive UI + transcript activity heuristic |
| Gemini CLI  | `gemini`    | No          | Yes    | Yes      | JSON       | Pane title + interactive UI                               |
| Shell       | `bash`      | No          | No     | No       | None       | Shell prompt idle detection                               |

## Choosing a Provider

**From Telegram**: When you create a new topic and select a directory, a provider picker appears with Claude (default), Codex, Gemini, and Shell options. After provider selection, CCGram asks for session mode:

- `✅ Standard` (normal approvals)
- `🚀 YOLO` (provider-specific permissive mode)

**From the terminal**: If you create a tmux window manually and start an agent CLI, CCGram auto-detects the provider from the running process name. When the pane command is a JS runtime wrapper (node, bun), it falls back to `ps -t` foreground process inspection to reliably identify the actual CLI. As a last resort, Gemini pane-title symbols (`✦`, `✋`, `◇`) are checked.

**Default provider**: Set `CCGRAM_PROVIDER=codex` (or `gemini`, `shell`) to change the default. Claude is the default if unset.

## Session Mode (Standard vs YOLO)

CCGram stores mode per window and reuses it for recover/continue/resume flows.

- `normal` mode launches the provider command as-is.
- `yolo` mode appends the provider-native permissive flag:
  - Claude: `--dangerously-skip-permissions`
  - Codex: `--dangerously-bypass-approvals-and-sandbox`
  - Gemini: `--yolo`

YOLO sessions are indicated in Telegram topic titles with a `🚀` badge and in `/sessions` with a `[YOLO]` tag. When Remote Control is active, a `📡` badge also appears in the topic title.

## Custom Launch Commands

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

## Provider-Specific Commands

Each provider exposes its own slash commands to the Telegram menu. Examples:

- **Claude**: `/clear`, `/compact`, `/cost`, `/doctor`, `/permissions`...
- **Codex**: `/model`, `/mode`, `/status`, `/diff`, `/compact`, `/mcp`...
- **Gemini**: `/chat`, `/clear`, `/compress`, `/model`, `/memory`, `/vim`...

---

## Claude Code

Claude Code has the richest integration — hook events (SessionStart, Notification, Stop, StopFailure, SessionEnd, SubagentStart, SubagentStop, TeammateIdle, TaskCompleted) provide instant session tracking, interactive UI detection, done/idle detection, API error alerting, session lifecycle cleanup, subagent activity monitoring, and agent team notifications.

The bot also detects Remote Control mode (📡 topic badge + one-tap activation button) and uses a pyte VT100 screen buffer as fallback for terminal status parsing. Multi-pane windows (e.g. from agent teams) are automatically scanned for blocked panes and surfaced as inline keyboard alerts.

### Hooks

Install hooks with `ccgram hook --install`. This is Claude-specific and not needed for other providers.

If hooks are missing, ccgram warns at startup with the fix command. Hooks are optional — terminal scraping works as fallback.

### Transcript

Claude transcripts are JSONL files under `~/.claude/projects/`. They are read incrementally (byte offsets) for efficient polling.

## Codex CLI

Codex CLI lacks a session hook, so session tracking relies on hookless transcript discovery plus provider detection from the running process name.

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

### Transcript

Codex transcripts are JSONL files under `~/.codex/sessions/`. They are read incrementally (byte offsets).

## Gemini CLI

Gemini CLI lacks a session hook. Session tracking relies on hookless transcript discovery plus provider detection.

Gemini sets pane titles (`Working: ✦`, `Action Required: ✋`, `Ready: ◇`) that CCGram reads for status, and its `@inquirer/select` permission prompts are detected as interactive UI. Gemini transcript discovery matches project hash/alias only (no cross-project full scan) to avoid wrong-session attachment.

### Launch Hardening

For ccgram-managed Gemini launches, CCGram injects `GEMINI_CLI_SYSTEM_SETTINGS_PATH=~/.ccgram/gemini-system-settings.json` with `tools.shell.enableInteractiveShell=false` to avoid node-pty `EBADF` crashes in tmux. If you set `CCGRAM_GEMINI_COMMAND`, your override is used as-is.

### Transcript

Gemini transcripts are JSON files (whole-file read, not incremental) under `~/.gemini/tmp/`.

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
