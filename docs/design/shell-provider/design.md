# Shell Provider

## Functional Responsibilities

- Implement `AgentProvider` protocol for shell windows (no hooks, no transcript, shell-specific idle detection)
- Define and inject prompt markers in two modes:
  - **Wrap mode** (default): append dimmed `⌘N⌘` marker after user's existing prompt (preserves Tide/Starship/etc.)
  - **Replace mode** (legacy): replace entire prompt with `{prefix}:N❯`
- Expose typed `PromptMatch` contract for prompt parsing (replaces raw `re.Match`)
- Detect shell idle state via prompt marker presence (`has_prompt_marker()`)
- Manage prompt setup lifecycle: auto-setup (directory browser), ask flow (external bind), lazy recovery (marker lost mid-session)
- Respect user's Skip choice per session (no lazy recovery if user declined setup)

## Encapsulated Knowledge

This module owns all knowledge that no other module should have:

- **Prompt marker regex patterns** — `_WRAP_RE` for wrap mode, `_compile_replace_re()` for replace mode; pattern construction, mode switching
- **PromptMatch semantics** — the mapping from regex capture groups to typed fields:
  ```python
  @dataclass(frozen=True)
  class PromptMatch:
      sequence_number: int   # monotonic counter for output isolation
      trailing_text: str     # command text after marker (empty = bare prompt)
      exit_code: int         # exit code from the sequence number (wrap) or explicit field (replace)
      raw_line: str          # original terminal line for debugging
  ```
- **Prompt injection method** — PS1/PROMPT_COMMAND override for bash/zsh, fish_prompt function for fish; session-scoped, never modifies shell config files
- **Sequence number management** — monotonic counter incremented on each command send, used by `shell_capture.py` to isolate command output
- **Lazy recovery logic** — detect marker loss (exec bash, profile reload), re-inject on next command send unless user chose Skip
- **Prompt mode resolution** — `CCGRAM_PROMPT_MODE` env var → "wrap" default; `CCGRAM_PROMPT_MARKER` prefix for replace mode
- **Setup lifecycle state machine** — auto-setup (explicit shell creation) vs ask flow (external bind / provider switch); Skip respected per session; fresh offer on provider switch back to shell

## Subdomain Classification

**Supporting** — The shell subsystem is actively developed (wrap/replace modes, multi-shell support) but the prompt contract itself is stable once formalized. The `PromptMatch` dataclass eliminates accidental volatility from the implicit regex group contract, leaving only essential volatility (new prompt features).

## Integration Contracts

### ← Shell Capture (depended on by)

- **Direction**: Shell Capture depends on Shell Provider
- **Contract type**: Contract (typed `PromptMatch` interface)
- **What is shared**: Prompt parsing results with named fields
- **Contract definition**:

  ```python
  @dataclass(frozen=True)
  class PromptMatch:
      sequence_number: int
      trailing_text: str
      exit_code: int
      raw_line: str

  def match_prompt(line: str) -> PromptMatch | None:
      """Match a prompt marker in line. Returns typed result or None."""
      ...
  ```

  Consumers access `.sequence_number`, `.trailing_text`, `.exit_code` instead of `.group(1)`, `.group(2)`.

### ← Polling Subsystem (depended on by)

- **Direction**: Polling depends on Shell Provider
- **Contract type**: Contract (idle detection + setup trigger)
- **What is shared**: Marker presence check and setup function
- **Contract definition**:
  - `has_prompt_marker(window_id: str) -> bool` — check if prompt marker is present in recent pane lines
  - `setup_shell_prompt(window_id: str) -> None` — inject prompt marker into shell

### → Tmux Manager (depends on)

- **Direction**: Shell Provider depends on Tmux Manager
- **Contract type**: Contract (pane operations)
- **What is shared**: Pane capture for marker detection, send_keys for marker injection
- **Contract definition**: `capture_pane(window_id) -> str`, `send_keys(window_id, text)`

### → Provider Protocol (implements)

- **Direction**: Shell Provider implements AgentProvider
- **Contract type**: Model (protocol implementation)
- **What is shared**: Full AgentProvider interface
- **Contract definition**: Implements all required protocol methods; `capabilities` returns `ProviderCapabilities(name="shell", has_hooks=False, has_transcript=False, ...)`

## Change Vectors

These are reasonable future changes that would require ONLY this module to change:

- **Adding a new prompt field** (e.g., shell PID, timestamp) — add field to `PromptMatch`, update regex and parsing; consumers that don't use the new field are unaffected
- **Adding a new prompt mode** (e.g., minimal mode) — add regex pattern, update `match_prompt()` mode switch; `PromptMatch` contract unchanged
- **Changing marker injection method** (e.g., different shell integration) — only `setup_shell_prompt()` changes
- **Supporting a new shell** (e.g., nushell, PowerShell) — add injection logic; `PromptMatch` contract unchanged
- **Changing sequence number strategy** — only internal counter management changes; consumers use `PromptMatch.sequence_number` unchanged
