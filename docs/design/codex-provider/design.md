# Codex Provider

## Functional Responsibilities

- Implement `AgentProvider` protocol for Codex CLI (resume, continue, JSONL transcripts, no hooks)
- Parse Codex JSONL transcript format for message extraction and history display
- Build Codex-specific status snapshots: token stats, session metadata, transcript info (via `build_status_snapshot()` protocol method)
- Check for assistant output since a given transcript offset (via `has_output_since()` protocol method)
- Format Codex interactive prompts for Telegram readability: edit approval diffs, numbered options, token consumption stats
- Provide Codex-specific launch arguments and command metadata

## Encapsulated Knowledge

This module owns all knowledge that no other module should have:

- **Codex JSONL entry types** — `session_meta`, `event_msg`, `response_item`, `token_count`; parsing logic for each entry type
- **Status snapshot construction** — aggregation of token stats, session info, transcript metadata into Telegram-formatted status message
- **Interactive prompt format** — regex patterns for Codex edit approval prompts (side-by-side diffs), numbered options (yes/no/esc), token consumption display, change count formatting
- **Prompt formatting rules** — line-by-line transformation for Telegram readability: diff markers, indentation preservation, option numbering
- **Transcript offset tracking** — `has_output_since(offset)` scans JSONL entries after byte offset for assistant output
- **Codex launch arguments** — `--resume`, `-c` for continue, `--model` for model override, etc.
- **Codex command metadata** — supported commands and their descriptions for Telegram menu registration

## Subdomain Classification

**Supporting** — Codex-specific logic changes when the Codex CLI evolves (transcript format changes, new interactive prompt types, new launch arguments). This is external dependency-driven volatility, not feature-driven — it's reactive, not proactive.

## Integration Contracts

### ← Command Orchestration (depended on by, via protocol)

- **Direction**: Command Orchestration depends on Codex Provider (through Provider Protocol)
- **Contract type**: Contract (optional protocol methods)
- **What is shared**: Status snapshot and output detection
- **Contract definition** (implemented from AgentProvider protocol):

  ```python
  def build_status_snapshot(
      self, window_id: str, transcript_path: str, offset: int
  ) -> str | None:
      """Build Codex status snapshot from JSONL transcript."""
      # Delegates to internal codex_status module
      ...

  def has_output_since(self, transcript_path: str, offset: int) -> bool:
      """Check if Codex has produced assistant output since offset."""
      ...
  ```

  Other providers return `None` / `False` by default. Command Orchestration calls these uniformly without checking provider name.

### ← Polling Subsystem (depended on by, via protocol)

- **Direction**: Polling depends on Codex Provider (through Provider Protocol)
- **Contract type**: Contract (terminal parsing)
- **What is shared**: Terminal status parsing with interactive prompt formatting
- **Contract definition** (implemented from AgentProvider protocol):
  ```python
  def parse_terminal_status(self, lines: list[str]) -> StatusUpdate | None:
      """Parse Codex terminal output. Applies interactive prompt formatting internally."""
      ...
  ```
  The interactive prompt formatter is called internally — consumers never see it.

### → Provider Protocol (implements)

- **Direction**: Codex Provider implements AgentProvider
- **Contract type**: Model (protocol implementation)
- **What is shared**: Full AgentProvider interface
- **Contract definition**: Implements all required protocol methods plus optional `build_status_snapshot()` and `has_output_since()`; `capabilities` returns `ProviderCapabilities(name="codex", has_hooks=False, supports_resume=True, supports_status_snapshot=True, ...)`

### → JSONL Parser (depends on, internal)

- **Direction**: Codex Provider depends on shared JSONL base class
- **Contract type**: Model (inheritance)
- **What is shared**: JSONL file reading, incremental parsing, byte offset tracking
- **Contract definition**: `_jsonl.JsonlTranscriptParser` base class with `read_entries(path, offset)` method

## Change Vectors

These are reasonable future changes that would require ONLY this module to change:

- **Changing Codex JSONL transcript format** — update entry parsing in codex.py; no bot.py or polling changes
- **Adding a new interactive prompt type** — update formatter; `parse_terminal_status()` contract unchanged
- **Adding status snapshot for another provider** — that provider implements `build_status_snapshot()`; no Codex changes needed
- **Changing Codex launch arguments** — update `get_launch_args()`; consumers use provider protocol uniformly
- **Adding new token stat fields** — update snapshot builder; command orchestration displays the string unchanged

## Internal Structure

After the refactoring, this module absorbs two previously top-level files:

| Current location                             | New location                                     | Purpose                                         |
| -------------------------------------------- | ------------------------------------------------ | ----------------------------------------------- |
| `src/ccgram/codex_status.py`                 | `src/ccgram/providers/codex_status.py` (private) | Status snapshot building, output-since checking |
| `src/ccgram/interactive_prompt_formatter.py` | `src/ccgram/providers/codex_format.py` (private) | Interactive prompt formatting for Telegram      |

Both become private implementation details of the Codex provider package — imported only by `providers/codex.py`, invisible to the rest of the system. The public surface is the `AgentProvider` protocol methods.
