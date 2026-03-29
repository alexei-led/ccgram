# Provider Protocol

## Functional Responsibilities

- Define the `AgentProvider` protocol — the uniform interface all CLI backends implement
- Define `ProviderCapabilities` dataclass — declares what each provider supports (hooks, resume, continue, transcripts, status snapshots, etc.)
- Provide provider resolution: `get_provider_for_window(window_id)` resolves the correct provider instance per window
- Provide provider detection: `detect_provider_from_pane()`, `detect_provider_from_command()`, `detect_provider_from_runtime()` for auto-identification
- Manage provider registry (`ProviderRegistry`): name → factory mapping, singleton cache per provider name
- Gate UX features per window based on capabilities (recovery keyboard buttons, hook checks, command menus)

## Encapsulated Knowledge

This module owns all knowledge that no other module should have:

- **Capability matrix** — which provider supports hooks, resume, continue, transcripts, incremental reads, status snapshots, pane title probing, etc.
- **Provider resolution order** — WindowState.provider_name → config default (`CCGRAM_PROVIDER`)
- **Process detection for auto-identification** — basename matching (claude/codex/gemini → provider name), ps-based TTY fallback for JS-runtime-wrapped CLIs, PGID caching for `process_detection.py`
- **Provider singleton lifecycle** — factory cache in `ProviderRegistry`, lazy instantiation
- **Launch command resolution** — `CCGRAM_<NAME>_COMMAND` env var → provider default; shell provider has no override
- **Optional capability method defaults** — `build_status_snapshot()` returns `None` by default; providers opt in by overriding

## Subdomain Classification

**Supporting** — The provider protocol evolves when new providers are added (infrequent) or when existing providers gain new capabilities. The protocol surface is moderately volatile — new optional methods are added periodically — but the core abstraction (uniform interface for CLI backends) is stable.

## Integration Contracts

### ← Polling Subsystem (depended on by)

- **Direction**: Polling depends on Provider Protocol
- **Contract type**: Contract (capability queries + terminal parsing)
- **What is shared**: Provider identity, capability flags, terminal status parsing
- **Contract definition**:

  ```python
  class AgentProvider(Protocol):
      @property
      def capabilities(self) -> ProviderCapabilities: ...
      def parse_terminal_status(self, lines: list[str]) -> StatusUpdate | None: ...

  def get_provider_for_window(window_id: str) -> AgentProvider: ...
  ```

### ← Command Orchestration (depended on by)

- **Direction**: Commands depend on Provider Protocol
- **Contract type**: Contract (capabilities + optional methods)
- **What is shared**: Command metadata, status snapshot building
- **Contract definition**:
  ```python
  class AgentProvider(Protocol):
      def get_command_metadata(self) -> dict: ...
      def build_status_snapshot(
          self, window_id: str, transcript_path: str, offset: int
      ) -> str | None:
          """Build provider-specific status snapshot. Returns None if unsupported."""
          return None  # default implementation
  ```

### ← Topic Orchestration (depended on by)

- **Direction**: Topic creation depends on Provider Protocol
- **Contract type**: Contract (detection functions)
- **What is shared**: Provider identity detection from running processes
- **Contract definition**:
  ```python
  def detect_provider_from_pane(
      pane_current_command: str, pane_tty: str, window_id: str
  ) -> str | None: ...
  def detect_provider_from_runtime(window_id: str) -> str | None: ...
  ```

### → Individual Providers (depends on)

- **Direction**: Registry holds provider implementations
- **Contract type**: Model (protocol implementation)
- **What is shared**: Full provider implementation details
- **Contract definition**: Each provider module (`claude.py`, `codex.py`, `gemini.py`, `shell.py`) implements `AgentProvider` and registers with `ProviderRegistry`

## Change Vectors

These are reasonable future changes that would require ONLY this module to change:

- **Adding a new provider** — implement `AgentProvider`, register with `ProviderRegistry`; no consumer changes
- **Adding a new capability flag** — add to `ProviderCapabilities`; consumers opt in to querying it
- **Changing detection heuristics** — modify `detect_provider_from_pane()` / `detect_provider_from_runtime()`; consumers call the same functions
- **Adding a new optional protocol method** — add with default `None`/no-op return; existing providers unaffected
- **Changing launch command resolution** — only `resolve_launch_command()` changes
