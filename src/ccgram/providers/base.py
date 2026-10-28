"""Provider protocol and shared event types for multi-agent CLI backends.

Pure definitions only — no imports from existing ccgram modules to avoid
circular dependencies. Every agent provider (Claude, Codex, Gemini) must
satisfy the ``AgentProvider`` protocol.

Event types:
  - SessionStartEvent: emitted when a new session is detected
  - AgentMessage: a parsed message from the agent's transcript
  - StatusUpdate: a parsed terminal status line

Capability descriptor:
  - ProviderCapabilities: declares what features the provider supports
"""

import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

# ── Type aliases for AgentMessage fields ─────────────────────────────────
MessageRole = Literal["user", "assistant"]
ContentType = Literal["text", "thinking", "tool_use", "tool_result", "local_command"]

# ── Shared validation ────────────────────────────────────────────────────
# Alphanumeric + hyphens/underscores — rejects shell metacharacters.
RESUME_ID_RE = re.compile(r"^[\w-]+$")

# Strict UUID v4 form for Claude session ids (validated in hook payloads + resume).
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# ── Event types ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SessionStartEvent:
    """Emitted when a provider session starts or is detected via hook."""

    session_id: str  # provider-specific session identifier (UUID for Claude)
    cwd: str  # absolute path to the project directory
    transcript_path: str  # path to the session's transcript file
    window_key: str  # tmux key, e.g. "ccgram:@0"


@dataclass(frozen=True, slots=True)
class ResumableSession:
    """A provider-owned session that can be resumed in its workspace."""

    session_id: str
    summary: str
    cwd: str
    provider_name: str
    mtime: float = 0.0
    msg_count: int | None = None


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """A single parsed message from the agent's transcript."""

    text: str
    role: MessageRole
    content_type: ContentType
    is_complete: bool = True
    phase: str | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class StatusUpdate:
    """Parsed terminal status line from the agent's pane.

    Two modes depending on ``is_interactive``:
      - **Normal status** (is_interactive=False): ``raw_text`` is the text
        after the spinner, ``display_label`` is a short formatted label
        like "…reading".
      - **Interactive UI** (is_interactive=True): ``raw_text`` is the full
        interactive pane content, ``display_label`` is the UI type name
        (same as ``ui_type``), e.g. "AskUserQuestion".
    """

    raw_text: str
    display_label: str
    is_interactive: bool = False
    ui_type: str | None = None  # "AskUserQuestion", "ExitPlanMode", etc.
    # True when the detection is a structural guess (the catch-all
    # selection shape) rather than a named pattern or action-hint
    # anchor: the keyboard may be shown, but it must not latch the
    # blocking interactive mode.
    ui_advisory: bool = False


@dataclass(frozen=True, slots=True)
class DiscoveredCommand:
    """A command/skill discovered by a provider."""

    name: str  # Original name (e.g. "spec:work", "committing-code")
    description: str
    source: Literal["builtin", "skill", "command"]


# ── Hook events ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HookEvent:
    """A structured event from the hook event log."""

    event_type: str  # "Notification", "Stop", etc.
    window_key: str  # "ccgram:@0"
    session_id: str
    data: dict[str, Any]
    timestamp: float


# ── Capabilities ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FirstRunPrompt:
    """A blocking first-run screen a launch has to clear without a human.

    ``needles`` must ALL appear (case-insensitive) in the captured pane
    before ``keys`` are sent; ``settled`` is a substring that proves the
    prompt is gone. Matching is exact-substring on purpose: a fuzzy match
    would send keys into an unrelated screen.
    """

    needles: tuple[str, ...]
    keys: tuple[str, ...]
    settled: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Declares what features a provider supports.

    Immutable after construction — providers return a fixed instance.
    """

    name: str  # e.g. "claude", "codex", "gemini"
    launch_command: str  # e.g. "claude", "codex"
    supports_hook: bool = False
    hook_install_managed_by_ccgram: bool = False
    supports_resume: bool = False
    supports_resume_picker: bool = False
    supports_continue: bool = False
    supports_structured_transcript: bool = False
    supports_incremental_read: bool = True  # False → whole-file JSON (e.g. Gemini)
    uses_pane_title: bool = False  # Provider reads OSC pane title for status
    # True only for providers whose terminal chrome matches Claude Code's
    # UI_PATTERNS + ─-separator status block, so the shared pyte fast-path
    # (parse_from_screen / parse_status_block_from_screen) is meaningful.
    # When False, the pyte path still populates rendered_text / RC state /
    # vim-insert but does not run Claude pattern extraction, so the
    # provider's own parse_terminal_status is not shadowed (issue #85).
    uses_pyte_status_parsing: bool = False
    builtin_commands: tuple[str, ...] = ()
    # When true, CommandCatalog appends user-defined commands discovered from
    # the configured command sources (currently ~/.claude skills/commands).
    supports_user_command_discovery: bool = False
    supports_status_snapshot: bool = False
    # True for shell-like providers that route inbound text through an LLM
    # → command flow with a prompt-marker terminal loop. Used by handlers to
    # gate shell-specific behavior without checking ``provider_name == "shell"``.
    chat_first_command_path: bool = False
    has_yolo_confirmation: bool = False
    # When true, the provider maintains a per-window task-state model seeded
    # from transcript entries. Callers should invoke seed_task_state() on
    # session start and apply_task_entries() on each new batch.
    supports_task_tracking: bool = False
    # Slash command names (without the leading "/") that open a modal in-TUI
    # picker the user must navigate via arrow keys / Enter / Esc. When a
    # command in this set is forwarded, the topic gets a one-line hint
    # pointing at /toolbar so users know how to drive the picker from
    # Telegram. Empty for providers without modal pickers.
    tui_picker_commands: frozenset[str] = frozenset()
    # First-run approval/trust screen this agent shows before its prompt.
    # None means "no recipe": the launcher then reports the block instead of
    # guessing at keys.
    first_run: FirstRunPrompt | None = None


# ── Provider protocol ────────────────────────────────────────────────────


class AgentProvider(Protocol):
    """Protocol that every agent CLI provider must satisfy.

    Lifecycle: the active provider is resolved once by ``get_provider()``
    and cached in ``providers._active``. The registry creates a fresh
    instance per ``get()`` call. All methods are stateless — they receive
    input and return results without side effects.
    """

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def make_launch_args(
        self,
        resume_id: str | None = None,
        use_continue: bool = False,
    ) -> str:
        """Build CLI args string for launching the agent.

        Returns a string like ``--resume abc123`` or ``--continue``.
        Empty string for a fresh session.
        """
        ...

    def discover_resumable_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[ResumableSession]:
        """List sessions owned by this provider, optionally for one workspace."""
        ...

    def parse_terminal_status(
        self, pane_text: str, *, pane_title: str = ""
    ) -> StatusUpdate | None:
        """Parse captured pane text into a StatusUpdate.

        Args:
            pane_text: Captured terminal pane content.
            pane_title: Terminal title set via OSC escapes (e.g. Gemini CLI).

        Returns None if no status line or interactive UI is detected.
        """
        ...

    def extract_bash_output(self, pane_text: str, command: str) -> str | None:
        """Extract ``!`` command output from a captured tmux pane.

        Returns the command echo and output lines, or None if not found.
        """
        ...

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        """Discover transcript for a hookless provider session.

        Scans the provider's session storage for the most recent transcript
        matching the given working directory. Returns a SessionStartEvent
        if found, None otherwise. Implementations may optionally honor
        ``max_age`` (seconds) to ignore stale transcript files.

        Only useful for providers without hook support (Codex, Gemini).
        Providers with hooks (Claude) return None.
        """
        ...

    def requires_pane_title_for_detection(self, pane_current_command: str) -> bool:
        """Whether provider detection requires pane title for given command."""
        ...

    def detect_from_pane_title(
        self, pane_current_command: str, pane_title: str
    ) -> bool:
        """Detect this provider from tmux pane title."""
        ...

    def discover_commands(self, base_dir: str) -> list[DiscoveredCommand]:
        """Discover provider-native commands.

        This method is expected to return provider-native commands (typically
        built-ins). Higher-level command composition (for example, appending
        user-defined commands from shared command sources) is done by
        ``CommandCatalog.get_provider_commands()``.
        """
        ...

    def build_status_snapshot(
        self,
        transcript_path: str,
        *,
        display_name: str,
        session_id: str = "",
        cwd: str = "",
    ) -> str | None:
        """Build a transcript-based status snapshot for display.

        Returns a formatted string suitable for Telegram, or None if the
        provider does not support status snapshots or the transcript is
        unreadable. Only providers with ``supports_status_snapshot=True``
        return non-None values.
        """
        ...

    def has_output_since(self, transcript_path: str, offset: int) -> bool:
        """Check whether the transcript has assistant output after *offset*.

        Used to detect whether a slash command (e.g. ``/status``) produced
        native transcript output, so the caller can skip the snapshot fallback.
        """
        ...

    async def scrape_current_mode(self, window_id: str) -> str | None:  # noqa: ARG002
        """Scrape the agent CLI's mode-line from the terminal and return a short label.

        Returns a short human-readable label (e.g. "Edit", "Plan", "YOLO")
        or None if the provider does not expose a mode indicator.
        """
        return None

    async def seed_task_state(  # noqa: ARG002
        self,
        window_id: str,
        session_id: str,
        transcript_path: str,
    ) -> None:
        """Seed task-tracking state from a transcript on session start.

        Called once when a new session is discovered, before incremental
        entry processing begins. Default: no-op.
        Only providers with ``supports_task_tracking=True`` implement this.
        """

    def apply_task_entries(  # noqa: ARG002
        self,
        window_id: str,
        session_id: str,
        entries: list[dict],
    ) -> None:
        """Apply parsed transcript entries to task-tracking state.

        Called on each batch of new transcript entries. Default: no-op.
        Only providers with ``supports_task_tracking=True`` implement this.
        """
