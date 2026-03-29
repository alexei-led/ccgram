# Polling Subsystem

## Functional Responsibilities

- Poll terminal state for all active windows at 1-second intervals via a thin coordinator loop
- Detect provider status (active, idle, done) via pyte screen parsing and provider-specific regex (TerminalStatusStrategy)
- Detect interactive UI prompts (AskUserQuestion, ExitPlanMode, permissions) and surface them to users (InteractiveUIStrategy)
- Manage topic lifecycle: autoclose timers, dead window detection, topic existence probing, unbound window TTL (TopicLifecycleStrategy)
- Monitor passive shell output and relay it to Telegram (ShellRelayStrategy)
- Detect and debounce Remote Control (RC) mode transitions
- Scan non-active panes for blocked interactive prompts (multi-pane awareness)
- Enqueue status updates and typing indicators to message queue

## Encapsulated Knowledge

This module owns all knowledge that no other module should have:

- **pyte screen buffer management** — content-hash caching, ANSI rendering, screen buffer lifecycle per window
- **Autoclose timer logic** — done/dead state entry timestamps, configurable timeouts, timer cancellation
- **Dead window detection heuristics** — topic probe failure counting (60s interval), window existence checks, dead notification deduplication
- **Multi-pane scanning strategy** — which panes to check, alert hash deduplication, pane count caching with TTL
- **RC debounce logic** — on/off transition detection, timing windows, `rc_off_since` tracking
- **Startup grace period** — suppress status during window initialization, `has_seen_status` flag
- **Unbound window TTL** — track how long a window has been unbound, auto-kill after configurable timeout
- **Poll state structures** — `WindowPollState` (per-window) and `TopicPollState` (per-topic) own all mutable polling state; no module-level dicts leak outside

## Subdomain Classification

**Core** — This is the system's most volatile module. Nearly every feature addition touches the polling loop: new provider capabilities, new UI detection patterns, new lifecycle states. The polling subsystem is where terminal monitoring, session state, and Telegram actions converge. High investment in boundary design is justified.

## Integration Contracts

### → Thread Router (depends on)

- **Direction**: Polling Subsystem depends on Thread Router
- **Contract type**: Contract (narrow interface)
- **What is shared**: The set of active (user_id, thread_id, window_id) tuples to poll
- **Contract definition**: `iter_thread_bindings() -> Iterator[tuple[int, int, str]]`

### → Session State (depends on)

- **Direction**: Polling Subsystem depends on Session State
- **Contract type**: Model (reads WindowState internals)
- **What is shared**: Window state fields: `session_id`, `cwd`, `provider_name`, `transcript_path`, `notification_mode`, `approval_mode`
- **Contract definition**: `WindowStateStore` protocol — `get_window_state(window_id) -> WindowState`, `get_display_name(window_id) -> str`

### → Provider Protocol (depends on)

- **Direction**: Polling Subsystem depends on Provider Protocol
- **Contract type**: Contract (capability queries)
- **What is shared**: Provider identity and terminal parsing capability
- **Contract definition**: `get_provider_for_window(window_id) -> AgentProvider`, `provider.parse_terminal_status(lines) -> StatusUpdate | None`, `provider.capabilities`

### → Tmux Manager (depends on)

- **Direction**: Polling Subsystem depends on Tmux Manager
- **Contract type**: Contract (existing stable interface)
- **What is shared**: Pane content and window metadata
- **Contract definition**: `capture_pane(window_id) -> str`, `list_panes(window_id) -> list`, `find_window_by_id(window_id) -> TmuxWindow | None`

### → Message Queue (depends on)

- **Direction**: Polling Subsystem depends on Message Queue
- **Contract type**: Contract (enqueue interface)
- **What is shared**: Status text and delivery metadata
- **Contract definition**: `enqueue_status_update(user_id, thread_id, window_id, text, ...)`, `clear_tool_msg_ids_for_topic(user_id, thread_id)`

### → Interactive UI (drives)

- **Direction**: Polling Subsystem drives Interactive UI
- **Contract type**: Functional (passes parsed prompt data)
- **What is shared**: Interactive prompt detection results and mode management
- **Contract definition**: `handle_interactive_ui(bot, user_id, thread_id, window_id, lines, ...)`, `clear_interactive_mode(user_id)`, `set_interactive_mode(user_id, window_id)`

### → Topic Emoji (drives)

- **Direction**: Polling Subsystem drives Topic Emoji
- **Contract type**: Contract (status → emoji mapping)
- **What is shared**: Window activity state (active/idle/done/dead)
- **Contract definition**: `update_topic_emoji(bot, user_id, thread_id, state, ...)`

### → Shell Capture (drives)

- **Direction**: Polling Subsystem drives Shell Capture
- **Contract type**: Contract (delegation)
- **What is shared**: Window ID for passive monitoring
- **Contract definition**: `check_passive_shell_output(bot, window_id, user_id, thread_id)`, `clear_shell_monitor_state(window_id)`

### → Cleanup (drives)

- **Direction**: Polling Subsystem drives Cleanup
- **Contract type**: Contract (lifecycle event)
- **What is shared**: Topic identity for state cleanup
- **Contract definition**: `clear_topic_state(bot, user_id, thread_id, window_id, ...)`

## Change Vectors

These are reasonable future changes that would require ONLY this module to change:

- **Adding a new provider status detection method** (e.g., a provider that uses pane title for status) — only TerminalStatusStrategy changes
- **Changing interactive UI prompt detection** (e.g., new prompt format) — only InteractiveUIStrategy changes
- **Adding a new topic lifecycle state** (e.g., "paused") — only TopicLifecycleStrategy changes
- **Modifying shell output monitoring** (e.g., new relay format) — only ShellRelayStrategy changes
- **Changing poll interval or backoff logic** — only polling_coordinator changes
- **Adding a new autoclose trigger** — only TopicLifecycleStrategy changes
- **Changing RC debounce timing** — only TerminalStatusStrategy changes
- **Adding a new multi-pane scanning heuristic** — only InteractiveUIStrategy changes
