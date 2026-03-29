# Architecture Overview

## Functional Requirements Summary

This architecture addresses five modularity imbalances identified in the [modularity review](../reviews/modularity-review.md):

1. **Status Polling Knowledge Sprawl** (CRITICAL) — `status_polling.py` (1,339 lines) holds model-level knowledge of 7 conceptual domains
2. **SessionManager State Accumulation** (CRITICAL) — `session.py` (945 lines, 40+ methods) mixes 8 unrelated concerns behind a single API surface exposed to 24 consumers
3. **bot.py Dispatch Monolith** (CRITICAL) — `bot.py` (2,018 lines, 49 imports) serves as a manual wiring hub for 30+ handlers with mixed orchestration logic
4. **Provider Abstraction Leakage** (SIGNIFICANT) — Codex-specific modules and provider name checks break the provider protocol's design intent
5. **Shell Prompt Implicit Contract** (SIGNIFICANT) — Raw `re.Match` groups create silent breakage risk between shell provider and capture modules

## Module Map

| Module                    | Type    | Files                                                                          | Description                                                             |
| ------------------------- | ------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| **Polling Subsystem**     | NEW     | `handlers/polling_coordinator.py`, `handlers/polling_strategies.py`            | Terminal state polling via 4 focused strategies + thin coordinator      |
| **Thread Router**         | NEW     | `thread_router.py`                                                             | Bidirectional topic↔window routing, chat ID resolution                  |
| **Session State**         | CHANGED | `session.py`                                                                   | Per-window state, preferences, persistence; exposes Protocol interfaces |
| **Bot Shell**             | CHANGED | `bot.py`                                                                       | Pure handler registration + application lifecycle                       |
| **Callback Dispatch**     | NEW     | `handlers/callback_registry.py`                                                | Self-registration callback routing via `@register` decorator            |
| **Command Orchestration** | NEW     | `handlers/command_orchestration.py`                                            | Command forwarding, menu caching, failure probing                       |
| **Topic Orchestration**   | NEW     | `handlers/topic_orchestration.py`                                              | Auto-topic creation, window adoption, rate limiting                     |
| **Provider Protocol**     | CHANGED | `providers/base.py`, `providers/__init__.py`                                   | Extended with optional `build_status_snapshot()` method                 |
| **Shell Provider**        | CHANGED | `providers/shell.py`                                                           | Typed `PromptMatch` contract replacing raw `re.Match`                   |
| **Codex Provider**        | CHANGED | `providers/codex.py`, `providers/codex_status.py`, `providers/codex_format.py` | Absorbs previously leaked top-level modules                             |

### File Migration Map

| Current                                       | Target                                                                                                                                          | Change     |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `handlers/status_polling.py` (1,339 lines)    | `handlers/polling_coordinator.py` (~200) + `handlers/polling_strategies.py` (~800)                                                              | Decomposed |
| `bot.py` (2,018 lines)                        | `bot.py` (~400) + `handlers/callback_registry.py` (~80) + `handlers/command_orchestration.py` (~250) + `handlers/topic_orchestration.py` (~200) | Decomposed |
| `session.py` (945 lines)                      | `session.py` (~600) + `thread_router.py` (~200) + `protocols.py` (~60)                                                                          | Extracted  |
| `codex_status.py` (237 lines)                 | `providers/codex_status.py`                                                                                                                     | Moved      |
| `interactive_prompt_formatter.py` (245 lines) | `providers/codex_format.py`                                                                                                                     | Moved      |

### Net File Count

- **Removed**: 2 top-level modules (`codex_status.py`, `interactive_prompt_formatter.py`) + 1 handler (`status_polling.py`)
- **Added**: 6 new modules (`polling_coordinator.py`, `polling_strategies.py`, `callback_registry.py`, `command_orchestration.py`, `topic_orchestration.py`, `thread_router.py`, `protocols.py`)
- **Net**: +4 files. Total lines across changed files: ~2,790 (down from ~4,302 original — 35% reduction due to eliminated duplication and ceremony)

## How the Modules Work Together

### Flow 1: User Sends Text Message → Agent

```
User types "hello" in topic (thread_id=42)
  → Bot Shell dispatches to text_handler
  → text_handler calls Thread Router: get_window_for_thread(user_id, 42) → "@0"
  → text_handler calls Session State: send_to_window("@0", "hello")
  → Session State delegates to Tmux Manager: send_keys("@0", "hello\n")
```

**Modules**: Bot Shell → text_handler → Thread Router → Session State → Tmux Manager
**Contracts**: Contract (routing) → Model (window state) → Contract (tmux)

### Flow 2: Agent Output → User

```
SessionMonitor detects new JSONL content
  → message_callback fires with (session_id, messages)
  → Thread Router: find_users_for_session(session_id) → [(user_id, thread_id)]
  → Thread Router: resolve_chat_id(user_id, thread_id) → chat_id
  → Message Queue: enqueue(user_id, thread_id, window_id, messages)
  → Queue worker: rate_limit_send → Telegram API
```

**Modules**: Session Monitor → Thread Router → Message Queue → Telegram API
**Contracts**: Functional (callback) → Contract (routing) → Contract (queue)

### Flow 3: Status Polling Cycle (1 second)

```
Polling Coordinator iterates Thread Router: iter_thread_bindings()
  For each (user_id, thread_id, window_id):
    → TerminalStatusStrategy: capture pane, parse via Provider Protocol
      If active status → enqueue status update, update topic emoji
      If interactive prompt → InteractiveUIStrategy: surface keyboard
      If RC detected → debounce transition
    → TopicLifecycleStrategy: check autoclose timers, probe topic existence
      If done/dead timeout → trigger cleanup
    → ShellRelayStrategy: check passive output (shell windows only)
```

**Modules**: Polling Coordinator → Polling Strategies → [Thread Router, Session State, Provider Protocol, Tmux Manager, Message Queue, Interactive UI, Topic Emoji, Shell Capture, Cleanup]
**Key design**: Coordinator has zero domain knowledge — each strategy owns its concerns.

### Flow 4: User Taps Inline Button

```
Telegram sends CallbackQuery with data="dir:select:/home/user/project"
  → Bot Shell dispatches to Callback Dispatch: dispatch(update, context)
  → Callback Dispatch: longest-prefix match "dir:" → handle_directory_callback
  → handle_directory_callback processes the selection
```

**Modules**: Bot Shell → Callback Dispatch → Handler module
**Contracts**: Contract (registration) → Contract (prefix decorator). Adding a new callback handler never touches Bot Shell or Callback Dispatch dispatch logic.

### Flow 5: Unknown `/command` Forwarded to Provider

```
User sends /status in Codex topic
  → Bot Shell dispatches to Command Orchestration: forward_command_handler
  → Command Orch: sync provider menu (3-tier cache)
  → Command Orch: validate against Provider Protocol capabilities
  → Command Orch: send_to_window via Session State
  → Command Orch: probe for failure (async delay + pane inspection)
  → Command Orch: provider.build_status_snapshot() → Codex returns formatted stats
  → Command Orch: deliver snapshot to user via Message Sender
```

**Modules**: Bot Shell → Command Orchestration → [Provider Protocol, Session State, Thread Router, Message Sender]
**Key design**: No `capabilities.name == "codex"` check — `build_status_snapshot()` is polymorphic. Claude/Gemini/Shell return None.

### Flow 6: New Tmux Window Auto-Creates Topic

```
External tmux window created
  → SessionMonitor fires NewWindowEvent
  → Topic Orchestration: handle_new_window(bot, event)
    → Tmux Manager: get window metadata
    → Provider Protocol: detect_provider_from_pane() → "codex"
    → Session State: set_window_provider("@5", "codex")
    → Thread Router: enumerate chats via iter_thread_bindings()
    → Telegram API: create_forum_topic(chat_id, "my-project")
    → Thread Router: bind_thread(user_id, new_thread_id, "@5")
```

**Modules**: Session Monitor → Topic Orchestration → [Tmux Manager, Provider Protocol, Session State, Thread Router, Telegram API]
**Key design**: Rate limiting per chat prevents Telegram flood control.

### Flow 7: Shell Command with Typed Prompt Contract

```
Shell command executes, output appears between two prompt markers
  → Polling calls ShellRelayStrategy → delegates to Shell Capture
  → Shell Capture: match_prompt(line) → PromptMatch(sequence_number=5, trailing_text="ls -la", exit_code=0, raw_line="...")
  → Shell Capture: match_prompt(next_marker_line) → PromptMatch(sequence_number=6, trailing_text="", exit_code=0, ...)
  → Output between markers extracted, exit code from PromptMatch.exit_code
  → Formatted output sent to user
```

**Modules**: Polling Subsystem → Shell Capture → Shell Provider (via PromptMatch)
**Key design**: Named fields (`.sequence_number`, `.exit_code`) instead of positional groups (`.group(1)`, `.group(2)`). Adding fields to PromptMatch doesn't break consumers.

## Coupling Assessment

| Integration                           | Strength       | Distance                        | Volatility     | Balanced?      | Rationale                                                                                           |
| ------------------------------------- | -------------- | ------------------------------- | -------------- | -------------- | --------------------------------------------------------------------------------------------------- |
| Polling Subsystem → Session State     | Model          | Low (same pkg)                  | High (core)    | **Yes**        | Strategies read WindowState fields — co-location justified by high mutual knowledge                 |
| Polling Subsystem → Thread Router     | Contract       | Low (same pkg)                  | High (core)    | **Borderline** | Different bounded contexts co-located for deployment; low coupling is intentional, not low cohesion |
| Polling Subsystem → Provider Protocol | Contract       | Low (same pkg)                  | Medium (supp.) | **Yes**        | Low-volatility protocol side grants exemption                                                       |
| Bot Shell → Callback Dispatch         | Contract       | Low (same pkg)                  | Low (generic)  | **Yes**        | Stable infrastructure — low volatility exempts from balance check                                   |
| Bot Shell → Command Orchestration     | Functional     | Low (same pkg)                  | High (core)    | **Yes**        | High strength at low distance for volatile module — correct cohesion                                |
| Bot Shell → Topic Orchestration       | Functional     | Low (same pkg)                  | Medium (supp.) | **Yes**        | Appropriate strength for distance and volatility                                                    |
| Command Orch. → Provider Protocol     | Contract       | Low (same pkg)                  | Medium (supp.) | **Yes**        | Capability queries without deep provider knowledge                                                  |
| Command Orch. → Session State         | Model          | Low (same pkg)                  | High (core)    | **Yes**        | Probing needs window state internals — justified co-location                                        |
| Thread Router ↔ Session State         | Functional     | Low (same pkg)                  | High (core)    | **Yes**        | Shared persistence lifecycle — bidirectional coordination by design                                 |
| Shell Capture → Shell Provider        | Contract       | Medium (handlers/ → providers/) | High (active)  | **Yes**        | Typed PromptMatch contract across package boundary — correct balance                                |
| Codex Provider → Codex helpers        | Model          | Low (same pkg)                  | Medium (supp.) | **Yes**        | Provider-specific knowledge co-located within provider package                                      |
| Bot layer → Codex internals           | **Eliminated** | —                               | —              | **Yes**        | No more direct dependency — bot calls provider protocol uniformly                                   |

## Design Decisions and Trade-offs

### Decision 1: Polling strategies in one module, not four

**Considered**: Four separate files (`terminal_status_strategy.py`, `interactive_ui_strategy.py`, `topic_lifecycle_strategy.py`, `shell_relay_strategy.py`).

**Chosen**: Single `polling_strategies.py` with four classes.

**Why**: The strategies share type imports (`WindowPollState`, `TopicPollState`), are always modified in the same PR (the poll interface is their shared evolution axis), and four tiny files would create cross-import friction without reducing coupling. The balance model says co-located modules with shared knowledge should have high strength — separate files at low distance would be low cohesion.

### Decision 2: Thread Router physically extracted, not Protocol-only

**Considered**: Protocol-only segregation (SessionManager implements `ThreadRouter` protocol, consumers type-hint against it).

**Chosen**: Physical extraction into `thread_router.py` class, plus Protocol interfaces on remaining SessionManager.

**Why**: Thread routing (82 calls, 9 methods) is the highest-usage concern with the cleanest data boundary. Physical extraction makes the dependency audit concrete — `from ccgram.thread_router import thread_router` is unambiguous. Protocol-only segregation still allows accidental access to the full SessionManager. The cost (200-line new file, updated imports across 24 consumers) is one-time; the benefit (prevented state accumulation) is ongoing.

### Decision 3: Callback registry with explicit `load_handlers()`, not auto-discovery

**Considered**: Automatic module discovery (scan `handlers/` for `@register` decorators).

**Chosen**: Explicit `load_handlers()` function with manual imports.

**Why**: Auto-discovery adds runtime magic that makes import errors harder to debug. Explicit imports mean a missing handler causes an ImportError at startup, not a silent missing callback at runtime. The cost is one import line per handler module — acceptable for a codebase with ~15 callback handler modules.

### Decision 4: Optional protocol method for status snapshots, not capability flag + dispatch

**Considered**: `ProviderCapabilities.supports_status_snapshot: bool` flag, with bot.py checking the flag and calling a separate function.

**Chosen**: `build_status_snapshot() -> str | None` as an optional method on `AgentProvider` with a default `None` return.

**Why**: The flag-and-dispatch pattern recreates the problem — consumers still need to know _how_ to call the snapshot builder. An optional method with a default return means consumers call it uniformly and handle `None`. No flag checks, no dispatch, no provider name strings.

### Decision 5: PromptMatch with exit_code field, not just sequence_number + trailing_text

**Considered**: Minimal dataclass with only `sequence_number` and `trailing_text` (matching current regex groups).

**Chosen**: Include `exit_code` and `raw_line` fields.

**Why**: `exit_code` is extracted from the sequence number in wrap mode but is a distinct semantic concept — callers want "what was the exit code?" not "what was group 1?". Including it in the dataclass makes the semantic intent explicit. `raw_line` aids debugging without requiring callers to reconstruct from fields.

### Decision 6: Incremental migration order

**Recommended order** (each step is independently mergeable):

1. **C2: Shell PromptMatch** — smallest change, zero risk, fixes silent breakage class. ~2 files changed.
2. **C1: Provider abstraction** — move files + add optional method. ~5 files changed.
3. **B1: Thread Router extraction** — physical split + Protocol definitions. ~26 files changed (import updates).
4. **A2: Bot dispatch extraction** — callback registry + orchestration modules. ~20 files changed.
5. **A1: Polling decomposition** — largest change, benefits from all prior work. ~5 files changed.

Each step reduces the blast radius of subsequent steps. Steps 1-2 are encapsulation repairs (low risk). Step 3 is interface segregation (medium risk, high consumer count). Steps 4-5 are decompositions (medium risk, high reward).

## Unresolved Risks

### Minor: Polling Subsystem → Thread Router low cohesion (Borderline)

The balance model flags contract-level coupling at low distance as potential low cohesion. This is an intentional trade-off: polling and routing are genuinely different bounded contexts. Merging them would recreate the original knowledge sprawl. If this becomes problematic (e.g., polling and routing are always changed together), consider merging them into a single module with clear internal boundaries.

### Minor: SessionManager still accumulates preferences

After Thread Router extraction, SessionManager still owns 5 concerns: window state, preferences, directory favorites, user offsets, and persistence. Further extraction (e.g., separate `UserPreferences` class) is possible but not currently justified — the Protocol interfaces prevent consumer coupling from growing. Revisit if SessionManager exceeds 800 lines again.

### Minor: Callback registry import ordering

The `@register` decorator pattern requires handler modules to be imported before `create_bot()` runs. If a handler module has a side effect on import (e.g., initializing a global), the import ordering matters. This is mitigated by the explicit `load_handlers()` function, but circular imports between handler modules could surface. Mitigation: handler modules should never import each other; shared logic lives in helper modules.

### Observation: 24 consumer import migration for Thread Router

Extracting Thread Router requires updating imports in ~24 files. This is a mechanical change but creates a large diff that may conflict with in-flight feature work. Mitigation: perform the migration in a dedicated PR with no functional changes, and coordinate timing with active feature branches.

---

_This architecture addresses the coupling imbalances identified by the [Balanced Coupling](https://coupling.dev) model analysis. All design decisions are grounded in the three coupling dimensions (integration strength, distance, volatility) and the balance rule._
