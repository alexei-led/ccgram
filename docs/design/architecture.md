# Architecture Overview

## Functional Requirements Summary

This architecture addresses the four modularity imbalances identified in the [third modularity review](../reviews/modularity-review.md) (2026-03-30):

1. **Inter-Agent Messaging Private-Interface Coupling** (SIGNIFICANT) — Handler-layer modules (`msg_broker`, `msg_spawn`, `msg_cmd`) bypass encapsulation boundaries via private imports, shared mutable dicts, and raw state.json parsing
2. **SessionManager God Object** (SIGNIFICANT) — 1,526 lines, ~50 public methods, 24 consumers, 6 unrelated concerns with no interface segregation
3. **Fragmented Per-Topic State** (SIGNIFICANT) — 14+ module-level mutable dicts across 9 modules with 4 inconsistent key types, orchestrated by cleanup.py via 14 lazy imports
4. **Cross-Strategy Encapsulation Violations** (MINOR) — `TopicLifecycleStrategy` accesses `TerminalStatusStrategy._states` directly; coordinator imports private constants

The architecture preserves 7 well-balanced integrations: Hook System, Provider Protocol, LLM/Whisper, Callback Registry, ThreadRouter, `mailbox.py` core, and Shell PromptMatch contract.

## Module Map

### Preserved Modules (no boundary changes)

| Module                    | Files                                                                          | Description                                                 |
| ------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| **Bot Shell**             | `bot.py`                                                                       | PTB Application lifecycle, handler registration             |
| **Thread Router**         | `thread_router.py`                                                             | Bidirectional topic↔window routing, chat ID resolution      |
| **Callback Dispatch**     | `handlers/callback_registry.py`                                                | Self-registering `@register` prefix-based callback routing  |
| **Provider Protocol**     | `providers/base.py`, `providers/__init__.py`, `providers/registry.py`          | AgentProvider protocol, capability matrix, detection        |
| **Shell Provider**        | `providers/shell.py`                                                           | PromptMatch contract, prompt injection, idle detection      |
| **Codex Provider**        | `providers/codex.py`, `providers/codex_status.py`, `providers/codex_format.py` | JSONL transcripts, status snapshots, interactive formatting |
| **Command Orchestration** | `handlers/command_orchestration.py`                                            | Command forwarding, 3-tier menu cache, status snapshots     |
| **LLM**                   | `llm/`                                                                         | Protocol + factory for OpenAI-compatible chat completions   |
| **Whisper**               | `whisper/`                                                                     | Protocol + factory for voice transcription                  |
| **Session Monitor**       | `session_monitor.py`                                                           | JSONL polling, hook event dispatch, session tracking        |
| **Tmux Manager**          | `tmux_manager.py`                                                              | Window/pane CRUD, send_keys, capture_pane                   |

### New Modules

| Module                   | Files                      | Description                                                                   |
| ------------------------ | -------------------------- | ----------------------------------------------------------------------------- |
| **Topic State Registry** | `topic_state_registry.py`  | Self-registering cleanup orchestration; replaces cleanup.py's 14 lazy imports |
| **User Preferences**     | `user_preferences.py`      | Starred dirs, MRU, read offsets (extracted from SessionManager)               |
| **Message Delivery**     | `handlers/msg_delivery.py` | `MessageDeliveryStrategy` singleton; breaks broker↔telegram cycle             |

### Redesigned Modules (boundary changes)

| Module                    | Files                                                               | Key Change                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Session State**         | `session.py`                                                        | Drop 9 pass-through methods; add `export_window_info()`; expose Protocol interfaces (`WindowStateStore`, `SessionIO`, `WindowModeConfig`) |
| **Mailbox**               | `mailbox.py`                                                        | Expose `sanitize_dir_name()`, `validate_no_traversal()` as public API                                                                     |
| **Spawn Request**         | `spawn_request.py`                                                  | Public accessor API (`get_pending`, `pop_pending`, `iter_pending`, `register_pending`); private `_pending_requests` dict                  |
| **Message Broker**        | `handlers/msg_broker.py`                                            | Uses mailbox public API (sanitize_dir_name, validate_no_traversal); imports spawn accessors; imports delivery strategy from msg_delivery  |
| **Messaging Telegram UI** | `handlers/msg_telegram.py`                                          | `resolve_topic()` promoted to public; imports delivery_strategy from msg_delivery                                                         |
| **Message Spawn**         | `handlers/msg_spawn.py`                                             | Uses public APIs from topic_orchestration and msg_telegram                                                                                |
| **Topic Orchestration**   | `handlers/topic_orchestration.py`                                   | `collect_target_chats()`, `create_topic_in_chat()` promoted to public                                                                     |
| **Polling Subsystem**     | `handlers/polling_coordinator.py`, `handlers/polling_strategies.py` | TerminalStatusStrategy gets public methods for probe/seen/unbound state; coordinator delegates threshold comparisons                      |
| **Messaging CLI**         | `msg_cmd.py`                                                        | Uses `export_window_info()` instead of raw state.json parsing                                                                             |

## How the Modules Work Together

### Flow 1: User Sends Text Message → Agent

```
User types "hello" in topic (thread_id=42)
  → Bot Shell dispatches to text_handler
  → text_handler calls Thread Router: get_window_for_thread(user_id, 42) → "@0"
  → text_handler calls Session State (SessionIO): send_to_window("@0", "hello")
  → Session State delegates to Tmux Manager: send_keys("@0", "hello\n")
```

**Modules**: Bot Shell → text_handler → Thread Router → Session State (SessionIO) → Tmux Manager
**Contracts**: Contract (routing) → Contract (SessionIO protocol) → Contract (tmux)

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

### Flow 3: Inter-Agent Message Delivery (NEW)

```
Agent A sends "ccgram msg send @5 'help with tests'"
  → Messaging CLI: create message via Mailbox.create()
  → Mailbox writes message JSON to @5's inbox directory
  → Polling Coordinator calls broker_delivery_cycle()
  → Message Broker: check window @5 idle via Provider Protocol
  → Message Broker: construct delivery path via Mailbox.sanitize_dir_name()
  → Message Broker: inject message via Tmux Manager send_keys
  → Message Broker: mark delivered via Message Delivery strategy
  → Messaging Telegram UI: notify_messages_delivered() → silent notification in both topics
```

**Modules**: CLI → Mailbox → Polling → Broker → [Provider, Mailbox, Tmux, Delivery] → Telegram UI
**Key design**: Broker never imports mailbox private functions; delivery strategy owned by msg_delivery module (no cycle)

### Flow 4: Agent Spawn Request (NEW)

```
Agent sends "ccgram msg spawn --provider claude --cwd ~/project"
  → Messaging CLI: create_spawn_request() via Spawn Request module
  → Spawn Request persists to mailbox/spawns/
  → Polling → Broker: scan_spawn_requests() via Spawn Request public API
  → Broker routes to Message Spawn handler
  → Message Spawn: post_spawn_approval_keyboard() → Telegram inline keyboard
  → User taps "Approve"
  → Message Spawn: creates window via Tmux Manager
  → Message Spawn: creates topic via Topic Orchestration public API (collect_target_chats, create_topic_in_chat)
  → Message Spawn: installs messaging skill via msg_skill
```

**Modules**: CLI → Spawn Request → Polling → Broker → Spawn Handler → [Tmux, Topic Orchestration, msg_skill]
**Key design**: Spawn handler uses public `collect_target_chats()` and `create_topic_in_chat()` from topic_orchestration; no private imports

### Flow 5: Topic Cleanup via Registry (NEW)

```
Topic closed or window dies
  → Polling detects dead window
  → cleanup.py: resolve window_id, qualified_id
  → Topic State Registry: clear_all(user_id, thread_id, window_id, qualified_id)
    → Calls all registered topic-scoped cleanups with (user_id, thread_id)
    → Resolves chat_id for legacy modules, calls (chat_id, thread_id) cleanups
    → Calls all window-scoped cleanups with window_id
    → Calls all qualified-scoped cleanups with qualified_id
```

**Modules**: Polling → cleanup.py → Topic State Registry → [all registered handler modules]
**Key design**: cleanup.py is ~20 lines (resolve IDs, call registry); no lazy imports. New features register via @topic_state.register("scope")

### Flow 6: Status Polling Cycle (1 second)

```
Polling Coordinator iterates Thread Router: iter_thread_bindings()
  For each (user_id, thread_id, window_id):
    → TerminalStatusStrategy: capture pane, parse via Provider Protocol
      Strategy owns threshold logic (MAX_PROBE_FAILURES, STARTUP_TIMEOUT)
      If active status → enqueue status update, update topic emoji
      If interactive prompt → InteractiveUIStrategy: surface keyboard
    → TopicLifecycleStrategy: check autoclose timers
      Calls TerminalStatusStrategy.clear_probe_failures(window_id) — public method
      Calls TerminalStatusStrategy.clear_seen_status(window_id) — public method
    → ShellRelayStrategy: check passive output
    → Broker delivery cycle: inject pending messages into idle windows
```

**Modules**: Coordinator → Strategies → [Thread Router, Session State, Provider, Tmux, Queue, Emoji, Shell Capture, Broker]
**Key design**: Coordinator delegates threshold decisions to strategies; TopicLifecycleStrategy accesses TerminalStatusStrategy only via public methods

### Flow 7: CLI Peer Discovery (REDESIGNED)

```
Agent runs "ccgram msg list-peers"
  → Messaging CLI: calls export_window_info() from Session State
  → Session State returns dict[str, WindowInfo] (explicit public contract)
  → Messaging CLI: passes to msg_discovery.list_peers()
  → Returns formatted peer list
```

**Key design**: CLI no longer parses state.json directly; uses `export_window_info()` public API

## Coupling Assessment

| Integration                      | Strength   | Distance                  | Volatility     | Balanced?  | Rationale                                                                  |
| -------------------------------- | ---------- | ------------------------- | -------------- | ---------- | -------------------------------------------------------------------------- |
| Hook System → Session Monitor    | Contract   | High (separate processes) | Low            | Yes        | JSONL file contract between processes — gold standard                      |
| Provider Protocol → Consumers    | Contract   | Low (same pkg)            | Medium (supp.) | Yes        | Capability flags gate behavior uniformly                                   |
| LLM / Whisper → Consumers        | Contract   | Low (same pkg)            | Low (generic)  | Yes        | Protocol + factory, zero cross-coupling                                    |
| Callback Registry → Handlers     | Contract   | Low (same pkg)            | Low (generic)  | Yes        | @register decorator absorbs handler volatility                             |
| ThreadRouter → Consumers         | Contract   | Low (same pkg)            | Low            | Yes        | Zero ccgram imports; clean public API                                      |
| Mailbox → Consumers              | Contract   | Low (same pkg)            | Medium (supp.) | Yes        | Zero internal imports; public sanitize_dir_name(), validate_no_traversal() |
| Shell PromptMatch → Consumers    | Contract   | Low (same pkg)            | Low (supp.)    | Yes        | Named dataclass fields; stable contract                                    |
| Topic State Registry → Handlers  | Contract   | Low (same pkg)            | High (core)    | Yes        | Self-registration pattern; adding state is local                           |
| msg_broker → Mailbox             | Contract   | Low (same pkg)            | High (core)    | Yes        | Public API only (sanitize_dir_name, validate_no_traversal)                 |
| msg_broker → Spawn Request       | Contract   | Low (same pkg)            | High (core)    | Yes        | Public accessors (get_pending, pop_pending)                                |
| msg_broker → Message Delivery    | Contract   | Low (same pkg)            | High (core)    | Yes        | Shared singleton with no back-reference                                    |
| msg_spawn → Topic Orchestration  | Contract   | Low (same pkg)            | Medium (supp.) | Yes        | Public functions (collect_target_chats, create_topic_in_chat)              |
| msg_spawn → Messaging Telegram   | Contract   | Low (same pkg)            | Medium (supp.) | Yes        | Public resolve_topic()                                                     |
| msg_cmd → Session State          | Contract   | Medium (CLI/bot)          | High (core)    | Yes        | Explicit export_window_info() API                                          |
| SessionManager → Consumers       | Contract   | Low (same pkg)            | High (core)    | Yes        | Narrow Protocol interfaces (WindowStateStore, SessionIO, WindowModeConfig) |
| Session State ↔ User Preferences | Functional | Low (same pkg)            | Low (generic)  | Yes        | Shared persistence lifecycle (same pattern as ThreadRouter)                |
| Polling Strategies → Coordinator | Model      | Low (same pkg)            | High (core)    | Borderline | Strategy singletons with public methods; coordinator delegates             |
| TopicLifecycle → TerminalStatus  | Contract   | Low (same file)           | Medium         | Yes        | Public methods (clear_probe_failures, clear_seen_status)                   |

## Design Decisions and Trade-offs

### Decision 1: Topic State Registry via self-registration (not central manifest)

**Considered**: Central manifest file listing all cleanup functions (like a DI container config).

**Chosen**: Self-registration at import time via `@topic_state.register(scope)` decorator.

**Why**: Matches the proven `callback_registry` pattern. Registration is co-located with the state it cleans — a developer adding per-window state sees both the state dict and its cleanup in the same file. No central file to update. The balance model says: the registry is generic infrastructure (low volatility); the registrations are in core/supporting modules (high volatility). Contract coupling at low distance — balanced.

**Trade-off**: Registration happens as import-time side effects. If a module isn't imported, its cleanup isn't registered. Mitigated by: handler modules are always imported via `load_handlers()` (same as callback registry).

### Decision 2: Extract MessageDeliveryStrategy to msg_delivery.py (not keep in broker)

**Considered**: Keep delivery strategy in msg_broker.py; have msg_telegram use a callback-based approach to avoid the import.

**Chosen**: Physical extraction to `msg_delivery.py`.

**Why**: The `msg_broker ↔ msg_telegram` circular dependency is a structural problem caused by co-locating the delivery strategy singleton with the broker that uses it. Both broker and telegram UI need access to the strategy — extracting it into a third module eliminates the cycle entirely. The balance model says: high-volatility shared state (delivery strategy) should be co-located with its primary consumers at low distance — a separate module within the same package satisfies this.

**Trade-off**: One more file. Minimal — the class is ~80 lines with clear ownership.

### Decision 3: Spawn Request accessor functions (not expose the dict)

**Considered**: Make `_pending_requests` a public module-level dict (the current de facto state).

**Chosen**: Private dict with `get_pending()`, `pop_pending()`, `iter_pending()`, `register_pending()` accessor functions.

**Why**: Three modules co-owning a single mutable dict with no accessor protocol is intrusive coupling in a high-volatility area. Accessor functions are the minimal contract-level encapsulation that allows the cache structure to change without coordinated edits. The balance model says: reduce integration strength from intrusive to contract while keeping distance low.

**Trade-off**: 4 one-line functions. The cost is nearly zero.

### Decision 4: Promote private functions to public (not create wrapper modules)

**Considered**: Create adapter/facade modules between msg_spawn and topic_orchestration.

**Chosen**: Simply promote `_collect_target_chats`, `_create_topic_in_chat`, `_resolve_topic` to public (drop underscore prefix, add docstrings).

**Why**: These functions already have stable signatures and are used across module boundaries. The underscore prefix is a lie — they are de facto public interfaces. Promoting them acknowledges reality. Creating wrapper modules would add distance without reducing strength. The balance model says: don't increase distance when strength is already appropriate for the distance.

**Trade-off**: None. This is pure recognition of existing contracts.

### Decision 5: Session State export_window_info() (not shared serialization format)

**Considered**: Define a shared `WindowInfo` type that both session.py and msg_cmd.py import from a common module.

**Chosen**: `export_window_info()` function on session.py that returns `dict[str, WindowInfo]` where `WindowInfo` is defined in `msg_discovery.py`.

**Why**: The CLI process cannot instantiate SessionManager (no bot token). The current code parses state.json directly, hardcoding field names. `export_window_info()` makes the dependency explicit — it's a contract-level function that session.py owns and msg_cmd.py calls. If SessionManager's serialization format changes, only this function needs updating (single point of change). The balance model says: reduce integration strength from intrusive (raw schema knowledge) to contract (explicit export function) while acknowledging the medium distance (CLI vs bot process).

**Trade-off**: The function loads state.json from disk in the CLI process, which duplicates some of SessionManager's load logic. Acceptable because it's a read-only snapshot.

### Decision 6: Protocol interfaces on SessionManager (not physical decomposition)

**Considered**: Split SessionManager into 3 physical classes (WindowStateStore, SessionIO, WindowModeConfig).

**Chosen**: Keep one physical class with Protocol interfaces for consumer-facing contracts.

**Why**: SessionManager's 6 concerns share persistence (state.json) and startup/shutdown lifecycle. Physical decomposition would require a persistence coordinator or shared state.json writes from multiple objects — adding complexity. Protocol interfaces achieve the same consumer-facing benefit (narrow dependency surface) without the internal restructuring cost. The balance model says: interface segregation reduces effective coupling strength without changing distance.

**Trade-off**: The physical class remains large (~1,200 lines after removing 9 pass-throughs and 6 preference methods). Protocol interfaces are only enforced by type checkers, not at runtime. Acceptable because the goal is consumer isolation, not runtime polymorphism.

## Unresolved Risks

### Minor: Polling Strategies → Coordinator model coupling (Borderline)

The coordinator imports strategy singletons and calls their public methods. This is model-level coupling at low distance. The balance model says this is borderline — acceptable if volatility stays medium. If the coordinator grows beyond ~800 lines or strategies start changing independently, consider a `PollResult` return dataclass so the coordinator routes without interpreting strategy semantics.

### Minor: Session State still accumulates per-window modes

After extracting UserPreferences, SessionManager still owns ~35 public methods across 4 concerns. Further extraction (e.g., `WindowModeConfig` as a physical class) is possible but not justified — the Protocol interfaces prevent consumer coupling from growing. Revisit if SessionManager exceeds 1,200 lines.

### Minor: Topic State Registry import ordering

Like the callback registry, the topic state registry depends on handler modules being imported before cleanup runs. If a handler module isn't imported (e.g., conditional import based on config), its state won't be cleaned. Mitigated by: all handler modules are imported via `load_handlers()` at startup.

### Minor: Message Delivery singleton testability

`MessageDeliveryStrategy` is a module-level singleton. Unit testing requires either importing and mutating it or adding a reset function. This matches the existing pattern for strategy singletons in polling_strategies.py.

### Observation: 15+ consumer import changes for SessionManager Protocol adoption

Introducing Protocol type annotations across 15+ handler files creates a large diff. Recommend: do this in a dedicated PR with no functional changes, similar to the ThreadRouter extraction migration. The Protocol interfaces can be adopted incrementally — consumers that don't type-annotate still work.

---

_This architecture addresses the coupling imbalances identified by the [Balanced Coupling](https://coupling.dev) model analysis. All design decisions are grounded in the three coupling dimensions (integration strength, distance, volatility) and the balance rule._
