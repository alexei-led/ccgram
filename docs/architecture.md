# ccgram Architecture

Generated from code state 2026-05-21.

## System Overview

ccgram maps each Telegram Forum topic to one terminal-multiplexer window running one agent CLI (Claude Code, Codex, Gemini, Pi, or Shell). All internal routing is keyed by window ID (`@0`, `@12`). Multiplexer access goes through the `multiplexer/` seam (`Multiplexer` Protocol); tmux is the default backend and herdr is selectable via `CCGRAM_MULTIPLEXER=herdr`.

```mermaid
graph TB
    Telegram["Telegram<br>(Forum topics)"]
    Bot["bot.py<br>(172 lines: factory + lifecycle)"]
    Bootstrap["bootstrap.py<br>post_init + post_shutdown"]
    Registry["handlers/registry.py<br>PTB handler wiring"]
    TC["telegram_client.py<br>TelegramClient Protocol<br>+ PTBTelegramClient adapter"]
    Handlers["handlers/<br>14 feature subpackages"]
    TmuxMgr["multiplexer/ seam <br> Multiplexer Protocol <br> (tmux default, herdr)"]
    Windows["multiplexer windows <br> (Claude, Codex, Gemini, Pi, Shell)"]
    Hook["hook.py<br>Claude Code hooks"]
    Monitor["session_monitor.py<br>poll loop"]
    State["State files<br>~/.ccgram/"]

    Telegram -- "updates" --> Bot
    Bot -- "post_init" --> Bootstrap
    Bot -- "register_all" --> Registry
    Registry -- "dispatch" --> Handlers
    Handlers -- "depend on Protocol" --> TC
    TC -- "PTBTelegramClient" --> Bot
    Handlers -- "send_keys / capture_pane (via multiplexer proxy)" --> TmuxMgr
    TmuxMgr --> Windows
    Windows -- "hook events" --> Hook
    Hook -- "session_map.json<br>events.jsonl" --> State
    Monitor -- "reads" --> State
    Monitor -- "NewMessage / NewWindowEvent" --> Handlers
```

## Module Layers

```mermaid
graph TD
    subgraph entry["Entry Points + Bootstrap"]
        CLI["cli.py / main.py"]
        BotPy["bot.py<br>(factory + lifecycle, 172 lines)"]
        BootstrapPy["bootstrap.py<br>post_init + post_shutdown"]
        RegistryPy["handlers/registry.py<br>PTB handler wiring"]
        HookPy["hook.py"]
    end

    subgraph protocol["Telegram Seam"]
        TCProto["telegram_client.py<br>TelegramClient Protocol<br>+ PTBTelegramClient adapter<br>+ FakeTelegramClient (tests)"]
    end

    subgraph handlers["Handler Layer — handlers/"]
        TopLevel["Top-level: callback_*, cleanup,<br>command_*, file_handler, hook_events,<br>inline, reactions, registry, response_builder,<br>sessions_dashboard, sync_command, upgrade, user_state"]
        TopicsPkg["topics/<br>topic_orchestration, topic_lifecycle,<br>directory_browser, directory_callbacks,<br>window_callbacks, new_command, worktree"]
        TextPkg["text/<br>text_handler"]
        InteractivePkg["interactive/<br>interactive_ui, interactive_callbacks"]
        StatusPkg["status/<br>status_bubble, status_bar_actions, topic_emoji, rc_probe"]
        LivePkg["live/<br>live_view, screenshot_callbacks, pane_callbacks"]
        SendPkg["send/<br>send_command, send_callbacks, send_security"]
        ToolbarPkg["toolbar/<br>toolbar_keyboard, toolbar_callbacks"]
        VoicePkg["voice/<br>voice_handler, voice_callbacks"]
        ShellPkg["shell/<br>shell_commands, shell_capture,<br>shell_context, shell_prompt_orchestrator"]
        MsgPipePkg["messaging_pipeline/<br>message_queue, message_routing,<br>message_sender, message_task,<br>tool_batch, topic_commands"]
        RecoveryPkg["recovery/<br>recovery_callbacks (dispatcher),<br>recovery_banner, resume_picker,<br>restore_command, resume_command,<br>transcript_discovery,<br>history, history_callbacks"]
        CommandsPkg["commands/<br>forward, menu_sync,<br>failure_probe, status_snapshot"]
        PollingPkg["polling/<br>polling_coordinator,<br>polling_types (pure), polling_state (stateful),<br>periodic_tasks,<br>window_tick/{decide, observe, apply}"]
    end

    subgraph query["Read-Only Query Layer"]
        WQ["window_query.py<br>read window state"]
        SQ["session_query.py<br>read session data"]
    end

    subgraph state["State Management (constructor DI — F2)"]
        SM["session.py<br>SessionManager<br>(constructs + owns stores)"]
        TR["thread_router.py<br>(callbacks via __init__)"]
        WS["window_state_store.py<br>(persistence kernel)"]
        WSP["window_state_ports/<br>pane/identity/worktree/<br>tool/lifecycle"]
        UP["user_preferences.py<br>(callback via __init__)"]
        SMS["session_map.py<br>SessionMapSync<br>(callback via __init__)"]
        SR["session_resolver.py"]
    end

    subgraph infra["Infrastructure"]
        TmuxMgr2["multiplexer/ seam<br>(tmux / herdr backends)"]
        WR["window_resolver.py"]
        SP["state_persistence.py"]
    end

    subgraph providers["Provider Abstraction"]
        Base["providers/base.py<br>AgentProvider protocol<br>ProviderCapabilities"]
        Claude["providers/claude.py"]
        Jsonl["providers/_jsonl.py<br>(Codex + Gemini + Pi base)"]
        Shell["providers/shell.py"]
    end

    subgraph monitor["Session Monitoring"]
        SesMon["session_monitor.py"]
        TReader["transcript_reader.py"]
        EvReader["event_reader.py"]
        SLifecycle["session_lifecycle.py"]
        IdleT["idle_tracker.py"]
    end

    BotPy --> BootstrapPy
    BotPy --> RegistryPy
    RegistryPy --> handlers
    handlers --> protocol
    protocol --> BotPy
    handlers --> query
    handlers --> WSP
    query --> WSP
    WSP --> WS
    query --> SR
    handlers --> SM
    SM --> TR & WS & UP & SMS
    SM --> SP
    SesMon --> TReader & EvReader & SLifecycle & IdleT
    SesMon --> SMS
    providers --> handlers
```

## State Flow: Topic → Window → Session

```mermaid
graph LR
    Topic["Telegram Topic<br>(thread_id)"]
    Window["tmux Window<br>(@id)"]
    Session["Claude Session<br>(uuid)"]

    Topic -- "thread_bindings<br>(thread_router.py)" --> Window
    Window -- "session_map.json<br>(written by hook)" --> Session

    WQ["window_query.py<br>read-only state"]
    SQ["session_query.py<br>read-only resolution"]
    SM["SessionManager<br>writes + startup"]

    Window -- "read" --> WQ
    Window -- "write" --> SM
    Session -- "read" --> SQ
```

## SessionManager Responsibilities

`SessionManager` constructs and owns the four state stores (`WindowStateStore`, `ThreadRouter`, `UserPreferences`, `SessionMapSync`) via constructor DI with explicit `schedule_save` callbacks. Its public surface is now small: startup orchestration (`__post_init__`, `resolve_stale_ids`), write coordination (`set_window_provider`, `set_window_cwd`, `set_*_mode`, `set_display_name`), and cross-cutting audit (`audit_state`, `prune_stale_state`, `prune_stale_window_states`).

Read paths bypass `SessionManager`:

- `window_query.py` — `get_window_provider()`, `get_approval_mode()`, `get_notification_mode()`, `view_window()`; feature-shaped reads delegate to `window_state_ports/*`.
- `window_state_ports/` — `pane_state`, `identity_state`, `worktree_state`, `tool_state`, `lifecycle_state`. Frozen projection dataclasses for handlers and Mini App, plus cohesive feature writes (pane upsert/remove/lifecycle, worktree metadata, batch mode, tool-call visibility, origin). Provider/session identity writes still delegate to `SessionManager.set_window_provider`.
- `session_query.py` — `resolve_session_for_window()`, `find_users_for_session()`, `get_recent_messages()`.
- `session_map_sync` (direct imports) — `load/prune/register`.
- `thread_router` (direct imports) — `get_display_name()`.

`WindowStateStore` remains the single persistence kernel for `WindowState`. Handler and Mini App reads of window state go through `window_query` or `window_state_ports/*` — never raw `WindowState` fields. Boundary enforced by `tests/ccgram/test_window_state_access_audit.py` (raw feature-field access permitted only in `window_state_store.py`, `window_state_ports/*`, `session.py`, `window_query.py`, and serialization tests) and `tests/ccgram/test_query_layer_only_for_handlers.py` (write/admin allow-list).

## Provider Protocol

```mermaid
classDiagram
    class ProviderCapabilities {
        +name: str
        +supports_hook: bool
        +supports_resume: bool
        +supports_task_tracking: bool
        +chat_first_command_path: bool
        +has_yolo_confirmation: bool
        ...15 more flags
    }

    class AgentProvider {
        <<Protocol>>
        +capabilities: ProviderCapabilities
        +make_launch_args() str
        +parse_transcript_line(line) dict
        +parse_transcript_entries(entries) list
        +parse_terminal_status(text) StatusUpdate
        +seed_task_state(wid, sid, path) ← NEW
        +apply_task_entries(wid, sid, entries) ← NEW
        +scrape_current_mode(wid) str
        ...8 more methods
    }

    class ClaudeProvider {
        +supports_task_tracking = True
        +seed_task_state() reads transcript
        +apply_task_entries() → claude_task_state
        +scrape_current_mode() parses mode-line
    }

    class JsonlProvider {
        +supports_task_tracking = False
        +seed_task_state() no-op
        +apply_task_entries() no-op
    }

    class CodexProvider
    class GeminiProvider
    class PiProvider
    class ShellProvider

    AgentProvider <|.. ClaudeProvider
    AgentProvider <|.. JsonlProvider
    JsonlProvider <|-- CodexProvider
    JsonlProvider <|-- GeminiProvider
    JsonlProvider <|-- PiProvider
    JsonlProvider <|-- ShellProvider
```

## Message Routing Flow

```mermaid
sequenceDiagram
    participant SessionMonitor
    participant MsgRouting as message_routing.py
    participant SQ as session_query.py
    participant WQ as window_query.py
    participant MsgQueue as message_queue.py
    participant Telegram

    SessionMonitor->>MsgRouting: NewMessage(session_id, text)
    MsgRouting->>SQ: find_users_for_session(session_id)
    SQ-->>MsgRouting: [(user_id, window_id, thread_id)]
    loop for each user
        MsgRouting->>WQ: get_notification_mode(window_id)
        WQ-->>MsgRouting: "all" | "errors_only" | "muted"
        alt not filtered
            MsgRouting->>MsgQueue: enqueue_content_message(...)
            MsgQueue->>Telegram: rate_limit_send → Bot API
        end
    end
```

## Hook Event Flow

```mermaid
sequenceDiagram
    participant Claude as Claude Code
    participant Hook as hook.py
    participant EventFiles as events.jsonl<br>session_map.json
    participant EventReader as event_reader.py
    participant SessionMonitor as session_monitor.py
    participant HookEvents as hook_events.py
    participant Telegram

    Claude->>Hook: hook event (stdin JSON)
    Hook->>EventFiles: append event + update map
    SessionMonitor->>EventReader: read_new_events(path, offset)
    EventReader-->>SessionMonitor: [HookEvent, ...]
    SessionMonitor->>HookEvents: dispatch_hook_event(event)
    HookEvents->>Telegram: status update / notification
```

## Shell Provider Architecture

```mermaid
graph TD
    ShellH["handlers/<br>shell_commands.py<br>shell_capture.py<br>shell_context.py<br>shell_prompt_orchestrator.py"]
    ShellProv["providers/<br>shell.py (thin)<br>shell_infra.py (utilities)"]
    JsonlBase["providers/_jsonl.py<br>(JsonlProvider base)"]

    ShellH -- "imports match_prompt,<br>KNOWN_SHELLS,<br>has_prompt_marker<br>(accepted leak: low volatility)" --> ShellProv
    ShellProv --> JsonlBase

    PS1["Terminal PS1<br>wrap mode: append ⌘N⌘<br>replace mode: {prefix}:N❯"]
    ShellH -- "setup_shell_prompt()" --> PS1

    LLM["llm/ (optional)<br>NL→command generation"]
    ShellH -- "get_completer()" --> LLM
```

## Session Monitoring Architecture

```mermaid
graph TB
    SM2["session_monitor.py<br>(coordinator)"]

    SM2 --> ER["event_reader.py<br>read_new_events(path, offset)<br>stateless pure I/O"]
    SM2 --> TR2["transcript_reader.py<br>per-session JSONL parsing<br>file mtime cache"]
    SM2 --> SL["session_lifecycle.py<br>reconcile() session map changes<br>handle_session_end()"]
    SM2 --> IT["idle_tracker.py<br>per-session activity timestamps"]

    TR2 -- "seed_task_state()<br>apply_task_entries()<br>(via provider protocol)" --> Claude2["ClaudeProvider<br>clause_task_state"]

    SM2 -- "load_session_map()<br>prune_session_map()" --> SMS2["session_map_sync"]
```

## Key Design Decisions

| Decision                                | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Window ID-centric routing (`@0`, `@12`) | Unique within a tmux server; window names are display-only                                                                                                                                                                                                                                                                                                                                                                                            |
| Hook-based event system                 | Instant stop/done/notification detection without terminal polling; events appended to `events.jsonl` and consumed incrementally                                                                                                                                                                                                                                                                                                                       |
| `window_query` / `session_query`        | Handlers read window/session state via free functions, never importing `SessionManager`. Direct `session_manager.<attr>` in `handlers/**` is restricted to a documented write/admin allow-list                                                                                                                                                                                                                                                        |
| `window_state_ports/` feature ports     | `WindowStateStore` is the single persistence kernel; `window_state_ports/{pane,identity,worktree,tool,lifecycle}_state` are thin adapters exposing frozen projections plus cohesive feature writes. Raw `WindowState`-field access outside the kernel, the ports, `session.py`, `window_query.py`, and serialization tests fails `test_window_state_access_audit.py`. Provider identity writes still delegate to `SessionManager.set_window_provider` |
| Provider protocol with capability flags | Gate UX features (resume, continue, hooks, YOLO, mode scraping, RC, picker hints) without `if provider == "claude"` checks                                                                                                                                                                                                                                                                                                                            |
| `supports_task_tracking` capability     | `transcript_reader` is provider-agnostic; only Claude implements task state                                                                                                                                                                                                                                                                                                                                                                           |
| Tool-call visibility on `WindowState`   | Per-window `tool_call_visibility` (`default`/`shown`/`hidden`) gates `_handle_content_task` before batch eligibility; hook events bypass                                                                                                                                                                                                                                                                                                              |
| Status-mode color schemes               | `CCGRAM_STATUS_MODE` selects `system` (green = working) or `user` (green = ready) — only emoji rendering changes, not internal state names                                                                                                                                                                                                                                                                                                            |
| Gemini JSONL incremental reads          | Gemini CLI v0.40+ uses append-only JSONL; provider inherits `JsonlProvider` byte-offset reader, dedupes by message id and pending tool_use                                                                                                                                                                                                                                                                                                            |
| Viewport screenshots                    | `/screenshot` and 📷 capture the current viewport with ANSI color; live view uses the same viewport capture at a smaller font size. `/last` (📄 Last toolbar button) delivers the last assistant reply text (AI providers, from transcript) or last command+output block (shell) as a message or `.txt` attachment for overflow                                                                                                                       |
| Picker hints                            | `ProviderCapabilities.tui_picker_commands` lists modal-opening slash commands; `forward._picker_hint()` adds a hint pointing at `/toolbar` when one is forwarded, with the hint text adapted to the resolved `ToolbarLayout`                                                                                                                                                                                                                          |
| `handlers/` feature subpackages         | Handlers are grouped into 14 feature subpackages; each `__init__.py` re-exports the public surface                                                                                                                                                                                                                                                                                                                                                    |
| Constructor DI for stores               | `SessionManager` constructs `WindowStateStore`/`ThreadRouter`/`UserPreferences`/`SessionMapSync` with explicit `schedule_save` callbacks; no `_wire_singletons` and no silent unwired defaults — `register_*_callback` fails loud                                                                                                                                                                                                                     |
| `bot.py` is a factory + lifecycle only  | 172 lines; `handlers/registry.py` owns PTB handler wiring; `bootstrap.py` owns `post_init` (ordered: `register_provider_commands` → `verify_hooks_installed` → `wire_runtime_callbacks` → `start_session_monitor` → `start_status_polling` → `start_miniapp_if_enabled`) and `post_shutdown`                                                                                                                                                          |
| `window_tick/decide,observe,apply`      | Pure decision kernel (`decide.py`, zero deps on tmux/PTB/singletons) + pure observer (`observe.py`, `TickContext` out) + side-effect applier (`apply.py`); `decide_tick` is unit-tested without mocks                                                                                                                                                                                                                                                 |
| `TelegramClient` Protocol               | Handlers depend on `TelegramClient` not `telegram.Bot`; `PTBTelegramClient` adapts in production, `FakeTelegramClient` records in tests. Only `bot.py`, `bootstrap.py`, `handlers/registry.py`, `telegram_client.py`, `telegram_request.py`, `telegram_sender.py` import from `telegram.ext` at runtime                                                                                                                                               |
| Pure types vs stateful polling          | `polling_types.py` holds contracts (stdlib + `providers.base.StatusUpdate` only); `polling_state.py` holds strategies + module-level singletons; `decide.py` imports only from `polling_types`. Pinned by `test_polling_types_purity.py`                                                                                                                                                                                                              |
| Recovery split                          | `recovery_callbacks.py` is a thin dispatcher; `recovery_banner.py` owns dead-window banner UX; `resume_picker.py` owns the resume picker + transcript scan. `recovery/__init__.py` re-exports the public surface                                                                                                                                                                                                                                      |
| Commands subpackage                     | `handlers/commands/` mirrors the `shell/` pattern: `forward.py`, `menu_sync.py`, `failure_probe.py`, `status_snapshot.py`. `commands/__init__.py` hosts `commands_command` + `toolbar_command`                                                                                                                                                                                                                                                        |
| Lazy-import contract                    | In-function `Import`/`ImportFrom` must carry `# Lazy: <reason>` (or live inside `if TYPE_CHECKING:` / `_reset_*_for_testing`). `scripts/lint_lazy_imports.py` runs in `make lint`; cycle regressions caught by `tests/integration/test_import_no_cycles.py`                                                                                                                                                                                           |
