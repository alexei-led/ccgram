# Modularity Review

**Scope**: Full codebase — `src/ccgram/` (~24,800 LOC across 50+ Python modules)
**Date**: 2026-03-28

## Executive Summary

ccgram bridges Telegram and tmux to manage AI coding agents remotely — each Forum topic maps 1:1 to a tmux window running an agent CLI. The system's provider protocol and hook-based event pipeline are well-designed integrations that demonstrate strong [modularity](https://coupling.dev/posts/core-concepts/modularity/). However, three modules — `status_polling.py`, `session.py`, and `bot.py` — have accumulated [model-level knowledge](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/) across 5+ conceptual domains each, creating rigidity bottlenecks in the system's most [volatile](https://coupling.dev/posts/dimensions-of-coupling/volatility/) areas. A fourth issue — provider-specific logic leaking outside the provider abstraction — undermines an otherwise clean design pattern.

## Coupling Overview

| Integration                     | [Strength](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/)   | [Distance](https://coupling.dev/posts/dimensions-of-coupling/distance/) | [Volatility](https://coupling.dev/posts/dimensions-of-coupling/volatility/) | [Balanced?](https://coupling.dev/posts/core-concepts/balance/) |
| ------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Hook System -> Monitoring       | [Contract](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/)   | High (separate processes)                                               | Low                                                                         | Yes                                                            |
| Provider Protocol -> Consumers  | [Model](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/)      | Low (same package)                                                      | Low-Medium                                                                  | Yes                                                            |
| Status Polling -> 6+ domains    | [Model](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/)      | Medium (conceptual)                                                     | High                                                                        | **No**                                                         |
| SessionManager -> 24 consumers  | [Functional](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/) | Low (same package)                                                      | High                                                                        | **No**                                                         |
| bot.py -> 30+ handlers          | [Functional](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/) | Low (same package)                                                      | High                                                                        | **No**                                                         |
| Bot layer -> Codex internals    | [Model](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/)      | Low (same package)                                                      | Medium                                                                      | **No**                                                         |
| Shell prompt -> capture/polling | [Model](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/)      | Low (same package)                                                      | High                                                                        | Borderline                                                     |

---

## Issue: Status Polling Knowledge Sprawl

**Integration**: `status_polling.py` -> providers, session state, tmux, terminal parsing, shell subsystem, interactive UI, hook events, message queue, topic emoji, Telegram Bot API
**Severity**: CRITICAL

### Knowledge Leakage

`status_polling.py` (1,339 lines) holds [model-level knowledge](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/) of nearly every domain in the system. It doesn't just call other subsystems through their public APIs — it understands their internal semantics:

- It knows that shell providers need `setup_shell_prompt()` called on provider transitions (lazy-imports `providers.shell.setup_shell_prompt` at lines 1143 and 1186).
- It interprets `WindowState` fields directly (`transcript_path`, `session_id`, `cwd`, `provider_name`) rather than operating through a focused interface.
- It maintains parallel state dictionaries (`WindowPollState`, `TopicPollState`, `_dead_notified`, `_pane_alert_hashes`) that extend and mirror session state — 7 module-level mutable state structures in total.
- It directly calls `session_manager.register_hookless_session()` for transcript discovery — a responsibility that belongs in the session/monitoring layer.
- It checks `provider_name` string values directly (e.g., `if provider_name in ("codex", "gemini", "shell")` at line 527) instead of querying provider capabilities.

### Complexity Impact

The module crosses terminal, session, UI, shell, and provider domains — at least 5 conceptual areas. While the physical [distance](https://coupling.dev/posts/dimensions-of-coupling/distance/) is low (same package), the conceptual distance is medium: a developer modifying shell prompt behavior must also understand topic lifecycle timers, pane scanning, and RC debounce logic to safely change this file. The cognitive load exceeds what any single developer can hold in working memory when debugging a polling issue.

### Cascading Changes

Concrete scenarios that force changes to this 1,339-line file:

- **Adding a new provider capability** (e.g., a provider that uses a different terminal status detection method) requires modifying the polling strategy in `update_status_message()` and `_handle_no_status()`.
- **Changing interactive UI detection** requires updating `_parse_with_pyte()`, `_check_interactive_only()`, and the multi-pane scanning logic.
- **Modifying shell prompt markers** requires updating the idle detection logic that calls `has_prompt_marker()` and `setup_shell_prompt()`.
- **Adding a new topic lifecycle state** (beyond active/idle/done/dead) requires changes in `_handle_no_status()`, `_start_autoclose_timer()`, and `_check_autoclose_timers()`.

Because the module is the system's most changed file (nearly every feature addition touches it), these cascading changes compound: any feature PR is likely to conflict with other in-flight work.

### Recommended Improvement

Decompose into focused polling strategies, each owning its domain knowledge:

1. **Terminal status strategy** — provider status parsing, pyte screen buffering, RC state, spinner detection
2. **Interactive UI strategy** — interactive prompt scanning, multi-pane alerts, interactive mode coordination
3. **Topic lifecycle strategy** — autoclose timers, dead window detection, topic existence probing, unbound window TTL
4. **Shell relay strategy** — passive shell output monitoring (already partially extracted to `shell_capture.py`)

A thin polling coordinator (< 200 lines) iterates bindings and delegates to each strategy. Each strategy owns its module-level state and presents a single `poll(window_id, ...) -> StatusResult` interface.

**Trade-off**: This introduces 4-5 new modules and requires defining the interfaces between them. The cost is justified because the current file is a [change amplifier](https://coupling.dev/posts/core-concepts/complexity/) — the decomposition converts one 1,339-line bottleneck into independently modifiable units.

---

## Issue: SessionManager State Accumulation

**Integration**: `session.py` (SessionManager) -> 24 consumer modules
**Severity**: CRITICAL

### Knowledge Leakage

`SessionManager` (1,621 lines, 40+ public methods) manages 6 unrelated data concerns through a single class:

1. **Thread/window bindings** — `thread_bindings`, `window_display_names`, `_window_to_thread` (routing core)
2. **Window state** — `window_states` with `WindowState` dataclass (session tracking)
3. **User read offsets** — `user_window_offsets` (unread detection)
4. **Group chat mappings** — `group_chat_ids` (multi-group routing)
5. **User preferences** — notification modes, approval modes, batch modes (embedded in `WindowState`)
6. **Directory favorites** — `user_dir_favorites` (UI preference)

All 24 consumers import the `session_manager` singleton and gain access to the full 40+ method API surface. A handler that only needs `resolve_chat_id()` also has implicit access to `cycle_batch_mode()`, `register_hookless_session()`, and directory favorites manipulation. This creates [functional coupling](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/) at maximum surface area.

### Complexity Impact

[Low cohesion](https://coupling.dev/posts/core-concepts/balance/) makes the class hard to reason about. A developer reading `message_queue.py` to understand message delivery must follow `session_manager.get_window_state()` into a 1,621-line class that also handles directory favorites, audit operations, and hookless session registration. The class has become a state attractor — every new feature that needs persistent per-window or per-user state adds fields and methods here because it's the only persistent state abstraction available.

### Cascading Changes

- **Adding a new persistent setting** (e.g., a per-window "auto-screenshot" preference) requires adding a field to `WindowState`, updating `to_dict()`/`from_dict()` serialization, adding getter/setter methods to `SessionManager`, and modifying any handler that needs the setting.
- **Changing the persistence format** (e.g., migrating from JSON to SQLite) would require rewriting the entire `SessionManager` class, affecting all 24 consumers.
- **Adding a new binding dimension** (e.g., per-pane bindings for agent teams) would require restructuring `thread_bindings` and `_window_to_thread`, affecting every module that iterates bindings.

### Recommended Improvement

Define narrow Protocol interfaces that consumers type-hint against:

```python
class ThreadRouter(Protocol):
    def resolve_window_for_thread(self, user_id: int, thread_id: int) -> str | None: ...
    def bind_thread(self, user_id: int, thread_id: int, window_id: str) -> None: ...
    def unbind_thread(self, user_id: int, thread_id: int) -> None: ...

class WindowStateStore(Protocol):
    def get_window_state(self, window_id: str) -> WindowState: ...
    def get_display_name(self, window_id: str) -> str: ...
```

`SessionManager` implements all protocols. Consumers import and depend on the narrow protocol, not the full class. No physical decomposition needed — just interface segregation.

**Trade-off**: Adds Protocol definitions and changes import patterns across 24 files. This is a moderate refactoring effort, but it prevents the state attractor pattern from accelerating — each new consumer explicitly declares which slice of state it needs, making dependency audits trivial.

---

## Issue: bot.py Dispatch Monolith

**Integration**: `bot.py` -> 30+ handler modules
**Severity**: CRITICAL

### Knowledge Leakage

`bot.py` (2,018 lines) imports from 41 modules and serves as a manual wiring hub. It has accumulated three distinct responsibilities:

1. **Telegram handler registration** — mapping commands and callbacks to handler functions via python-telegram-bot's `Application.add_handler()` (lines 1927-2016).
2. **Callback dispatch** — a cascading prefix-match chain routing callback queries to the correct handler based on `CB_*` string prefixes (the `callback_handler()` function).
3. **Orchestration logic** — provider menu management (LRU caches for scoped menus, lines 209-408), command failure probing (transcript inspection after sending commands, lines 913-983), auto-topic creation for new windows (lines 1665-1713), and lifecycle management (post_init/post_shutdown, lines 1749-1908).

The callback dispatch is [functionally coupled](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/) to every handler: adding a new callback requires importing the handler, importing its prefix constant, adding a prefix match case, and (often) registering a new command handler — a 4-step ceremony in a 2,000-line file.

### Complexity Impact

All 30+ handler imports execute at module load time. An import error in any handler prevents the entire bot from starting. The file is a merge conflict hotspot — any two features adding handlers will conflict in the import section and in `callback_handler()`.

### Cascading Changes

- **Adding a new callback handler** requires 4 changes in bot.py (import handler, import prefix, add match case, register command).
- **Renaming a callback prefix** requires changing both `callback_data.py` and the match case in bot.py.
- **Adding a new bot command** requires adding a `CommandHandler` registration in `create_bot()`.

Each new feature amplifies the file's size and import surface.

### Recommended Improvement

Extract a callback dispatch registry where handlers self-register:

```python
# In each handler module
@callback_registry.register(CB_DIR_SELECT, CB_DIR_BACK, CB_DIR_HOME)
async def handle_directory_callback(update, context): ...
```

`callback_handler()` becomes a 10-line loop over the registry. Handler modules self-register their prefixes, eliminating the import ceremony in bot.py. The orchestration logic (provider menus, command probing, auto-topic creation) should move to dedicated modules, reducing bot.py to ~400 lines of pure registration and lifecycle code.

**Trade-off**: A registry pattern adds one level of indirection. The benefit is that handler modules become self-contained — adding a new handler no longer requires touching bot.py.

---

## Issue: Provider Abstraction Leakage

**Integration**: `bot.py`, `codex_status.py` -> Codex provider internals
**Severity**: SIGNIFICANT

### Knowledge Leakage

The provider protocol (`AgentProvider` + `ProviderCapabilities`) is a well-designed [contract](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/) — providers implement a uniform interface and consumers query capabilities. However, Codex-specific logic has leaked outside the provider boundary:

- `codex_status.py` (237 lines, top-level module) contains Codex-specific JSONL transcript parsing to build status snapshots. It is only used by `bot.py`.
- `bot.py` has `_codex_status_probe_offset()` and `_maybe_send_codex_status_snapshot()` (lines 913-983) — functions that check `provider.capabilities.name == "codex"` to activate a special code path for `/status` and `/stats` commands.
- `interactive_prompt_formatter.py` (245 lines, top-level module) formats Codex-specific interactive prompts. It is only imported by `providers/codex.py`.

This means the orchestration layer (bot.py) has [model-level knowledge](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/) of a specific provider's transcript format and command semantics — exactly the knowledge the provider protocol was designed to encapsulate.

### Complexity Impact

The provider protocol promises that consumers interact with all providers uniformly. The Codex-specific paths in bot.py break this promise: a developer adding a new provider must now audit bot.py for hard-coded provider name checks to understand the full behavioral surface. The `codex_status.py` module existing at the top level (alongside provider-agnostic modules) is misleading — it looks like core infrastructure but is actually a single-provider helper.

### Cascading Changes

- **Adding a similar status snapshot for another provider** (e.g., Gemini) would require duplicating the `_maybe_send_codex_status_snapshot` pattern in bot.py, creating more provider-specific branches in the orchestration layer.
- **Changing Codex transcript format** requires changes in both `providers/codex.py` and the top-level `codex_status.py` — two files in different packages that share [model-level knowledge](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/) of the same data format.

### Recommended Improvement

Move the provider-specific logic behind the provider protocol:

1. Move `codex_status.py` into `providers/` and have `CodexProvider` expose status snapshot building through a new optional protocol method (e.g., `build_status_fallback(transcript_path, ...) -> str | None`).
2. Move `interactive_prompt_formatter.py` into `providers/` (it's already only used by `CodexProvider`).
3. Replace the `capabilities.name == "codex"` check in bot.py with a capability query (e.g., `capabilities.supports_status_fallback`).

**Trade-off**: Adds one optional method to the `AgentProvider` protocol. This is minimal cost for restoring the protocol's design intent — consumers should never need to know which provider they're talking to.

---

## Issue: Shell Prompt Implicit Contract

**Integration**: `providers/shell.py` -> `handlers/shell_capture.py`, `handlers/status_polling.py`
**Severity**: SIGNIFICANT

### Knowledge Leakage

The shell prompt marker (`⌘N⌘` in wrap mode, `{prefix}:N❯` in replace mode) is a critical [integration contract](https://coupling.dev/posts/dimensions-of-coupling/integration-strength/) between three modules:

- `shell.py` defines and injects the marker, exposes `match_prompt()` returning a raw `re.Match`.
- `shell_capture.py` imports `match_prompt()` and extracts exit codes and sequence numbers by accessing regex groups positionally (group 1 = sequence number, group 2 = trailing text).
- `status_polling.py` calls `has_prompt_marker()` and `setup_shell_prompt()` to detect idle state and trigger marker recovery.

The `match_prompt()` function is the single source of truth for the regex, which is good. But the _semantics_ of the match groups are implicit — `shell_capture.py` assumes group(1) is always the exit code integer and group(2) is always the command echo. These assumptions are not enforced at the API level and will break silently if the marker format evolves.

### Complexity Impact

The shell subsystem is among the most actively developed areas of the codebase (wrap/replace modes, multi-shell support, lazy recovery). Each change to the prompt format carries risk of silent breakage in output extraction — no type error, no test failure until runtime. This is [accidental volatility](https://coupling.dev/posts/dimensions-of-coupling/volatility/) introduced by the implicit contract, not by domain requirements.

### Cascading Changes

- **Adding a timestamp to the marker** would change the group positions, breaking `shell_capture.py`'s exit code extraction.
- **Switching prompt modes** (wrap <-> replace) changes the regex but not the group semantics — today. A future mode could break the assumption.
- **Adding a new marker field** (e.g., shell PID) would shift group indices in one mode but not the other.

### Recommended Improvement

Replace the raw `re.Match` return with a typed dataclass:

```python
@dataclass(frozen=True)
class PromptMatch:
    sequence_number: int
    trailing_text: str
    raw_line: str
```

`shell.py` exports `match_prompt() -> PromptMatch | None`. Consumers access named fields instead of positional groups. Adding new fields to the marker only requires updating `PromptMatch` and the parsing logic in `shell.py` — consumers that don't use the new fields are unaffected.

**Trade-off**: One dataclass and a minor function signature change. Minimal cost, eliminates a class of silent runtime failures.

---

_This analysis was performed using the [Balanced Coupling](https://coupling.dev) model by [Vlad Khononov](https://vladikk.com)._
