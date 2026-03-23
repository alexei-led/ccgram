# ps-based provider detection for tmux panes

## Overview

Replace basename-only `pane_current_command` detection with `ps -t <pane_tty>` foreground process group detection. All three CLIs (claude, codex, gemini) are Node.js scripts — `pane_current_command` shows `bun` or `node` instead of the CLI name. This breaks provider auto-detection when a user launches a CLI from within a shell session (provider stays "shell", messages route through NL→command pipeline instead of direct forwarding).

The new approach:

1. Get `pane_tty` from tmux (e.g., `/dev/ttys003`)
2. Run `ps -t <tty> -o pid=,pgid=,stat=,args=`
3. Filter for `+` in stat → foreground process group
4. Find group leader (pid == pgid) → match full args against provider patterns
5. Cache by `(window_id, fg_pgid)` — only re-classify when PGID changes

Supplements: pane-title detection kept as fallback, title stamping on launch for instant re-detection.

## Context (from discovery)

**Files involved:**

- `src/ccgram/tmux_manager.py` — TmuxWindow dataclass (line 91), `list_windows()` (line 161), `_find_foreign_window()` (line 244), `_scan_session_windows()` (line 725)
- `src/ccgram/providers/__init__.py` — `detect_provider_from_command()` (line 113), `should_probe_pane_title_for_provider_detection()` (line 139), `detect_provider_from_runtime()` (line 148)
- `src/ccgram/providers/base.py` — `AgentProvider` protocol (line 127), `requires_pane_title_for_detection()` (line 246), `detect_from_pane_title()` (line 250)
- `src/ccgram/providers/gemini.py` — `_GEMINI_WRAPPER_COMMANDS` (line 158), `needs_pane_title_for_detection()` (line 169), `detect_gemini_from_runtime()` (line 174)
- `src/ccgram/providers/claude.py` — detection stubs return False (line 185)
- `src/ccgram/providers/_jsonl.py` — base detection stubs (line 216)
- `src/ccgram/providers/shell.py` — `_KNOWN_SHELLS` (line 20)
- `src/ccgram/handlers/status_polling.py` — `_maybe_discover_transcript()` (line 1009)
- `src/ccgram/bot.py` — `_handle_new_window()` (line 1511)
- `tests/ccgram/test_provider_autodetect.py` — existing detection tests

**Live data (verified on running system):**

- Claude Code: `pane_current_command=bun`, `ps -t` leader: `bun /Users/alexei/.bun/bin/claude` (stat=S+)
- cc-team wrapper: leader: `bun /Users/alexei/.bun/install/global/node_modules/cc-team/cli.js`
- Shell: `pane_current_command=bash`, leader: `-fish` (stat=Ss background) or `bash ./scripts/restart.sh` (stat=S+ foreground)
- All foreground processes have `+` in stat; background shell has `Ss` (no `+`)
- TIOCGPGRP ioctl fails on macOS from outside the process group (ENOTTY) — ruled out

**Key patterns:**

- libtmux exposes `pane.pane_current_command`, `pane.pane_current_path`, `pane.pane_width`, `pane.pane_height` — `pane_tty` should also be available
- Provider detection has two integration points: `_handle_new_window()` (bot.py) and `_maybe_discover_transcript()` (status_polling.py)
- Detection is provider-driven: each provider declares `requires_pane_title_for_detection()` and `detect_from_pane_title()`

## Development Approach

- **Testing approach**: Regular (implement first, then test)
- Complete each task fully before moving to the next
- Make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes
- **CRITICAL: all tests must pass before starting next task**
- **CRITICAL: update this plan file when scope changes during implementation**
- Run `make fmt && make test && make lint` after each change

## Testing Strategy

- **Unit tests**: Mock `subprocess` calls to `ps` with known output; test classification logic with real argv samples from live system
- **Integration tests**: Test with actual tmux panes if feasible
- **Edge cases**: wrapper scripts (ce, cc-mirror), exec vs spawn, empty TTY, permission errors, malformed ps output

## Progress Tracking

- Mark completed items with `[x]` immediately when done
- Add newly discovered tasks with ➕ prefix
- Document issues/blockers with ⚠️ prefix
- Update plan if implementation deviates from original scope

## Implementation Steps

### Task 1: Add `pane_tty` to TmuxWindow dataclass

**Files:**

- Modify: `src/ccgram/tmux_manager.py`

- [x] Add `pane_tty: str = ""` field to `TmuxWindow` dataclass (after `pane_current_command`)
- [x] In `_sync_list_windows()`, read `pane.pane_tty` from libtmux and pass to TmuxWindow constructor
- [x] In `_find_foreign_window()`, add `#{pane_tty}` to the tmux format string and parse it
- [x] In `_scan_session_windows()`, add `#{pane_tty}` to the tmux format string and parse it
- [x] Verify existing tests still pass: `make test`

### Task 2: Implement ps-based foreground process detection

**Files:**

- Create: `src/ccgram/providers/process_detection.py`
- Create: `tests/ccgram/test_process_detection.py`

- [x] Create `process_detection.py` with module docstring explaining the approach
- [x] Implement `get_foreground_args(tty_path: str) -> str` — runs `ps -t <tty> -o pid=,pgid=,stat=,args=`, filters for `+` in stat, finds group leader (pid==pgid), returns full args string. Returns empty string on any error (permission, timeout, no processes)
- [x] Implement `classify_provider_from_args(args: str) -> str` — tokenizes args, skips wrapper tokens (`sudo`, `env`, `node`, `bun`, `npx`, `bunx`, `uv`, `python`, `python3`), matches remaining tokens against provider patterns. Returns provider name or empty string
- [x] Provider patterns: claude (`claude`, `ce`, `cc-mirror`, `zai`, `cc-team`), codex (`codex`), gemini (`gemini`), shell (reuse `_KNOWN_SHELLS` from shell.py)
- [x] Implement `detect_provider_from_tty(tty_path: str) -> str` — combines `get_foreground_args()` + `classify_provider_from_args()`. This is the main entry point
- [x] Write tests for `classify_provider_from_args()` with real argv samples: `bun /Users/x/.bun/bin/claude`, `bun /path/to/codex.js`, `node /path/to/gemini/dist/index.js`, `-fish`, `bash ./script.sh`, `sudo codex`, `env node /path/to/claude`
- [x] Write tests for `get_foreground_args()` mocking subprocess — normal output, empty output, permission error, timeout
- [x] Write tests for `detect_provider_from_tty()` end-to-end with mocked subprocess
- [x] Run tests: `make fmt && make test && make lint`

### Task 3: Add PGID-based caching

**Files:**

- Modify: `src/ccgram/providers/process_detection.py`
- Modify: `tests/ccgram/test_process_detection.py`

- [x] Add module-level cache: `_pgid_cache: dict[str, tuple[int, str]]` mapping `window_id → (fg_pgid, provider_name)`
- [x] Implement `detect_provider_cached(window_id: str, tty_path: str) -> str` — gets foreground PGID from ps output, checks cache, only re-classifies when PGID changes
- [x] Add `clear_detection_cache(window_id: str | None = None)` — clears one entry or all (for cleanup)
- [x] Write tests for cache hit/miss behavior
- [x] Write tests for cache invalidation on PGID change
- [x] Run tests: `make fmt && make test && make lint`

### Task 4: Integrate ps-based detection into provider detection chain

**Files:**

- Modify: `src/ccgram/providers/__init__.py`
- Modify: `tests/ccgram/test_provider_autodetect.py`

- [x] Add new function `detect_provider_from_pane(pane_current_command: str, *, pane_tty: str = "", window_id: str = "") -> str` — fast path: try `detect_provider_from_command()` first; if empty and `pane_tty` provided and command is a known runtime (`node`, `bun`, `npx`, `bunx`), call `detect_provider_cached(window_id, pane_tty)`; if still empty, fall through to pane-title path
- [x] Define `_JS_RUNTIMES = frozenset({"node", "bun", "npx", "bunx"})` for gating the ps-based path
- [x] Update exports in `__all__`
- [x] Write tests for `detect_provider_from_pane()` — fast path hit (command=codex), runtime trigger (command=bun, tty provided), no tty fallback, unknown command
- [x] Run tests: `make fmt && make test && make lint`

### Task 5: Wire detection into status polling and bot.py

**Files:**

- Modify: `src/ccgram/handlers/status_polling.py`
- Modify: `src/ccgram/bot.py`
- Modify: `tests/ccgram/test_status_polling.py`
- Modify: `tests/ccgram/test_provider_autodetect.py`

- [x] In `_maybe_discover_transcript()` (status_polling.py:1049): replace `detect_provider_from_command(w.pane_current_command)` with `detect_provider_from_pane(w.pane_current_command, pane_tty=w.pane_tty, window_id=window_id)`. Keep the existing pane-title fallback path after (for when ps-based also fails)
- [x] In `_handle_new_window()` (bot.py:1533): same replacement — use `detect_provider_from_pane()` with pane_tty
- [x] Add `clear_detection_cache(window_id)` calls in cleanup paths (topic close, window kill)
- [x] Update existing tests in `test_status_polling.py` to account for new detection flow
- [x] Update existing tests in `test_provider_autodetect.py` for `_handle_new_window` changes
- [x] Write new test: shell topic → pane_current_command changes to `bun` → ps-based detection identifies codex → provider switches
- [x] Run tests: `make fmt && make test && make lint`

### Task 6: Add pane-title detection for Claude Code spinner

**Files:**

- Modify: `src/ccgram/providers/claude.py`
- Modify: `tests/ccgram/test_provider_autodetect.py`

- ⚠️ Skipped — ps-based detection is the primary mechanism; Claude also has hooks for explicit provider setting. Adding spinner character detection risks false positives with other braille-using tools.

### Task 7: Add title stamping on provider launch

**Files:**

- Modify: `src/ccgram/tmux_manager.py`
- Modify: `src/ccgram/handlers/directory_callbacks.py` (or wherever provider launch happens)

- [x] Add `stamp_pane_title(window_id: str, provider_name: str)` to TmuxManager — sends `printf '\033]2;ccgram:<provider>\007'` via send_keys before the actual CLI launch command
- [x] Call `stamp_pane_title()` in the directory browser flow when creating a new topic with a specific provider
- [x] Add `ccgram:` prefix check in pane-title detection — if title starts with `ccgram:`, extract provider name directly (instant detection, no ps needed)
- [x] Write tests for title stamping and `ccgram:` prefix detection
- [x] Run tests: `make fmt && make test && make lint`

### Task 8: Verify acceptance criteria

- [x] Verify: all 2208 tests pass
- [ ] Verify: launching codex from shell topic via `!codex` triggers provider switch within 1 polling cycle (manual)
- [ ] Verify: exiting codex back to shell triggers switch back to shell provider (manual)

### Task 9: [Final] Update documentation

- [ ] Update CLAUDE.md provider detection section if needed
- [ ] Update architecture.md module inventory with new `process_detection.py`
- [ ] Move this plan to `docs/plans/completed/`

## Technical Details

### ps output format and parsing

```
ps -t /dev/ttys003 -o pid=,pgid=,stat=,args=
```

Sample output (real data):

```
 8617  8617 Ss   -fish                          ← background shell (no +)
 8668  8668 S+   bun /Users/alexei/.bun/bin/claude  ← FG group leader (pid==pgid, has +)
 8690  8668 S+   bun /var/folders/.../context7-mcp   ← FG member (pgid==8668)
```

**Foreground detection**: `+` in stat field (e.g., `S+`, `R+`, `T+`)
**Group leader**: pid == pgid among foreground processes
**Classification**: Skip wrapper tokens in args, match first meaningful token

### Classification token matching

```python
WRAPPER_TOKENS = {"sudo", "env", "node", "bun", "npx", "bunx", "uv", "python", "python3"}

# Match basename of first non-wrapper token in args
# "bun /Users/x/.bun/bin/claude" → skip "bun" → basename("/Users/x/.bun/bin/claude") → "claude" ✓
# "bun /path/to/cc-team/cli.js" → skip "bun" → path contains "cc-team" → "claude" ✓
```

### Detection priority chain

```
1. Fast path: detect_provider_from_command(pane_current_command)
   → Direct basename match (claude/codex/gemini/shell)
   → Returns immediately if matched

2. ps-based (new): detect_provider_cached(window_id, pane_tty)
   → Only when command is JS runtime (node/bun/npx/bunx)
   → Cached by PGID — subprocess only on PGID change
   → Returns if matched

3. Pane title (existing + extended): detect_provider_from_runtime()
   → Gemini markers (✦/✋/◇)
   → Claude spinners (braille chars)
   → ccgram: prefix (from title stamping)
   → Returns if matched

4. No match → empty string → provider unchanged
```

### Cache structure

```python
# window_id → (foreground_pgid, detected_provider_name)
_pgid_cache: dict[str, tuple[int, str]] = {}

# On each poll:
# 1. Parse ps output to get current FG PGID
# 2. If PGID matches cache → return cached provider (no classification)
# 3. If PGID changed → re-classify from args → update cache
```

## Post-Completion

**Manual verification:**

- Test with real shell topic → launch `codex` → verify provider switches
- Test with real shell topic → launch `claude` → verify provider switches
- Test provider switch back when CLI exits
- Verify no performance regression with `ps` calls on every poll cycle
