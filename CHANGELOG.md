# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2026-06-22

### Added

- herdr backend: tab-identity model — one ccgram topic = one herdr tab (`window_id = tab_id`)
- herdr `/new` flow: workspace picker step before agent selection when using herdr backend
- herdr topic labels: adaptive `<workspace> ▸ <tab>` format stamped into `WindowRef.window_name`
- herdr session re-resolution: stale `tab_id` re-mapped via shared `session_id` after herdr restart
- Bounded self-heal: startup rescan + per-poll `_emit_known_unbound_window_events` for tabs missed by hooks
- `Multiplexer` Protocol: `WorkspaceRef` dataclass and `list_workspaces()` method for workspace enumeration
- Non-colon pane delimiter (`CB_PANE_DELIMITER = "|"`) for herdr ids in callback data
- herdr multi-pane awareness: `_active_pane(tab_id)` resolution for all pane ops; `list_panes` enumerates full tab

### Changed

- herdr session-map keys use `herdr:<tab_id>` format (e.g. `herdr:w2:t1`)
- herdr hook identity: resolves `$HERDR_PANE_ID` → tab_id via `herdr pane info` before writing `session_map.json`
- `session_map_prefix()` and `session_map_prefix_for()` are canonical key builders — all call sites use them
- `__*__`-named workspaces/tabs skipped by herdr backend (self-hosting escape hatch)

### Fixed

- Warning log emitted when herdr socket is unavailable at hook fire time (session no longer silently unmonitored)
- `status_cmd.py` now uses canonical `session_map_prefix_for()` instead of a local duplicate

### Documentation

- Architecture doc: herdr tab-identity design revision
- `docs/guides.md`: updated herdr topic labels, workspace picker, `__*__` escape hatch

## [3.6.0] - 2026-06-21

### Added

- Multiplexer seam + herdr backend support ([#114](https://github.com/alexei-led/ccgram/pull/114))

### Documentation

- Add remote development Slidev deck
- Fix topic terminology in slides
- Remove CLAUDE.md

## [3.5.2] - 2026-06-02

### Added

- Add Pi followup command

### Documentation

- Label Caddy config fence
- Update CHANGELOG.md for v3.5.2

### Fixed

- Refresh stale Pi transcript paths
- Align Pi slash command handling
- Ignore stale dead autoclose timers

## [3.5.1] - 2026-05-31

### Documentation

- Update CHANGELOG.md for v3.5.1

## [3.5.0] - 2026-05-31

### Changed

- Remove inter-agent messaging subsystem ([#106](https://github.com/alexei-led/ccgram/pull/106))

### Documentation

- Update CHANGELOG.md for v3.5.0

## [3.4.1] - 2026-05-23

### Documentation

- Update CHANGELOG.md for v3.4.1

### Fixed

- Audit and cut log noise; per-level colors ([#98](https://github.com/alexei-led/ccgram/pull/98))

## [3.4.0] - 2026-05-23

### Added

- Window-state feature ports + /agent provider override ([#101](https://github.com/alexei-led/ccgram/pull/101))

### Documentation

- Update CHANGELOG.md for v3.4.0

## [3.3.3] - 2026-05-23

### Changed

- Cut session_map.json reads + collapse repetitive poll-loop logs

### Documentation

- Update CHANGELOG.md for v3.3.3

## [3.3.2] - 2026-05-23

### Changed

- Unify tool-call formatting + ephemeral bubble UX fixes ([#100](https://github.com/alexei-led/ccgram/pull/100))

### Documentation

- Update CHANGELOG.md for v3.3.2

## [3.3.1] - 2026-05-22

### Added

- Ephemeral tool-progress mode with ×N dedup, show tool calls by default

### Documentation

- Update CHANGELOG.md for v3.3.1

## [3.3.0] - 2026-05-22

### Added

- Readable viewport screenshots, /last, status-bar + toolbar revamp ([#96](https://github.com/alexei-led/ccgram/pull/96))

### Documentation

- Slim CLAUDE.md and architecture rules
- Slim docs/ai-agents/ — strip duplication with CLAUDE.md
- Align human-facing docs with current code state
- Update CHANGELOG and guides/architecture for v3.3.0

## [3.2.0] - 2026-05-21

### Documentation

- Update CHANGELOG.md for v3.2.0

## [3.1.3] - 2026-05-21

### Added

- Add Pi to provider picker

### Documentation

- Update CHANGELOG.md for v3.1.3
- Update CHANGELOG.md for v3.1.3

### Fixed

- Parenthesize except clause in verify_hooks_installed ([#90](https://github.com/alexei-led/ccgram/pull/90))

## [3.1.2] - 2026-05-16

### Documentation

- Update CHANGELOG.md for v3.1.2

### Fixed

- Register signal handlers via asyncio loop

## [3.1.1] - 2026-05-16

### Added

- Warn once for externally-launched Gemini windows ([#86](https://github.com/alexei-led/ccgram/pull/86))

### Documentation

- Normalize markdown formatting in CHANGELOG, CLAUDE.md, and architecture docs
- Update CHANGELOG.md for v3.1.1

### Fixed

- Strip cc-thingz hook-runner log lines from pane capture
- Gate Claude pyte chrome parsing to Claude provider
- Pending-creation race guard for directory flow ([#79](https://github.com/alexei-led/ccgram/pull/79))

## [3.1.0] - 2026-05-16

### Added

- RC remote-control feedback + git-worktree topics (v3.1.0) ([#84](https://github.com/alexei-led/ccgram/pull/84))

### Documentation

- Update CHANGELOG.md for v3.1.0

### Fixed

- Extract question from boxed prompts ([#83](https://github.com/alexei-led/ccgram/pull/83))

## [3.0.9] - 2026-05-15

### Documentation

- Update CHANGELOG.md for v3.0.9

### Fixed

- Prefer transcript-path provider over stale claim ([#81](https://github.com/alexei-led/ccgram/pull/81))
- Treat .claude<suffix>/projects/ as Claude ([#82](https://github.com/alexei-led/ccgram/pull/82))

## [3.0.8] - 2026-05-14

### Added

- Multi-provider hook support (Codex, Gemini, Pi) ([#80](https://github.com/alexei-led/ccgram/pull/80))

### Documentation

- Update CHANGELOG.md for v3.0.8

### Fixed

- Update hide_tool_calls default assertion to true

## [3.0.7] - 2026-05-04

### Added

- Add OpenAI TTS backend

### Documentation

- Update README and CLAUDE.md for OpenAI TTS and hide_tool_calls default
- Update CHANGELOG.md for v3.0.7

## [3.0.6] - 2026-05-04

### Documentation

- Update CHANGELOG.md for v3.0.6

### Fixed

- Avoid focusing created restart tmux window

## [3.0.5] - 2026-05-03

### Added

- Add Edge TTS voice replies ([#71](https://github.com/alexei-led/ccgram/pull/71))

### Documentation

- Update CHANGELOG.md for v3.0.5

## [3.0.4] - 2026-05-03

### Documentation

- Update CHANGELOG.md for v3.0.4

### Fixed

- Exclude codex_exec sessions from primary detection ([#73](https://github.com/alexei-led/ccgram/pull/73))

## [3.0.3] - 2026-05-03

### Documentation

- Update docs for CCGRAM_STATUS_MODE, tool-call visibility, and Gemini JSONL support
- Add modularity review and round-4 decouple plan
- Update CHANGELOG.md for v3.0.3

## [3.0.2] - 2026-04-29

### Added

- Add toggle to suppress tool-call messages ([#65](https://github.com/alexei-led/ccgram/pull/65))
- Add CCGRAM_STATUS_MODE for configurable topic emoji color scheme ([#68](https://github.com/alexei-led/ccgram/pull/68))

### Documentation

- Update CHANGELOG.md for v3.0.2

### Fixed

- Support JSONL transcripts and add /status snapshot ([#66](https://github.com/alexei-led/ccgram/pull/66))

## [3.0.1] - 2026-04-28

### Documentation

- Update CHANGELOG.md for v3.0.1

### Fixed

- Detect and drop hooks fired by nested claude instances

## [2.11.3] - 2026-04-26

### Documentation

- Update CHANGELOG.md for v2.11.3

### Fixed

- Preserve primary binding under nested SessionStart ([#63](https://github.com/alexei-led/ccgram/pull/63))

## [2.11.2] - 2026-04-26

### Documentation

- Update CHANGELOG.md for v2.11.2

### Fixed

- Preserve manually created tmux windows

## [2.11.1] - 2026-04-24

### Documentation

- Update CHANGELOG.md for v2.11.1

## [2.11.0] - 2026-04-19

### Added

- Add pi coding agent provider ([#59](https://github.com/alexei-led/ccgram/pull/59))

### Documentation

- Clean up design and modularity review artifacts [skip ci]
- Update CHANGELOG.md for v2.11.0

## [2.10.0] - 2026-04-16

### Changed

- Architecture round 2 — modularity fixes + target design ([#57](https://github.com/alexei-led/ccgram/pull/57))

### Documentation

- Update CHANGELOG.md for v2.10.0

## [2.9.0] - 2026-04-13

### Added

- /send command + TOML-configurable toolbar + modularity refactor ([#56](https://github.com/alexei-led/ccgram/pull/56))

### Documentation

- Update CHANGELOG.md for v2.9.0

## [2.8.2] - 2026-04-09

### Added

- Enhance Telegram UX — reduce noise, delays, and flood ([#54](https://github.com/alexei-led/ccgram/pull/54))

### Documentation

- Update CHANGELOG.md for v2.8.2

### Fixed

- Handle missing autoclose topics and zsh shell markers ([#53](https://github.com/alexei-led/ccgram/pull/53))

## [2.8.1] - 2026-04-07

### Added

- Enable PTB built-in rate limiter for Telegram API calls

### Documentation

- Sync documentation with v2.8.0 codebase changes
- Update CHANGELOG.md for v2.8.1

## [2.8.0] - 2026-04-07

### Added

- Terminal live view with auto-refreshing screenshots ([#52](https://github.com/alexei-led/ccgram/pull/52))

### Documentation

- Update CHANGELOG.md for v2.8.0

## [2.7.2] - 2026-04-06

### Documentation

- Update CHANGELOG.md for v2.7.2

### Fixed

- Disable interactive editors in agent windows

## [2.7.1] - 2026-04-04

### Documentation

- Update CHANGELOG.md for v2.7.1

### Fixed

- Use resilient HTTPX requests for all bot traffic ([#49](https://github.com/alexei-led/ccgram/pull/49))

## [2.7.0] - 2026-04-04

### Added

- Make Telegram output more informative with two-tier approach
- Add ccgram-messaging skill for inter-agent collaboration

### Documentation

- Add inter-agent messaging guide and README feature section [skip ci]
- Replace ASCII diagrams with styled Mermaid in messaging guide [skip ci]
- Add informative output implementation plan
- Update CHANGELOG.md for v2.7.0

### Fixed

- Auto-accept YOLO bypass permissions prompt on Claude window creation
- Tighten vim INSERT mode detection to avoid Claude status bar false positives
- Raise chrome line length limit to 250 for wide Claude status bars
- Remove noisy debug log on non-interactive pane capture
- Address code review findings for informative output

## [2.6.1] - 2026-04-02

### Added

- Auto-install messaging skill on Claude window creation

### Documentation

- Update CHANGELOG.md for v2.6.1

### Fixed

- Broaden session monitor exception handling to log all errors

## [2.6.0] - 2026-04-01

### Added

- Add file-based mailbox for inter-agent messaging
- Add peer discovery with declared overlay and filtering
- Add ccgram msg CLI subcommand group for inter-agent messaging
- Add messaging config extensions to Config singleton
- Add broker delivery with idle detection, rate limiting, and lifecycle integration
- Add Telegram notifications for inter-agent messaging
- Add Mailbox.broadcast() method for filtered broadcast messaging
- Add agent spawning with Telegram approval flow
- Add messaging skill auto-installation for Claude Code agents
- Add window self-identification via CCGRAM_WINDOW_ID env var
- Verify acceptance criteria and add deadlock prevention for --wait
- Add inter-agent messaging documentation
- Fix 4 per-topic state cleanup gaps in cleanup.py
- Promote mailbox private helpers to public API
- Promote 3 private functions to public API across module boundaries
- Replace shared mutable \_pending_requests dict with public accessor API
- Extract msg_delivery.py to break msg_broker ↔ msg_telegram circular dependency
- Add public methods to TerminalStatusStrategy; promote polling constants
- Extract UserPreferences from SessionManager (starred/MRU/offsets)
- Narrow SessionManager — remove 9 dead pass-throughs, add Protocols, add export_window_info
- Add TopicStateRegistry with self-registration pattern, dual-path cleanup
- Migrate topic-scoped handlers to self-register with TopicStateRegistry
- Migrate window-scoped handlers to self-register with TopicStateRegistry
- Migrate chat/qualified-scoped handlers to self-register with TopicStateRegistry
- Finalize TopicStateRegistry — simplify cleanup.py, ensure all registrations at startup
- Verify all modularity refactoring acceptance criteria
- Update documentation for modularity refactoring round 2
- Render TaskCreate batches as task lists
- Improve task tool batch rendering
- Add inter-agent messaging system and modularity refactoring

### Changed

- Move TopicStateRegistry to top-level ccgram package
- Extract periodic tasks and transcript discovery from polling coordinator
- Add strategy-level query methods to polling strategies
- Encapsulate delivery path in Mailbox and clean up messaging API
- Decompose SessionManager god object and fix modularity violations

### Documentation

- Add inter-agent messaging implementation plan
- Revise inter-agent messaging implementation plan
- Revise inter-agent messaging implementation plan
- Update modularity review for fourth iteration (2026-03-31)
- Update architecture docs and design specs for modularity refactoring
- Update CHANGELOG.md for v2.6.0

### Fixed

- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings
- Address codex review findings
- Address code review findings
- Address code review findings
- Show live Claude task board in status bubble
- Broaden task status normalization to accept common variants
- Re-inject stale pending messages on crash recovery instead of dropping them
- Catch ImportError in topic state registry safe cleanup handler
- Replace hardcoded startup_time with relative time.monotonic() - 31.0

## [2.5.0] - 2026-03-30

### Documentation

- Update CHANGELOG.md for v2.5.0

## [2.4.1] - 2026-03-28

### Documentation

- Update CHANGELOG.md for v2.4.1

### Fixed

- Handle NetworkError as transient in bot error handler
- Eliminate status message and topic rename noise on Stop events ([#46](https://github.com/alexei-led/ccgram/pull/46))

## [2.4.0] - 2026-03-28

### Added

- Detect and recreate deleted Telegram topics via /sync ([#45](https://github.com/alexei-led/ccgram/pull/45))

### Documentation

- Update CHANGELOG.md for v2.4.0

## [2.3.5] - 2026-03-28

### Documentation

- Update CHANGELOG.md for v2.3.5

### Fixed

- Is_general_topic fails when message_thread_id is None in forum groups ([#43](https://github.com/alexei-led/ccgram/pull/43))

## [2.3.4] - 2026-03-27

### Added

- Reduce General topic noise with pin-once + reaction fallback
- Reduce General topic noise with pin-once + reaction fallback ([#41](https://github.com/alexei-led/ccgram/pull/41))
- Add shutdown notification and signal diagnostics
- Hide underscore-prefixed tmux windows from window list

### Changed

- Remove unnecessary **future** annotations import

### Documentation

- Align provider emoji in README diagram with code [skip ci]
- Update CHANGELOG.md for v2.3.4

### Fixed

- Guard General topic handler with is_general_topic check
- Harden service resilience against crashes and silent degradation ([#42](https://github.com/alexei-led/ccgram/pull/42))

## [2.3.3] - 2026-03-26

### Added

- /release skill with LLM-crafted notes, portable project settings [skip ci]

### Documentation

- Update CHANGELOG.md for v2.3.3

### Fixed

- Detect interactive UI during message queue backlog ([#33](https://github.com/alexei-led/ccgram/pull/33))

## [2.3.2] - 2026-03-26

### Documentation

- Update CHANGELOG.md for v2.3.2

### Fixed

- Idempotent prompt markers, raw send_keys, POSIX fallback

## [2.3.1] - 2026-03-25

### Added

- Wrap prompt mode — preserve user's prompt (Tide, Starship, P10k)

### Documentation

- Update README and guides for shell provider [skip ci]
- Update CHANGELOG.md for v2.3.1

## [2.3.0] - 2026-03-24

### Added

- Shell provider — chat-first shell interface via Telegram ([#36](https://github.com/alexei-led/ccgram/pull/36))

### Documentation

- Update CHANGELOG.md for v2.3.0

## [2.2.5] - 2026-03-23

### Documentation

- Update CHANGELOG.md for v2.2.5

### Fixed

- Persist group routing on topic rebind ([#35](https://github.com/alexei-led/ccgram/pull/35))
- Recover stale provider mappings from transcript path

## [2.2.4] - 2026-03-20

### Added

- Switch to entity-based Telegram formatting ([#34](https://github.com/alexei-led/ccgram/pull/34))

### Documentation

- Update CHANGELOG.md for v2.2.4

## [2.2.3] - 2026-03-20

### Documentation

- Update CHANGELOG.md for v2.2.3

### Fixed

- Respect Telegram cooldown period and log version at startup

## [2.2.2] - 2026-03-20

### Documentation

- Update CHANGELOG.md for v2.2.2

### Fixed

- Handle Telegram flood control during startup command registration

## [2.2.1] - 2026-03-20

### Added

- Subagent context binding ([#32](https://github.com/alexei-led/ccgram/pull/32))

### Documentation

- Update CHANGELOG.md for v2.2.1

## [2.2.0] - 2026-03-20

### Added

- Smart notification batching for tool call chains ([#31](https://github.com/alexei-led/ccgram/pull/31))

### Documentation

- Update release process in CLAUDE.md [skip ci]
- Update CHANGELOG.md for v2.2.0

## [2.1.2] - 2026-03-20

### Documentation

- Update CHANGELOG.md for v2.1.1 [skip ci]
- Update CHANGELOG.md for v2.1.2 [skip ci]

### Fixed

- Update actions/checkout to v6, drop changelog push to protected main

## [2.1.1] - 2026-03-20

### Added

- Ack reactions on forwarded messages + assert_sendable guard ([#30](https://github.com/alexei-led/ccgram/pull/30))

### Documentation

- Update CHANGELOG.md for v2.1.0 [skip ci]

### Fixed

- Install Claude hooks via current interpreter ([#29](https://github.com/alexei-led/ccgram/pull/29))

## [2.1.0] - 2026-03-19

### Added

- Generalize external session discovery beyond emdash ([#27](https://github.com/alexei-led/ccgram/pull/27))
- Add voice message transcription via Whisper API ([#24](https://github.com/alexei-led/ccgram/pull/24))
- Restart run command, ANSI capture, relay cleanup ([#28](https://github.com/alexei-led/ccgram/pull/28))

### Documentation

- Improve BotFather setup instructions and add group ID

### Fixed

- Reset polling client after transport errors ([#26](https://github.com/alexei-led/ccgram/pull/26))

## [2.0.1] - 2026-03-16

### Added

- Auto-detect tmux session and prevent duplicate instances

## [2.0.0] - 2026-03-16

### Added

- Rename ccbot to ccgram (v2.0.0)

## [1.6.12] - 2026-03-14

### Fixed

- Break dead window notification infinite retry loop

## [1.6.11] - 2026-03-13

### Added

- Add emdash integration — auto-discover foreign tmux sessions

## [1.6.10] - 2026-03-12

### Fixed

- Upgrade to telegramify-markdown 1.0.0, add deptry for dep hygiene

## [1.6.9] - 2026-03-09

### Changed

- Remove dead code and legacy artifacts
- Consolidate module-level state in status_polling.py
- Extract retry-with-fallback helper in message_sender.py
- Add Protocol methods for Gemini pane-title detection
- Optimize polling loop with O(1) window lookup

### Fixed

- Ruff format status_polling.py
- Prevent unbounded state growth in long-running process
- Cache converted text in retry helper, hoist deferred import
- Remove orphaned poll state accumulation, tighten tests

## [1.6.8] - 2026-03-08

### Added

- Auto-enter INSERT mode when vim NORMAL mode detected

## [1.6.7] - 2026-03-08

### Fixed

- Handle RetryAfter in safe_reply/safe_edit/safe_send with sleep+retry

## [1.6.6] - 2026-03-08

### Added

- Auto-recover dead topics in /restore instead of showing recovery keyboard

## [1.6.5] - 2026-03-08

### Added

- Add /restore command, startup stale topic cleanup, sync improvements

### Fixed

- Register /restore in Telegram bot command menu

## [1.6.3] - 2026-03-05

### Added

- Bidirectional topic↔window name sync

## [1.6.2] - 2026-03-03

### Added

- Harden Gemini provider with launch settings, runtime detection, and hookless session resilience

## [1.6.1] - 2026-03-03

### Added

- Add Gemini transcript discovery, tool parsing, and command detection
- Detect Gemini from pane title when running under bun/node wrappers

### Documentation

- Update Gemini provider docs with transcript discovery and detection details

## [1.6.0] - 2026-03-03

### Added

- Add per-window approval mode with provider-specific YOLO flags

### Documentation

- Update llm.txt and ai-agents docs with ~15 missing modules
- Document YOLO session mode in README and guides

## [1.5.9] - 2026-03-03

### Added

- Add command catalog with provider-agnostic discovery and caching
- Add /commands handler with scoped provider menus and error probing

### Documentation

- Update README with command menu and provider-scoped features

## [1.5.8] - 2026-03-02

### Added

- Add Codex tool formatting parity and refactor tests

## [1.5.7] - 2026-03-02

### Added

- Replace inline query with callback for status history recall

## [1.5.4] - 2026-03-02

### Added

- Register Telegram menu commands from all providers, not just the default
- Improve Codex interactive edit prompt formatting in Telegram

### Fixed

- Strip leading slash from CC name mapping to prevent double-prefixed commands
- Keep idle status visible for hookless providers at shell prompts

## [1.5.1] - 2026-03-02

### Added

- Add /sync command for on-demand state audit and cleanup
- Register missing bot commands in Telegram menu
- Strict bidirectional topic-window enforcement in /sync
- Improve transcript discovery for hookless providers with unknown process names
- Recognize Codex selection UI cursor and action hints

### Fixed

- Use tuple syntax for multi-exception except clauses
- Improve Homebrew formula generation reliability

## [1.5.0] - 2026-03-02

### Added

- Transcript discovery for hookless providers (Codex/Gemini) ([#20](https://github.com/alexei-led/ccgram/pull/20))

## [1.4.5] - 2026-03-02

### Fixed

- Automatic cleanup of stale state entries in state.json

## [1.4.4] - 2026-03-02

### Fixed

- Suspend topic probe after consecutive timeouts to reduce log noise

## [1.4.3] - 2026-03-01

### Fixed

- Throttle repetitive polling debug logs to reduce noise
- Clean up partial-jsonl throttle state on session removal

## [1.4.2] - 2026-03-01

### Added

- Add integration tests for dispatch, monitor, state, and hook pipeline ([#18](https://github.com/alexei-led/ccgram/pull/18))

### Fixed

- V1.4.2 bug fixes — Gemini I/O cache, glob fallback cwd, bash capture tests ([#19](https://github.com/alexei-led/ccgram/pull/19))

## [1.4.1] - 2026-03-01

### Fixed

- Topic name preservation and session discovery without index ([#17](https://github.com/alexei-led/ccgram/pull/17))

## [1.4.0] - 2026-02-27

### Added

- Multi-pane support, team hook events, and hook install UX

## [1.3.3] - 2026-02-27

### Added

- Detect more permission prompts + add /screenshot command

## [1.3.2] - 2026-02-26

### Fixed

- Case-insensitive TOPIC_NOT_MODIFIED check prevents emoji update spam

## [1.3.1] - 2026-02-25

### Added

- Cherry-pick upstream improvements — 6 targeted fixes

### Documentation

- Add CLAUDE_CONFIG_DIR and CCBOT_SHOW_HIDDEN_DIRS to .env.example

## [1.3.0] - 2026-02-25

### Added

- Expand hook system to 5 Claude Code event types

## [1.2.1] - 2026-02-25

### Added

- Improve Telegram message formatting with emoji and visual hierarchy

### Changed

- Migrate to structlog, extract state persistence and window resolver

### Fixed

- Remove incompatible add_logger_name from structlog config

## [1.1.1] - 2026-02-24

### Fixed

- Simplify provider launch commands and clean up dead code

## [1.1.0] - 2026-02-24

### Added

- Add ScreenBuffer abstraction wrapping pyte VT100 emulator
- Add pyte dependency and ScreenBuffer abstraction
- Version-resilient spinner detection via Unicode categories
- Adaptive separator/chrome detection without hardcoded line counts
- Pyte-based screen parsing for interactive UI detection
- Integrate pyte into status polling pipeline
- Fix Gemini single-JSON transcript parsing via whole-file reading
- Verify acceptance criteria for resilient terminal parsing
- Update documentation for resilient terminal parsing

### Fixed

- Harden spinner detection to reject ASCII punctuation
- Gemini resume_id validation and whole-file transcript offset tracking
- ScreenBuffer edge cases and cleanup on topic close
- Auto-clear stale status messages and idle indicators
- Detect edit permission prompts via structural ❯ catch-all
- Remove fragile text parsing and Python 2 except syntax
- Review fixes — dead code removal, typing, and correctness

## [1.0.1] - 2026-02-22

### Added

- Add per-window provider_name to WindowState and get_provider_for_window()
- Replace global get_provider() with per-window resolution across all handlers
- Add provider selection to directory browser UI
- Auto-detect provider for externally created tmux windows
- Use per-window provider in recovery, resume, and sessions dashboard
- Verify acceptance criteria for per-window provider support
- Update documentation for per-window provider support
- Mark Task 2 complete — per-window provider resolution already in place
- Mark Task 3 complete — provider selection UI already in place
- Mark Task 4 complete — provider auto-detection already in place
- Mark Task 5 complete — recovery, resume, and dashboard already use per-window provider
- Mark Task 6 complete — acceptance criteria verified (964 tests pass, provider coverage 80%+)
- Complete documentation for per-window provider support
- Robust terminal status detection for non-Claude providers

### Documentation

- Update documentation and changelog for multi-provider v1.0.0
- Reposition as standalone project, keep attribution to original
- Remove FORK.md — standalone project, attribution in README
- Rename to Command & Control Bot across all docs and metadata

### Fixed

- Address code review findings for per-window provider support
- Address code review findings for per-window provider support
- Address code review findings for per-window provider support
- Address code review findings for per-window provider support
- Address code review findings for per-window provider support
- Avoid persisting empty provider for unrecognized tmux commands
- Correct Codex/Gemini transcript parsing and resume syntax

## [0.4.0] - 2026-02-20

### Added

- Add AgentProvider protocol, event types, and contract tests (TASK-034)
- Add provider registry, capability policy, and config integration (TASK-035)
- Add ClaudeProvider wrapping existing modules behind AgentProvider protocol (TASK-036)
- Expand AgentProvider protocol and ClaudeProvider with history, bash, and command discovery
- Add Codex and Gemini CLI provider MVPs
- Capability-aware UX for recovery, resume, doctor, and status
- Add command history recall buttons to idle status messages

### Changed

- Make UUID_RE public and move expandable quote sentinels to providers.base
- Route handler calls through provider abstraction
- Consolidate provider code and deduplicate tests
- Extract JsonlProvider base class and deduplicate utilities

### Documentation

- Add TASK-036/037 progress and future provider task specs
- Add provider configuration and architecture docs
- Mark EPIC-008 multi-agent provider architecture done

### Fixed

- Harden provider layer with type guards, ClassVar annotations, and test coverage
- Handle stale message replies gracefully after restart

## [0.3.7] - 2026-02-19

### Fixed

- Parse status line with Claude Code 4.6 two-separator layout

## [0.3.6] - 2026-02-19

### Added

- Short status labels in Telegram (…reading, …thinking, …testing)

## [0.3.5] - 2026-02-19

### Added

- Show typing indicator while Claude Code is active

## [0.3.4] - 2026-02-19

### Fixed

- Detect /model selection UI in terminal for interactive control

## [0.3.2] - 2026-02-18

### Fixed

- Safe_edit crash when editing a Message (upgrade command)

## [0.3.1] - 2026-02-18

### Added

- Add /upgrade command for self-updating via uv

### Fixed

- Settings UI pattern fails on narrow terminals, add /model to menu
- Enforce 1 topic = 1 window to prevent double message delivery
- Kill orphan process on timeout, parse version from upgrade output
- Show active emoji during session startup instead of idle
- Retry uv pip compile in release to handle PyPI index lag

## [0.2.18] - 2026-02-18

### Fixed

- Hook install deduplication, insertion point, and command portability

## [0.2.17] - 2026-02-18

### Changed

- Migrate CLI from argparse to Click

## [0.2.16] - 2026-02-18

### Fixed

- Reduce log noise with colored output, demoted levels, and silenced spam

## [0.2.15] - 2026-02-18

### Documentation

- Move CLI reference and config to guides, recommend uv for install

### Fixed

- /unbind ghost status, interactive UI robustness, status line parsing

## [0.2.14] - 2026-02-18

### Fixed

- Simplify Homebrew formula generator and use uv run in CI
- Prune stale session_map.json entries for dead tmux windows

## [0.2.13] - 2026-02-18

### Added

- Add CLI argument parsing with flag-to-env precedence
- Add `ccbot status` and `ccbot doctor` CLI subcommands
- Add `ccbot hook --status` and `--uninstall` subcommands
- Topic close grace period + unbound window TTL

### Documentation

- Update CLAUDE.md with CLI flags and config precedence
- Document new CLI subcommands in CLAUDE.md

### Fixed

- Use hatch-vcs generated version instead of hardcoded string
- Guard against double-click in directory confirm callback

## [0.2.11] - 2026-02-17

### Fixed

- Back off topic auto-creation after flood control
- Preserve display names when session map is stale

## [0.2.10] - 2026-02-17

### Added

- Enhance directory browser workflow
- Add file handler support
- Add session favorites & notification controls

### Fixed

- Improve status and screenshot callbacks

## [0.2.9] - 2026-02-15

### Fixed

- Rename topic immediately on tmux window rename

## [0.2.8] - 2026-02-15

### Fixed

- Prevent dual-instance conflict and interactive UI message flood

## [0.2.7] - 2026-02-13

### Added

- Detect Claude exit, sync window renames, auto-close stale topics

## [0.2.6] - 2026-02-13

### Fixed

- Unify logging and add proactive dead window recovery

## [0.2.5] - 2026-02-13

### Fixed

- Improve screenshot reliability and debounce topic emoji updates

## [0.2.4] - 2026-02-12

### Fixed

- Use transcript_path for direct JSONL reading + auto-topic for unbound windows

## [0.2.3] - 2026-02-12

### Fixed

- Use <br> for newlines in Mermaid diagram node labels
- Handle libtmux ObjectDoesNotExist and clean up startup noise

## [0.2.2] - 2026-02-12

### Documentation

- Rewrite README for clarity and add configuration reference
- Expand guides with session recovery and service setup
- Add downloads and typed badges, fix license badge
- Replace ASCII architecture diagram with Mermaid flowchart

### Fixed

- Address code review findings
- Address code review findings
- Address code review findings
- Address code review findings

## [0.2.1] - 2026-02-12

### Fixed

- Scope id-token permission to publish job only
- Correct homebrew bump action name
- Exclude auto-generated \_version.py from ruff checks
- Restore pypi-publish action ref to release/v1

## [0.2.0] - 2026-02-12

### Added

- Configurable config directory via CCBOT_DIR env var
- Friendly config error message and non-source install docs
- Local .env takes priority over config dir .env
- Add window picker for unbound topics + auto-rename duplicate windows
- Support ! command mode in send_keys
- Capture and display ! bash command output in topic
- Add /kill command and auto-create topics for new tmux windows
- Rename /start to /new with backward-compatible alias
- Add multi-instance config variables (TASK-001)
- Add group filter to all handlers (TASK-002)
- Add CC command discovery and menu registration (TASK-004)
- Add /sessions dashboard command (TASK-006)
- Demote /esc, /screenshot, /kill to inline buttons (TASK-007)
- Clean up documentation structure (TASK-015)
- Dead window detection and recovery UI (TASK-009)
- Python 3.14 tooling upgrade and callback handler refactor (TASK-021, TASK-023)
- Extract text handler into dedicated module (TASK-024)
- Fix TASK-024 spec status to match implementation
- Fix cold-start auto-topic creation with CCBOT_GROUP_ID (TASK-032)
- Add e2e integration tests for new-window sync flow (TASK-033)
- Harden exceptions and logging (TASK-026)
- Enable quality gate lint rules C901, PLR, N (TASK-027)
- Resolve CC command names in forward_command_handler (TASK-008)
- Implement Fresh/Continue/Resume recovery flows (TASK-010)
- Implement /resume command to browse and resume past sessions (TASK-011)
- UI modernization - topic emoji, enhanced dashboard, status buttons (TASK-012/013/014)
- Fork independence - attribution, CI/CD, README, LICENSE, packaging (TASK-016/017/018/019/020/028/029/030/031)
- Verify implementation and fix spec status mismatches (TASK-025/032/033)
- Wrap-up docs and archive plan
- Add Homebrew tap support and install instructions

### Changed

- Re-key internal routing from window_name to window_id
- Move pane parsing functions into terminal_parser.py
- Optimize hot-path I/O in session lookup and project scanning
- Remove dead code (UnreadInfo, clear_user_state, show_user_messages, kill command)
- Consolidate duplicated session_map parsing and interactive key dispatch
- Remove dead empty-history early return in send_history
- Normalize naming and centralize user-data keys (TASK-025)

### Documentation

- Simplify .env setup instructions
- Restructure CLAUDE.md following official best practices
- Update READMEs to reflect window_id-keyed routing
- Add ccbot redesign plan for multi-instance, commands, resume, and UI
- Add specctl CLI reference to CLAUDE.md
- Add multi-instance setup to README and CLAUDE.md (TASK-003)
- Add multi-instance variables to .env.example
- Remove redundant docs and unused workflows (TASK-015)

### Fixed

- Support multiple supergroups per user via composite group_chat_ids key
- Remove extraneous f-string prefixes in main.py
- Replace time.time() with monotonic() and deprecated get_event_loop()
- Narrow exception handling and clean up orphaned \_pending_tools
- Address code review findings
- Address code review findings
- Address code review findings
