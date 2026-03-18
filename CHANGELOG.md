# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-03-16

### Changed

- **BREAKING**: Renamed project from `ccbot` to `ccgram` (CCGram)
- **BREAKING**: CLI command renamed from `ccbot` to `ccgram`
- **BREAKING**: Config directory changed from `~/.ccbot/` to `~/.ccgram/`
- **BREAKING**: Environment variables renamed from `CCBOT_*` to `CCGRAM_*`
- Hook command changed from `ccbot hook` to `ccgram hook`
- PyPI package name changed from `ccbot` to `ccgram`
- Default tmux session name changed from `ccbot` to `ccgram`

### Migration

- Old `CCBOT_*` environment variables still work as fallback with deprecation warnings
- `ccgram hook --install` detects and replaces legacy `ccbot hook` entries
- `ccgram hook --uninstall` removes both old and new hook entries
- Session map keys with `ccbot:` prefix are auto-migrated on load
- If `~/.ccgram/` doesn't exist but `~/.ccbot/` does, a migration hint is logged

## [1.0.0] - 2026-02-22

### Added

- Multi-provider support: Claude Code, OpenAI Codex CLI, and Google Gemini CLI as agent backends
- Per-topic provider selection via directory browser (Claude default, Codex, Gemini)
- Auto-detection of provider from externally created tmux windows
- Provider-aware recovery UI (Fresh/Continue/Resume adapt to each provider's capabilities)
- Gemini CLI terminal status detection via pane title and interactive UI patterns
- Codex and Gemini transcript parsing with provider-specific formats
- Provider capability matrix gating UX features per-window

### Fixed

- Codex resume syntax corrected to `resume <id>` subcommand (was `exec resume`)
- Gemini resume accepts index numbers and "latest" (not just UUIDs)
- Both Codex and Gemini now correctly support Continue (resume last session)

## [0.2.11] - 2026-02-17

### Fixed

- Preserved window display names when the SessionStart hook map is stale, preventing topic/session labels from regressing during state resolution

## [0.2.10] - 2026-02-17

### Added

- Directory favorites sidebar plus starred MRU controls for faster session bootstrapping
- File handler uploads that forward captions to Claude Code alongside the document payload
- Notification toggle to pause/resume Telegram alerts per topic

### Changed

- Directory browser now shows status keyboard tweaks for clarity when picking working directories
- Status keyboard refreshed to better expose screenshot shortcuts and live indicators

### Fixed

- Session polling stability improvements that cover status, screenshot, and message filtering edge cases

## [0.2.0] - 2026-02-12

Major rewrite as an independent fork of [six-ddc/ccbot](https://github.com/six-ddc/ccbot).

### Added

- Topic-based sessions: 1 topic = 1 tmux window = 1 Claude session
- Interactive UI for AskUserQuestion, ExitPlanMode, and Permission prompts
- Sessions dashboard with per-session status and kill buttons
- Message history with paginated browsing (newest first)
- Auto-discovery of Claude Code skills and custom commands in Telegram menu
- Hook-based session tracking (SessionStart hook writes session_map.json)
- Per-user message queue with FIFO ordering and message merging
- Rate limiting (1.1s minimum interval per user)
- Multi-instance support via CCBOT_GROUP_ID and CCBOT_INSTANCE_NAME
- Auto-topic creation for manually created tmux windows (including cold-start)
- Fresh/Continue/Resume recovery flows for dead sessions
- /resume command to browse and resume past sessions
- Directory browser for new topic session creation
- MarkdownV2 output with automatic plain text fallback
- Terminal screenshot rendering (ANSI color support)
- Status line polling with spinner and working text
- Expandable quote formatting for thinking content
- Persistent state (thread bindings, read offsets survive restarts)
- Topic emoji status updates reflecting session state
- Configurable config directory via CCBOT_DIR env var

### Changed

- Internal routing keyed by tmux window ID instead of window name
- Python 3.14 required (up from 3.12)
- Replaced broad exception handlers with specific types
- Normalized variable naming (full names instead of short aliases)
- Enabled C901, PLR, N ruff quality gate rules

### Removed

- Non-topic mode (active_sessions, /list, General topic routing)
- Message truncation at parse layer (splitting only at send layer)

## [0.1.0] - 2026-02-07

Initial release by [six-ddc](https://github.com/six-ddc).

[Unreleased]: https://github.com/alexei-led/ccgram/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/alexei-led/ccgram/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/alexei-led/ccbot/compare/v0.2.11...v1.0.0
[0.2.11]: https://github.com/alexei-led/ccbot/compare/v0.2.10...v0.2.11
[0.2.10]: https://github.com/alexei-led/ccbot/compare/v0.2.0...v0.2.10
[0.2.0]: https://github.com/alexei-led/ccbot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/alexei-led/ccbot/releases/tag/v0.1.0
