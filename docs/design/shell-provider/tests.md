# Shell Provider — Test Specification

## Unit Tests

### PromptMatch Dataclass

- **test_prompt_match_frozen**: Create PromptMatch, attempt field mutation. Expected: `FrozenInstanceError` raised.
- **test_prompt_match_fields**: Create PromptMatch with all fields. Expected: `sequence_number`, `trailing_text`, `exit_code`, `raw_line` all accessible by name.
- **test_prompt_match_equality**: Two PromptMatch with same values. Expected: equal (`==` returns True).

### match_prompt — Wrap Mode

- **test_wrap_mode_bare_prompt**: Line "⌘0⌘". Expected: `PromptMatch(sequence_number=0, trailing_text="", exit_code=0, raw_line="⌘0⌘")`.
- **test_wrap_mode_with_trailing_text**: Line "⌘5⌘ ls -la". Expected: `PromptMatch(sequence_number=5, trailing_text="ls -la", ...)`.
- **test_wrap_mode_embedded_in_prompt**: Line "user@host ~/project ⌘3⌘". Expected: match found (search, not match), correct sequence_number=3.
- **test_wrap_mode_no_marker**: Line "regular output without marker". Expected: returns None.
- **test_wrap_mode_multi_digit_sequence**: Line "⌘123⌘". Expected: `sequence_number=123`.

### match_prompt — Replace Mode

- **test_replace_mode_bare_prompt**: Line "ccgram:0❯". Expected: `PromptMatch(sequence_number=0, trailing_text="", ...)`.
- **test_replace_mode_with_command**: Line "ccgram:2❯ git status". Expected: `PromptMatch(sequence_number=2, trailing_text="git status", ...)`.
- **test_replace_mode_custom_prefix**: Prefix set to "mybot", line "mybot:1❯". Expected: match with sequence_number=1.
- **test_replace_mode_no_match**: Line "other:0❯". Expected: returns None (wrong prefix).
- **test_replace_mode_uses_match_not_search**: Line "some text ccgram:0❯". Expected: returns None (replace mode uses `.match()`, must be at line start).

### has_prompt_marker

- **test_has_marker_present**: Pane capture ends with "⌘0⌘" in last 5 lines. Expected: returns True.
- **test_has_marker_absent**: Pane capture has no marker anywhere. Expected: returns False.
- **test_has_marker_checks_last_5_lines**: Marker present at line -6 (beyond the 5-line check window). Expected: returns False.
- **test_has_marker_empty_capture**: Pane capture is empty. Expected: returns False.
- **test_has_marker_none_capture**: `capture_pane` returns None. Expected: returns False.

### setup_shell_prompt

- **test_setup_injects_bash_marker**: Shell is bash. Expected: sends PS1/PROMPT_COMMAND override via `send_keys`.
- **test_setup_injects_fish_marker**: Shell is fish. Expected: sends `fish_prompt` function definition.
- **test_setup_injects_zsh_marker**: Shell is zsh. Expected: sends precmd function or PROMPT override.
- **test_setup_wrap_mode**: Prompt mode is "wrap". Expected: injected marker uses `⌘N⌘` format.
- **test_setup_replace_mode**: Prompt mode is "replace". Expected: injected prompt uses `{prefix}:N❯` format.

### Provider Capabilities

- **test_shell_capabilities**: Expected: `name="shell"`, `has_hooks=False`, `has_transcript=False`, `supports_resume=False`, `supports_continue=False`.

## Integration Contract Tests

- **test_match_prompt_returns_prompt_match_or_none**: Call `match_prompt()` with various inputs. Expected: return type is always `PromptMatch | None` — never `re.Match`.
- **test_prompt_match_consumed_by_shell_capture**: Create `PromptMatch`, access `.sequence_number` and `.trailing_text`. Expected: same values that `shell_capture.py` would extract (previously via `group(1)` and `group(2)`).
- **test_has_prompt_marker_calls_capture_pane**: Call `has_prompt_marker("@0")`. Expected: `tmux_manager.capture_pane("@0")` called.
- **test_setup_calls_send_keys**: Call `setup_shell_prompt("@0")`. Expected: `tmux_manager.send_keys("@0", ...)` called with prompt override text.
- **test_implements_agent_provider**: Verify ShellProvider satisfies `AgentProvider` protocol — has `capabilities`, `parse_terminal_status`, `get_command_metadata`.

## Boundary Tests

- **test_match_prompt_empty_string**: Call with "". Expected: returns None.
- **test_match_prompt_only_marker_chars**: Call with "⌘⌘" (no digits). Expected: returns None.
- **test_match_prompt_negative_number**: Call with "⌘-1⌘". Expected: returns None (regex requires `\d+`).
- **test_match_prompt_very_long_trailing**: Trailing text is 10,000 characters. Expected: match succeeds, trailing_text contains full text.
- **test_match_prompt_unicode_in_trailing**: Trailing text contains emoji and CJK characters. Expected: match succeeds, characters preserved.
- **test_has_prompt_marker_with_ansi_codes**: Pane capture contains ANSI escape sequences around marker. Expected: marker still detected (or documented limitation).
- **test_setup_already_set**: Prompt already has marker (re-setup). Expected: no double injection, idempotent.

## Behavior Tests

- **test_command_output_isolation**: Shell executes command, output appears between two prompt markers with sequence N and N+1. Using `match_prompt()` on both markers: Expected: sequence_number incremented, trailing_text on first marker contains command echo, second marker is bare (empty trailing_text).
- **test_exit_code_extraction**: Command exits with code 1. Expected: `PromptMatch.exit_code` is 1 on the output-terminating marker.
- **test_prompt_recovery_after_exec_bash**: User runs `exec bash`, marker lost. Next command send triggers lazy recovery. Expected: `setup_shell_prompt()` called, marker restored, subsequent output isolation works.
- **test_skip_respected**: User chose Skip on prompt setup offer. Expected: lazy recovery does not override, `setup_shell_prompt()` not called on next send.
- **test_provider_switch_fresh_offer**: Switch from shell → claude → shell. Expected: fresh setup offer shown (Skip from previous session not carried over).
