# Codex Provider — Test Specification

## Unit Tests

### Status Snapshot

- **test_build_status_snapshot_with_data**: Transcript contains session_meta + token_count entries. Expected: returns formatted string with token stats and session info.
- **test_build_status_snapshot_empty_transcript**: Transcript path exists but is empty. Expected: returns None (no data to build from).
- **test_build_status_snapshot_no_transcript**: Transcript path doesn't exist. Expected: returns None.
- **test_build_status_snapshot_partial_data**: Transcript has session_meta but no token_count. Expected: returns partial snapshot with available data.
- **test_has_output_since_true**: Transcript has assistant output after offset 1024. Expected: returns True.
- **test_has_output_since_false**: No new assistant entries after offset. Expected: returns False.
- **test_has_output_since_empty**: Transcript empty or doesn't exist. Expected: returns False.

### Interactive Prompt Formatting

- **test_format_edit_approval**: Terminal output contains Codex edit approval prompt with diff. Expected: formatted with readable diff markers for Telegram.
- **test_format_numbered_options**: Terminal shows yes/no/esc numbered list. Expected: formatted with inline numbers preserved.
- **test_format_token_stats**: Terminal shows token consumption line. Expected: formatted as readable stats.
- **test_format_no_interactive**: Terminal shows regular output (not interactive). Expected: returns input unchanged (pass-through).
- **test_format_preserves_indentation**: Diff content has significant whitespace. Expected: indentation preserved in formatted output.

### Terminal Status Parsing

- **test_parse_active_status**: Terminal lines contain Codex activity indicator. Expected: `StatusUpdate` with active=True.
- **test_parse_idle_status**: Terminal shows Codex prompt waiting for input. Expected: idle status.
- **test_parse_interactive_prompt**: Terminal shows edit approval. Expected: formatted interactive content returned.
- **test_parse_empty_lines**: No terminal content. Expected: returns None.

### Provider Capabilities

- **test_codex_capabilities**: Expected: `name="codex"`, `has_hooks=False`, `supports_resume=True`, `supports_continue=True`, `supports_status_snapshot=True`.

### Launch Arguments

- **test_resume_args**: Resume session with ID. Expected: launch args include `resume` and session identifier.
- **test_continue_args**: Continue last session. Expected: launch args include `-c` flag.
- **test_model_override**: Model specified. Expected: launch args include `--model <model>`.

## Integration Contract Tests

- **test_build_status_snapshot_protocol_contract**: Call `build_status_snapshot(window_id, transcript_path, offset)` on CodexProvider. Expected: returns `str | None` — matches AgentProvider protocol signature.
- **test_has_output_since_protocol_contract**: Call `has_output_since(transcript_path, offset)` on CodexProvider. Expected: returns `bool`.
- **test_parse_terminal_status_contract**: Call `parse_terminal_status(lines)` with `list[str]`. Expected: returns `StatusUpdate | None`.
- **test_get_command_metadata_contract**: Call `get_command_metadata()`. Expected: returns `dict` with string keys and descriptions.
- **test_implements_agent_provider**: Verify CodexProvider satisfies full `AgentProvider` protocol.
- **test_snapshot_consumed_by_command_orchestration**: Build snapshot, verify it's a plain string suitable for Telegram delivery via `safe_send`.

## Boundary Tests

- **test_corrupt_jsonl_entry**: Transcript contains malformed JSON line. Expected: line skipped, parsing continues, no crash.
- **test_truncated_transcript**: Transcript file truncated mid-line. Expected: partial line skipped, valid entries parsed.
- **test_very_large_transcript**: Transcript is 100MB. Expected: incremental read from offset, no full-file load.
- **test_offset_beyond_file_size**: Offset > file size (file was truncated/rotated). Expected: offset reset, parsed from beginning.
- **test_format_very_long_diff**: Interactive prompt contains 1000-line diff. Expected: formatted without truncation (truncation is send layer's job).
- **test_unicode_in_transcript**: JSONL entries contain emoji and non-ASCII. Expected: parsed correctly, no encoding errors.
- **test_concurrent_transcript_read**: Two callers read same transcript simultaneously. Expected: both get correct results, no file lock contention.

## Behavior Tests

- **test_status_command_full_flow**: User sends `/status` in Codex topic → Command Orchestration calls `provider.build_status_snapshot()` → CodexProvider reads JSONL transcript → returns formatted status with token counts, session info. Expected: user receives Codex-specific status without any `capabilities.name == "codex"` check in the calling code.
- **test_codex_prompt_surfaced**: Codex shows edit approval in terminal → polling calls `parse_terminal_status()` → CodexProvider formats the prompt → status update contains readable diff. Expected: user sees formatted approval prompt in Telegram.
- **test_provider_isolation**: Modify `codex_status.py` (now `providers/codex_status.py`). Expected: only `providers/codex.py` is affected — no changes needed in `bot.py`, `status_polling.py`, or any handler module.
- **test_other_provider_unaffected**: Call `build_status_snapshot()` on ClaudeProvider. Expected: returns None (Claude doesn't implement snapshots). No error, no Codex code path triggered.
