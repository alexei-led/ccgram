# Provider Protocol — Test Specification

## Unit Tests

### ProviderCapabilities

- **test_capabilities_default_values**: Create ProviderCapabilities with name only. Expected: `has_hooks=False`, `supports_resume=False`, `supports_status_snapshot=False`, etc.
- **test_capabilities_claude**: Create Claude capabilities. Expected: `has_hooks=True`, `supports_resume=True`, `supports_continue=True`, `supports_status_snapshot=False`.
- **test_capabilities_codex**: Create Codex capabilities. Expected: `has_hooks=False`, `supports_resume=True`, `supports_status_snapshot=True`.
- **test_capabilities_shell**: Create Shell capabilities. Expected: `has_hooks=False`, `supports_resume=False`, `supports_status_snapshot=False`.

### Provider Resolution

- **test_get_provider_for_window_from_state**: Window has `provider_name="codex"`. Expected: returns CodexProvider instance.
- **test_get_provider_for_window_default**: Window has no provider_name. Expected: returns provider from config default.
- **test_get_provider_for_window_unknown**: Window has `provider_name="nonexistent"`. Expected: falls back to config default with warning.

### Provider Detection

- **test_detect_from_command_claude**: Pane command is "claude". Expected: returns "claude".
- **test_detect_from_command_codex**: Pane command is "codex". Expected: returns "codex".
- **test_detect_from_command_gemini**: Pane command is "gemini". Expected: returns "gemini".
- **test_detect_from_command_shell**: Pane command is "bash" or "zsh". Expected: returns "shell".
- **test_detect_from_command_node_wrapper**: Pane command is "node". Expected: returns None (needs runtime detection).
- **test_detect_from_pane_with_tty_fallback**: Pane command is "node", TTY-based ps inspection finds "codex". Expected: returns "codex".
- **test_detect_from_runtime**: Pane title probe finds provider identifier. Expected: returns correct provider name.
- **test_detect_unknown_command**: Pane command is "vim". Expected: returns None.

### Provider Registry

- **test_registry_get_provider**: Register "claude" factory. Expected: `get("claude")` returns ClaudeProvider instance.
- **test_registry_singleton_cache**: Call `get("claude")` twice. Expected: same instance returned (singleton).
- **test_registry_unknown_provider**: Call `get("nonexistent")`. Expected: returns None or raises.

### Optional Protocol Methods

- **test_build_status_snapshot_default**: Call `build_status_snapshot()` on base/default. Expected: returns None.
- **test_has_output_since_default**: Call `has_output_since()` on base/default. Expected: returns False.

## Integration Contract Tests

- **test_all_providers_implement_protocol**: For each registered provider (claude, codex, gemini, shell), verify it satisfies `AgentProvider` protocol — has `capabilities`, `parse_terminal_status`, `get_command_metadata`, `build_status_snapshot`.
- **test_capabilities_immutable**: Access `provider.capabilities`, attempt mutation. Expected: ProviderCapabilities is frozen or read-only.
- **test_parse_terminal_status_returns_correct_type**: Call `parse_terminal_status([])` on each provider. Expected: returns `StatusUpdate | None`.
- **test_get_command_metadata_returns_dict**: Call `get_command_metadata()` on each provider. Expected: returns `dict` with string keys.
- **test_launch_command_resolution**: Set `CCGRAM_CODEX_COMMAND="custom-codex"`. Expected: `resolve_launch_command("codex")` returns "custom-codex".
- **test_launch_command_default**: No env override. Expected: `resolve_launch_command("claude")` returns provider's default command.

## Boundary Tests

- **test_detect_from_pane_empty_command**: Pane command is "". Expected: returns None, no crash.
- **test_detect_from_pane_none_tty**: TTY is None (window just created). Expected: falls back gracefully, no crash.
- **test_provider_resolution_with_empty_state**: Window exists in tmux but has no WindowState entry. Expected: default provider returned.
- **test_concurrent_provider_resolution**: Multiple threads call `get_provider_for_window()` simultaneously. Expected: consistent results, no cache corruption.
- **test_registry_register_override**: Register "claude" factory, then register again with different factory. Expected: last registration wins.

## Behavior Tests

- **test_provider_capability_gates_ux**: Claude provider has `has_hooks=True`. Expected: hook validation runs for Claude windows. Codex has `has_hooks=False`. Expected: hook validation skipped for Codex windows.
- **test_status_snapshot_polymorphic**: Codex provider returns status string from `build_status_snapshot()`. Claude provider returns None. Expected: consumers call the method uniformly and handle None correctly — no provider name checks.
- **test_provider_detection_full_chain**: Tmux window running Codex via node wrapper. Expected: `detect_provider_from_command()` returns None → `detect_provider_from_pane()` with TTY fallback finds "codex" → provider set correctly.
