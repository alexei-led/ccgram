from __future__ import annotations

import pytest

from ccgram.providers import _ensure_registered
from ccgram.providers.base import AgentProvider
from ccgram.providers.registry import registry


def _all_providers() -> list[AgentProvider]:
    """Enumerate every registered provider — picks up new providers without code edits."""
    _ensure_registered()
    return [registry.get(name) for name in registry.provider_names()]


_PROVIDERS = _all_providers()


@pytest.mark.parametrize("provider", _PROVIDERS, ids=lambda p: p.capabilities.name)
def test_picker_commands_are_bare_names(provider) -> None:
    caps = provider.capabilities
    leading_slash = {c for c in caps.tui_picker_commands if c.startswith("/")}
    assert not leading_slash, (
        f"{caps.name} tui_picker_commands must be bare names (no leading '/'); "
        f"found: {leading_slash}. forward.py looks up the bare cc_name."
    )


@pytest.mark.parametrize("provider", _PROVIDERS, ids=lambda p: p.capabilities.name)
def test_picker_commands_are_lowercase(provider) -> None:
    caps = provider.capabilities
    non_lower = {c for c in caps.tui_picker_commands if c != c.lower()}
    assert not non_lower, (
        f"{caps.name} tui_picker_commands must be lowercase; "
        f"forward.py normalises cc_name with .lower() before lookup. Found: {non_lower}"
    )


@pytest.mark.parametrize("provider", _PROVIDERS, ids=lambda p: p.capabilities.name)
def test_picker_commands_subset_of_builtin_commands(provider) -> None:
    caps = provider.capabilities
    builtin = {c.lstrip("/") for c in caps.builtin_commands}
    missing = caps.tui_picker_commands - builtin
    assert not missing, f"{caps.name} picker commands not in builtin set: {missing}"


def test_shell_has_no_picker_commands() -> None:
    """Shell forwards raw commands; no TUI picker can fire there."""
    shell = registry.get("shell")
    assert shell.capabilities.tui_picker_commands == frozenset()
