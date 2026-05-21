"""Drift guard: tui_picker_commands must be a subset of builtin_commands."""

from __future__ import annotations

import pytest

from ccgram.providers.claude import ClaudeProvider
from ccgram.providers.codex import CodexProvider
from ccgram.providers.gemini import GeminiProvider
from ccgram.providers.pi import PiProvider


_PROVIDERS = [ClaudeProvider(), CodexProvider(), GeminiProvider(), PiProvider()]


@pytest.mark.parametrize("provider", _PROVIDERS, ids=lambda p: p.capabilities.name)
def test_picker_commands_subset_of_builtin_commands(provider) -> None:
    caps = provider.capabilities
    builtin = {c.lstrip("/") for c in caps.builtin_commands}
    picker = {c.lstrip("/") for c in caps.tui_picker_commands}
    missing = picker - builtin
    assert not missing, f"{caps.name} picker commands not in builtin set: {missing}"
