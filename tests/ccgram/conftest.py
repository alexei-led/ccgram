"""Shared fixtures for ccgram unit tests.

Provides factories for building JSONL entries, content blocks,
and sample pane text for terminal parser tests.
"""

import os
import time

import pytest


@pytest.fixture(autouse=True)
async def _shutdown_queue_workers():
    """Kill background queue workers created as side-effects of handler calls.

    Tests that call real handler code (forward_command_handler, handle_text_message,
    update_status_message) may trigger enqueue_status_update → get_or_create_queue,
    which spawns an infinite asyncio worker task.  Without cleanup the event loop
    waits for these pending tasks, causing 30 s hangs on Linux CI.
    """
    yield
    from ccgram.handlers.messaging_pipeline.message_queue import (
        _queue_workers,
        shutdown_workers,
    )

    if _queue_workers:
        await shutdown_workers()


@pytest.fixture(autouse=True)
def _clean_reaction_dedupe_cache():
    """Reset the module-level reaction dedupe cache between tests.

    ``reactions._last_reaction`` suppresses a repeat reaction to the same
    (chat_id, message_id). That is correct in production and poison across
    tests: one that reacts and then takes a failure path leaves the entry
    behind, and the next test to react to the same ids has its call silently
    skipped. Six test files reach this code, and xdist ordering decides which
    pairing happens, so it surfaces only as a rare parallel-run failure.
    """
    from ccgram.handlers import reactions

    reactions._last_reaction.clear()
    yield
    reactions._last_reaction.clear()


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """Remove CCGRAM_*_COMMAND env vars so tests use provider defaults."""
    for key in list(os.environ):
        if key.startswith("CCGRAM_") and key.endswith("_COMMAND"):
            monkeypatch.delenv(key)


@pytest.fixture(autouse=True)
def _default_replace_prompt_mode():
    """Default to replace mode so existing tests using ccgram:N❯ markers pass."""
    from ccgram.config import config

    original = config.prompt_mode
    config.prompt_mode = "replace"
    yield
    config.prompt_mode = original


@pytest.fixture
def _wrap_mode():
    """Switch to wrap prompt mode for the test."""
    from ccgram.config import config

    original = config.prompt_mode
    config.prompt_mode = "wrap"
    yield
    config.prompt_mode = original


@pytest.fixture(autouse=True)
def _clean_proxy_env(monkeypatch):
    """Remove proxy env vars that cause PTB's Application builder to fail.

    PTB auto-detects socks proxies from all_proxy/ftp_proxy/etc. and tries to
    import httpx[socks], which may not be installed in the test environment.
    """
    for key in list(os.environ):
        lower = key.lower()
        if "proxy" in lower and key not in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)


# ── JSONL entry factories ────────────────────────────────────────────────


@pytest.fixture
def make_jsonl_entry():
    """Factory: build a raw JSONL dict (pre-parse_line)."""

    def _make(
        msg_type: str = "assistant",
        content: list | str = "",
        *,
        timestamp: str | None = None,
        session_id: str = "test-session-id",
        cwd: str = "/tmp/test",
    ) -> dict:
        entry: dict = {
            "type": msg_type,
            "message": {"content": content},
            "sessionId": session_id,
            "cwd": cwd,
        }
        if timestamp:
            entry["timestamp"] = timestamp
        else:
            entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return entry

    return _make


@pytest.fixture
def make_text_block():
    """Factory: build a text content block."""

    def _make(text: str) -> dict:
        return {"type": "text", "text": text}

    return _make


@pytest.fixture
def make_tool_use_block():
    """Factory: build a tool_use content block."""

    def _make(
        tool_id: str = "tool_1",
        name: str = "Read",
        input_data: dict | None = None,
    ) -> dict:
        return {
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": input_data or {},
        }

    return _make


@pytest.fixture
def make_tool_result_block():
    """Factory: build a tool_result content block."""

    def _make(
        tool_use_id: str = "tool_1",
        content: str | list = "result text",
        *,
        is_error: bool = False,
    ) -> dict:
        block: dict = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            block["is_error"] = True
        return block

    return _make


@pytest.fixture
def make_thinking_block():
    """Factory: build a thinking content block."""

    def _make(thinking: str = "deep thoughts") -> dict:
        return {"type": "thinking", "thinking": thinking}

    return _make


# ── Sample pane text for terminal parser ─────────────────────────────────


@pytest.fixture
def sample_pane_exit_plan():
    return (
        "  Would you like to proceed?\n"
        "  ─────────────────────────────────\n"
        "  Yes     No\n"
        "  ─────────────────────────────────\n"
        "  ctrl-g to edit in vim\n"
    )


@pytest.fixture
def sample_pane_ask_user_multi_tab():
    return "  ←  ☐ Option A\n     ☐ Option B\n     ☐ Option C\n  Enter to select\n"


@pytest.fixture
def sample_pane_ask_user_single_tab():
    return "  ☐ Option A\n  ☐ Option B\n  Enter to select\n"


@pytest.fixture
def sample_pane_permission():
    return "  Do you want to proceed?\n  Some permission details\n  Esc to cancel\n"


@pytest.fixture
def sample_pane_status_line():
    sep = "─" * 30
    return f"Some output text here\nMore output\n✻ Reading file src/main.py\n{sep}\n"


@pytest.fixture
def sample_pane_no_ui():
    return "$ echo hello\nhello\n$\n"
