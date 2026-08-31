"""Optional smoke tests for a running agterm control socket."""

from __future__ import annotations

import os

import pytest

from ccgram.multiplexer.agterm import AgtermManager


pytestmark = pytest.mark.agterm


def _live_manager() -> AgtermManager:
    return AgtermManager()


@pytest.mark.skipif(
    os.getenv("CCGRAM_AGTERM_LIVE") != "1",
    reason="set CCGRAM_AGTERM_LIVE=1 to use a live agterm socket",
)
async def test_live_socket_lists_workspaces() -> None:
    workspaces = await _live_manager().list_workspaces()

    assert all(workspace.workspace_id for workspace in workspaces)
    assert all(workspace.cwd for workspace in workspaces)


@pytest.mark.skipif(
    not os.getenv("CCGRAM_AGTERM_LIVE_SESSION_ID"),
    reason="set CCGRAM_AGTERM_LIVE_SESSION_ID to read a disposable live session",
)
async def test_live_socket_reads_session_text() -> None:
    result = await _live_manager().capture_scrollback(
        os.environ["CCGRAM_AGTERM_LIVE_SESSION_ID"], lines=20
    )

    assert result is not None
    assert isinstance(result.text, str)
