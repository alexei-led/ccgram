"""Failure and readback paths at the libtmux command boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import libtmux
import pytest

from ccgram.multiplexer.tmux import TmuxManager


def response(*lines: str, rc: int = 0, error: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        stdout=list(lines), returncode=rc, stderr=[error] if error else []
    )


@pytest.fixture
def window() -> MagicMock:
    window = MagicMock(spec=libtmux.Window)
    window.window_id = "@7"
    window.window_name = "project"
    window.active_pane = None
    window.cmd.return_value = response()
    return window


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ([], False),
        (["@ccgram_topic 1"], True),
        (["@ccgram_topic 0"], False),
        (["@ccgram_topic yes"], None),
        (["@ccgram_topic_extra 1"], False),
    ],
)
def test_only_local_confirmed_option_selects_window(
    window: MagicMock, lines: list[str], expected: bool | None
) -> None:
    window.cmd.return_value = response(*lines)
    result = TmuxManager._window_ref_for_reconciliation(window)
    assert result.topic_marked is expected
    assert result.topic_eligible
    window.cmd.assert_called_once_with("show-options", "-w")


@pytest.mark.parametrize("failure", ["exception", "status", "stderr"])
def test_option_failure_keeps_window_live_but_marker_unknown(
    window: MagicMock, failure: str
) -> None:
    if failure == "exception":
        window.cmd.side_effect = OSError("unavailable")
    else:
        window.cmd.return_value = response(
            "@ccgram_topic 1",
            rc=1 if failure == "status" else 0,
            error="unavailable" if failure == "stderr" else "",
        )
    result = TmuxManager._window_ref_for_reconciliation(window)
    assert result.window_id == "@7"
    assert result.topic_marked is None


@pytest.mark.parametrize("marked", [True, False])
async def test_set_mark_is_idempotent_with_readback(
    window: MagicMock, marked: bool
) -> None:
    mgr = TmuxManager("test")
    session = MagicMock()
    session.windows.get.return_value = window
    value = "1" if marked else "0"
    initial = "0" if marked else "1"
    window.cmd.side_effect = [
        response(f"@ccgram_topic {initial}"),
        response(),
        response(f"@ccgram_topic {value}"),
        response(f"@ccgram_topic {value}"),
    ]
    with patch.object(mgr, "get_session", return_value=session):
        assert await mgr.set_topic_mark("@7", marked)
        assert await mgr.set_topic_mark("@7", marked)
    writes = [
        call.args for call in window.cmd.call_args_list if call.args[0] == "set-option"
    ]
    assert writes == [("set-option", "-w", "@ccgram_topic", value)]


@pytest.mark.parametrize("marked", [True, False])
@pytest.mark.parametrize(
    "failure", ["read", "write", "readback", "unchanged", "missing"]
)
async def test_unverified_option_change_fails(
    window: MagicMock, failure: str, marked: bool
) -> None:
    mgr = TmuxManager("test")
    session = MagicMock()
    session.windows.get.return_value = None if failure == "missing" else window
    initial = response(f"@ccgram_topic {'0' if marked else '1'}")
    failed = response(rc=1, error="target gone")
    if failure == "read":
        window.cmd.return_value = failed
    elif failure == "write":
        window.cmd.side_effect = [initial, failed]
    elif failure == "readback":
        window.cmd.side_effect = [initial, response(), failed]
    elif failure == "unchanged":
        window.cmd.side_effect = [initial, response(), initial]
    with patch.object(mgr, "get_session", return_value=session):
        assert not await mgr.set_topic_mark("@7", marked)


async def test_missing_session_cannot_set_mark() -> None:
    mgr = TmuxManager("test")
    with patch.object(mgr, "get_session", return_value=None):
        assert not await mgr.set_topic_mark("@7", True)
