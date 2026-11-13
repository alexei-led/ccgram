"""Native marker parsing and socket-identity failure boundaries."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import libtmux
import pytest

from ccgram.multiplexer.tmux import TmuxManager

NONCE = "16a6dc07b0855345ac469253567dd3f7"


def response(*lines: str, rc: int = 0, error: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        stdout=list(lines), returncode=rc, stderr=[error] if error else []
    )


@pytest.fixture
def window() -> MagicMock:
    window = MagicMock(spec=libtmux.Window)
    window.window_id = "@7"
    window.cmd.return_value = response()
    return window


@pytest.mark.parametrize(
    ("lines", "selected", "owner"),
    [
        ([], False, None),
        ([f"@ccgram_topic v1:1:{NONCE}"], True, f"tmux:{NONCE}"),
        ([f"@ccgram_topic v1:0:{NONCE}"], False, f"tmux:{NONCE}"),
        (["@ccgram_topic 1"], None, None),
        (["@ccgram_topic 0"], None, None),
        (["@ccgram_topic v1:1:invalid"], None, None),
        ([f"@ccgram_topic v2:1:{NONCE}"], None, None),
        (["@ccgram_topic_extra 1"], False, None),
    ],
)
def test_only_local_versioned_token_selects_window(
    window: MagicMock, lines: list[str], selected: bool | None, owner: str | None
) -> None:
    window.cmd.return_value = response(*lines)
    actual = TmuxManager._read_topic_selection(window, "socket-incarnation")
    assert actual[0] is selected
    if owner is not None or selected is None:
        assert actual[1] == owner
    else:
        assert actual[1] is not None and actual[1].startswith("tmux:")
    window.cmd.assert_called_once_with("show-options", "-w")


@pytest.mark.parametrize("failure", ["exception", "status", "stderr"])
def test_option_failure_is_unknown(window: MagicMock, failure: str) -> None:
    if failure == "exception":
        window.cmd.side_effect = OSError("unavailable")
    else:
        window.cmd.return_value = response(
            f"@ccgram_topic v1:1:{NONCE}",
            rc=1 if failure == "status" else 0,
            error="unavailable" if failure == "stderr" else "",
        )
    assert TmuxManager._read_topic_selection(window, "socket-incarnation") == (
        None,
        None,
    )


@pytest.mark.parametrize(
    "result",
    [
        response(rc=1),
        response(error="unavailable"),
        response(),
        response("/nonexistent/socket"),
        response("/tmp"),
        response("/one", "/two"),
        response("\x00invalid"),
    ],
)
def test_socket_incarnation_requires_actual_confirmed_socket(
    result: SimpleNamespace,
) -> None:
    server = MagicMock(spec=libtmux.Server)
    server.cmd.return_value = result
    assert TmuxManager._socket_incarnation(server) is None
