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
    lines: list[str], selected: bool | None, owner: str | None
) -> None:
    actual = TmuxManager._selection_from_options(lines, "@7", "socket-incarnation")
    assert actual[0] is selected
    if owner is not None or selected is None:
        assert actual[1] == owner
    else:
        assert actual[1] is not None and actual[1].startswith("tmux:")


@pytest.mark.parametrize("failure", ["exception", "status", "stderr"])
def test_option_failure_is_unknown(failure: str) -> None:
    manager = TmuxManager(session_name="owned")
    server = MagicMock(spec=libtmux.Server)
    manager._server = server
    if failure == "exception":
        server.cmd.side_effect = OSError("unavailable")
    else:
        server.cmd.return_value = response(
            "@7",
            f"@ccgram_topic v1:1:{NONCE}",
            rc=1 if failure == "status" else 0,
            error="unavailable" if failure == "stderr" else "",
        )
    if failure == "exception":
        with pytest.raises(OSError):
            manager._read_topic_selections(["@7"], "scope")
    else:
        assert manager._read_topic_selections(["@7"], "scope") is None


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


@pytest.mark.parametrize("window_count", [0, 1, 64, 65, 256])
def test_inventory_batches_local_option_reads(monkeypatch, window_count):
    manager = TmuxManager(session_name="owned")
    server = MagicMock(spec=libtmux.Server)
    manager._server = server
    window_ids = [f"@{i}" for i in range(window_count)]

    def command(*args):
        if args[0] == "list-windows":
            return response(*window_ids)
        assert args[0] == "display-message"
        return response(
            *(args[i + 3] for i, part in enumerate(args) if part == "display-message")
        )

    server.cmd.side_effect = command
    native_windows = {wid: MagicMock(spec=libtmux.Window) for wid in window_ids}
    for wid, native_window in native_windows.items():
        native_window.window_id = wid
        native_window.cmd.return_value = response()
    monkeypatch.setattr(
        "ccgram.multiplexer.tmux.libtmux.Window",
        lambda **kw: native_windows[kw["window_id"]],
    )
    monkeypatch.setattr(manager, "_socket_incarnation", lambda _server: "same-scope")

    inventory = manager._topic_inventory()

    assert inventory is not None
    assert set(inventory[1]) == set(window_ids)
    assert all(marked is False and owner for marked, owner in inventory[1].values())
    assert (
        server.cmd.call_count + sum(w.cmd.call_count for w in native_windows.values())
        == 1 + (window_count + 63) // 64
    )


@pytest.mark.parametrize(
    "lines,targets,valid",
    [
        ([], [], True),
        (["@1", "@2"], ["@1", "@2"], True),
        (["@1"], ["@1", "@2"], False),
        (["@1", "@1", "@2"], ["@1", "@2"], False),
        (["@1", "@3"], ["@1", "@2"], False),
        (["unexpected", "@1"], ["@1"], False),
        (["@1", f"@ccgram_topic v1:1:{NONCE}", "@2"], ["@1", "@2"], True),
    ],
)
def test_batched_option_frames_require_exact_complete_window_set(lines, targets, valid):
    observed = TmuxManager._selections_from_frames(lines, targets, "scope")
    assert (observed is not None) is valid
    if observed is not None:
        assert set(observed) == set(targets)
        if any(line.startswith("@ccgram_topic") for line in lines):
            assert observed["@1"] == (True, f"tmux:{NONCE}")


def test_batched_inventory_detects_duplicate_owners_across_batches(monkeypatch):
    manager = TmuxManager(session_name="owned")
    server = MagicMock(spec=libtmux.Server)
    manager._server = server
    windows = [f"@{i}" for i in range(130)]

    def command(*args):
        if args[0] == "list-windows":
            return response(*windows)
        output = []
        for i, part in enumerate(args):
            if part == "display-message":
                window_id = args[i + 3]
                output.append(window_id)
                if window_id in {"@1", "@129"}:
                    output.append(f"@ccgram_topic v1:1:{NONCE}")
        return response(*output)

    server.cmd.side_effect = command
    monkeypatch.setattr(manager, "_socket_incarnation", lambda _server: "scope")
    result = manager._topic_inventory()
    assert result is not None
    assert result[1]["@1"] == (None, None)
    assert result[1]["@129"] == (None, None)
    assert result[1]["@2"][0] is False
    assert result[1]["@2"][1] is not None
