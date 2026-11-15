"""Operator commands for explicit Telegram-topic selection."""

import os
from types import SimpleNamespace
from unittest.mock import ANY

import pytest
from click.testing import CliRunner

from _topic_mark_helpers import make_witness

from ccgram.cli import apply_args_to_env, cli
from ccgram.state_writer_lock import StateWriterLease


@pytest.fixture(autouse=True)
def _isolated_topic_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CCGRAM_DIR", str(tmp_path))


@pytest.mark.parametrize("action", ["mark", "unmark", "status"])
def test_topic_command_help_needs_no_bot_configuration(monkeypatch, action):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ALLOWED_USERS", raising=False)

    result = CliRunner().invoke(cli, ["topic", action, "--help"])

    assert result.exit_code == 0
    assert "TARGET" in result.output


def test_topic_discovery_flag_is_available():
    result = CliRunner().invoke(cli, ["run", "--topic-discovery", "marked", "--help"])

    assert result.exit_code == 0
    assert "--topic-discovery" in result.output


def test_topic_discovery_flag_maps_to_environment(monkeypatch):
    monkeypatch.delenv("CCGRAM_TOPIC_DISCOVERY", raising=False)

    apply_args_to_env(topic_discovery="marked")

    assert os.environ.get("CCGRAM_TOPIC_DISCOVERY") == "marked"


@pytest.mark.parametrize("action", ["mark", "unmark"])
def test_topic_mutations_require_an_explicit_target(action):
    result = CliRunner().invoke(cli, ["topic", action])

    assert result.exit_code == 2
    assert "Missing argument 'TARGET'" in result.output


@pytest.mark.parametrize("target", ["missing", "active", "project"])
def test_target_selection_never_uses_names_or_active_focus(target):
    from ccgram.multiplexer.base import WindowRef
    from ccgram.topic_mark_cmd import _select_targets

    window = WindowRef("@12", "project", "/project")

    with pytest.raises(LookupError, match="Target not found"):
        _select_targets([window], target)


def test_target_selection_accepts_case_variant_exact_identity():
    from ccgram.multiplexer.base import WindowRef
    from ccgram.topic_mark_cmd import _select_targets

    window = WindowRef("ABC-123", "project", "/project")

    assert _select_targets([window], "abc-123") == [window]


def test_target_selection_rejects_ambiguous_alias():
    from ccgram.multiplexer.base import WindowRef
    from ccgram.topic_mark_cmd import _select_targets

    windows = [
        WindowRef("@12", "first", "/project", alias_window_ids=("old",)),
        WindowRef("@13", "second", "/project", alias_window_ids=("old",)),
    ]

    with pytest.raises(LookupError, match="ambiguous"):
        _select_targets(windows, "old")


@pytest.mark.asyncio
async def test_unconfirmed_backend_listing_cannot_change_selection():
    from unittest.mock import AsyncMock

    from ccgram.multiplexer.base import Multiplexer
    from ccgram.topic_mark_cmd import _topic_action

    backend = AsyncMock(spec=Multiplexer)
    backend.topic_mark_endpoint_scope.return_value = make_witness().endpoint_scope
    backend.list_windows_for_reconciliation.return_value = None

    with pytest.raises(RuntimeError, match="unconfirmed"):
        await _topic_action(backend, "mark", "@12")


@pytest.fixture
def marked_backend():
    from unittest.mock import AsyncMock

    from ccgram.multiplexer.base import Multiplexer, WindowRef

    state = {"marked": False}
    backend = AsyncMock(spec=Multiplexer)
    backend.capabilities = SimpleNamespace(name="tmux")
    backend.topic_mark_endpoint_scope = AsyncMock(
        return_value=make_witness().endpoint_scope
    )

    def listing():
        return [
            WindowRef(
                "ABC-123",
                "manager",
                "/project",
                topic_marked=state["marked"],
                topic_mark_owner="owner-A",
            )
        ]

    async def set_mark(window_id, marked, *, owner, write_guard):
        assert owner == "owner-A"
        assert window_id == "ABC-123"
        witness = make_witness()
        write_guard.arm(witness)
        state["marked"] = marked
        write_guard.terminal(witness.dispatch_id)
        return True

    backend.list_windows_for_reconciliation.side_effect = listing
    backend.find_window_by_id.side_effect = lambda target: next(
        (window for window in listing() if window.matches(target)), None
    )
    backend.set_topic_mark.side_effect = set_mark
    return backend, state


@pytest.mark.asyncio
@pytest.mark.parametrize(("action", "expected"), [("mark", True), ("unmark", False)])
async def test_marker_mutation_uses_verified_canonical_target(
    marked_backend, action, expected
):
    from ccgram.topic_mark_cmd import _topic_action

    backend, state = marked_backend
    state["marked"] = not expected

    result = await _topic_action(backend, action, "abc-123")

    assert result == {"target": "ABC-123", "marked": expected}
    assert state["marked"] is expected
    backend.set_topic_mark.assert_awaited_once_with(
        "ABC-123", expected, owner="owner-A", write_guard=ANY
    )
    backend.kill_window.assert_not_called()
    backend.send.assert_not_called()


@pytest.mark.asyncio
async def test_marker_write_failure_is_not_reported_as_success(marked_backend):
    from ccgram.topic_mark_cmd import _topic_action

    backend, _state = marked_backend
    backend.set_topic_mark.side_effect = None
    backend.set_topic_mark.return_value = False

    with pytest.raises(RuntimeError, match="could not be confirmed"):
        await _topic_action(backend, "mark", "ABC-123")


@pytest.mark.asyncio
async def test_unmarkable_split_cannot_change_its_primary_marker():
    from unittest.mock import AsyncMock

    from ccgram.multiplexer.base import Multiplexer, WindowRef
    from ccgram.topic_mark_cmd import _topic_action

    backend = AsyncMock(spec=Multiplexer)
    backend.topic_mark_endpoint_scope.return_value = make_witness().endpoint_scope
    backend.list_windows_for_reconciliation.return_value = [
        WindowRef("split-target", "split", "/project", topic_markable=False)
    ]

    with pytest.raises(ValueError, match="cannot be natively marked"):
        await _topic_action(backend, "mark", "split-target")

    backend.set_topic_mark.assert_not_called()


@pytest.mark.asyncio
async def test_status_preserves_unknown_and_explicit_only_native_states():
    from unittest.mock import AsyncMock

    from ccgram.multiplexer.base import Multiplexer, WindowRef
    from ccgram.topic_mark_cmd import _topic_action

    backend = AsyncMock(spec=Multiplexer)
    backend.list_windows_for_reconciliation.return_value = [
        WindowRef("primary", "manager", "/project", topic_marked=None),
        WindowRef("split", "worker", "/project", topic_markable=False),
    ]

    result = await _topic_action(backend, "status", None)

    assert result == {
        "targets": [
            {
                "id": "primary",
                "name": "manager",
                "marked": None,
                "markable": True,
                "owner": None,
            },
            {
                "id": "split",
                "name": "worker",
                "marked": False,
                "markable": False,
                "owner": None,
            },
        ],
        "pending_operations": [],
    }
    backend.set_topic_mark.assert_not_called()


def test_status_is_json_and_needs_no_bot_credentials(monkeypatch, marked_backend):
    import json

    from ccgram import topic_mark_cmd

    backend, _state = marked_backend
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ALLOWED_USERS", raising=False)
    monkeypatch.setenv("CCGRAM_MULTIPLEXER", "agterm")
    monkeypatch.setenv("CCGRAM_TOPIC_DISCOVERY", "marked")
    monkeypatch.setattr(topic_mark_cmd, "get_multiplexer", lambda name: backend)

    result = CliRunner().invoke(cli, ["topic", "status", "abc-123"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["backend"] == "agterm"
    assert payload["mode"] == "marked"
    assert payload["targets"][0]["id"] == "ABC-123"


def test_topic_status_keeps_native_diagnostics_off_json_stdout(tmp_path):
    import json
    import subprocess
    import sys
    from textwrap import dedent

    script = dedent("""
        import structlog
        from ccgram.cli import cli
        from ccgram.multiplexer.base import WindowRef
        from ccgram.multiplexer import registry

        class NativeBackend:
            async def list_windows_for_reconciliation(self):
                structlog.get_logger().warning("native_marker_read_unconfirmed")
                return [WindowRef("target", "session", "/project", topic_marked=None)]

        registry._FACTORIES["agterm"] = NativeBackend
        cli(["topic", "status"], prog_name="ccgram")
    """)
    env = dict(os.environ)
    env.update(
        CCGRAM_DIR=str(tmp_path),
        CCGRAM_MULTIPLEXER="agterm",
        CCGRAM_TOPIC_DISCOVERY="auto",
    )
    env.pop("TELEGRAM_BOT_TOKEN", None)
    env.pop("ALLOWED_USERS", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["targets"][0]["marked"] is None
    assert "native_marker_read_unconfirmed" in result.stderr


async def test_mutation_uses_captured_owner_without_target_relookup(marked_backend):
    from ccgram.topic_mark_cmd import _topic_action

    backend, state = marked_backend
    backend.find_window_by_id.side_effect = AssertionError("recycled target lookup")

    await _topic_action(backend, "mark", "ABC-123")

    assert state["marked"] is True
    backend.find_window_by_id.assert_not_awaited()
    backend.set_topic_mark.assert_awaited_once_with(
        "ABC-123", True, owner="owner-A", write_guard=ANY
    )


@pytest.mark.parametrize(
    ("saved_route", "saved_owner", "desired", "blocked"),
    [
        ("ABC-123", "owner-A", False, True),
        ("old-agent", "owner-A", False, True),
        ("abc-123", None, False, True),
        ("ABC-123", "different-owner", False, False),
        ("ABC-123", "owner-a", False, False),
        ("ABC-123", "owner-A", True, False),
    ],
)
async def test_pending_journal_blocks_only_conflicting_owner_intent(
    marked_backend, tmp_path, saved_route, saved_owner, desired, blocked
):
    import json

    from ccgram.topic_mark_cmd import _topic_action

    backend, _state = marked_backend
    state_file = tmp_path / "state.json"
    contents = json.dumps(
        {
            "other_state": {"preserve": True},
            "topic_mark_operations": [
                {
                    "window_id": saved_route,
                    "owner": saved_owner,
                    "desired": desired,
                    "attempt": None,
                    "detach_clear": False,
                }
            ],
        }
    )
    state_file.write_text(contents, encoding="utf-8")
    before = state_file.stat().st_mtime_ns

    if blocked:
        with pytest.raises(RuntimeError, match="(?i)pending"):
            await _topic_action(backend, "mark", "ABC-123")
        backend.set_topic_mark.assert_not_awaited()
    else:
        await _topic_action(backend, "mark", "ABC-123")
        backend.set_topic_mark.assert_awaited_once_with(
            "ABC-123", True, owner="owner-A", write_guard=ANY
        )

    if blocked:
        assert state_file.read_text(encoding="utf-8") == contents
        assert state_file.stat().st_mtime_ns == before
    else:
        saved = json.loads(state_file.read_text(encoding="utf-8"))
        assert saved["other_state"] == {"preserve": True}
        expected_pending = (
            []
            if saved_owner == "owner-A" and desired is True
            else json.loads(contents)["topic_mark_operations"]
        )
        assert saved.get("topic_mark_operations", []) == expected_pending


@pytest.mark.parametrize(
    "contents", [b"{", b"[]", b"\xff", b'{"topic_mark_operations": null}']
)
async def test_unconfirmed_journal_refuses_native_mutation(
    marked_backend, tmp_path, contents
):
    from ccgram.topic_mark_cmd import _topic_action

    backend, _state = marked_backend
    (tmp_path / "state.json").write_bytes(contents)

    with pytest.raises(RuntimeError, match="journal.*unconfirmed"):
        await _topic_action(backend, "mark", "ABC-123")

    backend.set_topic_mark.assert_not_awaited()


async def test_status_exposes_orphaned_journal_without_rewriting_state(
    marked_backend, tmp_path
):
    import json

    from ccgram.topic_mark_cmd import _topic_action

    backend, _state = marked_backend
    operation = {
        "window_id": "gone-agent",
        "owner": "old-owner",
        "desired": False,
        "attempt": None,
        "detach_clear": False,
    }
    (tmp_path / "state.json").write_text(
        json.dumps({"topic_mark_operations": [operation]}), encoding="utf-8"
    )

    all_status = await _topic_action(backend, "status", None)
    current_status = await _topic_action(backend, "status", "ABC-123")

    assert all_status["pending_operations"] == [operation]
    assert current_status["pending_operations"] == []
    targets = all_status["targets"]
    assert isinstance(targets, list)
    assert targets[0]["owner"] == "owner-A"
    backend.set_topic_mark.assert_not_awaited()


async def test_unreadable_journal_refuses_native_mutation(
    marked_backend, tmp_path, monkeypatch
):
    from pathlib import Path

    from ccgram.topic_mark_cmd import _topic_action

    backend, _state = marked_backend
    state_file = tmp_path / "state.json"
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if path == state_file:
            raise PermissionError("state unavailable")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

    with pytest.raises(RuntimeError, match="journal.*unconfirmed"):
        await _topic_action(backend, "unmark", "ABC-123")

    backend.set_topic_mark.assert_not_awaited()


async def test_live_writer_without_rpc_never_falls_back_to_native_mutation(
    marked_backend, tmp_path
):
    from ccgram.topic_mark_cmd import _topic_action

    backend, state = marked_backend
    path = tmp_path / "state.json"
    original = b'{"unrelated":{"keep":true}}\n'
    path.write_bytes(original)
    with StateWriterLease(path), pytest.raises((RuntimeError, OSError)):
        await _topic_action(backend, "mark", "ABC-123")

    backend.set_topic_mark.assert_not_awaited()
    backend.kill_window.assert_not_called()
    assert state["marked"] is False
    assert path.read_bytes() == original
