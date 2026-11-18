import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from ccgram.monitor_state import MonitorState, TrackedSession

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/smoke_agent_relay.py"


def load_smoke():
    spec = importlib.util.spec_from_file_location("smoke_agent_relay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_import_does_not_create_native_sessions(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected process")
        ),
    )
    assert load_smoke().USER_ID == 999001


def test_reply_requires_the_real_persisted_checkpoint_and_resolved_source(tmp_path):
    smoke = load_smoke()
    path = tmp_path / "session.jsonl"
    row = {
        "type": "message",
        "id": "reply",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "nonce"}],
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    state = MonitorState(state_file=tmp_path / "state.json")
    tracked = TrackedSession(session_id="session", file_path=str(path))
    state.update_session(tracked)
    monitor = SimpleNamespace(state=state)
    mapping = {"session_id": "session", "provider_name": "pi", "transcript_path": ""}

    assert smoke.acknowledged_reply(mapping, "nonce", monitor) is None
    tracked.last_byte_offset = path.stat().st_size - 1
    assert smoke.acknowledged_reply(mapping, "nonce", monitor) is None
    tracked.last_byte_offset = path.stat().st_size
    receipt = smoke.acknowledged_reply(mapping, "nonce", monitor)
    assert receipt == {
        "reply_end": path.stat().st_size,
        "watermark": path.stat().st_size,
        "internal_request_records": 0,
    }


async def test_workspace_cleanup_ignores_unrelated_global_sessions(monkeypatch):
    smoke = load_smoke()
    operations = []

    class Mux:
        async def list_windows_for_reconciliation(self):
            return [SimpleNamespace(matches=lambda _target: False)]

    def control(*args):
        operations.append(args)
        if args == ("tree",):
            return {
                "tree": {
                    "workspaces": [
                        {"id": "owned-workspace", "sessions": []},
                        {
                            "id": "foreign-workspace",
                            "sessions": [{"id": "user-session"}],
                        },
                    ]
                }
            }
        return {}

    monkeypatch.setattr(smoke, "control", control)
    await smoke.close_owned(Mux(), "closed-target", "owned", "owned-workspace")
    assert ("workspace", "delete", "--target", "owned-workspace") in operations


async def test_workspace_with_foreign_session_is_not_deleted(monkeypatch):
    smoke = load_smoke()
    operations = []

    class Mux:
        async def list_windows_for_reconciliation(self):
            return [SimpleNamespace(matches=lambda _target: False)]

    def control(*args):
        operations.append(args)
        return {
            "tree": {
                "workspaces": [
                    {"id": "owned-workspace", "sessions": [{"id": "user-session"}]}
                ]
            }
        }

    monkeypatch.setattr(smoke, "control", control)
    await smoke.close_owned(Mux(), "closed-target", "owned", "owned-workspace")
    assert not any(args[:2] == ("workspace", "delete") for args in operations)


async def test_unknown_workspace_sessions_cannot_trigger_cleanup(monkeypatch):
    import pytest

    smoke = load_smoke()
    operations = []

    class Mux:
        async def list_windows_for_reconciliation(self):
            return []

    def control(*args):
        operations.append(args)
        return {"tree": {"workspaces": [{"id": "owned-workspace"}]}}

    monkeypatch.setattr(smoke, "control", control)
    with pytest.raises(RuntimeError, match="Cannot confirm smoke workspace sessions"):
        await smoke.close_owned(Mux(), "closed-target", "owned", "owned-workspace")
    assert not any(args[:2] == ("workspace", "delete") for args in operations)


def test_native_control_addresses_the_parent_window(monkeypatch):
    smoke = load_smoke()
    commands = []

    def run(args, **_kwargs):
        commands.append(args)
        return SimpleNamespace(
            returncode=0, stdout='{"ok": true, "result": {}}', stderr=""
        )

    monkeypatch.setenv("AGTERM_WINDOW_ID", "parent-window")
    monkeypatch.setattr("subprocess.run", run)
    smoke.control("tree")
    assert commands == [["agtermctl", "tree", "--window", "parent-window", "--json"]]


async def test_failing_tree_call_reports_runtime_error_without_deleting(monkeypatch):
    import pytest

    smoke = load_smoke()
    commands = []

    class Mux:
        async def list_windows_for_reconciliation(self):
            return []

    def run(args, **_kwargs):
        commands.append(args)
        return SimpleNamespace(returncode=1, stdout="", stderr="window closed")

    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(RuntimeError, match="agterm tree failed: window closed"):
        await smoke.close_owned(Mux(), "closed-target", "owned", "workspace")
    assert len(commands) == 1
    assert commands[0][:2] == ["agtermctl", "tree"]


def test_missing_provider_is_rejected_before_creating_workspace_or_state(
    monkeypatch, tmp_path, capsys
):
    import pytest

    smoke = load_smoke()
    operations = []
    monkeypatch.setenv("AGTERM_ENABLED", "1")
    monkeypatch.setattr(
        "sys.argv", [str(SCRIPT), "--provider", "claude", "--cwd", str(tmp_path)]
    )
    monkeypatch.setattr("shutil.which", lambda _name: None)

    def unexpected(*args, **_kwargs):
        operations.append(args)
        raise AssertionError("Preflight must not create resources")

    monkeypatch.setattr(smoke, "control", unexpected)
    monkeypatch.setattr("tempfile.mkdtemp", unexpected)
    with pytest.raises(SystemExit) as exc:
        smoke.main()
    assert exc.value.code == 2
    assert "provider CLI is not installed" in capsys.readouterr().err
    assert operations == []
