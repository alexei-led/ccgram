from __future__ import annotations

import pytest

from ccgram.multiplexer.herdr_commands import command_request


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("status", "--json"), ("ping", {})),
        (("agent", "list"), ("agent.list", {})),
        (("workspace", "list"), ("workspace.list", {})),
        (("tab", "list"), ("tab.list", {})),
        (("pane", "list"), ("pane.list", {})),
        (("pane", "get", "w2:p1"), ("pane.get", {"pane_id": "w2:p1"})),
        (("pane", "close", "w2:p1"), ("pane.close", {"pane_id": "w2:p1"})),
        (("tab", "close", "w2:t1"), ("tab.close", {"tab_id": "w2:t1"})),
        (
            ("workspace", "close", "w2"),
            ("workspace.close", {"workspace_id": "w2"}),
        ),
        (
            ("tab", "rename", "w2:t1", "Renamed"),
            ("tab.rename", {"tab_id": "w2:t1", "label": "Renamed"}),
        ),
        (
            ("pane", "layout", "--pane", "w2:p1"),
            ("pane.layout", {"pane_id": "w2:p1"}),
        ),
        (
            ("pane", "process-info", "--pane", "w2:p1"),
            ("pane.process_info", {"pane_id": "w2:p1"}),
        ),
        (
            ("pane", "send-text", "w2:p1", "שלום\n世界"),
            ("pane.send_text", {"pane_id": "w2:p1", "text": "שלום\n世界"}),
        ),
        (
            ("pane", "send-keys", "w2:p1", "Ctrl-C", "Enter"),
            ("pane.send_keys", {"pane_id": "w2:p1", "keys": ["Ctrl-C", "Enter"]}),
        ),
        (
            ("pane", "run", "w2:p1", "printf 'שלום\\n世界'"),
            (
                "pane.send_input",
                {
                    "pane_id": "w2:p1",
                    "text": "printf 'שלום\\n世界'",
                    "keys": ["enter"],
                },
            ),
        ),
        (
            ("pane", "read", "w2:p1", "--source", "visible", "--format", "text"),
            (
                "pane.read",
                {
                    "pane_id": "w2:p1",
                    "source": "visible",
                    "format": "text",
                    "strip_ansi": True,
                },
            ),
        ),
        (
            (
                "pane",
                "read",
                "w2:p1",
                "--source",
                "recent",
                "--lines",
                "42",
                "--format",
                "text",
            ),
            (
                "pane.read",
                {
                    "pane_id": "w2:p1",
                    "source": "recent",
                    "lines": 42,
                    "format": "text",
                    "strip_ansi": True,
                },
            ),
        ),
        (
            (
                "pane",
                "read",
                "w2:p1",
                "--source",
                "recent-unwrapped",
                "--format",
                "ansi",
            ),
            (
                "pane.read",
                {
                    "pane_id": "w2:p1",
                    "source": "recent_unwrapped",
                    "format": "ansi",
                    "strip_ansi": False,
                },
            ),
        ),
        (
            (
                "pane",
                "report-metadata",
                "w2:p1",
                "--source",
                "ccgram",
                "--title",
                "ccgram:claude",
            ),
            (
                "pane.report_metadata",
                {"pane_id": "w2:p1", "source": "ccgram", "title": "ccgram:claude"},
            ),
        ),
        (
            ("workspace", "create", "--cwd", "/repo", "--no-focus"),
            ("workspace.create", {"cwd": "/repo", "focus": False}),
        ),
        (
            ("workspace", "create", "--cwd", "/repo", "--focus"),
            ("workspace.create", {"cwd": "/repo", "focus": True}),
        ),
        (
            ("tab", "create", "--cwd", "/repo", "--no-focus", "--workspace", "w2"),
            (
                "tab.create",
                {"cwd": "/repo", "focus": False, "workspace_id": "w2"},
            ),
        ),
        (
            (
                "tab",
                "create",
                "--cwd",
                "/repo",
                "--focus",
                "--workspace",
                "w2",
                "--label",
                "Tab α",
            ),
            (
                "tab.create",
                {
                    "cwd": "/repo",
                    "focus": True,
                    "workspace_id": "w2",
                    "label": "Tab α",
                },
            ),
        ),
        (
            (
                "worktree",
                "create",
                "--cwd",
                "/repo",
                "--branch",
                "feature/α",
                "--path",
                "/tmp/α",
                "--no-focus",
                "--json",
            ),
            (
                "worktree.create",
                {
                    "cwd": "/repo",
                    "branch": "feature/α",
                    "path": "/tmp/α",
                    "focus": False,
                },
            ),
        ),
        (
            (
                "worktree",
                "create",
                "--cwd",
                "/repo",
                "--branch",
                "feature",
                "--path",
                "/tmp/worktree",
                "--focus",
                "--workspace",
                "w2",
                "--json",
                "--label",
                "Tree",
            ),
            (
                "worktree.create",
                {
                    "cwd": "/repo",
                    "branch": "feature",
                    "path": "/tmp/worktree",
                    "focus": True,
                    "workspace_id": "w2",
                    "label": "Tree",
                },
            ),
        ),
    ],
)
def test_command_request_maps_manager_command(args, expected) -> None:
    assert command_request(args) == expected


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("status",),
        ("status", "--json", "extra"),
        ("unknown", "list"),
        ("pane", "unknown"),
        ("pane", "get"),
        ("pane", "get", "--help"),
        ("pane", "get", "p1", "--unknown"),
        ("pane", "send-text", "p1"),
        ("pane", "send-text", "p1", "--help", "extra"),
        ("pane", "send-keys", "p1"),
        ("pane", "layout", "p1", "--pane", "p1"),
        ("pane", "read", "p1", "--source", "visible"),
        ("pane", "read", "p1", "--source"),
        ("pane", "read", "p1", "--source", "visible", "--format", "json"),
        (
            "pane",
            "read",
            "p1",
            "--source",
            "visible",
            "--unknown",
            "x",
            "--format",
            "text",
        ),
        ("pane", "report-metadata", "p1", "--source", "ccgram", "--title"),
        ("pane", "report-metadata", "p1", "--source", "wrong", "--title", "x"),
        ("workspace", "create", "--cwd"),
        ("workspace", "create", "--cwd", "/repo", "--no-focus", "--unknown"),
        ("workspace", "create", "--cwd", "/repo", "--focus", "--no-focus"),
        ("tab", "create", "--cwd", "/repo", "--no-focus", "--workspace"),
        ("tab", "create", "--cwd", "/repo", "--no-focus"),
        (
            "tab",
            "create",
            "--cwd",
            "/repo",
            "--no-focus",
            "--workspace",
            "w2",
            "--unknown",
        ),
        (
            "worktree",
            "create",
            "--cwd",
            "/repo",
            "--branch",
            "main",
            "--path",
            "/tmp/tree",
            "--no-focus",
        ),
        (
            "worktree",
            "create",
            "--cwd",
            "/repo",
            "--branch",
            "main",
            "--path",
            "/tmp/tree",
            "--no-focus",
            "--json",
            "--unknown",
        ),
        (
            "worktree",
            "create",
            "--cwd",
            "/repo",
            "--branch",
            "main",
            "--path",
            "/tmp/tree",
            "--no-focus",
            "--json",
            "--label",
        ),
    ],
)
def test_command_request_rejects_malformed_or_unsupported(args) -> None:
    with pytest.raises(ValueError):
        command_request(args)


@pytest.mark.parametrize(
    "args",
    [
        ("pane", "read", "p1", "--source", "visible", "--lines", ""),
        ("pane", "read", "p1", "--source", "visible", "--lines", "-1"),
        ("pane", "read", "p1", "--source", "visible", "--lines", "one"),
        ("pane", "read", "p1", "--source", "visible", "--lines", "4294967296"),
    ],
)
def test_command_request_rejects_invalid_line_count(args) -> None:
    with pytest.raises(ValueError):
        command_request(args + ("--format", "text"))


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ("pane", "send-text", "p1", "--help"),
            ("pane.send_text", {"pane_id": "p1", "text": "--help"}),
        ),
        (
            ("pane", "run", "p1", "--help"),
            (
                "pane.send_input",
                {"pane_id": "p1", "text": "--help", "keys": ["enter"]},
            ),
        ),
        (
            ("pane", "send-keys", "p1", "--help"),
            ("pane.send_keys", {"pane_id": "p1", "keys": ["--help"]}),
        ),
        (
            (
                "pane",
                "report-metadata",
                "p1",
                "--source",
                "ccgram",
                "--title",
                "--help",
            ),
            (
                "pane.report_metadata",
                {"pane_id": "p1", "source": "ccgram", "title": "--help"},
            ),
        ),
    ],
)
def test_command_request_preserves_literal_text(args, expected) -> None:
    assert command_request(args) == expected
