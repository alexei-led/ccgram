from __future__ import annotations

import base64

import pytest

from ccgram.terminal_backends.base import TerminalOperationRejectedError
from ccgram.terminal_backends.cmux_native_client import FakeCmuxNativeClient


def _tree_payload():
    return {
        "windows": [
            {
                "id": "win-id",
                "ref": "window:1",
                "workspaces": [
                    {
                        "id": "ws-id",
                        "ref": "workspace:1",
                        "title": "dev",
                        "panes": [
                            {
                                "id": "pane-id",
                                "ref": "pane:1",
                                "surfaces": [
                                    {
                                        "id": "surface-a",
                                        "ref": "surface:1",
                                        "type": "terminal",
                                        "title": "claude",
                                        "focused": True,
                                        "selected_in_pane": True,
                                    },
                                    {
                                        "id": "surface-b",
                                        "ref": "surface:2",
                                        "type": "browser",
                                        "title": "localhost",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


class TestListTerminalSessions:
    async def test_parses_terminal_surfaces_from_system_tree(self) -> None:
        client = FakeCmuxNativeClient({"system.tree": _tree_payload()})

        sessions = await client.list_terminal_sessions()

        assert len(sessions) == 1
        session = sessions[0]
        assert session.terminal_id == "surface-a"
        assert session.surface_id == "surface-a"
        assert session.title == "claude"
        assert session.workspace_id == "ws-id"
        assert session.workspace_title == "dev"
        assert session.pane_id == "pane-id"
        assert session.window_id == "win-id"
        assert session.window_ref == "window:1"
        assert session.workspace_ref == "workspace:1"
        assert session.pane_ref == "pane:1"
        assert session.surface_ref == "surface:1"
        assert session.focused is True
        assert session.selected_in_pane is True
        assert client.requests == [("system.tree", {"all": True})]


class TestCapture:
    async def test_decodes_base64_read_text(self) -> None:
        encoded = base64.b64encode("screen text".encode()).decode()
        client = FakeCmuxNativeClient({"surface.read_text": {"base64": encoded}})

        text = await client.capture_screen("surface-a")

        assert text == "screen text"
        assert client.requests == [
            ("surface.read_text", {"surface_id": "surface-a", "lines": 200})
        ]

    async def test_accepts_plain_text_read_text(self) -> None:
        client = FakeCmuxNativeClient({"surface.read_text": {"text": "plain"}})

        assert await client.capture_screen("surface-a") == "plain"


class TestSend:
    async def test_send_text_uses_surface_send_text(self) -> None:
        client = FakeCmuxNativeClient({"surface.send_text": {}})

        assert await client.send_text("surface-a", "hello") is True
        assert client.requests == [
            ("surface.send_text", {"surface_id": "surface-a", "text": "hello"})
        ]

    async def test_send_key_uses_surface_send_key(self) -> None:
        client = FakeCmuxNativeClient({"surface.send_key": {}})

        assert await client.send_key("surface-a", "enter") is True
        assert client.requests == [
            ("surface.send_key", {"surface_id": "surface-a", "key": "enter"})
        ]


class TestErrors:
    async def test_unknown_fake_method_fails_test(self) -> None:
        client = FakeCmuxNativeClient({})

        with pytest.raises(AssertionError):
            await client.send_text("surface-a", "hello")

    async def test_close_uses_surface_close(self) -> None:
        client = FakeCmuxNativeClient({"surface.close": {}})

        assert await client.close_terminal_session("surface-a") is True
        assert client.requests == [("surface.close", {"surface_id": "surface-a"})]


def test_error_type_import_is_alive() -> None:
    assert TerminalOperationRejectedError.code == "rejected"
