"""Reported shell identity is not evidence that agterm is at an idle prompt."""

import json
from unittest.mock import AsyncMock

import pytest

from ccgram.multiplexer.agterm import AgtermManager
from ccgram.providers import detect_provider_from_command, detect_provider_from_pane
from ccgram.providers.shell_infra import setup_shell_prompt
from ccgram.handlers.shell.shell_prompt_orchestrator import ensure_setup

SESSION = "12B5B1EA-47F9-4EAF-BFBE-B8D1552D44C2"


@pytest.mark.parametrize(
    "pane_text", ["user@host> ", "user@host> echo unfinished", "read> "]
)
async def test_markerless_shell_does_not_cancel_local_input(monkeypatch, pane_text):
    from ccgram.handlers.shell import shell_commands

    mux = manager({"foregroundShell": "fish"})
    monkeypatch.setattr(shell_commands, "tmux_manager", mux)
    monkeypatch.setattr(mux, "capture_pane", AsyncMock(return_value=pane_text))
    send_keys = AsyncMock()
    monkeypatch.setattr(mux, "send_keys", send_keys)
    monkeypatch.setattr(shell_commands.asyncio, "sleep", AsyncMock())

    await shell_commands._cancel_stuck_input(SESSION)

    send_keys.assert_not_awaited()


def manager(node: dict) -> AgtermManager:
    async def runner(args, stdin=None):
        if list(args[:2]) == ["window", "list"]:
            result = {"windows": [{"id": "window", "open": True}]}
        elif args[0] == "tree":
            result = {
                "tree": {
                    "workspaces": [
                        {
                            "name": "code",
                            "sessions": [
                                {
                                    "id": SESSION,
                                    "name": "~/Workspace",
                                    "cwd": "/home/user/Workspace",
                                    **node,
                                },
                            ],
                        }
                    ]
                }
            }
        else:
            raise AssertionError(f"Unexpected terminal operation: {args}")
        return 0, json.dumps({"ok": True, "result": result}), ""

    return AgtermManager(runner=runner, own_session_id="", workspaces=None)


@pytest.mark.parametrize("shell", ["fish", "bash", "zsh", "sh"])
async def test_explicit_foreground_shell_is_detected_without_synthetic_argv(
    shell: str,
) -> None:
    mux = manager({"foregroundShell": shell})
    window = await mux.find_window_by_id(SESSION)
    assert window is not None
    assert window.pane_current_command == shell
    assert (
        await detect_provider_from_pane(window.pane_current_command, window_id=SESSION)
        == "shell"
    )
    assert not window.topic_eligible
    assert await mux.foreground(SESSION) is None


@pytest.mark.parametrize("shell", [None, "", "top", "node", 42, [], {}])
async def test_missing_or_invalid_shell_hint_remains_unknown(shell: object) -> None:
    mux = manager({"foregroundShell": shell})
    window = await mux.find_window_by_id(SESSION)
    assert window is not None
    assert window.pane_current_command == ""
    assert detect_provider_from_command(window.pane_current_command) == ""
    assert await mux.foreground(SESSION) is None


async def test_real_agent_argv_takes_precedence_over_shell_hint() -> None:
    mux = manager({"foregroundShell": "fish", "foreground": ["pi"]})
    window = await mux.find_window_by_id(SESSION)
    assert window is not None and window.pane_current_command == "pi"
    assert window.topic_eligible


@pytest.mark.parametrize("peer_shell", [None, "bash"])
async def test_split_uses_its_own_shell_hint_not_the_primary_hint(
    peer_shell: str | None,
) -> None:
    mux = manager(
        {
            "foregroundShell": "fish",
            "hasSplit": True,
            "splitForegroundShell": peer_shell,
        }
    )
    windows = await mux.list_windows_for_reconciliation()
    assert windows is not None and len(windows) == 2
    assert windows[0].pane_current_command == "fish"
    assert windows[1].pane_current_command == (peer_shell or "")
    assert await mux.foreground(windows[1].window_id) is None


async def test_unsupported_prompt_setup_does_not_offer_an_unusable_keyboard(
    monkeypatch,
):
    mux = manager({"foregroundShell": "fish"})
    monkeypatch.setattr("ccgram.multiplexer.multiplexer", mux)
    monkeypatch.setattr("ccgram.multiplexer.get_active_multiplexer", lambda: mux)
    client = AsyncMock()
    await ensure_setup(
        SESSION, "provider_switch", client=client, chat_id=-871, thread_id=871
    )
    client.send_message.assert_not_awaited()


async def test_shell_hint_never_injects_prompt_setup_into_a_possible_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mux = manager({"foregroundShell": "fish"})
    monkeypatch.setattr("ccgram.multiplexer.multiplexer", mux)
    monkeypatch.setattr("ccgram.multiplexer.get_active_multiplexer", lambda: mux)
    send = AsyncMock()
    capture = AsyncMock(return_value="read> ")

    await setup_shell_prompt(SESSION, send_keys_fn=send, capture_fn=capture)

    send.assert_not_awaited()
    capture.assert_not_awaited()
