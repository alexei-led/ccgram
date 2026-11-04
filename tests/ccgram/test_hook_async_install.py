import json
from pathlib import Path

import pytest

from ccgram.hook import (
    _current_hook_command,
    _HOOK_EVENT_TYPES,
    _install_hook,
    _install_json_hooks,
)


@pytest.mark.parametrize(
    "command",
    ["ccgram hook", "/old/python -m ccgram.main hook", _current_hook_command()],
)
@pytest.mark.parametrize("async_setting", [{}, {"async": False}])
def test_install_upgrades_sync_hooks_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, async_setting: dict
) -> None:
    settings_file = tmp_path / "settings.json"
    other_hook = {"type": "command", "command": "other-hook", "timeout": 10}
    settings = {
        "unrelated": True,
        "hooks": {
            event: [
                {
                    "matcher": ".*",
                    "hooks": [
                        dict(other_hook),
                        {
                            "type": "command",
                            "command": command,
                            "timeout": 7,
                            **async_setting,
                        },
                    ],
                }
            ]
            for event in _HOOK_EVENT_TYPES
        },
    }
    settings_file.write_text(json.dumps(settings))
    monkeypatch.setattr("ccgram.hook._claude_settings_file", lambda: settings_file)

    assert _install_hook() == 0
    upgraded = json.loads(settings_file.read_text())
    assert upgraded["unrelated"] is True
    for event in _HOOK_EVENT_TYPES:
        groups = upgraded["hooks"][event]
        assert len(groups) == 1
        assert groups[0]["matcher"] == ".*"
        assert groups[0]["hooks"] == [
            other_hook,
            {
                "type": "command",
                "command": _current_hook_command(),
                "timeout": 7,
                "async": True,
            },
        ]

    first_install = settings_file.read_text()
    assert _install_hook() == 0
    assert settings_file.read_text() == first_install


@pytest.mark.parametrize(
    ("provider", "flat", "events", "timeout", "asynchronous"),
    [
        ("codex", False, ("SessionStart", "Stop"), 5, True),
        ("pi", False, ("SessionStart", "Stop", "SessionEnd", "Notification"), 5, True),
        ("pi", True, ("SessionStart", "Stop", "SessionEnd", "Notification"), 5, True),
        (
            "gemini",
            False,
            ("SessionStart", "AfterAgent", "SessionEnd", "Notification"),
            5000,
            False,
        ),
    ],
)
@pytest.mark.parametrize("existing", [False, True])
def test_provider_hooks_use_supported_async_mode(
    tmp_path: Path,
    provider: str,
    flat: bool,
    events: tuple[str, ...],
    timeout: int,
    asynchronous: bool,
    existing: bool,
) -> None:
    path = tmp_path / "hooks.json"
    hook_map = (
        {
            event: [
                {
                    "matcher": ".*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"ccgram hook --provider {provider}",
                            "timeout": timeout,
                            "async": False,
                        }
                    ],
                }
            ]
            for event in events
        }
        if existing
        else {}
    )
    settings = (
        {"customOption": {"keep": True}, **hook_map} if flat else {"hooks": hook_map}
    )
    path.write_text(json.dumps(settings))

    assert _install_json_hooks(path, provider, events, timeout) == 0
    first_install = path.read_text()
    assert _install_json_hooks(path, provider, events, timeout) == 0
    assert path.read_text() == first_install
    settings = json.loads(first_install)
    installed_events = settings if flat else settings["hooks"]
    if flat:
        assert "hooks" not in settings
        assert settings["customOption"] == {"keep": True}
    for event in events:
        groups = installed_events[event]
        assert len(groups) == 1
        hooks = groups[0]["hooks"]
        assert len(hooks) == 1
        assert hooks[0]["command"] == _current_hook_command(provider)
        assert hooks[0]["timeout"] == timeout
        assert hooks[0].get("async", False) is asynchronous
        if provider == "gemini":
            assert "--background" in hooks[0]["command"]
        if existing:
            assert groups[0]["matcher"] == ".*"
