"""Operator commands for explicit Telegram-topic selection."""

import os

import pytest
from click.testing import CliRunner

from ccgram.cli import apply_args_to_env, cli


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
    backend.list_windows_for_reconciliation.return_value = None

    with pytest.raises(RuntimeError, match="unconfirmed"):
        await _topic_action(backend, "mark", "@12")


@pytest.fixture
def marked_backend():
    from unittest.mock import AsyncMock

    from ccgram.multiplexer.base import Multiplexer, WindowRef

    state = {"marked": False}
    backend = AsyncMock(spec=Multiplexer)

    def listing():
        return [
            WindowRef("ABC-123", "manager", "/project", topic_marked=state["marked"])
        ]

    async def set_mark(window_id, marked):
        assert window_id == "ABC-123"
        state["marked"] = marked
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
    backend.set_topic_mark.assert_awaited_once_with("ABC-123", expected)
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
            {"id": "primary", "name": "manager", "marked": None, "markable": True},
            {"id": "split", "name": "worker", "marked": False, "markable": False},
        ]
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
