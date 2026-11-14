"""A lifecycle hook is not evidence that the launched agent owns its pane."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.topics.window_launch_service import launch_window
from ccgram.handlers.user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT

from .test_window_launch_service import (
    _isolated_window_state_owner as _isolated_window_state_owner,
)
from .test_window_launch_service import (
    _MODULE,
    _launch_env,
    _make_context,
    _make_query,
    _request,
)


async def test_hooked_launch_waits_for_matching_foreground_before_binding(tmp_path):
    order = []
    snapshots = iter(
        [
            ("unknown", (), "claude"),
            ("shell", ("-zsh",), "shell"),
            ("other", ("codex",), "codex"),
            ("ready", ("claude",), "claude"),
        ]
    )

    async def resolve(window_id):
        assert window_id == "@5"
        label, argv, provider = next(snapshots, ("ready", ("claude",), "claude"))
        order.append(label)
        return SimpleNamespace(
            foreground_argv=argv, provider=provider, available=True
        ), None

    async def send(*_args, **_kwargs):
        order.append("send")
        return True, "ok"

    user_data = {PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "first request"}
    with (
        _launch_env(supports_hook=True) as m,
        patch(f"{_MODULE}asyncio.sleep", new_callable=AsyncMock),
        patch(f"{_MODULE}send_telegram_to_window", side_effect=send) as forward,
    ):
        m.adapter.resolve = AsyncMock(side_effect=resolve)
        m.router.commit_topic_provisioning.side_effect = lambda *_a, **_k: (
            order.append("bind") or True
        )
        result = await launch_window(
            _make_query(),
            _make_context(user_data),
            _request(cwd=str(tmp_path), pending_text="first request"),
        )

    assert result.success
    assert order[:3] == ["unknown", "shell", "other"]
    assert set(order[3:-2]) == {"ready"}
    assert order[-2:] == ["bind", "send"]
    forward.assert_awaited_once()
    assert PENDING_THREAD_TEXT not in user_data


@pytest.mark.parametrize(
    ("argv", "provider", "available"),
    [
        ((), "claude", True),
        (("-zsh",), "shell", True),
        (("codex",), "codex", True),
        (("claude",), "claude", False),
    ],
)
async def test_registered_hook_without_matching_foreground_keeps_request(
    tmp_path, argv, provider, available
):
    user_data = {PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "first request"}
    with (
        _launch_env(supports_hook=True) as m,
        patch(f"{_MODULE}asyncio.sleep", new_callable=AsyncMock),
        patch(
            "ccgram.multiplexer.reconciliation.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            f"{_MODULE}send_telegram_to_window",
            new_callable=AsyncMock,
            return_value=(True, "ok"),
        ) as forward,
    ):
        m.adapter.resolve = AsyncMock(
            return_value=(
                SimpleNamespace(
                    foreground_argv=argv, provider=provider, available=available
                ),
                None,
            )
        )
        result = await launch_window(
            _make_query(),
            _make_context(user_data),
            _request(cwd=str(tmp_path), pending_text="first request"),
        )

    assert not result.success
    assert "still starting" in (result.error_message or "")
    m.router.commit_topic_provisioning.assert_not_called()
    m.router.bind_thread.assert_not_called()
    m.adapter.close.assert_not_awaited()
    forward.assert_not_awaited()
    assert user_data == {PENDING_THREAD_ID: 42, PENDING_THREAD_TEXT: "first request"}


@pytest.mark.parametrize("available", [True, False])
async def test_hookless_launch_rejects_another_provider(tmp_path, available):
    with (
        _launch_env(launch_command="agy") as m,
        patch(f"{_MODULE}asyncio.sleep", new_callable=AsyncMock),
        patch(
            "ccgram.multiplexer.reconciliation.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        m.adapter.resolve.return_value = (
            SimpleNamespace(
                foreground_argv=("codex",), provider="codex", available=available
            ),
            None,
        )
        result = await launch_window(
            _make_query(),
            _make_context({PENDING_THREAD_ID: 42}),
            _request(cwd=str(tmp_path), provider_name="antigravity"),
        )
    assert not result.success
    m.router.commit_topic_provisioning.assert_not_called()


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("/usr/local/bin/claude",), True),
        (("node", "/opt/node_modules/@anthropic-ai/claude-code/cli.js"), True),
        (("/bin/sh", "/opt/custom-wrapper"), False),
        (("node", "/opt/opaque-wrapper.js"), False),
        (("codex",), False),
    ],
)
async def test_custom_launcher_requires_recognizable_foreground(argv, expected):
    from ccgram.handlers.topics.window_launch_service import agent_process_started
    from ccgram.providers.process_detection import classify_provider_from_argv

    adapter = MagicMock()
    adapter.resolve = AsyncMock(
        return_value=(
            SimpleNamespace(
                foreground_argv=argv,
                provider=classify_provider_from_argv(argv) or "unknown",
                available=True,
            ),
            None,
        )
    )

    assert (
        await agent_process_started("@5", adapter, expected_provider="claude")
        is expected
    )
