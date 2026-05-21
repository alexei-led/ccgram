from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.commands import forward_command_handler


_FW = "ccgram.handlers.commands.forward"
_MS = "ccgram.handlers.commands.menu_sync"
_FP = "ccgram.handlers.commands.failure_probe"
_SS = "ccgram.handlers.commands.status_snapshot"


def _make_update(
    *,
    user_id: int = 100,
    thread_id: int = 42,
    text: str = "/clear",
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    msg = AsyncMock()
    msg.text = text
    msg.message_thread_id = thread_id
    msg.chat.type = "supergroup"
    msg.chat.id = -100999
    msg.chat.is_forum = True
    msg.is_topic_message = True
    msg.get_bot = MagicMock(return_value=MagicMock(send_chat_action=AsyncMock()))
    update.message = msg
    update.callback_query = None
    return update


def _make_context() -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = {}
    return ctx


@pytest.fixture(autouse=True)
def _allow_user():
    with patch("ccgram.config.Config.is_user_allowed", return_value=True):
        yield


class TestForwardCommandResolution:
    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        self.mock_tr = MagicMock()
        self.mock_tr.resolve_window_for_thread.return_value = "@1"
        self.mock_tr.get_display_name.return_value = "project"
        self.mock_tr.set_group_chat_id = MagicMock()

        self.mock_ws = MagicMock()

        self.mock_wq = MagicMock()
        self.mock_wq.view_window.return_value = SimpleNamespace(
            transcript_path=None,
            session_id="sess-1",
            cwd="/work/repo",
            provider_name="claude",
        )
        self.mock_wq.get_window_provider.return_value = "claude"

        self.mock_tm = MagicMock()
        self.mock_tm.find_window_by_id = AsyncMock(
            return_value=MagicMock(window_id="@1")
        )
        self.mock_tm.capture_pane = AsyncMock(return_value="")
        self.mock_provider = SimpleNamespace(
            capabilities=SimpleNamespace(
                name="claude",
                supports_incremental_read=True,
                supports_status_snapshot=False,
            )
        )
        self.mock_probe_ctx = AsyncMock(return_value=(None, None, None))
        self.mock_probe_spawn = MagicMock()

        with (
            patch(f"{_FW}.thread_router", self.mock_tr),
            patch(f"{_FW}.window_store", self.mock_ws),
            patch(f"{_FW}.window_query", self.mock_wq),
            patch(
                f"{_FW}.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as self.mock_send_to_window,
            patch(f"{_FW}.tmux_manager", self.mock_tm),
            patch(
                f"{_FW}.get_provider_for_window",
                return_value=self.mock_provider,
            ),
            patch(
                f"{_FW}._build_provider_command_metadata",
                return_value={
                    "clear": "clear",
                    "compact": "compact",
                    "committing_code": "committing-code",
                    "spec_work": "spec:work",
                    "spec_new": "spec:new",
                    "status": "/status",
                },
            ),
            patch(
                f"{_FW}._capture_command_probe_context",
                self.mock_probe_ctx,
            ),
            patch(
                f"{_FW}._spawn_command_failure_probe",
                self.mock_probe_spawn,
            ),
            patch(
                f"{_FW}.sync_scoped_provider_menu",
                new_callable=AsyncMock,
            ),
        ):
            yield

    async def test_builtin_forwarded_as_is(self) -> None:
        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/clear")

    async def test_builtin_with_args(self) -> None:
        update = _make_update(text="/compact focus on auth")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/compact focus on auth")

    async def test_skill_name_resolved(self) -> None:
        update = _make_update(text="/committing_code")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/committing-code")

    async def test_custom_command_resolved(self) -> None:
        update = _make_update(text="/spec_work")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/spec:work")

    async def test_custom_command_with_args(self) -> None:
        update = _make_update(text="/spec_new task auth")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/spec:new task auth")

    async def test_leading_slash_mapping_not_double_prefixed(self) -> None:
        update = _make_update(text="/status")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/status")

    async def test_unknown_command_forwarded_as_is(self) -> None:
        update = _make_update(text="/unknown_thing")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/unknown_thing")

    async def test_cross_provider_command_forwarded_to_provider(self) -> None:
        update = _make_update(text="/cost")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/cost")

    async def test_botname_mention_stripped(self) -> None:
        update = _make_update(text="/clear@mybot")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/clear")

    async def test_botname_mention_stripped_with_args(self) -> None:
        update = _make_update(text="/compact@mybot some args")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/compact some args")

    async def test_confirmation_message_shows_resolved_name(self) -> None:
        update = _make_update(text="/committing_code")
        await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        assert "committing" in reply_text and "code" in reply_text

    async def test_clear_clears_session(self) -> None:
        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        self.mock_ws.clear_window_session.assert_called_once_with("@1")

    async def test_clear_enqueues_status_clear_and_resets_idle(self) -> None:
        from ccgram.handlers.polling.polling_state import terminal_poll_state

        _window_poll_state = terminal_poll_state._states

        terminal_poll_state.get_state("@1").has_seen_status = True
        try:
            with (
                patch(f"{_FW}.enqueue_status_update") as mock_enqueue,
            ):
                update = _make_update(text="/clear")
                await forward_command_handler(update, _make_context())

            mock_enqueue.assert_called_once()
            call_args = mock_enqueue.call_args
            assert call_args[0][1] == 100  # user_id
            assert call_args[0][2] == "@1"  # window_id
            assert call_args[0][3] is None  # status_text (clear)
            assert call_args[1]["thread_id"] == 42
            assert not (
                _window_poll_state.get("@1")
                and _window_poll_state["@1"].has_seen_status
            )
        finally:
            terminal_poll_state.reset_all_seen_status()

    async def test_no_session_bound(self) -> None:
        self.mock_tr.resolve_window_for_thread.return_value = None

        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_not_called()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "No session" in reply_text

    async def test_window_gone(self) -> None:
        self.mock_tm.find_window_by_id = AsyncMock(return_value=None)

        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_not_called()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "no longer exists" in reply_text

    async def test_send_failure(self) -> None:
        self.mock_send_to_window.return_value = (False, "Connection lost")

        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Connection lost" in reply_text

    async def test_unauthorized_user(self) -> None:
        with (
            patch("ccgram.config.Config.is_user_allowed", return_value=False),
            patch(f"{_FW}._build_provider_command_metadata") as mock_metadata,
        ):
            update = _make_update(text="/clear")
            await forward_command_handler(update, _make_context())

        mock_metadata.assert_not_called()
        self.mock_send_to_window.assert_not_called()

    async def test_no_message(self) -> None:
        update = _make_update(text="/clear")
        update.message = None

        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_not_called()

    async def test_status_snapshot_sends_reply(self) -> None:
        mock_path = MagicMock(spec=Path)
        mock_path.__str__ = MagicMock(return_value="/tmp/codex.jsonl")
        mock_path.stat.return_value.st_size = 1024
        _view = SimpleNamespace(
            transcript_path=mock_path,
            session_id="sess-1",
            cwd="/work/repo",
            provider_name="codex",
        )
        self.mock_wq.view_window.return_value = _view
        codex_provider = SimpleNamespace(
            capabilities=SimpleNamespace(
                name="codex",
                supports_incremental_read=True,
                supports_status_snapshot=True,
            ),
            build_status_snapshot=MagicMock(return_value="Status snapshot body"),
            has_output_since=MagicMock(return_value=False),
        )

        with (
            patch(f"{_FW}.get_provider_for_window", return_value=codex_provider),
            patch(f"{_SS}.get_provider_for_window", return_value=codex_provider),
            patch(f"{_SS}.window_query", self.mock_wq),
        ):
            update = _make_update(text="/status")
            await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/status")
        codex_provider.build_status_snapshot.assert_called_once_with(
            "/tmp/codex.jsonl",
            display_name="project",
            session_id="sess-1",
            cwd="/work/repo",
        )
        assert update.message.reply_text.call_count == 2
        assert "snapshot body" in update.message.reply_text.call_args_list[1].args[0]

    async def test_status_on_non_snapshot_provider_skips_snapshot(self) -> None:
        claude_provider = SimpleNamespace(
            capabilities=SimpleNamespace(
                name="claude",
                supports_incremental_read=True,
                supports_status_snapshot=False,
            ),
            build_status_snapshot=MagicMock(return_value=None),
        )

        with (
            patch(f"{_FW}.get_provider_for_window", return_value=claude_provider),
            patch(f"{_SS}.get_provider_for_window", return_value=claude_provider),
        ):
            update = _make_update(text="/status")
            await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/status")
        claude_provider.build_status_snapshot.assert_not_called()
        assert update.message.reply_text.call_count == 1

    async def test_status_snapshot_skips_fallback_when_native_reply_exists(
        self,
    ) -> None:
        mock_path2 = MagicMock(spec=Path)
        mock_path2.__str__ = MagicMock(return_value="/tmp/codex.jsonl")
        mock_path2.stat.return_value.st_size = 1024
        _view2 = SimpleNamespace(
            transcript_path=mock_path2,
            session_id="sess-1",
            cwd="/work/repo",
            provider_name="codex",
        )
        self.mock_wq.view_window.return_value = _view2
        codex_provider = SimpleNamespace(
            capabilities=SimpleNamespace(
                name="codex",
                supports_incremental_read=True,
                supports_status_snapshot=True,
            ),
            build_status_snapshot=MagicMock(return_value=None),
            has_output_since=MagicMock(return_value=True),
        )

        with (
            patch(f"{_FW}.get_provider_for_window", return_value=codex_provider),
            patch(f"{_SS}.get_provider_for_window", return_value=codex_provider),
            patch(f"{_SS}.window_query", self.mock_wq),
            patch(f"{_FW}._status_snapshot_probe_offset", return_value=0),
            patch(f"{_SS}.asyncio.sleep", new_callable=AsyncMock),
        ):
            update = _make_update(text="/status")
            await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with("@1", "/status")
        codex_provider.build_status_snapshot.assert_not_called()
        assert update.message.reply_text.call_count == 1

    async def test_arms_rc_probe_for_claude_remote_control(self) -> None:
        from ccgram.telegram_client import PTBTelegramClient

        with patch("ccgram.handlers.status.rc_probe.arm_rc_probe") as mock_arm:
            update = _make_update(text="/remote-control")
            await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with(
            "@1", "/remote-control project"
        )
        mock_arm.assert_called_once()
        args = mock_arm.call_args.args
        assert args[0] == "@1"
        assert isinstance(args[1], PTBTelegramClient)

    async def test_arms_rc_probe_for_rc_alias(self) -> None:
        with patch("ccgram.handlers.status.rc_probe.arm_rc_probe") as mock_arm:
            update = _make_update(text="/rc")
            await forward_command_handler(update, _make_context())

        mock_arm.assert_called_once()

    async def test_no_rc_probe_for_non_rc_command(self) -> None:
        with patch("ccgram.handlers.status.rc_probe.arm_rc_probe") as mock_arm:
            update = _make_update(text="/clear")
            await forward_command_handler(update, _make_context())

        mock_arm.assert_not_called()

    async def test_codex_remote_control_forwarded_arm_delegates_to_probe(self) -> None:
        codex_provider = SimpleNamespace(
            capabilities=SimpleNamespace(
                name="codex",
                supports_incremental_read=True,
                supports_status_snapshot=False,
            )
        )
        with (
            patch(f"{_FW}.get_provider_for_window", return_value=codex_provider),
            patch("ccgram.handlers.status.rc_probe.arm_rc_probe") as mock_arm,
        ):
            update = _make_update(text="/remote-control")
            await forward_command_handler(update, _make_context())

        # Gate is gone: the slash forwards to the provider. The capability
        # check now lives inside arm_rc_probe, which no-ops for non-Claude.
        self.mock_send_to_window.assert_called_once_with(
            "@1", "/remote-control project"
        )
        mock_arm.assert_called_once()
