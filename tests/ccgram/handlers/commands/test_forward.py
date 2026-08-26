from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.commands import forward_command_handler


_FW = "ccgram.handlers.commands.forward"
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
                tui_picker_commands=frozenset(),
            )
        )
        self.mock_probe_ctx = AsyncMock(return_value=(None, None, None))
        self.mock_probe_spawn = MagicMock()

        with (
            patch(f"{_FW}.thread_router", self.mock_tr),
            patch(f"{_FW}.window_store", self.mock_ws),
            patch(f"{_FW}.window_query", self.mock_wq),
            patch(
                f"{_FW}.send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as self.mock_send_to_window,
            patch(
                f"{_FW}.send_telegram_followup_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as self.mock_send_followup_to_window,
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
                    "new": "/new",
                    "scoped_models": "/scoped-models",
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

    @pytest.mark.parametrize(
        ("typed", "forwarded"),
        [
            pytest.param("/clear", "/clear", id="builtin"),
            pytest.param(
                "/compact focus on auth",
                "/compact focus on auth",
                id="builtin-with-args",
            ),
            pytest.param("/committing_code", "/committing-code", id="skill-name"),
            pytest.param("/spec_work", "/spec:work", id="custom-command"),
            pytest.param(
                "/spec_new task auth", "/spec:new task auth", id="custom-with-args"
            ),
            pytest.param("/status", "/status", id="mapping-already-slash-prefixed"),
            pytest.param("/unknown_thing", "/unknown_thing", id="unknown-command"),
            pytest.param("/cost", "/cost", id="cross-provider-command"),
            pytest.param("/clear@mybot", "/clear", id="botname-mention"),
            pytest.param(
                "/compact@mybot some args",
                "/compact some args",
                id="botname-mention-with-args",
            ),
            pytest.param(
                "/followup run tests",
                "/followup run tests",
                id="followup-on-non-pi-provider",
            ),
        ],
    )
    async def test_command_resolution(self, typed: str, forwarded: str) -> None:
        await forward_command_handler(_make_update(text=typed), _make_context())

        self.mock_send_to_window.assert_called_once_with(
            100, "@1", 42, forwarded, -100999
        )
        self.mock_send_followup_to_window.assert_not_called()

    async def test_confirmation_message_shows_resolved_name(self) -> None:
        update = _make_update(text="/committing_code")
        await forward_command_handler(update, _make_context())

        assert (
            update.message.reply_text.call_args[0][0]
            == "⚡ [project] Sent: /committing-code"
        )

    async def test_pi_followup_queues_followup_message(self) -> None:
        self.mock_provider.capabilities.name = "pi"
        update = _make_update(text="/followup run tests")
        await forward_command_handler(update, _make_context())

        self.mock_send_followup_to_window.assert_called_once_with(
            100, "@1", 42, "run tests", -100999
        )
        self.mock_send_to_window.assert_not_called()
        reply_text = update.message.reply_text.call_args[0][0]
        assert reply_text == "⏭️ [project] Follow-up queued."

    async def test_pi_followup_requires_message(self) -> None:
        self.mock_provider.capabilities.name = "pi"
        update = _make_update(text="/followup")
        await forward_command_handler(update, _make_context())

        self.mock_send_followup_to_window.assert_not_called()
        self.mock_send_to_window.assert_not_called()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "Usage: /followup <message>" in reply_text

    async def test_pi_new_forwarded_as_session_reset(self) -> None:
        self.mock_provider.capabilities.name = "pi"
        update = _make_update(text="/new")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with(100, "@1", 42, "/new", -100999)
        self.mock_ws.clear_window_session.assert_called_once_with("@1")

    async def test_pi_clear_alias_forwards_to_new(self) -> None:
        self.mock_provider.capabilities.name = "pi"
        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with(100, "@1", 42, "/new", -100999)
        self.mock_ws.clear_window_session.assert_called_once_with("@1")
        reply_text = update.message.reply_text.call_args[0][0]
        assert "Sent: /new" in reply_text

    async def test_pi_scoped_models_telegram_name_resolves_to_native_command(
        self,
    ) -> None:
        self.mock_provider.capabilities.name = "pi"
        self.mock_provider.capabilities.tui_picker_commands = frozenset(
            {"scoped-models"}
        )
        update = _make_update(text="/scoped_models")
        await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with(
            100, "@1", 42, "/scoped-models", -100999
        )
        reply_text = update.message.reply_text.call_args[0][0]
        assert "drive the picker" in reply_text

    @pytest.mark.parametrize(
        ("typed", "hint_expected"),
        [
            pytest.param("/model", True, id="picker-command"),
            pytest.param("/model@mybot", True, id="picker-command-with-botname"),
            pytest.param("/clear", False, id="non-picker-command"),
            pytest.param(
                "/model claude-opus-4-5", False, id="picker-command-with-args"
            ),
        ],
    )
    async def test_tui_picker_hint(self, typed: str, hint_expected: bool) -> None:
        self.mock_provider.capabilities.tui_picker_commands = frozenset({"model"})
        update = _make_update(text=typed)
        await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        assert ("drive the picker" in reply_text) is hint_expected
        assert ("/toolbar" in reply_text) is hint_expected

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
                tui_picker_commands=frozenset(),
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

        self.mock_send_to_window.assert_called_once_with(
            100, "@1", 42, "/status", -100999
        )
        self.mock_probe_ctx.assert_not_called()
        self.mock_probe_spawn.assert_not_called()
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
                tui_picker_commands=frozenset(),
            ),
            build_status_snapshot=MagicMock(return_value=None),
        )

        with (
            patch(f"{_FW}.get_provider_for_window", return_value=claude_provider),
            patch(f"{_SS}.get_provider_for_window", return_value=claude_provider),
        ):
            update = _make_update(text="/status")
            await forward_command_handler(update, _make_context())

        self.mock_send_to_window.assert_called_once_with(
            100, "@1", 42, "/status", -100999
        )
        self.mock_probe_ctx.assert_not_called()
        self.mock_probe_spawn.assert_not_called()
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
                tui_picker_commands=frozenset(),
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

        self.mock_send_to_window.assert_called_once_with(
            100, "@1", 42, "/status", -100999
        )
        codex_provider.build_status_snapshot.assert_not_called()
        assert update.message.reply_text.call_count == 1

    @pytest.mark.parametrize(
        ("typed", "provider_name", "armed"),
        [
            pytest.param("/remote-control", "claude", True, id="claude-remote-control"),
            pytest.param("/rc", "claude", True, id="claude-rc-alias"),
            pytest.param("/remote-control", "codex", True, id="codex-remote-control"),
            pytest.param("/clear", "claude", False, id="unrelated-command"),
        ],
    )
    async def test_rc_probe_arming(
        self, typed: str, provider_name: str, armed: bool
    ) -> None:
        """Both /remote-control and its /rc alias arm the probe, on any provider."""
        from ccgram.telegram_client import PTBTelegramClient

        self.mock_provider.capabilities.name = provider_name
        with patch("ccgram.handlers.status.rc_probe.arm_rc_probe") as mock_arm:
            await forward_command_handler(_make_update(text=typed), _make_context())

        if not armed:
            mock_arm.assert_not_called()
            return
        mock_arm.assert_called_once()
        assert mock_arm.call_args.args[0] == "@1"
        assert isinstance(mock_arm.call_args.args[1], PTBTelegramClient)
        self.mock_send_to_window.assert_called_once_with(
            100, "@1", 42, f"{typed} project", -100999
        )


def _real_provider(name: str):
    if name == "claude":
        from ccgram.providers.claude import ClaudeProvider

        return ClaudeProvider()
    if name == "codex":
        from ccgram.providers.codex import CodexProvider

        return CodexProvider()
    if name == "gemini":
        from ccgram.providers.gemini import GeminiProvider

        return GeminiProvider()
    if name == "pi":
        from ccgram.providers.pi import PiProvider

        return PiProvider()
    raise ValueError(name)


class TestForwardWithRealProvider:
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

        self._provider_patch = patch(f"{_FW}.get_provider_for_window")
        self._mock_get_provider = self._provider_patch.start()

        with (
            patch(f"{_FW}.thread_router", self.mock_tr),
            patch(f"{_FW}.window_store", self.mock_ws),
            patch(f"{_FW}.window_query", self.mock_wq),
            patch(
                f"{_FW}.send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as self.mock_send_to_window,
            patch(f"{_FW}.tmux_manager", self.mock_tm),
            patch(
                f"{_FW}._build_provider_command_metadata",
                return_value={},
            ),
            patch(
                f"{_FW}._capture_command_probe_context",
                AsyncMock(return_value=(None, None, None)),
            ),
            patch(f"{_FW}._spawn_command_failure_probe", MagicMock()),
            patch(f"{_FW}.sync_scoped_provider_menu", new_callable=AsyncMock),
            patch(f"{_FW}._maybe_send_status_snapshot", new_callable=AsyncMock),
            patch("ccgram.handlers.status.rc_probe.arm_rc_probe"),
        ):
            yield

        self._provider_patch.stop()

    @pytest.mark.parametrize(
        "provider_name,picker_cmd",
        [
            ("claude", "model"),
            ("codex", "model"),
            ("gemini", "model"),
            ("pi", "model"),
            ("claude", "effort"),
            ("codex", "personality"),
            ("gemini", "auth"),
            ("pi", "login"),
        ],
    )
    async def test_picker_command_produces_hint(
        self, provider_name: str, picker_cmd: str
    ) -> None:
        self._mock_get_provider.return_value = _real_provider(provider_name)
        update = _make_update(text=f"/{picker_cmd}")
        await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        assert f"Sent: /{picker_cmd}" in reply_text
        assert "drive the picker" in reply_text
        assert "/toolbar" in reply_text

    @pytest.mark.parametrize("provider_name", ["claude", "codex", "gemini", "pi"])
    async def test_non_picker_command_no_hint(self, provider_name: str) -> None:
        self._mock_get_provider.return_value = _real_provider(provider_name)
        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        assert "/toolbar" not in reply_text
        assert "drive the picker" not in reply_text

    @pytest.mark.parametrize("provider_name", ["claude", "codex", "gemini", "pi"])
    async def test_picker_command_with_args_no_hint(self, provider_name: str) -> None:
        self._mock_get_provider.return_value = _real_provider(provider_name)
        update = _make_update(text="/model some-value")
        await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        assert "/toolbar" not in reply_text
        assert "drive the picker" not in reply_text

    @pytest.mark.parametrize("provider_name", ["claude", "codex", "gemini", "pi"])
    async def test_uppercase_picker_command_still_fires_hint(
        self, provider_name: str
    ) -> None:
        self._mock_get_provider.return_value = _real_provider(provider_name)
        update = _make_update(text="/MODEL")
        await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        assert "drive the picker" in reply_text

    async def test_claude_hyphenated_picker_reachable_via_telegram_form(self) -> None:
        """`/release-notes` is hyphenated; Telegram bots only accept underscores.

        The user types `/release_notes`; the provider_map reverses it to the
        original `release-notes`. Verify the picker hint still fires.
        """
        from ccgram.providers.claude import ClaudeProvider

        claude = ClaudeProvider()
        assert "release-notes" in claude.capabilities.tui_picker_commands
        self._mock_get_provider.return_value = claude
        with patch(
            f"{_FW}._build_provider_command_metadata",
            return_value={"release_notes": "release-notes"},
        ):
            update = _make_update(text="/release_notes")
            await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        assert "drive the picker" in reply_text

    async def test_degraded_hint_when_toolbar_lacks_nav_keys(self) -> None:
        """Custom toolbar that drops up/down must not be promised in the hint."""
        from ccgram.providers.claude import ClaudeProvider
        from ccgram.toolbar_config import (
            BUILTIN_ACTIONS,
            ToolbarConfig,
            ToolbarLayout,
        )

        stripped = ToolbarConfig(
            layouts={
                "claude": ToolbarLayout(
                    style="emoji_text",
                    buttons=(("screen", "send", "close"),),
                )
            },
            actions=dict(BUILTIN_ACTIONS),
        )
        self._mock_get_provider.return_value = ClaudeProvider()
        with patch(
            "ccgram.handlers.toolbar.toolbar_keyboard.get_toolbar_config",
            return_value=stripped,
        ):
            update = _make_update(text="/model")
            await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        # Degraded copy: points at /toolbar but does not promise specific buttons.
        assert "/toolbar" in reply_text
        assert "drive the picker" in reply_text
        assert "🔼" not in reply_text
        assert "🔽" not in reply_text
        assert "Enter Esc" not in reply_text

    async def test_full_hint_when_toolbar_has_nav_keys(self) -> None:
        """Default toolbar has up/down/enter/esc — full hint with glyphs fires."""
        from ccgram.providers.claude import ClaudeProvider

        self._mock_get_provider.return_value = ClaudeProvider()
        update = _make_update(text="/model")
        await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        assert "🔼" in reply_text
        assert "🔽" in reply_text
        assert "Enter Esc" in reply_text

    async def test_shell_provider_never_emits_picker_hint(self) -> None:
        from ccgram.providers.shell import ShellProvider

        self._mock_get_provider.return_value = ShellProvider()
        update = _make_update(text="/model")
        await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        assert "drive the picker" not in reply_text
        assert "/toolbar" not in reply_text
