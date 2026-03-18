"""Tests for forward_command_handler CC command resolution."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.bot import (
    _command_known_in_other_provider,
    _extract_pane_delta,
    _extract_probe_error_line,
    _get_provider_command_metadata,
    _maybe_send_command_failure_message,
    _normalize_slash_token,
    _probe_transcript_command_error,
    _short_supported_commands,
    forward_command_handler,
)


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
    with patch("ccgram.bot.is_user_allowed", return_value=True):
        yield


class TestForwardCommandResolution:
    """Verify that sanitized Telegram command names are resolved to original CC names."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        self.mock_sm = MagicMock()
        self.mock_sm.resolve_window_for_thread.return_value = "@1"
        self.mock_sm.get_display_name.return_value = "project"
        self.mock_sm.send_to_window = AsyncMock(return_value=(True, ""))
        self.mock_sm.get_window_state.return_value = SimpleNamespace(
            transcript_path="",
            session_id="sess-1",
            cwd="/work/repo",
        )

        self.mock_tm = MagicMock()
        self.mock_tm.find_window_by_id = AsyncMock(
            return_value=MagicMock(window_id="@1")
        )
        self.mock_tm.capture_pane = AsyncMock(return_value="")
        self.mock_provider = SimpleNamespace(
            capabilities=SimpleNamespace(name="claude", supports_incremental_read=True)
        )
        self.mock_probe_ctx = AsyncMock(return_value=(None, None, None))
        self.mock_probe_spawn = MagicMock()

        with (
            patch("ccgram.bot.session_manager", self.mock_sm),
            patch("ccgram.bot.tmux_manager", self.mock_tm),
            patch(
                "ccgram.bot.get_provider_for_window", return_value=self.mock_provider
            ),
            patch(
                "ccgram.bot._get_provider_command_metadata",
                return_value=(
                    {
                        "clear": "clear",
                        "compact": "compact",
                        "committing_code": "committing-code",
                        "spec_work": "spec:work",
                        "spec_new": "spec:new",
                        "status": "/status",
                    },
                    {
                        "/clear",
                        "/compact",
                        "/committing-code",
                        "/spec:work",
                        "/spec:new",
                        "/status",
                    },
                ),
            ),
            patch("ccgram.bot._command_known_in_other_provider", return_value=False),
            patch(
                "ccgram.bot._capture_command_probe_context",
                self.mock_probe_ctx,
            ),
            patch(
                "ccgram.bot._spawn_command_failure_probe",
                self.mock_probe_spawn,
            ),
            patch("ccgram.bot._sync_scoped_provider_menu", new_callable=AsyncMock),
        ):
            yield

    async def test_builtin_forwarded_as_is(self) -> None:
        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/clear")

    async def test_builtin_with_args(self) -> None:
        update = _make_update(text="/compact focus on auth")
        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with(
            "@1", "/compact focus on auth"
        )

    async def test_skill_name_resolved(self) -> None:
        update = _make_update(text="/committing_code")
        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/committing-code")

    async def test_custom_command_resolved(self) -> None:
        update = _make_update(text="/spec_work")
        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/spec:work")

    async def test_custom_command_with_args(self) -> None:
        update = _make_update(text="/spec_new task auth")
        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/spec:new task auth")

    async def test_leading_slash_mapping_not_double_prefixed(self) -> None:
        update = _make_update(text="/status")
        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/status")

    async def test_unknown_command_forwarded_as_is(self) -> None:
        update = _make_update(text="/unknown_thing")
        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/unknown_thing")

    async def test_known_other_provider_command_is_rejected(self) -> None:
        with patch("ccgram.bot._command_known_in_other_provider", return_value=True):
            update = _make_update(text="/cost")
            await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_not_called()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "not supported" in reply_text
        assert "/commands" in reply_text

    async def test_botname_mention_stripped(self) -> None:
        update = _make_update(text="/clear@mybot")
        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/clear")

    async def test_botname_mention_stripped_with_args(self) -> None:
        update = _make_update(text="/compact@mybot some args")
        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/compact some args")

    async def test_confirmation_message_shows_resolved_name(self) -> None:
        update = _make_update(text="/committing_code")
        await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        # safe_reply escapes MarkdownV2 chars (- -> \-), so check unescaped
        assert "committing" in reply_text and "code" in reply_text

    async def test_clear_clears_session(self) -> None:
        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        self.mock_sm.clear_window_session.assert_called_once_with("@1")

    async def test_clear_enqueues_status_clear_and_resets_idle(self) -> None:
        from ccgram.handlers.status_polling import (
            _get_window_state,
            _window_poll_state,
            reset_seen_status_state,
        )

        _get_window_state("@1").has_seen_status = True
        try:
            with (
                patch(
                    "ccgram.handlers.message_queue.enqueue_status_update"
                ) as mock_enqueue,
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
            reset_seen_status_state()

    async def test_no_session_bound(self) -> None:
        self.mock_sm.resolve_window_for_thread.return_value = None

        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_not_called()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "No session" in reply_text

    async def test_window_gone(self) -> None:
        self.mock_tm.find_window_by_id = AsyncMock(return_value=None)

        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_not_called()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "no longer exists" in reply_text

    async def test_send_failure(self) -> None:
        self.mock_sm.send_to_window = AsyncMock(return_value=(False, "Connection lost"))

        update = _make_update(text="/clear")
        await forward_command_handler(update, _make_context())

        reply_text = update.message.reply_text.call_args[0][0]
        assert "Connection lost" in reply_text

    async def test_unauthorized_user(self) -> None:
        with (
            patch("ccgram.bot.is_user_allowed", return_value=False),
            patch("ccgram.bot._get_provider_command_metadata") as mock_metadata,
        ):
            update = _make_update(text="/clear")
            await forward_command_handler(update, _make_context())

        mock_metadata.assert_not_called()
        self.mock_sm.send_to_window.assert_not_called()

    async def test_no_message(self) -> None:
        update = _make_update(text="/clear")
        update.message = None

        await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_not_called()

    async def test_codex_status_sends_snapshot_reply(self) -> None:
        self.mock_sm.get_window_state.return_value = SimpleNamespace(
            transcript_path="/tmp/codex.jsonl",
            session_id="sess-1",
            cwd="/work/repo",
        )
        codex_provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))

        with (
            patch("ccgram.bot.get_provider_for_window", return_value=codex_provider),
            patch(
                "ccgram.bot.build_codex_status_snapshot",
                return_value="Codex status snapshot body",
            ) as mock_snapshot,
        ):
            update = _make_update(text="/status")
            await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/status")
        mock_snapshot.assert_called_once_with(
            "/tmp/codex.jsonl",
            display_name="project",
            session_id="sess-1",
            cwd="/work/repo",
        )
        assert update.message.reply_text.call_count == 2
        assert (
            "status snapshot body"
            in update.message.reply_text.call_args_list[1].args[0]
        )

    async def test_status_on_non_codex_skips_snapshot(self) -> None:
        claude_provider = SimpleNamespace(capabilities=SimpleNamespace(name="claude"))

        with (
            patch("ccgram.bot.get_provider_for_window", return_value=claude_provider),
            patch("ccgram.bot.build_codex_status_snapshot") as mock_snapshot,
        ):
            update = _make_update(text="/status")
            await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/status")
        mock_snapshot.assert_not_called()
        assert update.message.reply_text.call_count == 1

    async def test_codex_status_skips_fallback_when_native_reply_exists(self) -> None:
        self.mock_sm.get_window_state.return_value = SimpleNamespace(
            transcript_path="/tmp/codex.jsonl",
            session_id="sess-1",
            cwd="/work/repo",
        )
        codex_provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))

        with (
            patch("ccgram.bot.get_provider_for_window", return_value=codex_provider),
            patch("ccgram.bot._codex_status_probe_offset", return_value=0),
            patch("ccgram.bot.has_codex_assistant_output_since", return_value=True),
            patch("ccgram.bot.build_codex_status_snapshot") as mock_snapshot,
            patch("ccgram.bot.asyncio.sleep", new_callable=AsyncMock),
        ):
            update = _make_update(text="/status")
            await forward_command_handler(update, _make_context())

        self.mock_sm.send_to_window.assert_called_once_with("@1", "/status")
        mock_snapshot.assert_not_called()
        assert update.message.reply_text.call_count == 1


class TestCommandFailureProbe:
    async def test_probe_transcript_uses_incremental_reader_for_codex(
        self, tmp_path
    ) -> None:
        transcript = tmp_path / "session.jsonl"
        prefix = "ok\n"
        suffix = "unknown command: /status\n"
        transcript.write_text(prefix + suffix, encoding="utf-8")

        provider = SimpleNamespace(
            capabilities=SimpleNamespace(supports_incremental_read=True),
            parse_transcript_line=lambda line: (
                {"text": line.strip()} if line.strip() else None
            ),
            parse_transcript_entries=lambda entries, pending_tools: (
                [
                    SimpleNamespace(role="assistant", text=entry["text"])
                    for entry in entries
                ],
                pending_tools,
            ),
            read_transcript_file=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                NotImplementedError("incremental only")
            ),
        )

        result = await _probe_transcript_command_error(
            provider,
            str(transcript),
            len(prefix),
        )
        assert result == "unknown command: /status"

    async def test_probe_transcript_whole_file_not_implemented_returns_none(
        self, tmp_path
    ) -> None:
        transcript = tmp_path / "session.json"
        transcript.write_text("{}", encoding="utf-8")

        provider = SimpleNamespace(
            capabilities=SimpleNamespace(supports_incremental_read=False),
            read_transcript_file=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                NotImplementedError("not implemented")
            ),
            parse_transcript_entries=lambda entries, pending_tools: ([], pending_tools),
        )

        result = await _probe_transcript_command_error(provider, str(transcript), 0)
        assert result is None

    async def test_surfaces_transcript_error(self) -> None:
        provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
        message = AsyncMock()

        with (
            patch("ccgram.bot.asyncio.sleep", new_callable=AsyncMock),
            patch(
                "ccgram.bot._probe_transcript_command_error",
                new_callable=AsyncMock,
                return_value="unrecognized command '/cost'",
            ),
            patch("ccgram.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            await _maybe_send_command_failure_message(
                message,
                "@1",
                "project",
                "/cost",
                provider=provider,
                transcript_path="/tmp/codex.jsonl",
                since_offset=0,
                pane_before="",
            )

        mock_reply.assert_called_once()
        assert "failed" in mock_reply.call_args.args[1]
        assert "unrecognized command" in mock_reply.call_args.args[1]

    async def test_falls_back_to_pane_delta_when_transcript_has_no_error(self) -> None:
        provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
        message = AsyncMock()

        with (
            patch("ccgram.bot.asyncio.sleep", new_callable=AsyncMock),
            patch(
                "ccgram.bot._probe_transcript_command_error",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "ccgram.bot.tmux_manager.capture_pane",
                new_callable=AsyncMock,
                return_value="before\nunknown command: /cost",
            ),
            patch("ccgram.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            await _maybe_send_command_failure_message(
                message,
                "@1",
                "project",
                "/cost",
                provider=provider,
                transcript_path=None,
                since_offset=None,
                pane_before="before",
            )

        mock_reply.assert_called_once()
        assert "unknown command" in mock_reply.call_args.args[1]

    async def test_no_error_found_sends_no_message(self) -> None:
        provider = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
        message = AsyncMock()

        with (
            patch("ccgram.bot.asyncio.sleep", new_callable=AsyncMock),
            patch(
                "ccgram.bot._probe_transcript_command_error",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "ccgram.bot.tmux_manager.capture_pane",
                new_callable=AsyncMock,
                return_value="before\nall good",
            ),
            patch("ccgram.bot.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            await _maybe_send_command_failure_message(
                message,
                "@1",
                "project",
                "/cost",
                provider=provider,
                transcript_path=None,
                since_offset=None,
                pane_before="before",
            )

        mock_reply.assert_not_called()


class TestCommandHelperFunctions:
    def test_normalize_slash_token(self) -> None:
        assert _normalize_slash_token("COST") == "/cost"
        assert _normalize_slash_token("/STATUS now") == "/status"
        assert _normalize_slash_token("   ") == "/"

    def test_extract_probe_error_line(self) -> None:
        assert (
            _extract_probe_error_line("ok\nunrecognized command '/cost'\n")
            == "unrecognized command '/cost'"
        )
        assert (
            _extract_probe_error_line("all good\nERROR executing command /x\n")
            == "ERROR executing command /x"
        )
        assert _extract_probe_error_line("all good\nstill fine\n") is None

    def test_extract_pane_delta(self) -> None:
        assert _extract_pane_delta("line1\nline2", "line1\nline2\nline3") == "line3"
        assert _extract_pane_delta("A\nB", "B\nC\nD") == "C\nD"
        assert _extract_pane_delta("same", "same") == ""
        assert _extract_pane_delta(None, "only after") == "only after"
        assert _extract_pane_delta("abc", "xabcx\ndef") == "xabcx\ndef"

    def test_short_supported_commands_default(self) -> None:
        assert (
            _short_supported_commands(set())
            == "Use /commands to list available commands."
        )

    def test_short_supported_commands_truncates(self) -> None:
        supported = {f"/cmd{i}" for i in range(10)}
        summary = _short_supported_commands(supported, limit=3)
        assert summary.startswith("Try: ")
        assert " …" in summary
        assert summary.count("/cmd") == 3

    def test_command_known_in_other_provider(self) -> None:
        current = SimpleNamespace(capabilities=SimpleNamespace(name="codex"))
        claude = SimpleNamespace(capabilities=SimpleNamespace(name="claude"))
        gemini = SimpleNamespace(capabilities=SimpleNamespace(name="gemini"))

        def _supported(provider: SimpleNamespace) -> set[str]:
            if provider.capabilities.name == "claude":
                return {"/cost"}
            return set()

        with (
            patch(
                "ccgram.bot.registry.provider_names",
                return_value=["codex", "claude", "gemini"],
            ),
            patch(
                "ccgram.bot.registry.get",
                side_effect=lambda name: {"claude": claude, "gemini": gemini}[name],
            ),
            patch(
                "ccgram.bot._get_provider_command_metadata",
                side_effect=lambda provider: ({}, _supported(provider)),
            ),
        ):
            assert _command_known_in_other_provider("/cost", current) is True
            assert _command_known_in_other_provider("/not-here", current) is False

    def test_get_provider_command_metadata_builds_mapping_and_supported(self) -> None:
        provider = SimpleNamespace(
            capabilities=SimpleNamespace(name="codex", builtin_commands=("/builtin",))
        )
        discovered = [SimpleNamespace(name="/status", telegram_name="status")]

        with patch("ccgram.bot.discover_provider_commands", return_value=discovered):
            mapping, supported = _get_provider_command_metadata(provider)

        assert mapping == {"status": "/status"}
        assert supported == {"/status", "/builtin"}
