import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import InlineKeyboardMarkup

from ccgram.handlers.callback_data import (
    CB_RECOVERY_BROWSE,
    CB_RECOVERY_CANCEL,
    CB_RECOVERY_CONTINUE,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_RESUME,
)
from ccgram.thread_router import ThreadRouter
from ccgram.handlers.recovery.recovery_banner import (
    RecoveryBanner,
    RecoveryMode,
    _create_and_bind_window,
    _recovery_cwd_or_report,
    render_banner,
)

from ccgram.handlers.user_state import RECOVERY_WINDOW_ID

_RC = "ccgram.handlers.recovery.recovery_banner"

_MODES: list[RecoveryMode] = ["dead", "restore", "resume"]


def _ctx() -> Any:
    ctx = MagicMock()
    ctx.user_data = {}
    return ctx


@pytest.fixture()
def _full_caps():
    with patch(f"{_RC}.get_provider_for_window") as mock_gpw:
        caps = mock_gpw.return_value.capabilities
        caps.supports_continue = True
        caps.supports_resume = True
        caps.supports_resume_picker = True
        yield mock_gpw


def _banner(mode: RecoveryMode, **overrides: Any) -> RecoveryBanner:
    fields: dict[str, Any] = {
        "chat_id": -100,
        "thread_id": 42,
        "window_id": "@0",
        "provider": None,
        "display": "my-project",
        "cwd": "/tmp/myproj",
    }
    fields.update(overrides)
    return RecoveryBanner(mode=mode, **fields)


def _callback_datas(keyboard: InlineKeyboardMarkup) -> list[str]:
    return [
        b.callback_data
        for row in keyboard.inline_keyboard
        for b in row
        if isinstance(b.callback_data, str)
    ]


class TestRenderBannerText:
    @pytest.mark.parametrize(
        ("mode", "expected_title"),
        [
            ("dead", "⚠ Session `my-project` ended."),
            ("restore", "\U0001f504 Restore `my-project`."),
            ("resume", "⏪ Resume `my-project`."),
        ],
    )
    def test_title_names_the_mode_and_the_window(
        self, _full_caps, mode: RecoveryMode, expected_title: str
    ) -> None:
        text, _ = render_banner(_banner(mode))

        assert text.startswith(expected_title)

    @pytest.mark.parametrize("mode", _MODES)
    def test_every_mode_explains_the_buttons(
        self, _full_caps, mode: RecoveryMode
    ) -> None:
        text, _ = render_banner(_banner(mode))

        assert "Start fresh · Continue last session · Resume from list" in text

    def test_includes_cwd_when_present(self, _full_caps) -> None:
        text, _ = render_banner(_banner("dead"))

        assert "/tmp/myproj" in text

    def test_omits_cwd_when_blank(self, _full_caps) -> None:
        text, _ = render_banner(_banner("dead", cwd=""))

        assert "📂" not in text

    def test_falls_back_to_window_id_when_no_display(self, _full_caps) -> None:
        text, _ = render_banner(_banner("dead", display="", window_id="@7"))

        assert "@7" in text


class TestRenderBannerKeyboard:
    @pytest.mark.parametrize("mode", _MODES)
    def test_every_mode_gets_the_action_keyboard(
        self, _full_caps, mode: RecoveryMode
    ) -> None:
        _, kb = render_banner(_banner(mode))

        datas = _callback_datas(kb)
        assert len(kb.inline_keyboard[0]) == 3
        assert any(d.startswith(CB_RECOVERY_FRESH) for d in datas)
        assert any(d.startswith(CB_RECOVERY_CONTINUE) for d in datas)
        assert any(d.startswith(CB_RECOVERY_RESUME) for d in datas)
        assert kb.inline_keyboard[1][0].callback_data == CB_RECOVERY_CANCEL

    def test_callback_data_within_64_bytes_for_long_window_id(self, _full_caps) -> None:
        _, kb = render_banner(_banner("dead", window_id="@" + "x" * 60))

        for data in _callback_datas(kb):
            assert len(data.encode("utf-8")) <= 64


class TestRecoveryCwdOrReport:
    """The two failures behind "Directory no longer exists" (#176).

    A missing window state means the folder is unknown; a missing folder
    means the folder is gone. They used to share one message that asserted
    the second while the first was true, leaving the banner a dead end.
    """

    @pytest.fixture()
    def _query(self):
        query = MagicMock()
        query.answer = AsyncMock()
        return query

    async def test_returns_cwd_when_state_and_directory_are_present(
        self, _query, tmp_path
    ) -> None:
        with patch(f"{_RC}._cwd_for_window", return_value=str(tmp_path)):
            result = await _recovery_cwd_or_report(_query, "@5", _ctx())

        assert result == str(tmp_path)

    async def test_missing_state_offers_browse_and_keeps_the_flow_alive(
        self, _query
    ) -> None:
        ctx = _ctx()
        ctx.user_data[RECOVERY_WINDOW_ID] = "@5"
        with (
            patch(f"{_RC}._cwd_for_window", return_value=""),
            patch(f"{_RC}.safe_edit") as mock_edit,
        ):
            result = await _recovery_cwd_or_report(_query, "@5", ctx)

        assert result is None
        text = mock_edit.call_args[0][1]
        assert "Directory no longer exists" not in text
        assert "session state is gone" in text
        kb = mock_edit.call_args.kwargs["reply_markup"]
        assert isinstance(kb, InlineKeyboardMarkup)
        browse = kb.inline_keyboard[0][0].callback_data
        assert isinstance(browse, str) and browse.startswith(CB_RECOVERY_BROWSE)
        # Browse re-validates against this state, so it must survive.
        assert ctx.user_data[RECOVERY_WINDOW_ID] == "@5"

    async def test_missing_directory_keeps_the_filesystem_message(
        self, _query, tmp_path
    ) -> None:
        ctx = _ctx()
        ctx.user_data[RECOVERY_WINDOW_ID] = "@5"
        gone = str(tmp_path / "gone")
        with (
            patch(f"{_RC}._cwd_for_window", return_value=gone),
            patch(f"{_RC}.safe_edit") as mock_edit,
        ):
            result = await _recovery_cwd_or_report(_query, "@5", ctx)

        assert result is None
        assert "Directory no longer exists" in mock_edit.call_args[0][1]
        assert RECOVERY_WINDOW_ID not in ctx.user_data


class TestStaleRecoveryOffer:
    """Fresh, Continue and Resume all converge on one replacement.

    That replacement unbinds the thread before creating its successor, and the
    banner offering it may have been drawn minutes earlier from a lookup that
    could not distinguish "gone" from "could not ask". So the decision is
    re-read here, from one confirmed snapshot.
    """

    @staticmethod
    async def _verdict(snapshot, *, returned_to_shell: bool = False):
        from ccgram.handlers.recovery.recovery_banner import _stale_recovery_offer

        async def _snapshot(*_a, **_kw):
            return snapshot

        async def _returned(*_a, **_kw):
            return returned_to_shell

        with (
            patch("ccgram.multiplexer.reconciliation.window_snapshot", _snapshot),
            patch(
                "ccgram.handlers.telegram_origin.agent_origin_returned_to_shell",
                _returned,
            ),
        ):
            return await _stale_recovery_offer("@5")

    async def test_unavailable_listing_changes_nothing(self) -> None:
        verdict = await self._verdict((False, None))
        assert verdict is not None
        assert "Could not reach" in verdict

    async def test_confirmed_live_agent_changes_nothing(self) -> None:
        from ccgram.multiplexer.base import WindowRef

        window = WindowRef(window_id="@5", window_name="proj", cwd="/p")
        verdict = await self._verdict((True, window), returned_to_shell=False)
        assert verdict is not None
        assert "running again" in verdict

    async def test_confirmed_missing_proceeds(self) -> None:
        assert await self._verdict((True, None)) is None

    async def test_confirmed_live_shell_after_agent_exit_proceeds(self) -> None:
        from ccgram.multiplexer.base import WindowRef

        window = WindowRef(window_id="@5", window_name="proj", cwd="/p")
        assert await self._verdict((True, window), returned_to_shell=True) is None

    async def test_no_old_window_is_not_a_refusal(self) -> None:
        from ccgram.handlers.recovery.recovery_banner import _stale_recovery_offer

        assert await _stale_recovery_offer("") is None


@pytest.fixture()
def create_env():
    with (
        patch(f"{_RC}.tmux_manager") as tmux,
        patch(f"{_RC}.thread_router") as router,
        patch(f"{_RC}.session_manager") as session,
        patch(f"{_RC}.safe_edit", new_callable=AsyncMock) as edit,
        patch(f"{_RC}.get_provider") as get_provider,
        patch(f"{_RC}.resolve_launch_command", return_value="claude"),
        patch(
            f"{_RC}._stale_recovery_offer", new_callable=AsyncMock, return_value=None
        ),
        patch(f"{_RC}.PTBTelegramClient") as client_factory,
    ):
        provider = get_provider.return_value
        provider.capabilities.name = "claude"
        provider.capabilities.supports_hook = False
        router.resolve_chat_id.return_value = -100
        router.begin_topic_provisioning.return_value = SimpleNamespace(
            claim_id="claim-1"
        )
        router.commit_topic_provisioning.return_value = True
        tmux.create_window = AsyncMock(return_value=(True, "created", "project", "@5"))
        tmux.kill_window = AsyncMock(return_value=True)
        client_factory.return_value.edit_forum_topic = AsyncMock()
        yield SimpleNamespace(
            tmux=tmux,
            router=router,
            session=session,
            edit=edit,
            provider=provider,
            client=client_factory.return_value,
        )


def _create_query() -> tuple[AsyncMock, MagicMock]:
    query = AsyncMock()
    query.message = MagicMock()
    query.message.chat.id = -100
    query.message.chat.type = "supergroup"
    context = MagicMock()
    context.user_data = {}
    context.bot = AsyncMock()
    return query, context


class TestRecoveryProvisioning:
    async def test_replacement_keeps_old_binding_until_commit(self, create_env) -> None:
        query, context = _create_query()

        await _create_and_bind_window(
            query,
            100,
            42,
            "/tmp/project",
            context,
            old_window_id="@0",
        )

        create_env.router.begin_topic_provisioning.assert_called_once_with(
            100,
            -100,
            thread_id=42,
            previous_target_id="@0",
            kind="replacement",
        )
        create_env.router.unbind_thread.assert_not_called()
        create_env.router.commit_topic_provisioning.assert_called_once_with(
            "claim-1", window_name="project"
        )


class TestRecoveryProbeCancellation:
    async def test_cancelled_stale_offer_releases_claim_preserves_old_binding_and_allows_retry(
        self,
    ) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _window_id: True,
        )
        router.bind_thread(100, 42, "@0", chat_id=-100)
        query, context = _create_query()

        with (
            patch(f"{_RC}.thread_router", router),
            patch(f"{_RC}.session_manager") as session,
            patch(f"{_RC}.tmux_manager") as tmux,
            patch(
                f"{_RC}._stale_recovery_offer",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
        ):
            tmux.create_window = AsyncMock()
            with pytest.raises(asyncio.CancelledError):
                await _create_and_bind_window(
                    query,
                    100,
                    42,
                    "/tmp/project",
                    context,
                    old_window_id="@0",
                )

        assert router.get_window_for_chat_thread(-100, 42) == "@0"
        assert router.iter_topic_provisionings() == []
        tmux.create_window.assert_not_awaited()

        retry = router.begin_topic_provisioning(
            100,
            -100,
            thread_id=42,
            previous_target_id="@0",
            kind="replacement",
        )
        assert retry.claim_id
        router.abort_topic_provisioning(retry.claim_id, target_confirmed_absent=True)
        assert session.flush_state.call_count == 2

    async def test_failed_replacement_preserves_old_binding(self, create_env) -> None:
        query, context = _create_query()
        create_env.tmux.create_window = AsyncMock(
            return_value=(False, "Directory does not exist: /gone", "", "")
        )

        result = await _create_and_bind_window(
            query,
            100,
            42,
            "/gone",
            context,
            old_window_id="@0",
        )

        assert result is False
        create_env.router.unbind_thread.assert_not_called()
        create_env.router.abort_topic_provisioning.assert_called_once_with(
            "claim-1", target_confirmed_absent=True
        )

    async def test_cancellation_cleans_completed_replacement_target(
        self, create_env
    ) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def create_window(*_args, **_kwargs):
            started.set()
            await release.wait()
            return True, "created", "project", "@5"

        create_env.tmux.create_window = AsyncMock(side_effect=create_window)
        query, context = _create_query()
        task = asyncio.create_task(
            _create_and_bind_window(
                query,
                100,
                42,
                "/tmp/project",
                context,
                old_window_id="@0",
            )
        )
        await started.wait()
        task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        create_env.tmux.kill_window.assert_awaited_once_with("@5")
        create_env.router.abort_topic_provisioning.assert_called_once_with(
            "claim-1", target_confirmed_absent=True
        )
        create_env.router.unbind_thread.assert_not_called()
