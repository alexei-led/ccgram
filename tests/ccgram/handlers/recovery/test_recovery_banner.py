from typing import Any
from unittest.mock import patch

import pytest
from telegram import InlineKeyboardMarkup

from ccgram.handlers.callback_data import (
    CB_RECOVERY_BROWSE,
    CB_RECOVERY_CANCEL,
    CB_RECOVERY_CONTINUE,
    CB_RECOVERY_FRESH,
    CB_RECOVERY_RESUME,
)
from ccgram.handlers.recovery.recovery_banner import (
    RecoveryBanner,
    RecoveryMode,
    render_banner,
)

from ccgram.handlers.user_state import RECOVERY_WINDOW_ID

_RC = "ccgram.handlers.recovery.recovery_banner"


def _ctx() -> Any:
    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.user_data = {}
    return ctx


@pytest.fixture()
def _full_caps():
    with patch(f"{_RC}.get_provider_for_window") as mock_gpw:
        caps = mock_gpw.return_value.capabilities
        caps.supports_continue = True
        caps.supports_resume = True
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


class TestRecoveryBannerDataclass:
    def test_is_frozen(self) -> None:
        banner = _banner("dead")
        with pytest.raises(Exception):
            banner.window_id = "@1"  # type: ignore[misc]

    def test_default_provider_and_display_optional(self) -> None:
        banner = RecoveryBanner(chat_id=1, thread_id=2, window_id="@5", mode="dead")
        assert banner.provider is None
        assert banner.display == ""
        assert banner.cwd == ""


class TestRenderBannerDeadMode:
    def test_returns_text_and_keyboard(self, _full_caps) -> None:
        text, kb = render_banner(_banner("dead"))
        assert isinstance(text, str)
        assert isinstance(kb, InlineKeyboardMarkup)

    def test_dead_text_says_session_ended(self, _full_caps) -> None:
        text, _ = render_banner(_banner("dead"))
        assert "Session" in text
        assert "ended" in text
        assert "my-project" in text

    def test_includes_cwd_when_present(self, _full_caps) -> None:
        text, _ = render_banner(_banner("dead"))
        assert "/tmp/myproj" in text

    def test_omits_cwd_when_blank(self, _full_caps) -> None:
        text, _ = render_banner(_banner("dead", cwd=""))
        assert "📂" not in text

    def test_falls_back_to_window_id_when_no_display(self, _full_caps) -> None:
        text, _ = render_banner(_banner("dead", display="", window_id="@7"))
        assert "@7" in text

    def test_includes_help_text(self, _full_caps) -> None:
        text, _ = render_banner(_banner("dead"))
        assert "Start fresh" in text
        assert "Continue last session" in text
        assert "Resume from list" in text

    def test_keyboard_has_three_action_buttons(self, _full_caps) -> None:
        _, kb = render_banner(_banner("dead"))
        assert len(kb.inline_keyboard[0]) == 3
        assert kb.inline_keyboard[1][0].callback_data == CB_RECOVERY_CANCEL


class TestRenderBannerRestoreMode:
    def test_restore_text_announces_restore(self, _full_caps) -> None:
        text, _ = render_banner(_banner("restore"))
        assert "Restore" in text
        assert "my-project" in text

    def test_includes_help_text(self, _full_caps) -> None:
        text, _ = render_banner(_banner("restore"))
        assert "Start fresh" in text
        assert "Continue last session" in text

    def test_uses_action_keyboard(self, _full_caps) -> None:
        _, kb = render_banner(_banner("restore"))
        datas = [
            b.callback_data
            for b in kb.inline_keyboard[0]
            if isinstance(b.callback_data, str)
        ]
        assert any(d.startswith(CB_RECOVERY_FRESH) for d in datas)
        assert any(d.startswith(CB_RECOVERY_CONTINUE) for d in datas)
        assert any(d.startswith(CB_RECOVERY_RESUME) for d in datas)


class TestRenderBannerResumeMode:
    def test_resume_text_says_resume(self, _full_caps) -> None:
        text, _ = render_banner(_banner("resume"))
        assert "Resume" in text or "resume" in text.lower()
        assert "my-project" in text

    def test_includes_help_text(self, _full_caps) -> None:
        text, _ = render_banner(_banner("resume"))
        assert "Start fresh" in text


class TestRenderBannerKeyboardCapabilities:
    def test_hides_continue_when_unsupported(self) -> None:
        with patch(f"{_RC}.get_provider_for_window") as mock_gpw:
            caps = mock_gpw.return_value.capabilities
            caps.supports_continue = False
            caps.supports_resume = True
            _, kb = render_banner(_banner("dead"))

        datas = [
            b.callback_data
            for b in kb.inline_keyboard[0]
            if isinstance(b.callback_data, str)
        ]
        assert not any(d.startswith(CB_RECOVERY_CONTINUE) for d in datas)
        assert any(d.startswith(CB_RECOVERY_FRESH) for d in datas)
        assert any(d.startswith(CB_RECOVERY_RESUME) for d in datas)

    def test_hides_resume_when_unsupported(self) -> None:
        with patch(f"{_RC}.get_provider_for_window") as mock_gpw:
            caps = mock_gpw.return_value.capabilities
            caps.supports_continue = True
            caps.supports_resume = False
            _, kb = render_banner(_banner("dead"))

        datas = [
            b.callback_data
            for b in kb.inline_keyboard[0]
            if isinstance(b.callback_data, str)
        ]
        assert any(d.startswith(CB_RECOVERY_FRESH) for d in datas)
        assert any(d.startswith(CB_RECOVERY_CONTINUE) for d in datas)
        assert not any(d.startswith(CB_RECOVERY_RESUME) for d in datas)


class TestRenderBannerCallbackBudget:
    def test_callback_data_within_64_bytes_for_long_window_id(self, _full_caps) -> None:
        long_id = "@" + "x" * 60
        _, kb = render_banner(_banner("dead", window_id=long_id))
        for row in kb.inline_keyboard:
            for btn in row:
                assert isinstance(btn.callback_data, str)
                assert len(btn.callback_data) <= 64


class TestRecoveryCwdOrReport:
    """The two failures behind "Directory no longer exists" (#176).

    A missing window state means the folder is unknown; a missing folder
    means the folder is gone. They used to share one message that asserted
    the second while the first was true, leaving the banner a dead end.
    """

    @pytest.fixture()
    def _query(self):
        from unittest.mock import AsyncMock, MagicMock

        query = MagicMock()
        query.answer = AsyncMock()
        return query

    async def test_returns_cwd_when_state_and_directory_are_present(
        self, _query, tmp_path
    ) -> None:
        from ccgram.handlers.recovery.recovery_banner import _recovery_cwd_or_report

        with patch(f"{_RC}._cwd_for_window", return_value=str(tmp_path)):
            result = await _recovery_cwd_or_report(_query, "@5", _ctx())

        assert result == str(tmp_path)

    async def test_missing_state_offers_browse_and_keeps_the_flow_alive(
        self, _query
    ) -> None:
        from ccgram.handlers.recovery.recovery_banner import _recovery_cwd_or_report

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
        from ccgram.handlers.recovery.recovery_banner import _recovery_cwd_or_report

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
