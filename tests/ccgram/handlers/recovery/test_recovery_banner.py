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
from ccgram.handlers.recovery.recovery_banner import (
    RecoveryBanner,
    RecoveryMode,
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
