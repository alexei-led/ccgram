"""Tests for status message inline action buttons (Esc, Screenshot, RC)."""

from unittest.mock import patch

import pytest

from ccgram.handlers.callback_data import (
    CB_STATUS_ESC,
    CB_STATUS_RECALL,
    CB_STATUS_REMOTE,
    CB_STATUS_SCREENSHOT,
)
from ccgram.handlers.status.status_bubble import build_status_keyboard


def _all_callback_data(window_id: str) -> list[str]:
    kb = build_status_keyboard(window_id)
    return [
        btn.callback_data
        for row in kb.inline_keyboard
        for btn in row
        if isinstance(btn.callback_data, str)
    ]


class TestBuildStatusKeyboard:
    @pytest.mark.parametrize(
        "prefix",
        [CB_STATUS_ESC, CB_STATUS_SCREENSHOT, CB_STATUS_REMOTE],
    )
    def test_has_button_with_prefix(self, prefix: str) -> None:
        assert any(d.startswith(prefix) for d in _all_callback_data("@0"))

    def test_window_id_in_callback_data(self) -> None:
        data = _all_callback_data("@42")
        assert f"{CB_STATUS_ESC}@42" in data
        assert f"{CB_STATUS_SCREENSHOT}@42" in data
        assert f"{CB_STATUS_REMOTE}@42" in data

    def test_callback_data_truncated_to_64_bytes(self) -> None:
        long_id = "@" + "x" * 60
        kb = build_status_keyboard(long_id)
        prefixes = (
            CB_STATUS_ESC,
            CB_STATUS_SCREENSHOT,
            CB_STATUS_REMOTE,
        )
        for row in kb.inline_keyboard:
            for btn in row:
                cb = btn.callback_data
                assert isinstance(cb, str)
                assert len(cb) == 64
                assert any(cb.startswith(p) for p in prefixes)

    def test_no_history_single_row(self) -> None:
        kb = build_status_keyboard("@0")
        assert len(kb.inline_keyboard) == 1

    def test_history_adds_row(self) -> None:
        kb = build_status_keyboard("@0", history=["hello", "world"])
        assert len(kb.inline_keyboard) == 2
        assert kb.inline_keyboard[0][0].callback_data == f"{CB_STATUS_RECALL}@0:0"
        assert kb.inline_keyboard[0][1].callback_data == f"{CB_STATUS_RECALL}@0:1"

    def test_history_label_truncated(self) -> None:
        long_cmd = "a" * 30
        kb = build_status_keyboard("@0", history=[long_cmd])
        label = kb.inline_keyboard[0][0].text
        assert label.startswith("↑ ")
        assert label.endswith("…")
        assert len(label) <= 2 + 20 + 1

    def test_history_none_no_extra_row(self) -> None:
        kb = build_status_keyboard("@0", history=None)
        assert len(kb.inline_keyboard) == 1

    def test_history_empty_list_no_extra_row(self) -> None:
        kb = build_status_keyboard("@0", history=[])
        assert len(kb.inline_keyboard) == 1

    def test_history_callback_data_truncated_to_64_bytes(self) -> None:
        long_id = "@" + "x" * 60
        kb = build_status_keyboard(long_id, history=["cmd"])
        btn = kb.inline_keyboard[0][0]
        cb = btn.callback_data
        assert isinstance(cb, str)
        assert len(cb) == 64  # type: ignore[arg-type]
        assert cb.startswith(CB_STATUS_RECALL)  # type: ignore[union-attr]

    def test_rc_button_always_present(self) -> None:
        data = _all_callback_data("@0")
        assert any(d.startswith(CB_STATUS_REMOTE) for d in data)

    def test_rc_button_label_inactive(self) -> None:
        kb = build_status_keyboard("@0")
        rc_btn = [
            btn
            for row in kb.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
            and btn.callback_data.startswith(CB_STATUS_REMOTE)  # type: ignore[union-attr]
        ][0]
        assert rc_btn.text == "\U0001f4e1"

    def test_rc_button_label_active(self) -> None:
        kb = build_status_keyboard("@0", rc_active=True)
        rc_btn = [
            btn
            for row in kb.inline_keyboard
            for btn in row
            if isinstance(btn.callback_data, str)
            and btn.callback_data.startswith(CB_STATUS_REMOTE)  # type: ignore[union-attr]
        ][0]
        assert rc_btn.text == "\U0001f4e1✓"


class TestDashboardButtonRow:
    """Dashboard WebApp button is appended only when Mini App is enabled."""

    def test_no_dashboard_when_user_id_omitted(self) -> None:
        with patch("ccgram.handlers.status.status_bar_actions.config") as cfg:
            cfg.miniapp_base_url = "https://example.com"
            cfg.telegram_bot_token = "bot:abc"
            kb = build_status_keyboard("@0")
        for row in kb.inline_keyboard:
            for btn in row:
                assert btn.web_app is None

    def test_no_dashboard_when_miniapp_disabled(self) -> None:
        with patch("ccgram.handlers.status.status_bar_actions.config") as cfg:
            cfg.miniapp_base_url = ""
            cfg.telegram_bot_token = "bot:abc"
            kb = build_status_keyboard("@0", user_id=42)
        for row in kb.inline_keyboard:
            for btn in row:
                assert btn.web_app is None

    def test_dashboard_appended_when_enabled(self) -> None:
        with (
            patch("ccgram.handlers.status.status_bar_actions.config") as cfg,
            patch(
                "ccgram.handlers.status.status_bar_actions.sign_token",
                return_value="abc.def",
            ),
        ):
            cfg.miniapp_base_url = "https://example.com"
            cfg.telegram_bot_token = "bot:abc"
            kb = build_status_keyboard("@7", user_id=42)
        last_row = kb.inline_keyboard[-1]
        assert len(last_row) == 1
        btn = last_row[0]
        assert btn.text == "\U0001fa9f Dashboard"
        assert btn.web_app is not None
        assert btn.web_app.url == "https://example.com/app/abc.def"

    def test_dashboard_url_signed_with_window_and_user(self) -> None:
        captured: list[tuple[str, int]] = []

        def fake_sign(*, bot_token: str, window_id: str, user_id: int) -> str:
            captured.append((window_id, user_id))
            assert bot_token == "bot:abc"
            return "tok"

        with (
            patch("ccgram.handlers.status.status_bar_actions.config") as cfg,
            patch(
                "ccgram.handlers.status.status_bar_actions.sign_token",
                side_effect=fake_sign,
            ),
        ):
            cfg.miniapp_base_url = "https://example.com/"
            cfg.telegram_bot_token = "bot:abc"
            build_status_keyboard("@9", user_id=99)
        assert captured == [("@9", 99)]

    def test_history_row_does_not_replace_dashboard(self) -> None:
        with (
            patch("ccgram.handlers.status.status_bar_actions.config") as cfg,
            patch(
                "ccgram.handlers.status.status_bar_actions.sign_token",
                return_value="tok",
            ),
        ):
            cfg.miniapp_base_url = "https://example.com"
            cfg.telegram_bot_token = "bot:abc"
            kb = build_status_keyboard("@0", history=["a", "b"], user_id=42)
        assert len(kb.inline_keyboard) == 3
        assert kb.inline_keyboard[-1][0].web_app is not None
