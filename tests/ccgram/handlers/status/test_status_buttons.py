"""Tests for status message inline action buttons (Esc, Screenshot, Last, Get File)."""

from unittest.mock import patch

import pytest

from ccgram.handlers.callback_data import (
    CB_STATUS_ESC,
    CB_STATUS_GET_FILE,
    CB_STATUS_LAST_REPLY,
    CB_STATUS_RECALL,
    CB_STATUS_SCREENSHOT,
)
from ccgram.handlers.callback_tokens import resolve_callback_data
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
        [
            CB_STATUS_ESC,
            CB_STATUS_SCREENSHOT,
            CB_STATUS_LAST_REPLY,
            CB_STATUS_GET_FILE,
        ],
    )
    def test_has_button_with_prefix(self, prefix: str) -> None:
        assert any(d.startswith(prefix) for d in _all_callback_data("@0"))

    def test_window_id_in_callback_data(self) -> None:
        data = _all_callback_data("@42")
        assert f"{CB_STATUS_ESC}@42" in data
        assert f"{CB_STATUS_SCREENSHOT}@42" in data
        assert f"{CB_STATUS_LAST_REPLY}@42" in data
        assert f"{CB_STATUS_GET_FILE}@42" in data

    def test_long_window_id_uses_lossless_callback_tokens(self) -> None:
        long_id = "@" + "x" * 60
        kb = build_status_keyboard(long_id)
        prefixes = (
            CB_STATUS_ESC,
            CB_STATUS_SCREENSHOT,
            CB_STATUS_LAST_REPLY,
            CB_STATUS_GET_FILE,
        )
        for row in kb.inline_keyboard:
            for btn in row:
                cb = btn.callback_data
                assert isinstance(cb, str)
                assert len(cb.encode("utf-8")) <= 64
                assert any(cb.startswith(p) for p in prefixes)
                assert resolve_callback_data(cb, 1, lambda _uid, wid: wid == long_id)

    @pytest.mark.parametrize("history", [None, []], ids=["none", "empty"])
    def test_history_absent_keeps_single_row(self, history: list[str] | None) -> None:
        assert len(build_status_keyboard("@0", history=history).inline_keyboard) == 1

    def test_no_history_argument_keeps_single_row(self) -> None:
        assert len(build_status_keyboard("@0").inline_keyboard) == 1

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

    def test_history_long_window_id_uses_lossless_callback_token(self) -> None:
        long_id = "@" + "x" * 60
        kb = build_status_keyboard(long_id, history=["cmd"])
        btn = kb.inline_keyboard[0][0]
        cb = btn.callback_data
        assert isinstance(cb, str)
        assert len(cb.encode("utf-8")) <= 64
        assert cb.startswith(CB_STATUS_RECALL)
        assert resolve_callback_data(cb, 1, lambda _uid, wid: wid == long_id) == (
            f"{CB_STATUS_RECALL}{long_id}:0"
        )

    @pytest.mark.parametrize(
        ("prefix", "label"),
        [
            (CB_STATUS_LAST_REPLY, "\U0001f4c4 Last"),
            (CB_STATUS_GET_FILE, "\U0001f4e5 Get File"),
        ],
    )
    def test_button_label(self, prefix: str, label: str) -> None:
        kb = build_status_keyboard("@0")
        texts = [
            b.text
            for row in kb.inline_keyboard
            for b in row
            if isinstance(b.callback_data, str) and b.callback_data.startswith(prefix)
        ]
        assert texts == [label]


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
