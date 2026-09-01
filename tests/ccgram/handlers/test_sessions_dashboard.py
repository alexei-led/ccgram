"""Tests for /sessions dashboard command."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import InlineKeyboardMarkup

from ccgram.handlers.callback_data import (
    CB_SESSIONS_NEW,
    CB_SESSIONS_REFRESH,
    CB_STATUS_ESC,
    CB_STATUS_SCREENSHOT,
)
from ccgram.handlers.callback_tokens import resolve_callback_data
from ccgram.handlers.sessions_dashboard import (
    _build_dashboard,
    handle_sessions_refresh,
    sessions_command,
)
from ccgram.session import WindowState

_CB_KILL = "sess:kill:"


@pytest.fixture(autouse=True)
def deps() -> Iterator[SimpleNamespace]:
    with (
        patch("ccgram.handlers.sessions_dashboard.view_window") as view,
        patch("ccgram.handlers.sessions_dashboard.thread_router") as router,
        patch("ccgram.handlers.sessions_dashboard.tmux_manager") as mux,
        patch(
            "ccgram.handlers.sessions_dashboard.list_windows_for_reconciliation",
            new_callable=AsyncMock,
        ) as listing,
        patch("ccgram.handlers.sessions_dashboard.config") as config,
    ):
        router.get_all_thread_windows.return_value = {}
        router.get_display_name.side_effect = lambda wid: wid
        view.side_effect = lambda wid: WindowState()
        listing.return_value = []
        config.is_user_allowed.return_value = True

        def _sessions(
            *,
            windows: dict[int, str],
            alive: list[str] | None = None,
            names: dict[str, str] | None = None,
            state: WindowState | None = None,
            unavailable: bool = False,
        ) -> None:
            router.get_all_thread_windows.return_value = windows
            if names is not None:
                router.get_display_name.side_effect = lambda wid: names[wid]
            if state is not None:
                view.side_effect = lambda wid: state
            if unavailable:
                listing.return_value = None
                return
            live = [] if alive is None else alive
            listing.return_value = [MagicMock(window_id=wid) for wid in live]

        yield SimpleNamespace(
            view=view,
            router=router,
            mux=mux,
            listing=listing,
            config=config,
            sessions=_sessions,
        )


def _callback_data(keyboard: InlineKeyboardMarkup) -> list[str]:
    return [
        btn.callback_data
        for row in keyboard.inline_keyboard
        for btn in row
        if isinstance(btn.callback_data, str)
    ]


async def _one_alive_session(deps: SimpleNamespace, **kwargs) -> tuple:
    deps.sessions(windows={42: "@0"}, alive=["@0"], names={"@0": "myproject"}, **kwargs)
    return await _build_dashboard(100)


class TestBuildDashboard:
    async def test_no_sessions_offers_refresh_and_new_only(
        self, deps: SimpleNamespace
    ) -> None:
        text, keyboard = await _build_dashboard(100)

        assert "No active sessions" in text
        data = _callback_data(keyboard)
        assert CB_SESSIONS_REFRESH in data
        assert CB_SESSIONS_NEW in data
        assert not any(d.startswith(_CB_KILL) for d in data)

    async def test_alive_session_shows_name_and_cwd(
        self, deps: SimpleNamespace
    ) -> None:
        text, _kb = await _one_alive_session(
            deps, state=WindowState(cwd="/home/user/myproject")
        )

        assert "\U0001f7e2 myproject" in text
        assert "/home/user/myproject" in text

    async def test_session_without_cwd_shows_no_path_line(
        self, deps: SimpleNamespace
    ) -> None:
        text, _kb = await _one_alive_session(deps, state=WindowState(cwd=""))

        assert "    " not in text

    async def test_dead_session_marked_with_grey_dot(
        self, deps: SimpleNamespace
    ) -> None:
        deps.sessions(windows={42: "@0"}, alive=[], names={"@0": "oldproject"})

        text, _kb = await _build_dashboard(100)

        assert "⚫ oldproject" in text

    async def test_alive_and_dead_sessions_listed_together(
        self, deps: SimpleNamespace
    ) -> None:
        deps.sessions(
            windows={10: "@0", 20: "@5"},
            alive=["@0"],
            names={"@0": "alive", "@5": "dead"},
        )

        text, _kb = await _build_dashboard(100)

        assert "\U0001f7e2 alive" in text
        assert "⚫ dead" in text

    async def test_liveness_comes_from_the_complete_listing(
        self, deps: SimpleNamespace
    ) -> None:
        """Every id on this dashboard is already bound, so adoptability is
        beside the point: a live window the backend merely will not auto-adopt
        must still read as running."""
        deps.sessions(windows={10: "@0"}, alive=["@0"], names={"@0": "out-of-scope"})
        deps.mux.list_windows = AsyncMock(return_value=[])

        text, _kb = await _build_dashboard(100)

        assert "\U0001f7e2 out-of-scope" in text
        assert "⚫" not in text

    async def test_unreachable_multiplexer_is_not_reported_as_stopped(
        self, deps: SimpleNamespace
    ) -> None:
        """A backend that could not be asked is unknown, not stopped."""
        deps.sessions(windows={10: "@0"}, names={"@0": "proj"}, unavailable=True)

        text, _kb = await _build_dashboard(100)

        assert "⚪ proj" in text
        assert "⚫" not in text
        assert "\U0001f7e2" not in text

    @pytest.mark.parametrize(
        ("state", "expected_tag", "present"),
        [
            pytest.param(
                WindowState(cwd="/p", provider_name="codex"),
                "[codex]",
                True,
                id="non-default-provider-tagged",
            ),
            pytest.param(
                WindowState(cwd="/p", provider_name=""),
                "[",
                False,
                id="default-untagged",
            ),
            pytest.param(
                WindowState(cwd="/p", provider_name="codex", approval_mode="yolo"),
                "[YOLO]",
                True,
                id="yolo-mode-tagged",
            ),
        ],
    )
    async def test_session_tags(
        self,
        deps: SimpleNamespace,
        state: WindowState,
        expected_tag: str,
        present: bool,
    ) -> None:
        text, _kb = await _one_alive_session(deps, state=state)

        assert (expected_tag in text) is present

    async def test_alive_session_offers_all_actions(
        self, deps: SimpleNamespace
    ) -> None:
        _text, keyboard = await _one_alive_session(deps)

        data = _callback_data(keyboard)
        assert any(d.startswith(CB_STATUS_ESC) for d in data)
        assert any(d.startswith(CB_STATUS_SCREENSHOT) for d in data)
        assert any(d.startswith(_CB_KILL) for d in data)
        assert any(
            "Refresh" in btn.text for row in keyboard.inline_keyboard for btn in row
        )
        assert any("New" in btn.text for row in keyboard.inline_keyboard for btn in row)

    async def test_dead_session_offers_no_actions(self, deps: SimpleNamespace) -> None:
        deps.sessions(windows={42: "@0"}, alive=[], names={"@0": "deadproject"})

        _text, keyboard = await _build_dashboard(100)

        data = _callback_data(keyboard)
        assert not any(d.startswith(CB_STATUS_ESC) for d in data)
        assert not any(d.startswith(CB_STATUS_SCREENSHOT) for d in data)
        assert not any(d.startswith(_CB_KILL) for d in data)

    async def test_long_window_id_kill_button_uses_a_lossless_token(
        self, deps: SimpleNamespace
    ) -> None:
        """A herdr window id blows the 64-byte callback budget, so the kill
        button carries a token that must resolve back to the full id."""
        window_id = "herdr-session-v1-" + "a" * 64
        deps.sessions(windows={42: window_id}, alive=[window_id])

        _text, keyboard = await _build_dashboard(100)

        callbacks = [d for d in _callback_data(keyboard) if d.startswith(_CB_KILL)]
        assert len(callbacks) == 1
        assert len(callbacks[0].encode("utf-8")) <= 64
        assert (
            resolve_callback_data(callbacks[0], 100, lambda _uid, wid: wid == window_id)
            == f"{_CB_KILL}{window_id}"
        )


class TestSessionsCommand:
    @staticmethod
    def _update(*, user_id: int | None = 100, with_message: bool = True) -> MagicMock:
        update = MagicMock()
        update.effective_user = None if user_id is None else MagicMock(id=user_id)
        update.message = AsyncMock() if with_message else None
        return update

    async def test_replies_with_the_dashboard(self, deps: SimpleNamespace) -> None:
        update = self._update()

        with patch("ccgram.handlers.sessions_dashboard.safe_reply") as mock_reply:
            await sessions_command(update, MagicMock())

        mock_reply.assert_called_once()
        assert mock_reply.call_args[0][0] is update.message
        assert "No active sessions" in mock_reply.call_args[0][1]

    async def test_unauthorized_user_is_told_so(self, deps: SimpleNamespace) -> None:
        deps.config.is_user_allowed.return_value = False

        with patch("ccgram.handlers.sessions_dashboard.safe_reply") as mock_reply:
            await sessions_command(self._update(), MagicMock())

        assert "not authorized" in mock_reply.call_args[0][1]

    @pytest.mark.parametrize(
        "update_kwargs",
        [
            pytest.param({"user_id": None}, id="no-user"),
            pytest.param({"with_message": False}, id="no-message"),
        ],
    )
    async def test_incomplete_update_is_ignored(self, update_kwargs: dict) -> None:
        with patch("ccgram.handlers.sessions_dashboard.safe_reply") as mock_reply:
            await sessions_command(self._update(**update_kwargs), MagicMock())

        mock_reply.assert_not_called()


class TestSessionsRefresh:
    async def test_refresh_edits_the_dashboard_in_place(
        self, deps: SimpleNamespace
    ) -> None:
        query = AsyncMock()

        with patch("ccgram.handlers.sessions_dashboard.safe_edit") as mock_edit:
            await handle_sessions_refresh(query, 100)

        mock_edit.assert_called_once()
        assert mock_edit.call_args[0][0] is query
        assert "No active sessions" in mock_edit.call_args[0][1]


class TestDashboardIdentityFoldsCase:
    @pytest.mark.parametrize(
        ("bound", "listed"),
        [
            pytest.param("9f1c2d3e-4a5b", "9F1C2D3E-4A5B", id="bound-lower"),
            pytest.param("9F1C2D3E-4A5B", "9f1c2d3e-4a5b", id="bound-upper"),
        ],
    )
    async def test_a_case_variant_binding_reads_as_running(
        self, deps: SimpleNamespace, bound: str, listed: str
    ) -> None:
        """Raw set membership marked the session stopped and stripped its
        Esc, Screenshot and Kill controls.

        Both orderings, because the set and the lookup fold independently and
        one of them alone leaves the other half untested.
        """
        deps.sessions(windows={10: bound}, alive=[listed], names={bound: "proj"})

        text, _kb = await _build_dashboard(100)

        assert "\U0001f7e2 proj" in text
        assert "⚫" not in text
