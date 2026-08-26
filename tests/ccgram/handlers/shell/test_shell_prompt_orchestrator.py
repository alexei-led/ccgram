from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest

from ccgram.handlers.shell.shell_prompt_orchestrator import (
    CB_SHELL_SETUP,
    CB_SHELL_SKIP,
    Trigger,
    _OrchestratorState,
    _dispatch,
    _show_offer_keyboard,
    _state,
    accept_offer,
    clear_state,
    ensure_setup,
    record_skip,
)

_MOD = "ccgram.handlers.shell.shell_prompt_orchestrator"
WINDOW = "@99"


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    _state.clear()
    yield
    _state.clear()


@pytest.fixture
def mock_setup() -> Iterator[AsyncMock]:
    with patch(
        "ccgram.providers.shell_infra.setup_shell_prompt",
        new_callable=AsyncMock,
    ) as m:
        yield m


@pytest.fixture
def mock_has_marker() -> Iterator[AsyncMock]:
    with patch(
        "ccgram.providers.shell_infra.has_prompt_marker",
        new_callable=AsyncMock,
    ) as m:
        yield m


def _callback_update(data: str) -> tuple[AsyncMock, AsyncMock]:
    query = AsyncMock()
    query.data = data
    query.from_user.id = 1
    update = AsyncMock()
    update.callback_query = query
    return update, query


class TestEnsureSetup:
    @pytest.mark.parametrize(
        ("trigger", "marker_present", "skip_flag", "expected_clear"),
        [
            pytest.param("auto", False, False, True, id="auto-always-runs"),
            pytest.param("auto", True, False, True, id="auto-runs-even-with-marker"),
            pytest.param("lazy", False, False, False, id="lazy-runs-without-marker"),
            pytest.param(
                "external_bind", False, False, False, id="external-bind-offers-setup"
            ),
            pytest.param(
                "provider_switch",
                False,
                False,
                False,
                id="provider-switch-offers-setup",
            ),
        ],
    )
    async def test_runs_setup(
        self,
        mock_setup: AsyncMock,
        mock_has_marker: AsyncMock,
        trigger: Trigger,
        marker_present: bool,
        skip_flag: bool,
        expected_clear: bool,
    ) -> None:
        mock_has_marker.return_value = marker_present
        if skip_flag:
            record_skip(WINDOW)

        await ensure_setup(WINDOW, trigger)

        mock_setup.assert_awaited_once()
        assert mock_setup.await_args_list[-1].kwargs["clear"] is expected_clear

    @pytest.mark.parametrize(
        ("trigger", "marker_present", "skip_flag"),
        [
            pytest.param("lazy", True, False, id="lazy-marker-already-present"),
            pytest.param("lazy", False, True, id="lazy-user-skipped"),
            pytest.param(
                "external_bind", True, False, id="external-bind-marker-present"
            ),
            pytest.param("provider_switch", False, True, id="provider-switch-skipped"),
        ],
    )
    async def test_skips_setup(
        self,
        mock_setup: AsyncMock,
        mock_has_marker: AsyncMock,
        trigger: Trigger,
        marker_present: bool,
        skip_flag: bool,
    ) -> None:
        mock_has_marker.return_value = marker_present
        if skip_flag:
            record_skip(WINDOW)

        await ensure_setup(WINDOW, trigger)

        mock_setup.assert_not_awaited()

    async def test_external_bind_sends_offer_keyboard_when_client_present(
        self, mock_setup: AsyncMock, mock_has_marker: AsyncMock
    ) -> None:
        mock_has_marker.return_value = False

        with patch(
            f"{_MOD}.safe_send", new_callable=AsyncMock, return_value=AsyncMock()
        ) as mock_send:
            await ensure_setup(
                WINDOW,
                "external_bind",
                client=AsyncMock(),
                chat_id=-100,
                thread_id=5,
            )

        mock_send.assert_awaited_once()
        assert _state[WINDOW].was_offered is True
        mock_setup.assert_not_awaited()

    async def test_external_bind_does_not_reoffer(
        self, mock_setup: AsyncMock, mock_has_marker: AsyncMock
    ) -> None:
        mock_has_marker.return_value = False
        _state[WINDOW] = _OrchestratorState(was_offered=True)

        with patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send:
            await ensure_setup(
                WINDOW,
                "external_bind",
                client=AsyncMock(),
                chat_id=-100,
                thread_id=5,
            )

        mock_send.assert_not_awaited()
        mock_setup.assert_not_awaited()


class TestOfferKeyboard:
    async def test_sends_message_into_the_topic(
        self, mock_setup: AsyncMock, mock_has_marker: AsyncMock
    ) -> None:
        with patch(f"{_MOD}.safe_send", new_callable=AsyncMock) as mock_send:
            await _show_offer_keyboard(
                "@3", client=AsyncMock(), chat_id=-100, thread_id=42
            )

        assert mock_send.call_args[1]["message_thread_id"] == 42
        assert _state["@3"].was_offered is True
        mock_setup.assert_not_awaited()

    async def test_without_a_client_falls_back_to_running_setup(
        self, mock_setup: AsyncMock, mock_has_marker: AsyncMock
    ) -> None:
        await _show_offer_keyboard("@3")

        mock_setup.assert_awaited_once_with("@3", clear=False)
        assert _state["@3"].was_offered is True


class TestCallbackButtons:
    async def test_setup_button_runs_setup(
        self, mock_setup: AsyncMock, mock_has_marker: AsyncMock
    ) -> None:
        update, query = _callback_update(f"{CB_SHELL_SETUP}@5")

        with patch(
            "ccgram.handlers.callback_helpers.user_owns_window", return_value=True
        ):
            await _dispatch(update, AsyncMock())

        query.answer.assert_awaited_once()
        mock_setup.assert_awaited_once_with("@5", clear=False)
        assert _state["@5"].was_offered is True
        query.edit_message_text.assert_awaited_once()

    async def test_skip_button_records_skip(self) -> None:
        update, query = _callback_update(f"{CB_SHELL_SKIP}@5")

        with patch(
            "ccgram.handlers.callback_helpers.user_owns_window", return_value=True
        ):
            await _dispatch(update, AsyncMock())

        query.answer.assert_awaited_once()
        assert _state["@5"].skip_flag is True
        query.edit_message_text.assert_awaited_once()


class TestStateBookkeeping:
    async def test_accept_offer_runs_setup(
        self, mock_setup: AsyncMock, mock_has_marker: AsyncMock
    ) -> None:
        await accept_offer(WINDOW)

        mock_setup.assert_awaited_once_with(WINDOW, clear=False)
        assert _state[WINDOW].was_offered is True

    def test_record_skip_sets_flag(self) -> None:
        record_skip(WINDOW)
        assert _state[WINDOW].skip_flag is True

    def test_clear_state_removes_entry(self) -> None:
        record_skip(WINDOW)
        clear_state(WINDOW)
        assert WINDOW not in _state

    def test_clear_state_no_op_for_unknown_window(self) -> None:
        record_skip(WINDOW)
        clear_state("@unknown")
        assert set(_state) == {WINDOW}
