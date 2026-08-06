"""Tests for callback handler authorization checks."""

from unittest.mock import patch

from ccgram.handlers.callback_helpers import user_owns_window


class TestUserOwnsWindow:
    def test_owns_bound_window(self) -> None:
        with patch("ccgram.handlers.callback_helpers.thread_router") as mock_sm:
            mock_sm.get_all_thread_windows.return_value = {42: "@0", 99: "@5"}
            assert user_owns_window(100, "@0")
            assert user_owns_window(100, "@5")

    def test_does_not_own_unbound_window(self) -> None:
        with patch("ccgram.handlers.callback_helpers.thread_router") as mock_sm:
            mock_sm.get_all_thread_windows.return_value = {42: "@0"}
            assert not user_owns_window(100, "@99")

    def test_no_bindings(self) -> None:
        with patch("ccgram.handlers.callback_helpers.thread_router") as mock_sm:
            mock_sm.get_all_thread_windows.return_value = {}
            assert not user_owns_window(100, "@0")

    def test_chat_scoped_binding_authorizes_only_its_chat(self) -> None:
        with patch("ccgram.handlers.callback_helpers.thread_router") as mock_router:
            mock_router.iter_thread_bindings_with_chat.return_value = [
                (100, -1001, 42, "@0")
            ]

            assert user_owns_window(100, "@0", -1001)
            assert not user_owns_window(100, "@0", -1002)

    def test_different_user_does_not_own(self) -> None:
        with patch("ccgram.handlers.callback_helpers.thread_router") as mock_sm:
            mock_sm.get_all_thread_windows.side_effect = lambda uid: (
                {42: "@0", 99: "@5"} if uid == 100 else {}
            )
            assert user_owns_window(100, "@0")
            assert not user_owns_window(200, "@0")
