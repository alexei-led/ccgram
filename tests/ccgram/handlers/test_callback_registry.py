"""Tests for callback registry — registration, dispatch, and handler loading."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_registry import (
    _find_handler,
    _registry,
    dispatch,
    load_handlers,
    register,
)


@pytest.fixture
def clean_registry():
    """Clear the registry before and after each test."""
    saved = dict(_registry)
    _registry.clear()
    yield
    _registry.clear()
    _registry.update(saved)


class TestRegister:
    @pytest.fixture(autouse=True)
    def _setup(self, clean_registry):
        pass

    def test_register_single_prefix(self) -> None:
        @register("pfx:")
        async def handler(update, context):
            pass

        assert "pfx:" in _registry
        assert _registry["pfx:"] is handler

    def test_register_multiple_prefixes(self) -> None:
        @register("a:", "b:", "c:")
        async def handler(update, context):
            pass

        assert _registry["a:"] is handler
        assert _registry["b:"] is handler
        assert _registry["c:"] is handler

    def test_register_returns_original_function(self) -> None:
        async def handler(update, context):
            pass

        decorated = register("pfx:")(handler)
        assert decorated is handler

    def test_register_duplicate_prefix_raises(self) -> None:
        @register("dup:")
        async def first(update, context):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register("dup:")
            async def second(update, context):
                pass


class TestFindHandler:
    @pytest.fixture(autouse=True)
    def _setup(self, clean_registry):
        pass

    def test_matches_prefix(self) -> None:
        handler = AsyncMock()
        _registry["db:sel:"] = handler

        assert _find_handler("db:sel:some-data") is handler

    def test_longest_prefix_wins(self) -> None:
        short = AsyncMock()
        long = AsyncMock()
        _registry["st:"] = short
        _registry["st:esc:"] = long

        assert _find_handler("st:esc:@0") is long

    def test_no_match_returns_none(self) -> None:
        _registry["known:"] = AsyncMock()

        assert _find_handler("unknown:data") is None

    def test_exact_match(self) -> None:
        handler = AsyncMock()
        _registry["exact"] = handler

        assert _find_handler("exact") is handler


class TestDispatch:
    @pytest.fixture(autouse=True)
    def _setup(self, clean_registry):
        pass

    def _make_update(
        self,
        data: str,
        user_id: int = 42,
        chat_type: str = "supergroup",
        chat_id: int = -100,
        thread_id: int = 99,
    ) -> MagicMock:
        update = MagicMock()
        update.callback_query.data = data
        update.effective_user.id = user_id
        update.effective_chat.id = chat_id
        update.callback_query.message.chat.type = chat_type
        update.callback_query.message.chat.id = chat_id
        update.callback_query.message.message_thread_id = thread_id
        update.callback_query.answer = AsyncMock()
        return update

    @pytest.mark.asyncio
    async def test_dispatch_calls_handler(self) -> None:
        handler = AsyncMock()
        _registry["test:"] = handler
        update = self._make_update("test:data")
        context = MagicMock()

        with (
            patch("ccgram.handlers.callback_registry.config") as mock_config,
            patch("ccgram.handlers.callback_registry.thread_router"),
            patch("ccgram.handlers.callback_registry.get_thread_id"),
        ):
            mock_config.group_id = None
            mock_config.is_user_allowed.return_value = True

            await dispatch(update, context)

        handler.assert_awaited_once_with(update, context)

    @pytest.mark.asyncio
    async def test_dispatch_unauthorized_user(self) -> None:
        handler = AsyncMock()
        _registry["test:"] = handler
        update = self._make_update("test:data")
        context = MagicMock()

        with patch("ccgram.handlers.callback_registry.config") as mock_config:
            mock_config.group_id = None
            mock_config.is_user_allowed.return_value = False

            await dispatch(update, context)

        handler.assert_not_awaited()
        update.callback_query.answer.assert_awaited_once_with("Not authorized")

    @pytest.mark.asyncio
    async def test_dispatch_wrong_group(self) -> None:
        handler = AsyncMock()
        _registry["test:"] = handler
        update = self._make_update("test:data", chat_id=-999)
        context = MagicMock()

        with patch("ccgram.handlers.callback_registry.config") as mock_config:
            mock_config.group_id = -100

            await dispatch(update, context)

        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_no_match(self) -> None:
        _registry["known:"] = AsyncMock()
        update = self._make_update("unknown:data")
        context = MagicMock()

        with (
            patch("ccgram.handlers.callback_registry.config") as mock_config,
            patch("ccgram.handlers.callback_registry.thread_router"),
            patch("ccgram.handlers.callback_registry.get_thread_id"),
        ):
            mock_config.group_id = None
            mock_config.is_user_allowed.return_value = True

            await dispatch(update, context)

        _registry["known:"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_noop(self) -> None:
        update = self._make_update("noop")
        context = MagicMock()

        with (
            patch("ccgram.handlers.callback_registry.config") as mock_config,
            patch("ccgram.handlers.callback_registry.thread_router"),
            patch("ccgram.handlers.callback_registry.get_thread_id"),
        ):
            mock_config.group_id = None
            mock_config.is_user_allowed.return_value = True

            await dispatch(update, context)

        update.callback_query.answer.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_dispatch_records_group_chat_id(self) -> None:
        handler = AsyncMock()
        _registry["test:"] = handler
        update = self._make_update(
            "test:data", chat_type="supergroup", chat_id=-100, thread_id=55
        )
        context = MagicMock()

        with (
            patch("ccgram.handlers.callback_registry.config") as mock_config,
            patch("ccgram.handlers.callback_registry.thread_router") as mock_tr,
            patch(
                "ccgram.handlers.callback_registry.get_thread_id",
                return_value=55,
            ),
        ):
            mock_config.group_id = None
            mock_config.is_user_allowed.return_value = True

            await dispatch(update, context)

        mock_tr.set_group_chat_id.assert_called_once_with(42, 55, -100)


class TestLoadHandlers:
    def test_load_handlers_populates_registry(self) -> None:
        load_handlers()
        prefixes = set(_registry.keys())
        assert len(prefixes) > 0
        for expected in ("db:", "wb:", "aq:", "vc:", "sh:", "hp:", "rec:"):
            assert any(p.startswith(expected) for p in prefixes), (
                f"expected prefix starting with {expected!r} in registry"
            )

    def test_load_handlers_imports_modules(self) -> None:
        load_handlers()
        expected_modules = [
            "ccgram.handlers.directory_callbacks",
            "ccgram.handlers.window_callbacks",
            "ccgram.handlers.history_callbacks",
            "ccgram.handlers.screenshot_callbacks",
            "ccgram.handlers.interactive_callbacks",
            "ccgram.handlers.recovery_callbacks",
            "ccgram.handlers.resume_command",
            "ccgram.handlers.voice_callbacks",
            "ccgram.handlers.shell_commands",
        ]
        for mod in expected_modules:
            assert mod in sys.modules, f"{mod} not imported by load_handlers()"
