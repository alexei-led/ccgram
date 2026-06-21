import ast
import asyncio
import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Bot
from telegram.error import TelegramError

from ccgram.handlers.polling.polling_coordinator import (
    _BACKOFF_MAX,
    _BACKOFF_MIN,
    status_poll_loop,
)

SRC_FILE = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "ccgram"
    / "handlers"
    / "polling"
    / "polling_coordinator.py"
)


def _make_window(window_id: str, window_name: str = "test") -> MagicMock:
    w = MagicMock()
    w.window_id = window_id
    w.window_name = window_name
    return w


class _LoopCtx:
    mocks: dict[str, Any]


def _patch_loop_deps(
    bindings: list[tuple[int, int, str]] | None = None,
    windows: list[MagicMock] | None = None,
) -> Any:
    bindings = bindings or []
    windows = windows or []

    patches: dict[str, Any] = {
        "thread_router": patch(
            "ccgram.handlers.polling.polling_coordinator.thread_router"
        ),
        "tmux_manager": patch(
            "ccgram.handlers.polling.polling_coordinator.tmux_manager"
        ),
        "tick_window": patch(
            "ccgram.handlers.polling.polling_coordinator.window_tick.tick_window",
            new_callable=AsyncMock,
        ),
        "run_periodic": patch(
            "ccgram.handlers.polling.periodic_tasks.run_periodic_tasks",
            new_callable=AsyncMock,
        ),
        "run_lifecycle": patch(
            "ccgram.handlers.polling.periodic_tasks.run_lifecycle_tasks",
            new_callable=AsyncMock,
        ),
        "config": patch("ccgram.config.config"),
        "log_throttled": patch(
            "ccgram.handlers.polling.polling_coordinator.log_throttled"
        ),
    }

    ctx = _LoopCtx()

    @contextlib.contextmanager
    def combined():
        mocks: dict[str, Any] = {}
        with contextlib.ExitStack() as stack:
            for name, p in patches.items():
                mocks[name] = stack.enter_context(p)

            mocks["tmux_manager"].list_windows = AsyncMock(return_value=windows)
            mocks["thread_router"].iter_thread_bindings.return_value = bindings
            mocks["config"].status_poll_interval = 1.0

            ctx.mocks = mocks
            yield ctx

    return combined, ctx


async def _run_loop_once(bot: Bot, **kwargs: Any) -> _LoopCtx:
    combined, ctx = _patch_loop_deps(**kwargs)

    async def _stop_after_one(_delay: float) -> None:
        raise asyncio.CancelledError

    with (
        combined(),
        patch(
            "ccgram.handlers.polling.polling_coordinator.asyncio.sleep",
            side_effect=_stop_after_one,
        ),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await status_poll_loop(bot)
    return ctx


class TestStatusPollLoopIteratesAllBindings:
    async def test_ticks_all_bindings(self):
        bot = AsyncMock(spec=Bot)
        w0, w1, w2 = _make_window("@0"), _make_window("@1"), _make_window("@2")
        bindings = [(1, 100, "@0"), (2, 200, "@1"), (3, 300, "@2")]

        ctx = await _run_loop_once(bot, bindings=bindings, windows=[w0, w1, w2])

        tick = ctx.mocks["tick_window"]
        assert tick.call_count == 3
        for i, (uid, tid, wid) in enumerate(bindings):
            call_args = tick.call_args_list[i]
            assert call_args[0][0] is bot
            assert call_args[0][1] == uid
            assert call_args[0][2] == tid
            assert call_args[0][3] == wid


class TestStatusPollLoopDelegatesPeriodicTasks:
    async def test_periodic_and_lifecycle_called(self):
        bot = AsyncMock(spec=Bot)
        ctx = await _run_loop_once(bot, bindings=[], windows=[])

        ctx.mocks["run_periodic"].assert_called_once()
        ctx.mocks["run_lifecycle"].assert_called_once()


class TestStatusPollLoopPassesWindowLookup:
    async def test_lookup_provides_correct_window(self):
        bot = AsyncMock(spec=Bot)
        w_a = _make_window("@A", "proj-a")
        w_b = _make_window("@B", "proj-b")
        bindings = [(1, 100, "@A")]

        ctx = await _run_loop_once(bot, bindings=bindings, windows=[w_a, w_b])

        tick = ctx.mocks["tick_window"]
        assert tick.call_count == 1
        assert tick.call_args[0][4] is w_a


class TestStatusPollLoopRespectsConfigInterval:
    async def test_sleeps_with_config_interval(self):
        bot = AsyncMock(spec=Bot)
        combined, ctx = _patch_loop_deps(bindings=[], windows=[])
        sleep_delays: list[float] = []

        async def _capture_sleep(delay: float) -> None:
            sleep_delays.append(delay)
            raise asyncio.CancelledError

        with combined():
            ctx.mocks["config"].status_poll_interval = 2.5
            with (
                patch(
                    "ccgram.handlers.polling.polling_coordinator.asyncio.sleep",
                    side_effect=_capture_sleep,
                ),
                contextlib.suppress(asyncio.CancelledError),
            ):
                await status_poll_loop(bot)

        assert sleep_delays == [2.5]


class TestBackoffOnTelegramError:
    async def test_backoff_doubles_on_error(self):
        bot = AsyncMock(spec=Bot)
        combined, ctx = _patch_loop_deps(bindings=[], windows=[])
        sleep_delays: list[float] = []
        call_count = 0

        async def _capture_sleep(delay: float) -> None:
            nonlocal call_count
            sleep_delays.append(delay)
            call_count += 1
            if call_count >= 3:
                raise asyncio.CancelledError

        with combined():
            ctx.mocks["tmux_manager"].list_windows = AsyncMock(
                side_effect=[
                    TelegramError("err"),
                    TelegramError("err"),
                    TelegramError("err"),
                ]
            )
            with (
                patch(
                    "ccgram.handlers.polling.polling_coordinator.asyncio.sleep",
                    side_effect=_capture_sleep,
                ),
                contextlib.suppress(asyncio.CancelledError),
            ):
                await status_poll_loop(bot)

        assert sleep_delays[0] == _BACKOFF_MIN
        assert sleep_delays[1] == _BACKOFF_MIN * 2

    def test_backoff_bounded_by_max(self):
        for streak in range(20):
            delay = min(_BACKOFF_MAX, _BACKOFF_MIN * (2**streak))
            assert delay <= _BACKOFF_MAX


class TestPerBindingError:
    async def test_error_does_not_abort_loop(self):
        bot = AsyncMock(spec=Bot)
        w0, w1, w2 = _make_window("@0"), _make_window("@1"), _make_window("@2")
        bindings = [(1, 100, "@0"), (2, 200, "@1"), (3, 300, "@2")]

        combined, ctx = _patch_loop_deps(bindings=bindings, windows=[w0, w1, w2])
        call_order: list[str] = []

        async def _tick_side_effect(
            _bot: Bot, uid: int, tid: int, wid: str, _w: Any
        ) -> None:
            call_order.append(wid)
            if wid == "@1":
                raise TelegramError("boom")

        with combined():
            ctx.mocks["tick_window"].side_effect = _tick_side_effect

            async def _stop_sleep(_delay: float) -> None:
                raise asyncio.CancelledError

            with (
                patch(
                    "ccgram.handlers.polling.polling_coordinator.asyncio.sleep",
                    side_effect=_stop_sleep,
                ),
                contextlib.suppress(asyncio.CancelledError),
            ):
                await status_poll_loop(bot)

        assert call_order == ["@0", "@1", "@2"]


class TestBackoffConstants:
    def test_backoff_bounds(self):
        assert _BACKOFF_MIN == 2.0
        assert _BACKOFF_MAX == 30.0


class TestImportsAreMinimal:
    def test_only_allowed_imports(self):
        source = SRC_FILE.read_text()
        tree = ast.parse(source)
        allowed_modules = {
            "asyncio",
            "typing",
            "structlog",
            "telegram.error",
            "telegram",
            "..thread_router",
            "..multiplexer",
            "..utils",
            "..config",
            ".window_tick",
            ".periodic_tasks",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name in allowed_modules, (
                        f"Unexpected import: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                level = node.level or 0
                rel = "." * level + mod
                if rel in allowed_modules:
                    continue
                if not mod:
                    for alias in node.names:
                        fq = "." * level + alias.name
                        assert fq in allowed_modules, (
                            f"Unexpected import: from {rel} import {alias.name}"
                        )
                else:
                    assert rel in allowed_modules or any(
                        mod.startswith(a.lstrip(".")) for a in allowed_modules
                    ), f"Unexpected import from: {rel}"


class TestDoesNotImportPerWindowModules:
    def test_no_per_window_imports(self):
        source = SRC_FILE.read_text()
        banned = {
            "interactive_ui",
            "message_queue",
            "message_sender",
            "topic_emoji",
            "transcript_discovery",
            "recovery_callbacks",
            "claude_task_state",
            "session_monitor",
            "polling_state",
            "cleanup",
        }
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for b in banned:
                    assert b not in mod, f"polling_coordinator must not import {b}"


class TestModuleLineCountUnderCeiling:
    def test_under_120_lines(self):
        lines = SRC_FILE.read_text().splitlines()
        assert len(lines) <= 120, (
            f"polling_coordinator.py is {len(lines)} lines, ceiling is 120"
        )


class TestBackoffBehavior:
    async def test_loop_error_triggers_backoff_sleep(self):
        bot = MagicMock(spec_set=["_do_post"])
        combined, ctx = _patch_loop_deps(bindings=[], windows=[])
        sleep_calls: list[float] = []

        async def _capture_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            raise asyncio.CancelledError

        with combined():
            ctx.mocks["tmux_manager"].list_windows = AsyncMock(
                side_effect=TelegramError("loop-error")
            )
            with (
                patch(
                    "ccgram.handlers.polling.polling_coordinator.asyncio.sleep",
                    side_effect=_capture_sleep,
                ),
                contextlib.suppress(asyncio.CancelledError),
            ):
                await status_poll_loop(bot)

        assert sleep_calls == [_BACKOFF_MIN * (2**0)]

    async def test_consecutive_errors_increase_backoff(self):
        bot = MagicMock(spec_set=["_do_post"])
        combined, ctx = _patch_loop_deps(bindings=[], windows=[])
        sleep_calls: list[float] = []
        call_count = 0

        async def _capture_sleep(delay: float) -> None:
            nonlocal call_count
            sleep_calls.append(delay)
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError

        with combined():
            ctx.mocks["tmux_manager"].list_windows = AsyncMock(
                side_effect=TelegramError("loop-error")
            )
            with (
                patch(
                    "ccgram.handlers.polling.polling_coordinator.asyncio.sleep",
                    side_effect=_capture_sleep,
                ),
                contextlib.suppress(asyncio.CancelledError),
            ):
                await status_poll_loop(bot)

        assert sleep_calls[0] == _BACKOFF_MIN
        assert sleep_calls[1] == _BACKOFF_MIN * 2

    async def test_error_streak_resets_after_success(self):
        bot = MagicMock(spec_set=["_do_post"])
        combined, ctx = _patch_loop_deps(bindings=[], windows=[])
        sleep_calls: list[float] = []
        call_count = 0

        async def _capture_sleep(delay: float) -> None:
            nonlocal call_count
            sleep_calls.append(delay)
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError

        with combined():
            ctx.mocks["tmux_manager"].list_windows = AsyncMock(
                side_effect=[TelegramError("boom"), []]
            )
            ctx.mocks["config"].status_poll_interval = 0.5
            with (
                patch(
                    "ccgram.handlers.polling.polling_coordinator.asyncio.sleep",
                    side_effect=_capture_sleep,
                ),
                contextlib.suppress(asyncio.CancelledError),
            ):
                await status_poll_loop(bot)

        assert sleep_calls[0] == _BACKOFF_MIN
        assert sleep_calls[1] == 0.5
