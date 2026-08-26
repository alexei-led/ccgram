"""Tests for StatePersistence — debounced atomic JSON persistence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ccgram.state_persistence import StatePersistence


class TestScheduleSaveNoLoop:
    def test_saves_immediately_without_event_loop(self, tmp_path: Path) -> None:
        """No loop to debounce against — the write lands on the call itself."""
        path = tmp_path / "state.json"
        sp = StatePersistence(path, lambda: {"key": "value"})

        sp.schedule_save()

        assert json.loads(path.read_text()) == {"key": "value"}
        assert not sp._dirty
        assert sp._save_timer is None


class TestScheduleSaveWithLoop:
    async def test_timer_scheduled_when_loop_running(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        sp = StatePersistence(path, lambda: {"async": True})
        sp.schedule_save()
        assert sp._dirty
        assert sp._save_timer is not None
        sp._save_timer.cancel()
        sp._save_timer = None

    async def test_debounce_resets_timer_on_second_call(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        sp = StatePersistence(path, lambda: {})
        sp.schedule_save()
        first_timer = sp._save_timer
        sp.schedule_save()
        assert sp._save_timer is not first_timer
        assert sp._save_timer is not None
        sp._save_timer.cancel()
        sp._save_timer = None

    async def test_fires_after_delay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drive the debounced save without waiting wall-clock time: schedule
        # immediately, drain the loop until the timer fires, and assert the
        # write landed.
        path = tmp_path / "state.json"
        sp = StatePersistence(path, lambda: {"fired": True})

        loop = asyncio.get_running_loop()
        original_call_later = loop.call_later
        monkeypatch.setattr(
            loop,
            "call_later",
            lambda _delay, fn, *a, **k: original_call_later(0, fn, *a, **k),
        )

        sp.schedule_save()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert path.exists()
        assert json.loads(path.read_text()) == {"fired": True}
        assert not sp._dirty


class TestDoSaveErrorHandling:
    def test_serializer_error_is_logged_not_raised(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"

        def bad_serializer() -> dict:
            raise TypeError("not serializable")

        sp = StatePersistence(path, bad_serializer)
        sp._dirty = True
        sp._do_save()  # must not raise
        assert not path.exists()

    def test_dirty_not_cleared_on_save_error(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"

        def bad_serializer() -> dict:
            raise OSError("disk full")

        sp = StatePersistence(path, bad_serializer)
        sp._dirty = True
        sp._do_save()
        assert sp._dirty


class TestFlush:
    def test_flush_saves_when_dirty(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        sp = StatePersistence(path, lambda: {"flushed": True})
        sp._dirty = True
        sp.flush()
        assert path.exists()
        assert json.loads(path.read_text()) == {"flushed": True}

    def test_flush_skips_when_not_dirty(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        call_count = 0

        def counting_serializer() -> dict:
            nonlocal call_count
            call_count += 1
            return {}

        sp = StatePersistence(path, counting_serializer)
        sp.flush()
        assert call_count == 0

    async def test_flush_cancels_pending_timer(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        sp = StatePersistence(path, lambda: {"early": True})
        sp.schedule_save()
        assert sp._save_timer is not None
        sp.flush()
        assert sp._save_timer is None
        assert path.exists()


class TestLoad:
    def test_returns_empty_dict_when_file_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.json"
        sp = StatePersistence(path, lambda: {})
        assert sp.load() == {}

    def test_returns_parsed_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text('{"x": 42}')
        sp = StatePersistence(path, lambda: {})
        assert sp.load() == {"x": 42}

    def test_returns_empty_dict_on_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("not valid json {{{")
        sp = StatePersistence(path, lambda: {})
        assert sp.load() == {}

    def test_returns_empty_dict_on_os_error(self, tmp_path: Path) -> None:
        # A directory path raises IsADirectoryError (OSError subclass) on read_text.
        path = tmp_path  # tmp_path is a directory
        sp = StatePersistence(path, lambda: {})
        assert sp.load() == {}
