"""Debounced, atomic JSON state persistence.

Extracted from SessionManager to provide reusable state saving with:
  - schedule_save(): debounced 0.5s save (resets on each call).
  - do_save(serialize_fn): atomic write via temp+rename.
  - flush(): immediate save, with optional strict error propagation.
  - load(): read JSON and return raw dict.
"""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from .utils import atomic_write_json

logger = structlog.get_logger()

_SaveError = (OSError, TypeError, ValueError)


class StatePersistence:
    """Debounced, atomic JSON file persistence."""

    def __init__(self, path: Path, serialize_fn: Callable[[], dict[str, Any]]) -> None:
        self._path = path
        self._serialize_fn = serialize_fn
        self._save_timer: asyncio.TimerHandle | None = None
        self._dirty = False

    def schedule_save(self) -> None:
        """Schedule debounced save (0.5s delay, resets on each call)."""
        self._dirty = True
        if self._save_timer is not None:
            self._save_timer.cancel()
        try:
            loop = asyncio.get_running_loop()
            self._save_timer = loop.call_later(0.5, self._do_save)
        except RuntimeError:
            self._do_save()  # No event loop (tests) -> immediate

    def _do_save(self, *, strict: bool = False) -> None:
        """Write the current snapshot, logging or propagating save errors."""
        self._save_timer = None
        try:
            state = self._serialize_fn()
            atomic_write_json(self._path, state)
            self._dirty = False
        except _SaveError:
            if strict:
                raise
            logger.exception("Failed to save state")

    def flush(self, *, strict: bool = False) -> None:
        """Force an immediate save, optionally propagating save errors."""
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
        if strict or self._dirty:
            self._do_save(strict=strict)

    def load(self) -> dict[str, Any]:
        """Only a genuinely missing file means there is no durable work."""
        try:
            state = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if not isinstance(state, dict):
            raise ValueError(
                "State must be an object; refusing to discard durable work"
            )
        return state
