"""Injectable polling runtime bundle.

``PollingRuntime`` groups the five stateful strategy instances that the
polling subsystem needs across a full poll cycle.  In production the
module-level ``_default_runtime`` wraps the existing singletons from
``polling_state`` so there is no behavioural change and no double
registration of ``topic_state`` callbacks.

The ``create()`` classmethod builds a fully isolated bundle for use in
tests that need independent state without touching the default singletons.
Each ``create()`` call registers its own ``topic_state`` cleanup callbacks
(each strategy constructor does this), so tests should use the
``_reset_topic_state_registry`` snapshot/restore fixture from
``test_polling_strategies.py`` to keep the registry clean.

Import direction: this module imports from ``polling_state``; **never** the
reverse.  ``polling_types`` and ``window_tick.decide`` must not import here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polling_state import (
        InteractiveUIStrategy,
        PaneStatusStrategy,
        TerminalPollState,
        TerminalScreenBuffer,
        TopicLifecycleStrategy,
    )


@dataclass
class PollingRuntime:
    """Bundle of the five stateful polling strategy instances.

    Attributes mirror the five module-level singletons in ``polling_state``
    so callers can switch between ``get_default_runtime()`` and an isolated
    test bundle without touching call sites.
    """

    poll_state: "TerminalPollState"
    screen_buffer: "TerminalScreenBuffer"
    interactive: "InteractiveUIStrategy"
    lifecycle: "TopicLifecycleStrategy"
    pane_status: "PaneStatusStrategy"

    @classmethod
    def create(cls) -> "PollingRuntime":
        """Build a fully isolated runtime with fresh strategy instances.

        Each call constructs independent state dictionaries and registers
        its own ``topic_state`` cleanup callbacks.  Use in tests to avoid
        mutating the default singletons.
        """
        # Lazy: polling_state imports are deferred to avoid a module-level
        # cycle (polling_runtime is imported by polling_state via the
        # default-runtime construction at the bottom of that module).
        # Lazy: strategy classes carry heavyweight deps (pyte, structlog);
        # defer until an isolated runtime is actually requested.
        from .polling_state import (
            InteractiveUIStrategy,
            PaneStatusStrategy,
            TerminalPollState,
            TerminalScreenBuffer,
            TopicLifecycleStrategy,
        )

        poll_state = TerminalPollState()
        screen_buffer = TerminalScreenBuffer(poll_state)
        interactive = InteractiveUIStrategy()
        lifecycle = TopicLifecycleStrategy(poll_state)
        pane_status = PaneStatusStrategy(screen_buffer, interactive)
        return cls(
            poll_state=poll_state,
            screen_buffer=screen_buffer,
            interactive=interactive,
            lifecycle=lifecycle,
            pane_status=pane_status,
        )

    def reset_window(self, window_id: str) -> None:
        """Reset per-window polling state across all strategies.

        Equivalent to calling ``reset_window_polling_state(window_id)`` on
        the default runtime; provided here so isolated test runtimes can do
        the same without importing the module-level helper.
        """
        self.poll_state.clear_seen_status(window_id)
        self.screen_buffer.clear_screen_buffer(window_id)


# ── Default runtime ────────────────────────────────────────────────────────
#
# Wraps the existing module-level singletons from ``polling_state``.  The
# singletons are constructed once when ``polling_state`` is imported (same
# as before); this object just gives them a named bundle.  No new instances
# are created, so ``topic_state`` callbacks are not double-registered.


def _build_default_runtime() -> "PollingRuntime":
    # Lazy: same reasoning as PollingRuntime.create() — defer polling_state
    # import so the module graph stays acyclic at load time.
    # Lazy: polling_state has heavyweight side effects (singleton construction)
    from .polling_state import (
        interactive_strategy,
        lifecycle_strategy,
        pane_status_strategy,
        terminal_poll_state,
        terminal_screen_buffer,
    )

    return PollingRuntime(
        poll_state=terminal_poll_state,
        screen_buffer=terminal_screen_buffer,
        interactive=interactive_strategy,
        lifecycle=lifecycle_strategy,
        pane_status=pane_status_strategy,
    )


_default_runtime: PollingRuntime | None = None


def get_default_runtime() -> PollingRuntime:
    """Return the module-level default runtime (lazily initialised).

    Lazy initialisation avoids executing ``polling_state`` top-level code
    (singleton construction) at import time of this module, which preserves
    the F4 invariant: ``polling_types`` → ``decide`` can still be imported
    without triggering the singletons.
    """
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = _build_default_runtime()
    return _default_runtime
