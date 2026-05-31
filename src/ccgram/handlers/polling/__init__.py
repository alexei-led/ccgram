"""Polling subpackage — terminal status polling orchestration.

Bundles the modules that drive the per-window polling cycle:
``polling_coordinator`` (the outer loop), ``window_tick`` (per-window
work), ``polling_types`` (pure data types and constants), ``polling_state``
(stateful strategy classes + module-level singletons), and
``periodic_tasks`` (lifecycle ticking, live view).

Only the **pure** ``polling_types`` symbols are re-exported at the
package level so that callers can ``from ccgram.handlers.polling import
TickContext`` without paying the cost of loading the stateful
singletons. Stateful symbols (``terminal_screen_buffer``,
``lifecycle_strategy``, etc.) must be imported from the canonical
``polling_state`` submodule:

    from ccgram.handlers.polling.polling_state import lifecycle_strategy

This split is what makes the F4 pure-decision-kernel invariant
provable at the import level — see
``tests/ccgram/handlers/polling/test_polling_types_purity.py``.

``periodic_tasks`` is intentionally NOT re-exported here: it imports
``topics.topic_lifecycle``, which itself imports ``polling_state``,
and re-exporting would force the load through ``polling/__init__.py``.
Callers that need ``run_periodic_tasks`` / ``run_lifecycle_tasks``
import them directly from ``handlers.polling.periodic_tasks``.
"""

from .polling_types import (
    ACTIVITY_THRESHOLD,
    MAX_PROBE_FAILURES,
    PANE_COUNT_TTL,
    RC_DEBOUNCE_SECONDS,
    SHELL_COMMANDS,
    STARTUP_TIMEOUT,
    TYPING_INTERVAL,
    PaneTransition,
    TickContext,
    TickDecision,
    TopicPollState,
    WindowPollState,
    is_shell_prompt,
)

__all__ = [
    "ACTIVITY_THRESHOLD",
    "MAX_PROBE_FAILURES",
    "PANE_COUNT_TTL",
    "RC_DEBOUNCE_SECONDS",
    "SHELL_COMMANDS",
    "STARTUP_TIMEOUT",
    "TYPING_INTERVAL",
    "PaneTransition",
    "TickContext",
    "TickDecision",
    "TopicPollState",
    "WindowPollState",
    "is_shell_prompt",
]
