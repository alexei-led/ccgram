"""Delivery-poison latch for the delivered watermark (issue #179).

A leaf module: message_queue sets it on permanent send failures and
tool_batch may set it on batch flush failures, without importing
message_queue (the pipeline's no-back-edge invariant).
"""

from __future__ import annotations

# Users whose delivered watermark must freeze until restart.
_poisoned_users: set[int] = set()


def poison_delivery(user_id: int) -> None:
    """Freeze the delivered watermark for a user (permanent send failure)."""
    _poisoned_users.add(user_id)


def delivery_poisoned(user_id: int | None = None) -> bool:
    """True when delivery is poisoned (optionally for one user)."""
    return user_id in _poisoned_users if user_id is not None else bool(_poisoned_users)


def clear_poison() -> None:
    """Test helper: clear all poison."""
    _poisoned_users.clear()
