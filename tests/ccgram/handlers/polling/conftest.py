"""Shared fixtures for the polling test package."""

from __future__ import annotations

import pytest

from ccgram.topic_state_registry import topic_state


@pytest.fixture(autouse=True)
def _restore_topic_state_registry():
    """Undo cleanup registrations made by ``PollingRuntime.create()``.

    Every isolated runtime registers its own teardown callbacks with the
    global registry; without this the registry grows across tests and stale
    strategies get invoked by unrelated cleanups.
    """
    snapshot = {scope: list(bucket) for scope, bucket in topic_state._cleanups.items()}
    yield
    for scope, bucket in topic_state._cleanups.items():
        bucket[:] = snapshot[scope]
