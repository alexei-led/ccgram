from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from _topic_admission_helpers import AdmissionRace
from ccgram.config import config
from ccgram.handlers.topics import (
    topic_orchestration,
    topic_provisioning_recovery,
    window_callbacks,
)
from ccgram.providers import _ensure_registered
from ccgram.thread_router import ThreadRouter
from ccgram.window_state_ports import lifecycle_state
from ccgram.window_state_store import (
    WindowStateStore,
    get_window_store,
    install_window_store,
)


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("path", ["picker", "automatic", "explicit", "recovery"])
async def test_completed_unmark_cannot_overtake_final_admission(
    monkeypatch, path, cancelled
):
    import ccgram.multiplexer
    import ccgram.session

    previous = get_window_store()
    store = WindowStateStore(
        schedule_save=lambda: None, on_hookless_provider_switch=lambda _wid: None
    )
    install_window_store(store)
    router = ThreadRouter(schedule_save=lambda: None, has_window_state=store.has_window)
    race = AdmissionRace(
        store, router, {"picker": 3, "automatic": 1, "explicit": 1, "recovery": 2}[path]
    )
    manager = SimpleNamespace(flush_state=lambda: None)
    monkeypatch.setattr(config, "topic_discovery_mode", "marked")
    monkeypatch.setattr(ccgram.multiplexer, "multiplexer", race.native)
    monkeypatch.setattr(ccgram.session, "session_manager", manager)
    for module in (window_callbacks, topic_orchestration, topic_provisioning_recovery):
        monkeypatch.setattr(module, "thread_router", router)
        monkeypatch.setattr(module, "session_manager", manager)
    monkeypatch.setattr(topic_orchestration, "tmux_manager", race.native)
    monkeypatch.setattr(topic_provisioning_recovery, "multiplexer", race.native)
    lifecycle_state.record_topic_mark_owner("@5", "owner:@5")
    store.set_window_provider("@5", "claude")
    _ensure_registered()
    try:
        if path == "picker":
            query = MagicMock(answer=AsyncMock())
            update = MagicMock(
                callback_query=SimpleNamespace(
                    message=SimpleNamespace(chat=SimpleNamespace(id=-100999))
                )
            )
            result = await race.run(
                window_callbacks._select_and_bind(
                    query, 100, 42, "@5", race.window, update
                ),
                cancelled=cancelled,
            )
            assert result is (None if cancelled else True)
        else:
            claim = router.begin_topic_provisioning(
                100,
                -100999,
                target_id="@5",
                kind="topic_for_target",
                topic_mark_owner="owner:@5",
            )
            router.attach_provisioning_topic(claim.claim_id, 42)
            claim = router.get_topic_provisioning(claim.claim_id)
            assert claim is not None
            if path == "recovery":
                router.mark_provisioning_uncertain(claim.claim_id)
                claim = router.get_topic_provisioning(claim.claim_id)
                assert claim is not None
                adapter = MagicMock(
                    resolve=AsyncMock(
                        return_value=(
                            SimpleNamespace(
                                foreground_argv=("claude",),
                                provider="claude",
                                available=True,
                            ),
                            None,
                        )
                    )
                )
                result = await race.run(
                    topic_provisioning_recovery._commit_present_topic(
                        router, claim, adapter
                    ),
                    cancelled=cancelled,
                )
                assert result == (None if cancelled else "bound")
            else:
                result = await race.run(
                    topic_orchestration._finish_topic_creation(
                        claim.claim_id,
                        -100999,
                        "@5",
                        "selected",
                        42,
                        admission=race.window if path == "automatic" else None,
                        expected_owner="owner:@5",
                    ),
                    cancelled=cancelled,
                )
                assert result is (None if cancelled else True)
        assert router.get_window_for_thread(100, 42, -100999) == (
            None if cancelled else "@5"
        )
    finally:
        install_window_store(previous)
