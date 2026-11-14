from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest, RetryAfter

from ccgram.handlers.topics.topic_provisioning_recovery import (
    recover_topic_provisioning,
)
from ccgram.multiplexer.base import WindowRef
from ccgram.providers import _ensure_registered
from ccgram.telegram_rate_limiter import NO_RETRY_RATE_LIMIT_ARGS
from ccgram.thread_router import ThreadRouter


@pytest.fixture(autouse=True)
def _isolated_window_state_owner():
    from ccgram.window_state_store import (
        WindowStateStore,
        get_window_store,
        install_window_store,
    )

    previous = get_window_store()
    store = WindowStateStore(
        schedule_save=lambda: None, on_hookless_provider_switch=lambda _wid: None
    )
    install_window_store(store)
    try:
        yield
    finally:
        install_window_store(previous)


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.resolve = AsyncMock(
        return_value=(
            SimpleNamespace(
                foreground_argv=("claude",), provider="claude", available=True
            ),
            None,
        )
    )
    return adapter


def _router() -> ThreadRouter:
    return ThreadRouter(schedule_save=lambda: None, has_window_state=lambda _wid: False)


def _restored_claim(*, target_id="@2", thread_id=42, previous_target_id=None):
    original = _router()
    if previous_target_id:
        original.bind_thread(1, 42, previous_target_id, chat_id=-100)
    claim = original.begin_topic_provisioning(
        1,
        -100,
        thread_id=thread_id,
        target_id=target_id,
        previous_target_id=previous_target_id,
        kind="topic_for_target" if thread_id is None else "target_for_topic",
        topic_mark_owner="owner:" + target_id if target_id else None,
    )
    restored = _router()
    restored.from_dict(original.to_dict())
    return restored, claim


@pytest.fixture(autouse=True)
def _isolate_persistence():
    _ensure_registered()
    with (
        patch("ccgram.handlers.topics.topic_provisioning_recovery.session_manager"),
        patch("ccgram.session.session_manager"),
        patch(
            "ccgram.multiplexer.multiplexer",
            new=SimpleNamespace(
                find_window_by_id=AsyncMock(
                    side_effect=lambda wid: SimpleNamespace(
                        window_id=wid,
                        topic_markable=True,
                        topic_eligible=True,
                        topic_mark_owner="owner:" + wid,
                        topic_marked=True,
                    )
                ),
                set_topic_mark=AsyncMock(return_value=True),
            ),
        ) as native,
        patch("ccgram.handlers.topics.topic_provisioning_recovery.multiplexer", native),
        patch("ccgram.handlers.topics.topic_orchestration.tmux_manager", native),
        patch("ccgram.handlers.topics.topic_deletion.session_manager"),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_query.get_window_provider",
            return_value="claude",
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.clear_topic_state",
            new_callable=AsyncMock,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_query.resolve_window_alias",
            side_effect=lambda wid: wid,
        ),
    ):
        yield


@pytest.mark.parametrize("mode", ["auto", "marked"])
@pytest.mark.parametrize("kind", ["topic_for_target", "target_for_topic"])
async def test_reloaded_second_chat_claim_keeps_original_owner_after_bound_chat_observes_replacement(
    monkeypatch, mode, kind
):
    from ccgram.config import config
    from ccgram.window_state_ports import lifecycle_state

    monkeypatch.setattr(config, "topic_discovery_mode", mode)
    original = _router()
    original.bind_thread(1, 11, "@2", chat_id=-101)
    claim = original.begin_topic_provisioning(
        1,
        -100,
        thread_id=42,
        target_id="@2",
        kind=kind,
        topic_mark_owner="original-O",
    )
    lifecycle_state.record_topic_mark_owner("@2", "original-O")
    lifecycle_state.observe_topic_mark_owner("@2", "replacement-P")
    restored = _router()
    restored.from_dict(original.to_dict())
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=WindowRef(
                "@2",
                "replacement",
                "/p",
                topic_marked=True,
                topic_mark_owner="replacement-P",
            )
        ),
        set_topic_mark=AsyncMock(return_value=True),
    )
    client = AsyncMock()
    with (
        patch("ccgram.handlers.topics.topic_provisioning_recovery.multiplexer", native),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            AsyncMock(return_value=True),
        ),
    ):
        assert await recover_topic_provisioning(
            client, router=restored, adapter=_adapter()
        ) == {"unresolved": 1}
    assert restored.get_window_for_chat_thread(-101, 11) == "@2"
    assert restored.get_window_for_chat_thread(-100, 42) is None
    retained = restored.get_topic_provisioning(claim.claim_id)
    assert retained is not None and retained.topic_mark_owner == "original-O"
    assert lifecycle_state.get_topic_mark_owner("@2") == "replacement-P"
    native.set_topic_mark.assert_not_awaited()
    client.delete_forum_topic.assert_not_awaited()


@pytest.mark.parametrize("mode", ["auto", "marked"])
@pytest.mark.parametrize("kind", ["topic_for_target", "target_for_topic"])
@pytest.mark.parametrize("topic_present", [True, False])
async def test_recovery_rechecks_automatic_eligibility_without_filtering_explicit_creation(
    monkeypatch, mode, kind, topic_present
):
    from ccgram.config import config
    from ccgram.window_state_ports import lifecycle_state

    monkeypatch.setattr(config, "topic_discovery_mode", mode)
    original = _router()
    claim = original.begin_topic_provisioning(
        1,
        -100,
        thread_id=42,
        target_id="@2",
        kind=kind,
        topic_mark_owner="original-O",
    )
    restored = _router()
    restored.from_dict(original.to_dict())
    lifecycle_state.record_topic_mark_owner("@2", "original-O")
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=WindowRef(
                "@2",
                "excluded",
                "/p",
                topic_eligible=False,
                topic_marked=True,
                topic_mark_owner="original-O",
            )
        ),
        set_topic_mark=AsyncMock(return_value=True),
    )
    with (
        patch("ccgram.handlers.topics.topic_provisioning_recovery.multiplexer", native),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            AsyncMock(return_value=True),
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            AsyncMock(return_value=topic_present),
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            AsyncMock(return_value=True),
        ) as creator,
    ):
        result = await recover_topic_provisioning(
            AsyncMock(), router=restored, adapter=_adapter()
        )
    if kind == "topic_for_target":
        assert result == {"unresolved": 1}
        assert restored.iter_topic_provisionings() == [claim]
        creator.assert_not_awaited()
    else:
        assert result == ({"bound": 1} if topic_present else {"recreated": 1})
    native.set_topic_mark.assert_not_awaited()


@pytest.mark.parametrize("presence", [True, False, None])
async def test_restored_known_topic_follows_authoritative_presence(presence):
    router, claim = _restored_claim()
    client = AsyncMock()
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
        return_value=presence,
    ) as probe:
        outcome = await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        )

    probe.assert_awaited_once_with("@2", None)
    if presence is True:
        assert outcome == {"bound": 1}
        assert router.get_window_for_chat_thread(-100, 42) == "@2"
        assert not router.iter_topic_provisionings()
        client.delete_forum_topic.assert_not_awaited()
    elif presence is False:
        assert outcome == {"deleted": 1}
        assert not router.iter_topic_provisionings()
        assert not list(router.iter_retired_topics())
        client.delete_forum_topic.assert_awaited_once_with(
            -100, 42, rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS
        )
    else:
        assert outcome == {"unresolved": 1}
        assert router.iter_topic_provisionings() == [claim]
        client.delete_forum_topic.assert_not_awaited()


async def test_marked_explicit_creation_recovers_only_after_verified_selection(
    monkeypatch,
):
    from ccgram.config import config
    from ccgram.window_state_ports import lifecycle_state

    monkeypatch.setattr(config, "topic_discovery_mode", "marked", raising=False)
    router, claim = _restored_claim()
    lifecycle_state.set_topic_selection(
        "@2", detached=False, pending=True, owner="owner:@2"
    )
    client = AsyncMock()
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=SimpleNamespace(
                window_id="@2",
                topic_markable=True,
                topic_eligible=True,
                topic_mark_owner="owner:" + "@2",
                topic_marked=True,
            )
        ),
        set_topic_mark=AsyncMock(return_value=False),
    )
    with (
        patch("ccgram.multiplexer.multiplexer", native),
        patch("ccgram.handlers.topics.topic_provisioning_recovery.multiplexer", native),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"unresolved": 1}
        assert router.get_window_for_chat_thread(-100, 42) is None
        assert router.iter_topic_provisionings() == [claim]
        assert lifecycle_state.get_pending_topic_mark("@2") is True
        native.set_topic_mark.return_value = True
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"bound": 1}

    assert router.get_window_for_chat_thread(-100, 42) == "@2"
    assert lifecycle_state.get_pending_topic_mark("@2") is None


@pytest.mark.parametrize(
    "owner,marked,expected",
    [
        ("original-owner", True, "bound"),
        ("other-terminal-owner", True, "unresolved"),
        ("original-owner", False, "unresolved"),
        (None, True, "unresolved"),
    ],
)
async def test_automatic_claim_recovery_requires_saved_native_owner_without_marking(
    monkeypatch, owner, marked, expected
):
    from ccgram.config import config
    from ccgram.window_state_ports import lifecycle_state

    monkeypatch.setattr(config, "topic_discovery_mode", "marked")
    original = _router()
    claim = original.begin_topic_provisioning(
        1,
        -100,
        target_id="@2",
        kind="topic_for_target",
        topic_mark_owner="original-owner",
    )
    original.attach_provisioning_topic(claim.claim_id, 42)
    router = _router()
    router.from_dict(original.to_dict())
    lifecycle_state.record_topic_mark_owner("@2", "original-owner")
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=SimpleNamespace(
                window_id="@2",
                topic_markable=True,
                topic_eligible=True,
                topic_mark_owner=owner,
                topic_marked=marked,
            )
        ),
        set_topic_mark=AsyncMock(return_value=True),
    )
    with (
        patch("ccgram.multiplexer.multiplexer", native),
        patch("ccgram.handlers.topics.topic_provisioning_recovery.multiplexer", native),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        assert await recover_topic_provisioning(
            AsyncMock(), router=router, adapter=_adapter()
        ) == {expected: 1}
    native.set_topic_mark.assert_not_awaited()
    assert lifecycle_state.get_topic_mark_owner("@2") == "original-owner"


@pytest.mark.parametrize("kind", ["target_for_topic", "topic_for_target"])
@pytest.mark.parametrize(
    "saved_owner",
    [None, "saved-owner"],
    ids=["initially-unknown", "known-witness-lost"],
)
async def test_auto_recovery_waits_for_a_lost_known_owner_but_not_initial_unknown(
    monkeypatch,
    kind,
    saved_owner,
):
    from ccgram.config import config
    from ccgram.window_state_ports import lifecycle_state

    monkeypatch.setattr(config, "topic_discovery_mode", "auto")
    original = _router()
    original.begin_topic_provisioning(
        1, -100, target_id="@2", thread_id=42, kind=kind, topic_mark_owner=saved_owner
    )
    router = _router()
    router.from_dict(original.to_dict())
    lifecycle_state.record_topic_mark_owner("@2", saved_owner)
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=SimpleNamespace(
                window_id="@2",
                topic_markable=True,
                topic_eligible=True,
                topic_mark_owner=None,
                topic_marked=None,
            )
        ),
        set_topic_mark=AsyncMock(return_value=True),
    )
    with (
        patch("ccgram.multiplexer.multiplexer", native),
        patch("ccgram.handlers.topics.topic_provisioning_recovery.multiplexer", native),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        claims_before = router.iter_topic_provisionings()
        assert await recover_topic_provisioning(
            AsyncMock(), router=router, adapter=_adapter()
        ) == ({"bound": 1} if saved_owner is None else {"unresolved": 1})
        if saved_owner is not None:
            assert router.get_window_for_chat_thread(-100, 42) is None
            assert router.iter_topic_provisionings() == claims_before
            assert lifecycle_state.get_topic_mark_owner("@2") == saved_owner
            native.find_window_by_id.return_value.topic_mark_owner = saved_owner
            assert await recover_topic_provisioning(
                AsyncMock(), router=router, adapter=_adapter()
            ) == {"bound": 1}
        assert router.get_window_for_chat_thread(-100, 42) == "@2"
    native.set_topic_mark.assert_not_awaited()
    assert lifecycle_state.get_topic_mark_owner("@2") == saved_owner


async def test_detached_stale_creation_claim_never_remarks_terminal(monkeypatch):
    from ccgram.config import config
    from ccgram.window_state_ports import lifecycle_state

    monkeypatch.setattr(config, "topic_discovery_mode", "marked")
    router, _claim = _restored_claim()
    lifecycle_state.set_topic_selection(
        "@2", detached=True, pending=None, owner="owner:@2"
    )
    native = SimpleNamespace(
        find_window_by_id=AsyncMock(
            return_value=SimpleNamespace(
                window_id="@2",
                topic_markable=True,
                topic_eligible=True,
                topic_mark_owner="owner:@2",
                topic_marked=False,
            )
        ),
        set_topic_mark=AsyncMock(return_value=True),
    )
    with (
        patch("ccgram.multiplexer.multiplexer", native),
        patch("ccgram.handlers.topics.topic_provisioning_recovery.multiplexer", native),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        assert await recover_topic_provisioning(
            AsyncMock(), router=router, adapter=_adapter()
        ) == {"unresolved": 1}
    native.set_topic_mark.assert_not_awaited()


async def test_title_repair_rate_limit_retains_claim_then_stops_batch():
    router, claim = _restored_claim()
    client = AsyncMock()
    client.edit_forum_topic.side_effect = RetryAfter(60)
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
        return_value=True,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"rate_limited": 1}

    assert router.get_window_for_chat_thread(-100, 42) is None
    assert router.iter_topic_provisionings() == [claim]
    client.send_message.assert_not_awaited()
    client.delete_message.assert_not_awaited()
    client.edit_forum_topic.assert_awaited_once_with(
        -100,
        42,
        name="@2",
        rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS,
    )


async def test_active_creation_is_never_reconciled_as_abandoned():
    router = _router()
    claim = router.begin_topic_provisioning(
        1, -100, thread_id=42, target_id="@2", kind="target_for_topic"
    )
    client = AsyncMock()
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
        return_value=False,
    ) as probe:
        assert (
            await recover_topic_provisioning(client, router=router, adapter=_adapter())
            == {}
        )
    probe.assert_not_awaited()
    client.delete_forum_topic.assert_not_awaited()
    assert router.iter_topic_provisionings() == [claim]


@pytest.mark.parametrize(("target_id", "thread_id"), [(None, 42), ("@2", None)])
async def test_missing_remote_identity_remains_protected(target_id, thread_id):
    router, claim = _restored_claim(target_id=target_id, thread_id=thread_id)
    client = AsyncMock()
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
    ) as probe:
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"unresolved": 1}
    probe.assert_not_awaited()
    client.delete_forum_topic.assert_not_awaited()
    assert router.iter_topic_provisionings() == [claim]


async def test_creation_changed_during_probe_is_not_deleted():
    router, claim = _restored_claim()

    async def probe(*_args):
        router.attach_provisioning_target(claim.claim_id, "@3")
        return False

    client = AsyncMock()
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        side_effect=probe,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"changed": 1}
    assert router.iter_topic_provisionings()[0].target_id == "@3"
    client.delete_forum_topic.assert_not_awaited()


async def test_deleted_topic_is_unbound_and_recreated_for_live_target():
    router, claim = _restored_claim(previous_target_id="@dead")
    client = AsyncMock()

    async def recreate(
        _client,
        chat_id,
        target_id,
        topic_name,
        *,
        user_id,
        propagate_retry_after,
        claim_id,
        automatic,
    ):
        router.attach_provisioning_topic(claim_id, 77)
        return router.commit_topic_provisioning(claim_id, window_name=topic_name)

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            side_effect=recreate,
        ) as create,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"recreated": 1}

    assert router.get_window_for_chat_thread(-100, 42) is None
    assert router.get_window_for_chat_thread(-100, 77) == "@2"
    assert not router.iter_topic_provisionings()
    assert not list(router.iter_retired_topics())
    create.assert_awaited_once_with(
        client,
        -100,
        "@2",
        "@2",
        user_id=1,
        propagate_retry_after=True,
        claim_id=claim.claim_id,
        automatic=False,
    )


async def test_deleted_topic_recreation_uses_cached_window_name():
    router, claim = _restored_claim(previous_target_id="@dead")
    router.window_display_names["@2"] = "cached-project"
    client = AsyncMock()

    async def recreate(
        _client,
        _chat_id,
        _target_id,
        _topic_name,
        *,
        user_id,
        propagate_retry_after,
        claim_id,
        automatic,
    ):
        return True

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            side_effect=recreate,
        ) as create,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"recreated": 1}

    create.assert_awaited_once_with(
        client,
        -100,
        "@2",
        "cached-project",
        user_id=1,
        propagate_retry_after=True,
        claim_id=claim.claim_id,
        automatic=False,
    )


async def test_deleted_topic_preserves_existing_target_binding():
    router, _claim = _restored_claim()
    router.bind_thread(1, 77, "@2", chat_id=-100)
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            new_callable=AsyncMock,
        ) as create,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"released": 1}

    assert router.get_window_for_chat_thread(-100, 42) is None
    assert router.get_window_for_chat_thread(-100, 77) == "@2"
    create.assert_not_awaited()


async def test_existing_target_binding_blocks_commit_of_present_topic():
    router, claim = _restored_claim()
    router.bind_thread(1, 77, "@2", chat_id=-100)
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"unresolved": 1}

    assert router.iter_topic_provisionings() == [claim]
    assert router.get_window_for_chat_thread(-100, 77) == "@2"
    assert router.get_window_for_chat_thread(-100, 42) is None


async def test_hookless_agent_in_shell_keeps_present_topic_quarantined():
    """A pane still running its launch shell must not be committed.

    Committing would bind the topic and the next poll would read that shell
    as an exited agent and kill a window whose CLI was merely slow to start.
    """
    router, claim = _restored_claim()
    client = AsyncMock()
    provider = MagicMock()
    provider.capabilities.supports_hook = False
    provider.capabilities.chat_first_command_path = False
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_query.get_window_provider",
            return_value="antigravity",
        ),
        patch("ccgram.providers.registry.registry.get", return_value=provider),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.agent_process_started",
            new_callable=AsyncMock,
            return_value=False,
        ) as started,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"unresolved": 1}

    started.assert_awaited_once()
    observed = started.await_args
    assert observed is not None
    assert observed.args[0] == "@2"
    assert observed.args[1] is not None
    assert router.iter_topic_provisionings() == [claim]
    assert router.get_window_for_chat_thread(-100, 42) is None


async def test_hookless_agent_unknown_probe_also_holds_the_claim():
    """An unanswerable probe must not commit on a guess."""
    router, claim = _restored_claim()
    client = AsyncMock()
    provider = MagicMock()
    provider.capabilities.supports_hook = False
    provider.capabilities.chat_first_command_path = False
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_query.get_window_provider",
            return_value="antigravity",
        ),
        patch("ccgram.providers.registry.registry.get", return_value=provider),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.agent_process_started",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"unresolved": 1}

    assert router.iter_topic_provisionings() == [claim]


async def test_hookless_agent_in_shell_commits_once_the_cli_took_over():
    router, _claim = _restored_claim()
    client = AsyncMock()
    provider = MagicMock()
    provider.capabilities.supports_hook = False
    provider.capabilities.chat_first_command_path = False
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_query.get_window_provider",
            return_value="antigravity",
        ),
        patch("ccgram.providers.registry.registry.get", return_value=provider),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.agent_process_started",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"bound": 1}

    assert router.get_window_for_chat_thread(-100, 42) == "@2"


async def test_unknown_topic_probe_retains_claim():
    router, claim = _restored_claim()
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            new_callable=AsyncMock,
        ) as create,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"unresolved": 1}

    assert router.iter_topic_provisionings() == [claim]
    create.assert_not_awaited()


async def test_claim_changed_during_topic_probe_is_protected():
    router, claim = _restored_claim()
    client = AsyncMock()

    async def probe(*_args, **_kwargs):
        router.attach_provisioning_target(claim.claim_id, "@3")
        return False

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            side_effect=probe,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            new_callable=AsyncMock,
        ) as create,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"changed": 1}

    assert router.iter_topic_provisionings()[0].target_id == "@3"
    create.assert_not_awaited()


async def test_retry_after_stops_recovery_batch_without_settling_claims():
    router, first = _restored_claim()
    second = router.begin_topic_provisioning(
        1, -100, thread_id=43, target_id="@3", kind="target_for_topic"
    )
    router.mark_provisioning_uncertain(second.claim_id)
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ) as presence,
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            side_effect=RetryAfter(60),
        ),
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"rate_limited": 1}

    presence.assert_awaited_once_with("@2", None)
    assert [item.claim_id for item in router.iter_topic_provisionings()] == [
        first.claim_id,
        second.claim_id,
    ]


async def test_retry_after_during_real_recreation_stops_recovery_batch():
    router, first = _restored_claim()
    second = router.begin_topic_provisioning(
        1, -100, thread_id=43, target_id="@3", kind="target_for_topic"
    )
    router.mark_provisioning_uncertain(second.claim_id)
    client = AsyncMock()
    client.create_forum_topic.side_effect = RetryAfter(60)

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ) as probe,
        patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router",
            router,
        ),
        patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
        patch(
            "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
            {},
        ) as retry_until,
        patch(
            "ccgram.handlers.topics.topic_orchestration.time.monotonic",
            return_value=100.0,
        ),
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"rate_limited": 1}

    probe.assert_awaited_once()
    client.create_forum_topic.assert_awaited_once_with(
        chat_id=-100,
        name="@2",
    )
    claims = router.iter_topic_provisionings()
    assert [item.claim_id for item in claims] == [
        first.claim_id,
        second.claim_id,
    ]
    assert claims[0].thread_id is None
    assert claims[0].retry_thread_id == 42
    assert claims[0].retry_at > 0
    assert claims[0].uncertain is False
    assert router.owns_topic_provisioning(claims[0].claim_id) is False
    assert retry_until[-100] > 100.0


async def test_deferred_recreation_survives_reload_and_recovers_without_event():
    router, first = _restored_claim()
    client = AsyncMock()
    client.create_forum_topic.side_effect = BadRequest("denied")

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router",
            router,
        ),
        patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
        patch(
            "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
            {},
        ),
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"deferred": 1}

    deferred = router.get_topic_provisioning(first.claim_id)
    assert deferred is not None
    assert deferred.thread_id is None
    assert deferred.retry_thread_id == 42
    assert deferred.retry_at > 0
    assert deferred.uncertain is False

    restored = _router()
    restored.from_dict(router.to_dict())
    retry_client = AsyncMock()
    retry_client.create_forum_topic.return_value = MagicMock(message_thread_id=77)

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.time.time",
            return_value=deferred.retry_at + 1,
        ),
        patch(
            "ccgram.handlers.topics.topic_orchestration.thread_router",
            restored,
        ),
        patch("ccgram.handlers.topics.topic_orchestration.session_manager"),
        patch(
            "ccgram.handlers.topics.topic_orchestration._topic_create_retry_until",
            {},
        ),
    ):
        assert await recover_topic_provisioning(
            retry_client, router=restored, adapter=_adapter()
        ) == {"recreated": 1}

    assert restored.get_window_for_chat_thread(-100, 42) is None
    assert restored.get_window_for_chat_thread(-100, 77) == "@2"
    assert restored.iter_topic_provisionings() == []


async def test_uncertain_recreation_stays_quarantined_when_target_is_closed():
    router, claim = _restored_claim()
    prepared = router.prepare_topic_recreation(claim.claim_id)
    router.defer_topic_recreation(
        prepared.claim_id,
        retry_at=0.0,
        uncertain=True,
    )
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=False,
        ) as presence,
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
        ) as probe,
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            new_callable=AsyncMock,
        ) as create,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"unresolved": 1}

    presence.assert_not_awaited()
    probe.assert_not_awaited()
    create.assert_not_awaited()
    current = router.get_topic_provisioning(claim.claim_id)
    assert current is not None
    assert current.uncertain is True
    assert current.retry_thread_id == 42


async def test_recreation_flush_failure_releases_ownership_for_later_retry():
    router, claim = _restored_claim()
    client = AsyncMock()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            side_effect=[True, True],
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.session_manager.flush_state",
            side_effect=OSError("persist failed"),
        ) as flush,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"changed": 1}

    deferred = router.get_topic_provisioning(claim.claim_id)
    assert deferred is not None
    assert deferred.thread_id is None
    assert deferred.retry_thread_id == 42
    assert deferred.uncertain is False
    assert router.owns_topic_provisioning(claim.claim_id) is False

    flush.side_effect = None

    async def recreate(
        _client,
        _chat_id,
        _target_id,
        _topic_name,
        *,
        user_id,
        propagate_retry_after,
        claim_id,
        automatic,
    ):
        router.attach_provisioning_topic(claim_id, 77)
        return router.commit_topic_provisioning(claim_id)

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            side_effect=recreate,
        ),
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"recreated": 1}

    assert router.get_window_for_chat_thread(-100, 77) == "@2"
    assert router.iter_topic_provisionings() == []


async def test_failed_replacement_preserves_previous_binding():
    router, _claim = _restored_claim(previous_target_id="@1")
    client = AsyncMock()
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
        return_value=False,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"released": 1}
    assert router.get_window_for_chat_thread(-100, 42) == "@1"
    assert not router.iter_topic_provisionings()
    client.delete_forum_topic.assert_not_awaited()


async def test_rate_limit_preserves_remaining_recovery_claims():
    router, _claim = _restored_claim()
    second = router.begin_topic_provisioning(
        1, -100, thread_id=43, target_id="@3", kind="target_for_topic"
    )
    router.mark_provisioning_uncertain(second.claim_id)
    client = AsyncMock()
    client.delete_forum_topic.side_effect = RetryAfter(60)
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
        return_value=False,
    ) as probe:
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"rate_limited": 1}
    assert probe.await_count == 1
    assert [item.claim_id for item in router.iter_topic_provisionings()] == [
        second.claim_id
    ]
    retired = list(router.iter_retired_topics())
    assert len(retired) == 1
    assert retired[0].retry_at > 0


async def test_recovery_restores_recorded_title_without_probe_messages():
    router, _claim = _restored_claim()
    router.window_display_names["@2"] = "restored-project"
    client = AsyncMock()
    client.edit_forum_topic.side_effect = BadRequest("TOPIC_NOT_MODIFIED")
    with patch(
        "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
        new_callable=AsyncMock,
        return_value=True,
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=_adapter()
        ) == {"bound": 1}

    assert router.get_window_for_chat_thread(-100, 42) == "@2"
    client.edit_forum_topic.assert_awaited_once_with(
        -100,
        42,
        name="restored-project",
        rate_limit_args=NO_RETRY_RATE_LIMIT_ARGS,
    )
    client.send_message.assert_not_awaited()
    client.delete_message.assert_not_awaited()


@pytest.mark.parametrize("supports_hook", [True, False])
@pytest.mark.parametrize("age", [30, 181])
@pytest.mark.parametrize(
    ("argv", "foreground_provider"),
    [((), "claude"), (("-zsh",), "shell"), (("codex",), "codex")],
)
async def test_hook_registration_cannot_commit_without_matching_foreground(
    age, argv, foreground_provider, supports_hook
):
    router, claim = _restored_claim()
    client = AsyncMock()
    adapter = _adapter()
    adapter.resolve.return_value = (
        SimpleNamespace(
            foreground_argv=argv, provider=foreground_provider, available=True
        ),
        None,
    )
    provider = SimpleNamespace(
        capabilities=SimpleNamespace(
            supports_hook=supports_hook, chat_first_command_path=False
        )
    )
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_query.get_window_provider",
            return_value="claude",
        ),
        patch("ccgram.providers.registry.registry.get", return_value=provider),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.time.time",
            return_value=claim.created_at + age,
        ),
    ):
        assert await recover_topic_provisioning(
            client, router=router, adapter=adapter
        ) == {"unresolved": 1}
        assert router.iter_topic_provisionings() == [claim]
        assert router.get_window_for_chat_thread(-100, 42) is None
        client.delete_forum_topic.assert_not_awaited()

        adapter.resolve.return_value = (
            SimpleNamespace(
                foreground_argv=("claude",), provider="claude", available=True
            ),
            None,
        )
        assert await recover_topic_provisioning(
            client, router=router, adapter=adapter
        ) == {"bound": 1}

    assert router.get_window_for_chat_thread(-100, 42) == "@2"
    assert not router.iter_topic_provisionings()


async def test_recovery_without_provider_identity_remains_quarantined():
    router, claim = _restored_claim()
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_query.get_window_provider",
            return_value=None,
        ),
    ):
        outcome = await recover_topic_provisioning(
            AsyncMock(), router=router, adapter=_adapter()
        )
    assert outcome == {"unresolved": 1}
    assert router.iter_topic_provisionings() == [claim]
    assert router.get_window_for_chat_thread(-100, 42) is None


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize(
    ("argv", "provider", "available"),
    [
        ((), "claude", True),
        (("-zsh",), "shell", True),
        (("codex",), "codex", True),
        (("claude",), "claude", False),
    ],
)
async def test_deleted_topic_waits_for_ready_agent_before_recreation(
    retry, argv, provider, available
):
    router, claim = _restored_claim(previous_target_id="@dead" if not retry else None)
    if retry:
        prepared = router.prepare_topic_recreation(claim.claim_id)
        router.defer_topic_recreation(prepared.claim_id, retry_at=0.0, uncertain=False)
        saved = router.to_dict()
        router = _router()
        router.from_dict(saved)
        claim = router.iter_topic_provisionings()[0]
    adapter = _adapter()
    adapter.resolve.return_value = (
        SimpleNamespace(foreground_argv=argv, provider=provider, available=available),
        None,
    )

    async def create(
        _client,
        _chat_id,
        _target_id,
        topic_name,
        *,
        user_id,
        propagate_retry_after,
        claim_id,
        automatic,
    ):
        router.attach_provisioning_topic(claim_id, 77)
        return router.commit_topic_provisioning(claim_id, window_name=topic_name)

    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            side_effect=create,
        ) as create_topic,
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.clear_topic_state",
            new_callable=AsyncMock,
        ) as clear,
    ):
        assert await recover_topic_provisioning(
            AsyncMock(), router=router, adapter=adapter
        ) == {"unresolved": 1}
        assert router.iter_topic_provisionings() == [claim]
        create_topic.assert_not_awaited()
        clear.assert_not_awaited()

        adapter.resolve.return_value = (
            SimpleNamespace(
                foreground_argv=("claude",), provider="claude", available=True
            ),
            None,
        )
        assert await recover_topic_provisioning(
            AsyncMock(), router=router, adapter=adapter
        ) == {"recreated": 1}

    create_topic.assert_awaited_once()
    if retry:
        clear.assert_not_awaited()
    else:
        clear.assert_awaited_once_with(
            1, 42, client=ANY, window_id="@dead", chat_id=-100, window_dead=False
        )
    assert router.get_window_for_chat_thread(-100, 42) is None
    assert router.get_window_for_chat_thread(-100, 77) == "@2"
    assert not router.iter_topic_provisionings()


async def test_recreation_readiness_probe_cannot_commit_changed_claim():
    router, claim = _restored_claim()
    adapter = _adapter()

    async def resolve(_window_id):
        router.attach_provisioning_target(claim.claim_id, "@3")
        return SimpleNamespace(
            foreground_argv=("claude",), provider="claude", available=True
        ), None

    adapter.resolve.side_effect = resolve
    with (
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.window_presence",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.probe_topic_exists",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "ccgram.handlers.topics.topic_provisioning_recovery.create_topic_in_chat",
            new_callable=AsyncMock,
        ) as create,
    ):
        assert await recover_topic_provisioning(
            AsyncMock(), router=router, adapter=adapter
        ) == {"changed": 1}
    create.assert_not_awaited()
    assert router.iter_topic_provisionings()[0].target_id == "@3"
