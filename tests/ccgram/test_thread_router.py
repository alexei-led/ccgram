import uuid

import pytest

from ccgram.thread_router import RetiredTopic, _RETIRED_TOPIC_LIMIT, ThreadRouter


def test_observed_chat_cannot_claim_an_unknown_legacy_topic(router):
    router.bind_thread(100, 42, "@old")
    before = router.to_dict()
    router.set_group_chat_id(100, 42, -999)
    assert router.to_dict() == before
    assert list(router.iter_thread_bindings_with_chat()) == [(100, None, 42, "@old")]


def test_observed_chat_cannot_replace_recorded_legacy_chat(router):
    router.bind_thread(100, 42, "@old")
    router.group_chat_ids["100:42"] = -111
    before = router.to_dict()
    router.set_group_chat_id(100, 42, -999)
    assert router.to_dict() == before
    assert list(router.iter_thread_bindings_with_chat()) == [(100, -111, 42, "@old")]


def test_matching_recorded_chat_can_promote_legacy_binding(router):
    router.bind_thread(100, 42, "@old")
    router.group_chat_ids["100:42"] = -999
    router.set_group_chat_id(100, 42, -999)
    assert router.get_window_for_thread(100, 42, -999) == "@old"
    assert router.thread_bindings == {}


@pytest.fixture
def router() -> ThreadRouter:
    return ThreadRouter(
        schedule_save=lambda: None,
        has_window_state=lambda _wid: False,
    )


class TestBindThread:
    def test_bind_and_get(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        assert router.get_window_for_thread(100, 1) == "@1"

    def test_bind_sets_display_name(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1", window_name="proj")
        assert router.get_display_name("@1") == "proj"

    def test_bind_without_name_no_display(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        assert router.get_display_name("@1") == "@1"

    def test_bind_evicts_stale(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        router.bind_thread(100, 2, "@1")
        assert router.get_window_for_thread(100, 1) is None
        assert router.get_window_for_thread(100, 2) == "@1"

    def test_rebind_same_thread(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        router.bind_thread(100, 1, "@2")
        assert router.get_window_for_thread(100, 1) == "@2"


class TestUnbindThread:
    def test_unbind_returns_window_id(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        assert router.unbind_thread(100, 1) == "@1"

    def test_unbind_removes_binding(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        router.unbind_thread(100, 1)
        assert router.get_window_for_thread(100, 1) is None

    def test_unbind_nonexistent_returns_none(self, router: ThreadRouter) -> None:
        assert router.unbind_thread(100, 999) is None

    def test_unbind_cleans_group_chat_id(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        router.set_group_chat_id(100, 1, -999)
        router.unbind_thread(100, 1)
        assert router.resolve_chat_id(100, 1) == 100

    def test_unbind_removes_empty_user(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        router.unbind_thread(100, 1)
        assert 100 not in router.thread_bindings


class TestRetiredTopics:
    def test_persists_eligible_topic_across_restart(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 42, "@1", chat_id=-999)
        router.unbind_thread(
            100,
            42,
            chat_id=-999,
            retirement_reason="system_replacement",
            cleanup_eligible=True,
        )

        restored = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: False,
        )
        restored.from_dict(router.to_dict())

        retired = list(restored.iter_retired_topics())
        assert len(retired) == 1
        assert retired[0].chat_id == -999
        assert retired[0].thread_id == 42
        assert retired[0].reason == "system_replacement"
        assert retired[0].cleanup_eligible is True
        assert retired[0].retry_at == 0.0
        assert retired[0].closed is False

    def test_updates_retry_state_and_round_trips(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 42, "@1", chat_id=-999)
        router.unbind_thread(
            100,
            42,
            chat_id=-999,
            retirement_reason="system_replacement",
            cleanup_eligible=True,
        )
        topic = next(router.iter_retired_topics())

        updated = router.update_retired_topic(
            topic,
            retry_at=123.5,
            closed=True,
        )

        assert updated is not None
        assert updated.retry_at == 123.5
        assert updated.closed is True
        assert updated.cleanup_eligible is True

        restored = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: False,
        )
        restored.from_dict(router.to_dict())

        assert list(restored.iter_retired_topics()) == [updated]

    def test_stale_update_after_rebind_is_discarded(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 42, "@1", chat_id=-999)
        router.unbind_thread(
            100,
            42,
            chat_id=-999,
            retirement_reason="system_replacement",
            cleanup_eligible=True,
        )
        topic = next(router.iter_retired_topics())
        router.bind_thread(100, 42, "@2", chat_id=-999)

        saves: list[int] = []
        router._schedule_save = lambda: saves.append(1)
        assert router.update_retired_topic(topic, retry_at=123.5, closed=True) is None
        assert list(router.iter_retired_topics()) == []
        assert saves == []

    def test_restart_discards_record_for_an_active_rebound_topic(
        self, router: ThreadRouter
    ) -> None:
        router.from_dict(
            {
                "chat_thread_bindings": {"100:-999:42": "@2"},
                "retired_topics": [
                    {
                        "user_id": 100,
                        "chat_id": -999,
                        "thread_id": 42,
                        "reason": "system_replacement",
                        "cleanup_eligible": True,
                        "sequence": 1,
                    }
                ],
            }
        )

        assert router.get_window_for_chat_thread(-999, 42) == "@2"
        assert list(router.iter_retired_topics()) == []

    def test_default_unbind_preserves_remote_topic_intent(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 42, "@1", chat_id=-999)
        router.unbind_thread(100, 42, chat_id=-999)

        retired = list(router.iter_retired_topics())
        assert len(retired) == 1
        assert retired[0].reason == "keep_remote"
        assert retired[0].cleanup_eligible is False

    def test_rebind_clears_retired_topic_before_sync_can_delete(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 42, "@1", chat_id=-999)
        router.unbind_thread(
            100,
            42,
            chat_id=-999,
            retirement_reason="system_replacement",
            cleanup_eligible=True,
        )
        router.bind_thread(100, 42, "@2", chat_id=-999)

        assert list(router.iter_retired_topics()) == []

    def test_retention_drops_oldest_topics(self, router: ThreadRouter) -> None:
        for thread_id in range(1, _RETIRED_TOPIC_LIMIT + 3):
            router.bind_thread(100, thread_id, f"@{thread_id}", chat_id=-999)
            router.unbind_thread(
                100,
                thread_id,
                chat_id=-999,
                retirement_reason="keep_remote",
            )

        retired = list(router.iter_retired_topics())
        assert len(retired) == _RETIRED_TOPIC_LIMIT
        assert retired[0].thread_id == 3
        assert retired[-1].thread_id == _RETIRED_TOPIC_LIMIT + 2

    def test_pending_topics_survive_insertion_and_serialization(
        self, router: ThreadRouter
    ) -> None:
        for thread_id in range(1, _RETIRED_TOPIC_LIMIT + 6):
            router.bind_thread(100, thread_id, f"@{thread_id}", chat_id=-999)
            router.unbind_thread(
                100,
                thread_id,
                chat_id=-999,
                retirement_reason="system_replacement",
                cleanup_eligible=True,
            )

        restored = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: False,
        )
        restored.from_dict(router.to_dict())

        retired = list(restored.iter_retired_topics())
        assert len(retired) == _RETIRED_TOPIC_LIMIT + 5
        assert [topic.thread_id for topic in retired] == list(
            range(1, _RETIRED_TOPIC_LIMIT + 6)
        )

    def test_load_retains_pending_topics_and_newest_history(
        self, router: ThreadRouter
    ) -> None:
        raw_topics = [
            {
                "user_id": 100,
                "chat_id": -999,
                "thread_id": thread_id,
                "reason": "keep_remote",
                "cleanup_eligible": thread_id % 2 == 0,
                "sequence": thread_id,
            }
            for thread_id in range(1, 2 * _RETIRED_TOPIC_LIMIT + 6)
        ]

        router.from_dict({"retired_topics": raw_topics})

        retired = list(router.iter_retired_topics())
        assert [topic.thread_id for topic in retired] == [
            thread_id
            for thread_id in range(1, 2 * _RETIRED_TOPIC_LIMIT + 6)
            if thread_id % 2 == 0 or thread_id >= 7
        ]

    def test_load_rejects_invalid_retry_and_closed_values(
        self, router: ThreadRouter
    ) -> None:
        def raw_topic(**overrides: object) -> dict[str, object]:
            topic: dict[str, object] = {
                "user_id": 100,
                "chat_id": -999,
                "thread_id": 42,
                "reason": "system_replacement",
                "cleanup_eligible": True,
                "sequence": 1,
                "retry_at": 0.0,
                "closed": False,
            }
            topic.update(overrides)
            return topic

        router.from_dict(
            {
                "retired_topics": [
                    raw_topic(retry_at=-1.0),
                    raw_topic(retry_at=float("inf"), sequence=2),
                    raw_topic(retry_at=float("nan"), sequence=3),
                    raw_topic(closed=1, sequence=4),
                    raw_topic(sequence=5),
                ]
            }
        )

        retired = list(router.iter_retired_topics())
        assert [
            (topic.sequence, topic.retry_at, topic.closed) for topic in retired
        ] == [(5, 0.0, False)]

    def test_chatless_binding_is_not_treated_as_a_known_forum_topic(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 42, "@1")
        router.unbind_thread(100, 42, cleanup_eligible=True)

        assert list(router.iter_retired_topics()) == []


class TestTopicDeletionClaims:
    @staticmethod
    def _retire(router: ThreadRouter) -> RetiredTopic:
        router.bind_thread(100, 42, "@old", chat_id=-999)
        router.unbind_thread(
            100,
            42,
            chat_id=-999,
            retirement_reason="system_replacement",
            cleanup_eligible=True,
        )
        return next(router.iter_retired_topics())

    def test_has_active_topic_checks_all_users(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 42, "@1", chat_id=-999)
        router.bind_thread(200, 42, "@2", chat_id=-999)

        assert router.has_active_topic(-999, 42) is True
        assert router.has_active_topic(-998, 42) is False

    def test_has_active_topic_resolves_legacy_chat_metadata(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 42, "@1")
        router.group_chat_ids["100:42"] = -999

        assert router.has_active_topic(-999, 42) is True

    def test_begin_claim_rejects_bind_without_mutating_route(
        self, router: ThreadRouter
    ) -> None:
        topic = self._retire(router)
        assert router.begin_topic_deletion(topic) is True

        with pytest.raises(
            ValueError,
            match="Topic deletion is in progress; retry with a new topic",
        ):
            router.bind_thread(200, 42, "@new", chat_id=-999)

        assert router.get_window_for_chat_thread(-999, 42) is None
        assert list(router.iter_retired_topics()) == [topic]

    def test_duplicate_claim_is_rejected_and_release_allows_bind(
        self, router: ThreadRouter
    ) -> None:
        topic = self._retire(router)
        assert router.begin_topic_deletion(topic) is True
        assert router.begin_topic_deletion(topic) is False

        router.end_topic_deletion(topic)
        router.bind_thread(100, 42, "@new", chat_id=-999)

        assert router.get_window_for_chat_thread(-999, 42) == "@new"

    def test_stale_record_cannot_be_claimed_after_rebind(
        self, router: ThreadRouter
    ) -> None:
        topic = self._retire(router)
        router.bind_thread(100, 42, "@new", chat_id=-999)

        assert router.begin_topic_deletion(topic) is False

    def test_reset_releases_topic_deletion_claim(self, router: ThreadRouter) -> None:
        topic = self._retire(router)
        assert router.begin_topic_deletion(topic) is True

        router.reset()
        router.from_dict(
            {
                "retired_topics": [
                    {
                        "user_id": topic.user_id,
                        "chat_id": topic.chat_id,
                        "thread_id": topic.thread_id,
                        "reason": topic.reason,
                        "cleanup_eligible": topic.cleanup_eligible,
                        "sequence": topic.sequence,
                    }
                ]
            }
        )

        assert router.begin_topic_deletion(topic) is True


class TestTopicProvisioning:
    def test_begin_is_durable_and_owned(self, router: ThreadRouter) -> None:
        claim = router.begin_topic_provisioning(
            100,
            -999,
            target_id="target-1",
            kind="topic_for_target",
        )

        assert claim.thread_id is None
        assert claim.target_id == "target-1"
        assert router.iter_topic_provisionings() == [claim]
        assert router.owns_topic_provisioning(claim.claim_id) is True
        assert router.to_dict()["topic_provisioning"] == [
            {
                "claim_id": claim.claim_id,
                "user_id": 100,
                "chat_id": -999,
                "thread_id": None,
                "target_id": "target-1",
                "previous_target_id": None,
                "kind": "topic_for_target",
                "uncertain": False,
                "created_at": claim.created_at,
            }
        ]

    def test_attach_supersedes_target_and_commit_binds_atomically(
        self, router: ThreadRouter
    ) -> None:
        claim = router.begin_topic_provisioning(
            100,
            -999,
            target_id="provisional",
            kind="replacement",
        )
        claim = router.attach_provisioning_target(claim.claim_id, "durable")
        claim = router.attach_provisioning_topic(claim.claim_id, 42)

        assert claim.previous_target_id == "provisional"
        assert router.commit_topic_provisioning(claim.claim_id, window_name="proj")
        assert router.get_window_for_chat_thread(-999, 42) == "durable"
        assert router.get_display_name("durable") == "proj"
        assert router.iter_topic_provisionings() == []
        assert router.owns_topic_provisioning(claim.claim_id) is False

    def test_regular_bind_is_rejected_until_commit(self, router: ThreadRouter) -> None:
        claim = router.begin_topic_provisioning(
            100,
            -999,
            thread_id=42,
            target_id="durable",
            kind="target_for_topic",
        )

        with pytest.raises(ValueError, match="provisioning is in progress"):
            router.bind_thread(100, 42, "other", chat_id=-999)

        assert router.commit_topic_provisioning(claim.claim_id) is True
        assert router.get_window_for_chat_thread(-999, 42) == "durable"

    def test_other_user_cannot_bind_claimed_topic(self, router: ThreadRouter) -> None:
        router.begin_topic_provisioning(
            100,
            -999,
            thread_id=42,
            target_id="durable",
            kind="target_for_topic",
        )

        with pytest.raises(ValueError, match="provisioning is in progress"):
            router.bind_thread(200, 42, "other", chat_id=-999)

    def test_abort_retires_only_an_unbound_topic(self, router: ThreadRouter) -> None:
        claim = router.begin_topic_provisioning(
            100,
            -999,
            thread_id=42,
            target_id="durable",
            kind="target_for_topic",
        )
        aborted = router.abort_topic_provisioning(
            claim.claim_id, target_confirmed_absent=True
        )

        assert aborted == claim
        assert router.iter_topic_provisionings() == []
        retired = list(router.iter_retired_topics())
        assert len(retired) == 1
        assert retired[0].reason == "creation_failed"
        assert retired[0].cleanup_eligible is True

    def test_retired_topic_keeps_target_for_deletion_race(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 42, "durable", chat_id=-999)
        router.unbind_thread(
            100,
            42,
            chat_id=-999,
            retirement_reason="system_replacement",
            cleanup_eligible=True,
        )
        topic = next(router.iter_retired_topics())

        assert topic.target_id == "durable"
        assert router.begin_topic_deletion(topic) is True
        with pytest.raises(ValueError, match="deletion is in progress"):
            router.begin_topic_provisioning(
                100,
                -999,
                target_id="durable",
                kind="topic_for_target",
            )

        router.end_topic_deletion(topic)
        claim = router.begin_topic_provisioning(
            100,
            -999,
            target_id="durable",
            kind="topic_for_target",
        )
        router.abort_topic_provisioning(
            claim.claim_id,
            target_confirmed_absent=False,
            topic_confirmed_absent=True,
        )

    def test_attach_and_commit_reject_target_or_topic_under_deletion(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 42, "durable", chat_id=-999)
        router.unbind_thread(
            100,
            42,
            chat_id=-999,
            retirement_reason="system_replacement",
            cleanup_eligible=True,
        )
        topic = next(router.iter_retired_topics())
        assert router.begin_topic_deletion(topic) is True
        claim = router.begin_topic_provisioning(
            100,
            -999,
            target_id="new-target",
            kind="replacement",
        )

        with pytest.raises(ValueError, match="deletion is in progress"):
            router.attach_provisioning_target(claim.claim_id, "durable")
        with pytest.raises(ValueError, match="deletion is in progress"):
            router.attach_provisioning_topic(claim.claim_id, 42)

        router.end_topic_deletion(topic)
        router.attach_provisioning_target(claim.claim_id, "durable")
        router.attach_provisioning_topic(claim.claim_id, 42)
        assert router.commit_topic_provisioning(claim.claim_id) is True

    def test_abort_preserves_old_binding_and_ambiguous_claim(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 42, "old", chat_id=-999)
        claim = router.begin_topic_provisioning(
            100,
            -999,
            thread_id=42,
            target_id="new",
            kind="replacement",
        )

        uncertain = router.abort_topic_provisioning(
            claim.claim_id, target_confirmed_absent=False
        )

        assert uncertain is not None and uncertain.uncertain is True
        assert router.owns_topic_provisioning(claim.claim_id) is True
        assert router.get_window_for_chat_thread(-999, 42) == "old"
        assert list(router.iter_retired_topics()) == []

        router.mark_provisioning_uncertain(claim.claim_id)
        assert router.owns_topic_provisioning(claim.claim_id) is False

    def test_no_topic_failure_can_release_with_topic_absence_proof(
        self, router: ThreadRouter
    ) -> None:
        claim = router.begin_topic_provisioning(
            100,
            -999,
            target_id="live-target",
            kind="topic_for_target",
        )

        assert (
            router.abort_topic_provisioning(
                claim.claim_id,
                target_confirmed_absent=False,
                topic_confirmed_absent=True,
            )
            == claim
        )
        assert router.iter_topic_provisionings() == []

    def test_multi_chat_claims_and_restart_have_no_ttl(
        self, router: ThreadRouter
    ) -> None:
        first = router.begin_topic_provisioning(
            100,
            -1001,
            thread_id=42,
            target_id="target-a",
            kind="target_for_topic",
        )
        second = router.begin_topic_provisioning(
            100,
            -1002,
            thread_id=42,
            target_id="target-b",
            kind="target_for_topic",
        )
        raw = router.to_dict()
        raw["topic_provisioning"].extend(
            {
                **entry,
                "claim_id": str(uuid.uuid4()),
                "thread_id": 1000 + index,
                "target_id": f"old-{index}",
                "created_at": 1.0,
            }
            for index, entry in enumerate(raw["topic_provisioning"] * 60)
        )

        restored = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: False,
        )
        restored.from_dict(raw)

        assert restored.has_topic_provisioning(-1001, 42) is True
        assert restored.has_topic_provisioning(-1002, 42) is True
        assert len(restored.iter_topic_provisionings()) == 122
        assert restored.owns_topic_provisioning(first.claim_id) is False
        assert restored.owns_topic_provisioning(second.claim_id) is False


class TestPrivateTopicChats:
    def test_observed_chat_persists_across_restart(self, router: ThreadRouter) -> None:
        router.mark_private_topic_chat(100)

        restored = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: False,
        )
        restored.from_dict(router.to_dict())

        assert restored.is_private_topic_chat(100) is True

    def test_previous_direct_message_state_migrates_to_private_chat(
        self, router: ThreadRouter
    ) -> None:
        router.from_dict({"direct_message_topics": ["100:1"]})

        assert router.is_private_topic_chat(100) is True


class TestReverseIndex:
    def test_get_thread_for_window(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 42, "@5")
        assert router.get_thread_for_window(100, "@5") == 42

    def test_reverse_cleared_on_unbind(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 42, "@5")
        router.unbind_thread(100, 42)
        assert router.get_thread_for_window(100, "@5") is None

    def test_reverse_updated_on_evict(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        router.bind_thread(100, 2, "@1")
        assert router.get_thread_for_window(100, "@1") == 2


class TestIterThreadBindings:
    def test_iter_all(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        router.bind_thread(100, 2, "@2")
        router.bind_thread(200, 3, "@3")
        result = set(router.iter_thread_bindings())
        assert result == {(100, 1, "@1"), (100, 2, "@2"), (200, 3, "@3")}

    def test_iter_empty(self, router: ThreadRouter) -> None:
        assert list(router.iter_thread_bindings()) == []


class TestGetAllThreadWindows:
    def test_returns_user_bindings(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        router.bind_thread(100, 2, "@2")
        assert router.get_all_thread_windows(100) == {1: "@1", 2: "@2"}

    def test_unknown_user_returns_empty(self, router: ThreadRouter) -> None:
        assert router.get_all_thread_windows(999) == {}


class TestResolveWindowForThread:
    def test_none_thread_id(self, router: ThreadRouter) -> None:
        assert router.resolve_window_for_thread(100, None) is None

    def test_unbound_thread(self, router: ThreadRouter) -> None:
        assert router.resolve_window_for_thread(100, 42) is None

    def test_bound_thread(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 42, "@3")
        assert router.resolve_window_for_thread(100, 42) == "@3"


class TestResolveChatId:
    def test_with_stored_group_id(self, router: ThreadRouter) -> None:
        router.set_group_chat_id(100, 1, -999)
        assert router.resolve_chat_id(100, 1) == -999

    def test_without_group_id_fallback(self, router: ThreadRouter) -> None:
        assert router.resolve_chat_id(100, 1) == 100

    def test_with_default_group_id(self) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: False,
            default_group_id=-999,
        )
        assert router.resolve_chat_id(100, 1) == -999

    def test_stored_group_id_precedes_default_group_id(self) -> None:
        router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: False,
            default_group_id=-999,
        )
        router.set_group_chat_id(100, 1, -888)
        assert router.resolve_chat_id(100, 1) == -888

    def test_none_thread_id_fallback(self, router: ThreadRouter) -> None:
        router.set_group_chat_id(100, 1, -999)
        assert router.resolve_chat_id(100) == 100


class TestGetWindowForChatThread:
    def test_resolves_window(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1", chat_id=-999)
        assert router.get_window_for_chat_thread(-999, 1) == "@1"

    def test_no_match(self, router: ThreadRouter) -> None:
        assert router.get_window_for_chat_thread(-999, 1) is None

    def test_fallback_to_user_id(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1")
        assert router.get_window_for_chat_thread(100, 1) == "@1"


class TestDisplayNames:
    def test_get_fallback(self, router: ThreadRouter) -> None:
        assert router.get_display_name("@99") == "@99"

    def test_set_and_get(self, router: ThreadRouter) -> None:
        router.set_display_name("@1", "myproject")
        assert router.get_display_name("@1") == "myproject"

    def test_sync_display_names(self, router: ThreadRouter) -> None:
        router.window_display_names["@1"] = "old-name"
        changed = router.sync_display_names([("@1", "new-name")])
        assert changed is True
        assert router.get_display_name("@1") == "new-name"

    def test_sync_no_change(self, router: ThreadRouter) -> None:
        router.window_display_names["@1"] = "same"
        changed = router.sync_display_names([("@1", "same")])
        assert changed is False

    def test_sync_ignores_unknown(self, router: ThreadRouter) -> None:
        changed = router.sync_display_names([("@99", "something")])
        assert changed is False


class TestToDictRoundtrip:
    def test_roundtrip(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1", window_name="proj", chat_id=-999)
        router.bind_thread(200, 2, "@2")

        data = router.to_dict()
        new_router = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: False,
        )
        new_router.from_dict(data)

        assert new_router.get_window_for_thread(100, 1) == "@1"
        assert new_router.get_window_for_thread(200, 2) == "@2"
        assert new_router.resolve_chat_id(100, 1) == -999
        assert new_router.get_display_name("@1") == "proj"
        assert new_router.get_thread_for_window(100, "@1") == 1

    def test_from_dict_dedup(self, router: ThreadRouter) -> None:
        data = {
            "thread_bindings": {
                "100": {"1": "@1", "2": "@1"},
            },
            "group_chat_ids": {},
            "window_display_names": {},
        }
        assert router.from_dict(data) is True
        assert router.get_window_for_thread(100, 2) == "@1"
        assert router.get_window_for_thread(100, 1) is None

    def test_from_dict_normalizes_mixed_legacy_and_scoped_claims(
        self, router: ThreadRouter
    ) -> None:
        repaired = router.from_dict(
            {
                "thread_bindings": {"100": {"2": "@5"}},
                "group_chat_ids": {"100:2": -1001},
                "chat_thread_bindings": {"200:-1001:142": "@5"},
            }
        )

        assert repaired is True
        assert router.thread_bindings == {}
        assert router.get_window_for_thread(100, 2, -1001) is None
        assert router.get_window_for_thread(200, 142, -1001) == "@5"
        assert router.get_thread_for_window(200, "@5", -1001) == 142
        assert router.group_chat_ids == {}

    def test_from_dict_keeps_a_window_claim_in_each_chat(
        self, router: ThreadRouter
    ) -> None:
        repaired = router.from_dict(
            {
                "thread_bindings": {"100": {"2": "@5"}},
                "group_chat_ids": {"100:2": -1001},
                "chat_thread_bindings": {"200:-1002:142": "@5"},
            }
        )

        assert repaired is True
        assert router.get_window_for_thread(100, 2, -1001) == "@5"
        assert router.get_window_for_thread(200, 142, -1002) == "@5"


class TestChatScopedBindings:
    def test_same_user_can_bind_same_thread_id_in_two_chats(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 7, "@a", window_name="a", chat_id=-1001)
        router.bind_thread(100, 7, "@b", window_name="b", chat_id=-1002)

        assert router.get_window_for_chat_thread(-1001, 7) == "@a"
        assert router.get_window_for_chat_thread(-1002, 7) == "@b"
        assert router.resolve_window_for_thread(100, 7, -1001) == "@a"
        assert router.resolve_window_for_thread(100, 7, -1002) == "@b"
        assert {binding[2] for binding in router.iter_thread_bindings()} == {"@a", "@b"}

    def test_chatless_lookup_refuses_legacy_scoped_collision(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 7, "@legacy")
        router.bind_thread(100, 7, "@scoped", chat_id=-1001)

        assert router.get_window_for_thread(100, 7) is None
        assert router.get_window_for_thread(100, 7, -1001) == "@scoped"

    def test_chat_scoped_bindings_survive_round_trip(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 7, "@a", chat_id=-1001)
        restored = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: False,
        )
        restored.from_dict(router.to_dict())

        assert restored.get_window_for_chat_thread(-1001, 7) == "@a"
        assert list(restored.iter_thread_bindings()) == [(100, 7, "@a")]

    def test_cross_user_bind_evicts_existing_window_claim(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 2, "@5", chat_id=-1001)
        # Simulate metadata persisted by an earlier version before it learned
        # that chat-scoped rows make this fallback route redundant.
        router.group_chat_ids["100:2"] = -1001

        router.bind_thread(200, 142, "@5", chat_id=-1001)

        assert router.get_window_for_thread(100, 2, -1001) is None
        assert router.get_thread_for_window(100, "@5", -1001) is None
        assert router.get_window_for_thread(200, 142, -1001) == "@5"
        assert router.get_thread_for_window(200, "@5", -1001) == 142
        assert "100:2" not in router.group_chat_ids

    def test_from_dict_repairs_cross_user_duplicate_window_deterministically(
        self, router: ThreadRouter
    ) -> None:
        assert (
            router.from_dict(
                {
                    "chat_thread_bindings": {
                        "200:-1001:142": "@5",
                        "100:-1001:2": "@5",
                    }
                }
            )
            is True
        )

        assert router.get_window_for_thread(100, 2, -1001) is None
        assert router.get_window_for_thread(200, 142, -1001) == "@5"
        assert router.get_thread_for_window(200, "@5", -1001) == 142

    def test_same_window_routes_independently_in_different_chats(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 2, "@5", chat_id=-1001)
        router.bind_thread(200, 142, "@5", chat_id=-1002)

        restored = ThreadRouter(
            schedule_save=lambda: None,
            has_window_state=lambda _wid: False,
        )
        restored.from_dict(router.to_dict())

        assert restored.get_window_for_thread(100, 2, -1001) == "@5"
        assert restored.get_window_for_thread(200, 142, -1002) == "@5"


class TestUnbindChatScoped:
    def test_unbind_with_chat_id_removes_only_that_chat(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 7, "@a", chat_id=-1001)
        router.bind_thread(100, 7, "@b", chat_id=-1002)

        assert router.unbind_thread(100, 7, chat_id=-1001) == "@a"
        assert router.get_window_for_chat_thread(-1001, 7) is None
        assert router.get_window_for_chat_thread(-1002, 7) == "@b"

    def test_unbind_with_unknown_chat_id_returns_none(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 7, "@a", chat_id=-1001)
        assert router.unbind_thread(100, 7, chat_id=-9999) is None
        assert router.get_window_for_chat_thread(-1001, 7) == "@a"

    def test_chatless_unbind_infers_the_sole_chat_scoped_binding(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 7, "@a", chat_id=-1001)
        assert router.unbind_thread(100, 7) == "@a"
        assert router.get_window_for_chat_thread(-1001, 7) is None

    def test_chatless_unbind_refuses_when_the_thread_is_ambiguous(
        self, router: ThreadRouter
    ) -> None:
        """Two chats share thread id 7 — unbinding without a chat_id must not
        guess which one the caller meant."""
        router.bind_thread(100, 7, "@a", chat_id=-1001)
        router.bind_thread(100, 7, "@b", chat_id=-1002)

        assert router.unbind_thread(100, 7) is None
        assert router.get_window_for_chat_thread(-1001, 7) == "@a"
        assert router.get_window_for_chat_thread(-1002, 7) == "@b"

    def test_chatless_unbind_prefers_the_legacy_binding(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 7, "@legacy")
        router.bind_thread(100, 7, "@scoped", chat_id=-1001)

        assert router.unbind_thread(100, 7) == "@legacy"
        assert router.get_window_for_chat_thread(-1001, 7) == "@scoped"

    def test_unbinding_a_scoped_thread_keeps_the_users_other_bindings(
        self, router: ThreadRouter
    ) -> None:
        """The user's legacy bindings live in a per-user dict; removing a
        chat-scoped binding must not drop that dict along with it."""
        router.bind_thread(100, 1, "@other")
        router.bind_thread(100, 7, "@scoped", chat_id=-1001)

        assert router.unbind_thread(100, 7) == "@scoped"
        assert router.get_window_for_thread(100, 1) == "@other"
        assert router.get_window_for_chat_thread(-1001, 7) is None


class TestIterThreadBindingsWithChat:
    def test_yields_chat_id_for_scoped_and_legacy_bindings(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 1, "@legacy")
        router.bind_thread(100, 7, "@scoped", chat_id=-1001)

        assert set(router.iter_thread_bindings_with_chat()) == {
            (100, None, 1, "@legacy"),
            (100, -1001, 7, "@scoped"),
        }

    def test_legacy_binding_reports_its_stored_group_chat_id(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 1, "@1")
        router.group_chat_ids["100:1"] = -500

        assert list(router.iter_thread_bindings_with_chat()) == [(100, -500, 1, "@1")]

    def test_empty(self, router: ThreadRouter) -> None:
        assert list(router.iter_thread_bindings_with_chat()) == []


class TestPopDisplayName:
    def test_returns_and_removes_stored_name(self, router: ThreadRouter) -> None:
        router.set_display_name("@1", "proj")
        assert router.pop_display_name("@1") == "proj"
        assert router.get_display_name("@1") == "@1"

    def test_unknown_window_falls_back_to_window_id(self, router: ThreadRouter) -> None:
        assert router.pop_display_name("@missing") == "@missing"

    def test_pop_persists_only_when_a_name_was_removed(self) -> None:
        saves: list[int] = []
        router = ThreadRouter(
            schedule_save=lambda: saves.append(1),
            has_window_state=lambda _wid: False,
        )
        router.set_display_name("@1", "proj")
        saves.clear()

        router.pop_display_name("@missing")
        assert saves == []

        router.pop_display_name("@1")
        assert len(saves) == 1


class TestReset:
    def test_reset_clears_all(self, router: ThreadRouter) -> None:
        router.bind_thread(100, 1, "@1", window_name="proj")
        router.set_group_chat_id(100, 1, -999)
        router.reset()
        assert router.get_window_for_thread(100, 1) is None
        assert router.resolve_chat_id(100, 1) == 100
        assert router.get_display_name("@1") == "@1"
        assert list(router.iter_thread_bindings()) == []


class TestScheduleSave:
    def test_schedule_save_called_on_bind(self, router: ThreadRouter) -> None:
        calls = []
        router._schedule_save = lambda: calls.append(1)
        router.bind_thread(100, 1, "@1")
        assert len(calls) == 1

    def test_schedule_save_called_on_unbind(self, router: ThreadRouter) -> None:
        calls = []
        router.bind_thread(100, 1, "@1")
        router._schedule_save = lambda: calls.append(1)
        router.unbind_thread(100, 1)
        assert len(calls) == 1

    def test_schedule_save_called_on_set_group_chat_id(
        self, router: ThreadRouter
    ) -> None:
        calls = []
        router._schedule_save = lambda: calls.append(1)
        router.set_group_chat_id(100, 1, -999)
        assert len(calls) == 1

    def test_schedule_save_called_on_set_display_name(
        self, router: ThreadRouter
    ) -> None:
        calls = []
        router._schedule_save = lambda: calls.append(1)
        router.set_display_name("@1", "proj")
        assert len(calls) == 1
