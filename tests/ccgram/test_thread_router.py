import pytest

from ccgram.thread_router import _RETIRED_TOPIC_LIMIT, ThreadRouter


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
                retirement_reason="system_replacement",
                cleanup_eligible=True,
            )

        retired = list(router.iter_retired_topics())
        assert len(retired) == _RETIRED_TOPIC_LIMIT
        assert retired[0].thread_id == 3
        assert retired[-1].thread_id == _RETIRED_TOPIC_LIMIT + 2

    def test_chatless_binding_is_not_treated_as_a_known_forum_topic(
        self, router: ThreadRouter
    ) -> None:
        router.bind_thread(100, 42, "@1")
        router.unbind_thread(100, 42, cleanup_eligible=True)

        assert list(router.iter_retired_topics()) == []


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
        router.bind_thread(100, 1, "@1")
        router.set_group_chat_id(100, 1, -999)
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
        router.bind_thread(100, 1, "@1", window_name="proj")
        router.bind_thread(200, 2, "@2")
        router.set_group_chat_id(100, 1, -999)

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
