"""Workspace → chat routing of auto-created topics (CCGRAM_TOPIC_CHAT_ROUTES)."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.config import parse_topic_chat_routes
from ccgram.handlers.topics import topic_routes
from ccgram.handlers.topics.topic_orchestration import _handle_new_window_locked
from ccgram.handlers.topics.topic_routes import (
    relocate_misrouted_topics,
    route_target_chats,
)
from ccgram.monitor_events import NewWindowEvent
from ccgram.multiplexer.base import WindowRef, WorkspaceRef

PRIVATE = 1644026884
DIGIENTS = -1001
TABULA = -1002
ROUTES = {"Digients": DIGIENTS, "Tabula": TABULA}


class TestParse:
    def test_pairs(self) -> None:
        assert parse_topic_chat_routes(" Digients=-1001 , Tabula = -1002 ") == ROUTES

    def test_empty(self) -> None:
        assert parse_topic_chat_routes("") == {}

    def test_name_may_hold_equals_sign(self) -> None:
        assert parse_topic_chat_routes("a=b=-5") == {"a=b": -5}

    @pytest.mark.parametrize(
        "raw", ["Digients", "=-1001", "Digients=abc", "Digients=1001"]
    )
    def test_malformed_entries_are_skipped(self, raw: str) -> None:
        assert parse_topic_chat_routes(f"{raw},Tabula=-1002") == {"Tabula": TABULA}


class TestRouteTargetChats:
    def test_no_routes_keeps_the_usual_chats(self) -> None:
        assert route_target_chats({PRIVATE, DIGIENTS}, "Digients", {}) == {
            PRIVATE,
            DIGIENTS,
        }

    def test_routed_workspace_goes_to_its_chat_only(self) -> None:
        # Even when that group has never been seen before.
        assert route_target_chats({PRIVATE}, "Digients", ROUTES) == {DIGIENTS}

    def test_unrouted_workspace_stays_out_of_routed_groups(self) -> None:
        assert route_target_chats({PRIVATE, DIGIENTS, TABULA}, "Trading", ROUTES) == {
            PRIVATE
        }

    def test_unknown_workspace_counts_as_unrouted(self) -> None:
        assert route_target_chats({PRIVATE, DIGIENTS}, "", ROUTES) == {PRIVATE}


def _event(window_id: str = "hc-t9") -> NewWindowEvent:
    return NewWindowEvent(
        window_id=window_id, session_id="s", window_name="cua", cwd="/tmp"
    )


class TestNewWindowRouting:
    async def _created_in(self, workspace: str, routes: dict[str, int]) -> list[int]:
        create = AsyncMock(return_value=True)
        mod = "ccgram.handlers.topics.topic_orchestration"
        with (
            patch(f"{mod}._is_window_already_bound", return_value=False),
            patch(f"{mod}._is_pending_user_creation", return_value=False),
            patch(f"{mod}._auto_detect_provider", AsyncMock()),
            patch(f"{mod}.collect_target_chats", return_value={PRIVATE, TABULA}),
            patch(f"{mod}.configured_routes", return_value=routes),
            patch(f"{mod}.window_workspace", AsyncMock(return_value=workspace)),
            patch(f"{mod}.create_topic_in_chat", create),
        ):
            assert await _handle_new_window_locked(_event(), AsyncMock()) is True
        return sorted(call.args[1] for call in create.await_args_list)

    async def test_routed_session_gets_a_topic_in_its_group(self) -> None:
        assert await self._created_in("Digients", ROUTES) == [DIGIENTS]

    async def test_unrouted_session_stays_in_the_private_chat(self) -> None:
        assert await self._created_in("Trading", ROUTES) == [PRIVATE]

    async def test_without_routes_every_seen_chat_is_used(self) -> None:
        assert await self._created_in("Digients", {}) == [TABULA, PRIVATE]


def _window(window_id: str, workspace: str, *, eligible: bool = True) -> WindowRef:
    return WindowRef(
        window_id=window_id,
        window_name=f"name-{window_id}",
        cwd="/tmp",
        topic_eligible=eligible,
        workspace=workspace,
    )


class _Relocation:
    """Patches for relocate_misrouted_topics, recording create/retire order."""

    def __init__(
        self,
        windows: list[WindowRef] | None,
        bindings: Sequence[tuple[int, int | None, int, str]],
        *,
        routes: dict[str, int] = ROUTES,
        create_ok: bool = True,
    ) -> None:
        self.events: list[tuple[str, str, int]] = []
        self.windows = windows
        self.bindings = bindings
        self.routes = routes
        self.create_ok = create_ok

    async def _create(
        self, _client: object, chat_id: int, window_id: str, _name: str
    ) -> bool:
        self.events.append(("create", window_id, chat_id))
        return self.create_ok

    async def _retire(
        self, _client: object, _uid: int, _tid: int, window_id: str, **kwargs: object
    ) -> str:
        assert kwargs["retirement_reason"] == "relocated"
        chat_id = kwargs["chat_id"]
        assert isinstance(chat_id, int)
        self.events.append(("retire", window_id, chat_id))
        return "deleted"

    async def run(self) -> int:
        with (
            patch.object(topic_routes, "configured_routes", return_value=self.routes),
            patch(
                "ccgram.multiplexer.reconciliation.list_windows_for_reconciliation",
                AsyncMock(return_value=self.windows),
            ),
            patch.object(
                topic_routes.session_query,
                "iter_bound_topics",
                return_value=self.bindings,
            ),
            patch.object(topic_routes.window_query, "view_window", return_value=None),
            patch.object(topic_routes, "session_manager", MagicMock()),
            patch(
                "ccgram.handlers.topics.topic_orchestration.collect_target_chats",
                return_value={PRIVATE},
            ),
            patch(
                "ccgram.handlers.topics.topic_orchestration.create_topic_in_chat",
                side_effect=self._create,
            ),
            patch(
                "ccgram.handlers.topics.topic_routes.retire_topic_binding",
                side_effect=self._retire,
            ),
        ):
            return await relocate_misrouted_topics(AsyncMock())


class TestRelocation:
    async def test_moves_a_private_topic_into_its_group_create_first(self) -> None:
        reloc = _Relocation(
            [_window("hc-t1", "Digients")], [(PRIVATE, PRIVATE, 11, "hc-t1")]
        )
        assert await reloc.run() == 1
        assert reloc.events == [
            ("create", "hc-t1", DIGIENTS),
            ("retire", "hc-t1", PRIVATE),
        ]

    async def test_keeps_the_old_topic_when_creation_fails(self) -> None:
        reloc = _Relocation(
            [_window("hc-t1", "Digients")],
            [(PRIVATE, PRIVATE, 11, "hc-t1")],
            create_ok=False,
        )
        assert await reloc.run() == 0
        assert reloc.events == [("create", "hc-t1", DIGIENTS)]

    async def test_tab_moved_out_of_a_routed_space_goes_back_to_private(self) -> None:
        reloc = _Relocation(
            [_window("hc-t1", "Trading")], [(PRIVATE, DIGIENTS, 11, "hc-t1")]
        )
        assert await reloc.run() == 1
        assert reloc.events == [
            ("create", "hc-t1", PRIVATE),
            ("retire", "hc-t1", DIGIENTS),
        ]

    async def test_extra_binding_is_retired_without_creating(self) -> None:
        reloc = _Relocation(
            [_window("hc-t1", "Digients")],
            [(PRIVATE, DIGIENTS, 5, "hc-t1"), (PRIVATE, PRIVATE, 11, "hc-t1")],
        )
        assert await reloc.run() == 1
        assert reloc.events == [("retire", "hc-t1", PRIVATE)]

    async def test_correctly_placed_and_ineligible_windows_are_left_alone(
        self,
    ) -> None:
        reloc = _Relocation(
            [
                _window("hc-t1", "Digients"),
                _window("hc-t2", "Trading"),
                _window("hc-t3", "Digients", eligible=False),
            ],
            [
                (PRIVATE, DIGIENTS, 5, "hc-t1"),
                (PRIVATE, PRIVATE, 6, "hc-t2"),
                (PRIVATE, PRIVATE, 7, "hc-t3"),
            ],
        )
        assert await reloc.run() == 0
        assert reloc.events == []

    async def test_no_routes_or_no_listing_does_nothing(self) -> None:
        bindings = [(PRIVATE, PRIVATE, 11, "hc-t1")]
        windows = [_window("hc-t1", "Digients")]
        assert await _Relocation(windows, bindings, routes={}).run() == 0
        assert await _Relocation(None, bindings).run() == 0


class TestGroupJoinLog:
    @staticmethod
    def _update(chat_type: str, *, migrate_to: int | None = None) -> MagicMock:
        update = MagicMock()
        msg = update.effective_message
        msg.chat.type = chat_type
        msg.chat.id = -555
        msg.chat.title = "Digients"
        msg.chat.is_forum = False
        msg.migrate_to_chat_id = migrate_to
        update.effective_user.id = PRIVATE
        return update

    async def test_logs_the_group_id_and_its_supergroup_successor(self) -> None:
        with patch.object(topic_routes, "logger") as log:
            await topic_routes.log_group_join(
                self._update("group", migrate_to=-100555), MagicMock()
            )
        kwargs = log.info.call_args.kwargs
        assert kwargs["chat_id"] == -555
        assert kwargs["migrated_to"] == -100555
        assert kwargs["title"] == "Digients"

    async def test_private_chats_are_ignored(self) -> None:
        with patch.object(topic_routes, "logger") as log:
            await topic_routes.log_group_join(self._update("private"), MagicMock())
        log.info.assert_not_called()


class TestRoutedWorkspace:
    def test_label_for_a_routed_group(self) -> None:
        assert topic_routes.routed_label(DIGIENTS, ROUTES) == "Digients"
        assert topic_routes.routed_label(PRIVATE, ROUTES) is None
        assert topic_routes.routed_label(None, ROUTES) is None

    async def _resolve(
        self, chat_id: int, *, selectable: bool = True
    ) -> WorkspaceRef | None:
        mux = MagicMock()
        mux.capabilities.supports_workspace_selection = selectable
        mux.list_workspaces = AsyncMock(
            return_value=[
                WorkspaceRef("sp-t", "Tabula", "/r/tabula"),
                WorkspaceRef("sp-d", "Digients", "/r/digients"),
            ]
        )
        with (
            patch.object(topic_routes, "configured_routes", return_value=ROUTES),
            patch.object(topic_routes, "multiplexer", mux),
        ):
            return await topic_routes.routed_workspace(chat_id)

    async def test_a_routed_group_resolves_to_its_workspace(self) -> None:
        workspace = await self._resolve(DIGIENTS)
        assert workspace is not None
        assert (workspace.workspace_id, workspace.cwd) == ("sp-d", "/r/digients")

    async def test_unrouted_chats_and_unselectable_backends_resolve_to_none(
        self,
    ) -> None:
        assert await self._resolve(PRIVATE) is None
        assert await self._resolve(DIGIENTS, selectable=False) is None
