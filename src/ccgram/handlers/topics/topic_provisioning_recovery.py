"""Recover persisted topic creation after its initiating task has stopped."""

from collections import Counter
from contextlib import suppress
from functools import partial
from pathlib import Path
import time

import structlog
from telegram.error import RetryAfter

from ... import window_query
from ...config import config
from ...topic_selection import select_topic_target
from ...window_state_ports import lifecycle_state
from ...multiplexer import multiplexer
from ...multiplexer.base import canonical_window_id
from ...multiplexer.reconciliation import window_presence
from ...session_protocol.telegram import TelegramSessionAdapter
from ...providers import registry as provider_registry
from ...session import session_manager
from ...telegram_client import TelegramClient
from ...thread_router import ThreadRouter, TopicProvisioning, thread_router
from ..cleanup import clear_topic_state
from .topic_deletion import cleanup_retired_topic
from .topic_orchestration import _window_topic_lock, create_topic_in_chat
from .topic_probe import probe_topic_exists
from .window_launch_service import agent_process_started

logger = structlog.get_logger()


def _target_bound_in_chat(
    router: ThreadRouter,
    claim: TopicProvisioning,
    *,
    exclude_claim_topic: bool = True,
) -> bool:
    """Return whether another current topic already owns this target in chat."""
    assert claim.target_id is not None
    wanted_target = canonical_window_id(claim.target_id)
    for (
        user_id,
        chat_id,
        thread_id,
        window_id,
    ) in router.iter_thread_bindings_with_chat():
        resolved_chat_id = (
            chat_id
            if chat_id is not None
            else router.resolve_chat_id(user_id, thread_id)
        )
        if (
            resolved_chat_id == claim.chat_id
            and canonical_window_id(window_id) == wanted_target
            and (
                not exclude_claim_topic
                or (user_id, thread_id) != (claim.user_id, claim.thread_id)
            )
        ):
            return True
    return False


def _claim_is_current(
    router: ThreadRouter, claim: TopicProvisioning, *, owned: bool = False
) -> bool:
    """Return whether recovery still owns the same durable claim snapshot."""
    current = next(
        (
            item
            for item in router.iter_topic_provisionings()
            if item.claim_id == claim.claim_id
        ),
        None,
    )
    return current == claim and router.owns_topic_provisioning(claim.claim_id) is owned


def _cached_topic_name(router: ThreadRouter, target_id: str) -> str:
    """Choose a recovered topic name from cached window state."""
    view = window_query.view_window(target_id)
    if view is not None:
        if view.window_name:
            return view.window_name
        if view.cwd:
            return Path(view.cwd).name
    return router.get_display_name(target_id) or target_id


def _prepare_recreation(
    router: ThreadRouter, claim: TopicProvisioning
) -> TopicProvisioning | None:
    """Persist an in-flight replacement claim and take runtime ownership."""
    try:
        prepared = router.prepare_topic_recreation(claim.claim_id)
    except KeyError, TypeError, ValueError:
        return None
    try:
        session_manager.flush_state()
    except Exception:  # noqa: BLE001
        try:
            router.defer_topic_recreation(
                prepared.claim_id,
                retry_at=0.0,
                uncertain=False,
            )
        except KeyError, TypeError, ValueError:
            with suppress(KeyError, TypeError, ValueError):
                router.mark_provisioning_uncertain(prepared.claim_id)
        logger.warning(
            "Could not persist topic recreation preparation; released ownership",
            claim_id=prepared.claim_id,
        )
        return None
    return prepared


def _defer_recreation(
    router: ThreadRouter,
    claim: TopicProvisioning,
    *,
    retry_at: float,
    uncertain: bool,
) -> bool:
    """Persist a replacement retry and release runtime ownership."""
    try:
        router.defer_topic_recreation(
            claim.claim_id,
            retry_at=retry_at,
            uncertain=uncertain,
        )
    except KeyError, TypeError, ValueError:
        return False
    session_manager.flush_state()
    return True


def _prepare_and_defer(
    router: ThreadRouter,
    claim: TopicProvisioning,
    *,
    uncertain: bool,
) -> bool:
    """Prepare a replacement claim, then release it as retry state."""
    prepared = _prepare_recreation(router, claim)
    return prepared is not None and _defer_recreation(
        router,
        prepared,
        retry_at=0.0,
        uncertain=uncertain,
    )


def _drop_recreation(
    router: ThreadRouter, claim: TopicProvisioning, *, target_closed: bool
) -> bool:
    """Settle an exact replacement claim after a confirmed terminal outcome."""
    prepared = _prepare_recreation(router, claim)
    if prepared is None:
        return False
    try:
        router.abort_topic_provisioning(
            prepared.claim_id,
            target_confirmed_absent=target_closed,
            topic_confirmed_absent=not target_closed,
        )
    except KeyError, TypeError, ValueError:
        return False
    session_manager.flush_state()
    return True


def _recreation_outcome(
    router: ThreadRouter, claim: TopicProvisioning, *, created: bool
) -> str:
    """Translate creator completion into recovery state."""
    if created:
        return "recreated"
    current = router.get_topic_provisioning(claim.claim_id)
    if current is None:
        return "released"
    return "unresolved" if current.uncertain else "deferred"


async def _claim_agent_is_ready(
    claim: TopicProvisioning, adapter: TelegramSessionAdapter
) -> bool:
    """Require current provider evidence before recovering an agent binding."""
    assert claim.target_id is not None
    provider_name = window_query.get_window_provider(claim.target_id)
    if not provider_name:
        return False
    if provider_registry.get(provider_name).capabilities.chat_first_command_path:
        return True
    return (
        await agent_process_started(
            claim.target_id, adapter, expected_provider=provider_name
        )
        is True
    )


async def _recreate_deleted_topic(
    client: TelegramClient,
    router: ThreadRouter,
    backend: object | None,
    claim: TopicProvisioning,
    target_id: str,
    topic_name: str,
    adapter: TelegramSessionAdapter,
) -> str:
    """Recreate a deleted topic through the original durable claim."""
    assert claim.target_id is not None
    dead_window = (
        router.get_window_for_thread(claim.user_id, claim.thread_id, claim.chat_id)
        if claim.thread_id is not None
        else None
    )
    async with _window_topic_lock(canonical_window_id(target_id)):
        presence = await window_presence(target_id, backend)
        if not _claim_is_current(router, claim):
            return "changed"
        if presence is None:
            return (
                "unresolved"
                if _prepare_and_defer(router, claim, uncertain=True)
                else "changed"
            )
        if not presence:
            return (
                "released"
                if _drop_recreation(router, claim, target_closed=True)
                else "changed"
            )
        if _target_bound_in_chat(router, claim, exclude_claim_topic=False):
            return (
                "released"
                if _drop_recreation(router, claim, target_closed=False)
                else "changed"
            )
        ready = await _claim_agent_is_ready(claim, adapter)
        if not _claim_is_current(router, claim):
            return "changed"
        if not ready:
            return "unresolved"
        prepared, outcome = await _prepare_selected_recreation(router, claim, adapter)
        if prepared is None:
            return outcome
        return await _create_prepared_replacement(
            client, router, prepared, target_id, topic_name, dead_window
        )


async def _prepare_selected_recreation(
    router: ThreadRouter, claim: TopicProvisioning, adapter: TelegramSessionAdapter
) -> tuple[TopicProvisioning | None, str]:
    """Select the explicit target before taking replacement ownership."""
    if not await _select_recovered_target(claim, adapter):
        return None, "unresolved"
    if not _claim_is_current(router, claim):
        return None, "changed"
    return _prepare_recreation(router, claim), "changed"


async def _create_prepared_replacement(
    client: TelegramClient,
    router: ThreadRouter,
    prepared: TopicProvisioning,
    target_id: str,
    topic_name: str,
    dead_window: str | None,
) -> str:
    """Finish remote creation under the caller's target lock and ownership."""
    if dead_window is not None:
        assert prepared.retry_thread_id is not None
        await clear_topic_state(
            prepared.user_id,
            prepared.retry_thread_id,
            client=client,
            window_id=dead_window,
            chat_id=prepared.chat_id,
            window_dead=False,
        )
        if not _claim_is_current(router, prepared, owned=True):
            return "changed"
    try:
        created = await create_topic_in_chat(
            client,
            prepared.chat_id,
            target_id,
            topic_name,
            user_id=prepared.user_id,
            propagate_retry_after=True,
            claim_id=prepared.claim_id,
        )
    except RetryAfter:
        return "rate_limited"
    session_manager.flush_state()
    return _recreation_outcome(router, prepared, created=created)


async def _select_recovered_target(
    claim: TopicProvisioning, adapter: TelegramSessionAdapter
) -> bool:
    """Recover explicit creation selection, not automatic/admitted topics."""
    assert claim.target_id is not None
    if not await _claim_agent_is_ready(claim, adapter):
        return False
    saved_owner = lifecycle_state.get_topic_mark_owner(claim.target_id)
    window = await multiplexer.find_window_by_id(claim.target_id)
    if window is None:
        return False
    observed_owner = window.topic_mark_owner
    owner = observed_owner if observed_owner is not None else saved_owner
    require_selection = (
        config.topic_discovery_mode == "marked" or claim.kind == "target_for_topic"
    )
    if (
        (
            saved_owner is not None
            and observed_owner is not None
            and observed_owner != saved_owner
        )
        or (
            require_selection
            and (
                lifecycle_state.is_topic_detach_pending(claim.target_id, owner=owner)
                or lifecycle_state.is_topic_detached(claim.target_id)
            )
        )
        or (config.topic_discovery_mode == "marked" and saved_owner is None)
    ):
        return False
    if claim.kind != "target_for_topic":
        if config.topic_discovery_mode == "marked" and (
            window is None
            or not window.topic_markable
            or window.topic_marked is not True
            or not observed_owner
            or saved_owner != observed_owner
        ):
            return False
        return (
            lifecycle_state.record_topic_mark_owner(claim.target_id, observed_owner)
            or config.topic_discovery_mode != "marked"
        )
    if await select_topic_target(claim.target_id):
        return await _claim_agent_is_ready(
            claim, adapter
        ) and not lifecycle_state.is_topic_detach_pending(claim.target_id, owner=owner)
    logger.warning("recovered_topic_selection_pending", window_id=claim.target_id)
    return (
        config.topic_discovery_mode != "marked"
        and not lifecycle_state.is_topic_detach_pending(claim.target_id, owner=owner)
        and not lifecycle_state.is_topic_detached(claim.target_id)
    )


async def _commit_present_topic(
    router: ThreadRouter, claim: TopicProvisioning, adapter: TelegramSessionAdapter
) -> str:
    """Commit a proven-live topic without evicting a concurrent target bind."""
    assert claim.target_id is not None
    async with _window_topic_lock(canonical_window_id(claim.target_id)):
        if not _claim_is_current(router, claim):
            return "changed"
        if _target_bound_in_chat(router, claim):
            return "unresolved"
        if not await _select_recovered_target(claim, adapter):
            return "unresolved"
        if not _claim_is_current(router, claim):
            return "changed"
        if _target_bound_in_chat(router, claim):
            return "unresolved"
        committed = router.commit_topic_provisioning(claim.claim_id)
        session_manager.flush_state()
        return "bound" if committed else "changed"


async def _recover_present_topic(
    client: TelegramClient,
    router: ThreadRouter,
    backend: object | None,
    claim: TopicProvisioning,
    adapter: TelegramSessionAdapter,
) -> str:
    assert claim.target_id is not None and claim.thread_id is not None
    try:
        topic_exists = await probe_topic_exists(
            client,
            claim.chat_id,
            claim.thread_id,
            propagate_retry_after=True,
            topic_name=_cached_topic_name(router, claim.target_id),
        )
    except RetryAfter:
        return "rate_limited"
    if not _claim_is_current(router, claim):
        return "changed"
    if topic_exists is None:
        return "unresolved"
    if topic_exists:
        if not await _claim_agent_is_ready(claim, adapter):
            return "unresolved"
        return await _commit_present_topic(router, claim, adapter)

    topic_name = _cached_topic_name(router, claim.target_id)
    return await _recreate_deleted_topic(
        client,
        router,
        backend,
        claim,
        claim.target_id,
        topic_name,
        adapter,
    )


async def _recover_retry_topic(
    client: TelegramClient,
    router: ThreadRouter,
    backend: object | None,
    claim: TopicProvisioning,
    adapter: TelegramSessionAdapter,
) -> str:
    """Resume a persisted replacement claim without a new window event."""
    assert claim.target_id is not None and claim.retry_thread_id is not None
    target_id = window_query.resolve_window_alias(claim.target_id) or claim.target_id
    if target_id != claim.target_id:
        claim = router.attach_provisioning_target(claim.claim_id, target_id)
        session_manager.flush_state()
    if not _claim_is_current(router, claim):
        return "changed"
    return await _recreate_deleted_topic(
        client,
        router,
        backend,
        claim,
        target_id,
        _cached_topic_name(router, target_id),
        adapter,
    )


async def _recover_absent_target(
    client: TelegramClient,
    router: ThreadRouter,
    target_id: str,
    claim: TopicProvisioning,
) -> str:
    """Release and clean up a claim whose terminal window is confirmed gone."""
    assert claim.thread_id is not None
    router.abort_topic_provisioning(claim.claim_id, target_confirmed_absent=True)
    session_manager.flush_state()
    retired = next(
        (
            topic
            for topic in router.iter_retired_topics()
            if topic.chat_id == claim.chat_id and topic.thread_id == claim.thread_id
        ),
        None,
    )
    if retired is None:
        return "released"
    return await cleanup_retired_topic(
        client,
        retired,
        router=router,
        before_delete=partial(
            clear_topic_state,
            retired.user_id,
            retired.thread_id,
            client=client,
            window_id=target_id,
            chat_id=retired.chat_id,
            window_dead=True,
        ),
    )


async def _recover_known_topic(
    client: TelegramClient,
    router: ThreadRouter,
    backend: object | None,
    claim: TopicProvisioning,
    adapter: TelegramSessionAdapter,
) -> str:
    assert claim.target_id is not None and claim.thread_id is not None
    target_id = window_query.resolve_window_alias(claim.target_id) or claim.target_id
    if target_id != claim.target_id:
        claim = router.attach_provisioning_target(claim.claim_id, target_id)
        session_manager.flush_state()
    presence = await window_presence(target_id, backend)
    if not _claim_is_current(router, claim):
        return "changed"
    if presence is None:
        return "unresolved"
    if presence:
        return await _recover_present_topic(client, router, backend, claim, adapter)
    return await _recover_absent_target(client, router, target_id, claim)


async def _recover_claim(
    client: TelegramClient,
    router: ThreadRouter,
    backend: object | None,
    claim: TopicProvisioning,
    adapter: TelegramSessionAdapter,
) -> str:
    """Resolve one unowned claim according to its durable state."""
    if claim.thread_id is not None:
        return await _recover_known_topic(client, router, backend, claim, adapter)
    if claim.retry_thread_id is None:
        return "unresolved"
    if claim.uncertain:
        return "unresolved"
    if claim.retry_at > time.time():
        return "deferred"
    return await _recover_retry_topic(client, router, backend, claim, adapter)


async def recover_topic_provisioning(
    client: TelegramClient,
    *,
    router: ThreadRouter = thread_router,
    backend: object | None = None,
    adapter: TelegramSessionAdapter,
    limit: int = 20,
) -> dict[str, int]:
    """Resolve abandoned claims from current evidence, never from their age."""
    outcomes: Counter[str] = Counter()
    probes = 0
    for claim in router.iter_topic_provisionings():
        if router.owns_topic_provisioning(claim.claim_id):
            continue
        if claim.target_id is None:
            outcomes["unresolved"] += 1
            continue
        if probes >= limit:
            break
        if claim.thread_id is None and claim.retry_thread_id is None:
            outcomes["unresolved"] += 1
            continue
        if claim.thread_id is None and claim.uncertain:
            outcomes["unresolved"] += 1
            continue
        if (
            claim.thread_id is None
            and not claim.uncertain
            and claim.retry_at > time.time()
        ):
            outcomes["deferred"] += 1
            continue
        probes += 1
        outcome = await _recover_claim(client, router, backend, claim, adapter)
        outcomes[outcome] += 1
        if outcome == "rate_limited":
            break
    if any(key != "unresolved" for key in outcomes):
        logger.info("topic_provisioning_recovered", outcomes=dict(outcomes))
    return dict(outcomes)
