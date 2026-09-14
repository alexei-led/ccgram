"""Recover persisted topic creation after its initiating task has stopped."""

from collections import Counter
from functools import partial
from pathlib import Path

import structlog
from telegram.error import RetryAfter

from ... import window_query
from ...multiplexer.base import canonical_window_id
from ...multiplexer.reconciliation import window_presence
from ...session import session_manager
from ...telegram_client import TelegramClient
from ...thread_router import ThreadRouter, TopicProvisioning, thread_router
from ..cleanup import clear_topic_state
from .topic_deletion import cleanup_retired_topic
from .topic_orchestration import _window_topic_lock, create_topic_in_chat
from .topic_probe import probe_topic_exists

logger = structlog.get_logger()


def _target_bound_in_chat(
    router: ThreadRouter,
    claim: TopicProvisioning,
    *,
    exclude_claim_topic: bool = True,
) -> bool:
    """Return whether another current topic already owns this target in chat."""
    assert claim.target_id is not None and claim.thread_id is not None
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


def _claim_is_current(router: ThreadRouter, claim: TopicProvisioning) -> bool:
    """Return whether recovery still owns the same durable claim snapshot."""
    current = next(
        (
            item
            for item in router.iter_topic_provisionings()
            if item.claim_id == claim.claim_id
        ),
        None,
    )
    return current == claim and not router.owns_topic_provisioning(claim.claim_id)


def _cached_topic_name(router: ThreadRouter, target_id: str) -> str:
    """Choose a recovered topic name from cached window state."""
    view = window_query.view_window(target_id)
    if view is not None:
        if view.window_name:
            return view.window_name
        if view.cwd:
            return Path(view.cwd).name
    return router.get_display_name(target_id) or target_id


async def _recreate_deleted_topic(
    client: TelegramClient,
    router: ThreadRouter,
    backend: object | None,
    claim: TopicProvisioning,
    target_id: str,
    topic_name: str,
) -> str:
    """Recreate a deleted topic only after a locked, fresh target check."""
    assert claim.thread_id is not None
    async with _window_topic_lock(canonical_window_id(target_id)):
        presence = await window_presence(target_id, backend)
        if presence is None:
            return "unresolved"
        if not presence:
            return "released"
        if _target_bound_in_chat(router, claim, exclude_claim_topic=False):
            return "released"
        try:
            created = await create_topic_in_chat(
                client,
                claim.chat_id,
                target_id,
                topic_name,
                user_id=claim.user_id,
                propagate_retry_after=True,
            )
        except RetryAfter:
            return "rate_limited"
        session_manager.flush_state()
        return "recreated" if created else "released"


async def _commit_present_topic(router: ThreadRouter, claim: TopicProvisioning) -> str:
    """Commit a proven-live topic without evicting a concurrent target bind."""
    assert claim.target_id is not None
    async with _window_topic_lock(canonical_window_id(claim.target_id)):
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
) -> str:
    assert claim.target_id is not None and claim.thread_id is not None
    try:
        topic_exists = await probe_topic_exists(
            client,
            claim.chat_id,
            claim.thread_id,
            propagate_retry_after=True,
        )
    except RetryAfter:
        return "rate_limited"
    if not _claim_is_current(router, claim):
        return "changed"
    if topic_exists is None:
        return "unresolved"
    if topic_exists:
        return await _commit_present_topic(router, claim)

    dead_window = router.get_window_for_thread(
        claim.user_id, claim.thread_id, claim.chat_id
    )
    topic_name = _cached_topic_name(router, claim.target_id)
    aborted = router.abort_topic_provisioning(
        claim.claim_id,
        target_confirmed_absent=False,
        topic_confirmed_absent=True,
    )
    if aborted is None:
        return "changed"
    session_manager.flush_state()
    if dead_window is not None:
        await clear_topic_state(
            claim.user_id,
            claim.thread_id,
            client=client,
            window_id=dead_window,
            chat_id=claim.chat_id,
            window_dead=False,
        )
    return await _recreate_deleted_topic(
        client,
        router,
        backend,
        claim,
        claim.target_id,
        topic_name,
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
        return await _recover_present_topic(client, router, backend, claim)
    return await _recover_absent_target(client, router, target_id, claim)


async def recover_topic_provisioning(
    client: TelegramClient,
    *,
    router: ThreadRouter = thread_router,
    backend: object | None = None,
    limit: int = 20,
) -> dict[str, int]:
    """Resolve abandoned claims from current evidence, never from their age."""
    outcomes: Counter[str] = Counter()
    probes = 0
    for claim in router.iter_topic_provisionings():
        if router.owns_topic_provisioning(claim.claim_id):
            continue
        if claim.target_id is None or claim.thread_id is None:
            outcomes["unresolved"] += 1
            continue
        if probes >= limit:
            break
        probes += 1
        outcome = await _recover_known_topic(client, router, backend, claim)
        outcomes[outcome] += 1
        if outcome == "rate_limited":
            break
    if any(key != "unresolved" for key in outcomes):
        logger.info("topic_provisioning_recovered", outcomes=dict(outcomes))
    return dict(outcomes)
