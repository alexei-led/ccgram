"""Recover persisted topic creation after its initiating task has stopped."""

from collections import Counter
from functools import partial

import structlog

from ... import window_query
from ...multiplexer.reconciliation import window_presence
from ...session import session_manager
from ...telegram_client import TelegramClient
from ...thread_router import ThreadRouter, TopicProvisioning, thread_router
from ..cleanup import clear_topic_state
from .topic_deletion import cleanup_retired_topic

logger = structlog.get_logger()


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
    current = next(
        (
            item
            for item in router.iter_topic_provisionings()
            if item.claim_id == claim.claim_id
        ),
        None,
    )
    if current != claim or router.owns_topic_provisioning(claim.claim_id):
        return "changed"
    if presence is None:
        return "unresolved"
    if presence:
        committed = router.commit_topic_provisioning(claim.claim_id)
        session_manager.flush_state()
        return "bound" if committed else "changed"
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
