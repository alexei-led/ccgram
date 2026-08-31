"""Property-based tests for message queue merge logic.

Concrete input/output cases live in ``test_message_queue.py``; this file only
covers invariants that must hold for *any* sequence of queued tasks.
"""

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from ccgram.handlers.messaging_pipeline.message_queue import (
    MERGE_MAX_LENGTH,
    _merge_content_tasks,
)
from ccgram.handlers.messaging_pipeline.message_task import (
    ContentTask,
    ContentType,
    MessageTask,
)

_all_content_types: st.SearchStrategy[ContentType] = st.sampled_from(
    ["text", "tool_use", "tool_result"]
)
_window_ids = st.sampled_from(["@0", "@1", "@2", "@3"])


def _content_task(
    window_id: str = "@0",
    parts: tuple[str, ...] | None = None,
    content_type: ContentType = "text",
) -> ContentTask:
    return ContentTask(
        window_id=window_id,
        parts=parts or ("hello",),
        content_type=content_type,
        chat_id=-1001,
    )


@given(
    parts_list=st.lists(
        st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(categories=("L", "N", "P", "Z")),
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=50)
async def test_merged_length_never_exceeds_limit(parts_list: list[str]) -> None:
    first = _content_task(parts=(parts_list[0],))
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    lock = asyncio.Lock()
    for p in parts_list[1:]:
        await queue.put(_content_task(parts=(p,)))

    merged, _count = await _merge_content_tasks(queue, first, lock)
    total = sum(len(p) for p in merged.parts) + 2 * (len(merged.parts) - 1)
    assert total <= MERGE_MAX_LENGTH


@given(
    n_tasks=st.integers(min_value=1, max_value=8),
    content_types=st.lists(_all_content_types, min_size=8, max_size=8),
)
@settings(max_examples=50)
async def test_fifo_order_preserved(
    n_tasks: int, content_types: list[ContentType]
) -> None:
    types = content_types[:n_tasks]
    all_parts: list[str] = []
    tasks: list[ContentTask] = []
    for i, ct in enumerate(types):
        part = f"msg-{i}"
        all_parts.append(part)
        tasks.append(_content_task(parts=(part,), content_type=ct))

    first = tasks[0]
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    lock = asyncio.Lock()
    for t in tasks[1:]:
        await queue.put(t)

    merged, _count = await _merge_content_tasks(queue, first, lock)

    assert list(all_parts[: len(merged.parts)]) == list(merged.parts)

    remaining_parts: list[str] = []
    while not queue.empty():
        t = queue.get_nowait()
        assert isinstance(t, ContentTask)
        remaining_parts.extend(t.parts)
    assert list(merged.parts) + remaining_parts == all_parts


@given(
    window_ids=st.lists(
        _window_ids,
        min_size=2,
        max_size=6,
    )
)
@settings(max_examples=50)
async def test_different_window_breaks_chain(window_ids: list[str]) -> None:
    first = _content_task(window_id=window_ids[0])
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    lock = asyncio.Lock()
    for wid in window_ids[1:]:
        await queue.put(_content_task(window_id=wid))

    merged, count = await _merge_content_tasks(queue, first, lock)

    expected_merges = 0
    for wid in window_ids[1:]:
        if wid != window_ids[0]:
            break
        expected_merges += 1
    assert count == expected_merges
