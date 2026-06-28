"""Property-based tests for message queue merge logic."""

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from ccgram.handlers.messaging_pipeline.message_queue import (
    MERGE_MAX_LENGTH,
    _can_merge_tasks,
    _coalesce_status_updates,
    _merge_content_tasks,
)
from ccgram.handlers.messaging_pipeline.message_task import (
    ContentTask,
    ContentType,
    MessageTask,
    StatusUpdateTask,
)

# --- Strategies ---

_mergeable_types: st.SearchStrategy[ContentType] = st.sampled_from(["text"])
_unmergeable_types: st.SearchStrategy[ContentType] = st.sampled_from(
    ["tool_use", "tool_result"]
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
    )


# --- _can_merge_tasks unit properties ---


@given(ct=_mergeable_types)
def test_same_window_mergeable_types_can_merge(ct: ContentType) -> None:
    base = _content_task(content_type=ct)
    candidate = _content_task(content_type=ct)
    assert _can_merge_tasks(base, candidate) is True


@given(base_ct=_unmergeable_types, cand_ct=_all_content_types)
def test_unmergeable_base_blocks_merge(
    base_ct: ContentType, cand_ct: ContentType
) -> None:
    base = _content_task(content_type=base_ct)
    candidate = _content_task(content_type=cand_ct)
    assert _can_merge_tasks(base, candidate) is False


@given(cand_ct=_unmergeable_types)
def test_unmergeable_candidate_blocks_merge(cand_ct: ContentType) -> None:
    base = _content_task(content_type="text")
    candidate = _content_task(content_type=cand_ct)
    assert _can_merge_tasks(base, candidate) is False


@given(w1=_window_ids, w2=_window_ids)
def test_different_windows_block_merge(w1: str, w2: str) -> None:
    if w1 == w2:
        return
    base = _content_task(window_id=w1)
    candidate = _content_task(window_id=w2)
    assert _can_merge_tasks(base, candidate) is False


# --- _merge_content_tasks property tests ---


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
    total = sum(len(p) for p in merged.parts)
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


# --- Edge case tests ---


async def test_all_unmergeable_no_merge() -> None:
    first = _content_task(parts=("hello",))
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    lock = asyncio.Lock()

    for _ in range(3):
        await queue.put(_content_task(content_type="tool_use", parts=("tool",)))

    merged, count = await _merge_content_tasks(queue, first, lock)
    assert count == 0
    assert queue.qsize() == 3


async def test_exact_boundary() -> None:
    half = "x" * (MERGE_MAX_LENGTH // 2)
    first = _content_task(parts=(half,))
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    lock = asyncio.Lock()

    await queue.put(_content_task(parts=(half,)))
    await queue.put(_content_task(parts=("overflow",)))

    merged, count = await _merge_content_tasks(queue, first, lock)
    total = sum(len(p) for p in merged.parts)
    assert total <= MERGE_MAX_LENGTH
    assert count == 1
    assert queue.qsize() == 1


async def test_status_coalesce_keeps_latest_same_window_thread() -> None:
    first = StatusUpdateTask(text="old", window_id="@1", thread_id=10)
    queue: asyncio.Queue[MessageTask] = asyncio.Queue()
    lock = asyncio.Lock()

    await queue.put(StatusUpdateTask(text="mid", window_id="@1", thread_id=10))
    await queue.put(StatusUpdateTask(text="new", window_id="@1", thread_id=10))
    await queue.put(StatusUpdateTask(text="other-window", window_id="@2", thread_id=10))

    selected, dropped = await _coalesce_status_updates(queue, first, lock)
    assert selected.text == "new"
    assert dropped == 2
    assert queue.qsize() == 1
    remaining = queue.get_nowait()
    assert isinstance(remaining, StatusUpdateTask)
    assert remaining.text == "other-window"
