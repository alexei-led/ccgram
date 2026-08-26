import ast
import dataclasses
from pathlib import Path

import pytest

from ccgram.handlers.messaging_pipeline.message_task import (
    ContentTask,
    MessageTask,
    StatusClearTask,
    StatusUpdateTask,
    thread_key,
)


@pytest.mark.parametrize(
    "task",
    [
        ContentTask(window_id="@0", parts=("hello",)),
        StatusUpdateTask(window_id="@0", text="working..."),
        StatusClearTask(window_id="@0"),
    ],
    ids=["content", "status_update", "status_clear"],
)
def test_tasks_are_frozen(task: MessageTask):
    with pytest.raises(dataclasses.FrozenInstanceError):
        task.window_id = "@1"  # type: ignore[misc]


class TestContentTask:
    def test_defaults(self):
        task = ContentTask(window_id="@0", parts=("x",))
        assert task.content_type == "text"
        assert task.role == "assistant"
        assert task.tool_use_id is None
        assert task.tool_name is None
        assert task.thread_id is None

    def test_role_can_be_set_to_user(self):
        task = ContentTask(window_id="@0", parts=("x",), role="user")
        assert task.role == "user"

    def test_tool_use_fields(self):
        task = ContentTask(
            window_id="@0",
            parts=("result",),
            content_type="tool_result",
            tool_use_id="tu_123",
            tool_name="Read",
        )
        assert task.content_type == "tool_result"
        assert task.tool_use_id == "tu_123"
        assert task.tool_name == "Read"

    def test_hashable(self):
        task = ContentTask(window_id="@0", parts=("hello",))
        assert {task: 1}[task] == 1


class TestStatusUpdateTask:
    def test_defaults(self):
        task = StatusUpdateTask(window_id="@0", text="ok")
        assert task.thread_id is None

    def test_optional_text(self):
        task = StatusUpdateTask(window_id="@0", text=None)
        assert task.text is None


class TestStatusClearTask:
    def test_defaults(self):
        task = StatusClearTask(window_id="@0")
        assert task.thread_id is None

    def test_optional_window_id(self):
        task = StatusClearTask(window_id=None)
        assert task.window_id is None


class TestMessageTaskUnion:
    def test_union_covers_all_variants(self):
        args = set(MessageTask.__args__)
        assert args == {ContentTask, StatusUpdateTask, StatusClearTask}


class TestThreadKey:
    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (None, 0),
            (0, 0),
            (42, 42),
            (1, 1),
        ],
    )
    def test_normalises_thread_id(self, input_val, expected):
        assert thread_key(input_val) == expected


class TestModuleImports:
    def test_imports_nothing_from_handlers(self):
        src = Path("src/ccgram/handlers/messaging_pipeline/message_task.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module and node.module.startswith("ccgram.handlers"):
                raise AssertionError(f"forbidden import: from {node.module}")
            if node.level and node.level > 0:
                mod = node.module or ""
                raise AssertionError(
                    f"forbidden relative import: from {'.' * node.level}{mod}"
                )
