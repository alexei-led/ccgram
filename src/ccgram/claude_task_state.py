"""Claude task tracking from transcripts and hook-derived wait states.

Maintains an in-memory per-window snapshot of Claude Code tasks so Telegram can
render a live task list inside the existing status bubble. Transcript entries
are the source of truth; hook notifications only provide transient wait-state
headers and optimistic task completion hints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .topic_state_registry import topic_state

_WAITING_INPUT = "Waiting for input"
_PLAN_APPROVAL = "Plan approval needed"


@dataclass(frozen=True, slots=True)
class ClaudeTaskItem:
    """A single Claude task row suitable for UI rendering."""

    task_id: str
    subject: str
    description: str = ""
    active_form: str = ""
    status: str = "pending"
    blocked_by: tuple[str, ...] = ()
    order: int = 0
    owner: str = ""


@dataclass(frozen=True, slots=True)
class ClaudeTaskSnapshot:
    """Public snapshot of the current ordered Claude task list."""

    items: tuple[ClaudeTaskItem, ...]
    done_count: int
    open_count: int
    active_task_id: str | None = None

    @property
    def total_count(self) -> int:
        return len(self.items)


@dataclass(slots=True)
class _PendingTaskCreate:
    subject: str
    description: str
    active_form: str
    order: int


@dataclass(slots=True)
class _WindowTaskState:
    session_id: str = ""
    tasks: dict[str, ClaudeTaskItem] = field(default_factory=dict)
    pending_creates: dict[str, _PendingTaskCreate] = field(default_factory=dict)
    next_order: int = 1


def _normalize_status(raw_status: str | None) -> str:
    if not raw_status:
        return "pending"
    normalized = raw_status.lower().strip()
    if normalized in ("completed", "complete", "done", "finished"):
        return "completed"
    if normalized in ("in_progress", "in-progress", "active"):
        return "in_progress"
    return "pending"


def _as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _iter_message_blocks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    message = entry.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content", [])
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


class ClaudeTaskStateStore:
    """In-memory Claude task tracker keyed by tmux window ID."""

    def __init__(self) -> None:
        self._window_states: dict[str, _WindowTaskState] = {}
        self._wait_headers: dict[str, str] = {}
        self._last_status: dict[str, str] = {}

    def reset(self) -> None:
        self._window_states.clear()
        self._wait_headers.clear()
        self._last_status.clear()

    def clear_window(self, window_id: str) -> None:
        self._window_states.pop(window_id, None)
        self._wait_headers.pop(window_id, None)
        self._last_status.pop(window_id, None)

    def _replace_window(self, window_id: str, session_id: str) -> _WindowTaskState:
        state = _WindowTaskState(session_id=session_id)
        self._window_states[window_id] = state
        return state

    def _get_or_create(self, window_id: str, session_id: str) -> _WindowTaskState:
        state = self._window_states.get(window_id)
        if state is None or state.session_id != session_id:
            return self._replace_window(window_id, session_id)
        return state

    def has_snapshot(self, window_id: str) -> bool:
        state = self._window_states.get(window_id)
        return bool(state and state.tasks)

    def get_snapshot(self, window_id: str) -> ClaudeTaskSnapshot | None:
        state = self._window_states.get(window_id)
        if state is None or not state.tasks:
            return None

        items = tuple(sorted(state.tasks.values(), key=lambda item: item.order))
        done_count = sum(1 for item in items if item.status == "completed")
        active_task_id = next(
            (item.task_id for item in items if item.status == "in_progress"),
            None,
        )
        return ClaudeTaskSnapshot(
            items=items,
            done_count=done_count,
            open_count=len(items) - done_count,
            active_task_id=active_task_id,
        )

    def set_wait_header(self, window_id: str, header: str | None) -> bool:
        if not header:
            return self.clear_wait_header(window_id)
        if self._wait_headers.get(window_id) == header:
            return False
        self._wait_headers[window_id] = header
        return True

    def clear_wait_header(self, window_id: str) -> bool:
        return self._wait_headers.pop(window_id, None) is not None

    def get_wait_header(self, window_id: str) -> str | None:
        return self._wait_headers.get(window_id)

    def set_last_status(self, window_id: str, status_text: str) -> None:
        """Store the last non-idle status text for a window."""
        self._last_status[window_id] = status_text

    def get_last_status(self, window_id: str) -> str | None:
        """Retrieve the last non-idle status text for a window."""
        return self._last_status.get(window_id)

    def format_completion_text(self, window_id: str, num_turns: int = 0) -> str:
        """Build an enriched Ready message with task checklist and last status.

        Returns:
            Enriched text like::

                ✓ Ready
                ━━━━━━━━━━━━━━━━━━━━
                ✔ write unit tests
                ✔ run linter
                3/3 tasks done · 12 turns

            Falls back to ``"✓ Ready\\nLast: <status> · N turns"`` when no
            task checklist, or bare ``"✓ Ready"`` when nothing available.
        """
        from .handlers.callback_data import IDLE_STATUS_TEXT

        snapshot = self.get_snapshot(window_id)
        last_status = self.get_last_status(window_id)

        if snapshot is None and last_status is None:
            return IDLE_STATUS_TEXT

        lines: list[str] = [IDLE_STATUS_TEXT]

        if snapshot is not None:
            lines.append("\u2501" * 20)
            for item in snapshot.items[:8]:
                if item.status == "completed":
                    glyph = "\u2714"
                elif item.status == "in_progress":
                    glyph = "\u25d4"
                else:
                    glyph = "\u25fb"
                label = (
                    item.active_form
                    if item.status == "in_progress" and item.active_form
                    else item.subject
                )
                lines.append(f"{glyph} {label}")

            hidden = max(0, snapshot.total_count - 8)
            if hidden > 0:
                lines.append(f"+{hidden} more")

            summary_parts = [f"{snapshot.done_count}/{snapshot.total_count} tasks done"]
            if num_turns:
                summary_parts.append(f"{num_turns} turns")
            lines.append(" \u00b7 ".join(summary_parts))
        elif last_status:
            suffix = f" \u00b7 {num_turns} turns" if num_turns else ""
            lines.append(f"Last: {last_status}{suffix}")

        return "\n".join(lines)

    def rebuild_from_entries(
        self,
        window_id: str,
        session_id: str,
        entries: list[dict[str, Any]],
    ) -> bool:
        self._replace_window(window_id, session_id)
        return self.apply_entries(window_id, session_id, entries)

    def apply_entries(
        self,
        window_id: str,
        session_id: str,
        entries: list[dict[str, Any]],
    ) -> bool:
        if not window_id or not session_id:
            return False
        state = self._get_or_create(window_id, session_id)
        changed = False
        for entry in entries:
            entry_type = entry.get("type")
            if entry_type == "assistant":
                changed |= self._apply_assistant_entry(state, entry)
            elif entry_type == "user":
                changed |= self._apply_user_entry(state, entry)
        return changed

    def _apply_assistant_entry(
        self, state: _WindowTaskState, entry: dict[str, Any]
    ) -> bool:
        changed = False
        for block in _iter_message_blocks(entry):
            if block.get("type") != "tool_use":
                continue
            tool_name = block.get("name")
            input_data = block.get("input", {})
            if tool_name == "TaskCreate":
                changed |= self._apply_task_create(
                    state, _as_text(block.get("id")), input_data
                )
            elif tool_name == "TaskUpdate":
                changed |= self._apply_task_update(state, input_data)
            elif tool_name == "TodoWrite":
                changed |= self._replace_from_todos(state, input_data)
        return changed

    def _apply_user_entry(self, state: _WindowTaskState, entry: dict[str, Any]) -> bool:
        changed = False
        tool_result = entry.get("toolUseResult", {})
        if not isinstance(tool_result, dict):
            tool_result = {}
        for block in _iter_message_blocks(entry):
            if block.get("type") != "tool_result":
                continue
            tool_use_id = _as_text(block.get("tool_use_id"))
            if tool_use_id:
                task_data = tool_result.get("task")
                if isinstance(task_data, dict):
                    changed |= self._finalize_task_create(state, tool_use_id, task_data)
            tasks = tool_result.get("tasks")
            if isinstance(tasks, list):
                changed |= self._replace_from_task_list(state, tasks)
        return changed

    def mark_task_completed(
        self,
        window_id: str,
        session_id: str,
        task_id: str,
        *,
        subject: str = "",
    ) -> bool:
        state = self._window_states.get(window_id)
        if state is None or state.session_id != session_id:
            return False
        existing = state.tasks.get(task_id)
        if existing is None:
            return False
        if existing.status == "completed":
            return False
        state.tasks[task_id] = ClaudeTaskItem(
            task_id=existing.task_id,
            subject=existing.subject or subject,
            description=existing.description,
            active_form=existing.active_form,
            status="completed",
            blocked_by=existing.blocked_by,
            order=existing.order,
            owner=existing.owner,
        )
        return True

    def _apply_task_create(
        self,
        state: _WindowTaskState,
        tool_use_id: str,
        input_data: Any,
    ) -> bool:
        if not tool_use_id or not isinstance(input_data, dict):
            return False
        pending = _PendingTaskCreate(
            subject=_as_text(input_data.get("subject")),
            description=_as_text(input_data.get("description")),
            active_form=_as_text(input_data.get("activeForm")),
            order=state.next_order,
        )
        state.next_order += 1
        if state.pending_creates.get(tool_use_id) == pending:
            return False
        state.pending_creates[tool_use_id] = pending
        return True

    def _finalize_task_create(
        self,
        state: _WindowTaskState,
        tool_use_id: str,
        task_data: dict[str, Any],
    ) -> bool:
        pending = state.pending_creates.pop(tool_use_id, None)
        task_id = _as_text(task_data.get("id"))
        if not pending or not task_id:
            return False
        item = ClaudeTaskItem(
            task_id=task_id,
            subject=_as_text(task_data.get("subject")) or pending.subject,
            description=_as_text(task_data.get("description")) or pending.description,
            active_form=_as_text(task_data.get("activeForm")) or pending.active_form,
            status=_normalize_status(task_data.get("status")),
            blocked_by=tuple(
                str(value)
                for value in task_data.get("blockedBy", [])
                if str(value).strip()
            ),
            order=pending.order,
            owner=_as_text(task_data.get("owner")),
        )
        previous = state.tasks.get(task_id)
        if previous == item:
            return False
        state.tasks[task_id] = item
        return True

    def _apply_task_update(self, state: _WindowTaskState, input_data: Any) -> bool:
        if not isinstance(input_data, dict):
            return False
        task_id = _as_text(input_data.get("taskId"))
        if not task_id:
            return False
        existing = state.tasks.get(task_id)
        if existing is None:
            return False

        blocked_by = list(existing.blocked_by)
        for value in input_data.get("addBlockedBy", []):
            blocked = str(value).strip()
            if blocked and blocked not in blocked_by:
                blocked_by.append(blocked)
        remove = {str(value).strip() for value in input_data.get("removeBlockedBy", [])}
        if remove:
            blocked_by = [value for value in blocked_by if value not in remove]

        raw_status = input_data.get("status")
        updated = ClaudeTaskItem(
            task_id=existing.task_id,
            subject=_as_text(input_data.get("subject")) or existing.subject,
            description=_as_text(input_data.get("description")) or existing.description,
            active_form=_as_text(input_data.get("activeForm")) or existing.active_form,
            status=_normalize_status(raw_status)
            if isinstance(raw_status, str)
            else existing.status,
            blocked_by=tuple(blocked_by),
            order=existing.order,
            owner=existing.owner,
        )
        if updated == existing:
            return False
        state.tasks[task_id] = updated
        return True

    def _replace_from_task_list(
        self, state: _WindowTaskState, tasks: list[Any]
    ) -> bool:
        new_tasks: dict[str, ClaudeTaskItem] = {}
        next_order = 1
        for raw_task in tasks:
            if not isinstance(raw_task, dict):
                continue
            task_id = _as_text(raw_task.get("id"))
            if not task_id:
                continue
            new_tasks[task_id] = ClaudeTaskItem(
                task_id=task_id,
                subject=_as_text(raw_task.get("subject")),
                description=_as_text(raw_task.get("description")),
                active_form=_as_text(raw_task.get("activeForm")),
                status=_normalize_status(raw_task.get("status")),
                blocked_by=tuple(
                    str(value)
                    for value in raw_task.get("blockedBy", [])
                    if str(value).strip()
                ),
                order=next_order,
                owner=_as_text(raw_task.get("owner")),
            )
            next_order += 1

        if state.tasks == new_tasks:
            return False
        state.tasks = new_tasks
        state.pending_creates.clear()
        state.next_order = next_order
        return True

    def _replace_from_todos(self, state: _WindowTaskState, input_data: Any) -> bool:
        if not isinstance(input_data, dict):
            return False
        todos = input_data.get("todos", [])
        if not isinstance(todos, list):
            return False

        new_tasks: dict[str, ClaudeTaskItem] = {}
        next_order = 1
        for index, raw_todo in enumerate(todos, start=1):
            if not isinstance(raw_todo, dict):
                continue
            task_id = _as_text(raw_todo.get("id")) or str(index)
            subject = (
                _as_text(raw_todo.get("subject"))
                or _as_text(raw_todo.get("content"))
                or _as_text(raw_todo.get("task"))
            )
            new_tasks[task_id] = ClaudeTaskItem(
                task_id=task_id,
                subject=subject,
                description=_as_text(raw_todo.get("description")),
                active_form=_as_text(raw_todo.get("activeForm")),
                status=_normalize_status(raw_todo.get("status")),
                blocked_by=tuple(
                    str(value)
                    for value in raw_todo.get("blockedBy", [])
                    if str(value).strip()
                ),
                order=next_order,
                owner=_as_text(raw_todo.get("owner")),
            )
            next_order += 1

        if state.tasks == new_tasks:
            return False
        state.tasks = new_tasks
        state.pending_creates.clear()
        state.next_order = next_order
        return True


def classify_wait_message(message: str) -> str | None:
    """Normalize Claude hook notification text into a short status header."""
    stripped = message.strip()
    if not stripped:
        return None
    if stripped == "Claude is waiting for your input":
        return _WAITING_INPUT
    prefix = "Claude needs your permission to use "
    if stripped.startswith(prefix):
        tool_name = stripped.removeprefix(prefix).strip()
        if tool_name == "Updated plan":
            return _PLAN_APPROVAL
        if tool_name:
            return f"Approval needed: {tool_name}"
    return None


claude_task_state = ClaudeTaskStateStore()


def get_claude_task_snapshot(window_id: str) -> ClaudeTaskSnapshot | None:
    """Return the current Claude task snapshot for a window."""
    return claude_task_state.get_snapshot(window_id)


def get_claude_wait_header(window_id: str) -> str | None:
    """Return the current hook-derived wait header for a window."""
    return claude_task_state.get_wait_header(window_id)


@topic_state.register("window")
def clear_claude_task_window(window_id: str) -> None:
    """Clear Claude task and wait state for a window."""
    claude_task_state.clear_window(window_id)
