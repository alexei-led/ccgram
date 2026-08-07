"""Provider-side discovery helpers for resumable sessions."""

from __future__ import annotations

import json
from pathlib import Path

from ccgram.providers.base import ResumableSession
from ccgram.utils import read_session_metadata_from_jsonl

_IndexParseError = (json.JSONDecodeError, OSError)


def index_message_count(entry: dict) -> int | None:
    """Return a positive message-count hint from a Claude index entry."""
    for key in ("messageCount", "msgCount", "msg_count", "messages"):
        value = entry.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def discover_claude_sessions(
    projects_path: Path,
    *,
    cwd: str | None = None,
    limit: int | None = None,
) -> list[ResumableSession]:
    """Discover Claude sessions, optionally restricted to one exact workspace."""
    resolved_cwd = _resolve_path(cwd) if cwd else None
    if cwd and resolved_cwd is None:
        return []

    candidates: list[tuple[float, ResumableSession]] = []
    seen_ids: set[str] = set()
    if not projects_path.exists():
        return []

    try:
        project_dirs = list(projects_path.iterdir())
    except OSError:
        return []

    for project_dir in project_dirs:
        if not project_dir.is_dir():
            continue
        index_file = project_dir / "sessions-index.json"
        if index_file.exists():
            _scan_index(index_file, resolved_cwd, seen_ids, candidates)
        _scan_bare_jsonl(project_dir, resolved_cwd, seen_ids, candidates)

    candidates.sort(key=lambda item: item[0], reverse=True)
    sessions = [session for _, session in candidates]
    return sessions[:limit] if limit is not None else sessions


def _resolve_path(path: str) -> str | None:
    try:
        return str(Path(path).expanduser().resolve())
    except OSError, ValueError:
        return None


def _scan_index(
    index_file: Path,
    resolved_cwd: str | None,
    seen_ids: set[str],
    candidates: list[tuple[float, ResumableSession]],
) -> None:
    try:
        index_data = json.loads(index_file.read_text(encoding="utf-8"))
    except _IndexParseError:
        return
    if not isinstance(index_data, dict):
        return

    original_path = index_data.get("originalPath", "")
    entries = index_data.get("entries", [])
    if not isinstance(entries, list):
        return

    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        session = _parse_index_entry(
            raw_entry,
            original_path=original_path,
            resolved_cwd=resolved_cwd,
            seen_ids=seen_ids,
        )
        if session is not None:
            seen_ids.add(session.session_id)
            candidates.append((session.mtime, session))


def _parse_index_entry(
    entry: dict,
    *,
    original_path: object,
    resolved_cwd: str | None,
    seen_ids: set[str],
) -> ResumableSession | None:
    session_id = entry.get("sessionId", "")
    full_path = entry.get("fullPath", "")
    if not isinstance(session_id, str) or not session_id or session_id in seen_ids:
        return None
    if not isinstance(full_path, str) or not full_path:
        return None

    project_path = entry.get("projectPath", original_path)
    if not isinstance(project_path, str) or not project_path:
        return None
    normalized_project = _resolve_path(project_path)
    if normalized_project is None or (
        resolved_cwd is not None and normalized_project != resolved_cwd
    ):
        return None

    session_file = Path(full_path)
    if not session_file.is_file():
        return None
    try:
        mtime = session_file.stat().st_mtime
    except OSError:
        mtime = 0.0

    summary = entry.get("summary", "") or entry.get("firstPrompt", "")
    if not isinstance(summary, str):
        summary = ""
    return ResumableSession(
        session_id=session_id,
        summary=summary or session_id[:12],
        cwd=normalized_project,
        provider_name="claude",
        mtime=mtime,
        msg_count=index_message_count(entry),
    )


def _scan_bare_jsonl(
    project_dir: Path,
    resolved_cwd: str | None,
    seen_ids: set[str],
    candidates: list[tuple[float, ResumableSession]],
) -> None:
    try:
        jsonl_files = list(project_dir.glob("*.jsonl"))
    except OSError:
        return

    for jsonl_file in jsonl_files:
        session_id = jsonl_file.stem
        if session_id in seen_ids:
            continue
        file_cwd, summary = read_session_metadata_from_jsonl(jsonl_file)
        if not file_cwd:
            continue
        normalized_project = _resolve_path(file_cwd)
        if normalized_project is None:
            continue
        if resolved_cwd is not None and normalized_project != resolved_cwd:
            continue
        try:
            mtime = jsonl_file.stat().st_mtime
        except OSError:
            mtime = 0.0

        seen_ids.add(session_id)
        candidates.append(
            (
                mtime,
                ResumableSession(
                    session_id=session_id,
                    summary=summary or session_id[:12],
                    cwd=normalized_project,
                    provider_name="claude",
                    mtime=mtime,
                ),
            )
        )
