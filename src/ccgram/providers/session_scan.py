"""Shared session-bucket scanning for the JSONL providers (pi, omp).

Both agents store one JSONL transcript per session inside a per-cwd bucket
directory, and both find "the session for this window" the same way: list the
bucket newest-first, read each transcript's header, and take the first whose
header ``cwd`` resolves to the window's cwd.

The bucket *directory* is the provider's business — pi and omp encode a cwd
differently (see ``pi.encode_cwd_dirname`` / ``omp.encode_cwd_dirname``), so
callers pass the resolved directory in rather than a cwd.
"""

from __future__ import annotations

import time
from pathlib import Path

from ccgram.providers.base import SessionStartEvent
from ccgram.providers.pi_format import read_session_header


def candidate_transcripts(sessions_dir: Path) -> list[tuple[float, Path]]:
    """Return ``(mtime, path)`` tuples for a bucket's sessions, newest first."""
    if not sessions_dir.is_dir():
        return []
    results: list[tuple[float, Path]] = []
    try:
        for entry in sessions_dir.iterdir():
            if entry.suffix == ".jsonl" and entry.is_file():
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                results.append((mtime, entry))
    except OSError:
        return []
    results.sort(key=lambda pair: pair[0], reverse=True)
    return results


def newest_matching_transcript(
    sessions_dir: Path,
    cwd: str,
    *,
    window_key: str,
    max_age: float,
    scan_limit: int,
) -> SessionStartEvent | None:
    """Find the newest transcript in *sessions_dir* whose header cwd matches.

    ``max_age`` caps how old a candidate may be (0 or less disables the cap) so
    a dead window never adopts an unrelated historical transcript for the same
    cwd; ``scan_limit`` bounds how many recent files are inspected.
    """
    if not cwd:
        return None
    try:
        resolved_target = str(Path(cwd).resolve())
    except OSError:
        return None

    now = time.time()
    for mtime, path in candidate_transcripts(sessions_dir)[:scan_limit]:
        if max_age > 0 and now - mtime > max_age:
            break
        header = read_session_header(str(path))
        if not header:
            continue
        try:
            header_cwd = str(Path(header["cwd"]).resolve())
        except OSError:
            continue
        if header_cwd != resolved_target:
            continue
        return SessionStartEvent(
            session_id=header["id"],
            cwd=header["cwd"],
            transcript_path=str(path),
            window_key=window_key,
        )
    return None
