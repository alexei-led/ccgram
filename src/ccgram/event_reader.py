"""Incremental event-reader for the Claude Code hook event log (events.jsonl).

Pure I/O: reads new lines from events.jsonl by byte offset, parses them as
HookEvent objects, and returns both the events and the new offset. The caller
is responsible for persisting the offset (e.g., in MonitorState).

Key function: read_new_events().
"""

import json
from pathlib import Path

import aiofiles
import structlog

from .hooks.state_files import StateFileValidationError, parse_event_record
from .providers.base import HookEvent

logger = structlog.get_logger()


async def read_new_events(
    path: Path, current_offset: int
) -> tuple[list[HookEvent], int]:
    """Read new hook events from events.jsonl starting at current_offset.

    Returns (events, new_offset). On error returns ([], current_offset).
    Detects file truncation and resets offset to 0 automatically.
    """
    if not path.exists():
        return [], current_offset

    events: list[HookEvent] = []
    new_offset = current_offset

    try:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            await f.seek(0, 2)
            file_size = await f.tell()
            if current_offset > file_size:
                current_offset = 0
                new_offset = 0
            await f.seek(current_offset)

            async for line in f:
                line = line.strip()
                if not line:
                    new_offset = await f.tell()
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed event line (invalid JSON)")
                    new_offset = await f.tell()
                    continue

                try:
                    record = parse_event_record(data)
                except StateFileValidationError as exc:
                    logger.debug("Skipping invalid event record: %s", exc)
                    new_offset = await f.tell()
                    continue

                events.append(
                    HookEvent(
                        event_type=record.event,
                        window_key=record.window_key,
                        session_id=record.session_id,
                        data=record.data,
                        timestamp=record.ts,
                    )
                )
                new_offset = await f.tell()

    except OSError:
        logger.debug("Could not read events file %s", path)

    return events, new_offset
