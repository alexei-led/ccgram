#!/usr/bin/env python3
"""Publish the reviewed changelog section, including its upgrade instructions."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

_RELEASE_PREFIX = "## ["


def extract_release_notes(changelog: str, tag: str) -> str:
    version = tag.removeprefix("v")
    if not re.fullmatch(
        r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", tag
    ):
        raise ValueError("Expected a versioned release tag")
    heading = f"{_RELEASE_PREFIX}{version}]"
    lines = changelog.splitlines()
    matches = [
        index
        for index, line in enumerate(lines)
        if line == heading or line.startswith(f"{heading} ")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one changelog section for {tag}")
    start = matches[0] + 1
    end = next(
        (
            index
            for index in range(start, len(lines))
            if lines[index].startswith(_RELEASE_PREFIX)
        ),
        len(lines),
    )
    notes = "\n".join(lines[start:end]).strip()
    if not notes:
        raise ValueError(f"Changelog section for {tag} is empty")
    return f"{notes}\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    args = parser.parse_args(argv)
    try:
        notes = extract_release_notes(
            Path("CHANGELOG.md").read_text(encoding="utf-8"), args.tag
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
