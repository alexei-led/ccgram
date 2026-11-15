"""Release publishing must use the reviewed version's complete authored notes."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "extract_release_notes.py"


def _extract(
    tmp_path: Path, changelog: str | None, tag: str
) -> subprocess.CompletedProcess[str]:
    if changelog is not None:
        (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(_SCRIPT), tag],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=5,
    )


@pytest.mark.parametrize(
    ("changelog", "tag", "expected"),
    [
        (
            "# Changelog\n\n## [4.23.0] - 2026-11-15\n\n### Upgrade\n\nStop the old daemon.\n\n## [4.22.0] - 2026-11-09\nOld notes.\n",
            "v4.23.0",
            "### Upgrade\n\nStop the old daemon.\n",
        ),
        (
            "## [Unreleased]\nFuture notes.\n## [4.23.0]\nUnicode ✓ and [guide](https://example.com/guide).\n## [4.22.0]\nOld notes.\n",
            "v4.23.0",
            "Unicode ✓ and [guide](https://example.com/guide).\n",
        ),
        (
            "## [4.23.0]\nNew notes.\n## [4.2.0]\nLast notes.\n",
            "v4.2.0",
            "Last notes.\n",
        ),
        ("## [4.23.0]\r\n\r\nWindows newlines.\r\n", "v4.23.0", "Windows newlines.\n"),
    ],
)
def test_publisher_retains_only_requested_reviewed_notes(
    tmp_path: Path, changelog: str, tag: str, expected: str
) -> None:
    result = _extract(tmp_path, changelog, tag)
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("changelog", "tag", "error"),
    [
        ("## [4.22.0]\nOld notes.\n", "v4.23.0", "exactly one changelog section"),
        (
            "## [4.23.0]\nA.\n## [4.23.0]\nB.\n",
            "v4.23.0",
            "exactly one changelog section",
        ),
        ("## [4.23.0]\n\n## [4.22.0]\nOld.\n", "v4.23.0", "is empty"),
        ("## [Unreleased]\nFuture.\n", "vUnreleased", "versioned release tag"),
        ("## [4.23.0]\nNotes.\n", "4.23.0", "versioned release tag"),
        ("## [4.23.0]\nNotes.\n", "v04.23.0", "versioned release tag"),
        ("## [4.23.0]\nNotes.\n", 'v4.23.0";echo injected', "versioned release tag"),
        (None, "v4.23.0", "CHANGELOG.md"),
    ],
)
def test_invalid_notes_stop_publishing_without_partial_output(
    tmp_path: Path, changelog: str | None, tag: str, error: str
) -> None:
    result = _extract(tmp_path, changelog, tag)
    assert result.returncode == 2
    assert result.stdout == ""
    assert error in result.stderr
