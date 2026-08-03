"""Fitness gate for the guarded Herdr session identity boundary."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERDR = ROOT / "src/ccgram/multiplexer/herdr.py"


def test_one_canonical_digest_owner_and_no_layout_identity() -> None:
    source = HERDR.read_text()
    assert source.count("def herdr_session_target_id(") == 1
    assert "def canonical_session_bytes(" in source
    # Identity comes from the complete agent_session composite, not layout data.
    identity_section = source[
        source.index("def _session_composite") : source.index("class HerdrManager")
    ]
    for forbidden in ("focused", "title", "cwd", "directory", "screen", "layout"):
        assert forbidden not in identity_section


def test_legacy_target_predicate_is_exact_version_prefix() -> None:
    source = (ROOT / "src/ccgram/window_state_store.py").read_text()
    assert 'window_id.startswith("herdr-session-v1-")' in source
