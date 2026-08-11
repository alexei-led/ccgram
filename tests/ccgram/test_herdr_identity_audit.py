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
    # Only the composite→digest functions are audited: layout fields (cwd,
    # titles) legitimately ride along in HerdrLiveRecord as locator data for
    # topic discovery, but they must never influence the identity digest.
    identity_section = source[
        source.index("def _session_composite") : source.index("def _parse_live_record")
    ]
    for forbidden in ("focused", "title", "cwd", "directory", "screen", "layout"):
        assert forbidden not in identity_section


def test_persisted_target_predicate_uses_the_shared_exact_validator() -> None:
    source = (ROOT / "src/ccgram/window_state_store.py").read_text()
    validator = (ROOT / "src/ccgram/herdr_targets.py").read_text()
    assert "is_herdr_session_target(window_id)" in source
    assert "[0-9a-f]{{64}}" in validator
    assert "fullmatch(value)" in validator
