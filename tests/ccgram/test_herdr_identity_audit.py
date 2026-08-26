"""Fitness gate for the guarded Herdr session identity boundary."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERDR = ROOT / "src/ccgram/multiplexer/herdr.py"


# Layout/volatile record fields that must never feed a session composite. The
# terminal fallback composite deliberately uses ``terminal_id``; the ban is on
# display and filesystem state, not on locators as such.
FORBIDDEN = ("focused", "title", "cwd", "directory", "screen", "layout")


def test_one_canonical_digest_owner_and_no_layout_identity() -> None:
    source = HERDR.read_text()
    assert source.count("def herdr_session_target_id(") == 1
    assert "def canonical_session_bytes(" in source
    # Identity comes from the complete agent_session composite, not layout data.
    # The region stops at ``_parse_live_record``: that is the record assembler,
    # which legitimately carries locators and cwd alongside the identity it
    # derives. Its identity derivation is guarded by the next test instead.
    identity_section = source[
        source.index("def _session_composite") : source.index("def _parse_live_record")
    ]
    for forbidden in FORBIDDEN:
        assert forbidden not in identity_section


def test_record_assembler_builds_composites_only_from_agent_and_terminal() -> None:
    """``_parse_live_record`` may carry layout data, never hash it."""
    source = HERDR.read_text()
    section = source[
        source.index("def _parse_live_record") : source.index("class HerdrManager")
    ]
    constructions = re.findall(r"HerdrSessionComposite\((.*?)\)", section, re.DOTALL)
    assert constructions, "expected the terminal fallback composite construction"
    for args in constructions:
        for forbidden in FORBIDDEN:
            assert forbidden not in args


def test_persisted_target_predicate_uses_the_shared_exact_validator() -> None:
    source = (ROOT / "src/ccgram/window_state_store.py").read_text()
    validator = (ROOT / "src/ccgram/herdr_targets.py").read_text()
    assert "is_herdr_session_target(window_id)" in source
    assert "[0-9a-f]{{64}}" in validator
    assert "fullmatch(value)" in validator
