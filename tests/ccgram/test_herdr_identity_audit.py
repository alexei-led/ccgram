"""Fitness gate for the guarded Herdr session identity boundary."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERDR = ROOT / "src/ccgram/multiplexer/herdr.py"


# Layout/volatile record fields must never feed a session composite.
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


def test_record_assembler_uses_guarded_terminal_fallback() -> None:
    """Sessionless known agents use only their guarded terminal identity."""
    source = HERDR.read_text()
    section = source[
        source.index("def _parse_live_record") : source.index("class HerdrManager")
    ]
    assert "if composite is None:" in section
    assert 'agent not in {"claude", "pi", "codex", "gemini"}' in section
    assert 'HerdrSessionComposite("herdr", agent, "terminal", terminal_id)' in section


def test_persisted_target_predicate_uses_the_shared_exact_validator() -> None:
    source = (ROOT / "src/ccgram/window_state_store.py").read_text()
    validator = (ROOT / "src/ccgram/herdr_targets.py").read_text()
    assert "is_herdr_session_target(window_id)" in source
    assert "[0-9a-f]{{64}}" in validator
    assert "fullmatch(value)" in validator
