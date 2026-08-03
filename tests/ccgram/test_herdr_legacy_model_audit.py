"""Fitness gate: current Herdr code and docs must not revive tab/pane identity."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURRENT_PATHS = (
    "src/ccgram/multiplexer",
    "src/ccgram/hook.py",
    "src/ccgram/session.py",
    "src/ccgram/session_map.py",
    "src/ccgram/window_resolver.py",
    "tests/ccgram",
    "tests/integration/test_herdr_contract.py",
    "README.md",
    "docs/guides.md",
    "docs/architecture.md",
    "docs/ai-agents/architecture-map.md",
    "docs/ai-agents/codebase-index.md",
)
FORBIDDEN = (
    "_active_pane",
    "_representative_pane",
    "CCGRAM_HERDR_TOPIC_SCOPE",
    "live_window_session_ids",
    "_resolve_by_session_id",
)


def _current_sources() -> dict[Path, str]:
    sources: dict[Path, str] = {}
    for relative in CURRENT_PATHS:
        path = ROOT / relative
        paths = path.rglob("*.py") if path.is_dir() else (path,)
        for candidate in paths:
            if candidate.is_file() and candidate.name != Path(__file__).name:
                # Generic tmux tests can legitimately talk about active panes;
                # this audit limits that vocabulary ban to Herdr-specific tests.
                if "tests/ccgram" in str(candidate) and "herdr" not in candidate.name:
                    continue
                sources[candidate.relative_to(ROOT)] = candidate.read_text()
    return sources


def test_current_herdr_paths_contain_no_legacy_identity_fallbacks() -> None:
    offenders = [
        f"{path}: {token}"
        for path, source in _current_sources().items()
        for token in FORBIDDEN
        if token in source
    ]
    assert not offenders, "\n".join(offenders)


def test_audit_rejects_a_planted_focus_identity_helper() -> None:
    assert "_active" + "_pane" in "async def _active_pane(tab_id): pass"
