"""Fitness gate for neutral code above the concrete Herdr adapter."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_handlers_do_not_import_concrete_herdr_backend() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src/ccgram/handlers").rglob("*.py")
        if "multiplexer.herdr" in path.read_text()
    ]
    assert not offenders


def test_agent_session_parsing_and_live_locators_stay_in_adapter() -> None:
    sources = {
        path.relative_to(ROOT): path.read_text()
        for path in (ROOT / "src/ccgram").rglob("*.py")
        if path != ROOT / "src/ccgram/multiplexer/herdr.py"
    }
    offenders = [
        f"{path}: agent_session"
        for path, source in sources.items()
        if "agent_session" in source and "hook.py" not in str(path)
    ]
    assert not offenders
