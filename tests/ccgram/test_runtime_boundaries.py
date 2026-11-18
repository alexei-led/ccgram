"""Import boundaries between neutral event observation and application adapters."""

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[2] / "src" / "ccgram"
_FORBIDDEN = {
    "event_stream_monitor": (
        "ccgram.handlers",
        "ccgram.telegram",
        "ccgram.thread_router",
        "ccgram.bot",
        "telegram",
    ),
    "bot": ("ccgram.main",),
    "bootstrap": ("ccgram.handlers.polling.window_tick",),
}


def _forbidden_imports(source: str, forbidden: tuple[str, ...]) -> list[str]:
    violations = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = resolve_name("." * node.level + base, "ccgram")
            names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
        else:
            continue
        violations.extend(
            f"{node.lineno}: {name}" for name in names if name.startswith(forbidden)
        )
    return violations


@pytest.mark.parametrize("module", _FORBIDDEN)
def test_runtime_dependency_direction(module):
    source = (_SOURCE / f"{module}.py").read_text()
    violations = _forbidden_imports(source, _FORBIDDEN[module])
    assert not violations, f"{module}.py: " + "\n".join(violations)


@pytest.mark.parametrize(
    ("module", "source", "rejected"),
    [
        ("bot", "import ccgram.main as runner", True),
        ("bot", "from .main import _shutdown_signal", True),
        ("bot", "def callback():\n    from . import main", True),
        ("bot", "from ccgram import main as runner", True),
        ("event_stream_monitor", "from .handlers import polling", True),
        ("event_stream_monitor", "import telegram", True),
        ("event_stream_monitor", "from .telegram_client import TelegramClient", True),
        ("event_stream_monitor", "from .multiplexer import base", False),
        (
            "bootstrap",
            "from .handlers.polling.window_tick import _handle_dead_window_notification",
            True,
        ),
        (
            "bootstrap",
            "from .handlers.polling.window_death import handle_window_death",
            False,
        ),
    ],
)
def test_audit_rejects_planted_imports(module, source, rejected):
    assert bool(_forbidden_imports(source, _FORBIDDEN[module])) is rejected
