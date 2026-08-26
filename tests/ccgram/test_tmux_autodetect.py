"""Unit tests for tmux auto-detection and duplicate instance prevention."""

import subprocess
from unittest.mock import patch

import pytest

from ccgram.utils import check_duplicate_ccgram, detect_tmux_context


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


# Every way the tmux probe can fail to answer. Each entry is a `subprocess.run`
# stand-in; all of them must degrade to "no context".
UNUSABLE_PROBES = [
    pytest.param({"return_value": _completed("  \n")}, id="empty-output"),
    pytest.param({"return_value": _completed("error\n", returncode=1)}, id="exit-1"),
    pytest.param({"side_effect": subprocess.TimeoutExpired("tmux", 5)}, id="timeout"),
    pytest.param({"side_effect": FileNotFoundError}, id="tmux-not-installed"),
]


class TestDetectTmuxContext:
    @pytest.fixture(autouse=True)
    def _inside_tmux(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,12345,0")

    def test_returns_session_and_window_for_own_pane(self, monkeypatch):
        monkeypatch.setenv("TMUX_PANE", "%5")
        with patch(
            "ccgram.utils.subprocess.run", return_value=_completed("my-session\t@3\n")
        ) as mock_run:
            assert detect_tmux_context() == ("my-session", "@3")
        mock_run.assert_called_once_with(
            [
                "tmux",
                "display-message",
                "-t",
                "%5",
                "-p",
                "#{session_name}\t#{window_id}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_session_only_without_tmux_pane(self, monkeypatch):
        monkeypatch.delenv("TMUX_PANE", raising=False)
        with patch(
            "ccgram.utils.subprocess.run", return_value=_completed("my-session\n")
        ):
            assert detect_tmux_context() == ("my-session", None)

    def test_returns_none_none_outside_tmux(self, monkeypatch):
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setattr(
            "ccgram.utils.subprocess.run",
            lambda *a, **k: pytest.fail("no tmux probe outside tmux"),
        )
        assert detect_tmux_context() == (None, None)

    @pytest.mark.parametrize("probe", UNUSABLE_PROBES)
    @pytest.mark.parametrize(
        "pane", [pytest.param("%5", id="with-pane"), pytest.param(None, id="no-pane")]
    )
    def test_unusable_probe_yields_no_context(self, monkeypatch, probe, pane):
        if pane:
            monkeypatch.setenv("TMUX_PANE", pane)
        else:
            monkeypatch.delenv("TMUX_PANE", raising=False)
        with patch("ccgram.utils.subprocess.run", **probe):
            assert detect_tmux_context() == (None, None)


class TestCheckDuplicateCcgram:
    def test_detects_duplicate(self, monkeypatch):
        monkeypatch.setenv("TMUX_PANE", "%5")
        output = "%1\t@0\tfish\n%2\t@1\tccgram\n%5\t@3\tccgram\n"
        with patch("ccgram.utils.subprocess.run", return_value=_completed(output)):
            result = check_duplicate_ccgram("test-session")
        assert result is not None
        assert "Another ccgram instance" in result
        assert "test-session" in result
        assert "@1" in result

    @pytest.mark.parametrize(
        ("own_pane", "output"),
        [
            pytest.param("%2", "%1\t@0\tfish\n%2\t@1\tccgram\n", id="only-own-pane"),
            pytest.param("%5", "%1\t@0\tfish\n%2\t@1\tclaude\n", id="no-ccgram-pane"),
            pytest.param("%5", "bad-line\n%1\t@0\tfish\n", id="malformed-line"),
        ],
    )
    def test_no_duplicate_reported(self, monkeypatch, own_pane, output):
        monkeypatch.setenv("TMUX_PANE", own_pane)
        with patch("ccgram.utils.subprocess.run", return_value=_completed(output)):
            assert check_duplicate_ccgram("test-session") is None

    def test_skips_check_when_tmux_pane_empty(self, monkeypatch):
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.setattr(
            "ccgram.utils.subprocess.run",
            lambda *a, **k: pytest.fail("no tmux probe without TMUX_PANE"),
        )
        assert check_duplicate_ccgram("test-session") is None

    @pytest.mark.parametrize("probe", UNUSABLE_PROBES)
    def test_unusable_probe_reports_no_duplicate(self, monkeypatch, probe):
        monkeypatch.setenv("TMUX_PANE", "%5")
        with patch("ccgram.utils.subprocess.run", **probe):
            assert check_duplicate_ccgram("test-session") is None
