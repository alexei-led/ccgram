"""Unit tests for tmux auto-detection and duplicate instance prevention."""

import subprocess
from unittest.mock import patch

from ccgram.utils import check_duplicate_ccgram, detect_tmux_context


class TestDetectTmuxContext:
    def test_returns_both_inside_tmux(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,12345,0")
        monkeypatch.setenv("TMUX_PANE", "%5")
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="my-session\t@3\n", stderr=""
        )
        with patch("ccgram.utils.subprocess.run", return_value=completed) as mock_run:
            session, window_id = detect_tmux_context()
        assert session == "my-session"
        assert window_id == "@3"
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

    def test_returns_none_none_outside_tmux(self, monkeypatch):
        monkeypatch.delenv("TMUX", raising=False)
        assert detect_tmux_context() == (None, None)

    def test_session_only_without_tmux_pane(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,12345,0")
        monkeypatch.delenv("TMUX_PANE", raising=False)
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="my-session\n", stderr=""
        )
        with patch("ccgram.utils.subprocess.run", return_value=completed):
            session, window_id = detect_tmux_context()
        assert session == "my-session"
        assert window_id is None

    def test_returns_none_none_on_empty_output(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,12345,0")
        monkeypatch.setenv("TMUX_PANE", "%5")
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="  \n", stderr=""
        )
        with patch("ccgram.utils.subprocess.run", return_value=completed):
            assert detect_tmux_context() == (None, None)

    def test_returns_none_none_on_nonzero_returncode(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,12345,0")
        monkeypatch.setenv("TMUX_PANE", "%5")
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="error message\n", stderr=""
        )
        with patch("ccgram.utils.subprocess.run", return_value=completed):
            assert detect_tmux_context() == (None, None)

    def test_returns_none_none_on_timeout(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,12345,0")
        monkeypatch.setenv("TMUX_PANE", "%5")
        with patch(
            "ccgram.utils.subprocess.run",
            side_effect=subprocess.TimeoutExpired("tmux", 5),
        ):
            assert detect_tmux_context() == (None, None)

    def test_returns_none_none_on_missing_tmux_binary(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,12345,0")
        monkeypatch.setenv("TMUX_PANE", "%5")
        with patch("ccgram.utils.subprocess.run", side_effect=FileNotFoundError):
            assert detect_tmux_context() == (None, None)

    def test_no_pane_nonzero_returncode(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,12345,0")
        monkeypatch.delenv("TMUX_PANE", raising=False)
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="error\n", stderr=""
        )
        with patch("ccgram.utils.subprocess.run", return_value=completed):
            assert detect_tmux_context() == (None, None)

    def test_no_pane_timeout(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,12345,0")
        monkeypatch.delenv("TMUX_PANE", raising=False)
        with patch(
            "ccgram.utils.subprocess.run",
            side_effect=subprocess.TimeoutExpired("tmux", 5),
        ):
            assert detect_tmux_context() == (None, None)

    def test_no_pane_missing_binary(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,12345,0")
        monkeypatch.delenv("TMUX_PANE", raising=False)
        with patch("ccgram.utils.subprocess.run", side_effect=FileNotFoundError):
            assert detect_tmux_context() == (None, None)


class TestCheckDuplicateCcgram:
    def test_detects_duplicate(self, monkeypatch):
        monkeypatch.setenv("TMUX_PANE", "%5")
        output = "%1\t@0\tfish\n%2\t@1\tccgram\n%5\t@3\tccgram\n"
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=output, stderr=""
        )
        with patch("ccgram.utils.subprocess.run", return_value=completed):
            result = check_duplicate_ccgram("test-session")
        assert result is not None
        assert "Another ccgram instance" in result
        assert "test-session" in result
        assert "@1" in result

    def test_ignores_own_pane(self, monkeypatch):
        monkeypatch.setenv("TMUX_PANE", "%2")
        output = "%1\t@0\tfish\n%2\t@1\tccgram\n"
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=output, stderr=""
        )
        with patch("ccgram.utils.subprocess.run", return_value=completed):
            assert check_duplicate_ccgram("test-session") is None

    def test_no_duplicate(self, monkeypatch):
        monkeypatch.setenv("TMUX_PANE", "%5")
        output = "%1\t@0\tfish\n%2\t@1\tclaude\n"
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=output, stderr=""
        )
        with patch("ccgram.utils.subprocess.run", return_value=completed):
            assert check_duplicate_ccgram("test-session") is None

    def test_skips_check_when_tmux_pane_empty(self, monkeypatch):
        monkeypatch.delenv("TMUX_PANE", raising=False)
        assert check_duplicate_ccgram("test-session") is None

    def test_returns_none_on_nonzero_returncode(self, monkeypatch):
        monkeypatch.setenv("TMUX_PANE", "%5")
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="error\n", stderr=""
        )
        with patch("ccgram.utils.subprocess.run", return_value=completed):
            assert check_duplicate_ccgram("test-session") is None

    def test_returns_none_on_timeout(self, monkeypatch):
        monkeypatch.setenv("TMUX_PANE", "%5")
        with patch(
            "ccgram.utils.subprocess.run",
            side_effect=subprocess.TimeoutExpired("tmux", 5),
        ):
            assert check_duplicate_ccgram("test-session") is None

    def test_returns_none_on_missing_binary(self, monkeypatch):
        monkeypatch.setenv("TMUX_PANE", "%5")
        with patch("ccgram.utils.subprocess.run", side_effect=FileNotFoundError):
            assert check_duplicate_ccgram("test-session") is None

    def test_handles_malformed_lines(self, monkeypatch):
        monkeypatch.setenv("TMUX_PANE", "%5")
        output = "bad-line\n%1\t@0\tfish\n"
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=output, stderr=""
        )
        with patch("ccgram.utils.subprocess.run", return_value=completed):
            assert check_duplicate_ccgram("test-session") is None
