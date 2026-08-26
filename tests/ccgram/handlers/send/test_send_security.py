"""Tests for src/ccgram/handlers/send/send_security.py."""

from __future__ import annotations

import stat as stat_module
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ccgram.handlers.send.send_security import (
    check_gitleaks_rules,
    is_excluded_dir,
    is_gitignored,
    is_hidden,
    is_path_contained,
    matches_secret_pattern,
    validate_sendable,
)


def _git_result(returncode: int) -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    return result


def _touch(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


class TestIsPathContained:
    @pytest.mark.parametrize(
        ("relative", "expected"),
        [
            pytest.param("file.txt", True, id="file-at-root"),
            pytest.param("sub/file.txt", True, id="file-in-subdir"),
            pytest.param("../secret.txt", False, id="dotdot-traversal"),
        ],
    )
    def test_containment(self, tmp_path: Path, relative: str, expected: bool) -> None:
        assert is_path_contained(tmp_path / relative, tmp_path) is expected

    def test_root_itself_is_contained(self, tmp_path: Path) -> None:
        assert is_path_contained(tmp_path, tmp_path) is True

    def test_file_outside_root(self, tmp_path: Path) -> None:
        assert is_path_contained(tmp_path.parent / "other.txt", tmp_path) is False

    def test_symlink_escaping_root_denied(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.txt"
        outside.touch()
        link = tmp_path / "link.txt"
        link.symlink_to(outside)
        assert is_path_contained(link, tmp_path) is False

    def test_symlink_within_root_allowed(self, tmp_path: Path) -> None:
        target = tmp_path / "real.txt"
        target.touch()
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        assert is_path_contained(link, tmp_path) is True


class TestIsHidden:
    @pytest.mark.parametrize(
        ("relative", "expected"),
        [
            pytest.param(".env", True, id="dotfile"),
            pytest.param(".git/config", True, id="file-in-dotdir"),
            pytest.param("src/.hidden.py", True, id="dotfile-in-subdir"),
            pytest.param("main.py", False, id="plain-file"),
            pytest.param("src/module.py", False, id="plain-file-in-subdir"),
        ],
    )
    def test_hidden_components(
        self, tmp_path: Path, relative: str, expected: bool
    ) -> None:
        assert is_hidden(_touch(tmp_path, relative), tmp_path) is expected

    def test_root_itself_not_hidden(self, tmp_path: Path) -> None:
        assert is_hidden(tmp_path, tmp_path) is False


class TestMatchesSecretPattern:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param(".env", ".env", id="env"),
            pytest.param(".env.local", ".env.*", id="env-suffixed"),
            pytest.param("cert.pem", "*.pem", id="pem"),
            pytest.param("CERT.PEM", "*.pem", id="pem-uppercase"),
            pytest.param("id_rsa.key", "*.key", id="key"),
            pytest.param("keystore.p12", "*.p12", id="p12"),
            pytest.param("credential.json", "*credential*", id="credential-substring"),
            pytest.param("app.secret.yaml", "*secret*", id="secret-substring"),
            pytest.param("main.py", None, id="source-file"),
            pytest.param("config.toml", None, id="plain-config"),
        ],
    )
    def test_pattern_match(self, name: str, expected: str | None) -> None:
        assert matches_secret_pattern(Path(name)) == expected


class TestIsGitignored:
    @pytest.mark.parametrize(
        ("returncode", "expected"),
        [
            pytest.param(0, True, id="git-says-ignored"),
            pytest.param(1, False, id="git-says-tracked"),
        ],
    )
    def test_git_decides(self, tmp_path: Path, returncode: int, expected: bool) -> None:
        path = _touch(tmp_path, "file.log")
        with patch("subprocess.run", return_value=_git_result(returncode)):
            assert is_gitignored(path, tmp_path) is expected

    @pytest.mark.parametrize(
        "git_failure",
        [
            pytest.param(FileNotFoundError(), id="git-missing"),
            pytest.param(subprocess.TimeoutExpired(["git"], 5), id="git-timeout"),
            pytest.param(_git_result(128), id="git-fatal-error"),
        ],
    )
    @pytest.mark.parametrize(
        ("gitignore", "relative", "expected"),
        [
            pytest.param("*.log\n", "debug.log", True, id="pattern-matches"),
            pytest.param("*.log\n", "main.py", False, id="pattern-misses"),
            pytest.param("build/\n", "build/output.bin", True, id="dir-pattern"),
            pytest.param(None, "file.py", False, id="no-gitignore"),
        ],
    )
    def test_pathspec_fallback(
        self,
        tmp_path: Path,
        git_failure: object,
        gitignore: str | None,
        relative: str,
        expected: bool,
    ) -> None:
        if gitignore is not None:
            (tmp_path / ".gitignore").write_text(gitignore, encoding="utf-8")
        path = _touch(tmp_path, relative)
        with patch("subprocess.run") as mock_run:
            if isinstance(git_failure, MagicMock):
                mock_run.return_value = git_failure
            else:
                mock_run.side_effect = git_failure
            assert is_gitignored(path, tmp_path) is expected


_GITLEAKS_AWS_RULE = b"""
[[rules]]
id = "aws-key"
path = ".*credentials.*"
"""


class TestCheckGitleaksRules:
    @pytest.mark.parametrize(
        ("toml", "relative", "expected"),
        [
            pytest.param(None, "main.py", None, id="no-config"),
            pytest.param(
                _GITLEAKS_AWS_RULE, "aws-credentials.txt", "aws-key", id="rule-matches"
            ),
            pytest.param(_GITLEAKS_AWS_RULE, "main.py", None, id="rule-misses"),
            pytest.param(
                b"[[rules]]\npath = '.*[.]pfx$'\n",
                "cert.pfx",
                "gitleaks rule",
                id="rule-without-id-uses-default",
            ),
            pytest.param(
                b'[[rules]]\nid = "no-path"\n',
                "main.py",
                None,
                id="rule-without-path-skipped",
            ),
            pytest.param(
                b'[[rules]]\nid = "bad-regex"\npath = "[invalid("\n\n'
                b'[[rules]]\nid = "good-rule"\npath = ".*secret.*"\n',
                "my-secret.txt",
                "good-rule",
                id="invalid-regex-skipped",
            ),
            pytest.param(
                b"this is [not valid toml }{", "main.py", None, id="malformed"
            ),
            pytest.param(b"[title]\nversion = 1\n", "main.py", None, id="no-rules"),
        ],
    )
    def test_rule_evaluation(
        self,
        tmp_path: Path,
        toml: bytes | None,
        relative: str,
        expected: str | None,
    ) -> None:
        if toml is not None:
            (tmp_path / ".gitleaks.toml").write_bytes(toml)
        assert check_gitleaks_rules(tmp_path / relative, tmp_path) == expected

    def test_path_outside_cwd_skips_rules(self, tmp_path: Path) -> None:
        (tmp_path / ".gitleaks.toml").write_bytes(_GITLEAKS_AWS_RULE)
        outside = tmp_path.parent / "aws-credentials.txt"
        assert check_gitleaks_rules(outside, tmp_path) is None


class TestValidateSendable:
    def test_clean_file_returns_none(self, tmp_path: Path) -> None:
        path = _touch(tmp_path, "report.txt")
        with patch("subprocess.run", return_value=_git_result(1)):
            assert validate_sendable(path, tmp_path) is None

    def test_outside_cwd_denied(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "secret.txt"
        assert (
            validate_sendable(outside, tmp_path) == "File is outside project directory"
        )

    def test_hidden_file_denied(self, tmp_path: Path) -> None:
        path = _touch(tmp_path, ".env")
        assert validate_sendable(path, tmp_path) == "Hidden files cannot be sent"

    def test_secret_pattern_denied(self, tmp_path: Path) -> None:
        path = _touch(tmp_path, "server.pem")
        result = validate_sendable(path, tmp_path)
        assert result is not None
        assert "credentials" in result
        assert "*.pem" in result

    def test_gitleaks_rule_denied(self, tmp_path: Path) -> None:
        (tmp_path / ".gitleaks.toml").write_bytes(
            b'[[rules]]\nid = "my-rule"\npath = ".*private.*"\n'
        )
        path = _touch(tmp_path, "private-key.txt")
        result = validate_sendable(path, tmp_path)
        assert result is not None
        assert "gitleaks" in result
        assert "my-rule" in result

    def test_gitignored_denied(self, tmp_path: Path) -> None:
        path = _touch(tmp_path, "output.log")
        with patch("subprocess.run", return_value=_git_result(0)):
            assert validate_sendable(path, tmp_path) == "File is gitignored"

    def test_file_too_large_denied(self, tmp_path: Path) -> None:
        path = _touch(tmp_path, "huge.bin")
        mock_stat = MagicMock()
        mock_stat.st_size = 60 * 1024 * 1024
        mock_stat.st_mode = stat_module.S_IFREG | 0o644
        with (
            patch("subprocess.run", return_value=_git_result(1)),
            patch.object(Path, "stat", return_value=mock_stat),
            patch.object(Path, "is_file", return_value=True),
        ):
            result = validate_sendable(path, tmp_path)
        assert result is not None
        assert "too large" in result

    def test_unstattable_file_denied(self, tmp_path: Path) -> None:
        path = _touch(tmp_path, "vanished.txt")
        with (
            patch("subprocess.run", return_value=_git_result(1)),
            patch.object(Path, "stat", side_effect=OSError("gone")),
        ):
            assert validate_sendable(path, tmp_path) == "File not accessible"

    def test_not_regular_file_denied(self, tmp_path: Path) -> None:
        directory = tmp_path / "subdir"
        directory.mkdir()
        with patch("subprocess.run", return_value=_git_result(1)):
            assert validate_sendable(directory, tmp_path) == "Not a regular file"

    def test_state_file_denied(self, tmp_path: Path) -> None:
        path = _touch(tmp_path, "state.json")
        with (
            patch("subprocess.run", return_value=_git_result(1)),
            patch("ccgram.utils.ccgram_dir", return_value=tmp_path),
        ):
            result = validate_sendable(path, tmp_path)
        assert result is not None
        assert "state" in result.lower() or "refusing" in result.lower()


class TestIsExcludedDir:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param("node_modules", True, id="node_modules"),
            pytest.param("__pycache__", True, id="pycache"),
            pytest.param("dist", True, id="dist"),
            pytest.param("build", True, id="build"),
            pytest.param(".venv", True, id="venv"),
            pytest.param(".git", True, id="git"),
            pytest.param(".mydir", True, id="any-dot-prefix"),
            pytest.param("src", False, id="src"),
            pytest.param("tests", False, id="tests"),
        ],
    )
    def test_exclusion(self, name: str, expected: bool) -> None:
        assert is_excluded_dir(name) is expected
