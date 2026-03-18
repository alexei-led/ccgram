"""Tests for directory browser favorites and hidden dirs."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ccgram.handlers.directory_browser import build_directory_browser, get_favorites
from ccgram.session import SessionManager


@pytest.fixture
def mock_session_manager():
    with patch(
        "ccgram.handlers.directory_browser.session_manager", spec=SessionManager
    ) as mgr:
        mgr.get_user_starred.return_value = []
        mgr.get_user_mru.return_value = []
        yield mgr


class TestGetFavorites:
    def test_none_user_id_returns_empty(self) -> None:
        favorites, starred = get_favorites(None)
        assert favorites == []
        assert starred == set()

    def test_empty_when_no_favorites(self, mock_session_manager: Mock) -> None:
        favorites, starred = get_favorites(100)
        assert favorites == []
        assert starred == set()

    def test_starred_first_then_mru(
        self, tmp_path: Path, mock_session_manager: Mock
    ) -> None:
        starred_dir = str(tmp_path / "starred")
        mru_dirs = [str(tmp_path / "mru1"), str(tmp_path / "mru2")]
        for d in [starred_dir, *mru_dirs]:
            Path(d).mkdir()

        mock_session_manager.get_user_starred.return_value = [starred_dir]
        mock_session_manager.get_user_mru.return_value = mru_dirs

        favorites, starred = get_favorites(100)
        assert favorites == [starred_dir, *mru_dirs]
        assert starred == {starred_dir}

    @pytest.mark.parametrize(
        ("starred_names", "mru_names", "expected_count"),
        [
            (["exists", "missing"], [], 1),
            (["dup"], ["dup"], 1),
        ],
        ids=["filters_nonexistent", "deduplicates"],
    )
    def test_filtering(
        self,
        tmp_path: Path,
        mock_session_manager: Mock,
        starred_names: list[str],
        mru_names: list[str],
        expected_count: int,
    ) -> None:
        for name in {*starred_names, *mru_names} - {"missing"}:
            (tmp_path / name).mkdir()

        mock_session_manager.get_user_starred.return_value = [
            str(tmp_path / n) for n in starred_names
        ]
        mock_session_manager.get_user_mru.return_value = [
            str(tmp_path / n) for n in mru_names
        ]

        favorites, _starred = get_favorites(100)
        assert len(favorites) == expected_count

    def test_caps_at_three(self, tmp_path: Path, mock_session_manager: Mock) -> None:
        dirs = [tmp_path / f"dir{i}" for i in range(5)]
        for d in dirs:
            d.mkdir()

        mock_session_manager.get_user_starred.return_value = [
            str(dirs[0]),
            str(dirs[1]),
        ]
        mock_session_manager.get_user_mru.return_value = [str(d) for d in dirs[2:]]

        favorites, _starred = get_favorites(100)
        assert len(favorites) == 3

    def test_handles_oserror_on_is_dir(self, mock_session_manager: Mock) -> None:
        mock_session_manager.get_user_starred.return_value = ["/invalid/path"]

        favorites, _starred = get_favorites(100)
        assert favorites == []


class TestHiddenDirs:
    def test_hidden_dirs_excluded_by_default(
        self, tmp_path: Path, mock_session_manager: Mock
    ) -> None:
        (tmp_path / "visible").mkdir()
        (tmp_path / ".hidden").mkdir()

        with patch("ccgram.handlers.directory_browser.config") as mock_cfg:
            mock_cfg.show_hidden_dirs = False
            _text, _kb, subdirs = build_directory_browser(str(tmp_path))

        assert "visible" in subdirs
        assert ".hidden" not in subdirs

    def test_hidden_dirs_shown_when_enabled(
        self, tmp_path: Path, mock_session_manager: Mock
    ) -> None:
        (tmp_path / "visible").mkdir()
        (tmp_path / ".hidden").mkdir()

        with patch("ccgram.handlers.directory_browser.config") as mock_cfg:
            mock_cfg.show_hidden_dirs = True
            _text, _kb, subdirs = build_directory_browser(str(tmp_path))

        assert "visible" in subdirs
        assert ".hidden" in subdirs
