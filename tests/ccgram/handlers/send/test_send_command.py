"""Tests for src/ccgram/handlers/send/send_command.py."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import InlineKeyboardMarkup
from telegram.error import TelegramError

from ccgram.config import config
from ccgram.handlers.callback_data import (
    CB_SEND_CANCEL,
    CB_SEND_DIR,
    CB_SEND_FILE,
    CB_SEND_PAGE,
    CB_SEND_UP,
)
from ccgram.handlers.send.send_command import (
    _find_files,
    _format_file_label,
    _is_image,
    _list_directory,
    build_file_browser,
    build_search_results,
    send_command,
    upload_file,
)
from ccgram.handlers.user_state import (
    SEND_CWD_KEY,
    SEND_ITEMS_KEY,
    SEND_PAGE_KEY,
    SEND_PATH_KEY,
    SEND_WINDOW_ID_KEY,
)

_MOD = "ccgram.handlers.send.send_command"


@pytest.fixture
def allow_all_files() -> Iterator[MagicMock]:
    """Neutralise the security layer — its own behaviour is tested separately."""
    with patch(f"{_MOD}.validate_sendable", return_value=None) as mock:
        yield mock


def _make_file(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _callback_data(markup: InlineKeyboardMarkup) -> list[object]:
    return [btn.callback_data for row in markup.inline_keyboard for btn in row]


def _deny_all_but(allowed: str):
    def _validate(path: Path, cwd: Path) -> str | None:
        return None if path.name == allowed else "denied"

    return _validate


class TestIsImage:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param("photo.png", True, id="png"),
            pytest.param("photo.jpg", True, id="jpg"),
            pytest.param("photo.jpeg", True, id="jpeg"),
            pytest.param("anim.gif", True, id="gif"),
            pytest.param("image.webp", True, id="webp"),
            pytest.param("PHOTO.PNG", True, id="png-uppercase"),
            pytest.param("Img.JPG", True, id="jpg-mixed-case"),
            pytest.param("readme.txt", False, id="txt"),
            pytest.param("main.py", False, id="py"),
            pytest.param("report.pdf", False, id="pdf"),
            pytest.param("Makefile", False, id="no-extension"),
        ],
    )
    def test_extension_detection(
        self, tmp_path: Path, name: str, expected: bool
    ) -> None:
        assert _is_image(tmp_path / name) is expected


@pytest.mark.usefixtures("allow_all_files")
class TestFindFiles:
    @pytest.mark.parametrize(
        ("pattern", "expected"),
        [
            pytest.param("*.txt", {"a.txt", "b.txt"}, id="glob-star"),
            pytest.param("?.txt", {"a.txt", "b.txt"}, id="glob-question-mark"),
            pytest.param("a.txt", {"a.txt"}, id="exact-name"),
            pytest.param("a", {"a.txt"}, id="substring-fallback"),
            pytest.param("*.xyz", set(), id="no-match"),
        ],
    )
    def test_pattern_dispatch(
        self, tmp_path: Path, pattern: str, expected: set[str]
    ) -> None:
        for name in ("a.txt", "b.txt", "c.py"):
            _make_file(tmp_path / name)
        assert {p.name for p in _find_files(tmp_path, pattern)} == expected

    def test_exact_relative_path_returned_directly(self, tmp_path: Path) -> None:
        target = _make_file(tmp_path / "sub" / "report.txt")
        assert _find_files(tmp_path, "sub/report.txt") == [target]

    def test_depth_limit_respected(self, tmp_path: Path) -> None:
        shallow = _make_file(tmp_path / "a" / "file.txt")
        deep = _make_file(tmp_path / "a" / "b" / "c" / "deep.txt")
        with patch.object(config, "send_search_depth", 2):
            results = _find_files(tmp_path, "*.txt")
        assert shallow in results
        assert deep not in results

    def test_excluded_dirs_skipped(self, tmp_path: Path) -> None:
        normal = _make_file(tmp_path / "src" / "module.txt")
        excluded = _make_file(tmp_path / "node_modules" / "dep.txt")
        results = _find_files(tmp_path, "*.txt")
        assert normal in results
        assert excluded not in results

    def test_max_results_cap(self, tmp_path: Path) -> None:
        for i in range(10):
            _make_file(tmp_path / f"file{i}.txt")
        with patch.object(config, "send_max_results", 3):
            assert len(_find_files(tmp_path, "*.txt")) == 3

    def test_sorted_by_mtime_desc(self, tmp_path: Path) -> None:
        old = _make_file(tmp_path / "old.txt")
        new = _make_file(tmp_path / "new.txt")
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))
        assert _find_files(tmp_path, "*.txt") == [new, old]

    def test_denied_files_filtered_out(self, tmp_path: Path) -> None:
        _make_file(tmp_path / "allowed.txt")
        _make_file(tmp_path / "denied.txt")
        with patch(
            f"{_MOD}.validate_sendable", side_effect=_deny_all_but("allowed.txt")
        ):
            results = _find_files(tmp_path, "*.txt")
        assert {p.name for p in results} == {"allowed.txt"}


@pytest.mark.usefixtures("allow_all_files")
class TestListDirectory:
    def test_dirs_and_files_separated(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        _make_file(tmp_path / "file.txt")
        dirs, files = _list_directory(tmp_path, tmp_path)
        assert [p.name for p in dirs] == ["subdir"]
        assert [p.name for p in files] == ["file.txt"]

    @pytest.mark.parametrize("kind", ["dirs", "files"])
    def test_entries_sorted_case_insensitively(self, tmp_path: Path, kind: str) -> None:
        for name in ("zebra", "Alpha", "middle"):
            if kind == "dirs":
                (tmp_path / name).mkdir()
            else:
                _make_file(tmp_path / f"{name}.txt")
        dirs, files = _list_directory(tmp_path, tmp_path)
        names = [p.name for p in (dirs if kind == "dirs" else files)]
        assert names == sorted(names, key=str.lower)

    @pytest.mark.parametrize(
        "excluded", ["node_modules", "__pycache__", ".git"], ids=lambda n: n.strip(".")
    )
    def test_noise_and_hidden_dirs_excluded(
        self, tmp_path: Path, excluded: str
    ) -> None:
        (tmp_path / excluded).mkdir()
        (tmp_path / "src").mkdir()
        dirs, _ = _list_directory(tmp_path, tmp_path)
        assert [p.name for p in dirs] == ["src"]

    def test_denied_files_excluded(self, tmp_path: Path) -> None:
        _make_file(tmp_path / "allowed.txt")
        _make_file(tmp_path / "secret.pem")
        with patch(
            f"{_MOD}.validate_sendable", side_effect=_deny_all_but("allowed.txt")
        ):
            _, files = _list_directory(tmp_path, tmp_path)
        assert [p.name for p in files] == ["allowed.txt"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert _list_directory(tmp_path, tmp_path) == ([], [])


class TestFormatFileLabel:
    @pytest.mark.parametrize(
        ("size", "expected_size_text"),
        [
            pytest.param(500, "500 B", id="bytes"),
            pytest.param(2048, "2.0 KB", id="kilobytes"),
            pytest.param(2 * 1024 * 1024, "2.0 MB", id="megabytes"),
        ],
    )
    def test_human_readable_size(
        self, tmp_path: Path, size: int, expected_size_text: str
    ) -> None:
        path = tmp_path / "file.bin"
        path.write_bytes(b"x" * size)
        assert expected_size_text in _format_file_label(path, tmp_path)

    def test_short_path_kept_verbatim(self, tmp_path: Path) -> None:
        path = tmp_path / "hi.txt"
        path.write_bytes(b"x" * 10)
        label = _format_file_label(path, tmp_path)
        assert label.startswith("hi.txt")
        assert "…" not in label

    @pytest.mark.parametrize(
        "relative",
        [
            pytest.param("very/long/nested/somefile_with_a_long_name.txt", id="nested"),
            pytest.param("a" * 27 + "/" + "b" * 27 + ".txt", id="long-names"),
        ],
    )
    def test_long_path_truncated_but_keeps_size_suffix(
        self, tmp_path: Path, relative: str
    ) -> None:
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x" * 1024)
        label = _format_file_label(path, tmp_path)
        assert len(label) <= 30
        assert "…" in label
        assert "KB" in label

    def test_relative_path_used_inside_cwd(self, tmp_path: Path) -> None:
        path = _make_file(tmp_path / "src" / "module.py")
        assert _format_file_label(path, tmp_path).startswith("src/module.py")

    def test_outside_cwd_falls_back_to_name(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "other.txt"
        outside.write_bytes(b"x" * 10)
        assert "other.txt" in _format_file_label(outside, tmp_path)


class TestBuildFileBrowser:
    def test_returns_text_markup_and_item_paths(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        _make_file(tmp_path / "file.txt")
        text, markup, items = build_file_browser(tmp_path, tmp_path, 0)
        assert "\U0001f4c2" in text
        assert isinstance(markup, InlineKeyboardMarkup)
        assert [p.name for p in items] == ["subdir", "file.txt"]

    def test_dirs_listed_before_files(self, tmp_path: Path) -> None:
        (tmp_path / "adir").mkdir()
        _make_file(tmp_path / "zfile.txt")
        _, _, items = build_file_browser(tmp_path, tmp_path, 0)
        dir_indices = [i for i, p in enumerate(items) if p.is_dir()]
        file_indices = [i for i, p in enumerate(items) if p.is_file()]
        assert max(dir_indices) < min(file_indices)

    @pytest.mark.parametrize(
        ("relative", "prefix"),
        [
            pytest.param("mydir/", CB_SEND_DIR, id="dir-button"),
            pytest.param("report.txt", CB_SEND_FILE, id="file-button"),
        ],
    )
    def test_entry_buttons_use_their_callback_prefix(
        self, tmp_path: Path, relative: str, prefix: str
    ) -> None:
        if relative.endswith("/"):
            (tmp_path / relative.rstrip("/")).mkdir()
        else:
            _make_file(tmp_path / relative)
        _, markup, _ = build_file_browser(tmp_path, tmp_path, 0)
        assert any(
            isinstance(cb, str) and cb.startswith(prefix)
            for cb in _callback_data(markup)
        )

    def test_item_count_matches_dirs_plus_files(self, tmp_path: Path) -> None:
        for name in ("d1", "d2"):
            (tmp_path / name).mkdir()
        for name in ("f1.txt", "f2.txt"):
            _make_file(tmp_path / name)
        _, _, items = build_file_browser(tmp_path, tmp_path, 0)
        assert len(items) == 4

    def test_pages_show_different_files(self, tmp_path: Path) -> None:
        for i in range(12):
            _make_file(tmp_path / f"file{i:02d}.txt")

        def _file_buttons(page: int) -> set[str]:
            _, markup, _ = build_file_browser(tmp_path, tmp_path, page)
            return {
                cb
                for cb in _callback_data(markup)
                if isinstance(cb, str) and cb.startswith(CB_SEND_FILE)
            }

        assert _file_buttons(0)
        assert _file_buttons(0).isdisjoint(_file_buttons(1))

    def test_pagination_controls_only_when_multipage(self, tmp_path: Path) -> None:
        _make_file(tmp_path / "only.txt")
        _, single, _ = build_file_browser(tmp_path, tmp_path, 0)
        assert not any(
            isinstance(cb, str) and cb.startswith(CB_SEND_PAGE)
            for cb in _callback_data(single)
        )

        for i in range(12):
            _make_file(tmp_path / f"file{i:02d}.txt")
        _, multi, _ = build_file_browser(tmp_path, tmp_path, 0)
        assert any(
            isinstance(cb, str) and cb.startswith(CB_SEND_PAGE)
            for cb in _callback_data(multi)
        )
        assert any("/" in btn.text for row in multi.inline_keyboard for btn in row)

    def test_parent_button_only_below_cwd(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        _, at_root, _ = build_file_browser(tmp_path, tmp_path, 0)
        _, below_root, _ = build_file_browser(sub, tmp_path, 0)
        assert CB_SEND_UP not in _callback_data(at_root)
        assert CB_SEND_UP in _callback_data(below_root)

    def test_cancel_button_always_present(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        items: list[Path] = [Path()]
        for directory in (tmp_path, sub):
            _, markup, items = build_file_browser(directory, tmp_path, 0)
            assert CB_SEND_CANCEL in _callback_data(markup)
        assert items == []  # last iteration is the empty subdirectory


class TestBuildSearchResults:
    def test_one_button_per_match_plus_cancel(self, tmp_path: Path) -> None:
        paths = [_make_file(tmp_path / name) for name in ("alpha.txt", "beta.txt")]
        text, markup, shown = build_search_results(paths, tmp_path)
        callbacks = _callback_data(markup)
        file_cbs = [
            cb
            for cb in callbacks
            if isinstance(cb, str) and cb.startswith(CB_SEND_FILE)
        ]
        assert len(file_cbs) == 2
        assert CB_SEND_CANCEL in callbacks
        assert shown == paths
        assert "2" in text

    def test_results_capped_at_three_pages(self, tmp_path: Path) -> None:
        paths = [_make_file(tmp_path / f"f{i:02d}.txt") for i in range(30)]
        _, _, shown = build_search_results(paths, tmp_path)
        assert len(shown) == 24

    def test_empty_list_offers_only_cancel(self, tmp_path: Path) -> None:
        _, markup, shown = build_search_results([], tmp_path)
        callbacks = _callback_data(markup)
        assert shown == []
        assert CB_SEND_CANCEL in callbacks
        assert not any(
            isinstance(cb, str) and cb.startswith(CB_SEND_FILE) for cb in callbacks
        )


class TestUploadFile:
    @pytest.mark.parametrize(
        ("name", "sender"),
        [
            pytest.param("photo.png", "send_photo", id="png-as-photo"),
            pytest.param("img.jpg", "send_photo", id="jpg-as-photo"),
            pytest.param("report.pdf", "send_document", id="pdf-as-document"),
        ],
    )
    async def test_routes_by_file_type(
        self, tmp_path: Path, name: str, sender: str
    ) -> None:
        path = _make_file(tmp_path / name)
        bot = AsyncMock()
        await upload_file(bot, chat_id=-100, thread_id=5, path=path)
        used = getattr(bot, sender)
        unused = bot.send_document if sender == "send_photo" else bot.send_photo
        used.assert_awaited_once()
        unused.assert_not_awaited()
        assert used.call_args.kwargs["chat_id"] == -100
        assert used.call_args.kwargs["message_thread_id"] == 5
        assert used.call_args.kwargs["filename"] == name

    async def test_telegram_error_is_reraised(self, tmp_path: Path) -> None:
        path = _make_file(tmp_path / "file.txt")
        bot = AsyncMock()
        bot.send_document.side_effect = TelegramError("flood")
        with pytest.raises(TelegramError):
            await upload_file(bot, chat_id=-100, thread_id=5, path=path)


def _make_update(
    user_id: int = 1, thread_id: int = 42, text: str = "/send"
) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    msg = AsyncMock()
    msg.text = text
    msg.message_thread_id = thread_id
    msg.chat_id = -100123
    update.message = msg
    update.callback_query = None
    return update


def _make_context(user_data: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.user_data = {} if user_data is None else user_data
    ctx.bot = AsyncMock()
    return ctx


class TestSendCommand:
    @pytest.fixture(autouse=True)
    def _patches(self) -> Iterator[None]:
        with (
            patch("ccgram.config.Config.is_user_allowed", return_value=True),
            patch(f"{_MOD}.thread_router") as mock_tr,
            patch(f"{_MOD}.view_window") as mock_view,
            patch(f"{_MOD}.safe_reply", new_callable=AsyncMock) as mock_reply,
        ):
            self.mock_tr = mock_tr
            self.mock_view = mock_view
            self.mock_reply = mock_reply
            mock_tr.resolve_window_for_thread.return_value = "@1"
            mock_tr.resolve_chat_id.return_value = -100123
            ws = MagicMock()
            ws.cwd = None  # overridden per test
            mock_view.return_value = ws
            self.ws = ws
            yield

    def _reply_text(self) -> str:
        self.mock_reply.assert_awaited_once()
        return self.mock_reply.await_args_list[-1][0][1]

    async def test_no_message_returns_early(self) -> None:
        update = MagicMock()
        update.message = None
        ctx = _make_context()

        await send_command(update, ctx)

        self.mock_reply.assert_not_awaited()
        assert ctx.user_data == {}

    async def test_unauthorized_user_rejected(self) -> None:
        ctx = _make_context()
        with patch("ccgram.config.Config.is_user_allowed", return_value=False):
            await send_command(_make_update(), ctx)

        assert self._reply_text() == "Not authorized."
        assert SEND_WINDOW_ID_KEY not in ctx.user_data

    async def test_general_topic_rejected(self) -> None:
        ctx = _make_context()

        await send_command(_make_update(thread_id=1), ctx)

        assert self._reply_text() == "Use this command inside a topic."
        assert SEND_WINDOW_ID_KEY not in ctx.user_data

    async def test_unbound_topic_rejected(self) -> None:
        self.mock_tr.resolve_window_for_thread.return_value = None
        ctx = _make_context()

        await send_command(_make_update(), ctx)

        assert self._reply_text() == "No session bound to this topic."

    @pytest.mark.parametrize(
        "view_state",
        [
            pytest.param("missing-cwd", id="cwd-does-not-exist"),
            pytest.param("no-window-state", id="window-state-wiped-after-binding"),
        ],
    )
    async def test_unusable_working_directory_rejected(
        self, tmp_path: Path, view_state: str
    ) -> None:
        if view_state == "missing-cwd":
            self.ws.cwd = str(tmp_path / "nonexistent")
        else:
            self.mock_view.return_value = None
        ctx = _make_context()

        await send_command(_make_update(), ctx)

        assert self._reply_text() == "Working directory not available."

    @pytest.mark.usefixtures("allow_all_files")
    async def test_no_args_caches_browser_state(self, tmp_path: Path) -> None:
        self.ws.cwd = str(tmp_path)
        _make_file(tmp_path / "file.txt")
        ctx = _make_context()

        await send_command(_make_update(text="/send"), ctx)

        assert ctx.user_data[SEND_PATH_KEY] == str(tmp_path)
        assert ctx.user_data[SEND_CWD_KEY] == str(tmp_path)
        assert ctx.user_data[SEND_PAGE_KEY] == 0
        assert ctx.user_data[SEND_WINDOW_ID_KEY] == "@1"
        assert [p.name for p in ctx.user_data[SEND_ITEMS_KEY]] == ["file.txt"]

    @pytest.mark.usefixtures("allow_all_files")
    @pytest.mark.parametrize(
        "pattern",
        [pytest.param("*.txt", id="glob"), pytest.param("report", id="substring")],
    )
    async def test_single_match_uploads(self, tmp_path: Path, pattern: str) -> None:
        self.ws.cwd = str(tmp_path)
        target = _make_file(tmp_path / "report.txt")
        ctx = _make_context()

        with patch(f"{_MOD}.upload_file", new_callable=AsyncMock) as mock_up:
            await send_command(_make_update(text=f"/send {pattern}"), ctx)

        mock_up.assert_awaited_once()
        assert mock_up.call_args[0][1:] == (-100123, 42, target)
        assert mock_up.call_args[0][0].bot is ctx.bot

    @pytest.mark.usefixtures("allow_all_files")
    async def test_exact_path_uploads(self, tmp_path: Path) -> None:
        self.ws.cwd = str(tmp_path)
        target = _make_file(tmp_path / "exact.txt")
        ctx = _make_context()

        with patch(f"{_MOD}.upload_file", new_callable=AsyncMock) as mock_up:
            await send_command(_make_update(text="/send exact.txt"), ctx)

        mock_up.assert_awaited_once()
        assert mock_up.call_args[0][1:] == (-100123, 42, target)

    @pytest.mark.usefixtures("allow_all_files")
    async def test_multiple_matches_show_picker(self, tmp_path: Path) -> None:
        self.ws.cwd = str(tmp_path)
        for i in range(3):
            _make_file(tmp_path / f"file{i}.txt")
        ctx = _make_context()

        with patch(f"{_MOD}.upload_file", new_callable=AsyncMock) as mock_up:
            await send_command(_make_update(text="/send *.txt"), ctx)

        mock_up.assert_not_awaited()
        assert len(ctx.user_data[SEND_ITEMS_KEY]) == 3

    @pytest.mark.usefixtures("allow_all_files")
    async def test_no_match_reports_pattern(self, tmp_path: Path) -> None:
        self.ws.cwd = str(tmp_path)
        ctx = _make_context()

        await send_command(_make_update(text="/send *.xyz"), ctx)

        assert self._reply_text() == "No files found matching: *.xyz"

    async def test_exact_path_denied_by_security(self, tmp_path: Path) -> None:
        self.ws.cwd = str(tmp_path)
        _make_file(tmp_path / "secret.key")
        ctx = _make_context()

        with (
            patch(
                f"{_MOD}.validate_sendable",
                return_value="File appears to contain credentials",
            ),
            patch(f"{_MOD}.upload_file", new_callable=AsyncMock) as mock_up,
        ):
            await send_command(_make_update(text="/send secret.key"), ctx)

        mock_up.assert_not_awaited()
        assert "credentials" in self._reply_text()

    @pytest.mark.usefixtures("allow_all_files")
    async def test_exact_path_escaping_cwd_via_symlink_denied(
        self, tmp_path: Path
    ) -> None:
        cwd = tmp_path / "project"
        cwd.mkdir()
        self.ws.cwd = str(cwd)
        outside = _make_file(tmp_path / "outside.txt")
        (cwd / "link.txt").symlink_to(outside)
        ctx = _make_context()

        with patch(f"{_MOD}.upload_file", new_callable=AsyncMock) as mock_up:
            await send_command(_make_update(text="/send link.txt"), ctx)

        mock_up.assert_not_awaited()
        assert self._reply_text() == "Cannot send: file is outside project directory"

    @pytest.mark.usefixtures("allow_all_files")
    async def test_exact_path_in_excluded_dir_denied(self, tmp_path: Path) -> None:
        self.ws.cwd = str(tmp_path)
        _make_file(tmp_path / "node_modules" / "dep.txt")
        ctx = _make_context()

        with patch(f"{_MOD}.upload_file", new_callable=AsyncMock) as mock_up:
            await send_command(_make_update(text="/send node_modules/dep.txt"), ctx)

        mock_up.assert_not_awaited()
        assert self._reply_text() == "Cannot send: file is in an excluded directory"
