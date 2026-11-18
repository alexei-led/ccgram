"""Tests for file_handler helper functions."""

import errno
import re
import unicodedata
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from ccgram.handlers import file_handler
from ccgram.handlers.file_handler import (
    _generate_photo_filename,
    _sanitize_caption,
    _sanitize_filename,
    _unique_dest,
    _validate_dest_path,
)


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            # ASCII names pass through untouched
            ("document.pdf", "document.pdf"),
            ("file-name_123.txt", "file-name_123.txt"),
            # path traversal is stripped to the basename
            ("/etc/passwd", "passwd"),
            ("../../../etc/passwd", "passwd"),
            ("../../etc/passwd", "passwd"),
            # punctuation and whitespace still collapse to underscores
            ("hello world!.txt", "hello_world_.txt"),
            ("file@#$.txt", "file___.txt"),
            ("quote'and\"double.txt", "quote_and_double.txt"),
            ("semi;colon&amp.sh", "semi_colon_amp.sh"),
            ("tab\tand\nnewline.txt", "tab_and_newline.txt"),
            # names that sanitize down to nothing usable
            ("..", "unnamed"),
            (".", "unnamed"),
            ("...", "unnamed"),
            ("", "unnamed"),
            # non-Latin scripts survive, which is the point of this test
            ("Отчёт за 2025.pdf", "Отчёт_за_2025.pdf"),
            ("договор.docx", "договор.docx"),
            ("Λογαριασμός.pdf", "Λογαριασμός.pdf"),
            ("请求书.xlsx", "请求书.xlsx"),
            ("領収書.pdf", "領収書.pdf"),
            ("فاتورة.pdf", "فاتورة.pdf"),
            ("חשבונית.pdf", "חשבונית.pdf"),
            # Thai writes vowels as combining marks, which are not word
            # characters, so a \\w-only rule would punch holes in the word
            ("ใบเสร็จ.pdf", "ใบเสร็จ.pdf"),
            ("नमस्ते.pdf", "नमस्ते.pdf"),
            ("송장.pdf", "송장.pdf"),
            # mixed scripts and digits keep both halves
            ("invoice-Счёт-2025.pdf", "invoice-Счёт-2025.pdf"),
            ("café.txt", "café.txt"),
            # non-word characters go regardless of script
            ("Счёт №5-2025.pdf", "Счёт__5-2025.pdf"),
            ("файл (копия).txt", "файл__копия_.txt"),
            # emoji and other symbols are not word characters
            ("report📊.pdf", "report_.pdf"),
            # bidi override cannot be smuggled into the saved name
            ("safe‮gnp.exe", "safe_gnp.exe"),
            # a traversal attempt written in Cyrillic is still a basename
            ("../отчёты/итог.pdf", "итог.pdf"),
        ],
    )
    def test_sanitize(self, input_name: str, expected: str) -> None:
        assert _sanitize_filename(input_name) == expected

    def test_composes_before_substituting(self) -> None:
        """A decomposed name keeps its accents.

        macOS reports names in NFD, where the combining mark is a separate
        character and not a word character, so without composing first it
        would be replaced on its own and leave a bare letter behind.
        """
        decomposed = unicodedata.normalize("NFD", "Отчёт.pdf")
        assert decomposed != "Отчёт.pdf"
        assert _sanitize_filename(decomposed) == "Отчёт.pdf"

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("Отчёт.pdf", "Договор.pdf"),
            ("请求书.xlsx", "领收书.xlsx"),
            ("فاتورة.pdf", "ايصال.pdf"),
        ],
    )
    def test_distinct_non_latin_names_stay_distinct(
        self, first: str, second: str
    ) -> None:
        """Two names of equal length used to sanitize to the same underscores.

        The saved file then shadowed the earlier one, and the message quoting
        the path said nothing about which document it was.
        """
        assert _sanitize_filename(first) != _sanitize_filename(second)
        assert "_" not in _sanitize_filename(first)

    def test_truncates_long_names_preserving_extension(self) -> None:
        long = "a" * 250 + ".pdf"
        result = _sanitize_filename(long)
        assert len(result) <= 200
        assert result.endswith(".pdf")

    def test_truncates_multibyte_name_by_encoded_size(self) -> None:
        result = _sanitize_filename("界" * 200 + ".pdf")
        assert len(result.encode()) <= 200
        assert result.endswith(".pdf")


class TestUniqueDest:
    def test_returns_original_if_not_exists(self, tmp_path: Path) -> None:
        assert _unique_dest(tmp_path / "file.txt") == tmp_path / "file.txt"

    @pytest.mark.parametrize(
        ("existing_files", "expected_name"),
        [
            (["file.txt"], "file_1.txt"),
            (["file.txt", "file_1.txt", "file_2.txt"], "file_3.txt"),
            (["file"], "file_1"),
        ],
    )
    def test_increments_suffix(
        self, tmp_path: Path, existing_files: list[str], expected_name: str
    ) -> None:
        for name in existing_files:
            (tmp_path / name).write_text("x")
        assert _unique_dest(tmp_path / existing_files[0]) == tmp_path / expected_name

    def test_fallback_to_timestamp_after_100(self, tmp_path: Path) -> None:
        dest = tmp_path / "file.txt"
        for i in range(100):
            name = "file.txt" if i == 0 else f"file_{i}.txt"
            (tmp_path / name).write_text(str(i))
        result = _unique_dest(dest)
        assert result.name.startswith("file_") and result.name.endswith(".txt")
        assert result != dest

    def test_broken_symlink_treated_as_existing(self, tmp_path: Path) -> None:
        dest = tmp_path / "file.txt"
        dest.symlink_to(tmp_path / "nonexistent_target")
        assert _unique_dest(dest) == tmp_path / "file_1.txt"


class TestValidateDestPath:
    @pytest.mark.parametrize(
        ("rel_dest", "expected"),
        [
            ("file.txt", True),
            ("subdir/file.txt", True),
            ("../outside.txt", False),
        ],
    )
    def test_path_validation(
        self, tmp_path: Path, rel_dest: str, expected: bool
    ) -> None:
        upload = tmp_path / "upload"
        upload.mkdir()
        if "/" in rel_dest and not rel_dest.startswith(".."):
            (upload / Path(rel_dest).parent).mkdir(parents=True, exist_ok=True)
        assert _validate_dest_path(upload / rel_dest, upload) is expected

    def test_rejects_absolute_path_outside(self, tmp_path: Path) -> None:
        upload = tmp_path / "upload"
        upload.mkdir()
        assert _validate_dest_path(tmp_path / "outside.txt", upload) is False


class TestSanitizeCaption:
    @pytest.mark.parametrize(
        ("input_text", "expected"),
        [
            ("", ""),
            ("hello\x00\x01\x02world", "helloworld"),
            ("hello\x07\x1bworld", "helloworld"),
            ("line1\nline2\r\nline3\ttab", "line1 line2  line3\ttab"),
        ],
    )
    def test_sanitize(self, input_text: str, expected: str) -> None:
        assert _sanitize_caption(input_text) == expected

    def test_limits_to_500_chars(self) -> None:
        assert len(_sanitize_caption("a" * 600)) == 500


class TestGeneratePhotoFilename:
    def test_format(self) -> None:
        result = _generate_photo_filename("ABCDEFGHIJKLMNOP")
        assert re.match(r"^photo_\d{8}_\d{6}_ABCDEFGH\.jpg$", result)


@pytest.mark.parametrize("timeout_at", ["get_file", "download"])
async def test_photo_timeout_retries_with_caption(
    tmp_path: Path, timeout_at: str
) -> None:
    upload_dir = tmp_path / ".ccgram-uploads"
    filename = "photo_test.jpg"
    message = MagicMock()
    message.photo = [
        SimpleNamespace(file_unique_id="photo", file_id="fid", file_size=5)
    ]
    message.caption = "please\nlook\x00 here"
    message.chat.id = -100
    message.chat.send_action = AsyncMock()
    message.message_thread_id = 42

    attempts: list[Path] = []

    async def download(path: str) -> Path:
        dest = Path(path)
        attempts.append(dest)
        if timeout_at == "download" and len(attempts) == 1:
            dest.write_bytes(b"partial")
            # A different upload claims the intended name while this one is in flight.
            (upload_dir / filename).write_bytes(b"other upload")
            raise TimedOut()
        assert not dest.exists()
        dest.write_bytes(b"photo")
        return dest

    photo = SimpleNamespace(download_to_drive=download)
    get_file = AsyncMock(
        side_effect=[TimedOut() if timeout_at == "get_file" else photo, photo]
    )
    message.get_bot.return_value.get_file = get_file

    with (
        patch.object(
            file_handler, "_resolve_upload_dir", return_value=("@0", upload_dir, None)
        ),
        patch.object(file_handler, "_generate_photo_filename", return_value=filename),
        patch.object(file_handler.config, "is_user_allowed", return_value=True),
        patch.object(file_handler, "adapter_for"),
        patch.object(
            file_handler,
            "send_telegram_to_window",
            new_callable=AsyncMock,
            return_value=(True, "ok"),
        ) as send,
        patch.object(file_handler, "ack_reaction", new_callable=AsyncMock) as ack,
        patch.object(file_handler, "safe_reply", new_callable=AsyncMock) as reply,
    ):
        await file_handler.handle_photo_message(
            MagicMock(effective_user=SimpleNamespace(id=1), message=message),
            MagicMock(),
        )

    assert get_file.await_args_list == [call("fid"), call("fid")]
    saved_name = "photo_test_1.jpg" if timeout_at == "download" else filename
    saved = upload_dir / saved_name
    expected_files = {saved_name: b"photo"}
    if timeout_at == "download":
        expected_files[filename] = b"other upload"
    assert {
        path.name: path.read_bytes() for path in upload_dir.iterdir()
    } == expected_files
    assert not any(path.exists() for path in attempts)
    send.assert_awaited_once()
    assert send.await_args is not None
    assert str(saved) in send.await_args.args[3]
    assert send.await_args.args[3].endswith("\n\nUser note: please look here")
    ack.assert_awaited_once()
    reply.assert_awaited_once_with(
        message, f"📷 Uploaded `.ccgram-uploads/{saved_name}`"
    )


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        (NetworkError("reset"), 3),
        (BadRequest("invalid file"), 1),
        (RetryAfter(1), 1),
        (OSError("disk full"), 1),
    ],
)
async def test_download_failure_cleans_staging(
    tmp_path: Path, error: Exception, attempts: int
) -> None:
    async def download(path: Path) -> None:
        path.write_bytes(b"partial")
        raise error

    original = tmp_path / "report.txt"
    original.write_bytes(b"previous upload")
    message = MagicMock()
    get_file = AsyncMock(return_value=SimpleNamespace(download_to_drive=download))
    message.get_bot.return_value.get_file = get_file
    with patch.object(file_handler, "safe_reply", new_callable=AsyncMock) as reply:
        result = await file_handler._download_and_save(
            message, tmp_path, original.name, "fid", None, "File"
        )

    assert result is None
    assert get_file.await_args_list == [call("fid")] * attempts
    assert list(tmp_path.iterdir()) == [original]
    assert original.read_bytes() == b"previous upload"
    reply.assert_awaited_once_with(message, "❌ Failed to save file.")


@pytest.mark.parametrize("link_error", sorted({errno.ENOTSUP, errno.EOPNOTSUPP}))
async def test_upload_without_hard_links_reports_nested_path_once(
    tmp_path: Path, link_error: int
) -> None:
    upload_dir = tmp_path / ".ccgram-uploads"
    upload_dir.mkdir()
    original = upload_dir / "report.txt"
    original.write_bytes(b"previous upload")

    async def download(path: Path) -> None:
        path.write_bytes(b"complete download")

    message = MagicMock(caption=None, message_thread_id=42)
    message.chat.id = -100
    message.chat.send_action = AsyncMock()
    get_file = AsyncMock(return_value=SimpleNamespace(download_to_drive=download))
    message.get_bot.return_value.get_file = get_file

    with (
        patch.object(
            Path, "hardlink_to", side_effect=OSError(link_error, "not supported")
        ),
        patch.object(
            file_handler, "_resolve_upload_dir", return_value=("@0", upload_dir, None)
        ),
        patch.object(
            file_handler,
            "send_telegram_to_window",
            new_callable=AsyncMock,
            return_value=(True, "ok"),
        ) as send,
        patch.object(file_handler, "ack_reaction", new_callable=AsyncMock) as ack,
        patch.object(file_handler, "safe_reply", new_callable=AsyncMock) as reply,
    ):
        await file_handler._upload_and_notify(
            message,
            1,
            42,
            original.name,
            "fid",
            None,
            "File",
            "see {path}",
            "📎",
            adapter=MagicMock(),
        )

    get_file.assert_awaited_once_with("fid")
    [saved] = list(upload_dir.glob("*/*"))
    assert set(upload_dir.iterdir()) == {original, saved.parent}
    assert saved.read_bytes() == b"complete download"
    assert original.read_bytes() == b"previous upload"
    send.assert_awaited_once()
    assert send.await_args is not None
    assert send.await_args.args[3] == f"see {saved}"
    ack.assert_awaited_once()
    reply.assert_awaited_once_with(
        message, f"📎 Uploaded `{saved.relative_to(tmp_path)}`"
    )


@pytest.mark.parametrize(
    "link_error", [errno.EEXIST, errno.EIO, errno.EPERM, errno.ENOTSUP]
)
async def test_publication_failure_cleans_up(tmp_path: Path, link_error: int) -> None:
    message = MagicMock()
    download = AsyncMock(side_effect=lambda path: path.write_bytes(b"complete"))
    get_file = AsyncMock(return_value=SimpleNamespace(download_to_drive=download))
    message.get_bot.return_value.get_file = get_file
    with (
        patch.object(
            Path, "hardlink_to", side_effect=OSError(link_error, "link failed")
        ),
        patch.object(
            Path, "rename", side_effect=OSError(errno.EIO, "rename failed")
        ) as rename,
        patch.object(file_handler, "safe_reply", new_callable=AsyncMock) as reply,
    ):
        result = await file_handler._download_and_save(
            message, tmp_path, "report.txt", "fid", None, "File"
        )

    assert result is None
    assert rename.call_count == (1 if link_error == errno.ENOTSUP else 0)
    assert not list(tmp_path.iterdir())
    get_file.assert_awaited_once_with("fid")
    download.assert_awaited_once()
    reply.assert_awaited_once_with(message, "❌ Failed to save file.")


async def test_download_rechecks_upload_directory_before_publication(
    tmp_path: Path,
) -> None:
    message = MagicMock()
    download = AsyncMock(side_effect=lambda path: path.write_bytes(b"complete"))
    get_file = AsyncMock(return_value=SimpleNamespace(download_to_drive=download))
    message.get_bot.return_value.get_file = get_file
    real_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        # Simulate only the check result, without changing the filesystem.
        return bool(download.await_count) if path == tmp_path else real_is_symlink(path)

    with (
        patch.object(Path, "is_symlink", is_symlink),
        patch.object(Path, "hardlink_to") as link,
        patch.object(Path, "rename") as rename,
        patch.object(file_handler, "safe_reply", new_callable=AsyncMock) as reply,
    ):
        result = await file_handler._download_and_save(
            message, tmp_path, "report.txt", "fid", None, "File"
        )

    assert result is None
    get_file.assert_awaited_once_with("fid")
    download.assert_awaited_once()
    link.assert_not_called()
    rename.assert_not_called()
    assert not list(tmp_path.iterdir())
    reply.assert_awaited_once_with(message, "❌ Failed to save file.")


@pytest.mark.parametrize("file_size", [None, 5])
async def test_download_size_limits(tmp_path: Path, file_size: int | None) -> None:
    async def download(path: Path) -> None:
        path.write_bytes(b"large")

    message = MagicMock()
    get_file = AsyncMock(return_value=SimpleNamespace(download_to_drive=download))
    message.get_bot.return_value.get_file = get_file
    with (
        patch.object(file_handler, "_MAX_FILE_SIZE", 4),
        patch.object(file_handler, "safe_reply", new_callable=AsyncMock) as reply,
    ):
        result = await file_handler._download_and_save(
            message, tmp_path, "report.txt", "fid", file_size, "File"
        )

    assert result is None
    assert get_file.await_count == (1 if file_size is None else 0)
    assert not list(tmp_path.iterdir())
    reply.assert_awaited_once()
    assert reply.await_args is not None
    assert "File too large" in reply.await_args.args[1]


@pytest.mark.parametrize("symlink_dir", [True, False])
async def test_download_rejects_symlink_escape(
    tmp_path: Path, symlink_dir: bool
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    upload = tmp_path / "upload"
    if symlink_dir:
        upload.symlink_to(target, target_is_directory=True)
    else:
        upload.mkdir()
        (upload / "report.txt").symlink_to(target / "report.txt")
    message = MagicMock()
    get_file = AsyncMock()
    message.get_bot.return_value.get_file = get_file
    with patch.object(file_handler, "safe_reply", new_callable=AsyncMock) as reply:
        result = await file_handler._download_and_save(
            message, upload, "report.txt", "fid", None, "File"
        )

    assert result is None
    get_file.assert_not_awaited()
    assert not list(target.iterdir())
    reply.assert_awaited_once_with(
        message,
        "❌ Upload directory is invalid." if symlink_dir else "❌ Invalid filename.",
    )


class TestUploadTypingFailure:
    async def test_typing_failure_still_notifies_agent(self, tmp_path: Path) -> None:
        """Regression #257: a timed-out typing action must not drop the upload."""
        message = MagicMock()
        message.caption = None
        message.chat.id = -100
        message.chat.send_action = AsyncMock(side_effect=TimedOut())

        with (
            patch.object(
                file_handler,
                "_resolve_upload_dir",
                return_value=("@0", tmp_path, None),
            ),
            patch.object(
                file_handler,
                "_download_and_save",
                new_callable=AsyncMock,
                return_value="a.txt",
            ),
            patch.object(
                file_handler,
                "send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, "ok"),
            ) as mock_send,
            patch.object(file_handler, "ack_reaction", new_callable=AsyncMock),
            patch.object(file_handler, "safe_reply", new_callable=AsyncMock),
        ):
            await file_handler._upload_and_notify(
                message,
                1,
                42,
                "a.txt",
                "fid",
                10,
                "File",
                "see {path}",
                "📎",
                adapter=MagicMock(),
            )

        mock_send.assert_awaited_once()


class TestUploadNotifiesAbsolutePath:
    async def test_agent_message_uses_absolute_path(self, tmp_path: Path) -> None:
        """The agent message must carry an absolute path: an agent resolves a
        relative one against an arbitrary base (its home rather than the
        session cwd), and the failed read then looks like the upload never
        happened.
        """
        message = MagicMock()
        message.caption = None
        message.chat.id = -100
        message.chat.send_action = AsyncMock()

        with (
            patch.object(
                file_handler,
                "_resolve_upload_dir",
                return_value=("@0", tmp_path, None),
            ),
            patch.object(
                file_handler,
                "_download_and_save",
                new_callable=AsyncMock,
                return_value="a.txt",
            ),
            patch.object(
                file_handler,
                "send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=(True, "ok"),
            ) as mock_send,
            patch.object(file_handler, "ack_reaction", new_callable=AsyncMock),
            patch.object(
                file_handler, "safe_reply", new_callable=AsyncMock
            ) as mock_reply,
        ):
            await file_handler._upload_and_notify(
                message,
                1,
                42,
                "a.txt",
                "fid",
                10,
                "File",
                "see {path}",
                "📎",
                adapter=MagicMock(),
            )

        assert mock_send.await_args is not None
        agent_message = mock_send.await_args.args[3]
        assert str(tmp_path / "a.txt") in agent_message
        assert mock_reply.await_args is not None
        # The user-facing reply keeps the short relative form.
        assert ".ccgram-uploads/a.txt" in mock_reply.await_args.args[1]


class TestResolveUploadDir:
    @pytest.mark.parametrize(
        ("cwd", "expected_path", "expected_error"),
        [
            # a plain absolute cwd resolves to the upload dir inside it
            ("/tmp/repo", Path("/tmp/repo/.ccgram-uploads"), None),
            # a tilde form is rejected, not expanded against the wrong home
            ("~/repo", None, "Session working directory is not absolute."),
            # a relative cwd fails loudly instead of saving to a wrong place
            ("repo", None, "Session working directory is not absolute."),
            # a control character in the cwd would split the literal tmux send
            ("/tmp/re\npo", None, "Session working directory is not usable."),
            ("/tmp/re\x00po", None, "Session working directory is not usable."),
            ("/tmp/re\tpo", None, "Session working directory is not usable."),
            ("/tmp/re\x1fpo", None, "Session working directory is not usable."),
            ("/tmp/re\x7fpo", None, "Session working directory is not usable."),
            ("/tmp/re\x80po", None, "Session working directory is not usable."),
            ("/tmp/re\x9fpo", None, "Session working directory is not usable."),
            # a C1 control is executed by terminals on echo (U+009B is CSI)
            ("/tmp/re\x9bpo", None, "Session working directory is not usable."),
            # a nonprinting-but-legal Unicode char (NBSP) is not a control
            # character: the path is harmless in a literal tmux send
            ("/tmp/re\xa0po", Path("/tmp/re\xa0po/.ccgram-uploads"), None),
            # the pre-existing empty-cwd branch of the same function
            ("", None, "Session has no working directory."),
        ],
    )
    def test_cwd_forms(
        self, cwd: str, expected_path: Path, expected_error: str | None
    ) -> None:
        view = SimpleNamespace(cwd=cwd)
        with (
            patch.object(
                file_handler.thread_router,
                "resolve_window_for_thread",
                return_value="@0",
            ),
            patch.object(file_handler, "view_window", return_value=view),
        ):
            window_id, upload_path, error = file_handler._resolve_upload_dir(
                1, 42, -100
            )

        assert window_id == "@0"
        assert upload_path == expected_path
        assert error == expected_error
