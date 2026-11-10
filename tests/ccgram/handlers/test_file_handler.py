"""Tests for file_handler helper functions."""

import re
import unicodedata
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import TimedOut

from ccgram.handlers import file_handler
from ccgram.session_protocol.models import UNCONFIRMED_DELIVERY_MESSAGE
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
                return_value=("a.txt", 10),
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
                return_value=("a.txt", 10),
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


class TestSizeNote:
    def test_matching_size_is_confirmed(self) -> None:
        assert file_handler._size_note(6761, 6761) == (
            True,
            "6,761 bytes, matches Telegram",
        )

    def test_mismatch_is_flagged(self) -> None:
        ok, note = file_handler._size_note(6000, 6761)
        assert not ok and "6,761" in note and "6,000" in note

    def test_missing_telegram_size_is_not_confirmed(self) -> None:
        ok, note = file_handler._size_note(10, None)
        assert not ok and "no size" in note


class TestUploadReplies:
    async def _reply_for(
        self, tmp_path: Path, send_result: tuple[bool, str], saved_size: int
    ) -> str:
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
                return_value=("a.txt", saved_size),
            ),
            patch.object(
                file_handler,
                "send_telegram_to_window",
                new_callable=AsyncMock,
                return_value=send_result,
            ),
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
        assert mock_reply.await_args is not None
        return mock_reply.await_args.args[1]

    async def test_success_reports_matching_size(self, tmp_path: Path) -> None:
        reply = await self._reply_for(tmp_path, (True, "Sent"), 10)
        assert reply.startswith("📎 Uploaded") and "matches Telegram" in reply

    async def test_unconfirmed_with_matching_size_is_green(
        self, tmp_path: Path
    ) -> None:
        reply = await self._reply_for(
            tmp_path, (False, UNCONFIRMED_DELIVERY_MESSAGE), 10
        )
        assert reply.startswith("\u2705") and "could not be confirmed" in reply

    async def test_unconfirmed_with_size_mismatch_warns(self, tmp_path: Path) -> None:
        reply = await self._reply_for(
            tmp_path, (False, UNCONFIRMED_DELIVERY_MESSAGE), 9
        )
        assert reply.startswith("\u26a0\ufe0f")

    async def test_refused_send_is_a_failure(self, tmp_path: Path) -> None:
        reply = await self._reply_for(tmp_path, (False, "Session ended"), 10)
        assert reply.startswith("\u274c") and "Session ended" in reply


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
