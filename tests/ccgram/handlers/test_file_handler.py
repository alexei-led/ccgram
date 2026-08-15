"""Tests for file_handler helper functions."""

import re
import unicodedata
from pathlib import Path

import pytest

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
