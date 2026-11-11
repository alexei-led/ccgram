"""Claude Code paste wrappers are looked through when matching Telegram text."""

from ccgram.pasted_content import unwrap_pasted_content


def test_unwraps_a_single_paste() -> None:
    wrapped = (
        '\n\n<pasted_content id="cccd">\nhello\nworld\n</pasted_content id="cccd">\n'
    )
    assert unwrap_pasted_content(wrapped).strip() == "hello\nworld"


def test_plain_text_is_untouched() -> None:
    assert unwrap_pasted_content("a <b> c") == "a <b> c"


def test_closing_tag_without_id() -> None:
    wrapped = '<pasted_content id="x1">\nhi\n</pasted_content>'
    assert unwrap_pasted_content(wrapped) == "hi"
