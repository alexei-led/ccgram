import io

import pytest
from PIL import Image

from ccgram.screenshot import strip_non_sgr, text_to_image

SAMPLE_TEXT = "hello world\nfoo bar"
SAMPLE_ANSI = "\x1b[32mgreen\x1b[0m normal \x1b[31mred\x1b[0m"


async def test_default_produces_valid_png():
    png = await text_to_image(SAMPLE_TEXT, with_ansi=False)
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"
    assert img.mode == "RGB"


async def test_ansi_produces_valid_png():
    png = await text_to_image(SAMPLE_ANSI, with_ansi=True)
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"


async def test_live_mode_produces_valid_png():
    png = await text_to_image(SAMPLE_TEXT, with_ansi=False, live_mode=True)
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"
    assert img.mode == "P"


async def test_live_mode_smaller_than_default():
    regular = await text_to_image(SAMPLE_TEXT, with_ansi=False, live_mode=False)
    live = await text_to_image(SAMPLE_TEXT, with_ansi=False, live_mode=True)
    assert len(live) < len(regular)


async def test_live_mode_smaller_dimensions():
    regular = await text_to_image(SAMPLE_TEXT, with_ansi=False, live_mode=False)
    live = await text_to_image(SAMPLE_TEXT, with_ansi=False, live_mode=True)
    reg_img = Image.open(io.BytesIO(regular))
    live_img = Image.open(io.BytesIO(live))
    assert live_img.width < reg_img.width
    assert live_img.height < reg_img.height


async def test_default_unchanged_without_live_mode():
    png = await text_to_image(SAMPLE_TEXT, font_size=28, with_ansi=False)
    img = Image.open(io.BytesIO(png))
    assert img.mode == "RGB"
    assert img.format == "PNG"


async def test_live_mode_with_ansi_colors():
    png = await text_to_image(SAMPLE_ANSI, with_ansi=True, live_mode=True)
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"
    assert img.mode == "P"


async def test_live_mode_palette_size():
    png = await text_to_image(SAMPLE_TEXT, with_ansi=False, live_mode=True)
    img = Image.open(io.BytesIO(png))
    assert img.mode == "P"
    palette = img.getpalette()
    assert palette is not None
    unique_colors = len(set(zip(palette[::3], palette[1::3], palette[2::3])))
    assert unique_colors <= 32


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("\x1b[2Ahello", "hello", id="cursor_move"),
        pytest.param("\x1b]0;title\x07hello", "hello", id="osc_bel_terminated"),
        pytest.param("\x1b]0;title\x1b\\hello", "hello", id="osc_st_terminated"),
        pytest.param(
            "\x1b]8;;http://x\x1by\x1b\\link\x07hello",
            "link\x07hello",
            id="osc_with_embedded_esc",
        ),
        pytest.param("\x1b[?2004hhello", "hello", id="bracketed_paste"),
        pytest.param("plain text", "plain text", id="no_escapes"),
        pytest.param("", "", id="empty"),
    ],
)
def test_strip_non_sgr_removes_non_colour_sequences(text: str, expected: str):
    result = strip_non_sgr(text)
    assert result == expected
    assert "\x1b" not in result


def test_strip_non_sgr_preserves_sgr_colors():
    sgr = "\x1b[31mred\x1b[0m"
    result = strip_non_sgr(sgr)
    assert result == sgr


def test_strip_non_sgr_mixed_input():
    text = "\x1b[2A\x1b]0;t\x07\x1b[31mred\x1b[0m\x1b[?2004h"
    result = strip_non_sgr(text)
    assert result == "\x1b[31mred\x1b[0m"


def test_strip_non_sgr_plain_text_unchanged():
    assert strip_non_sgr("hello world") == "hello world"


async def test_render_with_non_sgr_escapes_produces_valid_png():
    mixed = "\x1b[2A\x1b]0;title\x07\x1b[32mgreen\x1b[0m\x1b[?2004h text"
    png = await text_to_image(mixed, with_ansi=True)
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"


async def test_render_sgr_survives_strip():
    colored = "\x1b[32mgreen\x1b[0m"
    png_colored = await text_to_image(colored, with_ansi=True)
    png_plain = await text_to_image("green", with_ansi=False)
    assert len(png_colored) > 0
    assert len(png_plain) > 0
