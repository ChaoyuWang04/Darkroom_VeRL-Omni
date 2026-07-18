"""Text-image rendering helpers for OCR-reward tests and benchmarks.

Simulates the "Qwen-Image renders the requested text" rollout output: a clean image
with a string drawn on it. Also provides garbage/blank generators for the
low-score sanity tests.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFont

# Latin-only font is enough for ASCII; CJK font covers Chinese (and Latin too).
_FONT_LATIN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def _is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _font(text: str, size: int) -> ImageFont.FreeTypeFont:
    path = _FONT_LATIN if _is_ascii(text) else _FONT_CJK
    return ImageFont.truetype(path, size)


def render_text_image(
    text: str,
    size: tuple[int, int] = (512, 512),
    font_size: int = 64,
    bg: str = "white",
    fg: str = "black",
) -> Image.Image:
    """Render ``text`` centered on a solid background. CJK-aware font selection."""
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    font = _font(text, font_size)
    # center using the text bbox
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size[0] - tw) // 2 - bbox[0]
    y = (size[1] - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=fg, font=font)
    return img


def render_blank_image(size: tuple[int, int] = (512, 512), bg: str = "white") -> Image.Image:
    """A blank image — OCR should find nothing -> score 0 for any non-empty gt."""
    return Image.new("RGB", size, bg)


def render_noise_image(size: tuple[int, int] = (512, 512), seed: int = 0) -> Image.Image:
    """Random RGB noise — no legible text -> low score for any non-empty gt."""
    rng = random.Random(seed)
    img = Image.new("RGB", size)
    px = img.load()
    for yy in range(size[1]):
        for xx in range(size[0]):
            px[xx, yy] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    return img
