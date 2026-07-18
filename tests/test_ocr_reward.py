"""Unit tests for src/rewards/ocr_reward.py.

Run from repo root:  pytest tests/test_ocr_reward.py -v
(or: python -m pytest ...). Adds repo root to sys.path so `import src...` works
without installation.
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rewards.ocr_reward import (  # noqa: E402
    OCRScorer,
    compute_score_ocr_local,
    ocr_text_score,
    to_pil,
)
from tests._render import (  # noqa: E402
    render_blank_image,
    render_noise_image,
    render_text_image,
)

# One shared engine for the whole module (warmup is ~0.3s).
SCORER = OCRScorer()


# --------------------------------------------------------------------------- #
# Pure scoring math (no OCR) — fast, deterministic
# --------------------------------------------------------------------------- #
def test_score_exact_match():
    assert ocr_text_score("Hello World", "Hello World") == 1.0


def test_score_substring_is_perfect():
    # gt fully contained in OCR output -> dist 0 -> score 1 (upstream behavior)
    assert ocr_text_score("noise Hello123 noise", "Hello123") == 1.0


def test_score_whitespace_and_case_insensitive():
    assert ocr_text_score("  HELLO\nworld ", "hello world") == 1.0


def test_score_one_char_off():
    # gt len 5, one substitution -> dist 1 -> score 0.8
    assert ocr_text_score("hallo", "hello") == pytest.approx(0.8)


def test_score_garbage_is_capped_at_zero():
    assert ocr_text_score("zzzzzzzzzzzzzzzz", "hello") == 0.0


def test_score_empty_ocr():
    assert ocr_text_score("", "hello") == 0.0


# --------------------------------------------------------------------------- #
# Image normalization
# --------------------------------------------------------------------------- #
def test_to_pil_from_float_chw_tensor():
    torch = pytest.importorskip("torch")
    t = torch.zeros(3, 32, 48)  # CHW, float in [0,1]
    img = to_pil(t)
    assert isinstance(img, Image.Image)
    assert img.size == (48, 32)  # PIL is (W, H)


def test_to_pil_from_uint8_hwc_numpy():
    arr = np.zeros((32, 48, 3), dtype=np.uint8)
    img = to_pil(arr)
    assert img.size == (48, 32)


# --------------------------------------------------------------------------- #
# End-to-end OCR scoring (real RapidOCR)
# --------------------------------------------------------------------------- #
def test_clear_english_text_scores_high():
    img = render_text_image("Hello123", font_size=72)
    out = SCORER.score(img, "Hello123")
    assert out["score"] > 0.9, out
    assert out["acc"] is True


def test_clear_chinese_text_scores_high():
    img = render_text_image("人工智能", font_size=96)
    out = SCORER.score(img, "人工智能")
    assert out["score"] > 0.9, out


def test_mixed_cn_en_text_scores_high():
    img = render_text_image("AI研究院2026", font_size=80)
    out = SCORER.score(img, "AI研究院2026")
    assert out["score"] > 0.9, out


def test_blank_image_scores_low():
    img = render_blank_image()
    out = SCORER.score(img, "Hello123")
    assert out["score"] < 0.3, out
    assert out["acc"] is False


def test_noise_image_scores_low():
    img = render_noise_image(seed=42)
    out = SCORER.score(img, "Hello123")
    assert out["score"] < 0.3, out


def test_compute_score_ocr_local_interface():
    img = render_text_image("Reward42", font_size=72)
    out = compute_score_ocr_local(img, "Reward42")
    assert set(out) == {"score", "acc", "genrm_response"}
    assert out["score"] > 0.9
    assert isinstance(out["genrm_response"], str)


def test_numpy_float_image_path():
    # the rollout often hands a float HWC array in [0,1]; make sure that path scores.
    img = render_text_image("Path2PIL", font_size=72)
    arr = np.asarray(img).astype(np.float32) / 255.0
    out = compute_score_ocr_local(arr, "Path2PIL")
    assert out["score"] > 0.9, out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
