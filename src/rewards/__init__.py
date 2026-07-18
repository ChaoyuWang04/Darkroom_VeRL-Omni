"""Framework-agnostic reward functions for VeRL-Omni diffusion RL."""

from src.rewards.ocr_reward import (
    OCRScorer,
    compute_score_ocr,
    compute_score_ocr_local,
    get_scorer,
    ocr_text_score,
    to_pil,
)

__all__ = [
    "OCRScorer",
    "compute_score_ocr",
    "compute_score_ocr_local",
    "get_scorer",
    "ocr_text_score",
    "to_pil",
]
