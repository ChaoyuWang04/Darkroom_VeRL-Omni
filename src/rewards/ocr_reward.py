"""Local OCR-based reward for diffusion RL text-rendering tasks.

Drop-in replacement for verl pr-5297's
``tests/experimental/reward_loop/reward_fn.py::compute_score_ocr``. The upstream
function ships the rendered image to a remote VLM judge (Qwen2.5-VL-3B served over
an OpenAI-compatible HTTP endpoint) used *purely as an OCR engine*, then scores the
returned text against ``ground_truth`` with a normalized Levenshtein distance.

This module keeps the **exact same scoring** but swaps the VLM judge for a local
OCR engine (RapidOCR / onnxruntime). That removes the judge GPU entirely — the
key move that lets Qwen-Image FlowGRPO + LoRA fit on 4x5090 (32GB) instead of
needing a colocated TP=4 judge (see docs/model_decision.md section 4).

Scoring contract (verbatim from upstream, so reward signal is identical):
    gt   = strip_whitespace(ground_truth).lower()
    text = strip_whitespace(ocr_text).lower()
    dist = 0 if gt in text else Levenshtein.distance(text, gt)
    dist = min(dist, len(gt))              # cap: garbage OCR costs at most len(gt)
    score = 1 - dist / len(gt)             # in [0, 1]
    acc   = (score == 1)

Public API:
    OCRScorer                     - reusable engine wrapper (lazy, per-process)
    ocr_text_score(text, gt)      - pure scoring helper (no OCR, no deps on engine)
    to_pil(image)                 - PIL/np.ndarray/torch.Tensor -> PIL.Image (RGB)
    compute_score_ocr_local(...)  - sync scorer: image + gt -> {score, acc, ...}
    compute_score_ocr(...)        - async drop-in matching pr-5297's signature
"""

from __future__ import annotations

import re
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

try:  # torch is part of the verl-omni env; keep import soft so the module is usable bare.
    import torch

    _TORCH_OK = True
except Exception:  # pragma: no cover - torch always present in target env
    _TORCH_OK = False

ImageLike = Union[Image.Image, np.ndarray, "Any"]

_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Image normalization (mirrors pr-5297 reward_fn.compute_score_ocr preprocess)
# --------------------------------------------------------------------------- #
def to_pil(image: ImageLike) -> Image.Image:
    """Convert a PIL image / HWC numpy array / CHW torch tensor to a PIL RGB image.

    Matches upstream behavior:
      - torch.Tensor: float, ``permute(1, 2, 0)`` (CHW->HWC), moved to cpu/numpy.
      - np.ndarray: HWC; float arrays in [0, 1] are scaled to uint8, uint8 passes
        through. This is more permissive than upstream (which assumes float [0,1])
        so already-uint8 rollouts don't get crushed to black.
    """
    if _TORCH_OK and isinstance(image, torch.Tensor):
        image = image.detach().float().permute(1, 2, 0).cpu().numpy()

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:  # grayscale -> RGB
            arr = np.stack([arr] * 3, axis=-1)
        assert arr.shape[-1] == 3, f"expected HWC with 3 channels, got {arr.shape}"
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0).round()
        arr = arr.clip(0, 255).astype(np.uint8)
        image = Image.fromarray(arr)

    assert isinstance(image, Image.Image), f"unsupported image type: {type(image)}"
    return image.convert("RGB")


# --------------------------------------------------------------------------- #
# Pure scoring (identical math to upstream; no OCR engine involved)
# --------------------------------------------------------------------------- #
def ocr_text_score(ocr_text: str, ground_truth: str) -> float:
    """Normalized Levenshtein score in [0, 1]. Identical to pr-5297's logic."""
    gt = _WS.sub("", ground_truth or "").lower()
    text = _WS.sub("", ocr_text or "").lower()

    if len(gt) == 0:  # upstream divides by len(gt); guard the degenerate case.
        return 1.0 if len(text) == 0 else 0.0

    dist = 0 if gt in text else _levenshtein(text, gt)
    dist = min(dist, len(gt))  # recognized many unrelated chars -> cap penalty
    return 1.0 - dist / len(gt)


def _levenshtein(a: str, b: str) -> int:
    # python-Levenshtein (C) is in the env and ~100x faster than a pure-python DP;
    # fall back to a small DP so the module never hard-depends on it.
    try:
        import Levenshtein

        return Levenshtein.distance(a, b)
    except Exception:  # pragma: no cover
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


# --------------------------------------------------------------------------- #
# OCR engine wrapper
# --------------------------------------------------------------------------- #
class OCRScorer:
    """Lazy, reusable RapidOCR wrapper. One instance per process.

    RapidOCR's onnxruntime sessions are heavy (~0.3s warmup) and not picklable, so
    we construct the engine lazily on first use. For multiprocessing, give each
    worker its *own* OCRScorer (see ``_pool_worker`` in the benchmark).
    """

    def __init__(
        self,
        use_gpu: bool = False,
        intra_op_num_threads: Optional[int] = None,
        **rapidocr_kwargs: Any,
    ) -> None:
        self.use_gpu = use_gpu
        # onnxruntime defaults to all cores per session. In a multiprocessing pool
        # that oversubscribes the CPU (N workers x M cores), collapsing throughput.
        # Pin each worker's engine to a small thread count so processes scale.
        self.intra_op_num_threads = intra_op_num_threads
        self._kwargs = rapidocr_kwargs
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            kw = dict(self._kwargs)
            if self.intra_op_num_threads is not None:
                # propagates Global -> Det/Cls/Rec sessions (see UpdateParameters)
                kw.setdefault("intra_op_num_threads", self.intra_op_num_threads)
            if self.use_gpu:
                # rapidocr-onnxruntime routes to CUDAExecutionProvider when these
                # are set AND onnxruntime-gpu is installed; otherwise silently CPU.
                kw.setdefault("det_use_cuda", True)
                kw.setdefault("rec_use_cuda", True)
                kw.setdefault("cls_use_cuda", True)
            self._engine = RapidOCR(**kw)
        return self._engine

    def recognize(self, image: ImageLike) -> str:
        """Run OCR and return all detected text joined in reading order (top->bottom,
        left->right). Returns "" when nothing is detected."""
        arr = np.asarray(to_pil(image))  # RGB HWC uint8
        result, _elapse = self.engine(arr)
        if not result:
            return ""
        # result items: [box(4x2), text, conf]; sort by box top-left (y, x).
        def _key(item):
            box = item[0]
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            return (round(min(ys) / 10.0), min(xs))  # bucket y so same-line stays L->R

        items = sorted(result, key=_key)
        return " ".join(str(it[1]) for it in items)

    def score(self, image: ImageLike, ground_truth: str) -> dict:
        text = self.recognize(image)
        s = ocr_text_score(text, ground_truth)
        return {"score": s, "acc": s == 1.0, "genrm_response": text}


# --------------------------------------------------------------------------- #
# Module-level default scorer + convenience functions
# --------------------------------------------------------------------------- #
_DEFAULT_SCORER: Optional[OCRScorer] = None


def get_scorer(use_gpu: bool = False) -> OCRScorer:
    """Process-global default scorer (built once)."""
    global _DEFAULT_SCORER
    if _DEFAULT_SCORER is None:
        _DEFAULT_SCORER = OCRScorer(use_gpu=use_gpu)
    return _DEFAULT_SCORER


def compute_score_ocr_local(
    solution_image: ImageLike,
    ground_truth: str,
    *,
    use_gpu: bool = False,
    **_ignored: Any,
) -> dict:
    """Sync local-OCR reward: rendered image + target text -> {score, acc, genrm_response}.

    ``genrm_response`` holds the raw recognized text (mirrors the upstream field name
    so downstream logging/eval code is unchanged). Extra kwargs are accepted and
    ignored so this is interface-compatible with the upstream router-based call.
    """
    return get_scorer(use_gpu=use_gpu).score(solution_image, ground_truth)


async def compute_score_ocr(
    data_source: str = None,
    solution_image: ImageLike = None,
    ground_truth: str = None,
    extra_info: dict = None,
    reward_router_address: str = None,  # accepted + ignored (no remote judge)
    reward_model_tokenizer: Any = None,  # accepted + ignored
    model_name: str = None,  # accepted + ignored
    **_ignored: Any,
) -> dict:
    """Async drop-in for pr-5297 ``reward_fn.compute_score_ocr``.

    Same positional signature; the ``reward_router_address`` / tokenizer / model_name
    args are accepted for compatibility but unused — OCR runs locally in a thread so
    the event loop is not blocked.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, get_scorer().score, solution_image, ground_truth
    )
