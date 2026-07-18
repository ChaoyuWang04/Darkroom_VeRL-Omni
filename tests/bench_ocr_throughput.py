"""OCR-reward throughput benchmark.

Simulates the real FlowGRPO reward load: score a batch of 512x512 rendered-text
images and measure images/second. The acceptance target is **>=4 img/s sustained**
(512 images <= 128s) so reward never bottlenecks a training step.

Modes:
  - cpu1        : single process, one RapidOCR engine
  - cpuN (pool) : multiprocessing.Pool with N workers, each its own engine
  - gpu         : RapidOCR on CUDAExecutionProvider (requires onnxruntime-gpu)

Run from repo root:
    python tests/bench_ocr_throughput.py            # default: 512 imgs, workers 1/4/8/16
    python tests/bench_ocr_throughput.py --n 256 --workers 4 8

Why CPU is the production path (not just a fallback): the whole point of the local
OCR reward is to free the GPU that the VLM judge used to occupy. Running OCR on the
training GPUs would re-introduce the contention we removed. So the number that
matters for 4x5090 is sustained CPU throughput with workers <= spare host cores.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rewards.ocr_reward import OCRScorer, ocr_text_score  # noqa: E402
from tests._render import render_text_image  # noqa: E402

# Diverse, realistic text-render targets (english / digits / chinese / mixed).
_SAMPLES = [
    "Hello World", "OpenAI2026", "Reward Model", "GRPO", "Qwen Image",
    "DiffusionRL", "Blackwell", "sm_120", "Levenshtein", "FlowGRPO",
    "人工智能", "深度学习", "文字渲染", "强化学习", "扩散模型",
    "AI研究院", "5090显卡", "RLHF训练", "Token2025", "VeRL Omni",
]


def make_batch(n: int) -> list[tuple[np.ndarray, str]]:
    """Render n images (uint8 HWC arrays) paired with their ground-truth text."""
    batch = []
    for i in range(n):
        text = _SAMPLES[i % len(_SAMPLES)]
        img = render_text_image(text, size=(512, 512), font_size=72)
        batch.append((np.asarray(img), text))
    return batch


# --- worker plumbing for the multiprocessing pool --------------------------- #
_WORKER_SCORER: OCRScorer | None = None


def _init_worker(use_gpu: bool, intra_threads: int) -> None:
    global _WORKER_SCORER
    # pin each worker to `intra_threads` onnx threads so N processes don't all try
    # to grab every core (the cause of flat multiprocess scaling).
    _WORKER_SCORER = OCRScorer(use_gpu=use_gpu, intra_op_num_threads=intra_threads)
    # warm the engine so steady-state throughput excludes one-time onnx session init
    _WORKER_SCORER.recognize(np.full((64, 128, 3), 255, dtype=np.uint8))


def _score_one(arg: tuple[np.ndarray, str]) -> float:
    img, gt = arg
    text = _WORKER_SCORER.recognize(img)
    return ocr_text_score(text, gt)


# --- benchmark runners ------------------------------------------------------ #
def run_single(batch, use_gpu: bool = False) -> tuple[float, float]:
    scorer = OCRScorer(use_gpu=use_gpu)
    scorer.recognize(np.full((64, 128, 3), 255, dtype=np.uint8))  # warmup
    t0 = time.perf_counter()
    scores = [ocr_text_score(scorer.recognize(img), gt) for img, gt in batch]
    dt = time.perf_counter() - t0
    return dt, float(np.mean(scores))


def run_pool(batch, workers: int, use_gpu: bool = False, intra_threads: int = 1) -> tuple[float, float]:
    t0 = time.perf_counter()
    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(use_gpu, intra_threads),
    ) as pool:
        scores = pool.map(_score_one, batch, chunksize=max(1, len(batch) // (workers * 4)))
    dt = time.perf_counter() - t0  # includes pool spawn + per-worker warmup (honest)
    return dt, float(np.mean(scores))


def _fmt(n, dt, mean_score):
    return f"{n / dt:6.2f} img/s   ({dt:6.1f}s for {n} imgs)   mean_score={mean_score:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="number of images")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--gpu", action="store_true", help="also try GPU (onnxruntime-gpu)")
    args = ap.parse_args()

    print(f"Rendering {args.n} images (512x512) ...", flush=True)
    batch = make_batch(args.n)

    print("\n=== CPU single process ===", flush=True)
    dt, ms = run_single(batch)
    print("cpu1   ", _fmt(args.n, dt, ms), flush=True)

    for w in args.workers:
        if w == 1:
            continue
        print(f"\n=== CPU pool, {w} workers ===", flush=True)
        dt, ms = run_pool(batch, w)
        print(f"cpu{w:<3}", _fmt(args.n, dt, ms), flush=True)

    if args.gpu:
        print("\n=== GPU (onnxruntime CUDAExecutionProvider) ===", flush=True)
        try:
            import onnxruntime as ort

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                print("gpu     UNAVAILABLE: onnxruntime build exposes only "
                      f"{ort.get_available_providers()} (CPU-only). "
                      "Needs onnxruntime-gpu; not installed to protect the "
                      "env-locked stack. See docs/reward_benchmark.md.", flush=True)
            else:
                dt, ms = run_single(batch, use_gpu=True)
                print("gpu    ", _fmt(args.n, dt, ms), flush=True)
        except Exception as e:
            print(f"gpu     ERROR: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
