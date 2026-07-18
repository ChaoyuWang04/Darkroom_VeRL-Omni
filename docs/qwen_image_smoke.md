# Task 0.3-C — Qwen-Image GGUF Int4 single-card smoke (RTX 5090 / sm_120)

**Status: ✅ WORKING.** Goal was pipeline connectivity + sm_120 kernel validation
(not fidelity), then feed the generated image through the local OCR reward (Task B)
for an end-to-end check.

## Result (measured 2026-06-18)

| metric | value |
|---|---|
| invocation | `--quantization gguf --gguf-model <Q4_K_M.gguf>`, 512×512, 10 steps, **no cpu-offload** |
| exit | 0 |
| wall (incl. load) | ~40s |
| diffusion exec | 1.40s (10 steps, 140ms/step) |
| **peak VRAM** | **30354 MiB** (of 32607) |
| **peak RAM** | **~6 GB** (of 30) |
| output | `data/smoke_qwen_image_gguf.png` — renders the word **HELLO**, legible |
| OCR end-to-end | `score=1.000 acc=True recognized='HELLO'` (Task B `compute_score_ocr_local`) |

## The recipe

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model   <base Qwen/Qwen-Image dir> \
  --quantization gguf --gguf-model <Qwen-Image Q4_K_M .gguf> \
  --prompt '... the single large bold black word "HELLO" ...' \
  --height 512 --width 512 --num-inference-steps 10 --seed 142 \
  --output data/smoke_qwen_image_gguf.png
# NOTE: NO --enable-cpu-offload   (see "cpu-offload trap" below)
```

Driver: `scripts/run_qwen_image_gguf_smoke.sh` (waits for GGUF download + a free GPU,
never kills other jobs, then runs the smoke and the OCR end-to-end check).

## Why this configuration

- **bf16 is impossible locally.** The bf16 transformer is 40.9GB and the
  text_encoder (Qwen2.5-VL-7B) 16.6GB; offload's destination is the 30GB RAM, so it
  can't hold them. Online FP8 also OOMs on a 40.9GB bf16 load transient. → GGUF Int4
  is the only path that fits. (Q4_K_M transformer ≈ 13GB.)
- **city96 vs QuantStack is moot.** Both `city96/Qwen-Image-gguf` and the official
  `QuantStack/Qwen-Image-GGUF` Q4_K_M are byte-identical in quant layout (verified
  per-tensor from both file headers: 1087 F32 / 580 Q4_K / 232 Q6_K / 28 Q5_K / 6 BF16).

## The cpu-offload trap (the non-obvious bit)

`--enable-cpu-offload` **breaks** this setup. With the GGUF transformer at ~13GB,
offload pins it in the 30GB RAM, and when the 16GB bf16 text encoder also transits
RAM during the warmup forward, the worker is **SIGKILL'd by the OOM-killer**
(symptom: `Diffusion worker(s) died unexpectedly (exitcode=None)` ~7s into the dummy
run, no CUDA error even under `CUDA_LAUNCH_BLOCKING=1`). Because the GGUF transformer
is small, the whole pipeline fits on the 32GB card — so keep everything on GPU and
the RAM wall disappears (peak RAM drops from OOM to ~6GB).

## Three upstream vllm-omni fixes this required

All in `~/code/upstream/vllm-omni` (main @ e26f5cb). Worth upstreaming as a PR.

1. **config plumbing** — `engine/async_omni_engine.py`, `_create_default_diffusion_stage_cfg`:
   the offline `Omni()` path dropped `quantization_config` (dict); only the legacy
   `quantization` (str) was forwarded, and the fallback injection read a different key
   (`diffusion_quantization_config`). Added one line forwarding `quantization_config`.
   Without it, `--quantization gguf` was silently ignored → full bf16 load → RAM OOM.

2. **BF16 ≠ quantized** — `diffusion/model_loader/gguf_adapters/base.py`,
   `gguf_quant_weights_iterator`: BF16 is stored raw, not a real GGUF quant type.
   Treating it as quantized emitted a `.qweight_type` that unquantized modules
   (Qwen-Image's precision-sensitive img_in/txt_in/proj_out/time_embed, kept fp per
   #2728) never registered → `KeyError: 'img_in.qweight_type'`. Excluded BF16 from the
   quantized path so those 6 boundary tensors load as plain `.weight`.

3. **mod linears honor quant_config** — `diffusion/models/qwen_image/qwen_image_transformer.py`:
   `img_mod.1`/`txt_mod.1` were hardcoded `quant_config=None`, but the GGUF *does*
   quantize them (Q4_K, 120 tensors). Changed to `quant_config=quant_config` — None in
   BF16 mode keeps them full precision; a GGUF config lets the quantized weights load.

(attention + MLP — the bulk of params — were already correctly GGUF-wired.)
