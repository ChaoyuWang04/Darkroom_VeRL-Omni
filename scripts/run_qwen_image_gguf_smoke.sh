#!/usr/bin/env bash
# Task 0.3-C: single-card Qwen-Image text-to-image smoke on RTX 5090 (sm_120).
#
# Strategy (see docs/qwen_image_smoke.md + memory):
#   - 32GB VRAM / 30GB RAM can't hold bf16 (54GB), and online FP8 OOMs on a 40.9GB
#     bf16 load transient. So we use a pre-quantized GGUF Int4 transformer
#     (Q4_K_M, ~13GB) loaded via vllm-omni's native `--quantization gguf
#     --gguf-model`, reusing the base model's text_encoder/vae/scheduler.
#   - CRITICAL: do NOT use --enable-cpu-offload. With the GGUF transformer at
#     ~13GB, the full pipeline (transformer 13GB + Qwen2.5-VL text_encoder 16GB +
#     VAE) fits on the 32GB card (measured peak 30.3GB, RAM stays ~6GB). cpu-offload
#     instead pins the transformer in the 30GB RAM and OOM-kills the worker when the
#     16GB text encoder also transits RAM. Keeping everything on GPU sidesteps the
#     RAM wall entirely. (Needs the card essentially empty -> FREE_NEEDED_MIB=31000.)
#   - Requires 3 upstream vllm-omni fixes (see docs): quantization_config plumbing,
#     BF16-not-quantized in the gguf iterator, and quant_config on QwenImage mod linears.
#   - The GPU is SHARED: a Blender/SceneGen dataset job has priority. This script
#     WAITS quietly for the GPU to be free before touching it. It never kills anything.
#
# Goal: validate pipeline connectivity + sm_120 kernels (NOT image fidelity), then
# feed the generated image through the local OCR reward for an end-to-end check.
set -uo pipefail

VERLPY=~/Downloads/ENTER/envs/verl-omni/bin/python
ROOT=~/code/projects/verl-omni
MODEL=$ROOT/models/Qwen/Qwen-Image
GGUF=$ROOT/models/gguf/qwen-image-Q4_K_M.gguf
OUT=$ROOT/data/smoke_qwen_image_gguf.png
GT="HELLO"                                  # the text we ask the model to render
PROMPT='A clean minimalist poster on a plain white background, with the single large bold black word "HELLO" printed in the center.'
T2I=~/code/upstream/vllm-omni/examples/offline_inference/text_to_image/text_to_image.py

FREE_NEEDED_MIB=${FREE_NEEDED_MIB:-31000}   # full pipeline peaks ~30.3GB on GPU (no offload)
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-21600}     # wait up to 6h for the GPU
DL_WAIT_MAX_S=${DL_WAIT_MAX_S:-3600}        # wait up to 1h for the GGUF download

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 1) wait for the GGUF download to finish ------------------------------------
log "waiting for GGUF: $GGUF"
waited=0
while [ ! -f "$GGUF" ]; do
  if ! pgrep -f dl_gguf.py >/dev/null && [ ! -f "$GGUF" ]; then
    # downloader gone but file missing -> maybe partial dir; keep checking briefly
    :
  fi
  sleep 10; waited=$((waited+10))
  [ $waited -ge $DL_WAIT_MAX_S ] && { log "ERROR: GGUF download timeout"; exit 2; }
done
log "GGUF present: $(du -h "$GGUF" | cut -f1)"

# --- 2) wait for the GPU to be free (priority: other jobs win) ------------------
log "waiting for a free GPU (need >=${FREE_NEEDED_MIB}MiB free, ${GPU_WAIT_MAX_S}s max)"
waited=0; ok=0
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -ge "$FREE_NEEDED_MIB" ]; then
    ok=$((ok+1))
    [ $ok -ge 3 ] && { log "GPU free (${free}MiB), 3 consecutive checks -> go"; break; }
  else
    [ $ok -ne 0 ] && log "GPU busy again (${free}MiB free), resetting"
    ok=0
  fi
  sleep 15; waited=$((waited+15))
  [ $waited -ge $GPU_WAIT_MAX_S ] && { log "ERROR: GPU never freed within budget"; exit 3; }
done

# --- 3) sample peak VRAM in the background during generation --------------------
PEAK_FILE=$(mktemp)
( m=0; while :; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${u:-0}" -gt "$m" ] && { m=$u; echo "$m" > "$PEAK_FILE"; }
    sleep 1
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

# --- 4) run the smoke -----------------------------------------------------------
log "launching Qwen-Image GGUF smoke (512x512, 10 steps, cpu-offload)"
START=$(date +%s)
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $VERLPY "$T2I" \
    --model "$MODEL" \
    --quantization gguf --gguf-model "$GGUF" \
    --prompt "$PROMPT" \
    --height 512 --width 512 \
    --num-inference-steps 10 \
    --seed 142 \
    --output "$OUT"
RC=$?
END=$(date +%s)
kill $SAMPLER 2>/dev/null
PEAK=$(cat "$PEAK_FILE" 2>/dev/null || echo "?")
rm -f "$PEAK_FILE"

log "smoke exit=$RC  wall=$((END-START))s  peak_vram=${PEAK}MiB  out=$OUT"
[ $RC -ne 0 ] && { log "smoke FAILED (see errors above)"; exit $RC; }

# --- 5) end-to-end: OCR the generated image with the Task-B reward --------------
log "running local OCR reward on the generated image (gt='$GT')"
$VERLPY - "$OUT" "$GT" <<'PY'
import sys
sys.path.insert(0, __import__('os').path.expanduser('~/code/projects/verl-omni'))
from PIL import Image
from src.rewards.ocr_reward import compute_score_ocr_local
img = Image.open(sys.argv[1]).convert("RGB")
out = compute_score_ocr_local(img, sys.argv[2])
print(f"OCR end-to-end: score={out['score']:.3f} acc={out['acc']} recognized={out['genrm_response']!r}")
PY
log "DONE"
