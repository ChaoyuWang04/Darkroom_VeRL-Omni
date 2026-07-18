# OCR Reward 吞吐 Benchmark（Task 0.3-B）

> 实测于 **2026-06-12**，本地 1×RTX 5090 工作站。
> 目标：本地 OCR reward 替换 VLM 判官后，**持续吞吐 ≥ 4 img/s**（512 张 512×512 ≤ 128s），
> 确保 reward 计算不成为 FlowGRPO 训练 step 的瓶颈。
> **结论：达标，且余量巨大（8 进程 27.8 img/s，≈ 目标的 6.9×）。**

---

## 1. 配置

| 项 | 值 |
|---|---|
| OCR 引擎 | **RapidOCR (rapidocr-onnxruntime 1.4.4)** · onnxruntime 1.26.0（**CPU build**） |
| 模型 | ch_PP-OCRv4 det + cls + rec（RapidOCR 自带，中英文） |
| 打分 | `score = 1 - Levenshtein(ocr, gt)/len(gt)`，与 pr-5297 `compute_score_ocr` **逐字一致**（python-Levenshtein 0.27.3） |
| 负载 | 512 张 512×512 渲染文字图（英文/数字/中文/中英混合各 1/4，PIL 渲染） |
| CPU | **8 物理核 / 16 逻辑核**（HT） |
| 代码 | `src/rewards/ocr_reward.py` · `tests/bench_ocr_throughput.py` |

为何选 RapidOCR 而非 PaddleOCR：onnxruntime 后端 `pip` 即装即用，无需编译 `paddlepaddle`（在 sm_120/新环境上易出坑），CPU 多进程友好。识别精度在本任务的清晰渲染文字场景已足够（mean_score 0.953）。

---

## 2. 结果（512 张，每进程 onnx `intra_op_num_threads=1`）

| 模式 | 吞吐 | 512 张耗时 | 达标(≥4) | mean_score |
|---|---|---|---|---|
| CPU 单进程 | **4.32 img/s** | 118.6 s | ✅ 勉强 | 0.953 |
| CPU pool ×4 | **13.75 img/s** | 37.2 s | ✅ | 0.953 |
| **CPU pool ×8** | **27.77 img/s** | **18.4 s** | ✅ **峰值** | 0.953 |
| CPU pool ×12 | 15.32 img/s | 33.4 s | ✅ | 0.953 |
| CPU pool ×16 | 16.59 img/s | 30.9 s | ✅ | 0.953 |
| GPU (onnxruntime CUDA) | **N/A** | — | — | 见 §4 |

**峰值在 8 进程 = 物理核数**。12/16 进程反而下降：每个 onnx 会话虽限 1 intra-op 线程，但 worker 数超过物理核后落到 HT 逻辑核上，调度/缓存争用使吞吐回退。**8 进程是本机最优**，且恰好给训练宿主留下另外 8 逻辑核的余量。

---

## 3. 关键发现：onnx 线程过订导致多进程不 scale（核心坑）

- **现象（修前）**：直接 `Pool(N)`，每个 RapidOCR 用默认 `intra_op_num_threads=-1`（吃满所有核）。
  结果多进程几乎不 scale：cpu1=3.9 → cpu4=5.6 → cpu8=6.0 → cpu16=6.6 img/s（4 进程只比单进程快 1.4×）。
- **根因**：`-1` 让**每个** worker 的 onnx 会话都想抓 16 个核 → N 个 worker × 16 线程严重过订，全在抢同一批核，
  上下文切换吃掉并行收益。这是 onnxruntime 多进程的经典陷阱（呼应你 CUDA sprint 里「SIMT 过订 vs 占用率」的直觉）。
- **修法**：每 worker `intra_op_num_threads=1`（RapidOCR 的 Global 值会传播到 Det/Cls/Rec 三个会话），
  让「进程级并行」取代「会话内线程级并行」。
  **效果**：cpu4 5.6→**13.8**、cpu8 6.0→**27.8** img/s（4.6× 提升）。
- **落地默认**：`OCRScorer(intra_op_num_threads=1)` + 进程池 worker 数 = 物理核数。

---

## 4. GPU 后端：不可用 + 主动跳过（有据）

- **不可用**：本环境 onnxruntime 是 **CPU build**，`get_available_providers()` 只有
  `['AzureExecutionProvider','CPUExecutionProvider']`，无 `CUDAExecutionProvider`。RapidOCR-GPU 需 `onnxruntime-gpu`。
- **为何不装 onnxruntime-gpu**（工程裁决，非偷懒）：
  1. **保护 env-lock 栈**：`onnxruntime-gpu` 与现装 `onnxruntime` 同名冲突，会扰动已实测锁定的环境（env_lock.md）。
  2. **sm_120 风险高**：onnxruntime 官方 CUDA wheel 落后 Blackwell，sm_120 cubin 大概率缺失（同 P3/TRELLIS 教训）。
  3. **架构上反而是错的**：本地 OCR reward 存在的全部意义就是**把判官从 GPU 上挪走、释放训练显存**
     （model_decision.md §4）。把 OCR 放回 GPU = 重新和训练抢卡，自相矛盾。
- **结论**：**CPU 多进程不是「保底」，而是本场景的正解**。8 进程 27.8 img/s 已 6.9× 达标，无需 GPU。

---

## 5. 对 FlowGRPO 训练的换算（4×5090）

- 训练默认 `rollout.n=16`、`train_batch_size=32` → 单 step rollout 产出 **512 张**（16×32）待打分图，正好是本 benchmark 的负载。
- 8 进程下 **512 张 ≈ 18.4 s**。官方 LoRA 吞吐 0.305 img/GPU/s × 4 卡 ≈ 1.2 img/s 的「生成」速度 → 生成 512 张 rollout 本身要 ~430 s。
  **OCR 打分(18 s) 仅占 rollout 生成时间的 ~4%**，完全不是瓶颈，且跑在 CPU 上与 GPU 训练**重叠**，实际可被 rollout 完全掩盖。
- 4×5090 云实例若是 8+ 物理核宿主，直接用 `intra_op_num_threads=1` + 8 worker 进程池；核更多则线性放大。

---

## 6. 复现

```bash
VERLPY=~/Downloads/ENTER/envs/verl-omni/bin/python
cd ~/code/projects/verl-omni
$VERLPY -m pytest tests/test_ocr_reward.py -v        # 15 passed（含中/英/混合 >0.9，空白/噪声 <0.3）
$VERLPY tests/bench_ocr_throughput.py --n 512 --workers 1 4 8 12 16 --gpu
```
