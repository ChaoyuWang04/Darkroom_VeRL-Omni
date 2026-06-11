# Task 0.2 模型选型裁决（仓库侦察证据版）

> 证据来源：实际 clone 的上游源码 + 官方公告，非二手信息。
> - `vllm-project/vllm-omni` @ `e26f5cb918caea66ca403155d82d9ce4fbb97982` (2026-06-11)
> - `verl-project/verl` main @ `41a52449db58996731f54b79d20b939f5297fc4d` (2026-06-10)
> - **FlowGRPO 训练代码不在 verl main**，在未合并的 PR 序列（详见 §1）
> - 官方公告：<https://vllm.ai/blog/2026-05-14-verl-omni>，RFC：verl issue #4639

---

## 核心裁决（TL;DR）

| 决策项 | 结论 |
|---|---|
| **选哪个模型** | **Qwen-Image (DiT, ~20B) + FlowGRPO + LoRA** —— 唯一 Released、E2E 完整、官方示例齐全的入口 |
| **Z-Image 能用吗** | ❌ **不能**。只在 RFC #4639 的 motivation 里被点名为「未来目标」，零代码、零示例。不要选。 |
| **官方最小配置** | LoRA：**4 GPU**；Full-FT：**8 GPU**（见 §2） |
| **4×5090 (128GB) 可行性** | LoRA 路径**可行但紧张**，关键瓶颈是 reward judge 占卡 —— 用「换轻量 reward」解决（见 §4），不要换模型 |
| **reward judge 能否换掉** | ✅ **能，且强烈建议**。判官本质是「VLM 当 OCR 用 + Levenshtein 打分」，可直接换 PaddleOCR/RapidOCR，省掉整张判官卡（见 §4） |

---

## 1. examples 里到底有哪些模型的训练脚本？（关键事实纠正）

**⚠️ 重要发现：verl main 分支没有任何 diffusion RL 训练代码。**

`examples/README.md` 的表格里**记载**了 `flowgrpo_trainer/` 和 `examples/vllm_omni/`，但这两个目录在 main（HEAD 2026-06-10）里**根本不存在**。即 README 先行、代码未合。FlowGRPO 实现位于一组**已 close 但未并入 main** 的 stacked PR：

| PR | 内容 | 状态 |
|---|---|---|
| **#5297** | `[fsdp,trainer,vllm_omni,algo]` 主体：QwenImage FlowGRPO 训练全链路 | closed（含全部示例与 config） |
| #5716 | `[2/n][rollout]` diffusion agent loop（去噪 rollout） | closed |
| #5713 | `[3/n][reward]` image-based rewards（rule-based & genrm） | closed |

**含义（对环境搭建的直接影响）**：装 verl「main 源码」**拿不到 FlowGRPO**。必须 checkout PR #5297 的代码（已 fetch 到本地 `pr-5297` 分支）。这是 pre-release 项目的典型形态——能力分布在 PR 而非 tag/main。

### 各模型族真实状态（来自官方公告 + 代码核对）

| 模型 | 架构 | 状态 | 算法 | 代码位置 |
|---|---|---|---|---|
| **Qwen-Image** | 纯 DiT, T2I (~20B) | ✅ **Released** | FlowGRPO / MixGRPO / GRPO-Guard | PR #5297（已验证存在完整示例） |
| BAGEL | 统一理解生成 (7B+7B MoT) | 🟡 PR-ready | FlowGRPO | 未并入 main |
| Qwen3-Omni-Thinker | AR (文/图/视频/音频) | 🟡 PR-ready | GSPO | 未并入 main |
| Wan2.2 | DiT, T2V | 🔶 WIP | DanceGRPO | roadmap |
| SD3.5 | DiT, T2I | 🔶 WIP | DPO | roadmap |
| HunyuanImage-3.0 | 统一理解生成 | 🔶 Planned | MixGRPO / SRPO | roadmap |
| **Z-Image** | — | ⛔ **RFC-only** | — | **无代码，仅 RFC motivation 提及** |

> 结论印证计划书：**优先选官方 E2E 覆盖最完整的 Qwen-Image FlowGRPO**，路径最稳；模型大小用 LoRA 解决，不要为了「小」去选支持不完整的模型。

---

## 2. FlowGRPO 示例默认配置（逐字段，来自 PR #5297 真实脚本）

PR #5297 提供 4 个脚本：`run_flowgrpo.sh`(LoRA 基准) / `run_flowgrpo_fast.sh`(smoke) / `run_flowgrpo_full_ft.sh`(全参) / `run_flowgrpo_async_reward.sh`(异步 reward)。

### 模型与训练后端
| 项 | 值 |
|---|---|
| 策略模型 | `Qwen/Qwen-Image`（DiT，~20B），tokenizer 单独路径 |
| 训练引擎 | **DiffusersFSDP**（`verl/workers/engine/fsdp/diffusers_impl.py`） |
| rollout 引擎 | `vllm_omni`（`actor_rollout_ref.rollout.name=vllm_omni`） |
| reward 引擎 | `vllm`（判官用标准 vllm 起 OpenAI server） |

### 分辨率 / 去噪 / batch（默认值来自 `config/rollout/diffusion_rollout.yaml` + 脚本覆盖）
| 参数 | 默认 (yaml) | 脚本覆盖值 | 说明 |
|---|---|---|---|
| `image_height × image_width` | **512 × 512** | — | 训练默认分辨率 |
| `num_inference_steps`（训练 rollout） | **10** | — | 去噪步数（训练） |
| `val_kwargs.num_inference_steps`（评估） | 40 | **50** | 评估用全步数 |
| `sde_window_size` | null | **2** | Denoising Reduction：每条轨迹只训 2 步 |
| `sde_window_range` | null | **[0, 5]** | 训练步从前 5 个高噪步里采 |
| `rollout.n`（组大小） | 1 | **16** | 每 prompt 采 16 条去噪轨迹（GRPO group） |
| `noise_level` | 0.7 | **1.2** | SDE 噪声注入强度 |
| `guidance_scale` | 4.5 | 4.0 | CFG；FlowGRPO 可设 1.0（RL 自带 CFG 蒸馏） |
| `data.train_batch_size` | — | **32** | |
| `data.max_prompt_length` | — | **1058** | Qwen-Image 文本编码器上下文较长 |
| `ppo_mini_batch_size` / `micro_batch_size_per_gpu` | — | 16 / 16 | |

### GPU 数量与型号（关键账）
| 脚本 | `trainer.n_gpus_per_node` | LoRA? | 判官部署 | GPU 总数 |
|---|---|---|---|---|
| `run_flowgrpo.sh` (LoRA 基准) | 4 | rank 64 / α 128 | **colocate**：judge `TP=4` 与训练共卡 | **4** |
| `run_flowgrpo_fast.sh` (smoke) | 4 | 同上 | colocate TP=4 | 4 |
| `run_flowgrpo_full_ft.sh` (全参) | 8 | ❌ 全参 + param_offload | colocate TP=4 | **8** |
| `run_flowgrpo_async_reward.sh` | 4 | LoRA | **独立资源池**：`enable_resource_pool=True, n_gpus=1, TP=1` | **5**（4+1） |

> 官方公告说「tested on 4–5 H800 / 4 H200」—— **「4–5」的来源就在这里**：colocate=4，async 独立判官=4+1=5。
> 官方吞吐：LoRA 0.305 img/GPU/s，全参 0.510 img/GPU/s（H800/H200）。

---

## 3. LoRA 支持？（是，已接入训练引擎）

✅ **原生支持，且是官方推荐路径**。证据（`run_flowgrpo.sh`）：
```
actor_rollout_ref.model.lora_rank=64
actor_rollout_ref.model.lora_alpha=128
actor_rollout_ref.model.target_modules=[
  'to_q','to_k','to_v','to_out.0',          # DiT 自注意力
  'add_q_proj','add_k_proj','add_v_proj','to_add_out',  # 文图 cross-attn
  'img_mlp.net.0.proj','img_mlp.net.2',     # 图像 MLP
  'txt_mlp.net.0.proj','txt_mlp.net.2']     # 文本 MLP
```
- LoRA 直接挂在 DiffusersFSDP 训练引擎上，target 覆盖 DiT 的 attn + mlp。
- 配合 `fsdp_config.param_offload=True` + `optimizer_offload=True` + `rollout.layered_summon=True`（分层收集权重，省峰值显存）。
- 这套是 4 卡能跑的核心——**全参要 8 卡，LoRA 才 4 卡**。
- ⚠️ 权重同步链路（FSDP 训练侧 LoRA → vllm_omni rollout 侧）是 pre-release 最易出 bug 处，Phase 1 重点盯（计划书已列）。

---

## 4. Reward judge 部署 & 能否换轻量？（最重要的省卡点）

### 现状部署
- 判官模型：脚本里写 `Qwen3-VL-8B-Instruct`，但 **`reward_fn.py` 里 `compute_score_ocr` 的代码默认值其实是 `Qwen2.5-VL-3B-Instruct`**（脚本覆盖成了 8B）。
- 部署方式：verl 用 vllm 起一个 **OpenAI 兼容 HTTP server**（`/v1/chat/completions`），reward 函数通过 `reward_router_address` 发图过去打分。
- 两种摆放：
  - **colocate**（默认）：判官 `TP=4` 和训练抢同 4 张卡，靠 `gpu_memory_utilization=0.5` 给判官留空间。
  - **async + 独立池**：判官独占 1 卡（`gpu_memory_utilization=0.9, TP=1`），共 5 卡。

### ✅ 能换，而且很容易——这是 4×5090 的关键解法
`compute_score_ocr` 的逻辑拆开看：
1. 把生成图发给 VLM，prompt 是 **"output only the text content from the image"** —— **纯粹把 VLM 当 OCR 用**，不是审美/语义评判。
2. 取回文本，和 ground_truth 做 **Levenshtein 距离**，`score = 1 - dist/len(gt)`。

因为 reward 是 `custom_reward_function`（你完全控制这个 Python 函数），所以有两条降卡路径：

| 方案 | 改动 | 省卡效果 | 推荐度 |
|---|---|---|---|
| **A. 换小 VLM** | `reward_model.model_path` → Qwen2.5-VL-3B（代码本来的默认），`TP=4→1` | 判官显存大降，仍占 GPU | 中 |
| **B. 换本地 OCR**（PaddleOCR / RapidOCR） | 改 `compute_score_ocr`：把 `chat_complete` 调用换成本地 OCR 推理，**保留 Levenshtein 打分逻辑** | **直接干掉判官卡**，reward 跑在 CPU/边角显存 | **高（首选）** |

> 任务本身是「文字渲染 OCR reward」——客观、二值性强（计划书 Phase 1 选它正是这个理由）。用专用 OCR 替代通用 VLM 不仅省卡，**OCR 引擎在纯文字识别上往往比 8B VLM 更准更快**，信号质量更高。
> 唯一注意：PaddleOCR 在 sm_120 上若用 GPU 后端需验证；纯文字场景跑 CPU 即可，不占训练卡。

---

## 5. gpu_memory_utilization 等显存参数在哪设？

**位置**：`verl/trainer/config/rollout/diffusion_rollout.yaml`，可被命令行覆盖。

| 参数 | 默认 | 含义（diffusion 语境） |
|---|---|---|
| `gpu_memory_utilization` | **0.5** | rollout 引擎占显存比例。**注意是 0.5 不是 LLM 常见的 0.9**——因为要给判官 colocate + 训练侧 offload 留空间 |
| `dtype` | bfloat16 | |
| `enforce_eager` | False | 允许 cudagraph（sm_120 上需实测是否 fallback） |
| `free_cache_engine` | True | rollout 后释放 KV/cache 引擎显存，给 update 腾地方 |
| `max_model_len` | null（脚本设 1058） | |

判官侧另有独立的 `reward.reward_model.rollout.gpu_memory_utilization`（async 脚本里设 0.9）。

> **diffusion 显存账与 LLM 的本质差异**：rollout 阶段没有 KV cache，但有「text encoder + DiT(N 步去噪中间 latent) + VAE」三组件常驻，峰值常出现在 VAE decode 或 text encoder 加载时刻（计划书 Phase 1 预期坑 #2）。`gpu_memory_utilization=0.5` 的保守默认正是为这套异构流水线 + colocate 判官留的余量。

---

## 6. 对 4×5090 的最终账与建议

**5090 = 32GB/卡，4 卡 = 128GB（vs H800 80GB×4 = 320GB）。显存只有官方测试机的 40%。**

裁决：
1. **走 LoRA 路径**（4 卡入口），不碰全参（要 8 卡）。
2. **reward 用方案 B（本地 OCR）替换判官**——直接释放 colocate 判官占用的显存/算力，这是 32GB 小卡能跑通的关键变量。
3. 起步配置进一步降档（计划书 Phase 1 已规划）：512×512 已是默认、`sde_window_size` 保持 2、必要时 `rollout.n` 从 16 降到 8、`gpu_memory_utilization` 视 OOM 下调。
4. 单卡（本地 1×5090）：32GB **装不下 Qwen-Image bf16 全量推理**（权重 ~40GB），单卡只做**代码 dry-run / FP8 量化连通性验证**，不追求训练效果（计划书 0.3）。
5. **若 LoRA + 本地 OCR 在 4×5090 上仍反复 OOM** → 升 8×5090 或换 4×A100-80G（计划书风险表）。建议在 Phase 0 用显存账先判，别等正式训练。

> 一句话：**模型选 Qwen-Image-LoRA-FlowGRPO（已锁定），省卡靠「本地 OCR 替判官」而非换模型。**
