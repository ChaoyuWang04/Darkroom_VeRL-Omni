# VeRL-Omni 环境锁定（env_lock.md）

> 实测复现并验证于 **2026-06-12**。所有 import 通过、sm_120 实测可用。
> - 硬件：RTX 5090 (32GB, **sm_120**) · CUDA driver 595.71.05 (CUDA 13.2)
> - conda env **`verl-omni`** (Python 3.12.13) · 263 包 · 9.8G
> - 安装器：uv 0.11.16 · 国内镜像直连（**剥离代理**，详见 §3 与 `setup_pitfalls.md` P6）

---

## 1. 核心版本锁定（实测组合）

| 组件 | 版本 | 备注 |
|---|---|---|
| **torch** | **2.11.0+cu130** | sm_120 实测 `get_device_capability()==(12,0)` + bf16 matmul OK |
| torchvision / torchaudio | 0.26.0+cu130 / 2.11.0+cu130 | |
| **vllm** | **0.22.0** | 预编译 wheel（cu129 build），`_C`/`_moe_C` 在 cu130 上 ABI 兼容 |
| **vllm-omni** | **0.22.0**（源码 editable） | commit pin 见 §2；版本号用 `VLLM_OMNI_VERSION_OVERRIDE` 修正 |
| **verl** | **0.9.0.dev0**（源码 editable） | commit pin 见 §2 |
| transformers | **5.8.1** | vllm(≥4.56) ∩ vllm-omni(<5.9) 的交集，二者兼容 |
| diffusers | 0.38.0 | vllm-omni 硬钉；Qwen-Image pipeline 走 verl 内置，足够 |
| flashinfer-python / -cubin | 0.6.11.post2 | vllm attention 后端 |
| fa3-fwd | 0.0.3 | 预编译 wheel（**非源码编**） |
| triton / xgrammar | 3.6.0 / 0.2.1 | |
| tensordict / ray / accelerate | 0.10.0 / 2.55.1 / 1.12.0 | verl 训练栈 |
| numpy | 1.26.4 | |
| nvidia-cublas / cudnn-cu13 / nccl-cu13 | 13.1.0.3 / 9.19.0.56 / 2.28.9 | **cu13 栈**（随 torch+cu130） |

> ⚠️ **CUDA 变体说明**：最终是 **cu130 栈**（torch+cu130 + nvidia-cu13 库），而非计划的 cu129。
> vllm 0.22 的 cu129 预编译 kernel 在 cu130 上**实测 ABI 兼容**（CUDA 13 后向兼容 12.9），
> `import vllm._C` 无 undefined symbol。详见 `setup_pitfalls.md` P7。

## 2. 上游源码 commit pin

| 仓库 | 路径 | 分支 | commit |
|---|---|---|---|
| vllm-omni | `~/code/upstream/vllm-omni` | main | `e26f5cb918caea66ca403155d82d9ce4fbb97982` |
| verl | `~/code/upstream/verl` | main | `41a52449db58996731f54b79d20b939f5297fc4d` |
| verl (FlowGRPO) | 同上 | **pr-5297**（✅ 已 checkout @Task0.3-A） | `1aa669362bea3f19d8548b1d0873d3d0442684c8` |
| verl (#5716 rollout) | 同上 | pr-5716（已 fetch） | `851dd35…`（pr-5297 已自含其内容，无需合并） |

> **跑 FlowGRPO 训练前必须切到 pr-5297**：`git -C ~/code/upstream/verl checkout pr-5297`。
> verl 是 editable 安装，main↔pr-5297 的 pip 依赖**完全相同**（pr-5297 只多 diffusion 代码文件），
> 切分支零成本、无需重装。依据见 `model_decision.md` §1 与 `setup_pitfalls.md` P4。
>
> **Task 0.3-A 完整性结论（2026-06-12 实测）**：pr-5297 分支 **自含 stacked PR 全部内容**
> （#5297 主体 + #5716 diffusion rollout + #5713 image reward），git 历史是一个把 `verl-omni` 整条
> feature 分支合入的 merge commit。验证点：① `examples/flowgrpo_trainer/` 4 个 run 脚本 + 3 个 config 齐全；
> ② `DiffusionAgentLoopWorker`/`DiffusionSingleTurnAgentLoop`/`DiffusionRolloutConfig`/`vllm_omni_rollout.py` 在；
> ③ `tests/experimental/reward_loop/reward_fn.py::compute_score_ocr` 在；④ import 全通过（verl 0.8.0.dev）。
> **无需 fetch/合并 5716/5713**。**唯一 API 漂移**：`AsyncOmniEngineArgs`，详见 `setup_pitfalls.md` **P9**（Phase 1 前修）。

## 3. 完整复现安装命令（剥离代理 + 国内镜像直连）

```bash
VERLPY=~/Downloads/ENTER/envs/verl-omni/bin/python
ALI=https://mirrors.aliyun.com/pytorch-wheels/cu129     # 有 cu129/cu130；torch 大 wheel 走这里
TUNA=https://pypi.tuna.tsinghua.edu.cn/simple            # 其余包

# 关键：剥离代理，国内源直连（经代理会触发 uv 死锁，见 P6）
strip() { env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy "$@"; }

# 1) torch 三件套（注：双 index 下 uv 实际选到 +cu130，sm_120 可用）
strip uv pip install --python $VERLPY torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url $ALI --extra-index-url $TUNA

# 2) vllm 预编译（只下 3 包，不动 torch）
strip uv pip install --python $VERLPY vllm==0.22.0 \
    --index-url $TUNA --extra-index-url $ALI --index-strategy unsafe-best-match

# 3) vllm-omni 源码（transformers 自动降到 5.8.1；VERSION_OVERRIDE 修版本号 warning）
cd ~/code/upstream/vllm-omni
VLLM_OMNI_TARGET_DEVICE=cuda VLLM_OMNI_VERSION_OVERRIDE=0.22.0 \
  strip uv pip install --python $VERLPY -e . \
    --index-url $TUNA --extra-index-url $ALI --index-strategy unsafe-best-match

# 4) verl 源码
cd ~/code/upstream/verl
strip uv pip install --python $VERLPY -e . \
    --index-url $TUNA --extra-index-url $ALI --index-strategy unsafe-best-match
```

## 4. 验收（import 全通过）

```bash
$VERLPY -c "import torch, vllm, vllm._C, vllm_omni, verl, diffusers, transformers; \
print(torch.__version__, torch.cuda.get_device_capability()); \
print('vllm', vllm.__version__, '| omni', vllm_omni.__version__, '| verl', verl.__version__)"
# → 2.11.0+cu130 (12, 0)
# → vllm 0.22.0 | omni 0.22.0 | verl 0.9.0.dev0
```
