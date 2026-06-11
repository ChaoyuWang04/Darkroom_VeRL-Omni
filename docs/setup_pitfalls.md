# VeRL-Omni 环境搭建踩坑记录

> 硬件：1×RTX 5090 (32GB, sm_120) · CUDA toolkit 13.2 · Ubuntu · 中国大陆（Clash 代理 @127.0.0.1:7890）
> conda 26.3.2 (`~/Downloads/ENTER`) · uv 0.11.16 · Python 3.12 env `verl-omni`

每条格式：**现象 → 根因 → 解决 → 教训**。

---

## P1. conda 创建 env 时 SSL EOF，但 git clone / curl 都正常

- **现象**：`conda create -n verl-omni python=3.12` 反复报
  `CondaSSLError ... SSLEOFError(8, 'EOF occurred in violation of protocol')`，
  对 tsinghua / sustech 镜像全挂；同时 `git clone`（走 ghfast.top）和 `curl` 同域名都 200 OK。
- **根因**：环境里设了 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY=http://127.0.0.1:7890`（Clash）。
  conda 自带的旧 OpenSSL 经代理做 TLS 握手时被中断；curl/git 用系统 TLS 栈能容忍。
  **而且国内镜像本就不需要走代理**——经代理反而绕远 + 触发握手 bug。
- **解决**：给 conda 命令单独剥离代理变量：
  ```bash
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY \
      -u https_proxy -u http_proxy -u all_proxy \
      conda create -n verl-omni python=3.12 -y \
      --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  ```
  （`--override-channels` 顺带绕开 `~/.condarc` 里挂掉的 sustech nvidia channel。）
- **教训**：**国内镜像走直连、国外源走代理**，二者分流。代理是为 GitHub/HF/pytorch.org 这类境外站准备的，硬塞给国内镜像只会引入 TLS 故障变量（呼应 5090 工作站「最小化引入变量 / 工具分工」教训）。

## P2. 工具选型：pip→tsinghua 已配好，但大 wheel 用 uv 更稳

- **现状（已存在的好配置）**：`~/.config/pip/pip.conf` 已是 tsinghua；`HF_ENDPOINT=https://hf-mirror.com` 已设；
  git remote 已用 `ghfast.top` 前缀代理。镜像基建沿用 TRELLIS 时期的，无需重搭。
- **决策**：vllm/torch 这类大包安装用 **uv**（`uv pip install --python $CONDA_PREFIX/bin/python ...`）。
  原因：uv 用 rustls，经 Clash 代理握手稳定（不像 conda 的 OpenSSL），且 `--torch-backend=cu129`
  能直接锁定 CUDA wheel 变体。

## P3. sm_120 + CUDA 13.2 的根本张力 → 安装策略：能用预编译 binary 绝不源码编 CUDA 扩展

- **背景**：系统 toolkit 是 **CUDA 13.2**（很新）；而 vllm 0.22.0 / torch 预编译 wheel 是 **cu129 (CUDA 12.9)**。
- **关键判断**：
  - 预编译 binary（vllm/torch/flash 的 cu129 wheel）**自带 12.9 runtime**，与系统 13.2 toolkit 共存无碍，
    且 cu129 wheel **已含 sm_120 kernel**（Blackwell）。→ **走预编译**。
  - 任何「源码编译 CUDA 扩展」都会用系统 `nvcc 13.2`，对 vllm/flash-attn 的源码而言版本过新，极易编译失败。
    → **避免源码编 vllm-core / flash-attn**。
- **落地策略**：
  - vllm-core：**预编译** `vllm==0.22.0`（cu129）—— 不源码编。
  - vllm-omni / verl(+PR #5297) / diffusers：**纯 Python 层，源码可编**（`pip install -e .`），不碰 CUDA 编译。
  - 印证 TRELLIS 教训：**社区验证的 binary 路径 > 在新硬件上蛮力源码编**。

## P4. 「装 verl main 源码」拿不到 FlowGRPO —— 能力在未合并 PR 里

- **现象**：verl main（HEAD 2026-06-10）的 `examples/` 没有 `flowgrpo_trainer/`，
  全仓搜 `flowgrpo`/`diffusers_fsdp`/`qwen_image` 几乎为零（只有 README 记载）。
- **根因**：FlowGRPO 实现位于 **已 close、未并入 main** 的 stacked PR：**#5297**（主体）/ #5716 / #5713。
- **解决**：fetch PR head 到本地分支再用：
  ```bash
  cd ~/code/upstream/verl
  git fetch --depth 1 https://ghfast.top/https://github.com/verl-project/verl.git pull/5297/head:pr-5297
  # 5716/5713 同理；5713 曾遇一次性 TLS 中断，重试即可
  ```
- **教训**：pre-release 项目「commit/PR pin」而非「版本号 pin」。装 main 之外，必须显式 checkout 功能 PR
  （详见 `model_decision.md` §1 与 `env_lock.md`）。

## P5. diffusers 版本冲突：vllm-omni 钉死 0.38.0，计划书要 main

- **现象**：vllm-omni `requirements/common.txt` **硬钉 `diffusers==0.38.0`**；
  而计划/任务要求装 diffusers main（Qwen-Image/Z-Image 支持）。
- **状态**：待安装阶段裁决。注意 verl PR #5297 自带 `verl/utils/diffusers/`（pipeline + SDE scheduler），
  Qwen-Image pipeline 走的是 **verl 内置 + vllm_omni custom_pipeline**，未必依赖 diffusers main。
- **倾向**：优先满足 vllm-omni 的 0.38.0 钉子（保证 import 不崩）；diffusers main 仅在确有缺失模型支持时
  再单独处理，避免破坏 vllm-omni。最终方案见 `env_lock.md`。

---

## （待补）安装过程中实际遇到的报错
> 安装 vllm/vllm-omni/verl 过程中的具体报错与解法，完成后补到此节。
