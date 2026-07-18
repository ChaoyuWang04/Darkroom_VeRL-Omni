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
- **✅ 已裁决（2026-06-12）**：装 **diffusers 0.38.0**（满足 vllm-omni 硬钉），import 全通过。
  verl PR #5297 自带 `verl/utils/diffusers/`（pipeline + SDE scheduler），Qwen-Image pipeline 走
  **verl 内置 + vllm_omni custom_pipeline**，不依赖 diffusers main；如后续确有缺失模型支持再单独处理。

---

## P6. uv 经代理下载死锁（futex），1 小时零字节 ⚠️核心坑

- **现象**：`uv pip install vllm==0.22.0 --torch-backend=cu129`（带 Clash 代理 7890）跑满 **1 小时**，
  env / uv cache 大小**零增长**、输出文件全空。进程活着但 CPU 0.7% 空转，内核等待点 `futex_do_wait`。
- **根因**：**不是网络问题**——代理连 download.pytorch.org → HTTP/2 200、清华 pypi → 200 都通，mihomo 正常。
  是 **uv 0.11.16 经 mihomo 代理下载（大文件/高并发）时的内部死锁**（futex，疑似 tokio/reqwest 在代理下的竞态），
  卡在用户态线程锁而非 socket recv，所以一个包都没下成。
- **解决**：**剥离代理 + 国内镜像直连**。`env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY … uv pip install …
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cu129 --extra-index-url 清华pypi`。
  torch 大 wheel 走阿里云 `pytorch-wheels`（有 cu128/cu129/cu130），其余走清华 pypi（连 tilelang/xgrammar/
  tokenspeed 都有镜像）。剥离代理后 **888ms 解析、几分钟装完**。
- **教训**：呼应 P1/P2——**国内源直连、只有境外站才走代理**；uv 经代理下 GB 级 wheel 不稳，torch 一律走国内镜像。
  诊断卡顿先看「**下载是否真在进行**」（cache/env 大小是否增长），而非只看进程是否存活——否则会像这次空跑 1 小时。

## P7. CUDA 变体落到 cu130（非计划的 cu129），但 ABI 兼容 ✅

- **现象**：阿里云 cu129 index + 清华 extra-index 装 torch，uv 在双 index 间选了清华的
  **torch 2.11.0+cu130**（默认 wheel，带 nvidia-cu13 库），而非计划的 +cu129。
- **根因**：清华 pypi 的 torch 2.11.0 默认 wheel 是 cu130 build；版本号与阿里云 +cu129 相同，
  `--index-strategy unsafe-best-match` 下 uv 选了清华那个。
- **裁决：实测后保留 cu130**。`get_device_capability()==(12,0)` + bf16 matmul OK；
  `import vllm._C`（vllm 0.22 的 **cu129 预编译** kernel）在 cu130 上**无 undefined symbol** ——
  CUDA 13 后向兼容 12.9，ABI 通过。故**不重装 cu129，省 2.7GB 重下**。
- **教训**：**预编译优先 + 实测验证 > 纸面假设**（修正 P3 对 cu129 的预期）。ABI 兼容性用 `import vllm._C`
  一行即可判定，不必为「变体不符预期」盲目重装。若必须 cu129：写 `torch==2.11.0+cu129`（local version 强制从阿里云取）。

## P8. transformers 版本冲突 + vllm-omni 版本号 warning（自解 / 一行修）

- **transformers**：vllm 0.22 单装时贪新到 **5.11.0**；装 vllm-omni（要 `<5.9`）时 uv **自动降到 5.8.1**
  —— 这是 vllm(`≥4.56`) ∩ vllm-omni(`<5.9`) 的交集，两者兼容，import 全过，**无需手动干预**（P5 冲突自解）。
- **vllm-omni 版本号 warning**：源码 editable 装出 `0.1.dev1`（**浅克隆缺 git tags**，setuptools_scm fallback），
  触发「与 vllm 0.22 major/minor 不匹配」warning（**warn-only，不影响功能**）。
  修法：重装时加 `VLLM_OMNI_VERSION_OVERRIDE=0.22.0` → 版本号正确、warning 消失。
  （`git fetch --tags` 后因浅克隆 `git describe` 仍失败，故用 override 而非 unshallow。）

---

## P9. pr-5297（较老）调用了 vllm-omni 已改名的 API：`AsyncOmniEngineArgs` ⚠️训练前必修

- **背景**：pr-5297 是 stacked PR 序列的合并分支，commit 较老（早于我们装的 vllm-omni `e26f5cb` @2026-06-11）。
  Task 0.3-A 用「import 级符号比对」逐一核对 pr-5297 调用的 15 处 vllm_omni API，**14 处全通过，1 处漂移**。
- **现象**：`verl/workers/rollout/vllm_rollout/vllm_omni_async_server.py`
  - L29 `from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs` → **ImportError**
  - L350 `AsyncOmniEngineArgs.from_cli_args(args)`
  - vllm-omni 现在的类名是 **`OmniAsyncEngineArgs`**（词序对调；`class OmniAsyncEngineArgs(AsyncEngineArgs, OmniEngineArgs)`，同一个类，`from_cli_args` 继承自 EngineArgs，可用）。
- **影响面（非仅 async 路径）**：该文件由 `verl/workers/rollout/replica.py::_load_vllm_omni()`（rollout 模式 `vllm_omni` 的注册加载器）惰性 import。
  4 个 FlowGRPO 脚本全部设 `actor_rollout_ref.rollout.name=vllm_omni`，故 **colocate 主路径（run_flowgrpo.sh）在 rollout init 时也会触发**，是训练 blocker。
  好在 `replica.py` 顶层 import 不受影响（惰性 import），import verl 本身全通过。
- **修法（2 行 rename，1:1 等价）**：`vllm_omni_async_server.py` 里 `AsyncOmniEngineArgs` → `OmniAsyncEngineArgs`（L29 + L350）。
- **状态**：**未改上游**（Task 0.3 本地 smoke 不走 verl rollout 封装，故非本阶段关键路径）；
  **记录在此，Phase 1 云端训练前应用此 patch**。深层「方法签名/属性级」漂移只能在 Phase 1 实跑 rollout 时才能完全暴露，import 级比对覆盖不到。
- **教训**：pre-release「commit pin」下，上游 A（verl PR）较老 + 上游 B（vllm-omni main）滚动更新 → API 漂移必然存在；
  `import 级符号逐一比对` 能廉价抓出「改名/删除」类漂移（呼应 P7 用 `import vllm._C` 一行判 ABI），但抓不到「同名不同签名」。
