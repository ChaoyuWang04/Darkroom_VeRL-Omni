# 计划书 Task 1：VeRL-Omni 多模态/Diffusion RL 后训练

> 目标硬件：本地 1×RTX 5090 (32GB, sm_120) 做 smoke test → 云端 4×5090 正式训练（必要时升级 8 卡）
> 核心目标：跑通 diffusion / omni 模型的 RL 后训练全链路，深度学习 VeRL-Omni（verl + vllm-omni）的架构，系统性踩坑
> 框架版本基线：VeRL-Omni pre-release（2026/05），vllm-omni ≥ 0.22.0
> 预计周期：3–4 周（含 buffer）

---

## 0. 全局认知：这个任务和普通 LLM RL 的本质区别

在动手前先建立三个心智模型，后面每个 Phase 都会反复验证它们：

1. **Rollout 不再是 token 序列，而是连续 latent 空间中的去噪轨迹。**
   GRPO 的 "response" 变成了一条 SDE/ODE 采样路径，log-prob 的计算对象是每一步去噪的高斯转移概率（Flow-GRPO 的核心 trick：把确定性 ODE 转成随机 SDE 才能定义 likelihood）。这是和 LLM RL 在数学层面最大的不同，建议先推一遍 Flow-GRPO 论文里的公式。

2. **单次 rollout 是异构多组件流水线：text encoder → DiT → VAE。**
   不同组件显存峰值、计算特性完全不同（DiT 是迭代 N 步的 compute-bound，VAE decode 是一次性 memory-bound）。这直接导致调度和显存管理比 LLM 复杂得多——也是 vllm-omni 存在的理由。

3. **Reward 本身就是一个（或多个）多模态模型。**
   OCR scorer、VLM judge、aesthetic model 都要占卡或占 API。VeRL-Omni 当前的异步化只做到了 async-reward 这一层（fully-async 还在 roadmap 上），所以 reward 服务的部署方式是影响吞吐的一等公民。

**和 Task 2 的关系**：这里学的是「rollout 形态异构化之后 RL 框架如何应对」；Task 2 学的是「rollout 时间长尾化之后 RL 框架如何应对」。两个任务合起来覆盖 RL infra 的两大前沿矛盾。

---

## Phase 0：单卡侦察 + 可行性裁决（本地 5090，1–2 天）

**目标**：在花一分钱租卡之前，搞清楚三件事：环境能不能装、模型选哪个、4×5090 到底够不够。

### 0.1 环境搭建（预期是本任务第一个大坑区）

- [ ] 新建独立 conda env，安装 vllm-omni（注意 release notes 中已声明 Blackwell 量化支持，但 sm_120 上的 wheel 兼容性需要实测；沿用你 TRELLIS.2 的经验——社区 issue 里的验证配置 > 官方文档）
- [ ] 安装 verl + VeRL-Omni（pre-release，大概率要装 main 分支源码而不是 pip 包）
- [ ] 国内镜像策略先行：PyPI 镜像 + HF-mirror + GitHub 代理，写进 env setup 脚本，不要临时救火
- [ ] 验收：`python -c "import vllm_omni"` 通过，verl-omni 的 example 目录能看到 Flow-GRPO 脚本

**学习点**：pre-release 项目的依赖锁定方式（commit pin 而不是版本号 pin）；vllm-omni 和 vLLM 主仓的版本对齐关系。

### 0.2 模型选型裁决（关键决策点）

VeRL-Omni 当前官方覆盖的模型族：Qwen-Image（纯 DiT）、Qwen-Omni（AR-DiT 混合）、BAGEL / HunyuanImage3.0（统一理解生成）。逐个做显存账：

| 候选 | 参数量 | bf16 权重 | 4×5090 (128GB) 可行性判断 |
|---|---|---|---|
| Qwen-Image (DiT) | ~20B | ~40GB | LoRA + FSDP 分片：勉强可行；full-param：不可行 |
| Qwen2.5-Omni-7B | ~10B 等效（Thinker+Talker） | ~20GB | LoRA 可行，full-param 紧张 |
| BAGEL-7B | ~14B (7B+7B MoT) | ~28GB | LoRA 可行 |

- [ ] 拉取 VeRL-Omni 的 example configs，确认每个模型的**官方最小 GPU 配置**（E2E 测试用了几张什么卡）
- [ ] 决策规则：**优先选官方 E2E 覆盖最完整的入口**（搜索结果显示是 Qwen-Image Flow-GRPO），路径最稳；模型大小问题用 LoRA 解决，而不是换一个支持不完整的小模型
- [ ] 如果 Qwen-Image 的 LoRA 路径在 4×5090 上算不过账（rollout 阶段 DiT + text encoder + VAE 常驻显存 > 边界），在这里就裁决升 8 卡，不要等正式训练时才发现

**学习点**：diffusion RL 的显存账和 LLM RL 完全不同——rollout 阶段没有 KV cache，但有 N 步去噪的中间 latent + 多组件常驻；学会给异构流水线做显存预算。

### 0.3 单卡能做什么（诚实预期）

32GB 装不下 Qwen-Image bf16 推理（40GB 权重），所以单卡 smoke test 的定义要调整：

- [ ] **代码侧 dry-run**：用 VeRL-Omni 的配置体系跑一个 dummy/tiny 模型（或 FP8 量化推理验证 pipeline 连通性），目标是验证「数据 → rollout → reward → update」四段代码路径都能走到，而不是验证训练效果
- [ ] **精读源码**（这是 Phase 0 最有价值的部分，列出问题清单带着读）：
  - VeRL-Omni 如何在 verl 的 HybridFlow 抽象上挂载 diffusion rollout？复用了 verl 的哪些 Worker，新写了哪些？
  - 去噪轨迹的 log-prob 在哪里计算、存储格式是什么？
  - async-reward 的实现：reward worker 和 rollout worker 之间的队列在哪？
- [ ] 产出：一张 VeRL-Omni 架构 Mermaid 图（组件 + 数据流），作为后面 blog 的骨架

**验收标准**：架构图画完 + 显存账算完 + 4 卡/8 卡裁决做出 + dry-run 路径打通。

---

## Phase 1：多卡最小训练闭环（云端 4×5090，3–5 天）

**目标**：Qwen-Image + Flow-GRPO + LoRA，跑通第一个 reward 上升的训练曲线。**这一阶段唯一 KPI 是"曲线在动且方向正确"，不是出好模型。**

### 步骤

- [ ] 租 4×5090 实例，先跑 Phase 0 写好的 env setup 脚本（验证脚本的可迁移性本身就是踩坑项）
- [ ] 任务选最简单的可验证 reward：**文字渲染（OCR reward）**——给 prompt 要求图中渲染指定文字，用 OCR 模型打分。理由：reward 客观、二值性强、Qwen-Image 本身擅长文字渲染所以信号密度高
- [ ] 配置降档起步：低分辨率（如 512×512）、少去噪步数、小 batch、LoRA rank 16–32
- [ ] 接 wandb，必看曲线：reward mean/std、KL（如有）、每阶段耗时分解（rollout / reward / update）
- [ ] 跑 50–100 step，确认 reward 上升、生成图无明显崩坏（diffusion RL 的经典失败模式：reward hacking 导致图像畸变但 OCR 分数高）

### 预期坑（提前列出，遇到时对号入座）

1. **权重同步**：LoRA 权重从 FSDP training 侧同步到 vllm-omni rollout 侧的机制是否完善（pre-release 框架在 LoRA + 异构引擎同步上最容易出 bug）
2. **显存峰值出现在 rollout 而非 update**：和 LLM RL 相反，多组件 pipeline 的峰值在 VAE decode 或 text encoder 加载时刻，OOM 时先查这里
3. **sm_120 算子覆盖**：某些 attention/采样 kernel 在 Blackwell 上可能 fallback 到慢速路径甚至报错——nsys 抓一段 timeline 看有无异常 kernel
4. **数值精度**：去噪轨迹 log-prob 对精度敏感，bf16 下 ratio 爆炸是已知问题模式

**学习点**：Flow-GRPO 的实际训练动力学；diffusion RL 的 reward hacking 形态（和 LLM 的截然不同，值得截图记录）；LoRA 在 RL 语境下的权重同步链路。

**验收标准**：100 step 内 reward 显著上升 + 生成样例肉眼可见改善 + 一份「踩坑记录 v1」。

---

## Phase 2：正式训练 + 系统性 Profiling（4×5090，5–7 天）

**目标**：把 Phase 1 的玩具配置升级为一次"认真"的训练，同时用你的 profiling 方法论解剖这个系统。

### 2.1 训练侧

- [ ] 提升分辨率/步数/batch 到显存边界，跑一次 500+ step 的完整训练
- [ ] 做一组小 ablation（二选一，控制预算）：
  - SDE 噪声水平 / 去噪步数对训练稳定性的影响
  - reward 组合：纯 OCR vs OCR + aesthetic scorer（观察多 reward 的权衡和 hacking 方向变化）
- [ ] 评估：固定 prompt 集，训练前后对比生成质量（OCR 准确率 + 人眼盲评）

### 2.2 系统侧（核心学习产出）

按你的标准流程：nsys 先定位，ncu 只打关键 kernel。

- [ ] 测量训练循环的时间分解：rollout（去噪）/ reward 计算 / 经验处理 / update / 权重同步，画成甘特图
- [ ] 验证 async-reward 的实际收益：开/关对比一次端到端吞吐——**预判**：4 卡小规模下收益可能很小（类比你 MoE 项目的教训：overlap 类优化的收益是规模和长尾的函数），但要拿数据说话
- [ ] 观察显存时序：rollout 多组件加载/卸载的峰值模式，和 verl colocate 模式的 offload 策略如何交互
- [ ] 记录 GPU 利用率 bubble 的来源（这和你 GPU-Bubble-Lab 直接呼应——diffusion RL 的 bubble 形态是新样本）

**学习点**：异构 rollout pipeline 的性能特征；async-reward 的收益边界；把 GPU-Bubble-Lab 的分析框架迁移到新 workload。

**验收标准**：一次完整训练 + 时间分解甘特图 + async-reward 开关对比数据。

---

## Phase 3：架构深读 + 对外产出（3–5 天，部分可与 Phase 2 并行）

**目标**：把实践经验升华为结构化理解，并产出公开物。

- [ ] 带着 Phase 1/2 的实战问题二刷源码，重点：
  - VeRL-Omni 对 verl 原有抽象的**侵入点清单**（哪些是优雅扩展，哪些是 hack）——这是评价框架设计质量的直接证据
  - vllm-omni 的 disaggregated pipeline（Encoder / LLM Core / DiT / VAE 各 stage）如何被 RL 训练复用
- [ ] 技术博客：《在 4×5090 上跑通 Diffusion RL：VeRL-Omni 解剖与踩坑实录》——按你 Gated Attention 那篇的标准写，架构图 + 显存账 + 时间分解 + 坑清单
- [ ] 给上游提 issue（必做）/ PR（如果遇到了能修的 bug）：pre-release 阶段的 issue 含金量最高，附复现脚本

### Phase 3.5（可选 stretch，预算允许再做）

- [ ] 尝试 Qwen-Omni（AR-DiT 混合架构）的 RL 路径——这是从 "diffusion RL" 到真正 "omni RL" 的一步，但支持成熟度更低，时间盒 3 天，跑不通就记录卡点退出

---

## 预算估算

| 项 | 配置 | 估时 | 估价（RunPod/Nebius 量级） |
|---|---|---|---|
| Phase 1 | 4×5090 | ~30 GPU·hr×4 | $80–150 |
| Phase 2 | 4×5090 | ~60 GPU·hr×4 | $150–300 |
| Phase 3.5 | 4×5090 或 8 卡 | 时间盒 | $100 封顶 |
| **合计** | | | **$330–550** |

省钱原则：所有调试在本地单卡 dry-run 模式完成；云端实例只跑已验证的脚本；每天训练结束立即存 checkpoint 到对象存储并停机。

---

## 风险与退出条件

| 风险 | 信号 | 应对 |
|---|---|---|
| sm_120 兼容性不可逾越 | 核心 kernel 报错且无社区 patch | 云端换 H100/A100 实例（放弃 5090 执念，多花 ~30% 预算） |
| 4×5090 显存不够 | Phase 0 显存账算不过 / Phase 1 反复 OOM | 升 8×5090 或 4×A100-80G |
| pre-release bug 密集到无法推进 | 一周内 blocker > 3 个且无响应 | 降级到 Flow-GRPO 原始仓库（SD3.5-Medium 2B，单卡可跑）完成学习目标，VeRL-Omni 留 issue 跟踪 |

---

## 总学习清单（自查表）

- [ ] 能白板推导 Flow-GRPO 的 SDE log-prob 公式
- [ ] 能画出 VeRL-Omni 完整数据流图并指出每个组件的显存/计算特性
- [ ] 能解释 diffusion RL 的 reward hacking 与 LLM 的差异
- [ ] 能给出 async-reward 在不同规模下的收益判断框架
- [ ] 产出：1 篇博客 + ≥1 个上游 issue + 踩坑清单
