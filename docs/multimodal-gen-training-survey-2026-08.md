# 多模态生成模型训练入门：写给文本后训练工程师（2026-08 版）

> 读者画像：做过纯文本 post-training（function calling 方向的 SFT / RL / runtime），熟悉 verl、Megatron/FSDP2、SGLang/vLLM、GRPO/GSPO。
> 目标：不是读文献，而是**理解各模态训练长什么样，然后跑通一个完整的训练任务**（目标 → 数据 → 训练 → 评估 → infra）。

---

## TL;DR（30 秒版）

1. **所有模态的训练都是同一个三段论**：Pretrain（海量弱标数据 + 生成损失）→ SFT（**同一个损失，换成小而精的数据**）→ RL（reward model 或 verifier + GRPO 族 / DPO 族）。视觉的 SFT 不换 loss，只换数据分布——这是和文本最重要的同构点。
2. **最大的范式差异**：文本是"离散 token + 自回归"，图像/视频/3D/音乐主流是"连续 latent + flow matching（DiT 架构）"。一次生成 = 一条 25~50 步的去噪轨迹。把每个去噪步看成一个"token"，整条轨迹看成一条 response，GRPO 就能搬过来——这正是 Flow-GRPO / DanceGRPO 干的事。
3. **客观指标是存在的，而且非常像 function calling 判卷**：图像有 GenEval（用检测器数物体/颜色/位置，对错分明）和 OCR（图里的字渲染对没对）；TTS 有 WER（用 Whisper 把生成语音转回文字算错误率）和说话人相似度 SIM。这些就是视觉/音频界的 "JSON 格式对不对"。主观侧则由学习型 reward model（HPS、ImageReward、VideoAlign）和人类 Elo 竞技场（LMArena、Artificial Analysis）覆盖。
4. **infra 有一一对应**：verl → **verl-omni**（verl 官方社区的多模态生成 RL 框架，2026 年 5 月发布，rollout 用 vLLM-Omni）；sglang/vllm → vLLM-Omni / sglang-omni / ComfyUI / xDiT；GRPO/GSPO → FlowGRPO / DanceGRPO / Diffusion-DPO。你现有的 verl 技能可以近乎无损迁移。
5. **runtime 问题的答案**：绝大多数生成模态是**单步 bandit**（prompt → 生成 → 打分，没有环境状态转移），不需要 agentic runtime，reward 服务化（异步多 reward serving）才是它们的"runtime"。真正有环境交互的是**世界模型**——它不是"被 RL 训练的对象"，它本身就是"给 agent 做 RL 的环境"。

---

## 0. 总对照表：文本栈 ↔ 各模态栈

先把地图摊开，后面每一章都是对这张表的展开。

| 环节 | 文本（你熟的） | 图像 | 视频 | 语音 TTS | 音乐 | 3D | 世界模型 |
|---|---|---|---|---|---|---|---|
| **基座范式** | AR Transformer，next-token | MMDiT + flow matching（也有 AR 统一模型路线） | 3D causal VAE + DiT + flow matching | AR codec-LM（离散音频 token）或 NAR flow matching | LM planner + DiT，或纯 AR | 两段式：几何 latent 流模型 + 纹理多视角扩散 | 视频模型 + 动作条件，蒸馏成因果实时流式 |
| **预训练损失** | 交叉熵 | flow matching / ε-MSE | 同左（时空 latent 上） | 交叉熵（token）或 FM | FM 为主 | rectified flow | FM + 动作条件 |
| **代表开源模型 (2026-08)** | Qwen3 / DeepSeek 系 | FLUX.2、Qwen-Image 2.x、Z-Image、GLM-Image、HunyuanImage 3.0 | Wan 2.2 (MoE A14B)、LTX-2.5、HunyuanVideo 1.5 | Step Audio EditX、CosyVoice、MOSS-TTS、F5-TTS | ACE-Step v1.5、YuE | TRELLIS / TRELLIS.2、Hunyuan3D 2.1 | Matrix-Game 3.0、LingBot-World、minWM |
| **SFT 是什么** | 精选指令数据上继续 CE | 精选图 + 同一 FM 损失；LoRA 学风格/主体/指令编辑 | LoRA 学风格/运镜/角色 | 声音克隆、情感、说话风格微调 | 几首歌训 LoRA 学曲风 | 领域资产微调（如 CAD、游戏风格） | 动作标注数据上的对齐微调 |
| **RL 优化什么** | 有用性/无害性/verifier 正确率 | 美学 + 图文对齐 + 组合正确性 + 文字渲染 | 上述 + 运动质量 + 物理合理性 | 可懂度(WER↓) + 音色相似 + 自然度 | 音乐性 + 歌词对齐 + 风格贴合 | （基本空白，研究机会） | 动作可控性、长程一致性 |
| **RL 算法** | GRPO/GSPO/PPO/DPO | Flow-GRPO、DanceGRPO、Diffusion-DPO、SRPO、Pref-GRPO | 同左（rollout 更贵） | GRPO/DPO（AR 路线直接用 LLM 那套） | DPO / intrinsic RL（ACE-Step 1.5） | DreamDPO 等零星工作 | 少直接 RL；它是 RL 的环境 |
| **Reward 来源** | RM / rule verifier / LLM judge | HPSv3、ImageReward、PickScore、UnifiedReward；GenEval 检测器、OCR | VideoAlign（VQ/MQ/TA）、VisionReward | −WER（Whisper 判卷）、SIM、UTMOS | 美学 RM、CLAP 对齐 | 渲染后用 2D RM / VLM judge | 动作跟随 F1、下游 agent 成功率 |
| **客观"判卷"指标** | JSON 合法、函数名/参数 exact match、单测通过 | GenEval、T2I-CompBench、OCR 准确率 | VBench 各维度、Physics-IQ、VideoPhy | **WER/CER**、SIM、时长控制误差 | 歌词 WER、节拍/调性检测 | 渲染 FID_CLIP、CLIP-score、CMMD、LPIPS | 动作跟随 F1、记忆保持时长、FPS |
| **主观/学习型指标** | RM 分、Chatbot Arena Elo | HPS、Aesthetic、LMArena/AA 图像竞技场 | VideoAlign、视频竞技场 | MOS/UTMOS、TTS-Arena | Audiobox-Aesthetics、听感盲测 | GPTEval3D、用户研究 | 人类可玩性评估 |
| **训练框架** | Megatron / FSDP2 (verl 内) | diffusers + accelerate/FSDP；LoRA 用 ai-toolkit / DiffSynth / kohya | FastVideo、musubi-tuner、diffusion-pipe | 标准 LLM 栈（AR 路线）/ 各家官方 repo | ACE-Step 官方（含 LoRA） | TRELLIS / Hunyuan3D 官方 repo | Matrix-Game / minWM |
| **推理引擎** | SGLang / vLLM | ComfyUI、diffusers、xDiT、Nunchaku | ComfyUI、xDiT（序列并行） | vLLM/SGLang 可直接 serve AR-TTS；sglang-omni | 官方 pipeline | 官方 pipeline | 自带实时流式 serving |
| **RL 框架** | verl / OpenRLHF / AReaL | **verl-omni**、DanceGRPO(基于 FastVideo)、flow_grpo | verl-omni（Wan2.2 / LTX2.3） | verl 思路可直接套 AR-TTS | — | — | —（agent 侧用经典 RL） |
| **需要 runtime 吗** | 需要（工具、多轮、env） | 不需要（单步 bandit）；agentic 生成刚兴起 | 不需要 | 不需要（全双工对话除外） | 不需要 | 不需要 | **它自己就是 runtime** |

---

## 1. 第一个心智转变：从"token 序列"到"去噪轨迹"

### 1.1 文本你熟的世界

策略是 π(token_t | prefix)，一次 rollout = 自回归解码出一条 response，log-prob 每步天然可得，KV cache 让长序列便宜。GRPO：同一 prompt 采 G 条 response，组内标准化 reward 当 advantage。

### 1.2 视觉/音频主流的世界：flow matching

2026 年图像/视频/3D/音乐的主流基座是 **DiT（Diffusion Transformer）+ flow matching（rectified flow）**，在 VAE 压缩后的**连续 latent** 上工作：

- 训练时：取干净样本 x₁，噪声 x₀~N(0,I)，插值 x_t = (1−t)x₀ + t·x₁，让网络预测速度场：

  **L = E‖ v_θ(x_t, t, c) − (x₁ − x₀) ‖²**

  直觉：教模型"在任意加噪程度下，往干净数据的方向指"。这就是视觉界的"交叉熵"，从预训练到 SFT 全程不换。
- 推理时：从纯噪声出发，沿 v_θ 解 ODE 走 20~50 步，得到一张图/一段视频的 latent，再过 VAE decoder。

**和 AR 的关键差异**（决定了 infra 长什么样）：

| | AR（文本） | Flow/Diffusion（视觉） |
|---|---|---|
| 一次生成 | N 个 token，逐个出 | 固定 25~50 个去噪步，每步全量 forward |
| KV cache | 有，长序列摊薄成本 | **没有**，每步都是完整一遍 DiT |
| log-prob | 每 token 天然可得 | ODE 是确定性的，**没有可算的策略概率** ← RL 的第一个坎 |
| 采样多样性 | 温度/top-p | 初始噪声 + 采样器随机性 |
| 输出空间 | 离散词表 | 连续 latent |

### 1.3 RL 怎么搬过来：ODE → SDE 这一步是全部秘密

确定性 ODE 没有概率，policy gradient 无从下手。Flow-GRPO / DanceGRPO 的核心 trick：**把 ODE 采样等价改写成 SDE 采样**——每个去噪步注入受控高斯噪声，于是每一步变成

π_θ(x_{t−1} | x_t, c) = N(μ_θ(x_t, t, c), σ_t² I)

log-prob 可算了，importance ratio 可算了，剩下的就是你熟的 GRPO：同 prompt 采 G 条去噪轨迹 → reward → 组内标准化 advantage → clip 更新。对应关系：

- 一条 response ↔ 一条去噪轨迹（每个去噪步 ≈ 一个"token"）
- rollout 引擎解码 ↔ SDE 采样（这也是为什么 rollout 是最大算力开销：G × 50 步 × 全量 DiT forward，还要过 VAE + reward model）
- 训推一致（你课里的 TIS）↔ **训练用 SDE、部署用 ODE 的分布错位**，同样需要小心；Flow-GRPO 还发现可以在 RL 时把去噪步数降到 8~10 步（denoising reduction）省 rollout，部署再用满步数

一个数量级感受：Flow-GRPO 把 SD3.5-M 的 GenEval 从 ~63% 拉到 ~95%，OCR 文字渲染准确率从 ~59% 拉到 ~92%——**verifiable reward 在视觉上同样暴力有效**，这和你在 RLVR 里见到的曲线是同一个故事。

### 1.4 另一条路线：把图像也变成 token（那就完全是你的主场）

- **离散化路线**：VQ tokenizer 把图变 token 序列，AR 预测。Emu3、Janus-Pro、HunyuanImage 3.0（80B MoE，原生自回归统一模型）走这条。
- **混合路线**：AR 骨干负责理解与规划，扩散头负责像素（Transfusion、BAGEL；GPT-4o/Nano Banana 一系普遍被认为是这类）。
- 意义：AR 统一模型的 RL **就是 LLM RL**——verl 原生能训，KV cache、GSPO、你全部的经验直接适用。BLIP3o-NEXT 有个很实用的结论：**RL 该打在"承担生成主体"的那个模块上**——Qwen-Image 里 AR 只是文本编码器，RL 打扩散侧更有效；BLIP3o 里 AR 产图 token，RL 打 AR 侧。
- 2026-08 现状：**画质天花板仍在 diffusion/flow 侧，理解在 AR 侧，统一模型两条腿并跑**。你入门建议先学 flow 侧（增量知识最大），AR 侧当作"技能已解锁"。

---

## 2. 统一三段论：SFT / RL 到底在各模态里"是什么"

### 2.1 SFT = 同一个损失，换数据分布

这是最反直觉也最重要的一点：视觉没有"指令微调换 loss"这回事。

- **质量对齐（quality tuning）**：Meta Emu 的经典发现——预训练几十亿图后，只用**几千张人工精选的顶级美学图**继续跑同一个 FM 损失，画质档次整体跃迁。对应文本里"少量高质量 SFT 数据 > 海量平庸数据"。
- **风格/主体 LoRA**：几十~几百张图训一个 LoRA（rank 16~64），学一个画风、一个角色、一个产品。这是社区体量最大的"SFT"形态（Civitai 上百万个）。音乐同理：ACE-Step 1.5 用几首歌训 LoRA 学你的曲风。
- **指令编辑 SFT**：(原图, 编辑指令, 目标图) 三元组训练，让模型听懂"把背景换成雪山"。数据多为合成（InstructPix2Pix 流水线：LLM 造指令 + 模型造配对图）。FLUX Kontext、Qwen-Image-Edit 属于这条线的产品化。
- **能力注入**：I2V（图生视频）、多视角、可控条件（相机轨迹、姿态、深度）都是"预训练底座 + 定向数据 SFT"出来的。

### 2.2 RL = reward 工程，算法反而是次要的

各模态 RL 的算法差异很小（都是 GRPO 变体或 DPO），**真正的分野在 reward 从哪来**：

1. **规则 verifier（客观，最像你的 function calling 判卷）**
   - 图像：GenEval——prompt 说"两只红色的猫在椅子左边"，就用目标检测器数猫、判颜色、验位置，逐项对错。OCR reward——prompt 要求图里写 "OPEN 24 HOURS"，用 OCR 读出来 exact match。
   - TTS：**−WER**——生成的语音丢给 Whisper 转回文字，和输入文本算词错误率；**SIM**——说话人向量余弦相似度。Seed-TTS 一系的 RL 就用这两个当 reward，纯 RLVR。
2. **学习型 reward model（对应你们的 preference RM）**
   - 图像：ImageReward、PickScore（Pick-a-Pic 人类投票训的）、HPSv2/v3（人类偏好）、Aesthetic 预测器、UnifiedReward（一个 RM 管理解+生成多任务）。
   - 视频：VideoAlign（视觉质量 VQ / 运动质量 MQ / 图文对齐 TA 三个头）、VisionReward。
   - 音频:UTMOS（预测人类 MOS 评分）、Audiobox-Aesthetics（Meta 的音频美学四维）。
3. **VLM/LLM as judge**：拿 Gemini/GPT/Qwen-VL 当裁判打分或成对比较。视频和 3D 尤其依赖（GPTEval3D）。
4. **人类偏好对（offline DPO 路线）**：Pick-a-Pic、HPD 这类"同 prompt 两张图选一张"的数据集 → Diffusion-DPO。不用 rollout，最便宜的入门 RL。

**Reward hacking 在视觉里比文本更猖獗、而且肉眼可见**：对着 HPS/美学分猛推几百步，模型会收敛到高饱和、油光、超对比度的"reward 脸"。标配解法和你熟的一样：多 reward 加权混合 + KL 正则 + 早停 + 留一组 unseen prompt 做 held-out 评估；2025 下半年的 Pref-GRPO 改用"组内成对胜率"代替绝对分，专治分数膨胀。

### 2.3 "需要 runtime 吗？"——一个诚实的回答

用你课程里的语言：文本 agentic RL 把训练对象从 P(y|x) 换成了 π(a_t|s_t, h_t)，最小训练单元是 trajectory。那视觉呢？

- **绝大多数视觉/音频生成 RL 目前退回到了 P(y|x)**：prompt 进、样本出、reward 打分，一步 bandit，没有外部环境状态转移，没有工具调用。所以**不需要 Gateway/Tools Factory 那套 runtime**；它们的"runtime"是 **reward serving**——verl-omni 把 HPSv3、GenRM-OCR、UnifiedReward 做成异步 HTTP scorer 服务，和 rollout 重叠执行（这个设计你一眼就能认出是谁的亲戚）。
- **正在长出"多步"的三个方向**：① agentic 生成——生成 → VLM 自检 → 重生成/编辑的闭环（NVIDIA Cosmos3-Super-T2I 自称 agentic 模型，Nano Banana Pro 把推理链嵌进出图）；② 统一模型先写 CoT 再出图（T2I-R1 等），CoT 部分就是标准 LLM RL；③ **世界模型**——见第 4.5 节，那里 s_t、a_t、h_t 全都回来了，而且世界模型自己扮演 env。
- 结论：**"环境交互"不是生成模态的训练需求，而是生成模态的终局产品形态**（会交互的视频=世界模型）。

---

## 3. 指标体系：把"好看/好听"拆成可验证的东西

你问的核心问题——"除了好看好听，有没有客观指标？"——答案是有，而且已经形成四层结构。按"多像 function calling 判卷"从高到低排：

### 3.1 第一层：规则可验证（对错分明，可直接当 RL reward）

| 模态 | 指标 | 判卷方式 | 文本类比 |
|---|---|---|---|
| 图像 | **GenEval** | 检测器数物体个数/颜色/相对位置，逐项 0/1 | 函数名+参数 exact match |
| 图像 | **OCR 准确率** | OCR 读出图中文字与要求比对 | JSON 字段值对不对 |
| 图像 | T2I-CompBench++、DPG-Bench | 组合关系/长 prompt 逐要素核对（部分用 VQA 判） | 多约束指令遵循率 |
| 视频 | **VBench / VBench-2.0** | 16~18 个维度各自的自动判定（主体一致性、运动平滑、物理常识…） | 分维度单测集 |
| 视频 | Physics-IQ、VideoPhy-2 | 物理事件是否按现实演化 | 逻辑一致性检查 |
| TTS | **WER/CER** | Whisper 转写 vs 输入文本 | 输出能否被 parser 解析 |
| TTS | **SIM** | 说话人 embedding 余弦相似度 | 风格约束遵循 |
| TTS | 时长控制误差 | 要求 3.2s 就量 3.2s（IndexTTS-2 的卖点） | 输出长度约束 |
| 音乐 | 歌词 WER、BPM/调性检测 | ASR + 信号分析 | 结构化约束 |
| 世界模型 | 动作跟随 F1、记忆保持时长 | 按键"向左"画面是否向左；离开再回来场景还在不在 | 工具调用执行正确率 |

### 3.2 第二层：学习型 reward model（对应 preference RM，可当 reward 也可当离线指标）

HPSv2/v3、ImageReward、PickScore、UnifiedReward（图）；VideoAlign、VisionReward（视频）；UTMOS、Audiobox-Aesthetics（音频）。注意：**用谁训练就别只用谁评估**，不然是在测过拟合——留一个没参与训练的 RM 和第一层规则指标做交叉验证，这是所有 RL for 视觉论文的标准姿势。

### 3.3 第三层：分布统计（历史遗留，正在退位）

FID/KID（生成图分布 vs 真图分布的距离）、FVD（视频版）、FAD（音频版）、CLIPScore/CLAP-score（模态-文本对齐的 embedding 余弦）。问题：和人类感受相关性一般，样本量要求大，对"指令遵循"不敏感。2026 年的论文还会报，但没人再拿它当北极星。3D 是例外：因为生成任务没有 ground truth 几何，主流协议（Hunyuan3D 系）就是**渲染成图后在 2D 空间算** FID_CLIP、CLIP-score、CMMD、LPIPS。

### 3.4 第四层：人类 Elo 竞技场（终审法院）

LMArena 文生图/视频榜、Artificial Analysis 的 Image/Video/Speech Arena、TTS-Arena。盲测成对比较 + Elo，和 Chatbot Arena 完全同构。厂商发布会引用的"第一名"基本都指这里。

### 3.5 数据集是**跟着指标反向构建**的

这直接回答你"如何根据指标构建数据集"：

- 想提 GenEval → RL 的 prompt 集就按 GenEval 的六个类别（计数/颜色/位置/共存…）程序化生成成千上万条组合 prompt，reward 用同款检测器判卷。**训练集 prompt 和评测集 prompt 同分布不同样本**——和你给 verifier 造题的思路一模一样。
- 想提文字渲染 → prompt 模板批量塞带引号的目标文本，reward = OCR exact match。
- 想提美学/偏好 → 同 prompt 采多样本，人类或 RM 排序 → 偏好对（Pick-a-Pic 的做法）→ DPO 或训 RM。
- 想提 TTS 稳健性 → 收集绕口令、数字串、中英混、长句等 hard case 文本当 prompt 池，reward = −WER。
- 预训练/SFT 数据侧则是另一套流水线（各模态章节细讲），但共同的第一定律是：**recaption 比架构重要**——用 VLM 给图/视频/音频重写密集描述，是 DALL-E 3 以来所有模型质量跃迁的最大单一因素。

---

## 4. 分模态详解

每个模态统一按「基座范式 → 数据 → SFT → RL → 指标 → infra」走一遍。

### 4.1 图像：整个体系的原型机（先学这个）

**基座范式（2026-08）**：MMDiT（文本流与图像流双流注意力）+ rectified flow，文本编码器从 CLIP/T5 演进为直接挂 VLM/LLM（Qwen-Image 挂 Qwen2.5-VL，FLUX.2 挂 Mistral VLM）。开源第一梯队：FLUX.2 家族（dev/Klein 开权重）、Qwen-Image 2.x（Apache 2.0，中英文字渲染最强）、Z-Image（6B，效率路线）、GLM-Image、HiDream O1（MIT）；统一 AR 路线的 HunyuanImage 3.0（80B MoE）。闭源天花板：Nano Banana Pro/2（Gemini 3 Image）、Seedream 4.5、GPT Image。有意思的信号：Artificial Analysis 开源榜首在 2026 年中被 NVIDIA Cosmos3-Super-Text2Image 拿走——一个"agentic"图像模型，出图前先推理规划，印证了 2.3 节说的多步化趋势。**编辑模型已是标配**而非附赠（Kontext、Qwen-Image-Edit、nano-banana 系），编辑能力主要靠 SFT 数据管线而非新架构。

**数据**：
- 预训练：十亿级网爬图文对（DataComp、COYO 这类；LAION 因合规问题退役为 Re-LAION）→ 去重（SSCD embedding）→ 美学过滤（aesthetic predictor 卡阈值）→ NSFW/水印过滤 → **VLM 重写密集 caption**（原生 alt-text 又短又错，recaption 后模型的指令遵循直接换代）。
- SFT：几千~几万张人工精选（质量对齐）；LoRA 则 20~200 张同风格图即可。
- 偏好：Pick-a-Pic / HPD（同 prompt 二选一的人类投票）。

**SFT 实操**：全参微调走 diffusers + accelerate/FSDP；LoRA 社区标准工具是 ai-toolkit（FLUX 系最顺）、kohya sd-scripts、DiffSynth-Studio（阿里系模型支持最好）、SimpleTuner。你的 5090（32GB）单卡训 FLUX/Qwen-Image LoRA 毫无压力，几百步、几小时出活。

**RL 演进史（20 分钟版）**：
1. 2023：DDPO/DPOK 把去噪过程当 MDP 上 PPO——思想正确，但小规模 prompt 集就崩，没法实用。
2. 2023 末：Diffusion-DPO——跳过 rollout，直接在 Pick-a-Pic 偏好对上做 DPO。便宜、稳，SDXL-DPO 是第一个大规模落地案例。**至今仍是"最低成本入门 RL"的推荐路径**。
3. 2025：**Flow-GRPO / DanceGRPO** 双子星，把 GRPO 带进 flow matching（ODE→SDE 那一手）。DanceGRPO（字节 Seed + HKU，基于 FastVideo 实现）在 SD/FLUX/HunyuanVideo/SkyReels-I2V 四个底座、五种 reward 上统一跑通，公认的"视觉版 GRPO 参考实现"。
4. 2025 下半年~2026：工程化与防 hacking——MixGRPO（只在部分步上 SDE，省算力）、Pref-GRPO（成对胜率代替绝对分）、SRPO（腾讯，直接用可微 reward 沿轨迹回传，几十分钟对齐 FLUX）、DiNa-LRM（在 latent 上直接打分，跳过 VAE decode，verl-omni 2026-07 引入）。
5. 大厂配方（Seedream/Qwen-Image 技术报告可见）：SFT（质量对齐）→ RLHF（多 RM 混合 + GRPO 变体）→ 蒸馏加速，三连已成标准出厂流程——和 LLM 的 pipeline 完全同构。

**指标**：第一层 GenEval / OCR / T2I-CompBench / DPG-Bench；第二层 HPS 系 / ImageReward / PickScore；第四层 LMArena、AA Arena。论文标准配置：训练 reward 曲线 + 两个 held-out 规则基准 + 一个没参与训练的 RM + 少量人评。

**Infra**：训练 diffusers 生态；推理端 ComfyUI（事实标准的节点式工作流）、xDiT（DiT 的序列并行推理，Ulysses/Ring，对应你熟的 SP）、Nunchaku（SVDQuant 4-bit）、TeaCache（跨步复用，"扩散版 KV cache 精神"）。RL 框架见第 5 节。

### 4.2 视频：图像的所有问题 × 时间轴，rollout 贵到改变系统设计

**基座范式**：3D causal VAE 把视频压成时空 latent（8×8×4 一类的压缩率），DiT 在上面做 flow matching。开源主力：**Wan 2.2**（Apache 2.0；A14B 是"MoE"但注意——它的两个专家按**去噪时间步/信噪比**路由（高噪专家管布局、低噪专家管细节），不是文本 MoE 的 per-token 路由，所以 GSPO 要解决的 routing 抖动问题在这里根本不存在，同名不同物）+ TI2V-5B 小杯（消费卡友好）；LTX-2.5（唯一开源的音视频**单次联合生成**，微调生态最全）；HunyuanVideo 1.5（8.3B）；MAGI-1（自回归分块路线）。Wan 3.0 已上 API 未放权重。闭源：Veo 3.x、Sora 2、Kling、Seedance 1.5——2025 年后全部带原生音频。**趋势主线：双向注意力的"离线出片"模型，正在被蒸馏成因果、流式、可交互的实时模型（Self-Forcing / CausVid / DMD 蒸馏），这条路的尽头就是世界模型**——视频和世界模型是同一棵技术树。

**数据流水线**（比图像多三道工序）：切镜头（PySceneDetect）→ 运动量过滤（光流分数，剔除静止和抖动）→ 美学过滤 → **VLM 逐段密集 recaption**（描述主体、动作、运镜、光线）。开源可用：Panda-70M、Koala-36M、OpenVid-1M。物理/交互类数据大量来自游戏引擎合成（下一节世界模型会看到极致版）。

**SFT**：LoRA 学角色/画风/特定运动/运镜，工具是 musubi-tuner、diffusion-pipe、DiffSynth-Studio。Wan2.2-TI2V-5B 的 LoRA 在你的 5090 上可跑（慢，但能跑）；14B 级别建议租 H100。

**RL**：算法与图像同款（DanceGRPO 论文本身就在 HunyuanVideo 上验证；verl-omni 已支持 Wan2.2，2026-08 刚加了 LTX-2.3 的音视频联合 FlowGRPO）。真正的不同在**系统**：一条 720p/5s 视频的 rollout = 50 步 × 巨长序列的 DiT forward + 3D VAE decode + 视频 RM 推理，单条几十秒到分钟级——rollout 占训练时间的比例远超文本 RL。于是催生了三类系统优化，你会觉得非常眼熟：① 降步数 rollout（训练时 8~10 步）；② latent 直接打分省 VAE decode（DiNa-LRM）；③ **训推分离/异步**——DigenRL（2026-06）专门做 diffusion RL 的 disaggregation：生成轴流水线、时间步并行、trainer 弹性支援 rollout、一步受限异步。是的，这就是你 Syncopate 关心的 sync-colocate vs async 问题在扩散世界的镜像，而且这个方向 2026 年才刚开垦。
Reward 主力是 VideoAlign 三个头（视觉质量/运动质量/图文对齐）分开加权——DanceGRPO 发现文本-视频对齐头不稳，只用 VQ+MQ 也能拿到 56%/181% 的相对提升。**物理合理性 reward 是公认未解难题**（Physics-IQ 能评测，但没法当稠密 reward 用）。

**指标**：VBench / VBench-2.0（16~18 维雷达图，行业标准）、Physics-IQ、VideoPhy-2；FVD 只剩仪式意义；终审是视频竞技场 Elo。

**Infra**：FastVideo（滑动块注意力 STA / 视频稀疏注意力 VSA + 蒸馏，DanceGRPO 的宿主框架）；推理并行靠 xDiT 系序列并行——视频 token 数动辄几十万，**序列并行在这里的地位 ≈ TP 在 LLM 里的地位**。

### 4.3 音频：对你迁移成本最低的模态（AR 路线 = 换了词表的 LLM）

音频要分三个子领域看，范式差异比图像/视频之间还大。

**(a) TTS / 语音**——两条路线：
- **AR codec-LM（主流，和你最亲）**：神经音频编解码器（EnCodec/SNAC/双码本一类）把波形离散成 token → 一个正经的 LLM 在音频 token 上做 next-token → 轻量解码器（常是 flow matching 头）还原波形。2026-08 开源 Elo 榜首 Step Audio EditX（Apache 2.0）就是教科书结构：双码本 tokenizer + 3B 音频 LLM + FM 解码器——它连"编辑"都在 token 空间做（生成→改情绪→再改，迭代式 refine）。CosyVoice 2/3、Fish、MOSS-TTS（Seed-TTS-eval 榜霸）、VibeVoice（长篇多说话人）同族。
- **NAR flow matching**：F5-TTS、MaskGCT，一次并行出整段，快，但可控性和长文弱一些。
- **对你的意义**：AR-TTS 的训练就是 LLM 训练（换 tokenizer + 音频数据），**vLLM/SGLang 能直接 serve，verl 能直接训**。RL 是纯 RLVR：reward = −WER（Whisper 判卷）+ SIM（音色相似）+ 可选 UTMOS，Seed-TTS 系就是这么把幻觉率打下来的。你 sglang-omni 做的 Qwen3-Omni（Thinker-Talker：思考者出语义、说话者出音频 token）正是这条路线的 omni 化终点，全双工对话（Moshi 一系)是它的交互形态。
- 数据：Emilia 式野外流水线是行业底座——播客/视频音频 → VAD 切段 → ASR 转写 → 说话人分离 → 质量打分过滤 → 得到十万小时级 (文本, 语音) 对。**这条流水线本身就是核心资产**，"数据构建即产品"在音频最明显。
- 指标：WER/CER、SIM、UTMOS、Seed-TTS-eval 基准包、TTS-Arena Elo。全模态里客观指标最硬的一家。

**(b) 音乐**：ACE-Step v1.5（2026-02）是当前开源答案，架构值得你看一眼——**LM 当规划器**（把用户一句话扩写成完整歌曲蓝图：结构/歌词/风格标签，走 CoT）→ **DiT 当渲染器**（在 DCAE 压缩的 mel latent 上 flow matching 出整首歌）。A100 上 2 秒一首完整歌、3090 十秒、<4GB 显存、几首歌就能 LoRA 个人曲风；对齐用的是"intrinsic RL"（靠模型内部信号，不依赖外部 RM——规避了音乐审美 RM 极难训的问题）。AR 路线代表 YuE 7B（歌词对齐好，慢）。指标：FAD、CLAP-score、歌词 WER、Audiobox-Aesthetics，但**音乐是主观权重最高的模态**，盲测听感仍是硬通货。
**(c) 音效/Foley**：视频配音效（V2A）2026 年热区，Hunyuan 系有开源模型,跟视频生成正在合流（LTX-2 直接联合生成）。

### 4.4 3D：你有科研基础，这里只补"训练/评估视角"的增量

你在 TRELLIS 系（SLAT + rectified flow）上已有一线经验，所以跳过范式科普，只讲三件你的 paper 视角之外的事：

1. **产业格局（2026-08）**：两段式（几何 latent 流模型 → 纹理多视角扩散烘焙）仍是主干——TRELLIS.2（单图出带 PBR 材质资产）、Hunyuan3D 2.1（开源含 PBR）/ 3.x~3.5（API，<60s、8K PBR）、SAM 3D（Meta）、Tripo/Rodin/Meshy 商业系。趋势：part-level 生成（可拆件）、PBR 材质标配、AR mesh 路线（LLaMA-Mesh 把面片当 token——又一个"变成 LLM 问题"的例子）。数据仍被 Objaverse(-XL) + 渲染 + VLM caption + 质量过滤统治，**高质量 3D 数据稀缺是整个赛道的第一约束**。
2. **评估协议**：生成任务没有 GT 几何，所以主流协议（Hunyuan3D 报告可抄）是渲染到 2D 后算 FID_CLIP / CLIP-score / CMMD / LPIPS，加 GPTEval3D（VLM 裁判）和用户研究；重建类才用 Chamfer/F-score。2026 年刚出的 Hy3D-Bench 想做 3D 界的 VBench，可以关注。
3. **RL 在 3D 基本是无人区**：DreamDPO/DreamReward 属早期零星工作，没有"3D 版 DanceGRPO"。原因：单次 rollout 贵、reward 定义难（几何正确性怎么打分？）、偏好数据几乎没有。**这是你的研究方向和这份调查的交点——把 verifiable reward（比如渲染后的多视角一致性检测、部件完整性检查、物理稳定性仿真）接到 SLAT 流模型的 SDE 采样上，就是一篇没人写过的 paper。**

### 4.5 世界模型：runtime 问题的真正答案

**定义收敛（2026 版）**：世界模型 = **动作条件化的流式视频模型** p(下一帧 | 历史帧, 用户动作)，实时可交互。它和视频生成不是两个领域，是同一条技术树的两个阶段——证据：LingBot-World 直接拿 **Wan2.2 当底座**改造。

**训练配方**（Matrix-Game 2.0 把全套开源了，最值得学习的样本）：
1. **数据**：游戏引擎程序化量产——Unreal Engine + GTA5 里跑 PPO 导航 agent 自动游走，毫秒级对齐地录下 (画面, 键鼠动作)，产出 1200 小时帧级标注数据。Genie 系的更狠：从**无标注**网络视频里用 latent action model 自监督地反推出"隐动作"，绕过标注。机器人侧则用真机遥操作数据（Open X-Embodiment 一类）。
2. **预训练**：大规模视频（可无动作标注）学世界的样子。
3. **动作对齐 SFT**：action injection 模块把帧级键鼠/控制信号注入 DiT，在标注数据上微调出"可被操纵"。
4. **实时化蒸馏**：双向注意力 → 块因果（block-causal）+ few-step 蒸馏（DMD 系），50 步离线模型变 1~4 步流式模型，25 FPS 实时。**蒸馏在这里不是部署优化，是产品成立的前提**。

**2026-08 版图**：开源——Matrix-Game 3.0（2026-03，长程记忆）、LingBot-World（2026-01，分钟级场景记忆）、minWM（2026-05，全栈开源框架，**最适合上手复现**）、HY-World 1.5；闭源——Genie 3（2026-01 以 Project Genie 产品化开放给 Ultra 订阅，720p/24fps、几分钟一致性、可提示世界事件；Waymo 2026-02 基于它做了自动驾驶仿真专用版）、NVIDIA Cosmos 3（2026-06，物理 AI 全模态世界模型）。

**和 RL 的关系（关键认知）**：不是"用 RL 训世界模型"，而是**"世界模型是给 agent 做 RL 的环境"**——Dreamer 4 让 agent 完全在世界模型的想象里训练出 Minecraft 挖钻石；DeepMind 的 SIMA 2 agent 在 Genie 3 生成的世界里学习。对照你的四层架构：Channel→Gateway→Agent Runtime→Tools Factory 里的 "env"，在具身智能里就由世界模型充当——**世界模型 = 可微分、可无限生成的 Tools Factory/环境仿真器**。这是"多模态"与"agentic RL"两条线未来汇合的地方，也是你两边背景最值钱的交叉点。

**指标**：动作可控性/跟随 F1（按 W 是否前进）、时序一致性、记忆保持时长（转头回来东西还在吗）、实时性 FPS/延迟，以及最硬的一条——**在里面训出来的 agent 拿到真环境里的成功率**（sim2real）。Matrix-Game 2.0 的公开协议：图像质量 0.61 / 时序一致性 0.94 / 动作可控性最高 0.95 @ 25FPS，可当 baseline 参考系。

---

## 5. Infra 对照：你的每个技能点落在哪

### 5.1 逐层映射

| 层 | 文本栈（你的） | 多模态生成栈（2026-08） | 备注 |
|---|---|---|---|
| RL 框架 | verl / OpenRLHF / AReaL | **verl-omni**（verl 官方社区出品）· DanceGRPO（基于 FastVideo）· flow_grpo 官方 repo | verl-omni 从 verl 的多模态分支独立成库，2026-05 由 vLLM 官方博客宣布 |
| rollout 引擎 | SGLang / vLLM | **vLLM-Omni**（verl-omni 的 rollout 后端）· sglang-omni | 你同时踩着 verl 和 sglang-omni 两条线，罕见的双边视角 |
| 训练引擎 | Megatron / FSDP2 | diffusers + accelerate/FSDP/DeepSpeed；FastVideo（视频专用）；大厂内部是 Megatron 改造版 DiT 并行 | DiT 并行以 DP+SP 为主，PP 少见（层数浅、激活大） |
| 并行策略 | TP/PP/DP/SP/EP | **SP（Ulysses/Ring）是一等公民**（视频 token 几十万），DP 次之 | 视频里 SP 的地位 ≈ 文本里 TP |
| RL 算法 | GRPO / GSPO / DAPO / PPO | FlowGRPO / DanceGRPO / MixGRPO / Pref-GRPO / Diffusion-DPO / SRPO | 核心都是"组内标准化 advantage"，差别在采样器改造 |
| reward 侧 | RM 服务 + rule verifier | HPSv3 / GenRM-OCR / UnifiedReward 做成异步 HTTP scorer（verl-omni 内建） | reward 推理本身就是一批 GPU 负载，必须服务化+异步化 |
| 训推一致 | TIS / IS 校正 | SDE(训) vs ODE(推) 的分布错位；降步 rollout vs 满步部署的错位 | 同一个幽灵，换了件衣服 |
| MoE | GSPO for MoE、EP | Wan2.2 的 MoE 按时间步路由（非 per-token），无路由抖动问题 | 名字相同，问题不同，别把 GSPO 直觉硬套 |
| 推理加速 | 量化/投机采样/prefix cache | few-step 蒸馏（DMD/LCM）· TeaCache 跨步复用 · Nunchaku 4-bit · 稀疏注意力（STA/VSA） | 蒸馏地位极高：既是部署优化，又是世界模型实时化的前提 |
| 异步/分离架构 | 你研究的 Syncopate 问题 | DigenRL（2026-06，扩散 RL 的 disaggregation+弹性 rollout） | 你的问题意识在扩散世界刚被提出，几乎空白 |

### 5.2 verl-omni 速览（对你最重要的一个 repo）

- 定位：多模态生成 RL 训练框架，覆盖三类模型——① 扩散生成（Qwen-Image、Wan2.2）；② 统一理解+生成（BAGEL、HunyuanImage-3.0）；③ 全模态（Qwen3-Omni）。
- 关键设计：vLLM-Omni 做 rollout（路由/批处理/embed cache）；多 reward 异步 serving 与 rollout 重叠；FlowGRPO + DiNa-LRM（latent 直接打分省 VAE decode，2026-07）；LTX-2.3 音视频联合 RL（2026-08）；N 卡与昇腾双后端。
- 对你的意义：**config 结构、Ray 编排、rollout-train 权重同步这些 verl 心智模型全部复用**，增量学习只剩"SDE 采样器 + reward 服务 + DiT 的 FSDP 包法"。这是你从文本跨到多模态的最短路径，没有之一。

---

## 6. 三条"跑起来"路线（按迁移平滑度排序，都给到可执行颗粒度）

> 原则：先跑通闭环（数据→训练→指标动了），再谈规模。每条路线的"评估"都含一个第一层客观指标——没有可判卷的指标就谈不上"训练成功"。

### 路线 A：AR-TTS 的 RLVR（最平滑，纯复用你的 LLM 技能，5090 单卡可起步）

- **任务**：拿一个开源 AR codec-LM TTS，用 GRPO 优化 reward = −WER + λ·SIM，目标是把 hard-case 文本（数字串/中英混/绕口令/长句）的错读率打下来。
- **模型**：CosyVoice 系或任一 0.5B~3B 级 AR-TTS（小、结构标准）；进阶可直接瞄准 Qwen3-Omni 的 talker 侧（与你 sglang-omni 的工作互喂）。
- **数据**：自建 prompt 池 2~5k 条 hard-case 文本（LLM 批量造）+ 3~10 条参考音色。不需要任何音频标注——reward 全自动。
- **评估**：Seed-TTS-eval 协议（WER + SIM），训练前后对比；留 500 条 unseen 文本做 held-out。
- **infra**：先用官方 repo 做 SFT/声音克隆热身一天；RL 阶段 policy 是个标准 causal LM，verl 的思路（甚至代码骨架）可以直接改造——rollout 出音频 token → 解码成波形 → Whisper/说话人模型算 reward。
- **为什么第一**：零新范式（没有 SDE、没有 DiT），却完整走一遍"多模态 reward 工程"，一两周出结果，且和你 sglang-omni 的 Qwen3-Omni perf track 直接协同。

### 路线 B：图像 flow 模型的 SFT → RL（最主流，正式入门 DiT 世界）

- **B1 热身（1~2 天，5090 单卡）**：ai-toolkit 或 DiffSynth-Studio 给 FLUX.2-dev / Qwen-Image 训一个风格 LoRA（30~100 张图，rank 32）；跑通 GenEval 和 HPSv2 的评测脚本，建立"改动→指标"的手感。产出：你的第一条视觉训练闭环。
- **B2 正餐（1~2 周，租 2~8×H100）**：**verl-omni 跑 Qwen-Image + FlowGRPO**，reward 选 GenEval 检测器或 OCR（二选一，别贪多），prompt 池按 3.5 节方法程序化生成 5~20k 条。观察三件事：reward 曲线、held-out GenEval、以及故意留一个美学 RM 不参与训练、专门监控 reward hacking（画面是否开始油腻）。
- **评估**：GenEval（训练目标）+ DPG-Bench（泛化）+ HPSv2（画质没退化）三角验证。
- **为什么第二**：这是 2026 年多模态 post-training 的"标准工种"，简历/面试的通用语言；且 verl-omni 让你的 verl 经验直接变现——你甚至有能力给它提 PR（它比 sglang-omni 更年轻，good first issue 更多）。

### 路线 C：视频 → 世界模型（进阶，通向你的长期交叉点）

- **C1（5090 可跑）**：Wan2.2-TI2V-5B 用 musubi-tuner 训运动/风格 LoRA，跑 VBench 子集评测，体感"时间轴"带来的所有新麻烦（数据切片、显存、评测慢）。
- **C2（租多卡）**：verl-omni 的 Wan2.2 FlowGRPO 配方，reward 用 VideoAlign 的 VQ+MQ 两个头（照 DanceGRPO 的结论跳过不稳的 TA 头）；顺便你就站在了 DigenRL 那类"扩散 RL 训推分离"问题的第一现场——Syncopate 的问题意识可以在这里长出第二篇东西。
- **C3（研究向）**：跑通 minWM 或 Matrix-Game 的推理与微调，理解 action injection + 因果蒸馏两个模块；再往后，"3D（你的科研）× 世界模型 × agentic RL（你的课程）"三线交汇——用世界模型当环境训 agent，是 2027 年最确定的方向之一。

### 显卡预算速查

| 任务 | 硬件 | 时长感 |
|---|---|---|
| 图像/音乐 LoRA SFT、AR-TTS SFT | 5090 单卡 | 小时级 |
| TTS GRPO（0.5~3B） | 5090 起步，2×H100 舒适 | 天级 |
| 图像 FlowGRPO（12~20B 底座） | 2~8×H100 | 天~周级 |
| 视频 5B LoRA | 5090（勉强）/ 1×H100 | 天级 |
| 视频 14B FlowGRPO | 8×H100 起 | 周级 |
| 世界模型复现 | 8×H100 起（推理 1~2 卡可玩） | 周~月级 |

---

## 7. 你可能"不知道自己不知道"的七件事

1. **SFT 不换损失函数**。视觉从预训练到 SFT 是同一个 flow matching loss，变的只有数据。第一次听说时几乎所有文本背景的人都会愣一下。
2. **RL 的可行性系于一个采样器 trick**（ODE→SDE）。不理解这一步，所有 diffusion RL 论文都像黑话；理解了，就只剩 GRPO。
3. **Reward hacking 是肉眼可见的**：高饱和、油光、塑料感 = 视觉版"谄媚"。所以视觉 RL 的标配是多 reward 混合 + KL + 早停 + unseen RM 监控，单 reward 猛推必翻车。
4. **rollout 成本结构倒挂**：文本 RL 里训练步贵、rollout 相对便宜；扩散 RL 里 rollout（G 条 × 50 步全量 forward + VAE + RM）才是大头。所以降步 rollout、latent 打分、异步分离这些系统活的杠杆比文本更大——你做 infra 出身，这是你相对纯算法背景者的比较优势。
5. **蒸馏的战略地位不同**：文本里蒸馏是"省钱"；视觉里 few-step 蒸馏是"实时交互能不能成立"——没有 DMD 类蒸馏就没有世界模型产品。
6. **"MoE"在 Wan2.2 里是假朋友**：按时间步路由的双专家，与 per-token 路由的 LLM MoE 机理不同，GSPO 的问题意识不适用，面试时说错会露怯。
7. **3D 的 RL 是空白**，音乐的 reward 是难题（ACE-Step 1.5 干脆用 intrinsic RL 绕开外部 RM）——这两处是"已知没人做好"的地图边缘，也就是研究机会所在。

---

## 8. 精选清单（repo 优先，按"先跑再读"排序）

**框架/代码（先跑）**
- verl-project/verl-omni —— 多模态生成 RL，本文核心推荐（vLLM 官方博客 2026-05 有发布文）
- XueZeyue/DanceGRPO —— 视觉 GRPO 参考实现（基于 FastVideo）
- yifan123/flow_grpo —— Flow-GRPO 官方实现，代码量小，最适合精读理解 ODE→SDE
- hao-ai-lab/FastVideo —— 视频训练/蒸馏基建（STA/VSA 稀疏注意力）
- ostris/ai-toolkit、modelscope/DiffSynth-Studio、kohya-ss/musubi-tuner —— LoRA SFT 三件套（图像/阿里系/视频）
- ace-step/ACE-Step-1.5 —— 音乐基座 + LoRA，消费卡即玩
- SkyworkAI/Matrix-Game、minWM —— 世界模型全栈开源样本
- xdit-project/xDiT、ComfyUI —— 推理侧两大件

**评测（判卷器就是 reward 的原型）**
- GenEval、T2I-CompBench++、DPG-Bench（图像规则判卷）；HPSv2/v3、ImageReward、PickScore、UnifiedReward（图像 RM）
- VBench / VBench-2.0、Physics-IQ、VideoPhy-2（视频）；VideoAlign（视频 RM）
- Seed-TTS-eval、UTMOS、TTS-Arena（语音）；FAD、CLAP、Audiobox-Aesthetics（音乐/通用音频）
- Hy3D-Bench、GPTEval3D（3D）；LMArena / Artificial Analysis 各竞技场（终审）

**读物（按性价比）**
1. Flow-GRPO 论文 —— 一篇看懂视觉 RL 的全部要点（SDE 改写、降步 rollout、KL 防 hacking）
2. DanceGRPO 论文 —— 多底座多 reward 的工程化视角
3. Wan 2.x 技术报告 —— 视频基座的数据/训练/评估全配方，工业报告里最坦诚的一档
4. Matrix-Game 2.0 论文 —— 世界模型从数据引擎到实时蒸馏的完整开源配方
5. ACE-Step v1.5 报告 —— LM 规划器 + DiT 渲染器的混合范式与 intrinsic RL
6. Seedream / Qwen-Image 技术报告 —— 大厂"SFT→RLHF→蒸馏"三连的标准出厂流程
7. BLIP3o-NEXT —— "RL 该打在哪个模块上"这一个问题的答案
8. Diffusion-DPO 论文 —— 最低成本 RL 路线的原点

---

## 附：一句话回答你开头的每个问题

- **其他模态的基础模型范式和语言一样吗？** 不一样但在靠近：生成主流是 DiT+flow matching（连续 latent），理解主流是 AR；统一模型和"万物 token 化"（AR-TTS、AR-mesh、HunyuanImage 3.0）在把两边缝起来。
- **SFT/RL 的目标是什么？** SFT = 用精选数据换分布（画质档次/风格/指令编辑/声音克隆）；RL = 对齐 reward（美学、组合正确性、文字渲染、WER、运动质量），风格对齐主要在 SFT/LoRA 层，RL 管的是"更对、更好看、更稳"。
- **有没有客观指标？** 有，且成体系：GenEval/OCR（图）、VBench/Physics-IQ（视频）、WER/SIM（语音）、动作跟随 F1（世界模型）——性质上等价于你的"JSON 合法/函数调用正确"。
- **如何根据指标构建数据集？** 指标反向生成 prompt 池 + 判卷器当 reward（RLVR 思路）；偏好数据 = 同 prompt 多采样 + 排序；预训练数据的胜负手是 VLM recaption。
- **有没有 SFT/RL/runtime？** SFT/RL 全都有且与文本同构；runtime 只有世界模型需要——因为它自己就是 runtime。
- **框架怎么选？** RL 认准 verl-omni（你的 verl 技能直接迁移），SFT 认准各模态 LoRA 三件套，推理认准 vLLM-Omni/sglang-omni/ComfyUI/xDiT。

*（成文于 2026-08-14；这个领域按月折旧，半年后请重新核对模型名与版本号，但第 1~3 节的框架性内容会更长寿。）*
