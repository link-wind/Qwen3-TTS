**Plan: 迁移 SFT+GRPO 情感控制链路（v1.1）**

基于当前仓库已有的 Base-SFT 与推理主链，先统一“控制 token 语义与拼接位置”，再扩展数据准备（VAD 强度离散化 + emphasis 标签），随后实现“仅 LLM 主干可训练”的 SFT，最后新增最小 GRPO 训练闭环与三任务奖励。这样可最小化侵入并保证训练/推理接口一致，同时避免全参数更新导致的语速漂移问题。

### Steps

1. 定义统一控制协议：在 `qwen_tts/core/models/modeling_qwen3_tts.py` 中新增 `ControlTokenPreprocessor` 类（或在 preprocess/forward 阶段统一处理），对齐 `emotion_token`、`intensity_token`、`emphasis_token` 的 prepend 位置与 prefill 规则。  
   prepend 顺序固定为 `[emotion][intensity]` + text（严格参考 EMORL-TTS）；emphasis 支持 `*强调词*` 形式。

2. 扩展数据模式与预处理：在 `finetuning/README.md` 和 `finetuning/prepare_data.py` 中增加 ESD/Expresso/EmotionTalk 等数据集的字段映射、VAD 距离到 neutral 质心的欧氏距离计算、**按 emotion 类别特定阈值** 离散化为 weak/medium/strong 的逻辑，以及 emphasis 标签处理。

3. 改造训练样本拼装：在 `finetuning/dataset.py` 的 `__getitem__` 与 `collate_fn` 中注入控制 token：  
   prepend `[emotion][intensity]` 到 input_ids，emphasis 以可监督形式进入 input_ids，同时在 labels 中 mask 掉 control tokens（仅对 speech codec tokens 计算 CE loss）。保留 ref_audio 处理逻辑，保证 3 秒 cloning 能力不受影响。

4. 落实“仅 LLM 主干可训练”：在 `finetuning/sft_12hz.py` 中重构参数分组 / 使用 LoRA：  
   仅更新 LLM 主干参数（重点是 speech token prediction heads 和相关 transformer layers），推荐 LoRA（rank 32–64，target_modules 限制在 q_proj、v_proj、o_proj 等），显式冻结 speech tokenizer / codec embedding 相关模块，避免全参数更新导致语速漂移。

5. 同步推理入口参数：在 `qwen_tts/inference/qwen3_tts_model.py` 与 `qwen_tts/cli/demo.py` 中暴露 `emotion`、`intensity`、`emphasis` 控制入参（或统一 `control_tags` 参数），确保推理时的文本拼接与训练阶段完全同构，支持直接输入带 `[happy][strong] *特别*` 的文本。

6. 新增最小 GRPO 训练闭环：在 `finetuning/` 下规划 `grpo_runner.py`、`grpo_rollout.py`、`reward_fns.py`、`grpo_dataset.py`，复用 `qwen_tts/inference/qwen3_tts_model.py` 做采样，并实现三任务奖励（严格参考 EMORL-TTS）：  
   SER accuracy、emotion-intensity fidelity、emphasis controllability，加权后用于 Group Relative Policy Optimization（此步在 SFT 可控接口验证通过后再实施）。

### Further Considerations

1. 强度离散阈值策略：优先采用 **Option B（按 emotion 类别阈值）**（最推荐，符合 EMORL 原方案），happy/sad/angry 等情感的 arousal 分布差异较大；Option A（全局阈值）用于快速实验；Option C（分数据集校准）作为后续优化。

2. emphasis 表示方式：优先采用 **Option A（span-token）**：`*强调词*`（简单且 Expresso 已验证有效）；Option B（phrase-level `[emphasis]词[/emphasis]`）作为备选；Option C（强弱二值控制）暂不优先。

3. 优先级与风险控制：  
   - **强烈建议先优先交付 “SFT 可控接口”**（Steps 1–5），验证 prepend tag 能稳定控制情感强度、强调词突出度，且 3 秒 cloning 能力未退化后，再进入 GRPO 阶段。  
   - 学习率建议使用 **2e-5 ~ 5e-5**（低于 EMORL 原 0.0002），配合 LoRA 可有效缓解语速漂移。  
   - SFT 后需做 ablation 测试（不同 [emotion][intensity] 组合的 prosody 区分度、CER、MOS）。  
   - 新增 `control_preprocessor.py` 统一管理 prepend 逻辑，减少代码重复。  
   - 所有修改必须保证原有 Base/CustomVoice 推理路径不受影响。

### Milestones（建议两阶段交付）

#### Phase A：SFT 可控接口（先完成）
1. **M1 - 控制 token 管线打通**：训练/推理均支持 `[emotion][intensity] + text`，并兼容 `*强调词*`。  
2. **M2 - 数据预处理可复现**：给定同一数据与阈值配置，`prepare_data.py` 输出稳定一致的 intensity/emphasis 标注。  
3. **M3 - 仅 LLM 可训练**：冻结 codec/tokenizer，训练日志可见可训练参数量显著下降。  
4. **M4 - 基线评估通过**：情感可控、强度层次可区分、克隆能力不退化。

#### Phase B：GRPO 精细化控制
1. **M5 - rollout+reward 闭环**：可批量采样并返回三奖励分。  
2. **M6 - GRPO 稳定训练**：reward 提升且无明显音质崩坏/语速漂移。  
3. **M7 - 对照实验**：SFT-only vs SFT+GRPO 在 SER、强度一致性、强调可控性上显著提升。

### 最小接口草案（用于代码落地）

1. `prepare_data.py` 新增字段（写入 jsonl）：
   - `emotion`: `happy|sad|angry|neutral|...`
   - `intensity`: `weak|medium|strong`
   - `emphasis_spans`: `[[start, end], ...]`（可为空）
   - `vad_score` / `dist_to_neutral`（调试可选）

2. `dataset.py` 样本输入约定：
   - 训练文本统一转为：`[emotion][intensity] 原始文本(含*强调词*)`
   - `labels` 对 control token 位置置 `-100`
   - 保持现有 `ref_audio` 分支不变

3. 推理 API 建议：
   - `infer(..., emotion=None, intensity=None, emphasis=None, control_tags=None)`
   - 若传 `control_tags`，优先级高于单独参数

### Definition of Done（DoD）

1. 训练集与验证集均能生成 3 档强度标签，且各 emotion 档位分布非塌缩。  
2. SFT 后，固定文本在不同 intensity 下可感知区分（主观 A/B 与客观 prosody 指标一致）。  
3. emphasis 词在能量/时长上有稳定提升，不引入明显错词。  
4. 3 秒 cloning 主观相似度与基线相比无显著下降。  
5. GRPO 后三任务奖励的加权总分较 SFT-only 显著提升。

### 逐文件改动清单（Implementation Checklist）

#### 1) 控制 token 与推理同构
- `qwen_tts/core/models/modeling_qwen3_tts.py`
   - 新增/接入 `ControlTokenPreprocessor`（或等效函数）
   - 统一 prepend 顺序：`[emotion][intensity] + text`
   - 接入 emphasis span 渲染（`*词*` 或 token 级标记）
   - 保持原 Base/CustomVoice 路径默认行为不变（无参时回退）

- `qwen_tts/inference/qwen3_tts_model.py`
   - `infer` / `inference_base` 增加 `emotion/intensity/emphasis/control_tags`
   - 推理前文本标准化与训练一致
   - 新增参数冲突优先级：`control_tags > emotion/intensity`

- `qwen_tts/cli/demo.py`
   - 增加 CLI 参数透传（可选）
   - 示例文案增加 `[happy][strong] *特别*`

#### 2) 数据预处理与标签构建
- `finetuning/README.md`
   - 新增数据 schema 说明（emotion/intensity/emphasis_spans）
   - 增加 ESD+Expresso 预处理命令示例

- `finetuning/prepare_data.py`
   - 增加数据源字段映射（ESD/Expresso/EmotionTalk）
   - 新增 VAD 推断入口（可插拔 estimator）
   - 计算 `dist_to_neutral` 并按 emotion 阈值离散化到 weak/medium/strong
   - 输出调试字段（`vad_score`, `dist_to_neutral`）

#### 3) 训练数据拼装与损失掩码
- `finetuning/dataset.py`
   - 在 `__getitem__` 组装带控制 token 的文本
   - 在 `collate_fn` 对 control token 位置写入 `-100`
   - 保留现有 ref_audio 逻辑，确保 cloning 分支不变

- `finetuning/sft_12hz.py`
   - 参数分组：仅 LLM 主干可训练
   - 显式冻结 codec/tokenizer 相关模块
   - （可选）接入 LoRA（rank、target_modules、dropout 可配置）
   - 日志打印可训练参数占比

#### 4) GRPO 最小闭环（新增文件）
- `finetuning/grpo_dataset.py`
   - prompt + 控制标签 + 参考标签的 batch 构建

- `finetuning/grpo_rollout.py`
   - 封装批量采样（复用推理接口）

- `finetuning/reward_fns.py`
   - `reward_ser_accuracy()`
   - `reward_emotion_intensity_fidelity()`
   - `reward_emphasis_controllability()`
   - `compose_reward(weights=...)`

- `finetuning/grpo_runner.py`
   - group 采样、相对优势、KL 约束、参数更新
   - 周期性评估与 checkpoint 保存

### 建议提交顺序（Commit Plan）

1. **commit 1: docs(schema+pipeline)**
    - 改 `finetuning/README.md`
    - 说明控制 token、数据字段、训练/推理一致性约束

2. **commit 2: preprocess(vad+intensity+emphasis)**
    - 改 `finetuning/prepare_data.py`
    - 输出 emotion/intensity/emphasis 字段与调试分数

3. **commit 3: dataset(control-token+mask)**
    - 改 `finetuning/dataset.py`
    - 完成 prepend 与 labels mask

4. **commit 4: sft(trainable-scope+lora)**
    - 改 `finetuning/sft_12hz.py`
    - 完成仅 LLM 可训练与 LoRA 配置

5. **commit 5: inference-api(control-args)**
    - 改 `qwen_tts/inference/qwen3_tts_model.py`、`qwen_tts/cli/demo.py`
    - 保证训练/推理同构

6. **commit 6: model-preprocessor(control-unification)**
    - 改 `qwen_tts/core/models/modeling_qwen3_tts.py`
    - 合并控制 token 入口，减少分散拼接逻辑

7. **commit 7: grpo-scaffold(minimal-loop)**
    - 新增 `grpo_dataset.py`、`grpo_rollout.py`、`reward_fns.py`、`grpo_runner.py`
    - 打通从采样到更新的最小可运行闭环

8. **commit 8: eval-and-ablation**
    - 新增/补充评估脚本与实验记录模板
    - 输出 SFT-only vs SFT+GRPO 对照结果

### 每个 Commit 的验收检查（Quick Gate）

1. **commit 2 gate**：随机抽样 100 条，检查 intensity 三档是否都有样本。  
2. **commit 3 gate**：断言 control token 的 label 全为 `-100`。  
3. **commit 4 gate**：打印 trainable 参数比例，确认 codec/tokenizer 已冻结。  
4. **commit 5 gate**：同一文本在 train/infer 拼接字符串完全一致。  
5. **commit 7 gate**：单机小 batch 能跑通 1 个 epoch 且无 NaN。  
6. **commit 8 gate**：三项核心指标至少 2 项优于 SFT-only。

