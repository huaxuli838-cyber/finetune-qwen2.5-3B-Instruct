# CFLUE DPO 实验报告

## 实验目标

在 Qwen2.5-3B-Instruct 完成 CFLUE SFT 的基础上，使用 CFLUE 单项选择题构造 preference pairs，通过 DeepSeek 大模型进行答案评判（judge），训练 DPO（Direct Preference Optimization），进一步提升在 **Fineval val**（1151 道单项选择题）上的准确率。

## 环境信息

- **GPU**：NVIDIA GeForce RTX 5090 D，32GB 显存
- **CUDA / PyTorch**：CUDA 13.2，PyTorch 2.7.0+cu128
- **关键库版本**：transformers 5.12.0，peft 0.19.1，trl 1.6.0，openai 2.41.1
- **基座模型**：`./qwen/Qwen2___5-3B-Instruct`
- **SFT LoRA adapter**：`./qwen_cflue_lora/checkpoint-1896`
- **评测集**：Fineval val，共 1151 道单项选择题

## 前置结果（SFT）

| 阶段 | 正确率 | 正确数 |
|---|---|---|
| 基座模型 | 67.59% | 778 / 1151 |
| SFT（3 epoch best checkpoint） | 73.76% | 849 / 1151 |

## DPO 数据构造方法

### 公共流程

1. 从 CFLUE 单选题中采样题目；
2. 对每个问题生成 4 组答案：
   - **greedy**（temperature=0）
   - **temperature=0.7**
   - **temperature=0.9**
   - **temperature=1.1**
3. 根据答案与 ground truth 的关系预分类：
   - **ground truth pair**：一个答案正确、一个答案错误，直接构造 chosen/rejected；
   - **judge pair**：两个答案都正确或都错误，调用 DeepSeek judge 判定优劣并给出置信度；
4. 过滤低置信度 judge pairs，保留有效 pairs；
5. 保存为 `cflue_dpo_data_v*.jsonl`，用于 DPO 训练。

### DPO v1（第一轮）

- **采样数量**：约 6,000 题
- **有效 pairs**：**5,775** 条
  - ground truth pairs：1,272 条
  - judge pairs：4,503 条
- **Judge 模型**：deepseek-chat
- **置信度过滤**：未引入，仅保留 judge 有明确胜负的 pairs
- **输出文件**：`cflue_dpo_data_v1_5775.jsonl`

### DPO v2（第二轮）

- **采样数量**：**11,000** 题
- **有效 pairs**：**10,608** 条
- **新增策略**：
  - 样本量从 6k 提升到 11k；
  - 引入 **judge 置信度评分（1–5）**，仅保留 **confidence = 5** 的高置信度 judge pairs。
- **Judge 模型**：deepseek-chat
- **输出文件**：`cflue_dpo_data_v2.jsonl`

## DPO 训练配置

两轮 DPO 训练使用相同超参数：

| 超参数 | 值 |
|---|---|
| 基础模型 | Qwen2.5-3B-Instruct |
| SFT adapter | checkpoint-1896 |
| LoRA r | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| 学习率 | 5e-7 |
| β | 0.1 |
| epoch | 1 |
| per_device_train_batch_size | 4 |
| gradient_accumulation_steps | 8 |
| max_length | 1024 |
| lr_scheduler | cosine |
| warmup_ratio | 0.03 |
| bf16 | True |
| gradient_checkpointing | True |

### DPO v1 训练

- **输出目录**：`./qwen_cflue_dpo/`
- **训练样本**：5,775
- **训练步数**：约 181 steps
- **最终 loss**：约 0.69

### DPO v2 训练

- **输出目录**：`./qwen_cflue_dpo_v2/`
- **训练样本**：10,608
- **训练步数**：332 steps
- **训练时长**：约 24 分 45 秒
- **最终 loss**：**0.6876**
- **rewards/accuracies**：**62.5%**
- **最终模型**：`./qwen_cflue_dpo_v2/final`

## Fineval val 评测结果

| 阶段 | 正确率 | 正确数 | 相比 SFT | 相比 DPO v1 |
|---|---|---|---|---|
| 基座模型 | 67.59% | 778 / 1151 | — | — |
| SFT（checkpoint-1896） | 73.76% | 849 / 1151 | — | — |
| **DPO v1** | **74.02%** | **852 / 1151** | **+0.26 pp** | — |
| **DPO v2** | **73.94%** | **851 / 1151** | **+0.18 pp** | **-0.08 pp** |

## 结果分析

1. **DPO 训练有效但收益有限**
   - 相较于 SFT，DPO v1 提升 0.26 pp，DPO v2 提升 0.18 pp，均验证了对齐训练能带来正向收益。
   - 但两轮 DPO 差距仅 1 题（851 vs 852），说明 **继续扩大同分布数据规模对 Fineval 的边际增益很小**。

2. **样本量翻倍未能带来提升**
   - DPO v2 将 pairs 从 5,775 提升到 10,608，增幅 83.7%，但正确率反而略低于 v1。
   - 可能原因：
     - **数据同质性**：CFLUE 单选题整体分布相似，增加数量并未显著扩充模型未覆盖的知识；
     - **置信度过滤过严**：仅保留 confidence=5 的样本，可能过滤掉部分对模型有学习价值的“边界”pairs；
     - **DPO 优化上限**：在固定 SFT adapter 上继续 DPO，模型对 preference 信号的拟合可能已接近饱和。

3. **DPO 训练指标健康**
   - v2 最终 loss 0.6876，rewards/accuracies 62.5%，reward margins 为正，说明模型确实学到了偏好；
   - 但最终评测指标没有同步提升，提示 **训练集上的偏好学习 ≠ 下游选择题准确率提升**。

## 后续优化方向

1. **放宽置信度过滤**：尝试保留 confidence ≥ 3 或 ≥ 4 的样本，增加边界样本；
2. **引入难度/多样性采样**：避免简单题重复，优先选择模型容易出错的题目构造 pairs；
3. **答案生成策略优化**：尝试更大温度范围或 top-p 采样，生成质量差异更明显的候选；
4. **DPO 超参调优**：尝试 β=0.05 / 0.2、lr=1e-6 / 2e-7、2 epoch 等；
5. **引入 KTO / IPO 等替代算法**：对比不同对齐目标在 Fineval 上的效果。

## 关键文件

| 文件 | 说明 |
|---|---|
| `cflue_dpo_data_v1_5775.jsonl` | DPO v1 训练数据（5,775 pairs） |
| `cflue_dpo_data_v2.jsonl` | DPO v2 训练数据（10,608 pairs） |
| `qwen_cflue_dpo/final` | DPO v1 最终模型 |
| `qwen_cflue_dpo_v2/final` | DPO v2 最终模型 |
| `fineval_eval_dpo.json` | DPO v1 Fineval 评测结果 |
| `fineval_eval_dpo_v2_results.json` | DPO v2 Fineval 评测结果 |
| `CFLUE_SFT_Fineval_Experiment_Report.md` | SFT 阶段实验报告 |

## 结论

DPO 对齐训练在 SFT 基础上能够小幅提升 Fineval 准确率，但 **单纯增加同分布 CFLUE 单选题样本量并收紧 judge 置信度过滤，并未带来进一步收益**。后续应重点关注数据多样性、置信度阈值、DPO 超参与对齐算法的选择。
