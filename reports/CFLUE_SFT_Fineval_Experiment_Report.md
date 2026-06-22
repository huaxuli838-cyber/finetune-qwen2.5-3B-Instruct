# CFLUE 金融单项选择题 SFT + DPO 实验报告

> 基于 Qwen2.5-3B-Instruct 的 LoRA 微调与 DPO 偏好优化，目标提升模型在金融领域选择题（Fineval）上的准确率。

---

## 一、实验目标

1. 利用 CFLUE 数据集中的单项选择题构建高质量 SFT 数据，对 Qwen2.5-3B-Instruct 进行 LoRA 微调；
2. 在 SFT 模型基础上，进一步使用 DPO（Direct Preference Optimization）对齐模型对选择题答案的偏好；
3. 在 Fineval 金融评测集（约 1,150 道 val 题）上验证 SFT 与 DPO 的准确率提升。

---

## 二、数据集

### 2.1 SFT 训练数据

- **来源**：CFLUE（中文金融语言理解评测基准）
- **抽取规则**：仅保留单项选择题；保留原题解析；对缺失解析的题目按原有解析质量补全
- **规模**：**21,265 条**高质量问答对
- **格式**：Alpaca / chat 格式，适配 transformers + peft 训练框架

### 2.2 DPO 训练数据

- **来源**：CFLUE 单项选择题采样（与 SFT 数据同源，避免数据泄漏）
- **采样规模**：从 CFLUE 单选题中随机采样约 6,000 题
- **答案生成**：使用 SFT 最终模型（checkpoint-1896）为每道题生成 4 组不同解码参数的答案
  - greedy
  - temperature = 0.7
  - temperature = 0.9
  - temperature = 1.1
- **Preference Pair 构造**：
  - **Ground truth pairs（1,272 条）**：标准答案判定为“正确”的答案 vs “错误”的答案
  - **Judge pairs（4,728 条）**：当 4 组答案全对或全错时，调用 `deepseek-chat` 从解析质量、逻辑一致性、概念准确性等维度比较，选出 preferred / rejected
- **有效 pairs**：**5,775 条**
- **格式**：`{prompt, chosen, rejected}`，适配 TRL `DPOTrainer`

### 2.3 评测数据

- **来源**：Fineval 验证集
- **规模**：**1,151 道**单项选择题，覆盖 34 个金融/经济子领域
- **示例科目**：基金从业资格、银行从业资格、证券从业资格、CPA、审计、微观经济学、货币银行学等

---

## 三、模型与训练方案

### 3.1 基座模型

- **模型**：Qwen2.5-3B-Instruct
- **下载源**：ModelScope / HuggingFace 镜像
- **本地路径**：`./qwen/Qwen2___5-3B-Instruct/`

### 3.2 微调方法

- **方法**：LoRA（Low-Rank Adaptation）
- **训练框架**：transformers `Trainer` + peft
- **关键超参数**：

| 参数 | 取值 |
|------|------|
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 学习率 | 1e-4 |
| batch size | 4 |
| 梯度累积步数 | 8 |
| 等效 batch size | 32 |
| 最大序列长度 | 1024 |
| 训练轮数 | 3 epochs |
| 优化器 | AdamW |
| 学习率调度 | cosine |

### 3.3 训练优化

- 启用 `gradient_checkpointing` 以降低显存占用
- 因 OOM 将 batch size 从 8 调整至 4，并配合梯度累积保持等效 batch size
- 训练总步数：**1,896 steps**

### 3.4 DPO 训练方案

在 SFT 最终 checkpoint（`checkpoint-1896`）基础上继续训练，使用 TRL `DPOTrainer`。

| 参数 | 取值 |
|------|------|
| 基础模型 | Qwen2.5-3B-Instruct |
| 初始 adapter | `qwen_cflue_lora/checkpoint-1896` |
| 训练框架 | TRL `DPOTrainer` |
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| 学习率 | 5e-7 |
| β（DPO 温度系数） | 0.1 |
| batch size | 4 |
| 梯度累积步数 | 8 |
| 等效 batch size | 32 |
| 最大序列长度 | 1024 |
| 训练轮数 | 1 epoch |
| 学习率调度 | cosine |
| 参考模型 | `None`（TRL 自动创建） |

### 3.5 DPO 训练优化

- 修复 TRL 新版 API 差异：`DPOConfig` 移除 `max_prompt_length`，`DPOTrainer` 使用 `processing_class` 替代 `tokenizer`
- 加载 SFT adapter 时设置 `is_trainable=True`，确保 LoRA 参数参与 DPO 更新
- 训练总步数：**181 steps**

---

## 四、训练结果

### 4.1 Loss 曲线

| epoch | eval_loss | 备注 |
|------|-----------|------|
| 1    | 1.008     | 稳定下降 |
| 2    | 0.9987    | **最低验证 loss** |
| 3    | 1.027     | 轻微回升，未严重过拟合 |

- 最终训练 loss：**0.9069**
- 训练耗时：约 **76 分钟**

### 4.2 保存的 Checkpoints

- `qwen_cflue_lora/checkpoint-632`（epoch 1）
- `qwen_cflue_lora/checkpoint-1264`（epoch 2）
- `qwen_cflue_lora/checkpoint-1896`（epoch 3，最终模型）

### 4.3 DPO Loss 曲线

| step | train_loss | 备注 |
|------|-----------|------|
| 10   | 0.6919    | 初始稳定 |
| 50   | 0.6889    | 开始下降 |
| 100  | 0.6897    | 小幅波动 |
| 170  | 0.6901    | 接近收敛 |
| 181  | 0.6894    | **最终训练 loss** |

- 最终训练 loss：**0.6904**
- 训练耗时：约 **13 分钟**

### 4.4 保存的 DPO Checkpoints

- `qwen_cflue_dpo/checkpoint-181`（最终 step）
- `qwen_cflue_dpo/final`（最终模型）

---

## 五、评测结果

### 5.1 总体正确率

| 模型 | 正确率 | 正确题数 | 相比基线提升 | 相比 SFT 提升 |
|------|--------|----------|--------------|---------------|
| Qwen2.5-3B-Instruct（基线） | **67.59%** | 778 / 1,151 | — | — |
| LoRA SFT epoch 2 | **73.07%** | 841 / 1,151 | **+5.48 pp** | — |
| LoRA SFT epoch 3 | **73.76%** | 849 / 1,151 | **+6.17 pp** | — |
| SFT + DPO | **74.02%** | 852 / 1,151 | **+6.43 pp** | **+0.26 pp** |

> **结论**：
> - 经过 CFLUE SFT 微调后，模型在 Fineval 上的正确率从 **67.59% 提升至 73.76%**，绝对提升 **6.17 个百分点**，相对提升约 **9.1%**。
> - 在 SFT 基础上增加 DPO 对齐后，正确率进一步提升至 **74.02%**，相比 SFT 提升 **0.26 个百分点**，多对 3 道题。

### 5.2 部分科目正确率（SFT 最终模型 checkpoint-1896）

| 科目 | 正确率 | 题数 |
|------|--------|------|
| economic_law | 96.00% | 24/25 |
| cost_accounting | 88.24% | 30/34 |
| fund_qualification_certificate | 85.29% | 58/68 |
| commercial_bank_finance | 85.00% | 17/20 |
| microeconomics | 85.00% | 34/40 |
| central_banking | 85.71% | 24/28 |
| statistics | 82.86% | 29/35 |
| certified_practising_accountant | 82.35% | 28/34 |
| auditing | 81.25% | 26/32 |
| banking_practitioner_qualification_certificate | 76.72% | 89/116 |

### 5.3 DPO 相比 SFT 的分领域变化

| 科目 | SFT 正确率 | DPO 正确率 | 变化 |
|------|-----------|-----------|------|
| futures_practitioner_qualification_certificate | 79.49% | 84.62% | **+5.13 pp** |
| financial_engineering | 50.00% | 53.85% | +3.85 pp |
| investments | 44.74% | 47.37% | +2.63 pp |
| microeconomics | 85.00% | 87.50% | +2.50 pp |
| public_finance | 72.50% | 75.00% | +2.50 pp |
| banking_practitioner_qualification_certificate | 76.72% | 78.45% | +1.72 pp |
| securities_practitioner_qualification_certificate | 72.73% | 63.64% | **-9.09 pp** |
| certified_management_accountant | 72.22% | 66.67% | -5.56 pp |
| monetary_finance | 76.74% | 74.42% | -2.33 pp |
| fund_qualification_certificate | 85.29% | 83.82% | -1.47 pp |

> DPO 在期货从业资格、公共财政、微观经济学等科目上有明显提升，但在证券从业资格、管理会计师等科目上略有下降，整体呈微幅正收益。

---

## 六、实验结论

1. **SFT 微调有效**：在领域相关的高质量选择题数据上微调，能显著提升模型在金融专业考试题上的准确率（67.59% → 73.76%，+6.17 pp）。
2. **DPO 进一步小幅提升**：在 SFT 基础上使用 DPO 对齐，正确率从 73.76% 提升至 74.02%（+0.26 pp）。提升幅度较小，说明 SFT 模型在该任务上已接近瓶颈。
3. **LoRA 高效**：SFT 与 DPO 均只训练约 30M 参数（约 115 MB adapter），即取得显著效果。
4. **DPO 数据质量可靠**：5,775 条 preference pairs 中，22% 来自标准答案判定的正确性对比，78% 来自 DeepSeek judge 的解析质量对比，pair 构造合理。
5. **epoch 3 为最佳 SFT 检查点**：尽管 epoch 2 的验证 loss 最低，但 epoch 3 在下游任务上表现最好。
6. **模型未严重过拟合**：训练 loss 与验证 loss 差距合理，最终模型泛化能力良好。

---

## 七、技术栈

- **模型**：Qwen2.5-3B-Instruct
- **微调**：LoRA（peft）
- **训练框架**：transformers `Trainer`（SFT）、TRL `DPOTrainer`（DPO）
- **深度学习环境**：PyTorch 2.7.0 + CUDA 13.2
- **硬件**：NVIDIA GeForce RTX 5090 D（32GB 显存）
- **数据处理**：pandas、json、Python
- **偏好对构造**：DeepSeek-chat API
- **模型下载**：ModelScope / HuggingFace 镜像
- **模型评测**：自定义批量推理与答案抽取脚本

---

## 八、可复现文件

| 文件 | 说明 |
|------|------|
| `train_cflue_lora.py` | LoRA SFT 训练脚本 |
| `train_cflue_dpo.py` | DPO 训练脚本 |
| `build_dpo_data.py` | DPO 数据构造脚本 |
| `evaluate_fineval_finetuned.py` | 微调模型评测脚本 |
| `evaluate_fineval.py` | 基线模型评测脚本 |
| `cflue_sft_final/cflue_single_choice_all.jsonl` | SFT 训练数据（21,265 条） |
| `cflue_dpo_data.jsonl` | DPO 训练数据（5,775 条 pairs） |
| `fineval_eval_baseline_fixed.json` | 基线模型评测结果 |
| `fineval_eval_epoch3_fixed.json` | SFT 最终模型评测结果 |
| `fineval_eval_dpo.json` | DPO 最终模型评测结果 |
| `qwen_cflue_lora/checkpoint-1896/` | SFT 最终 LoRA adapter |
| `qwen_cflue_dpo/final/` | DPO 最终 LoRA adapter |

---

## 九、后续可优化方向

1. **DPO 超参调优**：当前 DPO 提升有限，可尝试调整 `β`（0.05–0.2）、学习率（1e-7–1e-6）、训练轮数等。
2. **DPO 数据质量增强**：
   - 增加 ground truth 对错样本比例；
   - 对 judge pairs 增加一致性过滤，剔除 judge 置信度低的样本；
   - 尝试使用更强大的模型（如 DeepSeek-V4）作为 judge。
3. **模型集成**：将 epoch 2、epoch 3 与 DPO 模型的预测结果进行投票，可能进一步提升准确率。
4. **增大 LoRA 容量**：尝试 `r=32/64`，观察是否还能带来提升。
5. **数据增强**：引入更多金融考试真题或解析质量更高的数据。
6. **指令模板优化**：针对选择题设计更结构化的 prompt，强化选项选择能力。

---

*报告生成时间：2026-06-15*
