# CFLUE 微调通用能力评估报告（C-Eval val）

## 评估目标

验证 Qwen2.5-3B-Instruct 经过 CFLUE 金融单项选择题 SFT 与 DPO 训练后，**通用中文知识能力**是否出现明显下降。

## 评估基准

- **数据集**：C-Eval val（中文综合能力评测验证集）
- **题目数量**：**1,346 题**
- **覆盖科目**：52 个中文科目，涵盖 STEM、人文、社科等领域
- **评估方式**：与 Fineval 评测保持一致
  - prompt 模板为中文单项选择题格式；
  - 不使用金融专用 system prompt，避免领域 prompt 对通用评测的干扰；
  - greedy 解码，max_new_tokens=16；
  - 从模型输出中提取 A/B/C/D 选项字母。

## 模型版本

| 模型 | 说明 |
|---|---|
| 基座模型 | `Qwen2.5-3B-Instruct` |
| SFT | 基座 + `./qwen_cflue_lora/checkpoint-1896` |
| DPO v1 | SFT + DPO，5,775 preference pairs |
| DPO v2 | SFT + DPO，10,608 preference pairs（高置信度过滤） |

## 总体正确率

| 模型 | 正确率 | 正确数 | 相比基座 | 相比 SFT |
|---|---|---|---|---|
| 基座模型 | **68.13%** | 917 / 1346 | — | — |
| SFT | **69.91%** | 941 / 1346 | **+1.78 pp** | — |
| DPO v1 | **69.76%** | 939 / 1346 | **+1.63 pp** | -0.15 pp |
| DPO v2 | **69.84%** | 940 / 1346 | **+1.71 pp** | -0.07 pp |

## 分领域正确率

| 领域 | 基座 | SFT | DPO v1 | DPO v2 |
|---|---|---|---|---|
| STEM | 64.26% | 62.54% | 62.37% | 62.20% |
| 人文 | 72.51% | 77.78% | 76.61% | 77.78% |
| 社科 | 70.93% | 75.13% | 75.31% | 75.31% |

## 与 Fineval 表现的对比

| 模型 | Fineval val（金融专业） | C-Eval val（通用中文） |
|---|---|---|
| 基座模型 | 67.59% | 68.13% |
| SFT | 73.76% | 69.91% |
| DPO v1 | 74.02% | 69.76% |
| DPO v2 | 73.94% | 69.84% |

## 关键结论

1. **通用能力未下降，反而略有提升**
   - SFT 在 C-Eval val 上从 68.13% 提升到 69.91%（+1.78 pp）；
   - DPO v1 / v2 基本维持在 SFT 水平，分别只下降 0.15 pp / 0.07 pp，可视为统计波动。

2. **STEM 略有下降，人文社科有所提升**
   - STEM：SFT 后下降约 1.72 pp，DPO 后继续小幅下滑至 62.20%；
   - 人文：SFT 提升 5.26 pp，DPO v2 保持该水平；
   - 社科：SFT 提升 4.20 pp，DPO v1/v2 进一步提升到 75.31%。

3. **DPO 对通用能力影响极小**
   - 无论是 5.7k pairs 的 DPO v1，还是 10.6k pairs、经过高置信度过滤的 DPO v2，都没有造成通用能力显著下降；
   - 说明当前 CFLUE 数据的偏好信号没有导致明显的过拟合或灾难性遗忘。

4. **专业领域与通用能力呈弱正相关**
   - SFT 在 Fineval 上提升 6.17 pp，在 C-Eval 上同时提升 1.78 pp；
   - 可能原因：CFLUE 金融题库包含大量中文阅读、逻辑推理与专业知识判断，训练过程中增强了模型对中文复杂题干的理解能力。

## 主要下降科目（SFT vs 基座，下降 > 10 pp）

| 科目 | 基座 | SFT | 变化 |
|---|---|---|---|
| high_school_physics | 84.21% | 63.16% | -21.05 pp |
| high_school_chemistry | 73.68% | 52.63% | -21.05 pp |
| probability_and_statistics | 38.89% | 22.22% | -16.67 pp |
| urban_and_rural_planner | 73.91% | 60.87% | -13.04 pp |
| college_physics | 52.63% | 42.11% | -10.53 pp |
| ideological_and_moral_cultivation | 100.00% | 89.47% | -10.53 pp |
| middle_school_mathematics | 57.89% | 47.37% | -10.53 pp |
| high_school_geography | 94.74% | 84.21% | -10.53 pp |

## 主要提升科目（SFT vs 基座，提升 > 10 pp）

| 科目 | 基座 | SFT | 变化 |
|---|---|---|---|
| civil_servant | 51.06% | 72.34% | +21.28 pp |
| chinese_language_and_literature | 39.13% | 56.52% | +17.39 pp |
| middle_school_history | 81.82% | 100.00% | +18.18 pp |
| mao_zedong_thought | 83.33% | 95.83% | +12.50 pp |
| accountant | 65.31% | 75.51% | +10.20 pp |
| education_science | 72.41% | 82.76% | +10.34 pp |
| computer_network | 68.42% | 78.95% | +10.53 pp |
| operating_system | 57.89% | 68.42% | +10.53 pp |

## 后续建议

- 如果目标是**兼顾通用能力**：当前 SFT + DPO 方案已经足够安全，可以继续做领域对齐；
- 如果希望**恢复 STEM 能力**：可在 DPO 数据中混入少量 C-Eval / 高考理科题，作为正则化；
- 如果仅追求 **Fineval 准确率**：DPO v2 样本量翻倍并未带来提升，后续应转向数据质量、超参与算法优化，而非简单堆量。

## 关键文件

| 文件 | 说明 |
|---|---|
| `evaluate_ceval.py` | C-Eval 评测脚本 |
| `ceval_eval_base_results.json` | 基座模型评测详情 |
| `ceval_eval_sft_results.json` | SFT 模型评测详情 |
| `ceval_eval_dpo_v1_results.json` | DPO v1 评测详情 |
| `ceval_eval_dpo_v2_results.json` | DPO v2 评测详情 |
| `CFLUE_SFT_Fineval_Experiment_Report.md` | Fineval 专业评测报告 |
