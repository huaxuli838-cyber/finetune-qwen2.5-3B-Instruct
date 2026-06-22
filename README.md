# Finetune Qwen2.5-3B-Instruct for Chinese Finance QA

基于 **Qwen2.5-3B-Instruct** 的中文金融领域大模型微调项目。项目围绕 **CFLUE 金融单项选择题** 与 **高质量金融文本** 构建 SFT / DPO 训练数据，重点提升模型在金融专业考试题（Fineval）上的准确率，同时保持通用中文能力（C-Eval）。

> 基座模型：Qwen2.5-3B-Instruct  
> 微调方法：LoRA（peft）  
> 对齐方法：DPO / IPO（TRL）  
> 硬件：NVIDIA GeForce RTX 5090 D（32GB）  

---

## 目录

- [核心成果](#核心成果)
- [项目结构](#项目结构)
- [实验概览](#实验概览)
  - [实验 1：CFLUE 单项选择题 SFT](#实验-1cflue-单项选择题-sft)
  - [实验 2：SFT + DPO 偏好对齐](#实验-2sft--dpo-偏好对齐)
  - [实验 3：DPO v8 IPO 生成质量优化](#实验-3dpo-v8-ipo-生成质量优化)
  - [实验 4：开放式金融 SFT v2](#实验-4开放式金融-sft-v2)
- [快速复现](#快速复现)
- [文件说明](#文件说明)
- [注意事项](#注意事项)

---

## 核心成果

| 实验 | 评测集 | 基座 / 对照 | 最佳结果 | 关键提升 |
|------|--------|-------------|----------|----------|
| **CFLUE SFT** | Fineval val（1,151 题） | 67.59% | **73.76%** | **+6.17 pp** |
| **SFT + DPO v1** | Fineval val | 73.76%（SFT） | **74.02%** | +0.26 pp |
| **SFT / DPO** | C-Eval val（1,346 题） | 68.13% | **69.91%** | 通用能力未下降 |
| **DPO v8 IPO** | 300 题 pairwise 质量评估 | SFT 生成质量 | **25.0% 胜率**（SFT 仅 13.3%） | 生成质量显著提升 |
| **SFT v2** | Fineval val（严格格式） | 67.94%（宽松格式） | **71.76%** | 格式规范化提升 3.8 pp |

> **pp** = percentage points（百分点）

### 主要结论

1. **CFLUE SFT 是效果最好、最稳定的实验**：在约 2.1 万条金融单选题数据上 LoRA 微调 3 个 epoch，Fineval 准确率从 **67.59% 提升到 73.76%**（+6.17 pp）。
2. **DPO 对齐有正向但边际收益小**：在 SFT 基础上增加 DPO，Fineval 只提升 0.26 pp，说明该任务已接近同分布数据的上限。
3. **通用能力未受损**：C-Eval 上 SFT 后反而从 68.13% 提升到 69.91%，没有灾难性遗忘。
4. **生成质量与选项正确率存在 trade-off**：DPO v8 IPO 在生成质量评测中大幅领先，但 Fineval 选项正确率下降至 67.7%。
5. **输出格式对评测分数影响很大**：SFT v2 强制使用“答案：X / 解析：…”格式后，Fineval 从 67.94% 提升到 71.76%。

---

## 项目结构

```
finetune-qwen2.5-3B-Instruct/
├── README.md                       # 本文件
├── requirements.txt                # Python 依赖
├── .gitignore                      # 忽略大文件/模型/数据
├── scripts/
│   ├── download_model.py           # 下载 Qwen2.5-3B-Instruct
│   ├── data/                       # 数据构造脚本
│   │   ├── build_cflue_sft.py      # 从 CFLUE 构造 SFT 数据
│   │   ├── build_dpo_data.py       # 构造 DPO v1/v2 preference pairs
│   │   ├── build_dpo_data_v8.py    # 构造 DPO v8 IPO 数据（质量评分）
│   │   ├── build_finance_sft_data.py      # 从 BAAI 金融语料构造开放式 SFT v1
│   │   └── build_cflue_calc_sft.py        # 从 CFLUE 计算题改写为开放式问答
│   ├── training/                   # 训练脚本
│   │   ├── train_cflue_lora.py     # LoRA SFT 训练
│   │   └── train_cflue_dpo.py      # DPO / IPO 训练
│   └── evaluation/                 # 评测脚本
│       ├── evaluate_fineval.py              # Fineval 基座模型评测
│       ├── evaluate_fineval_finetuned.py    # Fineval 微调模型评测
│       ├── evaluate_fineval_v2_strict.py    # Fineval 严格格式评测
│       ├── evaluate_ceval.py                # C-Eval 通用能力评测
│       ├── evaluate_dpo_quality_v7.py       # pairwise 生成质量评测
│       └── eval_finance_sft_v2.py           # 开放式金融 SFT v2 推理测试
├── reports/                        # 实验报告
│   ├── CFLUE_SFT_Fineval_Experiment_Report.md
│   ├── CFLUE_DPO_Experiment_Report.md
│   ├── DPO_v8_Generation_Quality_Report.md
│   └── CFLUE_CEval_General_Capability_Report.md
├── results/                        # 评测结果摘要（不含完整 raw output）
│   ├── evaluation_summary.json
│   └── dpo_v8_quality_summary.json
└── examples/
    └── data_format.md              # 训练数据格式示例
```

---

## 实验概览

### 实验 1：CFLUE 单项选择题 SFT

- **数据**：CFLUE（中文金融语言理解评测基准）中的单项选择题，共 **21,265 条**。
- **方法**：LoRA r=16, alpha=32, 3 epochs，学习率 1e-4。
- **关键脚本**：
  - 数据：`scripts/data/build_cflue_sft.py`
  - 训练：`scripts/training/train_cflue_lora.py`
  - 评测：`scripts/evaluation/evaluate_fineval_finetuned.py`
- **结果**：

| 模型 | Fineval 正确率 | 正确数 |
|------|----------------|--------|
| 基座 | 67.59% | 778 / 1,151 |
| SFT epoch 2 | 73.07% | 841 / 1,151 |
| **SFT epoch 3** | **73.76%** | **849 / 1,151** |

### 实验 2：SFT + DPO 偏好对齐

- **数据**：在 CFLUE 单选题上采样，用 SFT 模型生成 4 组答案，结合标准答案对错与 DeepSeek judge 评分构造 preference pairs。
  - DPO v1：5,775 pairs
  - DPO v2：10,608 pairs（高置信度过滤）
- **方法**：基于 SFT checkpoint-1896，继续 DPO 训练 1 epoch，lr=5e-7，β=0.1。
- **关键脚本**：
  - 数据：`scripts/data/build_dpo_data.py`
  - 训练：`scripts/training/train_cflue_dpo.py`
  - 评测：`scripts/evaluation/evaluate_fineval_finetuned.py`
- **结果**：

| 模型 | Fineval 正确率 | 正确数 |
|------|----------------|--------|
| SFT | 73.76% | 849 / 1,151 |
| DPO v1 | **74.02%** | 852 / 1,151 |
| DPO v2 | 73.94% | 851 / 1,151 |

### 实验 3：DPO v8 IPO 生成质量优化

- **目标**：不只是提升选项正确率，而是提升回答的**专业性、推理清晰度、完整性和格式规范**。
- **数据**：7,000 题，每题 4 个回答，DeepSeek 按质量打分（不看选项对错），分差 ≥2 才构造 pair，最终 7,701 pairs。
- **方法**：IPO（`loss_type='ipo'`），lr=5e-7，β=0.1，1 epoch。
- **关键脚本**：
  - 数据：`scripts/data/build_dpo_data_v8.py`
  - 训练：`scripts/training/train_cflue_dpo.py --loss_type ipo`
  - 评测：`scripts/evaluation/evaluate_dpo_quality_v7.py`
- **结果**：

| 指标 | SFT | DPO v8 IPO |
|------|-----|------------|
| pairwise 胜率 | 13.3% | **25.0%** |
| 平局率 | 68.7% | 59.7% |
| 失败率 | 18.0% | 15.3% |
| 选项正确率 | **71.3%** | 67.7% |

> 生成质量提升明显，但选项正确率下降，说明存在 **quality-accuracy trade-off**。

### 实验 4：开放式金融 SFT v2

- **动机**：避免模型只学会做选择题，从 BAAI IndustryCorpus2 金融 high 子集和 CFLUE 计算题中构造**开放式**金融问答对。
- **数据**：
  - BAAI 金融语料 → 概念理解、政策解读、市场分析、实务应用、摘要抽取等 14,716 条
  - CFLUE 计算题改写 → 1,786 条开放式计算/推理题
  - 合并 **SFT v2：16,502 条**
- **关键脚本**：
  - 数据：`scripts/data/build_finance_sft_data.py`、`scripts/data/build_cflue_calc_sft.py`
  - 训练：`scripts/training/train_cflue_lora.py`
  - 评测：`scripts/evaluation/evaluate_fineval_v2_strict.py`
- **结果**：

| 模型 | Fineval 正确率 |
|------|----------------|
| 基座 | 67.59% |
| SFT v2（宽松格式） | 67.94% |
| **SFT v2（严格格式）** | **71.76%** |

> 开放式数据让模型具备更好的解释能力，但在严格考试题准确率上仍略低于 CFLUE 单选题 SFT（73.76%）。

---

## 快速复现

### 1. 环境准备

```bash
# 创建并激活虚拟环境（推荐）
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 设置 DeepSeek API key（用于数据构造与 judge）
export DEEPSEEK_API_KEY="sk-..."
```

### 2. 下载基座模型

```bash
python scripts/download_model.py
```

模型会下载到当前目录的 `qwen/Qwen2___5-3B-Instruct/` 下。

### 3. CFLUE SFT 复现

```bash
# 1) 准备 CFLUE 数据到 ./cflue_full/，然后构造 SFT 数据
python scripts/data/build_cflue_sft.py --input_dir ./cflue_full --output_dir ./cflue_sft

# 2) 训练 LoRA SFT（约 75 分钟）
python scripts/training/train_cflue_lora.py \
  --model_path ./qwen/Qwen2___5-3B-Instruct \
  --data_path ./cflue_sft/cflue_single_choice_all.jsonl \
  --output_dir ./qwen_cflue_lora \
  --num_train_epochs 3

# 3) Fineval 评测
python scripts/evaluation/evaluate_fineval_finetuned.py \
  --base_model_path ./qwen/Qwen2___5-3B-Instruct \
  --adapter_path ./qwen_cflue_lora/checkpoint-1896 \
  --output fineval_eval_sft.json
```

### 4. DPO 复现

```bash
# 1) 构造 DPO 数据
python scripts/data/build_dpo_data.py \
  --data_path ./cflue_sft/cflue_single_choice_all.jsonl \
  --base_model_path ./qwen/Qwen2___5-3B-Instruct \
  --adapter_path ./qwen_cflue_lora/checkpoint-1896 \
  --output ./cflue_dpo_data.jsonl

# 2) DPO 训练（约 13 分钟）
python scripts/training/train_cflue_dpo.py \
  --base_model_path ./qwen/Qwen2___5-3B-Instruct \
  --sft_adapter_path ./qwen_cflue_lora/checkpoint-1896 \
  --data_path ./cflue_dpo_data.jsonl \
  --output_dir ./qwen_cflue_dpo

# 3) Fineval 评测
python scripts/evaluation/evaluate_fineval_finetuned.py \
  --base_model_path ./qwen/Qwen2___5-3B-Instruct \
  --adapter_path ./qwen_cflue_dpo/final \
  --output fineval_eval_dpo.json
```

### 5. C-Eval 通用能力评测

```bash
python scripts/evaluation/evaluate_ceval.py \
  --base_model_path ./qwen/Qwen2___5-3B-Instruct \
  --adapter_path ./qwen_cflue_lora/checkpoint-1896 \
  --data_dir ./ceval/val \
  --output ceval_eval_sft.json
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/data/build_cflue_sft.py` | 从原始 CFLUE 数据中提取单项选择题，构造 Alpaca / chat 格式的 SFT 数据 |
| `scripts/data/build_dpo_data.py` | 用 SFT 模型生成多组答案，结合标准答案和 DeepSeek judge 构造 DPO preference pairs |
| `scripts/data/build_dpo_data_v8.py` | DPO v8 数据构造：DeepSeek 按生成质量打分，分差 ≥2 构造 pair |
| `scripts/data/build_finance_sft_data.py` | 从 BAAI IndustryCorpus2 金融 high 子采样构造开放式金融问答对 |
| `scripts/data/build_cflue_calc_sft.py` | 把 CFLUE 中的计算题改写成开放式计算/推理问答 |
| `scripts/training/train_cflue_lora.py` | LoRA SFT 训练脚本 |
| `scripts/training/train_cflue_dpo.py` | DPO / IPO 训练脚本（支持 `sigmoid`、`ipo` 等 loss_type） |
| `scripts/evaluation/evaluate_fineval.py` | Fineval 基座模型评测 |
| `scripts/evaluation/evaluate_fineval_finetuned.py` | Fineval 微调模型评测 |
| `scripts/evaluation/evaluate_fineval_v2_strict.py` | 强制“答案：X / 解析：…”格式的 Fineval 评测 |
| `scripts/evaluation/evaluate_ceval.py` | C-Eval 通用能力评测 |
| `scripts/evaluation/evaluate_dpo_quality_v7.py` | pairwise 生成质量评测 |
| `scripts/evaluation/eval_finance_sft_v2.py` | 开放式金融 SFT v2 推理测试 |

---

## 注意事项

1. **大文件不上传**：本仓库只包含代码、报告和结果摘要。模型权重、完整数据集、`.jsonl` 训练数据因体积过大已加入 `.gitignore`。
2. **DeepSeek API**：涉及 `build_dpo_data*.py`、`build_finance_sft_data.py`、`build_cflue_calc_sft.py` 的脚本需要 `DEEPSEEK_API_KEY` 环境变量。
3. **硬件要求**：训练脚本默认使用 `device_map='auto'` 和 bf16，建议在 ≥24GB 显存的 NVIDIA GPU 上运行。
4. **路径约定**：脚本默认使用相对路径。运行前请确保 `qwen/Qwen2___5-3B-Instruct/`、`fineval/val/`、`ceval/val/`、`cflue_full/` 等目录位于工作目录下。

---

## 许可证

本项目代码部分遵循 MIT 许可证。模型权重、数据集和评测集遵循其各自原始许可证。
