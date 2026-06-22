# Finetune Qwen2.5-3B-Instruct for Chinese Finance QA

基于 **Qwen2.5-3B-Instruct** 的中文金融领域大模型微调项目。项目重点探索两条路线：

1. **开放式金融 SFT**：从高质量金融文本（BAAI IndustryCorpus2 金融 high 子集）和 CFLUE 计算题中构造开放式问答对，提升模型在金融概念理解、政策解读、市场分析、计算推理等方面的真实能力。
2. **偏好对齐（IPO）**：用 CFLUE 单项选择题构造 preference pairs，优化模型回答的专业性、推理清晰度、完整性和格式规范。

> 本项目不将单项选择题直接作为 SFT 训练数据，而是仅将其用于 IPO 偏好对齐与能力评测。  
> 基座模型：Qwen2.5-3B-Instruct  
> 微调方法：LoRA（peft）  
> 对齐方法：IPO（TRL）  
> 硬件：NVIDIA GeForce RTX 5090 D（32GB）  

---

## 目录

- [核心成果](#核心成果)
- [项目结构](#项目结构)
- [实验概览](#实验概览)
  - [实验 1：开放式金融 SFT](#实验-1开放式金融-sft)
  - [实验 2：IPO 生成质量优化](#实验-2ipo-生成质量优化)
- [快速复现](#快速复现)
- [文件说明](#文件说明)
- [注意事项](#注意事项)

---

## 核心成果

| 实验 | 评测集 | 基座 / 对照 | 最佳结果 | 关键提升 |
|------|--------|-------------|----------|----------|
| **SFT** | Fineval val（1,151 题，严格格式） | 67.94%（宽松格式） | **71.76%** | 格式规范化提升 3.8 pp |
| **IPO** | 300 题 pairwise 质量评估 | SFT 生成质量 | **25.0% 胜率**（SFT 仅 13.3%） | 生成质量显著提升 |
| **Fineval 基座** | Fineval val | — | 67.59% | — |

> **pp** = percentage points（百分点）

### 主要结论

1. **开放式 SFT 有效**：在不使用单项选择题作为 SFT 数据的情况下，模型在 Fineval 上从基座 67.59% 提升到严格格式下的 **71.76%**。
2. **输出格式对评测分数影响很大**：强制使用“答案：X / 解析：…”格式后，SFT 的 Fineval 分数从 67.94% 提升到 **71.76%**。
3. **SFT 弱项**：投资学（36.84%）、金融工程（50.00%）、宏观经济学（54.84%）、计量经济学（55.56%）等科目仍有较大提升空间。
4. **IPO 显著提升生成质量**：在专业性、推理清晰度、完整性等维度上全面超过 SFT，pairwise 胜率从 13.3% 提升到 **25.0%**。
5. **生成质量与选项正确率存在 trade-off**：IPO 的生成质量更好，但 Fineval 选项正确率从 71.3% 下降到 67.7%。

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
│   │   ├── build_finance_sft_data.py      # 从 BAAI 金融语料构造开放式 SFT 数据
│   │   ├── build_cflue_calc_sft.py        # 从 CFLUE 计算题改写为开放式问答
│   │   └── build_ipo_data.py              # 构造 IPO preference pairs（质量评分）
│   ├── training/                   # 训练脚本
│   │   ├── train_cflue_lora.py     # LoRA SFT 训练
│   │   └── train_cflue_ipo.py      # IPO 训练
│   └── evaluation/                 # 评测脚本
│       ├── evaluate_fineval.py              # Fineval 基座模型评测
│       ├── evaluate_fineval_finetuned.py    # Fineval 微调模型评测
│       ├── evaluate_fineval_strict.py       # Fineval 严格格式评测
│       ├── evaluate_ceval.py                # C-Eval 通用能力评测
│       ├── eval_finance_sft.py              # 开放式金融 SFT 推理测试
│       └── evaluate_ipo_quality.py          # pairwise 生成质量评测
├── reports/                        # 实验报告
│   └── IPO_Generation_Quality_Report.md
├── results/                        # 评测结果摘要（不含完整 raw output）
│   ├── evaluation_summary.json
│   └── ipo_quality_summary.json
└── examples/
    └── data_format.md              # 训练数据格式示例
```

---

## 实验概览

### 实验 1：开放式金融 SFT

- **动机**：避免模型只学会做选择题，从高质量金融文本和 CFLUE 计算题中构造**开放式**金融问答对，提升真实金融能力。
- **数据**：
  - BAAI IndustryCorpus2 金融 high 子集 → 概念理解、政策解读、市场分析、实务应用、摘要抽取等 **14,716 条**
  - CFLUE 计算题改写 → **1,786 条**开放式计算/推理题
  - 合并 **SFT：16,502 条**
- **关键脚本**：
  - 数据：`scripts/data/build_finance_sft_data.py`、`scripts/data/build_cflue_calc_sft.py`
  - 训练：`scripts/training/train_cflue_lora.py`
  - 评测：`scripts/evaluation/evaluate_fineval_strict.py`
- **结果**：

| 模型 | Fineval 正确率 |
|------|----------------|
| 基座 | 67.59% |
| SFT（宽松格式） | 67.94% |
| **SFT（严格格式）** | **71.76%** |

### 实验 2：IPO 生成质量优化

- **目标**：不只是提升选项正确率，而是提升回答的**专业性、推理清晰度、完整性和格式规范**。
- **数据**：从 CFLUE 单项选择题中采样 7,000 题，每题用 SFT 模型生成 4 个回答；DeepSeek 按生成质量打分（不看选项对错），分差 ≥2 才构造 pair，最终 **7,701 pairs**。
- **方法**：IPO（`loss_type='ipo'`），基于 SFT checkpoint 继续训练，lr=5e-7，β=0.1，1 epoch。
- **关键脚本**：
  - 数据：`scripts/data/build_ipo_data.py`
  - 训练：`scripts/training/train_cflue_ipo.py --loss_type ipo`
  - 评测：`scripts/evaluation/evaluate_ipo_quality.py`
- **结果**：

| 指标 | SFT | IPO |
|------|-----|-----|
| pairwise 胜率 | 13.3% | **25.0%** |
| 平局率 | 68.7% | 59.7% |
| 失败率 | 18.0% | 15.3% |
| 选项正确率 | **71.3%** | 67.7% |

> 生成质量提升明显，但选项正确率下降，说明存在 **quality-accuracy trade-off**。

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

### 3. 开放式金融 SFT 复现

```bash
# 1) 构造开放式 SFT 数据
python scripts/data/build_finance_sft_data.py \
  --data_dir ./BAAI_IndustryCorpus2_finance_high/finance_economics/chinese/high \
  --output_dir ./finance_sft \
  --n_passages 5000

# 2) 从 CFLUE 计算题中补充开放式计算/推理题
python scripts/data/build_cflue_calc_sft.py \
  --cflue_path ./cflue_sft/cflue_single_choice_all.jsonl \
  --output_dir ./finance_sft \
  --n_target 1800

# 3) 训练 LoRA SFT
python scripts/training/train_cflue_lora.py \
  --model_path ./qwen/Qwen2___5-3B-Instruct \
  --data_path ./finance_sft/finance_sft_train.jsonl \
  --output_dir ./qwen_finance_sft \
  --num_train_epochs 3

# 4) Fineval 严格格式评测
python scripts/evaluation/evaluate_fineval_strict.py \
  --base_model_path ./qwen/Qwen2___5-3B-Instruct \
  --adapter_path ./qwen_finance_sft/final \
  --output fineval_sft_strict.json
```

### 4. IPO 复现（可选）

```bash
# 1) 构造 IPO 数据（需要一个已有的 SFT checkpoint 作为采样模型）
python scripts/data/build_ipo_data.py \
  --input_path ./cflue_sft/cflue_single_choice_all.jsonl \
  --base_model_path ./qwen/Qwen2___5-3B-Instruct \
  --sft_adapter_path ./qwen_finance_sft/final \
  --output ./cflue_ipo_data.jsonl

# 2) IPO 训练
python scripts/training/train_cflue_ipo.py \
  --base_model_path ./qwen/Qwen2___5-3B-Instruct \
  --sft_adapter_path ./qwen_finance_sft/final \
  --data_path ./cflue_ipo_data.jsonl \
  --output_dir ./qwen_cflue_ipo \
  --loss_type ipo

# 3) pairwise 生成质量评测
python scripts/evaluation/evaluate_ipo_quality.py \
  --base_model_path ./qwen/Qwen2___5-3B-Instruct \
  --sft_adapter_path ./qwen_finance_sft/final \
  --ipo_adapter_path ./qwen_cflue_ipo/final
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/data/build_finance_sft_data.py` | 从 BAAI IndustryCorpus2 金融 high 子采样构造开放式金融问答对 |
| `scripts/data/build_cflue_calc_sft.py` | 把 CFLUE 中的计算题改写成开放式计算/推理问答 |
| `scripts/data/build_ipo_data.py` | 用 SFT 模型生成多组答案，DeepSeek 按生成质量打分，构造 IPO preference pairs |
| `scripts/training/train_cflue_lora.py` | LoRA SFT 训练脚本 |
| `scripts/training/train_cflue_ipo.py` | IPO 训练脚本（支持 `sigmoid`、`ipo` 等 loss_type） |
| `scripts/evaluation/evaluate_fineval.py` | Fineval 基座模型评测 |
| `scripts/evaluation/evaluate_fineval_finetuned.py` | Fineval 微调模型评测（通用 adapter 评测脚本） |
| `scripts/evaluation/evaluate_fineval_strict.py` | 强制“答案：X / 解析：…”格式的 Fineval 评测 |
| `scripts/evaluation/evaluate_ceval.py` | C-Eval 通用能力评测 |
| `scripts/evaluation/eval_finance_sft.py` | 开放式金融 SFT 推理测试 |
| `scripts/evaluation/evaluate_ipo_quality.py` | pairwise 生成质量评测 |

---

## 注意事项

1. **大文件不上传**：本仓库只包含代码、报告和结果摘要。模型权重、完整数据集、`.jsonl` 训练数据因体积过大已加入 `.gitignore`。
2. **DeepSeek API**：涉及 `build_ipo_data.py`、`build_finance_sft_data.py`、`build_cflue_calc_sft.py` 的脚本需要 `DEEPSEEK_API_KEY` 环境变量。
3. **硬件要求**：训练脚本默认使用 `device_map='auto'` 和 bf16，建议在 ≥24GB 显存的 NVIDIA GPU 上运行。
4. **路径约定**：脚本默认使用相对路径。运行前请确保 `qwen/Qwen2___5-3B-Instruct/`、`fineval/val/`、`ceval/val/`、`cflue_full/` 等目录位于工作目录下。

---

## 许可证

本项目代码部分遵循 MIT 许可证。模型权重、数据集和评测集遵循其各自原始许可证。
