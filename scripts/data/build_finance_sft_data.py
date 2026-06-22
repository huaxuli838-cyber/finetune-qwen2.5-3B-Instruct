import os
import re
import json
import time
import argparse
import random
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ==================== 配置 ====================

SYSTEM_PROMPT = (
    "你是金融领域的数据构造专家。你擅长从高质量金融/经济文本中提炼出开放式、"
    "任务式的专业问答对，用于训练金融大模型。"
)

GEN_PROMPT = """请根据下面这篇金融/经济领域的高质量文本，设计 1-3 个用于训练金融大模型的问答对。

要求：
1. 每个问答对应属于以下某一类型：
   - 概念理解：解释文本中涉及的金融术语、概念、指标或业务含义；
   - 政策解读：解读文中提到的政策、监管规定、法规及其背景与影响；
   - 市场分析：分析文中市场走势、行业趋势、事件影响或竞争格局；
   - 实务应用：结合具体业务场景说明如何操作、尽调、风控、投融资或决策；
   - 计算推理：如果文本包含数据，可设计需要简单计算、比较或推理的问题；
   - 摘要抽取：对文本关键信息、核心观点或数据进行摘要、提炼。
2. 问题（instruction）必须是开放式问答或任务式指令，不能是单项选择题，也不能让模型仅从选项中做选择。
3. 答案（output）必须基于文本内容，专业、完整、条理清晰，必要时分点说明。严禁添加文本中没有依据的内容。
4. 同一篇文本的多个问答对应尽量覆盖不同类型，避免重复。
5. 输出必须且只能是一个 JSON 数组，不要任何额外说明。每个元素格式如下：
[
  {"type": "概念理解", "instruction": "问题文本", "output": "答案文本"},
  ...
]

文本：
{text}
"""

VALID_TYPES = {
    "概念理解", "政策解读", "市场分析", "实务应用", "计算推理", "摘要抽取"
}

# ==================== 工具函数 ====================


def create_deepseek_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def try_parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        # 尝试提取第一个 JSON 数组
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def call_deepseek(
    client: OpenAI,
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_retries: int = 5,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"[API error] {e}, retry in {wait}s... ({attempt+1}/{max_retries})")
            time.sleep(wait)
    return None


def load_existing_cache(cache_path: str) -> Dict[str, Dict]:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    key = item.get("_key")
                    if key:
                        cache[key] = item
                except Exception:
                    continue
    return cache


def append_cache(cache_path: str, item: Dict):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def validate_qa(item: Dict) -> bool:
    if not isinstance(item, dict):
        return False
    t = item.get("type", "").strip()
    instruction = item.get("instruction", "").strip()
    output = item.get("output", "").strip()
    if t not in VALID_TYPES:
        return False
    if len(instruction) < 8 or len(output) < 30:
        return False
    # 过滤掉类似选择题的问法
    if re.search(r"[A-D][\.．、]", instruction) and ("选项" in instruction or "下列" in instruction):
        return False
    return True


def generate_for_passage(
    client: OpenAI,
    passage: Dict,
    cache_path: str,
    existing_cache: Dict[str, Dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Dict:
    key = passage["_key"]
    if key in existing_cache:
        return existing_cache[key]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": GEN_PROMPT.replace("{text}", passage["text"])},
    ]
    content = call_deepseek(
        client,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    record = {
        "_key": key,
        "_id": passage.get("_id"),
        "source_file": passage.get("source_file"),
        "raw_text_len": passage.get("raw_text_len"),
        "content": content,
        "pairs": [],
        "status": "success" if content else "api_failed",
    }

    if content:
        parsed = try_parse_json(content)
        if isinstance(parsed, list):
            valid_pairs = [p for p in parsed if validate_qa(p)]
            record["pairs"] = valid_pairs
            if not valid_pairs:
                record["status"] = "no_valid_pairs"
        else:
            record["status"] = "parse_failed"

    append_cache(cache_path, record)
    existing_cache[key] = record
    return record


def sample_passages(
    data_dir: str,
    n_passages: int,
    min_len: int,
    max_len: int,
    min_quality: float,
    max_text_chars: int,
    random_state: int = 42,
) -> List[Dict]:
    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise ValueError(f"在 {data_dir} 下未找到 parquet 文件")

    # 先快速统计总行数
    total_rows = 0
    row_counts = []
    print("统计各文件行数...")
    for f in files:
        df = pd.read_parquet(f, columns=["_id"])
        row_counts.append(len(df))
        total_rows += len(df)
    print(f"总行数: {total_rows}")

    passages = []
    for f, n_rows in zip(files, row_counts):
        need = int(n_passages * (n_rows / total_rows)) + 50  # 多采一些，过滤后补足
        df = pd.read_parquet(f, columns=["text", "_id", "quality_score"])
        # 过滤
        mask = (
            df["text"].str.len().between(min_len, max_len)
            & (df["quality_score"] >= min_quality)
        )
        df = df[mask]
        if len(df) == 0:
            continue
        sample_n = min(need, len(df))
        sampled = df.sample(n=sample_n, random_state=random_state)
        for _, row in sampled.iterrows():
            text = row["text"][:max_text_chars].strip()
            passages.append({
                "_id": int(row["_id"]),
                "source_file": os.path.basename(f),
                "text": text,
                "raw_text_len": len(row["text"]),
                "_key": f"{os.path.basename(f)}_{int(row['_id'])}",
            })

    random.seed(random_state)
    random.shuffle(passages)
    passages = passages[:n_passages]
    print(f"最终采样 passages: {len(passages)}")
    return passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="./BAAI_IndustryCorpus2_finance_high/finance_economics/chinese/high")
    parser.add_argument("--output_dir", type=str, default="./finance_sft_v1")
    parser.add_argument("--n_passages", type=int, default=5000,
                        help="采样的原文档数，每个文档期望生成 1-3 个问答对")
    parser.add_argument("--min_len", type=int, default=500)
    parser.add_argument("--max_len", type=int, default=6000)
    parser.add_argument("--max_text_chars", type=int, default=4500,
                        help="传入模型的最大原文长度")
    parser.add_argument("--min_quality", type=float, default=4.05)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test", action="store_true",
                        help="只跑 3 条测试，验证 prompt 与输出格式")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, "qa_generation_cache.jsonl")

    # 采样原文
    passages = sample_passages(
        args.data_dir,
        n_passages=3 if args.test else args.n_passages,
        min_len=args.min_len,
        max_len=args.max_len,
        min_quality=args.min_quality,
        max_text_chars=args.max_text_chars,
        random_state=args.random_state,
    )

    # 加载已有缓存
    existing_cache = load_existing_cache(cache_path)
    print(f"已有缓存记录: {len(existing_cache)}")

    client = create_deepseek_client()

    # 生成
    records = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                generate_for_passage,
                client,
                p,
                cache_path,
                existing_cache,
                args.temperature,
                args.max_tokens,
            ): p
            for p in passages
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="生成问答对"):
            try:
                records.append(future.result())
            except Exception as e:
                print(f"[worker error] {e}")

    # 汇总、去重、输出
    all_pairs = []
    type_counter = {}
    status_counter = {}
    seen_instructions = set()

    for rec in records:
        status_counter[rec.get("status", "unknown")] = status_counter.get(rec.get("status", "unknown"), 0) + 1
        for p in rec.get("pairs", []):
            instr = p["instruction"].strip()
            if instr in seen_instructions:
                continue
            seen_instructions.add(instr)
            pair = {
                "instruction": instr,
                "output": p["output"].strip(),
                "category": p["type"].strip(),
                "source_id": rec.get("_id"),
                "source_file": rec.get("source_file"),
            }
            all_pairs.append(pair)
            type_counter[pair["category"]] = type_counter.get(pair["category"], 0) + 1

    # 划分训练/验证集
    random.seed(args.random_state)
    random.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * 0.05))
    train = all_pairs[n_val:]
    val = all_pairs[:n_val]

    # 保存
    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(all_pairs, os.path.join(args.output_dir, "finance_sft_all.jsonl"))
    save_jsonl(train, os.path.join(args.output_dir, "finance_sft_train.jsonl"))
    save_jsonl(val, os.path.join(args.output_dir, "finance_sft_val.jsonl"))

    # 报告
    report = {
        "n_passages": len(passages),
        "status_distribution": status_counter,
        "category_distribution": type_counter,
        "n_valid_pairs": len(all_pairs),
        "n_train": len(train),
        "n_val": len(val),
    }
    with open(os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n========== 生成报告 ==========")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"数据已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
