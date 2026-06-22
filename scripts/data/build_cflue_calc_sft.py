import os
import re
import json
import time
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import List, Dict, Optional

from openai import OpenAI
from tqdm import tqdm

# ==================== 配置 ====================

SYSTEM_PROMPT = (
    "你是金融与财会领域的教育专家。你擅长把单项选择题改写成开放式的计算/推理问答，"
    "用于训练金融大模型解决实际计算问题。"
)

REWRITE_PROMPT = """下面是一道金融/财会领域的单项选择题，包含题干、选项和解析。
请你把它改写成一个**开放式的计算/推理问答对**，要求：
1. 问题中**不要保留 A/B/C/D 选项**，也不要让模型只能从选项中选择；
2. 问题必须明确要求“计算”“求”“推导”或“判断并说明理由”，突出数值计算或逻辑推理过程；
3. 答案必须包含**清晰的计算/推导步骤**和**最终结果**；
4. 语言专业、简洁，保持原题的金融/财会含义不变；
5. 输出必须且只能是一个 JSON 对象，不要任何额外说明：
{{"instruction": "改写后的问题", "output": "改写后的分步解答"}}

原题：
{question}

正确答案与解析：
{analysis}
"""

CALC_KEYWORDS = [
    '计算', '净额', '等于', '=','＝','+','－','×','÷','/','*','-',
    '利率', '收益率', '折现', '终值', '现值', '年金', '复利', '单利',
    '市盈率', '市净率', '股息率', '费用率', '成本率', '利润率', '增长率',
    '折旧', '摊销', '资本化', '费用化', '名义利率', '实际利率', '到期收益率',
    'YTM', 'NPV', 'IRR'
]


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
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def call_deepseek(
    client: OpenAI,
    messages: List[Dict],
    temperature: float = 0.3,
    max_tokens: int = 1024,
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


def load_cache(cache_path: str) -> Dict[str, Dict]:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    key = item.get("id")
                    if key:
                        cache[key] = item
                except Exception:
                    continue
    return cache


def append_cache(cache_path: str, item: Dict):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def is_calc_question(item: Dict) -> bool:
    inst = item.get("instruction", "")
    out = item.get("output", "")
    choices = item.get("choices", {})
    if "<img" in inst or "src=" in inst:
        return False
    text = inst + " " + " ".join(str(v) for v in choices.values())
    has_num = bool(re.search(r"\d+\.?\d*", text))
    has_calc = any(kw in out or kw in inst for kw in CALC_KEYWORDS)
    has_operator = bool(re.search(r"[=＝+\-×÷*/．\.]", out))
    return has_num and has_calc and has_operator


def extract_question_and_analysis(item: Dict) -> Dict:
    inst = item.get("instruction", "")
    # 去掉开头的“请回答以下单项选择题，并给出简要解析。”
    inst = re.sub(r"^请回答以下单项选择题，并给出简要解析。\s*\n+题目：", "", inst).strip()
    output = item.get("output", "").strip()
    return {"question": inst, "analysis": output}


def validate_rewrite(obj: Dict) -> bool:
    if not isinstance(obj, dict):
        return False
    instruction = obj.get("instruction", "").strip()
    output = obj.get("output", "").strip()
    if len(instruction) < 10 or len(output) < 30:
        return False
    # 检查是否仍含 A/B/C/D 选项模式
    if re.search(r"\n[A-D][\.．、]", instruction):
        return False
    if re.search(r"[A-D][\.．、]", instruction[:200]) and ("选项" in instruction or "下列" in instruction):
        return False
    return True


def rewrite_one(
    client: OpenAI,
    item: Dict,
    cache_path: str,
    existing_cache: Dict[str, Dict],
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Dict:
    qid = item.get("id")
    if qid in existing_cache:
        return existing_cache[qid]

    qa = extract_question_and_analysis(item)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": REWRITE_PROMPT.format(
            question=qa["question"],
            analysis=qa["analysis"],
        )},
    ]
    content = call_deepseek(client, messages, temperature=temperature, max_tokens=max_tokens)

    record = {
        "id": qid,
        "subject": item.get("subject"),
        "certification": item.get("certification"),
        "original_question": qa["question"],
        "original_analysis": qa["analysis"],
        "raw": content,
        "rewrite": None,
        "status": "success" if content else "api_failed",
    }

    if content:
        parsed = try_parse_json(content)
        if validate_rewrite(parsed):
            record["rewrite"] = {
                "instruction": parsed["instruction"].strip(),
                "output": parsed["output"].strip(),
            }
            record["status"] = "success"
        else:
            record["status"] = "parse_failed"

    append_cache(cache_path, record)
    existing_cache[qid] = record
    return record


def select_calc_items(data_path: str, n_target: int, cap_per_subject: int, random_state: int = 42) -> List[Dict]:
    items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if is_calc_question(item):
                items.append(item)
    random.seed(random_state)
    random.shuffle(items)

    selected = []
    remain = []
    subj_counts = Counter()
    for it in items:
        s = it.get("subject", "")
        if subj_counts[s] < cap_per_subject:
            selected.append(it)
            subj_counts[s] += 1
        else:
            remain.append(it)

    if len(selected) < n_target:
        selected.extend(remain[: n_target - len(selected)])

    return selected[:n_target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cflue_path", type=str, default="./cflue_sft/cflue_single_choice_all.jsonl")
    parser.add_argument("--output_dir", type=str, default="./finance_sft")
    parser.add_argument("--n_target", type=int, default=1800)
    parser.add_argument("--cap_per_subject", type=int, default=200)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, "cflue_calc_rewrite_cache.jsonl")

    # 选择计算题
    selected = select_calc_items(
        args.cflue_path,
        n_target=3 if args.test else args.n_target,
        cap_per_subject=args.cap_per_subject,
        random_state=args.random_state,
    )
    print(f"选中计算题数量: {len(selected)}")

    # 学科分布
    subj_counts = Counter([it.get("subject", "") for it in selected])
    print("学科分布（前20）:", subj_counts.most_common(20))

    existing_cache = load_cache(cache_path)
    print(f"已有缓存: {len(existing_cache)}")

    client = create_deepseek_client()

    records = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                rewrite_one,
                client,
                it,
                cache_path,
                existing_cache,
                args.temperature,
                args.max_tokens,
            ): it
            for it in selected
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="改写计算题"):
            try:
                records.append(future.result())
            except Exception as e:
                print(f"[worker error] {e}")

    # 收集有效改写
    valid_pairs = []
    seen_instructions = set()
    for rec in records:
        if rec.get("status") != "success" or not rec.get("rewrite"):
            continue
        inst = rec["rewrite"]["instruction"]
        out = rec["rewrite"]["output"]
        if inst in seen_instructions:
            continue
        seen_instructions.add(inst)
        valid_pairs.append({
            "instruction": inst,
            "output": out,
            "category": "计算推理",
            "source_id": rec.get("id"),
            "source_subject": rec.get("subject"),
            "source_certification": rec.get("certification"),
        })

    print(f"有效改写数量: {len(valid_pairs)}")

    # 保存改写结果
    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(valid_pairs, os.path.join(args.output_dir, "finance_sft_calc_rewritten.jsonl"))

    # 合并到已有数据（如果存在）
    all_path = os.path.join(args.output_dir, "finance_sft_all.jsonl")
    if os.path.exists(all_path):
        existing = []
        with open(all_path, "r", encoding="utf-8") as f:
            for line in f:
                existing.append(json.loads(line))
        combined = existing + valid_pairs
        random.seed(args.random_state)
        random.shuffle(combined)
        n_val = max(1, int(len(combined) * 0.05))
        train = combined[n_val:]
        val = combined[:n_val]
        save_jsonl(combined, os.path.join(args.output_dir, "finance_sft_all.jsonl"))
        save_jsonl(train, os.path.join(args.output_dir, "finance_sft_train.jsonl"))
        save_jsonl(val, os.path.join(args.output_dir, "finance_sft_val.jsonl"))
        print(f"已合并: 总计 {len(combined)}, 训练 {len(train)}, 验证 {len(val)}")
    else:
        random.seed(args.random_state)
        random.shuffle(valid_pairs)
        n_val = max(1, int(len(valid_pairs) * 0.05))
        train = valid_pairs[n_val:]
        val = valid_pairs[:n_val]
        save_jsonl(valid_pairs, os.path.join(args.output_dir, "finance_sft_all.jsonl"))
        save_jsonl(train, os.path.join(args.output_dir, "finance_sft_train.jsonl"))
        save_jsonl(val, os.path.join(args.output_dir, "finance_sft_val.jsonl"))

    # 报告
    report = {
        "n_selected": len(selected),
        "n_rewritten_success": sum(1 for r in records if r.get("status") == "success"),
        "n_parse_failed": sum(1 for r in records if r.get("status") == "parse_failed"),
        "n_api_failed": sum(1 for r in records if r.get("status") == "api_failed"),
        "n_valid_pairs": len(valid_pairs),
        "subject_distribution": dict(subj_counts.most_common()),
    }
    with open(os.path.join(args.output_dir, "cflue_calc_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
