import os
import re
import json
import time
import argparse
import random
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import torch
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel

# ==================== 配置 ====================

SYSTEM_PROMPT = "你是一个金融专业知识助手，擅长解答金融资格考试相关的单项选择题。"

SCORE_PROMPT = """你是一位资深金融考试阅卷专家。下面是一道单项选择题，以及 {n_candidates} 个考生答案。
请对每个考生答案进行 1-5 分的综合质量评分。评分只看答案本身的生成质量，**不看最终选项是否正确**，重点考察：
- 专业性：术语准确、符合金融领域表达；
- 推理清晰度：逻辑是否连贯、有条理；
- 格式规范性：是否包含“答案：X”和“解析：...”等明确结构；
- 完整性：解析是否充分，不是过于简略；
- 语言流畅度：没有语法错误、重复、混乱。

评分标准：
5 = 优秀：解析专业、逻辑清晰、格式规范、术语准确、内容完整；
4 = 良好：整体质量较高，但略有瑕疵；
3 = 及格：基本可读，但专业性或清晰度一般，或格式不够规范；
2 = 较差：解析有明显错误、逻辑混乱或过于简略；
1 = 很差：答非所问、格式严重不规范或内容混乱。

题目：
{question}

考生答案：
{candidates_text}

请严格按以下 JSON 数组格式输出（不要输出任何额外说明）：
[
  {{"id": "0", "score": 整数, "professionalism": 整数(1-5), "clarity": 整数(1-5), "format": 整数(1-5), "completeness": 整数(1-5), "reason": "不超过30字的简要理由"}},
  ...
]"""

# ==================== 工具函数 ====================


def load_tokenizer(model_path: str):
    print(f"加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(base_model_path: str, sft_adapter_path: str):
    print(f"加载基础模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    print(f"加载 SFT LoRA adapter: {sft_adapter_path}")
    model = PeftModel.from_pretrained(model, sft_adapter_path)
    model.eval()
    return model


def build_chat_prompt(instruction: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_answer(text: str) -> str:
    text = text.strip()
    answer_markers = ['答案：', '答案是：', '正确答案是：', '答案:', '答案是:', '正确答案是:', '答案选']
    last_answer_text = None
    for marker in answer_markers:
        if marker in text:
            last_answer_text = text.rsplit(marker, 1)[-1]

    search_text = last_answer_text if last_answer_text is not None else text
    search_text = search_text.strip()
    search_text = re.sub(r'^(答案|答案选|正确答案是|选|我认为|选项|是)[:：]?\s*', '', search_text)
    search_text = search_text.strip()

    match = re.search(r'^[A-D]+', search_text.upper())
    if match:
        return match.group(0)
    match = re.search(r'\b([A-D]+)\b', search_text.upper())
    if match:
        return match.group(1)
    return search_text[:5].upper()


def has_reasonable_explanation(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return False
    return '解析' in text or '因为' in text or '因此' in text or '故' in text or '所以' in text


def load_cflue_data(path: str) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            answer = str(item.get('answer', '')).strip().upper()
            if answer and answer in 'ABCD':
                data.append(item)
    print(f"加载 CFLUE 单选题总数: {len(data)}")
    return data


def load_cache(cache_path: str) -> Dict:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    cache[item.get('id')] = item
                except Exception:
                    continue
        print(f"加载缓存: {cache_path} ({len(cache)} 条)")
    return cache


def append_cache(cache_path: str, items: List[Dict]):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'a', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ==================== 阶段 1：用 SFT 模型自采样回答 ====================


def generate_responses(model, tokenizer, data: List[Dict],
                       cache_path: str, n_samples: int = 4,
                       temperature: float = 0.8, top_p: float = 0.9,
                       max_new_tokens: int = 512,
                       batch_size: int = 8) -> Dict[str, Dict]:
    cache = load_cache(cache_path)
    todo = [item for item in data if item['id'] not in cache]
    print(f"需要生成回答的题目数: {len(todo)}")
    if not todo:
        return cache

    device = next(model.parameters()).device
    results = []

    for batch_start in tqdm(range(0, len(todo), batch_size), desc="生成模型回答"):
        batch_items = todo[batch_start:batch_start + batch_size]
        prompt_texts = [
            build_chat_prompt(item['instruction'], tokenizer)
            for item in batch_items
        ]
        ground_truths = [item['answer'].strip().upper() for item in batch_items]
        qids = [item['id'] for item in batch_items]

        per_item_texts = [[] for _ in batch_items]
        for sample_idx in range(n_samples):
            inputs = tokenizer(
                prompt_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024,
                add_special_tokens=False,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs['input_ids'].shape[1]

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                success = True
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                success = False

            if success:
                for i in range(len(batch_items)):
                    response_ids = outputs[i][input_len:]
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                    per_item_texts[i].append(response_text)
            else:
                for i, prompt_text in enumerate(prompt_texts):
                    single_inputs = tokenizer(
                        prompt_text,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=1024,
                        add_special_tokens=False,
                    )
                    single_inputs = {k: v.to(device) for k, v in single_inputs.items()}
                    single_input_len = single_inputs['input_ids'].shape[1]
                    try:
                        with torch.no_grad():
                            single_output = model.generate(
                                **single_inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=temperature,
                                top_p=top_p,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                        response_ids = single_output[0][single_input_len:]
                        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        response_text = ''
                    per_item_texts[i].append(response_text)

        for i, item in enumerate(batch_items):
            qid = qids[i]
            candidates = []
            texts = per_item_texts[i]
            for j in range(n_samples):
                response_text = texts[j] if j < len(texts) else ''
                pred = extract_answer(response_text)
                candidates.append({
                    'candidate_id': str(j),
                    'text': response_text,
                    'pred': pred,
                    'is_correct': (pred == ground_truths[i]),
                    'has_explanation': has_reasonable_explanation(response_text),
                })

            result = {
                'id': qid,
                'instruction': item['instruction'],
                'ground_truth': ground_truths[i],
                'candidates': candidates,
                'prompt_text': prompt_texts[i],
            }
            results.append(result)
            cache[qid] = result

        if len(results) % 100 == 0:
            append_cache(cache_path, results[-100:])
            torch.cuda.empty_cache()

    written_ids = set()
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                written_ids.add(json.loads(line).get('id'))
    remaining = [r for r in results if r['id'] not in written_ids]
    if remaining:
        append_cache(cache_path, remaining)

    print(f"完成 {len(cache)} 题的模型回答采样")
    return cache


# ==================== 阶段 2：DeepSeek 评分 ====================


def create_deepseek_client():
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def try_parse_json(text: str):
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def call_deepseek(client: OpenAI, messages: List[Dict], temperature: float = 0.0,
                  max_tokens: int = 1024, max_retries: int = 3) -> str:
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
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                return f"[API_ERROR: {e}]"
    return "[API_ERROR]"


def build_candidates_text(candidates: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(candidates):
        parts.append(f"[{i}]\n{c['text']}\n")
    return '\n'.join(parts)


def score_question(client: OpenAI, item: Dict) -> Dict:
    qid = item['id']
    candidates = item['candidates']
    candidates_text = build_candidates_text(candidates)
    prompt = SCORE_PROMPT.format(
        n_candidates=len(candidates),
        question=item['instruction'],
        candidates_text=candidates_text,
    )
    messages = [
        {"role": "system", "content": "你是一位严格、公正的金融考试阅卷专家。"},
        {"role": "user", "content": prompt},
    ]
    content = call_deepseek(client, messages, temperature=0.0, max_tokens=1024)
    parsed = try_parse_json(content)

    scores = []
    if isinstance(parsed, list):
        for entry in parsed:
            try:
                scores.append({
                    'candidate_id': str(entry.get('id', '')),
                    'score': int(entry.get('score', 3)),
                    'professionalism': int(entry.get('professionalism', 3)),
                    'clarity': int(entry.get('clarity', 3)),
                    'format': int(entry.get('format', 3)),
                    'completeness': int(entry.get('completeness', 3)),
                    'reason': str(entry.get('reason', '')),
                })
            except (ValueError, TypeError):
                continue

    if len(scores) != len(candidates):
        scores = []
        for c in candidates:
            text = c['text']
            has_format = '答案' in text and '解析' in text
            has_reasoning = has_reasonable_explanation(text)
            text_len = len(text.strip())
            if has_format and has_reasoning and text_len >= 80:
                local_score = 5
            elif has_format and has_reasoning and text_len >= 50:
                local_score = 4
            elif has_format or has_reasoning:
                local_score = 3
            elif text_len >= 20:
                local_score = 2
            else:
                local_score = 1
            scores.append({
                'candidate_id': c['candidate_id'],
                'score': local_score,
                'professionalism': 3,
                'clarity': 3,
                'format': 5 if has_format else 1,
                'completeness': 3,
                'reason': 'API解析失败，使用本地规则回退',
            })

    return {
        'id': qid,
        'scores': scores,
        'raw_response': content,
    }


def score_responses(response_cache: Dict[str, Dict], client: OpenAI,
                    cache_path: str, max_workers: int = 10) -> Dict[str, Dict]:
    cache = load_cache(cache_path)
    todo = [item for qid, item in response_cache.items() if qid not in cache]
    print(f"需要 DeepSeek 评分的题目数: {len(todo)}")
    if not todo:
        return cache

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(score_question, client, item): item for item in todo}
        for future in tqdm(as_completed(future_to_item), total=len(todo), desc="DeepSeek 评分"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"评分异常: {e}")

            if len(results) % 50 == 0:
                append_cache(cache_path, results[-50:])

    written_ids = set()
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                written_ids.add(json.loads(line).get('id'))
    remaining = [r for r in results if r['id'] not in written_ids]
    if remaining:
        append_cache(cache_path, remaining)

    for r in results:
        cache[r['id']] = r
    print(f"完成 {len(cache)} 题的评分")
    return cache


# ==================== 阶段 3：构造 DPO pairs ====================


def construct_pairs(response_cache: Dict[str, Dict], score_cache: Dict[str, Dict],
                    tokenizer, min_gap: int = 1,
                    min_chosen_score: int = 3,
                    max_rejected_score: int = 4) -> List[Dict]:
    pairs = []
    stats = Counter()

    for qid, item in response_cache.items():
        if qid not in score_cache:
            stats['missing_score'] += 1
            continue

        candidates = item['candidates']
        scores = score_cache[qid]['scores']
        if len(scores) != len(candidates):
            stats['score_count_mismatch'] += 1
            continue

        scored = []
        for c, s in zip(candidates, scores):
            scored.append({**c, **s})

        scored_sorted = sorted(scored, key=lambda x: x['score'], reverse=True)
        chosen_list = [x for x in scored_sorted if x['score'] >= min_chosen_score]
        rejected_list = [x for x in scored_sorted if x['score'] <= max_rejected_score]

        if not chosen_list or not rejected_list:
            stats['no_valid_pair'] += 1
            continue

        best = chosen_list[0]
        added_for_q = 0
        for cand in scored_sorted[1:]:
            if best['score'] - cand['score'] < min_gap:
                continue
            if best['text'].strip() == cand['text'].strip():
                stats['identical_skip'] += 1
                continue
            pairs.append({
                'prompt': item['prompt_text'],
                'chosen': best['text'],
                'rejected': cand['text'],
                'source_question': qid,
                'ground_truth': item['ground_truth'],
                'chosen_score': best['score'],
                'chosen_is_correct': best['is_correct'],
                'rejected_score': cand['score'],
                'rejected_is_correct': cand['is_correct'],
                'chosen_reason': best.get('reason', ''),
                'rejected_reason': cand.get('reason', ''),
            })
            added_for_q += 1
            stats['pair_added'] += 1

        if added_for_q == 0:
            stats['gap_too_small'] += 1

    print("\nPair 构造统计:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  最终有效 pairs: {len(pairs)}")
    return pairs


# ==================== 主流程 ====================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./cflue_sft_final/cflue_single_choice_all.jsonl')
    parser.add_argument('--base_model_path', type=str, default='./qwen/Qwen2___5-3B-Instruct')
    parser.add_argument('--sft_adapter_path', type=str, default='./qwen_cflue_lora/checkpoint-1896')
    parser.add_argument('--output', type=str, default='./cflue_dpo_data_v8.jsonl')
    parser.add_argument('--n_total', type=int, default=7000, help='总题目数')
    parser.add_argument('--existing_cache_dir', type=str, default='./dpo_v7_cache', help='复用已有缓存目录')
    parser.add_argument('--new_cache_dir', type=str, default='./dpo_v8_cache')
    parser.add_argument('--n_candidates', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_workers', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_generation', action='store_true')
    parser.add_argument('--skip_scoring', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.new_cache_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.base_model_path)
    all_data = load_cflue_data(args.input_path)

    # 1. 加载已有缓存
    existing_response_cache_path = os.path.join(args.existing_cache_dir, 'responses.jsonl')
    existing_score_cache_path = os.path.join(args.existing_cache_dir, 'scores.jsonl')

    existing_response_cache = load_cache(existing_response_cache_path)
    existing_score_cache = load_cache(existing_score_cache_path)
    print(f"复用已有 response: {len(existing_response_cache)} 题, score: {len(existing_score_cache)} 题")

    existing_qids = set(existing_response_cache.keys())

    # 2. 从剩余题目中采样新的题目
    remaining_data = [d for d in all_data if d['id'] not in existing_qids]
    n_new_needed = max(0, args.n_total - len(existing_response_cache))
    print(f"已有 {len(existing_response_cache)} 题，还需新增 {n_new_needed} 题，剩余可选 {len(remaining_data)} 题")

    if n_new_needed > len(remaining_data):
        raise ValueError(f"剩余题目不足：需要 {n_new_needed}，仅剩 {len(remaining_data)}")

    random.seed(args.seed)
    new_data = random.sample(remaining_data, n_new_needed)

    # 合并
    combined_response_cache = dict(existing_response_cache)
    combined_score_cache = dict(existing_score_cache)

    new_response_cache_path = os.path.join(args.new_cache_dir, 'responses.jsonl')
    new_score_cache_path = os.path.join(args.new_cache_dir, 'scores.jsonl')

    # 3. 生成新增题目的回答
    if not args.skip_generation:
        model = load_model(args.base_model_path, args.sft_adapter_path)
        new_response_cache = generate_responses(
            model, tokenizer, new_data,
            new_response_cache_path,
            n_samples=args.n_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        combined_response_cache.update(new_response_cache)
        del model
        torch.cuda.empty_cache()
    else:
        new_response_cache = load_cache(new_response_cache_path)
        combined_response_cache.update(new_response_cache)

    # 4. 评分（已有评分会自动跳过）
    if not args.skip_scoring:
        client = create_deepseek_client()
        new_score_cache = score_responses(combined_response_cache, client, new_score_cache_path, args.max_workers)
        combined_score_cache.update(new_score_cache)
    else:
        new_score_cache = load_cache(new_score_cache_path)
        combined_score_cache.update(new_score_cache)

    print(f"\n合并后总题目数: {len(combined_response_cache)}")
    print(f"合并后总评分题数: {len(combined_score_cache)}")

    # 5. 构造 pairs
    pairs = construct_pairs(combined_response_cache, combined_score_cache, tokenizer)

    print(f"\n保存 {len(pairs)} 条 DPO v8 数据到 {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print("完成！")


if __name__ == '__main__':
    main()
