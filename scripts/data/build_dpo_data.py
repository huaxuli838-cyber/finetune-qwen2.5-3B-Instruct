import os
import re
import json
import time
import argparse
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from openai import OpenAI


# ==================== 配置 ====================

GENERATION_CONFIGS = [
    {"name": "greedy", "do_sample": False, "temperature": 1.0, "top_p": 1.0},
    {"name": "temp0.7", "do_sample": True, "temperature": 0.7, "top_p": 0.9},
    {"name": "temp0.9", "do_sample": True, "temperature": 0.9, "top_p": 0.9},
    {"name": "temp1.1", "do_sample": True, "temperature": 1.1, "top_p": 0.95},
]


# ==================== 数据加载 ====================

def load_cflue_data(path: str, n_samples: int, seed: int = 42) -> List[Dict]:
    """加载 CFLUE 单选题数据并随机采样"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 只保留有明确标准答案的
            answer = item.get('answer', '').strip().upper()
            if answer and answer in 'ABCD':
                data.append(item)
    
    print(f"加载 CFLUE 单选题总数: {len(data)}")
    
    if 0 < n_samples < len(data):
        random.seed(seed)
        data = random.sample(data, n_samples)
        print(f"随机采样: {n_samples} 题")
    else:
        print(f"使用全部 {len(data)} 题")
    
    return data


# ==================== Prompt 构造 ====================

def build_chat_prompt(item: Dict, tokenizer) -> str:
    """构造与 SFT 训练一致的 chat prompt（不带 assistant 回复）"""
    system = item.get('system', '你是一个金融专业知识助手，擅长解答金融资格考试相关的单项选择题。')
    instruction = item.get('instruction', '')
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# ==================== 答案抽取 ====================

def extract_answer(text: str) -> str:
    """从模型输出中抽取答案选项 A-D"""
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


# ==================== 模型生成 ====================

def load_model_and_tokenizer(base_model_path: str, adapter_path: str):
    """加载基础模型 + SFT LoRA adapter"""
    print(f"加载 tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"加载基础模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    
    print(f"加载 LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, tokenizer


@torch.no_grad()
def generate_answers(
    model,
    tokenizer,
    prompts: List[str],
    gen_config: Dict,
    batch_size: int = 16,
    max_new_tokens: int = 384,
) -> List[str]:
    """为所有 prompt 批量生成答案"""
    generation_config = GenerationConfig(
        do_sample=gen_config["do_sample"],
        temperature=gen_config.get("temperature", 1.0),
        top_p=gen_config.get("top_p", 1.0),
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    all_outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"生成 [{gen_config['name']}]"):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )
        
        input_seq_len = inputs['input_ids'].shape[1]
        for j, output in enumerate(outputs):
            # 关键：使用完整输入序列长度（含 padding）来切片，避免 left padding 下切错位置
            generated = output[input_seq_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            all_outputs.append(text)
    
    return all_outputs


# ==================== DeepSeek Judge ====================

def create_deepseek_client():
    """创建 DeepSeek client，从环境变量读取 API key"""
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_judge(client: OpenAI, question: str, ground_truth: str, ans_a: str, ans_b: str, model: str = "deepseek-chat", max_retries: int = 3) -> Tuple[str, str, int]:
    """
    调用 DeepSeek 比较两个答案哪个更好，带重试机制，同时返回置信度。
    返回: (better_choice: 'A'/'B', reason: str, confidence: int 1-5)
    """
    prompt = f"""你是一位金融专业考试阅卷专家。请比较下面两个答案，判断哪个更好。

题目：
{question}

正确答案：{ground_truth}

=== 答案 A ===
{ans_a}

=== 答案 B ===
{ans_b}

请从以下维度评估：
1. 最终答案选项是否正确
2. 解析是否清晰、专业、有逻辑
3. 整体回答是否符合金融专业考试要求

请直接输出 JSON 格式，不要输出其他内容：
{{"better": "A" 或 "B", "confidence": 1-5 的整数（1=非常不确定，5=非常确定）, "reason": "简要理由（不超过50字）"}}"""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            content = response.choices[0].message.content.strip()
            
            # 尝试解析 JSON
            # 先去掉可能的 markdown 代码块
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            
            result = json.loads(content)
            better = result.get('better', '').strip().upper()
            reason = result.get('reason', '').strip()
            confidence = result.get('confidence', 3)
            try:
                confidence = int(confidence)
                confidence = max(1, min(5, confidence))
            except (ValueError, TypeError):
                confidence = 3
            
            if better in ['A', 'B']:
                return better, reason, confidence
            else:
                return 'A', f"解析失败，默认选A。原始输出：{content[:100]}", 1
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return 'A', f"API 调用失败：{str(e)}", 1


# ==================== Pair 构造 ====================

def construct_pairs(
    data: List[Dict],
    chat_prompts: List[str],
    generated: Dict[str, List[str]],
    client: OpenAI,
    use_judge: bool = True,
    max_workers: int = 16,
    min_judge_confidence: int = 3,
) -> List[Dict]:
    """
    根据生成结果构造 DPO preference pairs。
    chat_prompts: 与 SFT 一致的 chat-format prompt（含 add_generation_prompt）
    generated: {config_name: [response_text, ...]}
    """
    # 第一步：分类所有答案，收集需要 judge 的 case
    pre_results = []
    judge_tasks = []
    
    for idx, item in enumerate(tqdm(data, desc="预分类答案")):
        ground_truth = item['answer'].strip().upper()
        instruction = item['instruction']
        prompt_text = chat_prompts[idx]
        
        # 收集所有生成答案
        responses = []
        for cfg in GENERATION_CONFIGS:
            resp = generated[cfg['name']][idx]
            pred = extract_answer(resp)
            is_correct = (pred == ground_truth)
            responses.append({
                'text': resp,
                'pred': pred,
                'correct': is_correct,
                'config': cfg['name'],
            })
        
        correct_resps = [r for r in responses if r['correct']]
        wrong_resps = [r for r in responses if not r['correct']]
        
        # 情况 1：有对有错，直接用 ground truth 构造 pair
        if correct_resps and wrong_resps:
            pre_results.append({
                'idx': idx,
                'prompt_text': prompt_text,
                'source_question': item.get('id', f'q_{idx}'),
                'ground_truth': ground_truth,
                'chosen': correct_resps[0]['text'],
                'rejected': wrong_resps[0]['text'],
                'pair_type': 'correct_vs_wrong',
                'need_judge': False,
            })
        # 情况 2：全对或全错，需要 judge
        elif use_judge and len(responses) >= 2:
            ans_a = responses[0]['text']
            ans_b = responses[-1]['text']
            pre_results.append({
                'idx': idx,
                'prompt_text': prompt_text,
                'source_question': item.get('id', f'q_{idx}'),
                'ground_truth': ground_truth,
                'ans_a': ans_a,
                'ans_b': ans_b,
                'need_judge': True,
            })
            judge_tasks.append((idx, instruction, ground_truth, ans_a, ans_b))
        else:
            pre_results.append({
                'idx': idx,
                'need_judge': False,
                'skip': True,
            })
    
    # 第二步：并行调用 judge
    judge_results = {}
    if judge_tasks and use_judge:
        print(f"\n并行调用 DeepSeek judge: {len(judge_tasks)} 个任务，workers={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, instruction, ground_truth, ans_a, ans_b in judge_tasks:
                future = executor.submit(
                    call_judge, client, instruction, ground_truth, ans_a, ans_b
                )
                future_to_idx[future] = idx
            
            for future in tqdm(as_completed(future_to_idx), total=len(judge_tasks), desc="Judge 评分"):
                idx = future_to_idx[future]
                better, reason, confidence = future.result()
                judge_results[idx] = (better, reason, confidence)
    
    # 第三步：组装最终 pairs
    pairs = []
    gt_count = 0
    judge_count = 0
    skip_count = 0
    
    for res in pre_results:
        if res.get('skip', False):
            skip_count += 1
            continue
        
        if not res['need_judge']:
            chosen = res['chosen']
            rejected = res['rejected']
            pair_type = res['pair_type']
            gt_count += 1
        else:
            idx = res['idx']
            better, reason, confidence = judge_results.get(idx, ('A', '未知', 1))
            judge_count += 1
            
            # 过滤低置信度 judge pair
            if confidence < min_judge_confidence:
                skip_count += 1
                continue
            
            if better == 'A':
                chosen, rejected = res['ans_a'], res['ans_b']
            else:
                chosen, rejected = res['ans_b'], res['ans_a']
            pair_type = f'judge_conf{confidence}_{reason}'
        
        # 过滤 chosen 和 rejected 相同的情况
        if chosen.strip() == rejected.strip():
            skip_count += 1
            continue
        
        pairs.append({
            'prompt': res['prompt_text'],
            'chosen': chosen,
            'rejected': rejected,
            'source_question': res['source_question'],
            'ground_truth': res['ground_truth'],
            'pair_type': pair_type,
        })
    
    print(f"\nPair 构造统计:")
    print(f"  Ground truth pairs: {gt_count}")
    print(f"  Judge pairs: {judge_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Total valid pairs: {len(pairs)}")
    
    return pairs


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./cflue_sft_final/cflue_single_choice_all.jsonl')
    parser.add_argument('--base_model_path', type=str, default='./qwen/Qwen2___5-3B-Instruct')
    parser.add_argument('--adapter_path', type=str, default='./qwen_cflue_lora/checkpoint-1896')
    parser.add_argument('--n_samples', type=int, default=11000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_new_tokens', type=int, default=384)
    parser.add_argument('--output', type=str, default='./cflue_dpo_data.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_judge', action='store_true', help='不使用 DeepSeek judge')
    parser.add_argument('--max_workers', type=int, default=16, help='DeepSeek judge 并发数')
    parser.add_argument('--min_judge_confidence', type=int, default=3, help='Judge 置信度阈值（1-5），低于该值的 judge pair 会被过滤')
    args = parser.parse_args()
    
    # 加载数据
    data = load_cflue_data(args.data_path, args.n_samples, args.seed)
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.adapter_path)
    
    # 构造所有 prompt
    print("构造 prompts...")
    prompts = [build_chat_prompt(item, tokenizer) for item in data]
    
    # 生成答案
    print("\n开始生成答案...")
    all_generated = {}
    for cfg in GENERATION_CONFIGS:
        outputs = generate_answers(
            model,
            tokenizer,
            prompts,
            cfg,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        all_generated[cfg['name']] = outputs
    
    # 创建 DeepSeek client
    client = None
    if not args.no_judge:
        client = create_deepseek_client()
        print("\nDeepSeek judge 已就绪")
    
    # 构造 pairs
    print("\n构造 DPO preference pairs...")
    pairs = construct_pairs(
        data, prompts, all_generated, client,
        use_judge=not args.no_judge,
        max_workers=args.max_workers,
        min_judge_confidence=args.min_judge_confidence,
    )
    
    # 保存
    print(f"\n保存 {len(pairs)} 条 DPO 数据到 {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print("完成！")


if __name__ == '__main__':
    main()
