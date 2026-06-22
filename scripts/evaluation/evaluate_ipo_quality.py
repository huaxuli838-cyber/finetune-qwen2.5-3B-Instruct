import os
import json
import argparse
import random
import re
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from openai import OpenAI


def load_model(base_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_responses(model, tokenizer, prompts, batch_size=8, max_new_tokens=512):
    outputs = []
    device = model.device
    for i in tqdm(range(0, len(prompts), batch_size), desc="生成回答"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs['input_ids'].shape[1]
        for seq in generated:
            out = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            outputs.append(out)
    return outputs


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


JUDGE_PAIRWISE_PROMPT = """你是一位资深金融领域评估专家。请对两个模型（模型A和模型B）针对同一金融单项选择题的回答进行成对比较。

**题目**：
{question}

**选项**：
{options}

**模型A的回答**：
{answer_a}

**模型B的回答**：
{answer_b}

**评估要求**：
请只比较两个回答的生成质量，**不要看最终选项字母是否正确**。重点评估：
1. **专业性（Professionalism）**：术语使用是否准确恰当，是否体现金融领域深度知识。
2. **推理清晰度（Reasoning Clarity）**：逻辑链条是否完整、步骤是否清晰、是否易于理解。
3. **格式规范性（Format）**：是否包含“答案：X”和“解析：...”等清晰结构。
4. **内容完整性（Completeness）**：解析是否充分，不是过于简略。
5. **语言流畅度（Fluency）**：没有语法错误、重复、混乱。

请严格按以下 JSON 格式输出，不要添加任何额外内容：
{{
  "winner": "A" 或 "B" 或 "tie",
  "reason": "综合判断理由（50字以内）",
  "dimensions": {{
    "professionalism": {{"A": 1-5整数, "B": 1-5整数, "reason": "简要理由"}},
    "reasoning_clarity": {{"A": 1-5整数, "B": 1-5整数, "reason": "简要理由"}},
    "format": {{"A": 1-5整数, "B": 1-5整数, "reason": "简要理由"}},
    "completeness": {{"A": 1-5整数, "B": 1-5整数, "reason": "简要理由"}},
    "fluency": {{"A": 1-5整数, "B": 1-5整数, "reason": "简要理由"}}
  }}
}}"""


def try_parse_json(text):
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def call_judge(client, question, options_text, ans_a, ans_b, max_retries=3):
    prompt = JUDGE_PAIRWISE_PROMPT.format(
        question=question,
        options=options_text,
        answer_a=ans_a,
        answer_b=ans_b,
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model='deepseek-chat',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            content = resp.choices[0].message.content.strip()
            result = try_parse_json(content)
            if result and 'winner' in result and 'dimensions' in result:
                return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return {
        'winner': 'tie',
        'reason': 'judge 调用失败或解析失败',
        'dimensions': {
            'professionalism': {'A': 3, 'B': 3, 'reason': '失败'},
            'reasoning_clarity': {'A': 3, 'B': 3, 'reason': '失败'},
            'format': {'A': 3, 'B': 3, 'reason': '失败'},
            'completeness': {'A': 3, 'B': 3, 'reason': '失败'},
            'fluency': {'A': 3, 'B': 3, 'reason': '失败'},
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default='./qwen/Qwen2___5-3B-Instruct')
    parser.add_argument('--sft_adapter_path', type=str, default='./qwen_finance_sft/final')
    parser.add_argument('--ipo_adapter_path', type=str, default='./qwen_cflue_ipo/final')
    parser.add_argument('--test_prompts', type=str, default='./ipo_eval_test_prompts.jsonl')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--output', type=str, default='ipo_quality_eval_results.json')
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.test_prompts, encoding='utf-8')]
    prompts = [it['prompt'] for it in items]
    questions = [it['question'] for it in items]
    options_list = [it['choices'] for it in items]
    answers = [it['answer'] for it in items]

    print('加载 SFT 模型...')
    model, tokenizer = load_model(args.base_model_path, args.sft_adapter_path)
    print('生成 SFT 回答...')
    sft_responses = generate_responses(model, tokenizer, prompts, batch_size=args.batch_size)
    del model
    torch.cuda.empty_cache()

    print('加载 IPO 模型...')
    model, tokenizer = load_model(args.base_model_path, args.ipo_adapter_path)
    print('生成 IPO 回答...')
    ipo_responses = generate_responses(model, tokenizer, prompts, batch_size=args.batch_size)
    del model
    torch.cuda.empty_cache()

    # 本地选项正确率统计
    sft_correct = sum(1 for resp, gt in zip(sft_responses, answers) if extract_answer(resp) == gt)
    ipo_correct = sum(1 for resp, gt in zip(ipo_responses, answers) if extract_answer(resp) == gt)
    print(f"\n选项正确率：SFT {sft_correct}/{len(items)} ({sft_correct/len(items)*100:.1f}%), "
          f"IPO {ipo_correct}/{len(items)} ({ipo_correct/len(items)*100:.1f}%)")

    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError('请设置环境变量 DEEPSEEK_API_KEY')
    client = OpenAI(api_key=api_key, base_url='https://api.deepseek.com')

    judge_inputs = []
    for idx, q in enumerate(questions):
        sft_first = random.choice([True, False])
        ans_a = sft_responses[idx] if sft_first else ipo_responses[idx]
        ans_b = ipo_responses[idx] if sft_first else sft_responses[idx]
        opts = '\n'.join([f"{k}. {v}" for k, v in sorted(options_list[idx].items())])
        judge_inputs.append({
            'idx': idx,
            'question': q,
            'options': opts,
            'ground_truth': answers[idx],
            'answer_a': ans_a,
            'answer_b': ans_b,
            'a_is_sft': sft_first,
        })

    print('调用 DeepSeek judge 进行成对评估（只看生成质量，不看选项正确性）...')
    judge_results = [None] * len(judge_inputs)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {
            executor.submit(call_judge, client, ji['question'], ji['options'], ji['answer_a'], ji['answer_b']): ji['idx']
            for ji in judge_inputs
        }
        for future in tqdm(as_completed(future_to_idx), total=len(judge_inputs), desc='Judge'):
            idx = future_to_idx[future]
            judge_results[idx] = future.result()

    sft_win = ipo_win = tie = 0
    dim_scores = {
        'professionalism': {'sft': [], 'ipo': []},
        'reasoning_clarity': {'sft': [], 'ipo': []},
        'format': {'sft': [], 'ipo': []},
        'completeness': {'sft': [], 'ipo': []},
        'fluency': {'sft': [], 'ipo': []},
    }
    results = []
    for ji, jr, sft_resp, ipo_resp in zip(judge_inputs, judge_results, sft_responses, ipo_responses):
        idx = ji['idx']
        a_is_sft = ji['a_is_sft']
        winner_label = jr['winner']
        if winner_label == 'tie':
            tie += 1
            winner_model = 'tie'
        else:
            winner_model = 'sft' if (winner_label == 'A' and a_is_sft) or (winner_label == 'B' and not a_is_sft) else 'ipo'
            if winner_model == 'sft':
                sft_win += 1
            else:
                ipo_win += 1

        model_scores = {'sft': {}, 'ipo': {}}
        for dim, scores in jr['dimensions'].items():
            if dim not in dim_scores:
                continue
            if a_is_sft:
                model_scores['sft'][dim] = scores['A']
                model_scores['ipo'][dim] = scores['B']
            else:
                model_scores['sft'][dim] = scores['B']
                model_scores['ipo'][dim] = scores['A']
            dim_scores[dim]['sft'].append(model_scores['sft'][dim])
            dim_scores[dim]['ipo'].append(model_scores['ipo'][dim])

        results.append({
            'idx': idx,
            'question': ji['question'],
            'ground_truth': ji['ground_truth'],
            'sft_response': sft_resp,
            'ipo_response': ipo_resp,
            'winner': winner_model,
            'judge_raw': jr,
            'dimension_scores': model_scores,
        })

    summary = {
        'total': len(items),
        'sft_win': sft_win,
        'ipo_win': ipo_win,
        'tie': tie,
        'sft_win_rate': sft_win / len(items),
        'ipo_win_rate': ipo_win / len(items),
        'tie_rate': tie / len(items),
        'sft_option_accuracy': sft_correct / len(items),
        'ipo_option_accuracy': ipo_correct / len(items),
        'dimension_averages': {
            dim: {
                'sft': sum(scores['sft']) / len(scores['sft']),
                'ipo': sum(scores['ipo']) / len(scores['ipo']),
            }
            for dim, scores in dim_scores.items()
        },
        'results': results,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n===== 评估汇总 =====')
    print(f"总样本数: {summary['total']}")
    print(f"SFT 胜: {sft_win} ({summary['sft_win_rate']*100:.1f}%)")
    print(f"IPO 胜: {ipo_win} ({summary['ipo_win_rate']*100:.1f}%)")
    print(f"平局: {tie} ({summary['tie_rate']*100:.1f}%)")
    print(f"选项正确率: SFT {summary['sft_option_accuracy']*100:.1f}% vs IPO {summary['ipo_option_accuracy']*100:.1f}%")
    print('\n各维度平均分（SFT vs IPO）：')
    for dim, scores in summary['dimension_averages'].items():
        print(f"  {dim}: {scores['sft']:.2f} vs {scores['ipo']:.2f} (IPO {'+' if scores['ipo']>scores['sft'] else ''}{scores['ipo']-scores['sft']:+.2f})")
    print(f'\n详细结果已保存: {args.output}')


if __name__ == '__main__':
    main()
