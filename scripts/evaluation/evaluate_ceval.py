import os
import json
import argparse
import re
import csv
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_ceval_val(data_dir='./ceval/val'):
    records = []
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob('*_val.csv'))
    for f in files:
        subject = f.stem.replace('_val', '')
        with open(f, 'r', encoding='utf-8', newline='') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                records.append({
                    'id': row.get('id'),
                    'subject': subject,
                    'question': str(row['question']),
                    'A': str(row['A']),
                    'B': str(row['B']),
                    'C': str(row['C']),
                    'D': str(row['D']),
                    'answer': str(row['answer']).strip().upper(),
                })
    return records


def build_messages(question, choices, system_prompt=None):
    instruction = "请回答以下单项选择题，直接给出选项字母。\n\n"
    instruction += f"题目：{question}\n"
    for k in ['A', 'B', 'C', 'D']:
        if k in choices:
            instruction += f"{k}. {choices[k]}\n"
    instruction += "\n答案："
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    return messages


def extract_answer(text):
    text = text.strip()
    # 优先匹配“答案：”后的内容
    markers = ['答案：', '答案是：', '正确答案是：', '答案:', '答案是:', '正确答案是:', '答案选']
    search_text = text
    for marker in markers:
        if marker in text:
            search_text = text.rsplit(marker, 1)[-1]
            break
    search_text = search_text.strip()
    search_text = re.sub(r'^(答案|答案选|正确答案是|选|我认为|选项|是)[:：]?\s*', '', search_text)
    search_text = search_text.strip()
    # 优先匹配行首或独立选项字母
    m = re.search(r'^[A-D]', search_text.upper())
    if m:
        return m.group(0)
    m = re.search(r'\b([A-D])\b', search_text.upper())
    if m:
        return m.group(1)
    # 兜底：看前5个字符
    for ch in search_text.upper():
        if ch in 'ABCD':
            return ch
    return ''


def batch_evaluate(model, tokenizer, records, batch_size=16, system_prompt=None):
    all_prompts = []
    for r in records:
        choices = {'A': r['A'], 'B': r['B'], 'C': r['C'], 'D': r['D']}
        messages = build_messages(r['question'], choices, system_prompt)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(prompt)

    all_outputs = []
    device = model.device
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="评测中"):
        batch_prompts = all_prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_seq_len = inputs['input_ids'].shape[1]
        for output in outputs:
            generated = output[input_seq_len:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            all_outputs.append(decoded)

    correct = 0
    results = []
    for r, out in zip(records, all_outputs):
        pred = extract_answer(out)
        gold = r['answer']
        is_correct = pred == gold
        if is_correct:
            correct += 1
        results.append({
            'subject': r['subject'],
            'id': r['id'],
            'question': r['question'],
            'gold': gold,
            'predict': pred,
            'raw_output': out,
            'correct': is_correct,
        })

    accuracy = correct / len(records) if records else 0
    return accuracy, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default='./qwen/Qwen2___5-3B-Instruct')
    parser.add_argument('--adapter_path', type=str, default=None, help='LoRA adapter path；为空则评测基座模型')
    parser.add_argument('--data_dir', type=str, default='./ceval/val')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output', type=str, default='ceval_eval_results.json')
    parser.add_argument('--system_prompt', type=str, default=None, help='system prompt')
    args = parser.parse_args()

    print(f'加载基础模型: {args.base_model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )

    if args.adapter_path:
        print(f'加载 LoRA adapter: {args.adapter_path}')
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    records = load_ceval_val(args.data_dir)
    print(f'共 {len(records)} 道题')

    accuracy, results = batch_evaluate(model, tokenizer, records, batch_size=args.batch_size, system_prompt=args.system_prompt)

    subject_stats = {}
    for r in results:
        sub = r['subject']
        if sub not in subject_stats:
            subject_stats[sub] = {'total': 0, 'correct': 0}
        subject_stats[sub]['total'] += 1
        if r['correct']:
            subject_stats[sub]['correct'] += 1

    print(f'\n总体正确率: {accuracy*100:.2f}% ({sum(r["correct"] for r in results)}/{len(results)})')
    print('\n各科目正确率:')
    for sub in sorted(subject_stats.keys()):
        s = subject_stats[sub]
        acc = s['correct'] / s['total'] * 100
        print(f'  {sub}: {acc:.2f}% ({s["correct"]}/{s["total"]})')

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'accuracy': accuracy,
            'total': len(results),
            'correct': sum(r['correct'] for r in results),
            'subject_accuracy': {sub: subject_stats[sub]['correct']/subject_stats[sub]['total'] for sub in subject_stats},
            'results': results,
        }, f, ensure_ascii=False, indent=2)
    print(f'\n详细结果已保存: {args.output}')


if __name__ == '__main__':
    main()
