import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_fineval_val(data_dir='./fineval/val'):
    """加载 Fineval val 数据集的所有 CSV 文件"""
    records = []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    for fname in files:
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path)
        subject = fname.replace('_val.csv', '')
        for _, row in df.iterrows():
            records.append({
                'id': row['id'],
                'subject': subject,
                'question': str(row['question']),
                'A': str(row['A']),
                'B': str(row['B']),
                'C': str(row['C']),
                'D': str(row['D']),
                'answer': str(row['answer']).strip().upper(),
            })
    return records


def build_prompt(question, choices):
    """构建 zero-shot 评测 prompt"""
    prompt = (
        "请回答以下选择题，只输出选项字母（如 A、B、C、D 或多选如 AB），不要解释。\n\n"
        f"题目：{question}\n"
    )
    for k in ['A', 'B', 'C', 'D']:
        if k in choices:
            prompt += f"{k}. {choices[k]}\n"
    prompt += "答案："
    return prompt


def extract_answer(text):
    """从模型输出中提取答案字母
    
    模型输出会重复 prompt，因此需要从最后的'答案：'或'正确答案是'之后提取
    """
    text = text.strip()
    
    # 优先从最后一个答案标记之后提取
    answer_markers = ['答案：', '答案是：', '正确答案是：', '答案:', '答案是:', '正确答案是:', '答案选']
    last_answer_text = None
    for marker in answer_markers:
        if marker in text:
            # 取最后一个标记之后的内容
            last_answer_text = text.rsplit(marker, 1)[-1]
    
    search_text = last_answer_text if last_answer_text is not None else text
    search_text = search_text.strip()
    
    # 去掉开头的解释词
    search_text = re.sub(r'^(答案|答案选|正确答案是|选|我认为|选项|是)[:：]?\s*', '', search_text)
    search_text = search_text.strip()
    
    # 提取开头的连续 A-D 字母组合
    match = re.search(r'^[A-D]+', search_text.upper())
    if match:
        return match.group(0)
    
    # 备选：在答案区域找独立的大写字母组合
    match = re.search(r'\b([A-D]+)\b', search_text.upper())
    if match:
        return match.group(1)
    
    return search_text[:5].upper()


def batch_inference(model, tokenizer, prompts, batch_size=8, max_new_tokens=16):
    """批量推理"""
    all_outputs = []
    device = model.device
    for i in tqdm(range(0, len(prompts), batch_size), desc="推理中"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # 解码生成的部分（使用完整输入长度切片，避免 left padding 下切错）
        input_seq_len = inputs['input_ids'].shape[1]
        for idx, output in enumerate(outputs):
            generated = output[input_seq_len:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            all_outputs.append(decoded)
    return all_outputs


def evaluate(records, model, tokenizer, batch_size=8):
    """评测并计算正确率"""
    prompts = []
    for r in records:
        choices = {'A': r['A'], 'B': r['B'], 'C': r['C'], 'D': r['D']}
        prompts.append(build_prompt(r['question'], choices))
    
    outputs = batch_inference(model, tokenizer, prompts, batch_size=batch_size)
    
    correct = 0
    results = []
    for r, out in zip(records, outputs):
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
    parser.add_argument('--model_path', type=str, default='./qwen/Qwen2___5-3B-Instruct')
    parser.add_argument('--data_dir', type=str, default='./fineval/val')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output', type=str, default='fineval_eval_results.json')
    args = parser.parse_args()

    print(f'加载模型: {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()
    print('模型加载完成')

    print(f'加载数据: {args.data_dir}')
    records = load_fineval_val(args.data_dir)
    print(f'共 {len(records)} 道题')

    accuracy, results = evaluate(records, model, tokenizer, batch_size=args.batch_size)

    print(f'\n总体正确率: {accuracy*100:.2f}% ({sum(r["correct"] for r in results)}/{len(results)})')
    
    # 按科目统计
    subject_stats = {}
    for r in results:
        sub = r['subject']
        if sub not in subject_stats:
            subject_stats[sub] = {'total': 0, 'correct': 0}
        subject_stats[sub]['total'] += 1
        if r['correct']:
            subject_stats[sub]['correct'] += 1
    
    print('\n各科目正确率:')
    for sub in sorted(subject_stats.keys()):
        s = subject_stats[sub]
        acc = s['correct'] / s['total'] * 100
        print(f'  {sub}: {acc:.2f}% ({s["correct"]}/{s["total"]})')

    # 保存结果
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
