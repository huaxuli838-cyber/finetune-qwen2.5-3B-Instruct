import json
import os
import ast
import argparse
from collections import Counter


def parse_choices(choices_str):
    """安全解析 choices 字符串为 dict"""
    try:
        choices = ast.literal_eval(choices_str)
        if isinstance(choices, dict):
            # 按 A, B, C, D, E 排序
            return {k: choices[k] for k in sorted(choices.keys())}
    except Exception:
        pass
    return None


def format_question_prompt(question, choices):
    """格式化题目和选项为 prompt"""
    lines = [f"题目：{question.strip()}"]
    for key in sorted(choices.keys()):
        lines.append(f"{key}. {choices[key]}")
    return "\n".join(lines)


def format_output(answer, analysis=None):
    """格式化输出，包含答案和解析"""
    output = f"答案：{answer.strip().upper()}"
    if analysis and str(analysis).strip():
        analysis_text = str(analysis).strip()
        output += f"\n解析：{analysis_text}"
    return output


def build_sft_item(item, idx):
    """将单条 CFLUE 数据转换为 SFT 格式"""
    choices = parse_choices(item['choices'])
    if choices is None:
        return None
    
    answer = str(item.get('answer', '')).strip().upper()
    if not answer:
        return None
    
    question_prompt = format_question_prompt(item['question'], choices)
    
    instruction = (
        "请回答以下单项选择题，并给出简要解析。\n\n"
        f"{question_prompt}"
    )
    
    output = format_output(answer, item.get('analysis'))
    
    return {
        "id": f"cflue_single_{idx}",
        "certification": item.get('名称', ''),
        "subject": item.get('科目', ''),
        "chapter": item.get('章节', ''),
        "task": item.get('task', '单项选择题'),
        "instruction": instruction,
        "input": "",
        "output": output,
        "answer": answer,
        "choices": choices,
        "has_analysis": bool(item.get('analysis') and str(item['analysis']).strip()),
    }


def load_split(filepath):
    """加载单个 split 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./cflue_full')
    parser.add_argument('--output_dir', type=str, default='./cflue_sft')
    parser.add_argument('--min_analysis', action='store_true', help='只保留带解析的数据')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    splits = {}
    for split_name in ['train', 'val', 'test']:
        path = os.path.join(args.input_dir, f'{split_name}.json')
        if os.path.exists(path):
            splits[split_name] = load_split(path)
    
    print('原始数据量:')
    for split_name, data in splits.items():
        print(f'  {split_name}: {len(data)} 条')
    
    # 提取单项选择题
    sft_data_by_split = {}
    global_idx = 0
    for split_name, data in splits.items():
        sft_items = []
        for item in data:
            if item.get('task') != '单项选择题':
                continue
            sft_item = build_sft_item(item, global_idx)
            global_idx += 1
            if sft_item is None:
                continue
            if args.min_analysis and not sft_item['has_analysis']:
                continue
            sft_items.append(sft_item)
        sft_data_by_split[split_name] = sft_items
    
    # 统计
    total = sum(len(items) for items in sft_data_by_split.values())
    print(f'\n单项选择题数据量:')
    for split_name, items in sft_data_by_split.items():
        with_analysis = sum(1 for x in items if x['has_analysis'])
        print(f'  {split_name}: {len(items)} 条（含解析: {with_analysis}, 无解析: {len(items)-with_analysis}）')
    print(f'  总计: {total} 条')
    
    # 科目分布
    all_items = []
    for items in sft_data_by_split.values():
        all_items.extend(items)
    
    subject_counts = Counter([x['subject'] for x in all_items])
    print(f'\n科目数量: {len(subject_counts)}')
    print('科目分布（前15）:')
    for sub, cnt in subject_counts.most_common(15):
        print(f'  {sub}: {cnt}')
    
    # 选项数量分布
    choice_counts = Counter([len(x['choices']) for x in all_items])
    print(f'\n选项数量分布: {dict(sorted(choice_counts.items()))}')
    
    # 保存各 split
    for split_name, items in sft_data_by_split.items():
        output_path = os.path.join(args.output_dir, f'cflue_single_choice_{split_name}.jsonl')
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'已保存: {output_path} ({len(items)} 条)')
    
    # 保存合并版
    combined_path = os.path.join(args.output_dir, 'cflue_single_choice_all.jsonl')
    with open(combined_path, 'w', encoding='utf-8') as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'已保存合并版: {combined_path} ({len(all_items)} 条)')
    
    # 保存一份不带元信息的简洁版（更通用的 SFT 格式）
    simple_items = []
    for item in all_items:
        simple_items.append({
            'instruction': item['instruction'],
            'input': item['input'],
            'output': item['output'],
        })
    simple_path = os.path.join(args.output_dir, 'cflue_single_choice_alpaca.jsonl')
    with open(simple_path, 'w', encoding='utf-8') as f:
        for item in simple_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'已保存 Alpaca 简洁版: {simple_path} ({len(simple_items)} 条)')
    
    print('\n完成！')


if __name__ == '__main__':
    main()
