import os
import json
import argparse
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig


def load_dpo_data(data_path: str):
    """加载 DPO 训练数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                'prompt': item['prompt'],
                'chosen': item['chosen'],
                'rejected': item['rejected'],
            })
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default='./qwen/Qwen2___5-3B-Instruct')
    parser.add_argument('--sft_adapter_path', type=str, default='./qwen_cflue_lora/checkpoint-1896')
    parser.add_argument('--data_path', type=str, default='./cflue_dpo_data.jsonl')
    parser.add_argument('--output_dir', type=str, default='./qwen_cflue_dpo')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-7)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--loss_type', type=str, default='sigmoid', choices=['sigmoid', 'hinge', 'ipo', 'exo_pair'], help='DPO loss type')
    args = parser.parse_args()
    
    # 加载 tokenizer
    print(f"加载 tokenizer: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    print(f"加载基础模型: {args.base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    
    # 加载 SFT LoRA adapter
    print(f"加载 SFT LoRA adapter: {args.sft_adapter_path}")
    model = PeftModel.from_pretrained(model, args.sft_adapter_path, is_trainable=True)
    
    # 打印 trainable 参数
    model.print_trainable_parameters()
    
    # 加载 DPO 数据
    print(f"加载 DPO 数据: {args.data_path}")
    train_dataset = load_dpo_data(args.data_path)
    print(f"DPO 训练样本数: {len(train_dataset)}")
    
    # DPO 训练参数
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        loss_type=args.loss_type,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        seed=args.seed,
        gradient_checkpointing=True,
    )
    
    # 初始化 DPOTrainer
    print("初始化 DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL 会自动创建 reference model
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # 开始训练
    print("开始 DPO 训练...")
    trainer.train()
    
    # 保存最终模型
    print(f"保存最终模型到 {args.output_dir}/final")
    trainer.save_model(os.path.join(args.output_dir, 'final'))
    
    print("DPO 训练完成！")


if __name__ == '__main__':
    main()
