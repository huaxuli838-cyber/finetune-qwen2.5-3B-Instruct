import os
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


class SFTDataset(Dataset):
    """SFT 数据集"""
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        system = item.get('system', '')
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        user_content = instruction
        if input_text:
            user_content += '\n' + input_text
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": output})
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # 找到 assistant 回复的起始位置，只计算 assistant 部分的 loss
        # Qwen2.5 chat template 中 assistant 内容在 "assistant\n" 之后
        assistant_start = prompt.rfind("assistant\n")
        if assistant_start == -1:
            assistant_start = 0
        else:
            assistant_start += len("assistant\n")
        
        model_inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = model_inputs["input_ids"]
        labels = [-100] * len(input_ids)
        
        # 计算 assistant 部分对应的 token 位置
        prompt_prefix = prompt[:assistant_start]
        prefix_ids = self.tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
        start_idx = len(prefix_ids)
        
        for i in range(start_idx, len(input_ids)):
            labels[i] = input_ids[i]
        
        model_inputs["labels"] = labels
        return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./qwen/Qwen2___5-3B-Instruct')
    parser.add_argument('--data_path', type=str, default='./cflue_sft_final/cflue_single_choice_all.jsonl')
    parser.add_argument('--output_dir', type=str, default='./qwen_cflue_lora')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'right'  # 训练时用右填充
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    
    # LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 启用梯度检查点以节省显存
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    # 加载数据集
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)
    
    # 简单划分 train/val（90/10）
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f'训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}')
    
    # 计算 warmup steps
    total_steps = (len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)) * args.num_train_epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type='cosine',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        seed=args.seed,
        report_to='none',
        load_best_model_at_end=False,
        dataloader_num_workers=4,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # 保存最终模型
    final_dir = os.path.join(args.output_dir, 'final')
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f'训练完成，模型保存至: {final_dir}')


if __name__ == '__main__':
    main()
