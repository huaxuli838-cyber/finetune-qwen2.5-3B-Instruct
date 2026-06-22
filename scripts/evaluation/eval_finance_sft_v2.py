import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = './qwen/Qwen2___5-3B-Instruct'
ADAPTER_PATH = './qwen_finance_sft_v2/final'
OUTPUT_PATH = './qwen_finance_sft_v2/inference_test_results.jsonl'

TEST_PROMPTS = [
    {
        "category": "概念理解",
        "temperature": 0.7,
        "question": "请用通俗易懂的语言解释什么是“净息差（NIM）”，并说明它为什么是衡量商业银行盈利能力的重要指标。"
    },
    {
        "category": "概念理解",
        "temperature": 0.7,
        "question": "解释“M2 货币供应量”与“社会融资规模”的区别与联系。"
    },
    {
        "category": "政策解读",
        "temperature": 0.7,
        "question": "2024 年中国人民银行下调存款准备金率和政策利率，会对实体经济、房地产市场和资本市场分别产生哪些影响？"
    },
    {
        "category": "政策解读",
        "temperature": 0.7,
        "question": "个人养老金制度全面实施后，对银行、基金、保险和资本市场分别意味着什么？"
    },
    {
        "category": "市场分析",
        "temperature": 0.7,
        "question": "美联储加息周期若进入尾声，对新兴市场股市、汇率以及中国出口企业会带来哪些机遇与风险？"
    },
    {
        "category": "市场分析",
        "temperature": 0.7,
        "question": "在当前低利率环境下，分析银行、保险、券商三类金融机构面临的共同挑战与差异化应对策略。"
    },
    {
        "category": "实务应用",
        "temperature": 0.7,
        "question": "一家外贸企业预计未来三个月将收到 100 万美元货款，担心美元贬值，请给出三种可行的汇率风险对冲方案，并比较其优缺点。"
    },
    {
        "category": "实务应用",
        "temperature": 0.7,
        "question": "银行在对制造业中小企业进行信贷尽调时，应重点分析哪些财务指标和非财务因素？"
    },
    {
        "category": "计算推理",
        "temperature": 0.1,
        "question": "某公司股票当前股价为 50 元，每股收益（EPS）为 5 元，每股净资产为 20 元。请计算该公司的市盈率和市净率，并简要说明这两个指标的投资含义。"
    },
    {
        "category": "计算推理",
        "temperature": 0.1,
        "question": "某项目初始投资 1000 万元，预计未来三年每年末分别产生现金流 400 万元、500 万元、600 万元，折现率为 10%。请计算该项目的净现值（NPV），并判断项目是否值得投资。"
    },
    {
        "category": "计算推理",
        "temperature": 0.1,
        "question": "某债券面值 1000 元，票面利率 8%，每年付息一次，剩余期限 5 年，市场到期收益率为 6%。请计算该债券的当前理论价格（保留两位小数）。"
    },
    {
        "category": "摘要抽取",
        "temperature": 0.7,
        "question": "请对以下政策要点进行摘要，提炼出核心目标和主要措施：国务院常务会议提出，要加大对中小微企业金融支持力度，引导金融机构降低实际贷款利率，推动普惠小微贷款增量扩面，优化贷款期限结构，完善敢贷愿贷能贷会贷长效机制。"
    },
]


def main():
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    for item in TEST_PROMPTS:
        messages = [{"role": "user", "content": item["question"]}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=item["temperature"],
                top_p=0.9,
                do_sample=item["temperature"] > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        result = {
            "category": item["category"],
            "temperature": item["temperature"],
            "question": item["question"],
            "answer": response,
        }
        results.append(result)
        print(f"\n=== {item['category']} ===")
        print(f"Q: {item['question']}")
        print(f"A: {response[:500]}{'...' if len(response) > 500 else ''}")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
