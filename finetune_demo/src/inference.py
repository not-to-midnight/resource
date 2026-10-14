"""推理：加载底座 + LoRA adapter，进行单轮对话

用法：
    python src/inference.py --prompt "LoRA 的原理是什么？"
"""

import argparse

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter", default=None, help="LoRA adapter 目录，默认取配置里的 outputs/adapter")
    parser.add_argument("--prompt", default="介绍一下具身智能")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    adapter = args.adapter or f"{cfg['output']['dir']}/adapter"

    # 分词器直接复用 adapter 目录里保存的那份
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()

    messages = [
        {"role": "system", "content": cfg["data"]["system_prompt"]},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # 只取新生成的部分（去掉输入 prompt）
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"用户: {args.prompt}")
    print(f"模型: {reply}")


if __name__ == "__main__":
    main()
