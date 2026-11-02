"""推理：加载底座 + LoRA adapter，进行单轮对话

用法：
    python src/inference.py --prompt "LoRA 的原理是什么？"

和 train.py 的区别：多了 PeftModel.from_pretrained 这一步，
它负责把 adapter 的增量权重"挂"回底座上。
"""

import argparse

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter 目录，默认取配置里的 outputs/adapter",
    )
    parser.add_argument("--prompt", default="介绍一下具身智能")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # adapter 路径：命令行给了就用命令行的，否则用配置里约定的 outputs/adapter
    adapter = args.adapter or f"{cfg['output']['dir']}/adapter"

    # 分词器直接复用 adapter 目录里保存的那份，保证和训练时完全一致
    tokenizer = AutoTokenizer.from_pretrained(adapter)

    # 先加载原始底座（只推理不训练，所以不用管 use_cache）
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], torch_dtype=torch.bfloat16, device_map="auto"
    )
    # 再把 LoRA 增量权重叠加到底座上
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()  # 切到推理模式：关掉 dropout，结果更稳定

    # 组装对话。system + user 的写法要和训练时保持一致，
    # 否则模型看到的是没见过的格式，效果会明显下降。
    messages = [
        {"role": "system", "content": cfg["data"]["system_prompt"]},
        {"role": "user", "content": args.prompt},
    ]
    # add_generation_prompt=True：在末尾补上 "<|im_start|>assistant\n"，
    # 相当于告诉模型"该你回答了"。
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 纯推理不需要梯度，torch.no_grad() 能省显存并提速
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,  # 最多生成 256 个新 token
            do_sample=True,  # 采样生成（设为 False 就变成死板的贪心解码）
            temperature=0.7,  # 温度：越高越随机，越低越保守
            top_p=0.9,  # 只在累计概率前 90% 的词里采样
        )

    # generate 返回的是"输入 + 新生成"拼在一起的结果，
    # 所以要按输入长度把前半段切掉，只保留新生成的部分。
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"用户: {args.prompt}")
    print(f"模型: {reply}")


if __name__ == "__main__":
    main()
