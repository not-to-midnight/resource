"""把 LoRA adapter 合并回底座，导出成可直接部署的完整模型

用法：
    python src/merge_lora.py \
        --base Qwen/Qwen2.5-1.5B-Instruct \
        --adapter outputs/adapter \
        --out outputs/merged

为什么要合并？
    adapter 模式部署时，运行环境还得装 peft 才能加载；
    合并之后就是一个普通模型，体积和底座一样大，但部署更简单。
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="底座模型名或本地路径")
    parser.add_argument("--adapter", required=True, help="LoRA adapter 目录")
    parser.add_argument("--out", required=True, help="合并后模型的输出目录")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.adapter)

    # 核心一步：把 BA 的增量真正加到 W 上（W' = W + BA），
    # 并拆掉 adapter 的外壳，得到一个普通模型。
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print(f"合并完成，可直接用 transformers 加载: {args.out}")


if __name__ == "__main__":
    main()
