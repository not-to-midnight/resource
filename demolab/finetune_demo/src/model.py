"""模型与分词器加载，以及 LoRA 注入

分工：

    load_tokenizer -> 文本 <-> token id 互转，并套用聊天模板
    load_model     -> 加载底座大模型（15 亿参数，全部冻结不训练）
    inject_lora    -> 给底座挂上可训练的小矩阵（LoRA）
"""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    # 因果语言模型需要一个 pad token 才能做 batch padding。
    # Qwen 默认没有 pad_token，这里直接拿 eos_token 顶替。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(cfg: dict):
    # 从字符串取出 torch 的 dtype 对象，例如 "bfloat16" -> torch.bfloat16
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        torch_dtype=dtype,  # 权重精度：bf16 显存约减半，且训练稳定
        device_map=cfg.get("device_map", "auto"),  # auto = 自动分配到 GPU/CPU
        trust_remote_code=True,
    )
    # 开启 gradient_checkpointing（梯度检查点）后必须关掉 use_cache，
    # 否则 transformers 会直接报错。前者用"重算"换显存，后者是推理加速缓存。
    model.config.use_cache = False
    return model


def inject_lora(model, cfg: dict):
    """冻结底座权重，只训练旁路插入的低秩矩阵 A / B。

    原理：原本某一层的权重 W 不动，输出从 Wx 改成 Wx + BAx。
    其中 A、B 是非常"瘦"的矩阵（秩 r=16），
    参数量只有原来的千分之几，这就是 LoRA 省显存的原因。
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 任务是因果语言模型（预测下一个 token）
        r=cfg["r"],  # 秩：越大表达能力越强、显存占用越多
        lora_alpha=cfg["alpha"],  # 缩放系数，一般取 r 的 1~2 倍
        lora_dropout=cfg["dropout"],  # 随机丢弃，防过拟合
        target_modules=cfg["target_modules"],  # 挂到哪些层上（注意力和 MLP 的投影层）
    )
    # get_peft_model 会把底座所有参数设成 requires_grad=False（冻结），
    # 只留下 LoRA 那部分可训练，并返回包装后的模型。
    return get_peft_model(model, lora_config)
