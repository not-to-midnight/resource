"""模型与分词器加载，以及 LoRA 注入"""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    # 因果语言模型需要一个 pad token，缺失时用 eos 顶替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(cfg: dict):
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        torch_dtype=dtype,
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=True,
    )
    # 开启 gradient_checkpointing 时必须关闭 use_cache，否则会报错
    model.config.use_cache = False
    return model


def inject_lora(model, cfg: dict):
    """冻结底座权重，只训练插入的低秩矩阵 A / B"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
    )
    return get_peft_model(model, lora_config)
