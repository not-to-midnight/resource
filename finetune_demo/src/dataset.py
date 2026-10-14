"""数据加载与预处理：JSONL -> 带标签屏蔽的训练样本

关键点：
1. 用 tokenizer 自带的 chat template 把对话渲染成模型期望的格式；
2. 只对 assistant 回复计算 loss，prompt 部分标签设为 -100 屏蔽掉。
"""

import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """读取 chat 格式的 JSONL，逐条编码为训练样本"""

    def __init__(self, path: str, tokenizer: Any, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load(path)

    @staticmethod
    def _load(path: str) -> list[dict]:
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        messages = self.samples[idx]["messages"]

        # 1) 渲染完整对话（含 assistant 回复）
        full_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )

        # 2) 只渲染到 user 结束，得到 prompt 长度
        #    因为模板是确定性拼接，这段必然是 full_ids 的前缀
        prompt_ids = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )

        # 3) 截断 + 构造 labels
        input_ids = full_ids[: self.max_length]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(input_ids))
        for i in range(prompt_len):
            labels[i] = -100  # 屏蔽 prompt，只训练回复

        return {"input_ids": input_ids, "labels": labels}


@dataclass
class SFTDataCollator:
    """把不等长样本 padding 成 batch，并把 padding 位置的标签也设为 -100"""

    tokenizer: Any

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, labels, attention_mask = [], [], []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_n)
            labels.append(f["labels"] + [-100] * pad_n)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
