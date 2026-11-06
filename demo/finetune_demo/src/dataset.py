"""数据加载与预处理：JSONL -> 带标签屏蔽的训练样本

微调里"数据处理"这一步的核心就两件事：

    1. 把对话按模型要求的格式渲染成 token 序列（用 chat template）；
    2. 决定哪些 token 参与 loss 计算（这里是只让 assistant 回复参与）。

这个文件改完，模型层和训练层都不用动。
"""

import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """SFT = Supervised Fine-Tuning（监督微调）。

    继承 torch 的 Dataset 后必须实现两个方法：

        __len__      告诉 DataLoader 一共有多少条样本
        __getitem__  告诉 DataLoader "第 i 条样本长什么样"

    DataLoader 会拿着这个类自己按索引取数据、拼成 batch。
    """

    def __init__(self, path: str, tokenizer: Any, max_length: int = 512):
        self.tokenizer = tokenizer
        # 单条样本的最大 token 数。超了直接截断，防止个别长样本把显存撑爆
        self.max_length = max_length
        self.samples = self._load(path)

    @staticmethod
    def _load(path: str) -> list[dict]:
        """读 JSONL：一行是一个 JSON 对象，空行跳过。"""
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
        """把第 idx 条对话转成模型能吃的 token + 标签。

        返回的 labels 就是"标准答案"：训练时模型预测得越接近它，loss 越小。
        """
        messages = self.samples[idx]["messages"]

        # 1) 渲染完整对话（含 assistant 回复）。
        #    Qwen 有自己的对话模板，拼出来大致是这样：
        #      <|im_start|>user\n问题<|im_end|>\n<|im_start|>assistant\n回答<|im_end|>
        #    apply_chat_template 会自动拼成这个格式；
        #    tokenize=True 表示顺便把文本转成 token id（模型只认 id）。
        full_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )

        # 2) 再只渲染到"user 说完"为止，用来量出 prompt 有多长。
        #    add_generation_prompt=True 会在末尾补上 "<|im_start|>assistant\n"，
        #    这正好是回答开始的位置，所以它的长度就是 prompt 的长度。
        #    因为模板是纯字符串拼接，这一段必然是 full_ids 的前缀。
        prompt_ids = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )

        # 3) 截断 + 造 labels
        input_ids = full_ids[: self.max_length]
        labels = list(input_ids)  # 先复制一份，下面只改 prompt 那一段
        prompt_len = min(len(prompt_ids), len(input_ids))
        for i in range(prompt_len):
            labels[i] = -100  # -100 是 PyTorch 约定的"忽略值"，该位置不参与 loss
        # 屏蔽 prompt 的作用：模型只会去学"怎么回答"，
        # 不会去学"复述问题"，训练信号更聚焦。

        return {"input_ids": input_ids, "labels": labels}


@dataclass
class SFTDataCollator:
    """批量组装器（collate function）。

    DataLoader 每次取出的一批样本长度并不一样（问题有长有短），
    没法直接堆成一个矩阵。collator 的活就是：

        把这一批样本用 padding 补齐到同一长度，再拼成 tensor 返回。
    """

    tokenizer: Any

    def __call__(self, features: list[dict]) -> dict:
        # 这一批里最长的样本有多长 —— 大家统一补到这个长度
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, labels, attention_mask = [], [], []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            # 真实 token 后面补上 pad token
            input_ids.append(f["input_ids"] + [pad_id] * pad_n)
            # padding 位置同样标 -100，避免模型去学"补位符"
            labels.append(f["labels"] + [-100] * pad_n)
            # attention_mask：1 = 真实 token，0 = padding。
            # 让模型的注意力只落在真实 token 上，忽略为了对齐而补的填充位。
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
