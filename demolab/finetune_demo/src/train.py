"""训练主脚本：读配置 -> 加载数据/模型 -> 注入 LoRA -> Trainer 训练

整个微调的骨架就在这个文件里，五步对应下面五个编号注释。
"""

import argparse
from pathlib import Path

import yaml
from transformers import Trainer, TrainingArguments

from dataset import SFTDataCollator, SFTDataset
from model import inject_lora, load_model, load_tokenizer


def main():
    # argparse：解析命令行参数。--config 可指定别的 yaml，默认 config.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # 读 yaml 配置，safe_load 会把文件解析成 Python 字典
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 输出目录；exist_ok=True 表示目录已存在也不报错
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 分词器 & 底座模型 ----
    tokenizer = load_tokenizer(cfg["model"]["name"])
    model = load_model(cfg["model"])

    # ---- 2. 注入 LoRA（这一步之后，只有极小部分参数会被训练）----
    model = inject_lora(model, cfg["lora"])
    # 打印可训练参数占比 —— 最能直观说明 LoRA 的价值，跑一次就能看到
    model.print_trainable_parameters()

    # ---- 3. 数据集 ----
    train_ds = SFTDataset(
        cfg["data"]["train_file"], tokenizer, max_length=cfg["data"]["max_length"]
    )
    collator = SFTDataCollator(tokenizer)  # 负责 padding + 拼 batch
    print(f"训练样本数: {len(train_ds)}")

    # ---- 4. 训练参数 ----
    t = cfg["train"]
    training_args = TrainingArguments(
        output_dir=str(out_dir),  # checkpoint 存到哪
        num_train_epochs=t["epochs"],  # 整个数据集完整过几遍
        per_device_train_batch_size=t["batch_size"],  # 单张卡一次喂几条样本
        gradient_accumulation_steps=t["grad_accum"],  # 攒几次梯度才更新一次参数
        # 等效 batch = batch_size * grad_accum = 2 * 4 = 8
        learning_rate=t["lr"],  # 学习率，LoRA 通常用 1e-4 ~ 3e-4
        lr_scheduler_type=t["lr_scheduler"],  # cosine = 余弦衰减，越训越小
        warmup_ratio=t["warmup_ratio"],  # 前 3% 步数慢慢升温，防止开局就崩
        logging_steps=t["logging_steps"],  # 每 5 步打印一次 loss
        save_strategy="epoch",  # 每个 epoch 结束存一次 checkpoint
        bf16=t["bf16"],  # 用 bf16 混合精度训练，省显存
        gradient_checkpointing=t["gradient_checkpointing"],  # 梯度检查点，进一步省显存
        report_to="none",  # 不上报到 wandb 等第三方平台
    )

    # ---- 5. 训练 ----
    # Trainer 内部就是个循环：取 batch -> 前向算 loss -> 反向求梯度 -> 更新参数
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()

    # ---- 6. 只保存 LoRA adapter ----
    # 注意：这里存的是 LoRA 小矩阵，不含底座，所以通常只有几十 MB
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter 已保存到: {adapter_dir}")


if __name__ == "__main__":
    main()
