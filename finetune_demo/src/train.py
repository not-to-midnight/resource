"""训练主脚本：读配置 -> 加载数据/模型 -> 注入 LoRA -> Trainer 训练"""

import argparse
from pathlib import Path

import yaml
from transformers import Trainer, TrainingArguments

from dataset import SFTDataCollator, SFTDataset
from model import inject_lora, load_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 分词器 & 底座模型 ----
    tokenizer = load_tokenizer(cfg["model"]["name"])
    model = load_model(cfg["model"])

    # ---- 2. 注入 LoRA ----
    model = inject_lora(model, cfg["lora"])
    model.print_trainable_parameters()  # 打印可训练参数占比

    # ---- 3. 数据集 ----
    train_ds = SFTDataset(
        cfg["data"]["train_file"], tokenizer, max_length=cfg["data"]["max_length"]
    )
    collator = SFTDataCollator(tokenizer)
    print(f"训练样本数: {len(train_ds)}")

    # ---- 4. 训练参数 ----
    t = cfg["train"]
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"],
        lr_scheduler_type=t["lr_scheduler"],
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        save_strategy="epoch",
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        report_to="none",
    )

    # ---- 5. 训练 ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()

    # ---- 6. 只保存 LoRA adapter（通常几十 MB）----
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter 已保存到: {adapter_dir}")


if __name__ == "__main__":
    main()
