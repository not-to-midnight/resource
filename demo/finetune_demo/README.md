# 大模型微调示例（LoRA / Qwen2.5-1.5B）

一个最小可运行的 LoRA 微调完整流程示例：**数据 → 模型注入 LoRA → 训练 → 推理 → 合并导出**。
底座用 `Qwen2.5-1.5B-Instruct`，8GB 显存即可跑通。

---

## 1. 整体流程

```
data/sample_data.jsonl          # 原始对话数据
        │
        ▼  dataset.py          # 套 chat template + 屏蔽 prompt 的 loss
   tokenized samples
        │
        ▼  model.py            # 加载底座 + 注入 LoRA（冻结原权重）
   peft model (可训练参数 ~0.5%)
        │
        ▼  train.py            # Trainer 训练，只更新 LoRA 矩阵
   outputs/adapter/              # 几十 MB 的 LoRA adapter
        │
        ├──▶ inference.py        # 底座 + adapter 直接推理
        └──▶ merge_lora.py       # 合并成完整模型，独立部署
```

---

## 2. 目录结构

```
finetune_demo/
├── config.yaml              # 所有超参集中在这里（模型/LoRA/数据/训练/输出）
├── requirements.txt         # 依赖
├── data/
│   └── sample_data.jsonl    # 训练数据（chat 格式，一行一条）
├── src/
│   ├── dataset.py           # 【数据层】JSONL -> 带 mask 的 tensor
│   ├── model.py             # 【模型层】加载底座 + LoRA 注入
│   ├── train.py             # 【训练层】Trainer 主循环
│   ├── inference.py         # 【推理层】加载 adapter 对话
│   └── merge_lora.py        # 【导出层】adapter 合并回底座
└── outputs/                 # 训练产物（已 gitignore）
```

**分层职责**：`dataset.py` 只管数据、`model.py` 只管模型构造、`train.py` 负责编排，
三者互不耦合——这是微调工程最常见的拆法，换模型或换数据都不用动另外两层。

---

## 3. 数据格式

每行一个 JSON，`messages` 是标准 chat 结构：

```json
{"messages": [
  {"role": "user", "content": "LoRA 的原理是什么？"},
  {"role": "assistant", "content": "LoRA 冻结原始权重，在旁路插入两个低秩矩阵……"}
]}
```

> 想做多轮对话，就往 `messages` 里继续追加 `user`/`assistant` 即可。
> 需要固定人设时，把 `system` 写进 `config.yaml` 的 `data.system_prompt`，
> 训练和推理会自动用同一份。

---

## 4. 运行

```bash
cd finetune_demo

# 1) 安装依赖
pip install -r requirements.txt

# 2) 训练（生成 outputs/adapter/）
python src/train.py --config config.yaml

# 3) 推理测试
python src/inference.py --prompt "LoRA 的原理是什么？"

# 4) 可选：合并成独立模型
python src/merge_lora.py \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --adapter outputs/adapter \
    --out outputs/merged
```

---

## 5. 三个关键概念（看代码时对照）

### 5.1 为什么只训练 LoRA 矩阵
`model.py` 里 `get_peft_model` 会冻结底座全部权重，只在指定层（`q_proj`/`v_proj` 等）
旁路挂上两个低秩矩阵 A、B。可训练参数从 15 亿降到约 800 万（**0.5%**），
所以显存占用和存储体积都大幅下降。`train.py` 里的 `print_trainable_parameters()`
会把这个比例打印出来——跑一次就能直观看到。

### 5.2 为什么屏蔽 prompt 的 loss
`dataset.py` 里把 prompt 位置的 `labels` 设为 `-100`（PyTorch 会忽略该位置）。
不屏蔽的话模型会去学"复现问题"，训练信号被稀释；屏蔽后梯度只来自回答部分，
训练目标更聚焦。这是 SFT 最容易出错、也最值得先看的一处。

### 5.3 adapter 与合并的区别
- **adapter 模式**：底座 + 小文件，推理时要 `PeftModel.from_pretrained`，省空间、可热插拔；
- **合并模式**：`merge_and_unload()` 把增量权重加回底座，导出的是一个普通模型，
  部署端不需要装 peft，但体积和底座一样大。

---

## 6. 参数速查

| 位置 | 参数 | 作用 | 默认 |
|------|------|------|------|
| `lora.r` | 秩 | 表达能力 vs 显存 | 16 |
| `lora.alpha` | 缩放 | 一般取 r 的 1~2 倍 | 32 |
| `lora.target_modules` | 注入层 | 越全效果越好、越慢 | 7 个投影层 |
| `train.lr` | 学习率 | LoRA 通常比全参大 10 倍 | 2e-4 |
| `train.grad_accum` | 梯度累积 | 小显存换等效大 batch | 4 |
| `data.max_length` | 截断长度 | 越大越吃显存 | 512 |

---

## 7. 常见问题

| 问题 | 解决 |
|------|------|
| CUDA OOM | 调小 `train.batch_size`、调大 `grad_accum`、调小 `data.max_length` |
| bf16 报错（老卡） | `config.yaml` 里改 `dtype: float16`、`bf16: false` |
| 训练 loss 不降 | 检查 `dataset.py` 的 mask 是否正确；学习率调大或数据量增加 |
| 模型输出复读问题 | prompt 没屏蔽干净，检查 `prompt_len` 的计算 |
| 推理答非所问 | adapter 路径不对，或训练轮数太少；先用 8 条样例数据确认流程跑通 |
