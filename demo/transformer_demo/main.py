# 基于 Transformer 的中英文翻译模型（从原始 Annotated-Transformer 重建，含 bug fix + checkpoint）

import os
import math

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

torchtext.disable_torchtext_deprecation_warning()

# 把工作目录切到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Step 1: 数据预处理
# ============================================================

file_path = "data/cmn.txt"

data = []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            english_sentence = parts[0].strip()
            chinese_sentence = parts[1].strip()
            data.append([english_sentence, chinese_sentence])

df = pd.DataFrame(data, columns=["English", "Chinese"])
df["English"].to_csv("data/english_sentences.txt", index=False, header=False)
df["Chinese"].to_csv("data/chinese_sentences.txt", index=False, header=False)
print(df.head())

# ============================================================
# Step 2: 分词 & 词表
# ============================================================

tokenizer_en = get_tokenizer("basic_english")


def tokenizer_zh(text):
    return list(text)


def build_vocab(sentences, tokenizer):
    def yield_tokens(sentences):
        for sentence in sentences:
            yield tokenizer(sentence)

    vocab = build_vocab_from_iterator(
        yield_tokens(sentences), specials=["<unk>", "<pad>", "<bos>", "<eos>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


with open("data/english_sentences.txt", "r", encoding="utf-8") as f:
    english_sentences = [line.strip() for line in f]

with open("data/chinese_sentences.txt", "r", encoding="utf-8") as f:
    chinese_sentences = [line.strip() for line in f]

en_vocab = build_vocab(english_sentences, tokenizer_en)
zh_vocab = build_vocab(chinese_sentences, tokenizer_zh)
print(f"英文词汇表大小：{len(en_vocab)}")
print(f"中文词汇表大小：{len(zh_vocab)}")


def process_sentence(sentence, tokenizer, vocab):
    tokens = tokenizer(sentence)
    tokens = ["<bos>"] + tokens + ["<eos>"]
    indices = [vocab[token] for token in tokens]
    return indices


en_sequences = [
    process_sentence(sentence, tokenizer_en, en_vocab) for sentence in english_sentences
]
zh_sequences = [
    process_sentence(sentence, tokenizer_zh, zh_vocab) for sentence in chinese_sentences
]


class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx]), torch.tensor(
            self.trg_sequences[idx]
        )


def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)
    src_batch = pad_sequence(
        src_batch, padding_value=en_vocab["<pad>"], batch_first=True
    )
    trg_batch = pad_sequence(
        trg_batch, padding_value=zh_vocab["<pad>"], batch_first=True
    )
    return src_batch, trg_batch


dataset = TranslationDataset(en_sequences, zh_sequences)
train_data, val_data = train_test_split(dataset, test_size=0.1)

batch_size = 32
train_dataloader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

# ============================================================
# Step 3: Transformer 模型
# ============================================================


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k**-0.5

    def forward(self, Q, K, V, mask=None):
        scores = (
            torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        )  # [B, H, L, L]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)
        self.dropout = nn.Dropout(0.1)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.w_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # bug fix: mask 已由 make_*_mask 做成 [B,1,1,S] 或 [B,1,S,S]，这里不能再 unsqueeze
        output, attn = self.attention(Q, K, V, mask=mask)

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )
        output = self.fc(output)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].to(x.device)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        attn_output = self.self_attn(x, x, x, trg_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)
        return x


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, trg, enc_output, src_mask=None, trg_mask=None):
        x = self.embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, trg_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout
        )
        self.decoder = Decoder(
            trg_vocab_size, d_model, num_heads, d_ff, num_layers, dropout
        )
        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def make_src_mask(self, src):
        src_mask = (src != en_vocab["<pad>"]).unsqueeze(1).unsqueeze(2)
        return src_mask.to(device)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != zh_vocab["<pad>"]).unsqueeze(1).unsqueeze(3)
        trg_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask.unsqueeze(0).unsqueeze(1)
        return trg_mask.to(device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, src_mask, trg_mask)
        output = self.fc_out(dec_output)
        return output


# ============================================================
# Step 4: 训练 & 评估 & 推理
# ============================================================

d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 3
dropout = 0.1

model = Transformer(
    src_vocab_size=len(en_vocab),
    trg_vocab_size=len(zh_vocab),
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    dropout=dropout,
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, output.size(-1)),
            trg[:, 1:].contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                trg[:, 1:].contiguous().view(-1),
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


def translate_sentence(sentence, model, en_vocab, zh_vocab, tokenizer_en, device, max_len=50):
    model.eval()
    tokens = ["<bos>"] + tokenizer_en(sentence) + ["<eos>"]
    src_indices = [en_vocab[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        enc_output = model.encoder(src_tensor, model.make_src_mask(src_tensor))

    trg_indices = [zh_vocab["<bos>"]]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.decoder(
                trg_tensor, enc_output,
                model.make_src_mask(src_tensor),
                model.make_trg_mask(trg_tensor),
            )
            output = model.fc_out(output)
            next_token = output.argmax(dim=-1)[:, -1].item()
        trg_indices.append(next_token)
        if next_token == zh_vocab["<eos>"]:
            break

    zh_tokens = zh_vocab.lookup_tokens(trg_indices[1:-1])
    return "".join(zh_tokens)


# ============================================================
# Step 5: checkpoint 保存 / 加载
# ============================================================

checkpoint_path = "checkpoint.pt"


def save_checkpoint(model, optimizer, path):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": dict(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_layers=num_layers,
                src_vocab_size=len(en_vocab),
                trg_vocab_size=len(zh_vocab),
            ),
        },
        path,
    )
    print(f"参数已保存到 {path}")


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"已从 {path} 载入参数")


# ============================================================
# Step 6: 主流程
# ============================================================

n_epochs = 10

if os.path.exists(checkpoint_path):
    load_checkpoint(model, optimizer, checkpoint_path)
else:
    for epoch in range(n_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss = evaluate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    save_checkpoint(model, optimizer, checkpoint_path)

# 推理测试
model.eval()
test_sentences = [
    "How are you?",
    "I love you.",
    "What is your name?",
    "Good morning.",
]
print("\n=== 翻译测试 ===")
for s in test_sentences:
    print(f"  {s!r:30} → {translate_sentence(s, model, en_vocab, zh_vocab, tokenizer_en, device)}")
