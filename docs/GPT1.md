

# 1.代码

```python
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformer import MultiHeadAttention, PositionWiseFeedForward
from torch import nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import GPT2TokenizerFast

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 让CUDA报错定位准确
os.environ["TORCH_USE_CUDA_DSA"] = "1"    # 启用设备端断言，显示具体断言内容


# ——————————————————————1.硬件与参数配置————————————————————
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备{DEVICE}")

# 模型超参数（轻量化设计，控制现存占用）
VOCAB_SIZE = 50257  # GPT2小词汇表，适配tokenizer
N_LAYER = 2  # decoder block层数，GPT1原版12层
N_HEAD = 4  # 多头注意力头数，原版12头
N_EMBD = 128  # embedding维度，原版768
N_HIDDEN = 4 * N_EMBD  # 隐层维度4倍放大词嵌入维度
SEQ_LEN = 64  # 原版1024
DROPOUT = 0.1
IGNORE_INDEX = -100

# 训练参数
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 5
GRAD_CLIP = 1.0  # 梯度裁剪防止爆炸

# 数据配置（使用小数据集子集）
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
MAX_TRAIN_SAMPLES = 10000  # 仅用1万样本训练（减少数据加载压力）
MAX_VAL_SAMPLES = 1000  # 1千样本验证


# ——————————————————————2.数据预处理————————————————————
class TextDataset(Dataset):
    # 将文本转化为序列对(input_ids, targets)的列表
    def __init__(self, texts, tokenizer, seq_len, ignore_index):
        self.data = []
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id  # 获取tokenizer定义的pad_token_id，通常是0

        for text in texts:
            token_ids = tokenizer.encode(text)
            # 添加起始和结尾token
            token_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]
            # 滑动窗口大小取1
            i = 0
            while i + seq_len < len(token_ids) - 1:
                input_chunk = token_ids[i : i + seq_len]
                target_chunk = token_ids[i + 1 : i + 1 + seq_len]
                i = i + 1
                self.data.append((torch.tensor(input_chunk), torch.tensor(target_chunk)))
            input_chunk = token_ids[i : len(token_ids) - 1] + [tokenizer.pad_token_id] * (seq_len - len(token_ids) + i + 1)
            target_chunk = token_ids[i + 1 : ] + [ignore_index] * (seq_len - len(token_ids) + i + 1)
            self.data.append((torch.tensor(input_chunk), torch.tensor(target_chunk)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_and_preprocess_data(tokenizer):
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
    )
    train_texts = [text for text in dataset["train"]["text"] if text.strip()][:MAX_TRAIN_SAMPLES]
    val_texts = [text for text in dataset["validation"]["text"] if text.strip()][:MAX_VAL_SAMPLES]

    train_dataset = TextDataset(train_texts, tokenizer, SEQ_LEN, IGNORE_INDEX)
    val_dataset = TextDataset(val_texts, tokenizer, SEQ_LEN, IGNORE_INDEX)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader


# ——————————————————————3.定义模型————————————————————
class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_hidden, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionWiseFeedForward(
            embed_dim=n_embd,
            hidden_dim=n_hidden,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(n_embd)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        attn_output, _ = self.attn(x, x, x, attn_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        x = x + self.dropout2(self.ffn(x))
        x = self.norm2(x)
        return x


class GPT1(nn.Module):
    def __init__(self, vocab_size, n_embd, seq_len, n_layers, n_head, n_hidden, dropout, pad_token_id,
                 ignore_index=-100):
        super().__init__()
        # pad id，用于生成掩码
        self.pad_token_id = pad_token_id
        # ignore_index，用于计算损失时忽略，不能以正常词表索引作ignore_index，会参与计算损失
        self.ignore_index = ignore_index
        # 最大序列长度，用于校验
        self.seq_len = seq_len
        # 词表大小，用于计算损失
        self.vocab_size = vocab_size

        # decoder-only：词嵌入+位置嵌入，无段嵌入
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)

        # decoder block堆叠
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_hidden=n_hidden,
                dropout=dropout
            )
                for _ in range(n_layers)]
        )

        # 最终层归一化+语言模型头
        self.norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # 权重共享：词嵌入与输出层共享权重（GPT优化）
        self.lm_head.weight = self.token_emb.weight

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """参数初始化（GPT风格）"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 训练时要提供targets用于计算损失，推理时不用
    def forward(self, input_ids, targets=None):
        # (b, n)
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.seq_len, "输入序列过长"

        # 词嵌入+位置嵌入
        pos = torch.arange(0, seq_len, dtype=torch.long).to(device=input_ids.device)
        # (b, n, d)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        # 未来位置掩码，对位置i的token掩盖掉i+1及之后位置的token，下三角矩阵
        # (1, 1, n, n)
        future_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).unsqueeze(0).unsqueeze(
            0).bool()
        # (b, 1, 1, n)
        padding_mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        attn_mask = future_mask & padding_mask

        # 经过decoder blocks
        # (b, n, d)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, attn_mask)

        # 最终归一化+语言模型头
        # (b, n, vocab_size)
        x = self.norm(x)
        logits = self.lm_head(x)

        # 计算损失
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.ignore_index
            )
        return logits, loss


# ——————————————————————4.训练与验证逻辑————————————————————
def train_one_epoch(model, loader, optimizer, epoch):
    """单轮训练"""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1} Train")

    for input_ids, targets in pbar:
        input_ids = input_ids.to(DEVICE)
        targets = targets.to(DEVICE)

        # 前向传播
        logits, loss = model(input_ids, targets)

        # 反向传播+梯度裁剪
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # 更新参数
        optimizer.step()

        # 统计损失
        total_loss += loss.item()
        pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    return avg_loss


def validate(model, loader):
    """验证模型"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(DEVICE)
            targets = targets.to(DEVICE)

            logits, loss = model(input_ids, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


# ——————————————————————5.文本生成函数————————————————————
def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.5):
    """温度(=1.5) + 随机采样，让生成随机性更强"""
    model.eval()

    # (b, n)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 取最后SEQ_LEN个token，避免超出最大序列长度
            input_ids_chunk = input_ids[ :, -SEQ_LEN: ]
            # (b, n, vocab_size)
            logits, _ = model(input_ids_chunk)

            # 取最后一个token并应用温度
            logits = logits[ : , -1, : ] / temperature

            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


# ——————————————————————6.主函数：训练+预测————————————————————
def main():
    # 1. 加载分词器（用GPT2的轻量级分词器，适配GPT-1）
    tokenizer = GPT2TokenizerFast.from_pretrained(
        "gpt2",
        # 下载一次后就可以本地加载
        local_files_only=True
    )
    print(f"词表大小：{tokenizer.vocab_size}")
    # 复用eos_token，不能新加special token，否则nn.Embedding会解析出错
    tokenizer.pad_token = tokenizer.eos_token

    # 2.加载数据
    train_loader, val_loader = load_and_preprocess_data(tokenizer)
    print(f"训练集批次：{len(train_loader)}, 验证集批次：{len(val_loader)}")

    # 3.初始化模型
    model = GPT1(
        vocab_size=tokenizer.vocab_size,
        n_embd=N_EMBD,
        seq_len=SEQ_LEN,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        n_hidden=N_HIDDEN,
        dropout=DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=IGNORE_INDEX
    ).to(DEVICE)
    print(f"模型参数总量：{sum(p.numel() for p in model.parameters()):,}")

    # 4. 初始化优化器（用AdamW）
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # 5. 训练循环
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        val_loss = validate(model, val_loader)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "gpt1_mini_best.pth")
            print(f"最佳模型验证损失：{val_loss:.4f}")

    # 6. 加载模型并生成文本
    model.load_state_dict(torch.load("gpt1_mini_best.pth"))
    print("\n——————————生成文本示例————————")
    prompts = [
        "The meaning of life is",
        "In the future, artificial intelligence will",
        "Once upon a time, there was a"
    ]
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Gererated text: {generate_text(model, tokenizer, prompt)}")


if __name__ == main():
    main()
```