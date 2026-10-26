import math

import torch
from torch import nn
import torch.nn.functional as F


# =========================
# 1. 准备一个很小的翻译数据集
# =========================

torch.manual_seed(7)
torch.set_num_threads(1)

# 有 CUDA 就使用 CUDA，否则使用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 每一项是「英语句子，法语句子」。
# 数据很少，所以模型的任务是学会并复现这些例子，而不是通用翻译。
pairs = [
    ("i eat apples", "je mange des pommes"),
    ("i eat pears", "je mange des poires"),
    ("you eat apples", "tu manges des pommes"),
    ("you eat pears", "tu manges des poires"),
    ("i sleep", "je dors"),
    ("you sleep", "tu dors"),
]


class Vocab:
    """建立词元与整数索引的双向对应关系。"""

    def __init__(self, sentences):
        # <pad>：把不同长度的句子补到同样长度。
        # <bos>：解码器开始生成句子时的起点。
        # <eos>：表示句子结束。
        words = {word for sentence in sentences for word in sentence.split()}
        self.itos = ["<pad>", "<bos>", "<eos>"] + sorted(words)

        # itos[index] -> 词元；stoi[词元] -> index。
        self.stoi = {word: index for index, word in enumerate(self.itos)}

    def encode(self, sentence):
        """把一句话转换为整数索引列表。"""
        return [self.stoi[word] for word in sentence.split()]

    def __len__(self):
        return len(self.itos)


# 英语和法语各有自己的词表；两个词表的同一索引不表示翻译关系。
src_vocab = Vocab([english for english, _ in pairs])
tgt_vocab = Vocab([french for _, french in pairs])


def pad_batch(sequences, pad_id):
    """把一个批次的索引列表补齐，得到形状为 [批量大小, 句子长度] 的张量。"""
    max_len = max(len(sequence) for sequence in sequences)
    padded = [
        sequence + [pad_id] * (max_len - len(sequence))
        for sequence in sequences
    ]
    return torch.tensor(padded, dtype=torch.long, device=device)


# 英语输入：句子末尾加 <eos>。
src = pad_batch(
    [src_vocab.encode(english) + [src_vocab.stoi["<eos>"]] for english, _ in pairs],
    src_vocab.stoi["<pad>"],
)

# 法语完整序列：开头加 <bos>，结尾加 <eos>。
tgt_full = pad_batch(
    [[tgt_vocab.stoi["<bos>"]] + tgt_vocab.encode(french) + [tgt_vocab.stoi["<eos>"]] for _, french in pairs],
    tgt_vocab.stoi["<pad>"],
)

# 例子：完整法语序列是
# <bos> je mange des pommes <eos>
#
# 解码器输入：<bos> je mange des pommes
# 正确答案：   je   mange des pommes <eos>
#
# 两行错开一位。输入位置 t 的任务是预测答案位置 t。
dec_in = tgt_full[:, :-1]
labels = tgt_full[:, 1:]


# =========================
# 2. 位置编码
# =========================

class PositionalEncoding(nn.Module):
    """为每个词元向量加入其位置的信息。"""

    def __init__(self, d_model, max_len=100):
        super().__init__()

        # pos 的形状：[最大长度, 1]；值依次为 0、1、2……
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        # even 是偶数特征下标：0、2、4……
        even = torch.arange(0, d_model, 2, dtype=torch.float32)

        # angle[pos, j] = pos / 10000^(2j/d_model)。
        # 这里 even 的值已经是 2j，所以写成 even / d_model。
        angle = pos / (10000 ** (even / d_model))

        table = torch.zeros(max_len, d_model)
        table[:, 0::2] = torch.sin(angle)  # 偶数特征使用 sin。
        table[:, 1::2] = torch.cos(angle)  # 奇数特征使用 cos。

        # 形状变成 [1, 最大长度, d_model]。
        # register_buffer 表示它会随模型移动到 CPU/GPU，但不是待训练参数。
        self.register_buffer("table", table.unsqueeze(0))

    def forward(self, x):
        # x：[B, T, d_model]。
        # 第 i 个位置的输出 = 该词元的嵌入向量 + 第 i 个位置的编码向量。
        return x + self.table[:, :x.shape[1]]


# =========================
# 3. 手写多头注意力
# =========================

class MultiHeadAttention(nn.Module):
    """
    注意力计算：
        Q = query · W_Q
        K = key_value · W_K
        V = key_value · W_V
        A = softmax(QK^T / sqrt(d_head) + 掩码)
        输出 = concat(各头的 A·V) · W_O

    注意：query 和 key_value 可以是同一个张量，也可以不同。
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 四个线性层的权重都是通过训练学到的。
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # 输入 x：[B, T, d_model]。
        # 先把最后一维分成 n_heads 个头，再把头移到第二维。
        # 输出：[B, n_heads, T, d_head]。
        batch, steps, _ = x.shape
        return x.reshape(batch, steps, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, query, key_value, key_valid, causal=False):
        """
        query：[B, Tq, d_model]，提供「由哪个位置发起查询」。
        key_value：[B, Tk, d_model]，提供「有哪些位置可供查询及读取」。
        key_valid：[B, Tk]，True 表示该键位置是真实词元，False 表示 <pad>。
        causal：True 时，查询位置不能读取其后面的目标词元。
        """

        q = self.split_heads(self.w_q(query))      # [B, H, Tq, Dh]
        k = self.split_heads(self.w_k(key_value))  # [B, H, Tk, Dh]
        v = self.split_heads(self.w_v(key_value))  # [B, H, Tk, Dh]

        # 每个查询位置与每个键位置做点积。
        # k.transpose(-2, -1) 的形状是 [B, H, Dh, Tk]。
        # scores[b, h, i, j] 是第 b 句、第 h 个头中，
        # 查询位置 i 与键位置 j 的匹配分数。
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        # scores：[B, H, Tq, Tk]。

        # 把英语或法语中的 <pad> 键位置屏蔽掉。
        # -1e9 经 softmax 后对应的权重几乎为 0。
        scores = scores.masked_fill(~key_valid[:, None, None, :], -1e9)

        if causal:
            # 仅用于解码器的目标侧自注意力。
            # allowed[i, j] 为 True 表示第 i 个查询位置允许查看第 j 个键位置。
            # 下三角形使 j <= i：可以看自己及之前，不能看未来。
            tq, tk = query.shape[1], key_value.shape[1]
            allowed = torch.tril(torch.ones(tq, tk, dtype=torch.bool, device=query.device))
            scores = scores.masked_fill(~allowed[None, None], -1e9)

        # 沿键位置 Tk 做 softmax。
        # 对固定的 b、h、i，各个 j 的注意力权重之和为 1。
        weights = F.softmax(scores, dim=-1)  # [B, H, Tq, Tk]

        # 第 i 个查询位置的结果是所有 V 向量的加权和。
        context = weights @ v  # [B, H, Tq, Dh]

        # 把多个头拼接回 d_model 维。
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(query.shape[0], query.shape[1], -1)  # [B, Tq, d_model]

        return self.w_o(context)  # [B, Tq, d_model]


# =========================
# 4. FFN、编码器层、解码器层
# =========================

class FeedForward(nn.Module):
    """对每个位置分别使用同一个两层 MLP；位置之间不会在这里相互读取。"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # 对每个位置的向量 u：
        # FFN(u) = ReLU(uW1 + b1)W2 + b2。
        # [B, T, d_model] -> [B, T, d_ff] -> [B, T, d_model]。
        return self.linear2(F.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """一个编码器层：自注意力 → Add & Norm → FFN → Add & Norm。"""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_valid):
        # query、key、value 都来自英语当前位置所在的整句表示 x。
        # 因此一个英语位置可以读取所有非 <pad> 的英语位置。
        x = self.norm1(x + self.self_attn(x, x, src_valid))

        # FFN 分别处理每个位置；残差中的 x 保留送入 FFN 前的信息。
        return self.norm2(x + self.ffn(x))


class DecoderLayer(nn.Module):
    """
    一个解码器层：
    目标侧因果自注意力 → Add & Norm
    读取编码器输出的交叉注意力 → Add & Norm
    FFN → Add & Norm
    """

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc, tgt_valid, src_valid):
        # 第一步：法语位置之间做自注意力。
        # 因果掩码保证第 i 个位置最多只能看第 i 个及之前的位置。
        x = self.norm1(x + self.self_attn(x, x, tgt_valid, causal=True))

        # 第二步：法语位置读取英语编码结果。
        # Q 来自当前法语表示 x；K、V 来自英语编码结果 enc。
        # 因此分数矩阵的后两维是 [法语位置数, 英语位置数]。
        x = self.norm2(x + self.cross_attn(x, enc, src_valid))

        # 第三步：对每个法语位置分别做 FFN。
        return self.norm3(x + self.ffn(x))


# =========================
# 5. 拼成完整的 Transformer
# =========================

class Transformer(nn.Module):
    def __init__(self, src_size, tgt_size, d_model=32, n_heads=4, d_ff=64, n_layers=2):
        super().__init__()

        self.d_model = d_model

        # 英语词表与法语词表不同，因此分别建立嵌入层。
        self.src_embedding = nn.Embedding(src_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_size, d_model)

        self.src_position = PositionalEncoding(d_model)
        self.tgt_position = PositionalEncoding(d_model)

        # 本例堆叠两个编码器层、两个解码器层。
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])

        # 把每个法语位置的 d_model 维向量变成目标词表大小的 logits。
        self.output = nn.Linear(d_model, tgt_size)

    def encode(self, src_ids, src_valid):
        # src_ids：[B, S]，其中 S 是英语序列长度。
        # 嵌入后：[B, S, d_model]。
        x = self.src_embedding(src_ids) * math.sqrt(self.d_model)
        x = self.src_position(x)

        # 整个英语句子依次通过各编码器层。
        for layer in self.encoder:
            x = layer(x, src_valid)

        # enc：[B, S, d_model]。
        # 每个英语位置都得到结合其他英语位置后的表示。
        return x

    def decode(self, tgt_ids, enc, tgt_valid, src_valid):
        # tgt_ids：[B, T]，其中 T 是当前法语输入长度。
        x = self.tgt_embedding(tgt_ids) * math.sqrt(self.d_model)
        x = self.tgt_position(x)

        for layer in self.decoder:
            x = layer(x, enc, tgt_valid, src_valid)

        # logits：[B, T, 法语词表大小]。
        # logits[b, t, v] 是第 b 句、第 t 个预测位置对词元 v 的原始分数。
        return self.output(x)

    def forward(self, src_ids, tgt_ids, src_pad_id, tgt_pad_id):
        # True 表示真实词元；False 表示补齐用的 <pad>。
        src_valid = src_ids != src_pad_id
        tgt_valid = tgt_ids != tgt_pad_id

        # 先完成英语编码，再进行法语解码。
        enc = self.encode(src_ids, src_valid)
        return self.decode(tgt_ids, enc, tgt_valid, src_valid)


model = Transformer(len(src_vocab), len(tgt_vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


# =========================
# 6. 训练
# =========================

for epoch in range(1, 301):
    model.train()
    optimizer.zero_grad()

    # 一次处理整个批次。
    # 虽然训练时所有法语词元都已提供给 dec_in，
    # 因果掩码仍保证位置 t 看不到位置 t 后面的词元。
    logits = model(src, dec_in, src_vocab.stoi["<pad>"], tgt_vocab.stoi["<pad>"])

    # CrossEntropyLoss 接收 logits，不需要事先做 softmax。
    # 把 [B, T, V] 与 [B, T] 展平，逐个预测位置计算损失。
    # 标签为 <pad> 的位置不计入损失。
    loss = F.cross_entropy(
        logits.reshape(-1, len(tgt_vocab)),
        labels.reshape(-1),
        ignore_index=tgt_vocab.stoi["<pad>"],
    )

    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 100 == 0:
        print(f"epoch={epoch:3d}, loss={loss.item():.4f}")


# =========================
# 7. 预测：逐词生成
# =========================

@torch.no_grad()
def translate(sentence, max_steps=12):
    model.eval()

    # 待翻译的英语句子先变成索引，再完整经过编码器。
    src_ids = torch.tensor(
        [src_vocab.encode(sentence) + [src_vocab.stoi["<eos>"]]],
        dtype=torch.long,
        device=device,
    )
    src_valid = src_ids != src_vocab.stoi["<pad>"]
    enc = model.encode(src_ids, src_valid)

    # 起初只知道法语句子开始符号。
    generated = [tgt_vocab.stoi["<bos>"]]

    for _ in range(max_steps):
        # 把目前已生成的整个法语前缀送入解码器。
        tgt_ids = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_valid = tgt_ids != tgt_vocab.stoi["<pad>"]

        logits = model.decode(tgt_ids, enc, tgt_valid, src_valid)

        # 最后一个位置的 logits 用来决定「下一个法语词元」。
        next_id = int(logits[0, -1].argmax())

        if next_id == tgt_vocab.stoi["<eos>"]:
            break

        generated.append(next_id)

    # 去掉开头的 <bos>，把索引转换回法语词元。
    return " ".join(tgt_vocab.itos[index] for index in generated[1:])


print("prediction:", translate("i eat apples"))