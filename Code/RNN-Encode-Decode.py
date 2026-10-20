import math
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# 小语料在 CPU 上即可训练；固定随机种子便于复现。
torch.manual_seed(1)
torch.set_num_threads(1)
device = torch.device("cpu")

# 每一项都是 (英文源句, 法语目标句)。
pairs = [
    ("go .", "va ."),
    ("hi .", "salut ."),
    ("i go .", "je vais ."),
    ("you go .", "tu vas ."),
    ("i am here .", "je suis ici ."),
    ("you are here .", "tu es ici ."),
    ("i am calm .", "je suis calme ."),
    ("you are calm .", "tu es calme ."),
]


# ==================== 1. 建立两份独立词表 ====================

class Vocab:
    """把一种语言的词元映射到编号，也能把编号映射回词元。"""

    def __init__(self, sentences):
        special = ["<pad>", "<unk>", "<bos>", "<eos>"]
        words = sorted({word for sentence in sentences for word in sentence.lower().split()})

        self.idx_to_token = special + [word for word in words if word not in special]
        self.token_to_idx = {word: i for i, word in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        # 预测时遇到训练词表中没有的词，用 <unk> 的编号。
        return self.token_to_idx.get(token, self.token_to_idx["<unk>"])

    def to_tokens(self, indices):
        return [self.idx_to_token[i] for i in indices]


src_vocab = Vocab([src for src, _ in pairs])  # 英文词表
tgt_vocab = Vocab([tgt for _, tgt in pairs])  # 法语词表

# 每条序列最终占 7 个位置；其中包含 <eos> 和可能的 <pad>。
num_steps = 7


def encode_sentence(sentence, vocab, num_steps):
    """句子 → 词元编号；添加 <eos>；再截断或填充。"""
    ids = [vocab[word] for word in sentence.lower().split()]
    ids.append(vocab["<eos>"])

    ids = ids[:num_steps]  # 太长则截断。
    valid_len = len(ids)  # 有效长度包含 <eos>，不包含后面补的 <pad>。
    ids += [vocab["<pad>"]] * (num_steps - valid_len)

    return ids, valid_len


# 同一行的英文和法文始终是一对翻译。
src_rows, src_lengths, tgt_rows, tgt_lengths = [], [], [], []
for src_sentence, tgt_sentence in pairs:
    src_ids, src_len = encode_sentence(src_sentence, src_vocab, num_steps)
    tgt_ids, tgt_len = encode_sentence(tgt_sentence, tgt_vocab, num_steps)

    src_rows.append(src_ids)
    src_lengths.append(src_len)
    tgt_rows.append(tgt_ids)
    tgt_lengths.append(tgt_len)

# 每次从四个张量的相同行取数据，保证句子配对不会被打乱。
dataset = TensorDataset(torch.tensor(src_rows), torch.tensor(src_lengths), torch.tensor(tgt_rows), torch.tensor(tgt_lengths))
train_iter = DataLoader(dataset, batch_size=4, shuffle=True)


# ==================== 2. 英文编码器 ====================

class Seq2SeqEncoder(nn.Module):
    """Embedding 把英文编号变成向量；GRU 读完整条英文序列。"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X):
        # X：(批量大小, 英文序列长度)，里面是英文词表编号。
        X = self.embedding(X)
        # 现在是 (批量大小, 英文序列长度, embed_size)。

        X = X.permute(1, 0, 2)
        # GRU 默认要求 (英文序列长度, 批量大小, embed_size)。

        output, state = self.rnn(X)
        # output：最后一层在每个英文时间步的输出。
        # state：每一层读完英文序列后的最终隐状态。
        return output, state


# ==================== 3. 法语解码器 ====================

class Seq2SeqDecoder(nn.Module):
    """根据编码器状态和法语侧输入，预测下一词元的 logits。"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 每一步输入 = 法语词元向量 + 编码上下文向量。
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs):
        # enc_outputs[0] 是编码器每一步的 output；
        # enc_outputs[1] 是编码器各层的最终 state。
        # 本模型用整个 state 初始化解码器的各层隐状态。
        return enc_outputs[1]

    def forward(self, X, state):
        # X：(批量大小, 法语输入长度)，里面是法语词表编号。
        X = self.embedding(X).permute(1, 0, 2)
        # 现在是 (法语输入长度, 批量大小, embed_size)。

        # state[-1] 是传入状态的最后一层隐状态，形状 (批量大小, num_hiddens)。
        # 把它复制到本次 forward 的每个法语时间步，作为 context。
        context = state[-1].repeat(X.shape[0], 1, 1)

        # 沿特征维拼接：每步得到 embed_size + num_hiddens 维输入。
        X_and_context = torch.cat((X, context), dim=2)

        output, state = self.rnn(X_and_context, state)

        # 把每步的隐藏向量变成“对法语词表中每个词元的分数”。
        logits = self.dense(output).permute(1, 0, 2)
        # logits：(批量大小, 法语输入长度, 法语词表大小)。
        return logits, state


# ==================== 4. 先编码英文，再解码法语 ====================

class EncoderDecoder(nn.Module):
    """把两个网络按执行顺序连接起来。"""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):
        enc_outputs = self.encoder(enc_X)              # 先读完整条英文序列。
        dec_state = self.decoder.init_state(enc_outputs)  # 再建立法语解码器初始状态。
        return self.decoder(dec_X, dec_state)          # 最后处理法语侧输入。


# ==================== 5. 屏蔽 <pad> 的损失 ====================

class MaskedSoftmaxCELoss(nn.Module):
    """只让真实目标词元（包含 <eos>）参与损失。"""

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred, label, valid_len):
        # pred：(批量大小, 时间步数, 法语词表大小)，是 logits。
        # CrossEntropyLoss 要求类别维在中间，所以改成 (批量大小, 词表大小, 时间步数)。
        per_token_loss = self.cross_entropy(pred.permute(0, 2, 1), label)

        # 对每条句子：位置 < 有效长度为 True；填充位置为 False。
        positions = torch.arange(label.shape[1], device=label.device)[None, :]
        mask = positions < valid_len[:, None]

        # <pad> 位置乘 0。与书中一致，每条句子除以固定的 num_steps。
        return (per_token_loss * mask).mean(dim=1)


# ==================== 6. 创建并训练模型 ====================

embed_size = 16      # 每个词元经过 Embedding 后是 16 维。
num_hiddens = 32     # 每层 GRU 的隐状态是 32 维。
num_layers = 2       # 编码器和解码器都是两层 GRU，便于直接传递 state。
dropout = 0.1        # 两层 GRU 之间训练时使用的 dropout。

encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder, decoder).to(device)

loss_fn = MaskedSoftmaxCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

num_epochs = 120
for epoch in range(num_epochs):
    net.train()
    total_loss = 0.0

    for X, X_valid_len, Y, Y_valid_len in train_iter:
        X = X.to(device)                    # 英文词元编号
        Y = Y.to(device)                    # 正确法语译文编号，包含 <eos>、<pad>
        Y_valid_len = Y_valid_len.to(device)

        # 对批次中每条句子都创建一个开头的 <bos>。
        bos = torch.full((Y.shape[0], 1), tgt_vocab["<bos>"], dtype=torch.long, device=device)

        # 强制教学：输入是 <bos> + 正确译文去掉最后一位。
        # 例如输入 <bos> je vais .，目标是 je vais . <eos>。
        dec_input = torch.cat((bos, Y[:, :-1]), dim=1)

        optimizer.zero_grad()                  # 清掉上一个批次的梯度。
        logits, _ = net(X, dec_input)          # 内部先完整编码英文，再解码法语。
        loss = loss_fn(logits, Y, Y_valid_len).sum()
        loss.backward()                        # 对本批损失反向传播。
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # 防止梯度过大。
        optimizer.step()                       # 更新编码器、解码器和嵌入层参数。

        total_loss += loss.item()

    if (epoch + 1) % 30 == 0:
        print(f"第 {epoch + 1} 轮：每句平均损失 {total_loss / len(dataset):.4f}")


# ==================== 7. 逐词元预测 ====================

@torch.no_grad()
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    """先编码整句英文，再从 <bos> 开始逐词元生成法语。"""
    net.eval()  # 关闭训练时的 dropout。

    src_ids, _ = encode_sentence(src_sentence, src_vocab, num_steps)
    enc_X = torch.tensor([src_ids], dtype=torch.long, device=device)
    # enc_X 的形状是 (1, num_steps)：批量中只有一条英文句子。

    enc_outputs = net.encoder(enc_X)              # 先完整编码英文。
    dec_state = net.decoder.init_state(enc_outputs)

    dec_X = torch.tensor([[tgt_vocab["<bos>"]]], dtype=torch.long, device=device)
    output_ids = []

    for _ in range(num_steps):
        # 每次只送入一个法语词元；GRU 内部状态由 dec_state 延续。
        logits, dec_state = net.decoder(dec_X, dec_state)

        # 对词表维取最大值的位置，得到本步预测词元的编号。
        pred_id = int(logits.argmax(dim=2).item())

        if pred_id == tgt_vocab["<eos>"]:
            break  # 预测到句末标记就停止，不把它放进译文。

        output_ids.append(pred_id)
        dec_X = torch.tensor([[pred_id]], dtype=torch.long, device=device)
        # 下一步的输入是“刚刚预测的词”，而非正确答案。

    return " ".join(tgt_vocab.to_tokens(output_ids))


# ==================== 8. 用 BLEU 比较预测与参考译文 ====================

def bleu(pred_seq, label_seq, k):
    """计算书中的 BLEU：短句惩罚 × 各级片段匹配率的乘积。"""
    pred_tokens = pred_seq.split()
    label_tokens = label_seq.split()
    len_pred = len(pred_tokens)
    len_label = len(label_tokens)

    if len_pred < k or k < 1:
        return 0.0  # 连 k 元片段都构不成时，返回 0。

    score = math.exp(min(0, 1 - len_label / len_pred))

    for n in range(1, k + 1):
        num_matches = 0
        label_subs = defaultdict(int)

        # 统计参考译文中，每种连续 n 词片段出现的次数。
        for i in range(len_label - n + 1):
            ngram = " ".join(label_tokens[i:i + n])
            label_subs[ngram] += 1

        # 逐个检查预测片段；每匹配一次就扣掉一次可匹配次数。
        for i in range(len_pred - n + 1):
            ngram = " ".join(pred_tokens[i:i + n])
            if label_subs[ngram] > 0:
                num_matches += 1
                label_subs[ngram] -= 1

        p_n = num_matches / (len_pred - n + 1)
        score *= p_n ** (0.5 ** n)

    return score


# 这里评估的是训练时见过的句子，目的仅是检查完整流程。
print("\n训练后逐句预测：")
for english, reference in pairs:
    translation = predict_seq2seq(net, english, src_vocab, tgt_vocab, num_steps, device)
    score = bleu(translation, reference, k=2)
    print(f"{english} → {translation} | 参考：{reference} | BLEU-2：{score:.3f}")