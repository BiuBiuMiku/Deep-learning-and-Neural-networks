"""
《动手学深度学习（PyTorch版）》9.1.3：
使用nn.GRU简洁实现字符级语言模型

依赖：
    pip install torch

说明：
1. 不依赖d2l。
2. 自动读取《时光机器》文本，不保存数据文件。
3. 使用nn.GRU封装GRU内部的门和时间步循环。
4. 使用nn.Linear把隐状态转换成词表logits。
5. 使用torch.optim.SGD更新参数。
6. 默认训练100轮；书中使用500轮。
"""

import math
import random
import re
import time
import urllib.request
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F


# ============================================================
# 1. 超参数
# ============================================================

DATA_URL = "https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"

BATCH_SIZE = 32
NUM_STEPS = 35
NUM_HIDDENS = 256
NUM_LAYERS = 1

# 快速测试可设为2；书中使用500
NUM_EPOCHS = 100

LEARNING_RATE = 1.0
MAX_TOKENS = 10_000
CLIPPING_THETA = 1.0
RANDOM_SEED = 42


# ============================================================
# 2. 读取数据
# ============================================================

def read_time_machine():
    """读取并清洗《时光机器》文本。"""

    print("正在读取《时光机器》数据……")

    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        text = response.read().decode("utf-8")

    return [
        re.sub("[^A-Za-z]+", " ", line).strip().lower()
        for line in text.splitlines()
    ]


# ============================================================
# 3. 字符词表
# ============================================================

class Vocab:
    """建立字符与索引之间的双向映射。"""

    def __init__(self, tokens):
        counter = Counter(
            token
            for line in tokens
            for token in line
        )

        # 0号索引留给未知字符
        self.idx_to_token = ["<unk>"]

        self.idx_to_token += [
            token
            for token, frequency in sorted(
                counter.items(),
                key=lambda item: (-item[1], item[0])
            )
        ]

        self.token_to_idx = {
            token: index
            for index, token in enumerate(self.idx_to_token)
        }

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """字符转索引。"""

        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)

        return [self[token] for token in tokens]

    def to_tokens(self, indices):
        """索引转字符。"""

        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]

        return [
            self.idx_to_token[index]
            for index in indices
        ]


def load_corpus_time_machine(max_tokens=10_000):
    """将文本转换成一维字符索引序列。"""

    lines = read_time_machine()

    # 字符级词元化
    tokens = [
        list(line)
        for line in lines
    ]

    vocab = Vocab(tokens)

    corpus = [
        vocab[token]
        for line in tokens
        for token in line
    ]

    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    return corpus, vocab


# ============================================================
# 4. 顺序划分数据
# ============================================================

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    按顺序生成小批量。

    X和Y形状：
        (batch_size, num_steps)

    Y比X向后错开一个字符。
    """

    offset = random.randint(0, num_steps)

    num_tokens = (
        (len(corpus) - offset - 1)
        // batch_size
        * batch_size
    )

    Xs = torch.tensor(
        corpus[offset:offset + num_tokens],
        dtype=torch.long
    )

    Ys = torch.tensor(
        corpus[offset + 1:offset + 1 + num_tokens],
        dtype=torch.long
    )

    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)

    num_batches = Xs.shape[1] // num_steps

    for start in range(
        0,
        num_batches * num_steps,
        num_steps
    ):
        X = Xs[:, start:start + num_steps]
        Y = Ys[:, start:start + num_steps]

        yield X, Y


class SeqDataLoader:
    """使数据可以被反复遍历多个epoch。"""

    def __init__(
        self,
        batch_size,
        num_steps,
        max_tokens=10_000
    ):
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.corpus, self.vocab = load_corpus_time_machine(
            max_tokens=max_tokens
        )

    def __iter__(self):
        return seq_data_iter_sequential(
            corpus=self.corpus,
            batch_size=self.batch_size,
            num_steps=self.num_steps
        )


# ============================================================
# 5. 使用nn.GRU定义模型
# ============================================================

class GRULanguageModel(nn.Module):
    """
    使用PyTorch高级API实现的字符级GRU语言模型。

    结构：

        字符索引
            ↓
        独热编码
            ↓
        nn.GRU
            ↓
        所有时间步的隐状态
            ↓
        nn.Linear
            ↓
        词表上的logits
    """

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        num_layers=1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

        # nn.GRU内部包含：
        # 1. 更新门；
        # 2. 重置门；
        # 3. 候选隐状态；
        # 4. 沿时间步的循环计算。
        self.gru = nn.GRU(
            input_size=vocab_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            batch_first=False
        )

        # nn.GRU只输出隐状态。
        #
        # 语言模型还需要将每个隐状态映射成词表上的logits。
        self.output_layer = nn.Linear(
            in_features=num_hiddens,
            out_features=vocab_size
        )

    def forward(self, inputs, state):
        """
        前向传播。

        inputs原始形状：
            (batch_size, num_steps)

        state形状：
            (num_layers, batch_size, num_hiddens)
        """

        # ----------------------------------------------------
        # 第一步：转置并独热编码
        # ----------------------------------------------------
        #
        # 转置前：
        #   (batch_size, num_steps)
        #
        # 转置后：
        #   (num_steps, batch_size)
        #
        # 独热编码后：
        #   (num_steps, batch_size, vocab_size)
        X = F.one_hot(
            inputs.T.long(),
            num_classes=self.vocab_size
        ).float()

        # ----------------------------------------------------
        # 第二步：调用nn.GRU
        # ----------------------------------------------------
        #
        # Y保存所有时间步的隐状态：
        #   (num_steps, batch_size, num_hiddens)
        #
        # state保存最后时间步的隐状态：
        #   (num_layers, batch_size, num_hiddens)
        Y, state = self.gru(X, state)

        # ----------------------------------------------------
        # 第三步：合并时间步和批量维
        # ----------------------------------------------------
        #
        # 原始Y：
        #   (num_steps, batch_size, num_hiddens)
        #
        # 展平后：
        #   (num_steps * batch_size, num_hiddens)
        Y = Y.reshape(
            -1,
            Y.shape[-1]
        )

        # ----------------------------------------------------
        # 第四步：计算词表logits
        # ----------------------------------------------------
        #
        # 输出形状：
        #   (num_steps * batch_size, vocab_size)
        output = self.output_layer(Y)

        return output, state

    def begin_state(self, batch_size, device):
        """
        创建初始隐状态。

        普通单向GRU的状态形状：

            (num_layers, batch_size, num_hiddens)

        当前配置下：

            (1, batch_size, 256)
        """

        return torch.zeros(
            (
                self.num_layers,
                batch_size,
                self.num_hiddens
            ),
            device=device
        )


# ============================================================
# 6. 文本预测
# ============================================================

def predict_ch8(prefix, num_preds, net, vocab, device):
    """根据前缀生成后续字符。"""

    if not prefix:
        raise ValueError("prefix不能为空")

    state = net.begin_state(
        batch_size=1,
        device=device
    )

    outputs = [vocab[prefix[0]]]

    def get_input():
        return torch.tensor(
            [[outputs[-1]]],
            device=device
        )

    with torch.no_grad():

        # 使用真实前缀预热隐状态
        for character in prefix[1:]:
            _, state = net(get_input(), state)
            outputs.append(vocab[character])

        # 正式生成
        for _ in range(num_preds):
            y_hat, state = net(get_input(), state)

            next_index = int(
                y_hat.argmax(dim=1).item()
            )

            outputs.append(next_index)

    return "".join(vocab.to_tokens(outputs))


# ============================================================
# 7. 训练一个epoch
# ============================================================

def train_epoch_ch8(
    net,
    train_iter,
    loss,
    optimizer,
    device
):
    """训练一个完整epoch。"""

    state = None
    total_loss = 0.0
    total_tokens = 0

    start_time = time.perf_counter()

    for X, Y in train_iter:

        # 第一个小批量初始化状态
        if state is None:
            state = net.begin_state(
                batch_size=X.shape[0],
                device=device
            )
        else:
            # 保留上一批量最终状态的数值，
            # 但切断上一批量的计算图。
            #
            # 当前GRU的state是一个张量。
            state = state.detach()

        # 标签顺序必须与模型输出的“时间优先”顺序一致
        y = Y.T.reshape(-1)

        X = X.to(device)
        y = y.to(device)

        # 前向传播
        y_hat, state = net(X, state)

        # CrossEntropyLoss直接接收logits
        batch_loss = loss(
            y_hat,
            y.long()
        )

        # PyTorch梯度默认累加，所以先清空旧梯度
        optimizer.zero_grad()

        # 通过时间反向传播
        batch_loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(
            net.parameters(),
            max_norm=CLIPPING_THETA
        )

        # 更新nn.GRU和nn.Linear中的全部参数
        optimizer.step()

        num_tokens = y.numel()

        total_loss += batch_loss.item() * num_tokens
        total_tokens += num_tokens

    elapsed_time = time.perf_counter() - start_time

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)
    speed = total_tokens / elapsed_time

    return perplexity, speed


# ============================================================
# 8. 完整训练
# ============================================================

def train_ch8(
    net,
    train_iter,
    vocab,
    lr,
    num_epochs,
    device
):
    """训练多个epoch并定期生成文本。"""

    loss = nn.CrossEntropyLoss()

    # 优化器会自动获得：
    # 1. nn.GRU的全部内部参数；
    # 2. nn.Linear的全部参数。
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr
    )

    for epoch in range(num_epochs):
        perplexity, speed = train_epoch_ch8(
            net=net,
            train_iter=train_iter,
            loss=loss,
            optimizer=optimizer,
            device=device
        )

        should_print = (
            epoch == 0
            or (epoch + 1) % 10 == 0
            or epoch + 1 == num_epochs
        )

        if should_print:
            generated = predict_ch8(
                prefix="time traveller ",
                num_preds=50,
                net=net,
                vocab=vocab,
                device=device
            )

            print(
                f"epoch {epoch + 1:>3d} | "
                f"困惑度 {perplexity:>7.2f} | "
                f"速度 {speed:>9.1f} 词元/秒"
            )

            print(generated)

    print("\n训练完成。")

    print(
        predict_ch8(
            prefix="time traveller ",
            num_preds=50,
            net=net,
            vocab=vocab,
            device=device
        )
    )

    print(
        predict_ch8(
            prefix="traveller",
            num_preds=50,
            net=net,
            vocab=vocab,
            device=device
        )
    )


# ============================================================
# 9. 主程序
# ============================================================

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"计算设备：{device}")

    train_iter = SeqDataLoader(
        batch_size=BATCH_SIZE,
        num_steps=NUM_STEPS,
        max_tokens=MAX_TOKENS
    )

    vocab = train_iter.vocab

    print(f"词表大小：{len(vocab)}")
    print(f"语料词元数：{len(train_iter.corpus)}")

    model = GRULanguageModel(
        vocab_size=len(vocab),
        num_hiddens=NUM_HIDDENS,
        num_layers=NUM_LAYERS
    ).to(device)

    # --------------------------------------------------------
    # 检查张量形状
    # --------------------------------------------------------

    sample_X, sample_Y = next(iter(train_iter))

    sample_state = model.begin_state(
        batch_size=sample_X.shape[0],
        device=device
    )

    with torch.no_grad():
        sample_output, sample_new_state = model(
            sample_X.to(device),
            sample_state
        )

    print("\n形状检查：")
    print(f"输入X：{tuple(sample_X.shape)}")
    print(f"标签Y：{tuple(sample_Y.shape)}")
    print(f"初始状态：{tuple(sample_state.shape)}")
    print(f"输出logits：{tuple(sample_output.shape)}")
    print(f"最终状态：{tuple(sample_new_state.shape)}")

    print("\n训练前预测：")

    print(
        predict_ch8(
            prefix="time traveller ",
            num_preds=20,
            net=model,
            vocab=vocab,
            device=device
        )
    )

    print("\n开始训练：")

    train_ch8(
        net=model,
        train_iter=train_iter,
        vocab=vocab,
        lr=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        device=device
    )


if __name__ == "__main__":
    main()