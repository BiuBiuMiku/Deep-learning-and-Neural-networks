"""
《动手学深度学习（PyTorch版）》9.2.2：
从零开始实现LSTM字符级语言模型

依赖：
    pip install torch

说明：
1. 不依赖d2l。
2. 自动读取《时光机器》文本，不保存数据文件。
3. LSTM的所有门、参数和时间步循环全部手写。
4. 使用顺序划分，相邻小批量之间传递(H, C)。
5. 默认训练100轮；书中使用500轮。
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

# 快速验证可以改成2；书中使用500
NUM_EPOCHS = 100

LEARNING_RATE = 1.0
MAX_TOKENS = 10_000
CLIPPING_THETA = 1.0
RANDOM_SEED = 42


# ============================================================
# 2. 读取并清洗数据
# ============================================================

def read_time_machine():
    """读取并清洗《时光机器》文本。"""

    print("正在读取《时光机器》数据……")

    # 直接读入内存，不创建数据文件
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

        # 索引0留给未知字符
        self.idx_to_token = ["<unk>"]

        # 按出现频率从高到低排列字符
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
        """字符转换成索引。"""

        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)

        return [self[token] for token in tokens]

    def to_tokens(self, indices):
        """索引转换回字符。"""

        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]

        return [
            self.idx_to_token[index]
            for index in indices
        ]


def load_corpus_time_machine(max_tokens=10_000):
    """建立字符级语料库和词表。"""

    lines = read_time_machine()

    # 字符级词元化：
    # "time" -> ['t', 'i', 'm', 'e']
    tokens = [
        list(line)
        for line in lines
    ]

    vocab = Vocab(tokens)

    # 展平并转换成字符索引
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
    生成顺序相邻的小批量。

    X和Y形状均为：
        (batch_size, num_steps)

    Y相对X向后错开一个字符。
    """

    # 每个epoch随机跳过开头的一小段
    offset = random.randint(0, num_steps)

    # 保证词元数量可以被batch_size整除
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
    """使语料库可以反复按epoch遍历。"""

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
# 5. 初始化LSTM参数
# ============================================================

def get_lstm_params(vocab_size, num_hiddens, device):
    """
    创建LSTM的全部可学习参数。

    因为使用独热编码：
        num_inputs = vocab_size

    因为预测下一个字符：
        num_outputs = vocab_size
    """

    num_inputs = vocab_size
    num_outputs = vocab_size

    def normal(shape):
        """使用标准差0.01的正态分布初始化权重。"""

        return (
            torch.randn(size=shape, device=device)
            * 0.01
        )

    def three():
        """
        创建一组门或候选记忆所需的三个参数：

        1. 输入权重：
           (num_inputs, num_hiddens)

        2. 隐状态权重：
           (num_hiddens, num_hiddens)

        3. 偏置：
           (num_hiddens,)
        """

        return (
            normal((num_inputs, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device=device)
        )

    # 输入门：
    # I = sigmoid(X @ W_xi + H @ W_hi + b_i)
    W_xi, W_hi, b_i = three()

    # 遗忘门：
    # F_gate = sigmoid(X @ W_xf + H @ W_hf + b_f)
    W_xf, W_hf, b_f = three()

    # 输出门：
    # O = sigmoid(X @ W_xo + H @ W_ho + b_o)
    W_xo, W_ho, b_o = three()

    # 候选记忆：
    # C_tilde = tanh(X @ W_xc + H @ W_hc + b_c)
    W_xc, W_hc, b_c = three()

    # 隐状态到词表logits的输出层
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [
        W_xi, W_hi, b_i,
        W_xf, W_hf, b_f,
        W_xo, W_ho, b_o,
        W_xc, W_hc, b_c,
        W_hq, b_q
    ]

    # 告诉PyTorch这些参数需要计算梯度
    for param in params:
        param.requires_grad_(True)

    return params


# ============================================================
# 6. 初始化双状态(H, C)
# ============================================================

def init_lstm_state(batch_size, num_hiddens, device):
    """
    创建LSTM初始状态。

    H：隐状态
    C：记忆元

    两者形状都是：
        (batch_size, num_hiddens)
    """

    H = torch.zeros(
        (batch_size, num_hiddens),
        device=device
    )

    C = torch.zeros(
        (batch_size, num_hiddens),
        device=device
    )

    # LSTM状态是包含两个张量的元组
    return H, C


# ============================================================
# 7. 手写LSTM前向传播
# ============================================================

def lstm(inputs, state, params):
    """
    手写LSTM前向传播。

    inputs形状：
        (num_steps, batch_size, vocab_size)

    state：
        (H, C)

    返回：
        所有时间步的logits；
        最后时间步的(H, C)。
    """

    (
        W_xi, W_hi, b_i,
        W_xf, W_hf, b_f,
        W_xo, W_ho, b_o,
        W_xc, W_hc, b_c,
        W_hq, b_q
    ) = params

    # 从状态元组中分别取出隐状态和记忆元
    H, C = state

    outputs = []

    # inputs的第0维是时间步
    for X in inputs:

        # ----------------------------------------------------
        # 第一步：输入门
        # ----------------------------------------------------
        #
        # 决定候选记忆C_tilde写入多少。
        I = torch.sigmoid(
            X @ W_xi
            + H @ W_hi
            + b_i
        )

        # ----------------------------------------------------
        # 第二步：遗忘门
        # ----------------------------------------------------
        #
        # 决定旧记忆C保留多少。
        #
        # 使用F_gate作为变量名，避免与
        # torch.nn.functional的别名F重名。
        F_gate = torch.sigmoid(
            X @ W_xf
            + H @ W_hf
            + b_f
        )

        # ----------------------------------------------------
        # 第三步：输出门
        # ----------------------------------------------------
        #
        # 决定新记忆C有多少暴露成隐状态H。
        O = torch.sigmoid(
            X @ W_xo
            + H @ W_ho
            + b_o
        )

        # ----------------------------------------------------
        # 第四步：候选记忆
        # ----------------------------------------------------
        #
        # 根据当前输入X和上一个隐状态H，
        # 提出准备写入记忆元的新内容。
        C_tilde = torch.tanh(
            X @ W_xc
            + H @ W_hc
            + b_c
        )

        # ----------------------------------------------------
        # 第五步：更新记忆元
        # ----------------------------------------------------
        #
        # 新记忆 =
        #   遗忘门允许保留的旧记忆
        #   +
        #   输入门允许写入的候选记忆
        C = (
            F_gate * C
            + I * C_tilde
        )

        # ----------------------------------------------------
        # 第六步：计算新隐状态
        # ----------------------------------------------------
        #
        # 先用tanh把记忆压缩到(-1, 1)，
        # 再由输出门决定暴露多少。
        H = O * torch.tanh(C)

        # ----------------------------------------------------
        # 第七步：计算当前字符logits
        # ----------------------------------------------------
        #
        # 只有H进入输出层，C不直接参与输出。
        Y = H @ W_hq + b_q

        outputs.append(Y)

    # 每个Y形状：
    #   (batch_size, vocab_size)
    #
    # 拼接后：
    #   (num_steps * batch_size, vocab_size)
    outputs = torch.cat(outputs, dim=0)

    # H和C都要传给下一时间步或下一个小批量
    return outputs, (H, C)


# ============================================================
# 8. 模型封装
# ============================================================

class LSTMModelScratch:
    """从零实现的字符级LSTM语言模型。"""

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        device
    ):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens

        self.params = get_lstm_params(
            vocab_size=vocab_size,
            num_hiddens=num_hiddens,
            device=device
        )

    def __call__(self, X, state):
        """
        X原始形状：
            (batch_size, num_steps)

        转置和独热编码后：
            (num_steps, batch_size, vocab_size)
        """

        inputs = F.one_hot(
            X.T.long(),
            num_classes=self.vocab_size
        ).float()

        return lstm(
            inputs=inputs,
            state=state,
            params=self.params
        )

    def begin_state(self, batch_size, device):
        """初始化(H, C)。"""

        return init_lstm_state(
            batch_size=batch_size,
            num_hiddens=self.num_hiddens,
            device=device
        )


# ============================================================
# 9. 文本预测
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

        # 使用真实前缀预热H和C
        for character in prefix[1:]:
            _, state = net(get_input(), state)
            outputs.append(vocab[character])

        # 自由生成
        for _ in range(num_preds):
            y_hat, state = net(get_input(), state)

            next_index = int(
                y_hat.argmax(dim=1).item()
            )

            outputs.append(next_index)

    return "".join(vocab.to_tokens(outputs))


# ============================================================
# 10. 梯度裁剪
# ============================================================

def grad_clipping(net, theta):
    """把所有参数的整体梯度范数限制在theta以内。"""

    squared_sum = sum(
        torch.sum(param.grad ** 2)
        for param in net.params
        if param.grad is not None
    )

    norm = torch.sqrt(squared_sum)

    if norm > theta:
        with torch.no_grad():
            scale = theta / norm

            for param in net.params:
                if param.grad is not None:
                    param.grad.mul_(scale)


# ============================================================
# 11. 手写SGD
# ============================================================

def sgd(params, lr):
    """
    更新参数，并在更新后清空梯度。

    损失已经求过平均，因此不再额外除以批量大小。
    """

    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


# ============================================================
# 12. 训练一个epoch
# ============================================================

def train_epoch_ch8(net, train_iter, loss, lr, device):
    """完整训练一个epoch。"""

    state = None
    total_loss = 0.0
    total_tokens = 0

    start_time = time.perf_counter()

    for X, Y in train_iter:

        if state is None:
            # 第一个小批量初始化(H, C)
            state = net.begin_state(
                batch_size=X.shape[0],
                device=device
            )
        else:
            # state是(H, C)元组。
            #
            # 两个张量都要保留数值并切断旧计算图。
            for state_tensor in state:
                state_tensor.detach_()

        # 让标签顺序与模型输出顺序一致
        y = Y.T.reshape(-1)

        X = X.to(device)
        y = y.to(device)

        # 前向传播，同时得到新的(H, C)
        y_hat, state = net(X, state)

        batch_loss = loss(
            y_hat,
            y.long()
        )

        # 反向传播
        batch_loss.backward()

        # 梯度裁剪
        grad_clipping(
            net,
            theta=CLIPPING_THETA
        )

        # 更新参数并清空梯度
        sgd(
            net.params,
            lr=lr
        )

        num_tokens = y.numel()

        total_loss += batch_loss.item() * num_tokens
        total_tokens += num_tokens

    elapsed_time = time.perf_counter() - start_time

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)
    speed = total_tokens / elapsed_time

    return perplexity, speed


# ============================================================
# 13. 完整训练
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

    for epoch in range(num_epochs):
        perplexity, speed = train_epoch_ch8(
            net=net,
            train_iter=train_iter,
            loss=loss,
            lr=lr,
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
# 14. 主程序
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

    model = LSTMModelScratch(
        vocab_size=len(vocab),
        num_hiddens=NUM_HIDDENS,
        device=device
    )

    # --------------------------------------------------------
    # 训练前检查张量形状
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
    print(f"初始H：{tuple(sample_state[0].shape)}")
    print(f"初始C：{tuple(sample_state[1].shape)}")
    print(f"输出logits：{tuple(sample_output.shape)}")
    print(f"最终H：{tuple(sample_new_state[0].shape)}")
    print(f"最终C：{tuple(sample_new_state[1].shape)}")

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