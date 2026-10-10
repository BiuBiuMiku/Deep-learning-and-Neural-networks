"""
《动手学深度学习（PyTorch版）》9.1.2：
从零开始实现门控循环单元GRU

依赖：
    pip install torch

说明：
1. 不依赖d2l。
2. 自动读取《时光机器》数据，不保存数据文件。
3. GRU的参数、门和时间步循环全部手写。
4. 数据按顺序划分，因此相邻小批量之间传递隐状态。
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

# 一个小批量包含32条子序列
BATCH_SIZE = 32

# 每条子序列包含35个时间步
NUM_STEPS = 35

# 每个样本的隐状态包含256个数
NUM_HIDDENS = 256

# 为方便运行，默认100轮
# 书中使用500轮
NUM_EPOCHS = 100

LEARNING_RATE = 1.0

# 只使用语料库的前10000个字符
MAX_TOKENS = 10_000

# 梯度整体L2范数的最大值
CLIPPING_THETA = 1.0

RANDOM_SEED = 42


# ============================================================
# 2. 读取并清洗数据
# ============================================================

def read_time_machine():
    """
    从网络读取《时光机器》，并清洗文本。

    返回：
        lines：清洗后的文本行列表。
    """

    print("正在读取《时光机器》数据……")

    # 数据直接读取到内存，不写入本地文件
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        text = response.read().decode("utf-8")

    lines = []

    for line in text.splitlines():
        # 非英文字母替换为空格
        line = re.sub("[^A-Za-z]+", " ", line)

        # 统一转成小写
        line = line.strip().lower()

        lines.append(line)

    return lines


# ============================================================
# 3. 字符词表
# ============================================================

class Vocab:
    """
    建立字符和整数索引之间的双向映射。

    token_to_idx：
        字符 -> 索引

    idx_to_token：
        索引 -> 字符
    """

    def __init__(self, tokens):
        # 统计所有字符的出现次数
        counter = Counter(
            token
            for line in tokens
            for token in line
        )

        # 0号索引留给未知字符
        self.idx_to_token = ["<unk>"]

        # 按频率从高到低排列
        token_frequencies = sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0])
        )

        self.idx_to_token += [
            token
            for token, frequency in token_frequencies
        ]

        # 建立字符到索引的反向映射
        self.token_to_idx = {
            token: index
            for index, token in enumerate(self.idx_to_token)
        }

    def __len__(self):
        """返回词表大小。"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """将字符或字符列表转换成索引。"""

        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)

        return [self[token] for token in tokens]

    def to_tokens(self, indices):
        """将索引或索引列表转换回字符。"""

        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]

        return [
            self.idx_to_token[index]
            for index in indices
        ]


# ============================================================
# 4. 创建字符级语料库
# ============================================================

def load_corpus_time_machine(max_tokens=10_000):
    """
    将文本转换成一维字符索引序列。

    返回：
        corpus：字符索引序列
        vocab：字符词表
    """

    lines = read_time_machine()

    # 字符级词元化：
    # "time" -> ['t', 'i', 'm', 'e']
    tokens = [
        list(line)
        for line in lines
    ]

    vocab = Vocab(tokens)

    # 展平所有文本行，并转换成字符索引
    corpus = [
        vocab[token]
        for line in tokens
        for token in line
    ]

    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    return corpus, vocab


# ============================================================
# 5. 顺序划分序列
# ============================================================

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    使用顺序划分生成小批量。

    相邻小批量在原始文本中相邻，因此可以把当前批量的
    最终隐状态交给下一个批量。

    X和Y的形状：
        (batch_size, num_steps)

    Y相对于X向后错开一个字符。
    """

    # 每个epoch随机跳过开头的一小段
    offset = random.randint(0, num_steps)

    # 标签需要向后错开一位，所以长度减1；
    # 同时保证词元数量能被batch_size整除
    num_tokens = (
        (len(corpus) - offset - 1)
        // batch_size
        * batch_size
    )

    # 输入序列
    Xs = torch.tensor(
        corpus[offset:offset + num_tokens],
        dtype=torch.long
    )

    # 标签序列，比输入向后移动一个位置
    Ys = torch.tensor(
        corpus[offset + 1:offset + 1 + num_tokens],
        dtype=torch.long
    )

    # 将一个长序列分成batch_size行
    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)

    # 每行能够切出多少个长度为num_steps的片段
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
    """封装语料库，使其可以反复按epoch遍历。"""

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
# 6. 初始化GRU参数
# ============================================================

def get_gru_params(vocab_size, num_hiddens, device):
    """
    创建GRU的全部可学习参数。

    vocab_size：
        词表大小。因为使用独热编码，所以也等于输入长度。

    num_hiddens：
        隐藏单元数量。
    """

    num_inputs = vocab_size
    num_outputs = vocab_size

    def normal(shape):
        """
        使用均值0、标准差0.01的正态分布初始化权重。
        """

        return (
            torch.randn(size=shape, device=device)
            * 0.01
        )

    def three():
        """
        创建一组：

            输入权重
            隐状态权重
            偏置

        形状分别为：

            (num_inputs, num_hiddens)
            (num_hiddens, num_hiddens)
            (num_hiddens,)
        """

        return (
            normal((num_inputs, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device=device)
        )

    # 更新门参数：
    #
    # Z = sigmoid(X @ W_xz + H @ W_hz + b_z)
    W_xz, W_hz, b_z = three()

    # 重置门参数：
    #
    # R = sigmoid(X @ W_xr + H @ W_hr + b_r)
    W_xr, W_hr, b_r = three()

    # 候选隐状态参数：
    #
    # H_tilde = tanh(
    #     X @ W_xh
    #     + (R * H) @ W_hh
    #     + b_h
    # )
    W_xh, W_hh, b_h = three()

    # 输出层参数：
    #
    # Y = H @ W_hq + b_q
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [
        W_xz, W_hz, b_z,
        W_xr, W_hr, b_r,
        W_xh, W_hh, b_h,
        W_hq, b_q
    ]

    # 这些张量都需要通过反向传播学习
    for param in params:
        param.requires_grad_(True)

    return params


# ============================================================
# 7. 初始化GRU隐状态
# ============================================================

def init_gru_state(batch_size, num_hiddens, device):
    """
    创建全0初始隐状态。

    H形状：
        (batch_size, num_hiddens)

    GRU只有一个向下一时间步传递的隐状态H，
    因此返回单元素元组：

        (H,)

    外面的元组只是统一模型接口，没有增加新的状态。
    """

    H = torch.zeros(
        (batch_size, num_hiddens),
        device=device
    )

    return (H,)


# ============================================================
# 8. GRU前向传播
# ============================================================

def gru(inputs, state, params):
    """
    手写GRU的前向传播。

    inputs形状：
        (num_steps, batch_size, vocab_size)

    state：
        单元素元组(H,)

    输出：
        所有时间步的logits；
        最后一个时间步的隐状态。
    """

    (
        W_xz, W_hz, b_z,
        W_xr, W_hr, b_r,
        W_xh, W_hh, b_h,
        W_hq, b_q
    ) = params

    # 从单元素元组中取出H
    H, = state

    outputs = []

    # inputs第0维是时间步
    #
    # 每次循环中的X形状：
    #   (batch_size, vocab_size)
    for X in inputs:

        # ----------------------------------------------------
        # 第一步：计算更新门Z
        # ----------------------------------------------------
        #
        # Z形状：
        #   (batch_size, num_hiddens)
        #
        # sigmoid保证每个门值位于0和1之间。
        #
        # Z越接近1：
        #   最终状态越倾向保留旧H。
        #
        # Z越接近0：
        #   最终状态越倾向采用候选状态。
        Z = torch.sigmoid(
            X @ W_xz
            + H @ W_hz
            + b_z
        )

        # ----------------------------------------------------
        # 第二步：计算重置门R
        # ----------------------------------------------------
        #
        # R控制生成候选状态时参考多少旧状态。
        #
        # R越接近1：
        #   候选状态充分参考过去。
        #
        # R越接近0：
        #   候选状态基本忽略过去。
        R = torch.sigmoid(
            X @ W_xr
            + H @ W_hr
            + b_r
        )

        # ----------------------------------------------------
        # 第三步：计算候选隐状态
        # ----------------------------------------------------
        #
        # R * H是逐元素乘法：
        # 先使用重置门过滤旧状态，
        # 再通过W_hh参与候选状态计算。
        #
        # H_tilde形状：
        #   (batch_size, num_hiddens)
        H_tilde = torch.tanh(
            X @ W_xh
            + (R * H) @ W_hh
            + b_h
        )

        # ----------------------------------------------------
        # 第四步：得到最终隐状态
        # ----------------------------------------------------
        #
        # 旧状态的权重：
        #   Z
        #
        # 候选状态的权重：
        #   1 - Z
        #
        # 这是对每个隐藏特征分别进行的加权混合。
        H = (
            Z * H
            + (1 - Z) * H_tilde
        )

        # ----------------------------------------------------
        # 第五步：计算当前时间步的词表logits
        # ----------------------------------------------------
        #
        # H形状：
        #   (batch_size, num_hiddens)
        #
        # Y形状：
        #   (batch_size, vocab_size)
        Y = H @ W_hq + b_q

        outputs.append(Y)

    # outputs中有num_steps个张量，
    # 每个张量形状为：
    #   (batch_size, vocab_size)
    #
    # 拼接后：
    #   (num_steps * batch_size, vocab_size)
    outputs = torch.cat(outputs, dim=0)

    return outputs, (H,)


# ============================================================
# 9. 从零实现模型的通用封装
# ============================================================

class GRUModelScratch:
    """从零实现的字符级GRU语言模型。"""

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        device
    ):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens

        self.params = get_gru_params(
            vocab_size=vocab_size,
            num_hiddens=num_hiddens,
            device=device
        )

    def __call__(self, X, state):
        """
        X原始形状：
            (batch_size, num_steps)

        X.T后：
            (num_steps, batch_size)

        独热编码后：
            (num_steps, batch_size, vocab_size)
        """

        inputs = F.one_hot(
            X.T.long(),
            num_classes=self.vocab_size
        ).float()

        return gru(
            inputs=inputs,
            state=state,
            params=self.params
        )

    def begin_state(self, batch_size, device):
        """创建初始隐状态。"""

        return init_gru_state(
            batch_size=batch_size,
            num_hiddens=self.num_hiddens,
            device=device
        )


# ============================================================
# 10. 文本预测
# ============================================================

def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    根据给定前缀生成后续字符。

    前缀首先用于预热隐状态，然后才开始自由生成。
    """

    if not prefix:
        raise ValueError("prefix不能为空")

    # 只生成一条文本，所以batch_size=1
    state = net.begin_state(
        batch_size=1,
        device=device
    )

    # 先加入前缀的第一个字符
    outputs = [vocab[prefix[0]]]

    def get_input():
        """
        取最后一个字符作为下一时间步输入。

        形状：
            (batch_size=1, num_steps=1)
        """

        return torch.tensor(
            [[outputs[-1]]],
            device=device
        )

    # 预测不需要梯度
    with torch.no_grad():

        # 预热：
        # 输入前一个真实字符，更新状态，
        # 再把下一个真实字符放入outputs。
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
# 11. 梯度裁剪
# ============================================================

def grad_clipping(net, theta):
    """
    将所有参数看成一个整体，计算整体梯度L2范数。

    如果范数超过theta，则按统一比例缩小所有梯度。
    """

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
# 12. 手写SGD
# ============================================================

def sgd(params, lr):
    """
    使用当前梯度更新参数，并在更新后清空梯度。

    当前损失已经由CrossEntropyLoss求过平均，
    所以不再额外除以小批量大小。
    """

    with torch.no_grad():
        for param in params:
            param -= lr * param.grad

            # 为下一个小批量清空梯度
            param.grad.zero_()


# ============================================================
# 13. 训练一个epoch
# ============================================================

def train_epoch_ch8(net, train_iter, loss, lr, device):
    """完整训练一个epoch。"""

    state = None

    # 累计损失总和
    total_loss = 0.0

    # 累计词元数量
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
            # 顺序划分会将上一批量的最终状态传过来。
            #
            # 保留H的数值，但切断上一批量的计算图，
            # 将BPTT限制在当前批量的NUM_STEPS个时间步内。
            for state_tensor in state:
                state_tensor.detach_()

        # 标签顺序需要与模型的“时间优先”输出一致
        #
        # (batch_size, num_steps)
        #          ↓ 转置并展平
        # (num_steps * batch_size,)
        y = Y.T.reshape(-1)

        X = X.to(device)
        y = y.to(device)

        # 前向传播
        y_hat, state = net(X, state)

        # y_hat形状：
        #   (num_steps * batch_size, vocab_size)
        #
        # y形状：
        #   (num_steps * batch_size,)
        #
        # CrossEntropyLoss直接接收logits
        batch_loss = loss(
            y_hat,
            y.long()
        )

        # 反向传播
        batch_loss.backward()

        # 在更新参数前裁剪梯度
        grad_clipping(
            net,
            theta=CLIPPING_THETA
        )

        # 更新参数，并在sgd内部清空梯度
        sgd(
            net.params,
            lr=lr
        )

        num_tokens = y.numel()

        # batch_loss是平均损失，乘词元数得到损失总和
        total_loss += batch_loss.item() * num_tokens
        total_tokens += num_tokens

    elapsed_time = time.perf_counter() - start_time

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)
    speed = total_tokens / elapsed_time

    return perplexity, speed


# ============================================================
# 14. 完整训练
# ============================================================

def train_ch8(
    net,
    train_iter,
    vocab,
    lr,
    num_epochs,
    device
):
    """训练多个epoch，并定期生成文本。"""

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
# 15. 主程序
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

    model = GRUModelScratch(
        vocab_size=len(vocab),
        num_hiddens=NUM_HIDDENS,
        device=device
    )

    # --------------------------------------------------------
    # 运行训练前的形状检查
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
    print(f"输出logits：{tuple(sample_output.shape)}")
    print(f"最终H：{tuple(sample_new_state[0].shape)}")

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