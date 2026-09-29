"""
《动手学深度学习（PyTorch版）》8.5：
循环神经网络从零开始实现

依赖：
    pip install torch

说明：
1. 不依赖 d2l 包。
2. 运行时会从网络读取《时光机器》英文文本，但不会保存数据文件。
3. 默认训练100轮，便于实际运行。
4. 书中使用500轮；想接近书中结果，可把 NUM_EPOCHS 改成500。
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
# 1. 全局配置
# ============================================================

# 《时光机器》数据地址
DATA_URL = "https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"

# 每个小批量包含多少条子序列
BATCH_SIZE = 32

# 每条子序列包含多少个时间步
NUM_STEPS = 35

# 隐状态向量的长度
NUM_HIDDENS = 256

# 为了方便运行，默认设为100。
# 书中使用500，可以改成：
# NUM_EPOCHS = 500
NUM_EPOCHS = 100

# 学习率
LEARNING_RATE = 1.0

# 只使用语料库的前10000个字符，和书中一致
MAX_TOKENS = 10_000

# False：顺序划分，相邻批量可以传递隐状态
# True：随机采样，每个批量必须重新初始化隐状态
USE_RANDOM_ITER = False

# 梯度裁剪阈值
CLIPPING_THETA = 1.0

# 固定随机种子，使每次运行更容易复现
RANDOM_SEED = 42


# ============================================================
# 2. 读取并清洗《时光机器》文本
# ============================================================

def read_time_machine():
    """
    下载并清洗《时光机器》文本。

    返回：
        lines：列表，每个元素是一行经过清洗的英文文本。

    清洗规则：
        1. 非英文字母替换为空格；
        2. 转换成小写；
        3. 删除行首和行尾空格。
    """

    print("正在读取《时光机器》数据……")

    # 数据直接读入内存，不写入本地文件
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        text = response.read().decode("utf-8")

    lines = []

    for line in text.splitlines():
        # [^A-Za-z]+ 表示连续的一个或多个非英文字母字符
        cleaned_line = re.sub("[^A-Za-z]+", " ", line)

        # 统一转换为小写，并删除两端空格
        cleaned_line = cleaned_line.strip().lower()

        lines.append(cleaned_line)

    return lines


# ============================================================
# 3. 词表：建立词元和索引的双向关系
# ============================================================

class Vocab:
    """
    字符词表。

    建立两个方向的映射：

        token_to_idx：
            字符 -> 索引

        idx_to_token：
            索引 -> 字符
    """

    def __init__(self, tokens):
        """
        参数：
            tokens：二维词元列表。

        例如：
            [
                ['t', 'i', 'm', 'e'],
                ['m', 'a', 'c', 'h', 'i', 'n', 'e']
            ]
        """

        # 把所有行中的字符取出来并统计频率
        counter = Counter(
            token
            for line in tokens
            for token in line
        )

        # 索引0留给未知词元
        self.idx_to_token = ["<unk>"]

        # 按出现频率从高到低排列；
        # 如果频率相同，则按字符本身排序，保证结果稳定
        sorted_tokens = sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0])
        )

        # sorted_tokens中的每个元素形如：
        # ('e', 1234)
        #
        # 我们这里只需要字符，不需要计数
        self.idx_to_token += [
            token
            for token, frequency in sorted_tokens
        ]

        # 反向建立：
        # 字符 -> 索引
        self.token_to_idx = {
            token: index
            for index, token in enumerate(self.idx_to_token)
        }

    def __len__(self):
        """返回词表大小。"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """
        把字符转换成索引。

        单个字符：
            vocab['t'] -> 5

        字符列表：
            vocab[['t', 'i', 'm', 'e']] -> [5, 7, 12, 1]
        """

        # 输入不是列表或元组，说明只有一个字符
        if not isinstance(tokens, (list, tuple)):
            # 如果字符不在词表中，返回未知词元索引0
            return self.token_to_idx.get(tokens, 0)

        # 输入是字符列表时，递归转换其中的每个字符
        return [self[token] for token in tokens]

    def to_tokens(self, indices):
        """
        把索引转换回字符。

        单个索引：
            vocab.to_tokens(5) -> 't'

        索引列表：
            vocab.to_tokens([5, 7, 12, 1]) -> ['t', 'i', 'm', 'e']
        """

        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]

        return [
            self.idx_to_token[index]
            for index in indices
        ]


# ============================================================
# 4. 把文本转换成字符索引序列
# ============================================================

def load_corpus_time_machine(max_tokens=10_000):
    """
    读取文本，建立字符词表，并把所有字符转换成索引。

    返回：
        corpus：一维字符索引序列
        vocab：字符词表
    """

    lines = read_time_machine()

    # 字符级词元化：
    # "time" -> ['t', 'i', 'm', 'e']
    tokens = [
        list(line)
        for line in lines
    ]

    # 根据所有字符建立词表
    vocab = Vocab(tokens)

    # 把二维字符列表展平，并把每个字符转换成索引
    corpus = [
        vocab[token]
        for line in tokens
        for token in line
    ]

    # 只保留前 max_tokens 个字符
    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    return corpus, vocab


# ============================================================
# 5. 随机采样
# ============================================================

def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    使用随机采样生成小批量子序列。

    随机采样的特点：
        不同小批量之间不一定在原文中相邻。

    因此：
        每个小批量都必须重新初始化隐状态。

    参数：
        corpus：
            一维字符索引序列。

        batch_size：
            一个小批量包含多少条子序列。

        num_steps：
            每条子序列包含多少个时间步。
    """

    # 随机偏移一小段，使每个epoch的切分位置有所变化
    offset = random.randint(0, num_steps - 1)
    corpus = corpus[offset:]

    # 因为标签要比输入向后错开一个位置，所以长度要减1
    num_subsequences = (len(corpus) - 1) // num_steps

    # 每条子序列在corpus中的起始位置
    initial_indices = list(
        range(0, num_subsequences * num_steps, num_steps)
    )

    # 随机打乱子序列顺序
    random.shuffle(initial_indices)

    def get_subsequence(position):
        """从指定位置取出长度为num_steps的子序列。"""
        return corpus[position:position + num_steps]

    # 完整小批量的数量
    num_batches = num_subsequences // batch_size

    for batch_start in range(0, num_batches * batch_size, batch_size):
        batch_indices = initial_indices[
            batch_start:batch_start + batch_size
        ]

        # 输入序列
        X = [
            get_subsequence(position)
            for position in batch_indices
        ]

        # 标签序列比输入向后移动一个字符
        Y = [
            get_subsequence(position + 1)
            for position in batch_indices
        ]

        yield torch.tensor(X), torch.tensor(Y)


# ============================================================
# 6. 顺序划分
# ============================================================

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    使用顺序划分生成小批量子序列。

    顺序划分的特点：
        下一个小批量在原文中紧接着当前小批量。

    因此：
        当前批量的最终隐状态可以传给下一个批量。

    但是：
        每个小批量开始前需要对隐状态调用detach，
        保留隐状态的数值，同时切断过去的计算图。
    """

    # 随机选择开头偏移量，减少每个epoch都从完全相同位置开始的问题
    offset = random.randint(0, num_steps)

    # 确保剩余词元可以被batch_size整除
    num_tokens = (
        (len(corpus) - offset - 1)
        // batch_size
        * batch_size
    )

    # 输入序列
    Xs = torch.tensor(
        corpus[offset:offset + num_tokens]
    )

    # 标签序列相对输入向后错开一个位置
    Ys = torch.tensor(
        corpus[offset + 1:offset + 1 + num_tokens]
    )

    # 将一个长序列分成batch_size行
    #
    # 形状：
    #   (num_tokens,)
    #       ↓
    #   (batch_size, 每行长度)
    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)

    # 每一行可以切出多少个长度为num_steps的小批量
    num_batches = Xs.shape[1] // num_steps

    for start in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, start:start + num_steps]
        Y = Ys[:, start:start + num_steps]

        yield X, Y


# ============================================================
# 7. 数据加载器
# ============================================================

class SeqDataLoader:
    """
    封装语料库和采样方式，使它可以直接放进for循环。
    """

    def __init__(
        self,
        batch_size,
        num_steps,
        use_random_iter=False,
        max_tokens=10_000
    ):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.use_random_iter = use_random_iter

        self.corpus, self.vocab = load_corpus_time_machine(
            max_tokens=max_tokens
        )

    def __iter__(self):
        """
        每次开始一个新的epoch时，选择对应的数据采样方法。
        """

        if self.use_random_iter:
            return seq_data_iter_random(
                self.corpus,
                self.batch_size,
                self.num_steps
            )

        return seq_data_iter_sequential(
            self.corpus,
            self.batch_size,
            self.num_steps
        )


# ============================================================
# 8. 初始化从零实现的RNN参数
# ============================================================

def get_params(vocab_size, num_hiddens, device):
    """
    创建RNN中的所有可学习参数。

    vocab_size：
        词表大小，也就是输入独热向量的长度。

    num_hiddens：
        隐藏单元数量，也就是隐状态向量的长度。
    """

    num_inputs = vocab_size
    num_outputs = vocab_size

    def normal(shape):
        """
        使用均值为0、标准差为0.01的正态分布初始化权重。
        """
        return torch.randn(size=shape, device=device) * 0.01

    # 输入X_t到隐状态H_t的权重
    #
    # X_t形状：
    #   (batch_size, vocab_size)
    #
    # W_xh形状：
    #   (vocab_size, num_hiddens)
    #
    # 相乘结果：
    #   (batch_size, num_hiddens)
    W_xh = normal((num_inputs, num_hiddens))

    # 上一个隐状态H_{t-1}到新隐状态H_t的权重
    #
    # H_{t-1}形状：
    #   (batch_size, num_hiddens)
    #
    # W_hh形状：
    #   (num_hiddens, num_hiddens)
    W_hh = normal((num_hiddens, num_hiddens))

    # 隐状态偏置
    b_h = torch.zeros(num_hiddens, device=device)

    # 隐状态H_t到输出logits O_t的权重
    #
    # H_t形状：
    #   (batch_size, num_hiddens)
    #
    # W_hq形状：
    #   (num_hiddens, vocab_size)
    #
    # 输出形状：
    #   (batch_size, vocab_size)
    W_hq = normal((num_hiddens, num_outputs))

    # 输出层偏置
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]

    # 告诉PyTorch：
    # 这些张量都是需要通过反向传播学习的参数
    for param in params:
        param.requires_grad_(True)

    return params


# ============================================================
# 9. 初始化隐状态
# ============================================================

def init_rnn_state(batch_size, num_hiddens, device):
    """
    初始化RNN隐状态。

    返回的是单元素元组：
        (H,)

    H的形状：
        (batch_size, num_hiddens)

    每条子序列都有一份独立的隐状态。
    """

    H = torch.zeros(
        (batch_size, num_hiddens),
        device=device
    )

    # 末尾的逗号表示单元素元组
    return (H,)


# ============================================================
# 10. RNN前向计算
# ============================================================

def rnn(inputs, state, params):
    """
    手写普通RNN的前向传播。

    inputs形状：
        (num_steps, batch_size, vocab_size)

    state：
        单元素元组(H,)

    参数共享：
        所有时间步共用同一组W_xh、W_hh、W_hq。
    """

    W_xh, W_hh, b_h, W_hq, b_q = params

    # 从单元素元组中取出隐状态
    H, = state

    outputs = []

    # inputs的第0维是时间步
    #
    # 每次循环中的X形状：
    #   (batch_size, vocab_size)
    for X in inputs:
        # RNN隐状态更新公式：
        #
        # H_t = tanh(
        #     X_t @ W_xh
        #     + H_{t-1} @ W_hh
        #     + b_h
        # )
        H = torch.tanh(
            X @ W_xh
            + H @ W_hh
            + b_h
        )

        # 根据当前隐状态计算当前时间步的logits：
        #
        # O_t = H_t @ W_hq + b_q
        #
        # Y形状：
        #   (batch_size, vocab_size)
        Y = H @ W_hq + b_q

        outputs.append(Y)

    # outputs中有num_steps个张量，
    # 每个张量形状为(batch_size, vocab_size)。
    #
    # 沿第0维拼接后：
    #   (num_steps * batch_size, vocab_size)
    outputs = torch.cat(outputs, dim=0)

    # 返回所有时间步的logits和最终隐状态
    return outputs, (H,)


# ============================================================
# 11. 从零开始的RNN模型封装
# ============================================================

class RNNModelScratch:
    """
    从零实现的RNN语言模型。

    注意：
        这个类没有继承nn.Module。

    因此：
        参数保存在self.params中；
        参数更新使用我们自己的sgd函数。
    """

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        device
    ):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens

        self.params = get_params(
            vocab_size,
            num_hiddens,
            device
        )

    def __call__(self, X, state):
        """
        执行前向传播。

        原始X形状：
            (batch_size, num_steps)

        每个元素是一个字符索引。
        """

        # X.T把形状转换为：
        #   (num_steps, batch_size)
        #
        # 独热编码后变成：
        #   (num_steps, batch_size, vocab_size)
        X = F.one_hot(
            X.T.long(),
            self.vocab_size
        ).float()

        return rnn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """创建初始隐状态。"""

        return init_rnn_state(
            batch_size,
            self.num_hiddens,
            device
        )


# ============================================================
# 12. 使用模型生成文本
# ============================================================

def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    根据前缀生成后续字符。

    参数：
        prefix：
            用户给出的前缀，例如"time traveller "。

        num_preds：
            需要继续生成多少个字符。

        net：
            已训练的RNN。

        vocab：
            字符词表。

        device：
            CPU或GPU。
    """

    if not prefix:
        raise ValueError("prefix不能为空")

    # 预测时批量大小为1，因为只生成一条文本
    state = net.begin_state(
        batch_size=1,
        device=device
    )

    # 先把前缀的第一个字符放进结果
    outputs = [vocab[prefix[0]]]

    def get_input():
        """
        把刚刚生成的最后一个字符变成形状(1, 1)的输入。
        """
        return torch.tensor(
            [[outputs[-1]]],
            device=device
        )

    # 预测阶段不需要建立反向传播计算图
    with torch.no_grad():

        # 预热阶段：
        # 使用真实前缀更新隐状态。
        #
        # 循环中输入的是前一个字符，
        # 然后把真实的下一个字符放进outputs。
        for character in prefix[1:]:
            _, state = net(get_input(), state)
            outputs.append(vocab[character])

        # 正式生成阶段
        for _ in range(num_preds):
            y_hat, state = net(get_input(), state)

            # y_hat形状是(1, vocab_size)
            #
            # argmax(dim=1)找到logits最大的字符索引
            next_index = int(
                y_hat.argmax(dim=1).item()
            )

            outputs.append(next_index)

    # 把索引转换回字符并拼接成字符串
    return "".join(vocab.to_tokens(outputs))


# ============================================================
# 13. 梯度裁剪
# ============================================================

def grad_clipping(net, theta):
    """
    把所有参数的整体梯度范数限制在theta以内。

    公式：
        g <- min(1, theta / ||g||) * g

    这里的g不是某一个参数的梯度，
    而是把所有参数梯度看作一个整体后的梯度向量。
    """

    # 计算所有参数梯度的平方和
    squared_sum = sum(
        torch.sum(param.grad ** 2)
        for param in net.params
        if param.grad is not None
    )

    # 整体L2范数
    norm = torch.sqrt(squared_sum)

    # 如果范数超过阈值，就按相同比例缩小所有梯度
    if norm > theta:
        with torch.no_grad():
            scale = theta / norm

            for param in net.params:
                if param.grad is not None:
                    param.grad.mul_(scale)


# ============================================================
# 14. 手写随机梯度下降
# ============================================================

def sgd(params, lr, batch_size):
    """
    手写小批量随机梯度下降。

    参数更新公式：
        param <- param - lr * grad / batch_size

    本程序中的损失已经调用mean求过平均，
    所以训练时会传入batch_size=1，避免再次平均。
    """

    # 更新参数时不需要PyTorch记录计算图
    with torch.no_grad():
        for param in params:
            # 使用当前梯度更新参数
            param -= lr * param.grad / batch_size

            # 更新完成后立即清空梯度
            #
            # 因此下一批训练前不需要再单独调用zero_grad
            param.grad.zero_()


# ============================================================
# 15. 训练一个epoch
# ============================================================

def train_epoch_ch8(
    net,
    train_iter,
    loss,
    lr,
    device,
    use_random_iter=False
):
    """
    完整遍历一次训练集。

    返回：
        perplexity：当前epoch的困惑度
        speed：每秒处理的词元数量
    """

    # 开始时还没有隐状态
    state = None

    # 累计所有词元的交叉熵总和
    total_loss = 0.0

    # 累计处理的词元数量
    total_tokens = 0

    start_time = time.perf_counter()

    for X, Y in train_iter:

        # ----------------------------------------------------
        # 第一步：处理隐状态
        # ----------------------------------------------------

        if state is None or use_random_iter:
            # 第一个批量：
            #   必须初始化隐状态。
            #
            # 随机采样：
            #   不同批量不连续，每个批量都重新初始化。
            state = net.begin_state(
                batch_size=X.shape[0],
                device=device
            )

        else:
            # 顺序划分：
            #   保留上一个批量的隐状态数值，
            #   但切断它与旧计算图的梯度联系。
            #
            # 从零实现的state固定为单元素元组(H,)，
            # 所以直接遍历其中的张量即可。
            for state_tensor in state:
                state_tensor.detach_()

        # ----------------------------------------------------
        # 第二步：整理标签形状
        # ----------------------------------------------------

        # Y原始形状：
        #   (batch_size, num_steps)
        #
        # Y.T后：
        #   (num_steps, batch_size)
        #
        # 展平后：
        #   (num_steps * batch_size,)
        #
        # 这样才能和模型按“时间优先”排列的输出对应。
        y = Y.T.reshape(-1)

        # 把输入和标签移动到模型所在设备
        X = X.to(device)
        y = y.to(device)

        # ----------------------------------------------------
        # 第三步：前向传播
        # ----------------------------------------------------

        # y_hat形状：
        #   (num_steps * batch_size, vocab_size)
        #
        # state：
        #   当前批量最后一个时间步的隐状态
        y_hat, state = net(X, state)

        # ----------------------------------------------------
        # 第四步：计算交叉熵
        # ----------------------------------------------------

        # CrossEntropyLoss接收：
        #
        # y_hat：
        #   每个位置对所有字符的logits。
        #
        # y：
        #   每个位置的正确字符索引。
        #
        # 不需要提前对y_hat使用softmax。
        l = loss(y_hat, y.long()).mean()

        # ----------------------------------------------------
        # 第五步：反向传播
        # ----------------------------------------------------

        # 计算所有参数的梯度
        l.backward()

        # 在更新参数之前裁剪梯度，防止梯度爆炸
        grad_clipping(net, CLIPPING_THETA)

        # ----------------------------------------------------
        # 第六步：更新参数并清空梯度
        # ----------------------------------------------------

        # l已经是所有词元的平均损失，
        # 因此这里传batch_size=1，避免重复平均。
        #
        # sgd内部在参数更新后调用param.grad.zero_()。
        sgd(
            net.params,
            lr,
            batch_size=1
        )

        # ----------------------------------------------------
        # 第七步：累计统计量
        # ----------------------------------------------------

        num_tokens = y.numel()

        # l是平均损失，乘词元数得到当前批量的损失总和
        total_loss += l.item() * num_tokens
        total_tokens += num_tokens

    elapsed_time = time.perf_counter() - start_time

    # 整个epoch每个词元的平均交叉熵
    average_loss = total_loss / total_tokens

    # 困惑度 = exp(平均交叉熵)
    perplexity = math.exp(average_loss)

    # 每秒处理的词元数
    speed = total_tokens / elapsed_time

    return perplexity, speed


# ============================================================
# 16. 完整训练函数
# ============================================================

def train_ch8(
    net,
    train_iter,
    vocab,
    lr,
    num_epochs,
    device,
    use_random_iter=False
):
    """
    训练多个epoch，并定期生成文本。
    """

    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        perplexity, speed = train_epoch_ch8(
            net=net,
            train_iter=train_iter,
            loss=loss,
            lr=lr,
            device=device,
            use_random_iter=use_random_iter
        )

        # 第1轮、每10轮和最后一轮显示结果
        should_print = (
            epoch == 0
            or (epoch + 1) % 10 == 0
            or epoch + 1 == num_epochs
        )

        if should_print:
            generated_text = predict_ch8(
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

            print(generated_text)

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
# 17. 程序入口
# ============================================================

def main():
    # 固定随机性，便于复现
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # 有GPU就使用GPU，否则使用CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"计算设备：{device}")

    # 创建数据加载器
    train_iter = SeqDataLoader(
        batch_size=BATCH_SIZE,
        num_steps=NUM_STEPS,
        use_random_iter=USE_RANDOM_ITER,
        max_tokens=MAX_TOKENS
    )

    vocab = train_iter.vocab

    print(f"词表大小：{len(vocab)}")
    print(f"语料词元数：{len(train_iter.corpus)}")
    print(
        "采样方式："
        + ("随机采样" if USE_RANDOM_ITER else "顺序划分")
    )

    # 创建从零实现的RNN
    net = RNNModelScratch(
        vocab_size=len(vocab),
        num_hiddens=NUM_HIDDENS,
        device=device
    )

    # 在正式训练前检查一个小批量的形状
    sample_X, sample_Y = next(iter(train_iter))

    print(f"输入X形状：{tuple(sample_X.shape)}")
    print(f"标签Y形状：{tuple(sample_Y.shape)}")

    # 使用随机参数预测，输出通常没有意义
    print("\n训练前预测：")

    print(
        predict_ch8(
            prefix="time traveller ",
            num_preds=20,
            net=net,
            vocab=vocab,
            device=device
        )
    )

    print("\n开始训练：")

    train_ch8(
        net=net,
        train_iter=train_iter,
        vocab=vocab,
        lr=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        device=device,
        use_random_iter=USE_RANDOM_ITER
    )


if __name__ == "__main__":
    main()