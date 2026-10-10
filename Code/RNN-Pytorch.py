"""
《动手学深度学习（PyTorch版）》8.6：
使用PyTorch高级API简洁实现循环神经网络

依赖：
    pip install torch

说明：
1. 不依赖d2l包。
2. 自动从网络读取《时光机器》数据，但不保存数据文件。
3. 使用nn.RNN封装循环隐藏层。
4. 使用nn.Linear把隐状态转换成词表上的logits。
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

# 《时光机器》文本地址
DATA_URL = "https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"

# 每个小批量包含32条子序列
BATCH_SIZE = 32

# 每条子序列包含35个时间步
NUM_STEPS = 35

# RNN隐状态H_t的长度
NUM_HIDDENS = 256

# 当前只使用单层RNN
NUM_LAYERS = 1

# 为了便于实际运行，默认100轮
# 书中使用500轮，可自行修改
NUM_EPOCHS = 100

# 学习率
LEARNING_RATE = 1.0

# 使用前10000个字符，与书中一致
MAX_TOKENS = 10_000

# 梯度裁剪阈值
CLIPPING_THETA = 1.0

# False：顺序划分
# True：随机采样
USE_RANDOM_ITER = False

# 固定随机种子，方便复现实验
RANDOM_SEED = 42


# ============================================================
# 2. 读取并清洗文本
# ============================================================

def read_time_machine():
    """
    读取并清洗《时光机器》文本。

    返回：
        lines：字符串列表，每个元素是一行文本。
    """

    print("正在读取《时光机器》数据……")

    # 数据直接读取到内存，不保存到本地文件
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        text = response.read().decode("utf-8")

    lines = []

    for line in text.splitlines():
        # 把非英文字母替换为空格
        line = re.sub("[^A-Za-z]+", " ", line)

        # 转成小写并清理两端空格
        line = line.strip().lower()

        lines.append(line)

    return lines


# ============================================================
# 3. 字符词表
# ============================================================

class Vocab:
    """
    建立字符和索引之间的双向映射。

    token_to_idx：
        字符 -> 索引

    idx_to_token：
        索引 -> 字符
    """

    def __init__(self, tokens):
        """
        tokens是二维字符列表，例如：

            [
                ['t', 'i', 'm', 'e'],
                ['m', 'a', 'c', 'h', 'i', 'n', 'e']
            ]
        """

        # 统计每个字符出现的次数
        counter = Counter(
            token
            for line in tokens
            for token in line
        )

        # 索引0留给未知词元
        self.idx_to_token = ["<unk>"]

        # 按出现频率从高到低排序
        token_frequencies = sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0])
        )

        self.idx_to_token += [
            token
            for token, frequency in token_frequencies
        ]

        # 建立反向映射：字符 -> 索引
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

        示例：
            vocab['t']
            vocab[['t', 'i', 'm', 'e']]
        """

        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)

        return [self[token] for token in tokens]

    def to_tokens(self, indices):
        """
        把索引转换回字符。

        示例：
            vocab.to_tokens(5)
            vocab.to_tokens([5, 7, 3])
        """

        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]

        return [
            self.idx_to_token[index]
            for index in indices
        ]


# ============================================================
# 4. 创建语料库
# ============================================================

def load_corpus_time_machine(max_tokens=10_000):
    """
    将《时光机器》转换成字符索引序列。

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

    vocab = Vocab(tokens)

    # 把二维字符列表展平，再把每个字符转换成索引
    corpus = [
        vocab[token]
        for line in tokens
        for token in line
    ]

    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    return corpus, vocab


# ============================================================
# 5. 随机采样
# ============================================================

def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    随机抽取子序列。

    不同小批量之间通常不连续，因此训练时每个小批量
    都要重新初始化隐状态。
    """

    # 随机跳过开头的一小段，使每轮切分位置不完全相同
    offset = random.randint(0, num_steps - 1)
    corpus = corpus[offset:]

    # 标签要比输入向后移动一个字符，所以长度减1
    num_subsequences = (len(corpus) - 1) // num_steps

    # 所有子序列的起点
    initial_indices = list(
        range(0, num_subsequences * num_steps, num_steps)
    )

    # 打乱子序列次序
    random.shuffle(initial_indices)

    def get_subsequence(position):
        return corpus[position:position + num_steps]

    # 能组成多少个完整小批量
    num_batches = num_subsequences // batch_size

    for batch_start in range(
        0,
        num_batches * batch_size,
        batch_size
    ):
        batch_indices = initial_indices[
            batch_start:batch_start + batch_size
        ]

        # 输入序列
        X = [
            get_subsequence(position)
            for position in batch_indices
        ]

        # 标签比输入向后错开一个字符
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
    顺序切分子序列。

    相邻小批量在原文中连续，因此可以把上一个小批量的
    最终隐状态传给下一个小批量。
    """

    offset = random.randint(0, num_steps)

    # 保证词元总数能被batch_size整除
    num_tokens = (
        (len(corpus) - offset - 1)
        // batch_size
        * batch_size
    )

    # 输入字符
    Xs = torch.tensor(
        corpus[offset:offset + num_tokens]
    )

    # 标签字符：向后错开一个位置
    Ys = torch.tensor(
        corpus[offset + 1:offset + 1 + num_tokens]
    )

    # 形状从：
    #   (num_tokens,)
    #
    # 变成：
    #   (batch_size, 每行长度)
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


# ============================================================
# 7. 数据迭代器
# ============================================================

class SeqDataLoader:
    """封装语料库和采样方式。"""

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
        """每个epoch开始时创建新的数据迭代器。"""

        if self.use_random_iter:
            return seq_data_iter_random(
                corpus=self.corpus,
                batch_size=self.batch_size,
                num_steps=self.num_steps
            )

        return seq_data_iter_sequential(
            corpus=self.corpus,
            batch_size=self.batch_size,
            num_steps=self.num_steps
        )


# ============================================================
# 8. 使用nn.RNN定义语言模型
# ============================================================

class RNNModel(nn.Module):
    """
    使用PyTorch高级API实现的字符级RNN语言模型。

    完整结构：

        字符索引
            ↓
        独热编码
            ↓
        nn.RNN循环隐藏层
            ↓
        每个时间步的隐状态
            ↓
        nn.Linear输出层
            ↓
        词表中每个字符的logits
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

        # ----------------------------------------------------
        # nn.RNN只负责循环隐藏层
        # ----------------------------------------------------
        #
        # input_size：
        #   每个时间步输入向量的长度。
        #
        # 这里使用独热编码，所以输入长度等于词表大小。
        #
        # hidden_size：
        #   隐状态H_t的长度。
        #
        # num_layers：
        #   RNN层数。当前使用1层。
        #
        # batch_first=False：
        #   输入形状采用：
        #   (num_steps, batch_size, input_size)
        self.rnn = nn.RNN(
            input_size=vocab_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            batch_first=False
        )

        # ----------------------------------------------------
        # nn.RNN输出的是隐状态，不是类别logits
        # ----------------------------------------------------
        #
        # 所以还需要一个全连接层：
        #
        #   隐状态长度num_hiddens
        #            ↓
        #   词表大小vocab_size
        self.linear = nn.Linear(
            in_features=num_hiddens,
            out_features=vocab_size
        )

    def forward(self, inputs, state):
        """
        前向传播。

        参数：
            inputs：
                字符索引，形状为
                (batch_size, num_steps)

            state：
                初始隐状态，形状为
                (num_layers, batch_size, num_hiddens)

        返回：
            output：
                所有位置的词表logits，形状为
                (num_steps * batch_size, vocab_size)

            state：
                最后一个时间步的隐状态，形状为
                (num_layers, batch_size, num_hiddens)
        """

        # ----------------------------------------------------
        # 第一步：转置
        # ----------------------------------------------------
        #
        # 原始inputs：
        #   (batch_size, num_steps)
        #
        # 转置后：
        #   (num_steps, batch_size)
        inputs = inputs.T

        # ----------------------------------------------------
        # 第二步：独热编码
        # ----------------------------------------------------
        #
        # 编码前：
        #   每个字符只是一个整数索引。
        #
        # 编码后：
        #   每个字符变成长度为vocab_size的向量。
        #
        # X形状：
        #   (num_steps, batch_size, vocab_size)
        X = F.one_hot(
            inputs.long(),
            num_classes=self.vocab_size
        )

        # one_hot返回整数张量，
        # 而nn.RNN需要浮点数输入
        X = X.float()

        # ----------------------------------------------------
        # 第三步：调用nn.RNN
        # ----------------------------------------------------
        #
        # Y包含每个时间步、每个样本的隐状态。
        #
        # Y形状：
        #   (num_steps, batch_size, num_hiddens)
        #
        # state是最后一个时间步的隐状态。
        #
        # state形状：
        #   (num_layers, batch_size, num_hiddens)
        Y, state = self.rnn(X, state)

        # ----------------------------------------------------
        # 第四步：整理所有隐状态
        # ----------------------------------------------------
        #
        # 原始Y：
        #   (num_steps, batch_size, num_hiddens)
        #
        # 展平后：
        #   (num_steps * batch_size, num_hiddens)
        #
        # 每一行表示某个时间步中某个样本的隐状态。
        Y = Y.reshape(
            -1,
            Y.shape[-1]
        )

        # ----------------------------------------------------
        # 第五步：计算词表logits
        # ----------------------------------------------------
        #
        # 输入：
        #   (num_steps * batch_size, num_hiddens)
        #
        # 输出：
        #   (num_steps * batch_size, vocab_size)
        #
        # 每一行表示一次“预测下一个字符”的logits。
        output = self.linear(Y)

        return output, state

    def begin_state(self, batch_size, device):
        """
        创建全0初始隐状态。

        普通单向nn.RNN的状态形状是：

            (num_layers, batch_size, num_hiddens)

        例如：
            num_layers = 1
            batch_size = 32
            num_hiddens = 256

        得到：
            (1, 32, 256)
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
# 9. 文本预测
# ============================================================

def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    使用训练后的RNN继续生成字符。

    prefix：
        给定前缀，例如"time traveller "。

    num_preds：
        在前缀后继续生成多少个字符。
    """

    if not prefix:
        raise ValueError("prefix不能为空")

    # 预测时只生成一条序列，所以batch_size=1
    state = net.begin_state(
        batch_size=1,
        device=device
    )

    # 先保存前缀的第一个字符
    outputs = [vocab[prefix[0]]]

    def get_input():
        """
        把outputs中最后一个字符作为下一时间步输入。

        返回形状：
            (batch_size=1, num_steps=1)
        """

        return torch.tensor(
            [[outputs[-1]]],
            device=device
        )

    # 预测不需要计算梯度
    with torch.no_grad():

        # ----------------------------------------------------
        # 预热阶段
        # ----------------------------------------------------
        #
        # 使用真实前缀逐步更新隐状态。
        #
        # 例如前缀是"time"：
        #
        # 输入t，加入真实字符i
        # 输入i，加入真实字符m
        # 输入m，加入真实字符e
        for character in prefix[1:]:
            _, state = net(get_input(), state)
            outputs.append(vocab[character])

        # ----------------------------------------------------
        # 正式生成阶段
        # ----------------------------------------------------
        for _ in range(num_preds):
            y_hat, state = net(get_input(), state)

            # y_hat形状：
            #   (1, vocab_size)
            #
            # 找出logits最大的字符索引
            next_index = int(
                y_hat.argmax(dim=1).item()
            )

            outputs.append(next_index)

    # 把字符索引转换回字符串
    return "".join(vocab.to_tokens(outputs))


# ============================================================
# 10. 训练一个epoch
# ============================================================

def train_epoch_ch8(
    net,
    train_iter,
    loss,
    optimizer,
    device,
    use_random_iter=False
):
    """
    训练完整的一个epoch。

    当前函数只处理PyTorch高级API模型，因此：
        不需要判断net是不是nn.Module；
        不需要判断optimizer类型；
        不需要兼容手写sgd。
    """

    state = None

    # 累计所有词元的损失总和
    total_loss = 0.0

    # 累计词元数量
    total_tokens = 0

    start_time = time.perf_counter()

    for X, Y in train_iter:

        # ----------------------------------------------------
        # 第一步：管理隐状态
        # ----------------------------------------------------

        if state is None or use_random_iter:
            # 第一个批量或者随机采样：
            # 重新初始化隐状态
            state = net.begin_state(
                batch_size=X.shape[0],
                device=device
            )
        else:
            # 当前只使用普通nn.RNN，
            # 所以state本身就是一个张量。
            #
            # 保留状态数值，但切断上一批量的计算图。
            state.detach_()

        # ----------------------------------------------------
        # 第二步：整理标签
        # ----------------------------------------------------
        #
        # 原始Y：
        #   (batch_size, num_steps)
        #
        # 转置后：
        #   (num_steps, batch_size)
        #
        # 展平后：
        #   (num_steps * batch_size,)
        #
        # 顺序与模型输出完全对应。
        y = Y.T.reshape(-1)

        X = X.to(device)
        y = y.to(device)

        # ----------------------------------------------------
        # 第三步：前向传播
        # ----------------------------------------------------

        y_hat, state = net(X, state)

        # y_hat：
        #   (num_steps * batch_size, vocab_size)
        #
        # y：
        #   (num_steps * batch_size,)
        #
        # CrossEntropyLoss直接接收logits，
        # 不需要提前执行softmax。
        l = loss(
            y_hat,
            y.long()
        )

        # ----------------------------------------------------
        # 第四步：清空旧梯度
        # ----------------------------------------------------
        #
        # PyTorch默认累加梯度。
        # optimizer.step()不会自动清空梯度，
        # 所以每个批量反向传播前都要清空。
        optimizer.zero_grad()

        # ----------------------------------------------------
        # 第五步：反向传播
        # ----------------------------------------------------

        l.backward()

        # ----------------------------------------------------
        # 第六步：梯度裁剪
        # ----------------------------------------------------
        #
        # clip_grad_norm_会：
        # 1. 计算所有参数梯度的整体L2范数；
        # 2. 如果超过CLIPPING_THETA，按比例缩小梯度；
        # 3. 保留原始梯度方向。
        nn.utils.clip_grad_norm_(
            net.parameters(),
            max_norm=CLIPPING_THETA
        )

        # ----------------------------------------------------
        # 第七步：更新参数
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # 第八步：累计统计量
        # ----------------------------------------------------

        num_tokens = y.numel()

        # l是当前批量的平均交叉熵，
        # 乘词元数后得到当前批量的损失总和
        total_loss += l.item() * num_tokens
        total_tokens += num_tokens

    elapsed_time = time.perf_counter() - start_time

    # 所有词元的平均交叉熵
    average_loss = total_loss / total_tokens

    # 困惑度
    perplexity = math.exp(average_loss)

    # 每秒处理的词元数量
    speed = total_tokens / elapsed_time

    return perplexity, speed


# ============================================================
# 11. 完整训练
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
    """训练多个epoch，并定期生成文本。"""

    # 交叉熵默认会对当前批量所有位置求平均
    loss = nn.CrossEntropyLoss()

    # PyTorch优化器会自动管理net中的所有可学习参数：
    #
    # nn.RNN内部参数
    # nn.Linear内部参数
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
            text = predict_ch8(
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

            print(text)

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
# 12. 主程序
# ============================================================

def main():
    # 固定随机种子
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # 有GPU则使用GPU，否则使用CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"计算设备：{device}")

    # 创建数据迭代器
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

    # 创建高级API模型，并移动到计算设备
    net = RNNModel(
        vocab_size=len(vocab),
        num_hiddens=NUM_HIDDENS,
        num_layers=NUM_LAYERS
    ).to(device)

    # --------------------------------------------------------
    # 训练前检查张量形状
    # --------------------------------------------------------

    sample_X, sample_Y = next(iter(train_iter))

    sample_state = net.begin_state(
        batch_size=sample_X.shape[0],
        device=device
    )

    with torch.no_grad():
        sample_output, sample_new_state = net(
            sample_X.to(device),
            sample_state
        )

    print("\n张量形状检查：")
    print(f"输入X：{tuple(sample_X.shape)}")
    print(f"标签Y：{tuple(sample_Y.shape)}")
    print(f"初始隐状态：{tuple(sample_state.shape)}")
    print(f"模型输出：{tuple(sample_output.shape)}")
    print(f"最终隐状态：{tuple(sample_new_state.shape)}")

    # 使用随机参数预测，结果通常没有意义
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