import random
import torch


# 固定Python和PyTorch的随机种子，保证实验尽可能可复现。
random.seed(42)
torch.manual_seed(42)


# ============================================================
# 1. 手写数据生成函数
# ============================================================

def synthetic_data(w, b, num_examples):
    """根据 y = Xw + b + 噪声，生成人工线性回归数据集。"""

    # 生成特征矩阵。
    # X形状：[num_examples, len(w)]，本例中是[1000, 2]。
    X = torch.normal(mean=0, std=1, size=(num_examples, len(w)))

    # 使用真实参数计算标签。
    # X形状：[1000, 2]。
    # w形状：[2]。
    # y形状：[1000]。
    y = X @ w + b

    # 为每个标签加入少量噪声。
    y += torch.normal(mean=0, std=0.01, size=y.shape)

    # 将标签改为二维列向量，形状由[1000]变为[1000, 1]。
    return X, y.reshape(-1, 1)


# 数据背后的真实参数。
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

# 生成训练数据。
features, labels = synthetic_data(true_w, true_b, num_examples=1000)

print("完整特征形状：", features.shape)   # [1000, 2]
print("完整标签形状：", labels.shape)     # [1000, 1]


# ============================================================
# 2. 手写小批量数据迭代器
# ============================================================

def data_iter(batch_size, features, labels):
    """打乱样本编号，然后逐批返回对应的特征和标签。"""

    # 获取样本总数，本例中是1000。
    num_examples = len(features)

    # 生成全部样本编号：[0, 1, 2, ..., 999]。
    indices = list(range(num_examples))

    # 原地打乱编号，但不会打乱features和labels本身。
    random.shuffle(indices)

    # 从0开始，每次前进batch_size个位置。
    for i in range(0, num_examples, batch_size):

        # 计算当前批次的结束位置。
        # min保证最后一批不会超过数据集边界。
        end = min(i + batch_size, num_examples)

        # 从打乱后的编号中取出当前批次的编号。
        batch_indices = torch.tensor(indices[i:end])

        # 使用完全相同的编号读取特征和标签，保证二者不会错位。
        # yield每次只返回一批，然后暂停函数，等待下一次读取。
        yield features[batch_indices], labels[batch_indices]


# 每个小批量使用10个样本。
batch_size = 10


# ============================================================
# 3. 手动创建并初始化模型参数
# ============================================================

# 两个输入特征需要两个权重。
# w形状：[2, 1]。
#
# requires_grad=True表示PyTorch需要追踪与w有关的计算，
# 以便执行backward后将梯度保存在w.grad中。
w = torch.normal(mean=0, std=0.01, size=(2, 1), requires_grad=True)

# 只有一个输出，因此只有一个偏置。
# b形状：[1]。
# 反向传播后，b的梯度保存在b.grad中。
b = torch.zeros(1, requires_grad=True)

print("初始权重：", w)
print("初始偏置：", b)


# ============================================================
# 4. 手写线性回归模型
# ============================================================

def linreg(X, w, b):
    """计算线性回归预测值 y_hat = Xw + b。"""

    # X形状：[batch_size, 2]。
    # w形状：[2, 1]。
    # X @ w结果形状：[batch_size, 1]。
    # b通过广播加到批量中的每个预测值上。
    return X @ w + b


# ============================================================
# 5. 手写平方损失函数
# ============================================================

def squared_loss(y_hat, y):
    """计算每个样本的平方损失，不在函数中求和或求平均。"""

    # 将真实标签调整成与预测值相同的形状，防止错误广播。
    # 例如将[10]调整为[10, 1]。
    y = y.reshape(y_hat.shape)

    # 返回每个样本各自的损失。
    # 输出形状与y_hat相同，例如[10, 1]。
    #
    # 除以2是为了让求导结果从2*(y_hat-y)简化为(y_hat-y)。
    return (y_hat - y) ** 2 / 2


# ============================================================
# 6. 手写小批量随机梯度下降
# ============================================================

def sgd(params, lr, current_batch_size):
    """使用当前批量的平均梯度更新所有参数。"""

    # 参数更新不是模型前向计算的一部分，不需要被自动求导记录。
    with torch.no_grad():

        # params是[w, b]，循环会依次更新权重和偏置。
        for param in params:

            # loss.sum().backward()计算的是当前批量的总梯度。
            # 除以current_batch_size后得到平均梯度。
            #
            # 更新公式：
            # param = param - 学习率 * 平均梯度。
            param -= lr * param.grad / current_batch_size

            # PyTorch会累加梯度。
            # 当前梯度已经使用完，必须清零，避免混入下一批。
            param.grad.zero_()


# ============================================================
# 7. 手写完整训练循环
# ============================================================

learning_rate = 0.03
num_epochs = 3

for epoch in range(num_epochs):

    # 每个epoch都会重新调用data_iter，因此样本顺序会重新打乱。
    for X, y in data_iter(batch_size, features, labels):

        # ---------- 第一步：前向传播 ----------

        # 使用当前w和b计算预测值。
        # y_hat形状：[当前批量大小, 1]。
        y_hat = linreg(X, w, b)

        # ---------- 第二步：计算每个样本的损失 ----------

        # loss不是一个标量，而是当前批量中每个样本的损失。
        # loss形状：[当前批量大小, 1]。
        loss = squared_loss(y_hat, y)

        # ---------- 第三步：反向传播 ----------

        # 先把当前批量的所有样本损失加起来，得到一个标量。
        # backward自动计算总损失对w和b的梯度。
        #
        # 计算结果保存到：
        # w.grad。
        # b.grad。
        loss.sum().backward()

        # ---------- 第四步：手动更新参数 ----------

        # X.shape[0]是当前批次的实际样本数。
        # 使用实际数量可以正确处理最后一个不足batch_size的批次。
        sgd([w, b], lr=learning_rate, current_batch_size=X.shape[0])

    # 完成一个epoch后，计算整个训练集的平均损失。
    with torch.no_grad():

        # 使用当前参数预测全部1000个样本。
        train_y_hat = linreg(features, w, b)

        # squared_loss返回每个样本的损失，因此最后调用mean求平均。
        train_loss = squared_loss(train_y_hat, labels).mean()

    print(f"epoch {epoch + 1}, loss {train_loss.item():.6f}")


# ============================================================
# 8. 查看训练结果
# ============================================================

with torch.no_grad():

    # w形状是[2, 1]，true_w形状是[2]。
    # 比较之前先将它们调整为相同形状。
    learned_w = w.reshape(true_w.shape)

    # b已经是形状[1]的张量。
    learned_b = b


print("\n真实权重：", true_w)
print("学到权重：", learned_w)
print("权重误差：", true_w - learned_w)

print("\n真实偏置：", true_b)
print("学到偏置：", learned_b)
print("偏置误差：", true_b - learned_b)