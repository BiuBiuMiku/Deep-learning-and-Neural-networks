import torch
from torch import nn
from torch.utils import data


# 固定随机种子，使每次运行生成相同的随机数据和初始参数，方便复现实验。
torch.manual_seed(42)


# ============================================================
# 1. 生成合成数据
# ============================================================

def synthetic_data(w, b, num_examples):
    """根据 y = Xw + b + 噪声，生成人工线性回归数据集。"""

    # 生成特征矩阵。
    # num_examples：样本数量，这里是1000。
    # len(w)：每个样本的特征数量，这里是2。
    # X的形状：[1000, 2]。
    X = torch.normal(mean=0, std=1, size=(num_examples, len(w)))

    # 使用真实参数生成没有噪声的标签。
    # X形状：[1000, 2]。
    # w形状：[2]。
    # X @ w结果形状：[1000]。
    # 标量b会通过广播加到每个样本上。
    y = X @ w + b

    # 为每个标签添加少量正态分布噪声，模拟现实中的测量误差。
    # 噪声均值为0，标准差为0.01，形状与y完全相同。
    y += torch.normal(mean=0, std=0.01, size=y.shape)

    # 把标签从[1000]改成[1000, 1]，与模型输出形状保持一致。
    return X, y.reshape(-1, 1)


# 数据背后的真实参数。
# 训练时模型并不知道它们，只能通过features和labels反推出它们。
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

# 生成1000个样本。
features, labels = synthetic_data(true_w, true_b, num_examples=1000)

print("完整特征形状：", features.shape)   # [1000, 2]
print("完整标签形状：", labels.shape)     # [1000, 1]


# ============================================================
# 2. 使用PyTorch DataLoader构造小批量数据
# ============================================================

def load_array(data_arrays, batch_size, is_train=True):
    """把特征和标签绑定起来，并创建小批量数据迭代器。"""

    # data_arrays是(features, labels)元组。
    # *data_arrays会把元组拆开，相当于TensorDataset(features, labels)。
    # TensorDataset保证features[i]始终与labels[i]对应。
    dataset = data.TensorDataset(*data_arrays)

    # batch_size：每次返回多少个样本。
    # shuffle=True：每个epoch开始时重新打乱训练数据。
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


# 每次使用10个样本计算损失和梯度。
batch_size = 10

# 创建数据加载器。
data_iter = load_array((features, labels), batch_size=batch_size, is_train=True)

# 取出第一批数据，检查形状是否正确。
X_batch, y_batch = next(iter(data_iter))

print("批量特征形状：", X_batch.shape)   # [10, 2]
print("批量标签形状：", y_batch.shape)   # [10, 1]


# ============================================================
# 3. 使用nn.Linear定义线性回归模型
# ============================================================

# nn.Linear(2, 1)表示：
# 每个样本输入2个特征，输出1个预测值。
#
# 它内部自动创建：
# weight形状：[1, 2]。
# bias形状：[1]。
#
# 它执行的计算是：
# y_hat = X @ weight.T + bias
#
# Sequential是一个按顺序存放网络层的容器。
net = nn.Sequential(nn.Linear(in_features=2, out_features=1))

print("模型结构：")
print(net)


# ============================================================
# 4. 初始化模型参数
# ============================================================

# 将权重初始化为均值0、标准差0.01的正态分布随机数。
# normal_末尾的下划线表示直接修改原张量。
nn.init.normal_(net[0].weight, mean=0, std=0.01)

# 将偏置初始化为0。
nn.init.zeros_(net[0].bias)

print("初始权重：", net[0].weight)
print("初始偏置：", net[0].bias)


# ============================================================
# 5. 定义损失函数
# ============================================================

# MSELoss计算平均平方误差：
# loss = mean((y_hat - y) ** 2)
#
# 默认返回一个标量，即当前批量的平均损失。
loss_fn = nn.MSELoss()


# ============================================================
# 6. 定义优化器
# ============================================================

# 学习率控制每次参数更新的步长。
learning_rate = 0.03

# net.parameters()返回模型中所有需要训练的参数：
# 1. Linear层的weight。
# 2. Linear层的bias。
#
# SGD会按照 param = param - learning_rate * param.grad 更新参数。
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


# ============================================================
# 7. 训练模型
# ============================================================

# 一个epoch表示完整使用一次训练集。
num_epochs = 3

for epoch in range(num_epochs):

    # DataLoader每次返回一个小批量：
    # X形状：[10, 2]。
    # y形状：[10, 1]。
    for X, y in data_iter:

        # ---------- 第一步：前向传播 ----------

        # 将当前批量送入模型，得到预测值。
        # y_hat = X @ weight.T + bias。
        # y_hat形状：[10, 1]。
        y_hat = net(X)

        # ---------- 第二步：计算损失 ----------

        # 比较预测值y_hat与真实标签y。
        # loss是当前批量的平均平方误差，是一个标量。
        loss = loss_fn(y_hat, y)

        # ---------- 第三步：清空旧梯度 ----------

        # PyTorch默认累加梯度，因此每批反向传播前必须清空上一批梯度。
        optimizer.zero_grad()

        # ---------- 第四步：反向传播 ----------

        # 自动计算loss对weight和bias的梯度。
        # 结果分别存入net[0].weight.grad和net[0].bias.grad。
        loss.backward()

        # ---------- 第五步：更新参数 ----------

        # SGD根据刚刚计算出的梯度更新weight和bias。
        optimizer.step()

    # 训练完一个epoch后，检查整个训练集上的平均损失。
    # 这里只进行评估，不需要计算梯度，所以使用no_grad节省内存。
    with torch.no_grad():
        train_y_hat = net(features)
        train_loss = loss_fn(train_y_hat, labels)

    print(f"epoch {epoch + 1}, loss {train_loss.item():.6f}")


# ============================================================
# 8. 查看训练结果
# ============================================================

with torch.no_grad():

    # PyTorch Linear层的weight形状是[1, 2]。
    # true_w形状是[2]，所以比较前要调整为相同形状。
    learned_w = net[0].weight.reshape(true_w.shape)

    # 偏置形状是[1]。
    learned_b = net[0].bias


print("\n真实权重：", true_w)
print("学到权重：", learned_w)
print("权重误差：", true_w - learned_w)

print("\n真实偏置：", true_b)
print("学到偏置：", learned_b)
print("偏置误差：", true_b - learned_b)