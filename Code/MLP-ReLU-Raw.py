import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

# 固定随机种子，便于复现实验结果
torch.manual_seed(42)

# 有可用 NVIDIA 显卡时使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. 读取 Fashion-MNIST 数据
# ============================================================

batch_size = 256

# 图片转为 float32 张量，并将像素从 0~255 缩放到 0~1
transform = transforms.ToTensor()

# 训练集：60000 张图片
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)

# 测试集：10000 张图片
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

# 训练集每轮随机打乱
train_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 测试集不必打乱
test_iter = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ============================================================
# 2. 第 3 章的通用训练工具
# ============================================================

class Accumulator:
    """在多个数值上累加，例如损失总和、正确数量、样本总数"""

    def __init__(self, n):
        # 创建 n 个初始值为 0 的累计变量
        self.data = [0.0] * n

    def add(self, *args):
        # 将传入的每个值，累加到对应位置
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, index):
        # 允许写 metric[0]、metric[1] 等
        return self.data[index]

def accuracy(logits, y):
    """返回一个批量中预测正确的样本数量"""

    # logits: [批量大小, 类别数]
    # 每行最大 logit 所在的位置，就是预测类别
    predictions = logits.argmax(dim=1)

    # True 视为 1，False 视为 0；sum() 得到猜对数量
    return (predictions == y).sum().item()

def evaluate_accuracy(net, data_iter):
    """计算模型在指定数据集上的准确率"""

    correct_count = 0
    total_count = 0

    # 评估时不记录梯度，节省内存与计算
    with torch.no_grad():
        for X, y in data_iter:
            # 将当前批量移到模型所在设备
            X = X.to(device)
            y = y.to(device)

            # net(X) 输出 [批量大小, 10] 的 logits
            logits = net(X)

            # 累计正确数量和总样本数量
            correct_count += accuracy(logits, y)
            total_count += y.numel()

    return correct_count / total_count

def train_epoch_ch3(net, train_iter, loss_fn, updater):
    """训练模型一个 epoch，并返回平均训练损失和训练准确率"""

    # metric[0]：全部样本的损失总和
    # metric[1]：全部样本中预测正确的数量
    # metric[2]：全部已处理样本数量
    metric = Accumulator(3)

    # 遍历训练集的每一个小批量
    for X, y in train_iter:
        # 将当前批量移到模型所在设备
        X = X.to(device)
        y = y.to(device)

        # 前向传播：得到每张图的 10 个 logits
        logits = net(X)

        # 获得每张图片各自的交叉熵损失，形状为 [批量大小]
        losses = loss_fn(logits, y)

        # 清空上一批次遗留的参数梯度
        updater.zero_grad()

        # 对当前批量的平均损失反向传播
        losses.mean().backward()

        # 根据梯度更新模型参数
        updater.step()

        # 累计当前批量的指标
        metric.add(losses.sum().item(), accuracy(logits, y), y.numel())

    # 返回平均训练损失与训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss_fn, num_epochs, updater):
    """训练多个 epoch，并在每轮结束后评估测试准确率"""

    for epoch in range(num_epochs):
        # 训练集完整训练一轮
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss_fn, updater)

        # 用从未参与更新的测试集评估
        test_acc = evaluate_accuracy(net, test_iter)

        print(f"epoch {epoch + 1}: train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")

# ============================================================
# 3. 第 4 章：手写多层感知机 MLP
# ============================================================

# 每张图片展平后有 784 个像素特征
num_inputs = 784

# Fashion-MNIST 有 10 个类别
num_outputs = 10

# 隐藏层有 256 个神经元，这是超参数
num_hiddens = 256

# W1：输入层 -> 隐藏层，形状 [784, 256]
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, device=device) * 0.01)

# b1：隐藏层偏置，形状 [256]
b1 = nn.Parameter(torch.zeros(num_hiddens, device=device))

# W2：隐藏层 -> 输出层，形状 [256, 10]
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, device=device) * 0.01)

# b2：输出层偏置，形状 [10]
b2 = nn.Parameter(torch.zeros(num_outputs, device=device))

# 交给优化器更新的全部参数
params = [W1, b1, W2, b2]

def relu(X):
    """手写 ReLU：逐元素保留正数，将负数变为 0"""
    return torch.max(X, torch.zeros_like(X))

def net(X):
    """图片 -> 隐藏层 -> ReLU -> 输出层 logits"""

    # [批量大小, 1, 28, 28] -> [批量大小, 784]
    X = X.reshape((-1, num_inputs))

    # 输入层 -> 隐藏层
    # [批量大小, 784] @ [784, 256] + [256] -> [批量大小, 256]
    H = relu(X @ W1 + b1)

    # 隐藏层 -> 输出层
    # [批量大小, 256] @ [256, 10] + [10] -> [批量大小, 10]
    # 返回 logits，不在这里写 Softmax
    return H @ W2 + b2

# ============================================================
# 4. 损失函数、优化器与开始训练
# ============================================================

# 输入 logits，内部稳定地完成 Softmax 和交叉熵
# reduction="none"：返回每张图各自的损失
loss_fn = nn.CrossEntropyLoss(reduction="none")

# 学习率与训练轮数
lr = 0.1
num_epochs = 10

# SGD 会更新 params 中的 W1、b1、W2、b2
updater = torch.optim.SGD(params, lr=lr)

# 调用上面自己写的第 3 章训练函数
train_ch3(net, train_iter, test_iter, loss_fn, num_epochs, updater)