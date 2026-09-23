import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

# 固定随机种子，方便复现结果
torch.manual_seed(42)

# 有 NVIDIA 显卡时使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 第 3.5 节：读取 Fashion-MNIST 数据
# ============================================================

def load_data_fashion_mnist(batch_size):
    """下载 Fashion-MNIST，并返回训练集、测试集的数据迭代器"""

    # 原始图片 -> float32 张量；像素从 0~255 缩放到 0~1
    transform = transforms.ToTensor()

    # train=True：训练集，共 60000 张图
    train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)

    # train=False：测试集，共 10000 张图
    test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

    # 训练时随机打乱图片；Windows 下 num_workers=0 最稳妥
    train_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 测试时不更新参数，不需要随机打乱
    test_iter = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter

# ============================================================
# 第 3.6 节：准确率、评估与 train_ch3 训练函数
# ============================================================

class Accumulator:
    """在 n 个数值上累加，例如损失总和、正确数量、样本总数"""

    def __init__(self, n):
        # 创建 n 个初始值为 0 的累计变量
        self.data = [0.0] * n

    def add(self, *args):
        # 将传入值逐个累加到 self.data 对应位置
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, index):
        # 允许写 metric[0]、metric[1] 等
        return self.data[index]

def accuracy(y_hat, y):
    """返回当前批量中预测正确的样本数量"""

    # y_hat 是 [批量大小, 类别数] 的 logits
    # 每一行最大 logit 所在的位置，就是预测类别
    predictions = y_hat.argmax(dim=1)

    # True 当作 1，False 当作 0；求和得到猜对数量
    return (predictions == y).sum().item()

def evaluate_accuracy(net, data_iter):
    """计算模型在指定数据集上的准确率"""

    # 评估模式会影响 Dropout、BatchNorm；当前模型没有它们，但这是标准写法
    net.eval()

    correct_count = 0
    total_count = 0

    # 测试时不需要梯度计算图
    with torch.no_grad():
        for X, y in data_iter:
            # 将当前小批量移到模型所在设备
            X = X.to(device)
            y = y.to(device)

            # 得到 [批量大小, 10] 的 logits
            y_hat = net(X)

            # 累计正确数量和总样本数量
            correct_count += accuracy(y_hat, y)
            total_count += y.numel()

    return correct_count / total_count

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个 epoch，返回平均训练损失和训练准确率"""

    # metric[0]：损失总和
    # metric[1]：预测正确数量
    # metric[2]：处理过的样本总数
    metric = Accumulator(3)

    # 一个 epoch：遍历训练集的全部小批量
    for X, y in train_iter:
        X = X.to(device)
        y = y.to(device)

        # 前向传播：图片 -> logits
        y_hat = net(X)

        # 这里 loss 返回每张图各自的损失，形状为 [批量大小]
        l = loss(y_hat, y)

        # 清空上一批残留梯度
        updater.zero_grad()

        # 对当前批量的平均交叉熵损失反向传播
        l.mean().backward()

        # 根据每个参数的梯度更新参数
        updater.step()

        # 累计当前批量的训练指标
        metric.add(l.sum().item(), accuracy(y_hat, y), y.numel())

    # 损失总和 / 样本总数，正确总数 / 样本总数
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练多个 epoch，并在每轮结束后计算测试准确率"""

    for epoch in range(num_epochs):
        # 在训练集上完整训练一轮
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)

        # 用从未参与参数更新的测试集评估
        test_acc = evaluate_accuracy(net, test_iter)

        print(f"epoch {epoch + 1}: train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")

# ============================================================
# 第 4.3 节：nn.Sequential 简洁实现多层感知机
# ============================================================

# Sequential 严格按从左到右的顺序传递数据：
#
# 图片 [批量大小, 1, 28, 28]-15-
# -> Flatten
# -> [批量大小, 784]
# -> Linear(784, 256)
# -> [批量大小, 256]
# -> ReLU
# -> [批量大小, 256]
# -> Linear(256, 10)
# -> [批量大小, 10] logits
#
# Sequential 中可以继续堆叠更多层，例如：
# nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)
#
# 但它只适合单一路径：A -> B -> C。
# 若有分支、拼接或残差连接，则需要继承 nn.Module 并自己写 forward()。

net = nn.Sequential(nn.Flatten(), nn.Linear(in_features=784, out_features=256), nn.ReLU(), nn.Linear(in_features=256, out_features=10))

def init_weights(module):
    """初始化每个全连接层的权重和偏置"""

    # net.apply 会遍历 Flatten、Linear、ReLU 等全部子模块
    # 只有 Linear 层包含当前需要初始化的 weight 和 bias
    if type(module) == nn.Linear:
        # 原地将 weight 填充为均值 0、标准差 0.01 的正态分布
        nn.init.normal_(module.weight, mean=0.0, std=0.01)

        # 显式将 bias 初始化为 0
        nn.init.zeros_(module.bias)

# 初始化两个 Linear 层的参数
net.apply(init_weights)

# 将整个模型移到 GPU 或 CPU
net.to(device)

# ============================================================
# 损失函数、优化器、训练入口
# ============================================================

# 输入必须是 logits，不能先手动执行 Softmax
# CrossEntropyLoss 内部稳定地完成 Softmax、log 和交叉熵
# reduction="none"：返回批量中每张图各自的损失
loss = nn.CrossEntropyLoss(reduction="none")

# 小批量大小、学习率、训练轮数
batch_size = 256
lr = 0.1
num_epochs = 10

# net.parameters() 自动收集两个 Linear 层的 weight 和 bias
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 读取数据
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 调用上面贴出的第 3 章训练函数
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)