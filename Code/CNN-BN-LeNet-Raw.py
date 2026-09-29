import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BatchNorm(nn.Module):
    """
    自己实现的批量规范化层。

    支持两种输入：
    1. 全连接层输出：(批量大小, 特征数)
    2. 卷积层输出：(批量大小, 通道数, 高度, 宽度)
    """

    def __init__(self, num_features, num_dims, eps=1e-5, momentum=0.9):
        super().__init__()

        if num_dims not in (2, 4):
            raise ValueError("num_dims 必须是 2 或 4")

        # num_features：
        # 全连接层中是输出特征数；卷积层中是输出通道数。
        #
        # 前面的 1 表示：同一个特征/通道的参数，要广播到批量中的所有样本。
        # 卷积层最后两个 1 表示：还要广播到该通道的所有空间位置。
        shape = (1, num_features) if num_dims == 2 else (1, num_features, 1, 1)

        # gamma：标准化之后的缩放参数，初始为 1。
        # beta：标准化之后的平移参数，初始为 0。
        # nn.Parameter 表示它们由反向传播计算梯度，再由优化器更新。
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        # 训练时逐批积累的均值、方差，供预测时使用。
        # register_buffer 会让它们随模型一起移动到 GPU、保存和加载，
        # 但优化器不会把它们当成需要学习的参数。
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))

        self.num_dims = num_dims
        self.eps = eps
        self.momentum = momentum

    def forward(self, X):
        # 检查输入是二维还是四维，以及特征数/通道数是否正确。
        if X.ndim != self.num_dims or X.shape[1] != self.gamma.shape[1]:
            raise ValueError("输入形状与 BN 设置不匹配")

        if self.training:
            # 全连接层 X 的形状是 (N, F)：
            # 沿第 0 维，也就是批量维求均值和方差。
            #
            # 卷积层 X 的形状是 (N, C, H, W)：
            # 沿批量、高、宽三个维度统计；保留通道维。
            dims = (0,) if self.num_dims == 2 else (0, 2, 3)

            # keepdim=True 保留被压缩的维度，方便结果广播回 X。
            # 例如 (N, F) 求均值后得到 (1, F)。
            mean = X.mean(dim=dims, keepdim=True)

            # 按书中的公式计算当前批量的方差：
            # 先计算每个值到均值的距离，平方后再取平均。
            var = ((X - mean) ** 2).mean(dim=dims, keepdim=True)

            # 训练时，使用“当前批量”的均值和方差规范化当前 X。
            # eps 防止方差为 0 时出现除以 0。
            X_hat = (X - mean) / torch.sqrt(var + self.eps)

            # 同时更新供预测使用的移动统计量。
            # 本代码 momentum=0.9 表示保留 90% 旧值，加入 10% 当前批量值。
            # 统计量不通过反向传播学习，所以更新放在 no_grad 中。
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(mean.detach(), alpha=1 - self.momentum)
                self.running_var.mul_(self.momentum).add_(var.detach(), alpha=1 - self.momentum)
        else:
            # 预测时不使用当前预测批量重新求均值和方差。
            # 这样同一个样本的结果不会受“和谁一起预测”影响。
            X_hat = (X - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # 先规范化，再通过可学习的 gamma、beta 调整尺度和位置。
        return self.gamma * X_hat + self.beta


def make_net():
    """构造书中加入自定义 BN 的 LeNet。"""
    return nn.Sequential(
        # 输入：(N, 1, 28, 28)
        # 5×5 卷积后：(N, 6, 24, 24)
        # BN 按 6 个输出通道分别处理，再交给 Sigmoid。
        nn.Conv2d(1, 6, kernel_size=5),
        BatchNorm(num_features=6, num_dims=4),
        nn.Sigmoid(),

        # 空间尺寸减半：(N, 6, 12, 12)
        nn.AvgPool2d(kernel_size=2, stride=2),

        # 第二次 5×5 卷积后：(N, 16, 8, 8)
        nn.Conv2d(6, 16, kernel_size=5),
        BatchNorm(num_features=16, num_dims=4),
        nn.Sigmoid(),

        # 再次减半：(N, 16, 4, 4)
        nn.AvgPool2d(kernel_size=2, stride=2),

        # 展平后，每个样本有 16×4×4=256 个数。
        nn.Flatten(),

        # 两个隐藏全连接层：BN 按输出特征分别处理。
        nn.Linear(16 * 4 * 4, 120),
        BatchNorm(num_features=120, num_dims=2),
        nn.Sigmoid(),

        nn.Linear(120, 84),
        BatchNorm(num_features=84, num_dims=2),
        nn.Sigmoid(),

        # Fashion-MNIST 有 10 类，因此输出 10 个 logits。
        # 这里不加 Sigmoid 或 softmax；交叉熵损失直接接收 logits。
        nn.Linear(84, 10)
    )


def accuracy(net, data_loader, device):
    """在整个数据集上计算分类准确率。"""
    net.eval()  # BN 改用训练期间积累的移动统计量。
    correct = 0
    total = 0

    with torch.no_grad():  # 评估时不需要保存反向传播所需的计算图。
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            logits = net(X)
            predictions = logits.argmax(dim=1)  # 取 logit 最大的类别。
            correct += (predictions == y).sum().item()
            total += y.numel()

    return correct / total


def main():
    # 与书中这一小节相同的主要训练参数。
    batch_size = 256   # 每次用 256 张图像计算一次梯度并更新参数。
    num_epochs = 10    # 完整遍历训练集 10 次。
    lr = 1.0           # SGD 学习率。

    # 有 CUDA GPU 就使用 GPU，否则使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("训练设备：", device)

    # Fashion-MNIST 图像原本是 28×28 灰度图。
    # ToTensor 将图像转为张量，像素值缩放到 [0, 1]。
    transform = transforms.ToTensor()

    # 首次运行会下载数据到当前目录的 data 文件夹。
    train_data = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

    # 训练集打乱顺序；测试集无需打乱。
    # num_workers=0 在 Windows 上直接运行也比较省心。
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    net = make_net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # 每轮测试之后 accuracy() 会调用 eval()，
        # 所以下一轮训练开始时必须重新调用 train()。
        net.train()

        loss_sum = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            # 清除上一批留下的梯度；PyTorch 默认会累积梯度。
            optimizer.zero_grad()

            # 前向传播：得到 (批量大小, 10) 的原始 logits。
            logits = net(X)

            # 计算当前批量的平均交叉熵损失。
            loss = loss_fn(logits, y)

            # 反向传播计算参数梯度，然后更新网络参数。
            # 这里会更新卷积/全连接权重以及 BN 的 gamma、beta。
            loss.backward()
            optimizer.step()

            # 把“批量平均损失”乘以批量样本数，
            # 以便最后计算整轮真正的样本平均损失。
            batch_count = y.size(0)
            loss_sum += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += batch_count

        # 测试时切换到 eval()，BN 不再更新移动统计量。
        test_acc = accuracy(net, test_loader, device)

        print(
            f"轮次 {epoch + 1:2d} | "
            f"训练损失 {loss_sum / total:.3f} | "
            f"训练准确率 {correct / total:.3f} | "
            f"测试准确率 {test_acc:.3f}"
        )

    # net[1] 是网络中的第一个 BN 层，负责第一个卷积层的 6 个输出通道。
    # detach() 表示这里只查看数值，不继续追踪梯度。
    print("第一个 BN 学到的 gamma：", net[1].gamma.detach().flatten())
    print("第一个 BN 学到的 beta：", net[1].beta.detach().flatten())


if __name__ == "__main__":
    main()