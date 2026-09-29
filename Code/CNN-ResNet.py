import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. 残差块：主分支学习改变量，直连分支保留或变换输入。
# ============================================================
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()

        # input_channels：块的输入通道数。
        # num_channels：块的输出通道数。
        # strides：第一个卷积的步幅，1 保持空间尺寸，2 大约减半。
        #
        # kernel_size=3、padding=1：
        # 当 stride=1 时，卷积前后的高宽相同。
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # 第二个卷积保持通道数和空间尺寸。
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

        # 当主分支改变通道数或空间尺寸时，直连分支也必须调整。
        # 否则两条路径的张量形状不同，无法逐元素相加。
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        # 主分支第一步：卷积 → BN → ReLU。
        Y = F.relu(self.bn1(self.conv1(X)))

        # 主分支第二步：卷积 → BN。
        # 这里先不做 ReLU，使残差分支可以提供正或负的修正。
        Y = self.bn2(self.conv2(Y))

        # 直连分支：形状不变时直接传递 X；否则用 1×1 卷积调整。
        shortcut = X if self.conv3 is None else self.conv3(X)

        # 两条路径逐元素相加，最后才做 ReLU。
        return F.relu(Y + shortcut)


# ============================================================
# 2. 残差模块：将多个残差块串起来。
# ============================================================
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    """
    input_channels：进入模块时的通道数。
    num_channels：模块输出通道数。
    num_residuals：这个模块包含多少个残差块。
    first_block：是否为第一个残差模块。
    """
    blocks = []

    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 后三个模块的第一个残差块负责：
            # 1. 增加通道数；
            # 2. 将高宽减半；
            # 3. 用 1×1 卷积调整直连分支。
            blocks.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            # 第一个模块，以及其他模块中的后续块，都保持形状。
            # first_block=True 时，输入和输出通道数应相同。
            blocks.append(Residual(num_channels, num_channels))

    return blocks


# ============================================================
# 3. ResNet-18：入口 + 四个残差模块 + 分类器。
# ============================================================
def make_resnet18():
    # 输入为 Fashion-MNIST 灰度图，所以输入通道数是 1。
    # 图像会被调整为 96×96。
    #
    # 7×7 卷积：96×96 → 48×48。
    # 最大汇聚：48×48 → 24×24。
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    # 每个模块包含两个残差块。
    # 星号 * 将返回的残差块列表解包，逐个交给 nn.Sequential。
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # (N, 64, 24, 24)
    b3 = nn.Sequential(*resnet_block(64, 128, 2))                  # (N, 128, 12, 12)
    b4 = nn.Sequential(*resnet_block(128, 256, 2))                 # (N, 256, 6, 6)
    b5 = nn.Sequential(*resnet_block(256, 512, 2))                 # (N, 512, 3, 3)

    return nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.AdaptiveAvgPool2d((1, 1)),  # 每个通道汇总成一个数：(N, 512, 1, 1)。
        nn.Flatten(),                 # 展平为 (N, 512)。
        nn.Linear(512, 10)             # 输出 10 个类别的 logits。
    )


def init_weights(module):
    """与书中 train_ch6 的主要初始化方式对应。"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)

    # BN 保持默认初始化：
    # gamma 为 1，beta 为 0，运行均值为 0，运行方差为 1。


# ============================================================
# 4. 测试：使用预测模式，计算整个测试集的准确率。
# ============================================================
def evaluate_accuracy(net, data_loader, device):
    net.eval()

    # eval() 会让 BN 使用保存的运行均值、运行方差，
    # 而不是根据当前测试批量重新计算并更新它们。
    correct = 0
    total = 0

    # no_grad() 控制是否记录梯度，与 eval() 的作用不同。
    # 测试不需要反向传播，所以不保存计算图。
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            logits = net(X)  # 形状：(当前批量大小, 10)。

            # 沿类别维取最大值对应的下标，得到预测类别。
            # 求类别时无需先做 softmax，因为它不会改变大小顺序。
            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total


# ============================================================
# 5. 数据加载和完整训练流程。
# ============================================================
def main():
    # 书中这一小节使用的训练参数。
    batch_size = 256   # 每批样本数。
    num_epochs = 10    # 完整遍历训练集的次数。
    lr = 0.05          # SGD 学习率。

    # 有 CUDA GPU 就使用 GPU，否则使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("训练设备：", device)

    # Fashion-MNIST 原图是 28×28。
    # Resize 将其调整为 96×96；ToTensor 将像素转为 [0, 1] 的张量。
    # 每个样本最终形状为 (1, 96, 96)，其中 1 是灰度通道。
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])

    # 首次运行时下载；已有数据时直接读取。
    train_data = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

    # 训练时打乱样本顺序；测试时不需要打乱。
    # num_workers=0 使用主进程加载，便于在 Windows 上直接运行。
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    net = make_resnet18()
    net.apply(init_weights)
    net = net.to(device)

    # 标签是整数类别，不需要手动转换成独热向量。
    # 模型输出原始 logits，不在模型末尾添加 softmax。
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # 上一轮测试调用过 eval()，这一轮必须切回训练模式。
        # 此时 BN 使用当前训练批量统计量，并更新运行统计量。
        net.train()

        loss_sum = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            # 清除上一批留下的梯度。
            optimizer.zero_grad(set_to_none=True)

            # 前向传播：从图像得到每个类别的 logits。
            logits = net(X)

            # 默认返回当前批量的平均交叉熵。
            loss = loss_fn(logits, y)

            # 反向传播计算梯度；优化器据此更新参数。
            # 卷积、全连接和 BN 的 gamma、beta 都会参与学习。
            loss.backward()
            optimizer.step()

            # 累积整轮指标。
            # 最后一批可能不足 256 个样本，所以使用实际批量大小。
            batch_count = y.size(0)
            loss_sum += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += batch_count

        train_loss = loss_sum / total
        train_acc = correct / total
        test_acc = evaluate_accuracy(net, test_loader, device)

        # 训练准确率是在这一轮参数持续更新的过程中累计的；
        # 测试准确率使用这一轮结束后的参数计算。
        print(
            f"轮次 {epoch + 1:2d}/{num_epochs} | "
            f"训练损失 {train_loss:.4f} | "
            f"训练准确率 {train_acc:.4f} | "
            f"测试准确率 {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()