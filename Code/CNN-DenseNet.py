import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. 卷积块：从已有特征中生成固定数量的新通道。
# ============================================================
def conv_block(input_channels, num_channels):
    """
    input_channels：此前积累的特征通道数。
    num_channels：本层新增的通道数，也就是增长率。
    """
    return nn.Sequential(
        # BN 在卷积前，所以处理输入通道，而不是输出通道。
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),

        # 3×3 卷积同时处理空间邻域和输入通道。
        # 默认 stride=1，padding=1，因此高宽保持不变。
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )


# ============================================================
# 2. 稠密块：每层生成新特征，再追加到已有特征中。
# ============================================================
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        layers = []

        for i in range(num_convs):
            # i 从 0 开始。
            # 本层输入 = 原始通道 + 此前 i 层新增的通道。
            current_channels = input_channels + i * num_channels
            layers.append(conv_block(current_channels, num_channels))

        # 注册子模块，使参数能被优化器找到，并随模型移动设备。
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        # X 始终表示“到当前为止积累的全部特征”。
        for block in self.net:
            Y = block(X)

            # 输入形状为 (批量, 通道, 高, 宽)，因此 dim=1 是通道维。
            # 保留已有 X，把新生成的 Y 追加到后面。
            X = torch.cat((X, Y), dim=1)

        return X


# ============================================================
# 3. 过渡层：压缩通道，并将高宽减半。
# ============================================================
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),

        # 1×1 卷积在每个位置重新组合通道，实现可学习的压缩。
        nn.Conv2d(input_channels, num_channels, kernel_size=1),

        # 每个通道独立汇聚；2×2 窗口、步幅2使高宽减半。
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


# ============================================================
# 4. 完整 DenseNet：入口 + 稠密块/过渡层 + 分类器。
# ============================================================
def make_densenet():
    # Fashion-MNIST 是灰度图，所以输入通道数为1。
    # 96×96 → 48×48 → 24×24，入口最终输出64个通道。
    stem = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    growth_rate = 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    current_channels = 64
    layers = [stem]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        # 每个稠密块含4个卷积块，每个卷积块新增32个通道。
        layers.append(DenseBlock(num_convs, current_channels, growth_rate))
        current_channels += num_convs * growth_rate

        # 只在稠密块之间添加过渡层，最后一个块后不添加。
        if i != len(num_convs_in_dense_blocks) - 1:
            compressed_channels = current_channels // 2
            layers.append(transition_block(current_channels, compressed_channels))
            current_channels = compressed_channels

    # 通道变化：64 → 192 → 96 → 224 → 112 → 240 → 120 → 248。
    # 最后一个稠密块的输出形状为 (N, 248, 3, 3)。
    layers.extend([
        nn.BatchNorm2d(current_channels),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),  # 每个通道汇总成一个数。
        nn.Flatten(),                 # (N, 248, 1, 1) → (N, 248)。
        nn.Linear(current_channels, 10)
    ])

    # 最后一层输出10个 logits，不添加 softmax。
    return nn.Sequential(*layers)


def init_weights(module):
    """与书中 train_ch6 的主要初始化方式对应。"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)

    # BN 使用默认初始化：缩放参数为1，平移参数为0。


# ============================================================
# 5. 测试：固定模型参数，使用 BN 的运行统计量。
# ============================================================
def evaluate_accuracy(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0

    # eval() 控制 BN 的计算模式；no_grad() 控制是否记录梯度。
    # 两者作用不同，测试时通常一起使用。
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            logits = net(X)
            predictions = logits.argmax(dim=1)

            # softmax 不改变类别分数的大小顺序，所以无需先算概率。
            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total


# ============================================================
# 6. 数据加载和完整训练流程。
# ============================================================
def main():
    # 与书中这一小节相同的主要训练设置。
    batch_size = 256
    num_epochs = 10
    lr = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("训练设备：", device)

    # 原图为28×28，调整为96×96。
    # ToTensor 将像素值转为 [0, 1]，单个样本形状为 (1, 96, 96)。
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])

    # 首次运行下载数据；已有数据时直接读取。
    train_data = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

    # 训练时打乱样本顺序；测试时无需打乱。
    # num_workers=0 便于在 Windows 上直接运行。
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    net = make_densenet()
    net.apply(init_weights)
    net = net.to(device)

    # 接收原始 logits 和整数类别标签，不需要独热标签或额外 softmax。
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # 上一轮测试后模型处于 eval 模式，必须重新切回训练模式。
        # BN 此时使用当前批量统计量，并更新保存的运行统计量。
        net.train()

        loss_sum = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            # 清除上一批梯度，避免无意累积。
            optimizer.zero_grad(set_to_none=True)

            # 前向传播 → 损失 → 反向传播 → 参数更新。
            logits = net(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            # 最后一批可能不足256个样本，使用实际批量大小统计。
            batch_count = y.size(0)
            loss_sum += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += batch_count

        train_loss = loss_sum / total
        train_acc = correct / total
        test_acc = evaluate_accuracy(net, test_loader, device)

        # 训练指标在参数持续更新的过程中累计；
        # 测试指标使用本轮结束后的参数计算。
        print(
            f"轮次 {epoch + 1:2d}/{num_epochs} | "
            f"训练损失 {train_loss:.4f} | "
            f"训练准确率 {train_acc:.4f} | "
            f"测试准确率 {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()