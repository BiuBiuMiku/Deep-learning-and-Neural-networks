import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_net():
    """使用 PyTorch 自带 BN 的 LeNet。"""
    return nn.Sequential(
        # 输入：(N, 1, 28, 28)
        # 卷积输出 6 个通道；BatchNorm2d(6) 为每个通道单独统计。
        nn.Conv2d(1, 6, kernel_size=5),
        nn.BatchNorm2d(6),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        # 第二层卷积输出 16 个通道。
        nn.Conv2d(6, 16, kernel_size=5),
        nn.BatchNorm2d(16),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        # 两次卷积、汇聚后，形状为 (N, 16, 4, 4)，展平得 256。
        nn.Flatten(),

        # 全连接输出 120 个特征；BatchNorm1d(120) 为每个特征单独统计。
        nn.Linear(16 * 4 * 4, 120),
        nn.BatchNorm1d(120),
        nn.Sigmoid(),

        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.Sigmoid(),

        # 输出 10 个原始 logits，直接交给交叉熵损失。
        nn.Linear(84, 10)
    )


def evaluate(net, data_loader, device):
    """计算测试准确率。"""
    net.eval()  # 自带 BN 改用训练期间保存的运行统计量。
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = net(X)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    batch_size = 256
    num_epochs = 10
    lr = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fashion-MNIST：28×28 灰度图，10 个类别。
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    net = make_net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        net.train()  # 使用当前批量统计量，并更新运行统计量。
        loss_sum = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            logits = net(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            batch_count = y.size(0)
            loss_sum += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += batch_count

        test_acc = evaluate(net, test_loader, device)
        print(
            f"轮次 {epoch + 1:2d} | "
            f"训练损失 {loss_sum / total:.3f} | "
            f"训练准确率 {correct / total:.3f} | "
            f"测试准确率 {test_acc:.3f}"
        )

    # 自带 BN 中，weight 对应手写版的 gamma，bias 对应 beta。
    first_bn = net[1]
    print("第一个 BN 的 gamma：", first_bn.weight.detach())
    print("第一个 BN 的 beta：", first_bn.bias.detach())
    print("第一个 BN 的运行均值：", first_bn.running_mean)


if __name__ == "__main__":
    main()