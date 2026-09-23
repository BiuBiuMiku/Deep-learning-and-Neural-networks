import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

# 固定随机种子：方便多次运行时结果更接近
torch.manual_seed(42)

# 每次训练取 256 张图片
batch_size = 256

# 将图片转为 float32 张量，并把像素从 0~255 缩放到 0~1
transform = transforms.ToTensor()

# 下载并读取训练集
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)

# 下载并读取测试集
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

# 训练时随机打乱数据
train_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 测试时不打乱数据
test_iter = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 模型由两层顺序组成：
# 1. Flatten：将 [批量大小, 1, 28, 28] 变成 [批量大小, 784]
# 2. Linear：将 784 个特征映射为 10 个 logits
# 注意：这里故意不写 Softmax
net = nn.Sequential(nn.Flatten(), nn.Linear(in_features=784, out_features=10))

def init_weights(m):
    """初始化 Linear 层的参数"""

    # net.apply 会遍历 Sequential 内的每个模块
    if type(m) == nn.Linear:
        # 原地将权重初始化为均值 0、标准差 0.01 的正态分布
        nn.init.normal_(m.weight, mean=0.0, std=0.01)

        # 显式将偏置设为 0，便于和手写版本对应
        nn.init.zeros_(m.bias)

# 对网络中的 Linear 层执行初始化
net.apply(init_weights)

# CrossEntropyLoss 输入的是 logits，不是 Softmax 概率
# 它内部稳定地完成 Softmax、log 和交叉熵计算
# 默认 reduction="mean"：直接返回当前批量的平均损失，一个标量
criterion = nn.CrossEntropyLoss()

# 将 net.parameters() 中的 weight 和 bias 交给 PyTorch 的 SGD 管理
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

def accuracy_from_logits(logits, y):
    """根据 logits 统计一个批量中预测正确的图片数量"""

    # Softmax 不改变大小顺序，所以直接选最大 logit 即可
    predictions = logits.argmax(dim=1)

    return (predictions == y).sum().item()

def evaluate_accuracy(data_iter):
    """计算模型在给定数据集上的准确率"""

    # 评估模式；当前模型没有 Dropout、BatchNorm，但这是标准习惯
    net.eval()

    correct_count = 0
    total_count = 0

    # 评估时关闭梯度记录
    with torch.no_grad():
        for X, y in data_iter:
            # net 输出 [批量大小, 10] 的 logits
            logits = net(X)

            correct_count += accuracy_from_logits(logits, y)
            total_count += y.numel()

    return correct_count / total_count

# 完整看训练集的次数
num_epochs = 10

for epoch in range(num_epochs):
    # 训练模式；以后有 Dropout、BatchNorm 时会影响其行为
    net.train()

    train_loss_sum = 0.0
    train_correct_count = 0
    train_total_count = 0

    for X, y in train_iter:
        # 清空上一批留下的梯度
        optimizer.zero_grad()

        # 前向传播：得到 logits [批量大小, 10]
        logits = net(X)

        # 输入 logits 和真实类别编号，得到平均交叉熵损失
        loss = criterion(logits, y)

        # 从 loss 反向计算所有参数的梯度
        loss.backward()

        # SGD 根据参数梯度更新 Linear 的 weight 和 bias
        optimizer.step()

        # loss 是批量平均损失，乘回批量大小后累计为总损失
        train_loss_sum += loss.item() * y.numel()
        train_correct_count += accuracy_from_logits(logits, y)
        train_total_count += y.numel()

    train_loss = train_loss_sum / train_total_count
    train_acc = train_correct_count / train_total_count
    test_acc = evaluate_accuracy(test_iter)

    print(f"epoch {epoch + 1}: train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")