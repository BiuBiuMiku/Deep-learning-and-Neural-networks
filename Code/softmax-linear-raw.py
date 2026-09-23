import torch
import torchvision
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

# shuffle=True：每个 epoch 随机打乱训练图片
# num_workers=0：Windows 下最稳妥；熟悉后可尝试设为 2 或 4 加速读取
train_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 测试不更新参数，也不必打乱
test_iter = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 一张 28×28 灰度图展平后共有 784 个输入特征
num_inputs = 784

# Fashion-MNIST 一共有 10 个服装类别
num_outputs = 10

# W 的形状是 [784, 10]
# 每个像素分别连接到每个类别输出神经元
# requires_grad=True：反向传播时计算 W.grad
W = torch.normal(mean=0.0, std=0.01, size=(num_inputs, num_outputs), requires_grad=True)

# b 的形状是 [10]，每个类别一个偏置
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(logits):
    """将每一行 logits 转换为概率分布"""

    # 每行减去本行最大 logit，不改变 Softmax 结果，但避免 exp 数值溢出
    shifted_logits = logits - logits.max(dim=1, keepdim=True).values

    # 对每个 logit 计算 e 的幂，结果都大于 0
    exp_logits = torch.exp(shifted_logits)

    # 每行求和，形状是 [批量大小, 1]
    partition = exp_logits.sum(dim=1, keepdim=True)

    # 广播除法：每一行除以自己的总和，使每行概率和为 1
    return exp_logits / partition

def net(X):
    """图片张量 -> 10 个类别的预测概率"""

    # X 原本是 [批量大小, 1, 28, 28]
    # 展平后变成 [批量大小, 784]
    X_flat = X.reshape(X.shape[0], -1)

    # [批量大小, 784] @ [784, 10] + [10] -> [批量大小, 10]
    logits = X_flat @ W + b

    # 将 logits 转成每个类别的预测概率
    return softmax(logits)

def cross_entropy(y_hat, y):
    """返回批量中每张图片各自的交叉熵损失"""

    # y_hat 是 [批量大小, 10] 的预测概率
    # y 是 [批量大小] 的真实类别编号
    # 取每一行中“真实类别”对应的预测概率
    correct_class_probabilities = y_hat[range(len(y_hat)), y]

    # -log(正确类别概率)：概率越小，损失越大
    return -torch.log(correct_class_probabilities)

def sgd(params, lr, batch_size):
    """手写的小批量随机梯度下降"""

    # 更新参数时不记录计算图
    with torch.no_grad():
        for param in params:
            # param.grad 是这个批量所有样本梯度的总和
            # 除以 batch_size 得到平均梯度
            param -= lr * param.grad / batch_size

            # PyTorch 默认累积梯度，更新后必须清零
            param.grad.zero_()

def accuracy(y_hat, y):
    """返回一个批量中预测正确的图片数量"""

    # 每行最大概率的位置，就是预测类别编号
    predictions = y_hat.argmax(dim=1)

    # True 转为 1，False 转为 0，再求和
    return (predictions == y).sum().item()

def evaluate_accuracy(data_iter):
    """计算当前 W、b 在给定数据集上的准确率"""

    correct_count = 0
    total_count = 0

    # 评估只前向计算，不需要梯度
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            correct_count += accuracy(y_hat, y)
            total_count += y.numel()

    return correct_count / total_count

# 学习率：每次参数更新的步长
lr = 0.1

# 完整看训练集的次数
num_epochs = 10

for epoch in range(num_epochs):
    # 记录这一轮的损失总和、正确数量、样本总数
    train_loss_sum = 0.0
    train_correct_count = 0
    train_total_count = 0

    # 一个 epoch 中，遍历训练集的所有小批量
    for X, y in train_iter:
        # 前向传播：获得 [批量大小, 10] 的预测概率
        y_hat = net(X)

        # 获得 [批量大小] 的每样本交叉熵损失
        losses = cross_entropy(y_hat, y)

        # 损失求和后反向传播，得到 W.grad 和 b.grad
        losses.sum().backward()

        # 用当前批量大小更新 W、b，并清空梯度
        sgd([W, b], lr=lr, batch_size=X.shape[0])

        # 累计训练指标
        train_loss_sum += losses.sum().item()
        train_correct_count += accuracy(y_hat, y)
        train_total_count += y.numel()

    # 计算这一轮的平均训练损失、训练准确率、测试准确率
    train_loss = train_loss_sum / train_total_count
    train_acc = train_correct_count / train_total_count
    test_acc = evaluate_accuracy(test_iter)

    print(f"epoch {epoch + 1}: train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")