import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -------------------- 1. 选择计算设备 --------------------

# 有可用的NVIDIA GPU时使用CUDA，否则使用CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("训练设备：", device)


# -------------------- 2. 加载Fashion-MNIST --------------------

# ToTensor会把图片转换成浮点张量。
# 单张图片的形状是：1×28×28。
transform = transforms.ToTensor()

# 训练集用于计算梯度和更新参数。
train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)

# 测试集只用于评价模型。
test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

# 每次取256张训练图片。
# shuffle=True表示每轮训练前打乱训练样本。
# Windows环境使用num_workers=0最稳妥。
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# -------------------- 3. 创建LeNet --------------------

conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
sigmoid1 = nn.Sigmoid()
pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
sigmoid2 = nn.Sigmoid()
pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

flatten = nn.Flatten()

linear1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
sigmoid3 = nn.Sigmoid()

linear2 = nn.Linear(in_features=120, out_features=84)
sigmoid4 = nn.Sigmoid()

linear3 = nn.Linear(in_features=84, out_features=10)

# Sequential会按照这里的顺序逐层执行。
net = nn.Sequential(conv1, sigmoid1, pool1, conv2, sigmoid2, pool2, flatten, linear1, sigmoid3, linear2, sigmoid4, linear3)


# -------------------- 4. 初始化参数 --------------------

def init_weights(layer):
    """初始化卷积层和全连接层。"""

    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)

        # 如果当前层具有偏置，就把偏置初始化为0。
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


# apply会递归访问net中的所有层。
net.apply(init_weights)

# 把全部模型参数移动到CPU或GPU。
net.to(device)


# -------------------- 5. 损失函数和优化器 --------------------

# 输入必须是原始logit，不需要提前做Softmax。
loss_function = nn.CrossEntropyLoss()

# 使用书中的小批量随机梯度下降。
# 0.9是书中针对这个Sigmoid LeNet使用的学习率。
learning_rate = 0.9
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


# -------------------- 6. 测试集准确率 --------------------

def evaluate_accuracy(model, data_loader, current_device):
    """计算模型在指定数据集上的分类准确率。"""

    # 切换到评估模式。
    model.eval()

    correct_count = 0
    sample_count = 0

    # 评估时不需要保存梯度。
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(current_device)
            labels = labels.to(current_device)

            logits = model(images)

            # 在10个logit中选择最大的类别下标。
            predictions = logits.argmax(dim=1)

            # 统计预测正确的样本数量。
            correct_count += (predictions == labels).sum().item()
            sample_count += labels.numel()

    return correct_count / sample_count


# -------------------- 7. 训练模型 --------------------

num_epochs = 10

for epoch in range(num_epochs):

    # 切换到训练模式。
    net.train()

    loss_sum = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in train_loader:

        # 数据和模型必须位于同一个设备。
        images = images.to(device)
        labels = labels.to(device)

        # 清除上一批次的梯度。
        optimizer.zero_grad()

        # 前向传播，得到形状为“批量大小×10”的logit。
        logits = net(images)

        # 计算当前批次的平均交叉熵损失。
        loss = loss_function(logits, labels)

        # 计算所有卷积核、全连接层权重和偏置的梯度。
        loss.backward()

        # 使用SGD更新所有可训练参数。
        optimizer.step()

        # 当前批次的样本数可能小于batch_size，所以使用labels.numel()。
        current_batch_size = labels.numel()

        # loss.item()是当前批次的平均损失。
        # 乘以批次样本数，得到当前批次的损失总和。
        loss_sum += loss.item() * current_batch_size

        # 取得每张图片预测logit最大的类别。
        predictions = logits.argmax(dim=1)

        # 累加正确预测数量和总样本数量。
        correct_count += (predictions == labels).sum().item()
        sample_count += current_batch_size

    # 计算整轮训练的平均损失和准确率。
    train_loss = loss_sum / sample_count
    train_accuracy = correct_count / sample_count

    # 每轮结束后，在测试集上评估一次。
    test_accuracy = evaluate_accuracy(net, test_loader, device)

    print(f"epoch {epoch + 1:2d}, loss {train_loss:.3f}, train acc {train_accuracy:.3f}, test acc {test_accuracy:.3f}")