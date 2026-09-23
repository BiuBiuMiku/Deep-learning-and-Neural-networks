import torch
from torch import nn


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    """
    创建一个NiN块。

    in_channels：输入通道数。
    out_channels：输出通道数。
    kernel_size：第一个普通卷积的卷积核大小。
    stride：第一个普通卷积的步幅。
    padding：第一个普通卷积的填充大小。
    """

    # 普通卷积负责观察周围区域，提取空间特征。
    spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    # 1×1卷积只处理当前位置的通道，类似在每个位置运行一个全连接层。
    channel_mlp_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    # 再进行一次通道组合，让每个位置上的小网络更深。
    channel_mlp_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    return nn.Sequential(
        spatial_conv,
        nn.ReLU(),
        channel_mlp_1,
        nn.ReLU(),
        channel_mlp_2,
        nn.ReLU()
    )


# 搭建完整NiN模型。
net = nn.Sequential(
    # 输入：(N, 1, 224, 224)
    # 使用11×11的大卷积核快速扩大感受野，并通过步幅4快速缩小图片。
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),

    # 使用3×3最大汇聚继续缩小特征图。
    nn.MaxPool2d(kernel_size=3, stride=2),

    # 提取更高级特征，通道数从96增加到256。
    # padding=2使5×5卷积前后的高度和宽度保持不变。
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),

    nn.MaxPool2d(kernel_size=3, stride=2),

    # 通道数从256增加到384。
    # padding=1使3×3卷积前后的高度和宽度保持不变。
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),

    nn.MaxPool2d(kernel_size=3, stride=2),

    # 训练时随机把一半特征变成0，减轻过拟合。
    nn.Dropout(p=0.5),

    # Fashion-MNIST有10个类别，因此把通道数变成10。
    # 训练后，每个通道会逐渐负责一个类别的证据。
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),

    # 对每个类别通道的所有空间位置取平均。
    # 输出形状固定变成(N, 10, 1, 1)。
    nn.AdaptiveAvgPool2d(output_size=(1, 1)),

    # 把(N, 10, 1, 1)变成(N, 10)。
    # 最后的10个数是logits，不是概率。
    nn.Flatten()
)


# 创建一张模拟的灰度图片。
# 形状依次是：批量大小、通道数、高度、宽度。
X = torch.randn((1, 1, 224, 224))

print("输入形状：", X.shape)

# 逐层运行，检查每一层的输出形状。
for layer in net:
    X = layer(X)
    print(f"{layer.__class__.__name__:20s} 输出形状：{X.shape}")

# 最终输出10个logits。
logits = X
print("\n最终logits：")
print(logits)

# 如果需要查看概率，再使用softmax。
probabilities = torch.softmax(logits, dim=1)
print("\n分类概率：")
print(probabilities)

# 概率之和应当接近1。
print("\n概率总和：")
print(probabilities.sum(dim=1))

# 取概率最大类别的下标作为预测结果。
predicted_class = probabilities.argmax(dim=1)
print("\n预测类别：")
print(predicted_class)